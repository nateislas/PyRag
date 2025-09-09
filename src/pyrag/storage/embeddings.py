"""Embedding service for generating text embeddings.

Default model: sentence-transformers/all-MiniLM-L6-v2 (384 dims, fast).
"""

import asyncio
from pathlib import Path
from typing import List, Union

import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from ..config import get_config
from ..logging import get_logger

logger = get_logger(__name__)


class EmbeddingService:
    """Service for generating text embeddings using SentenceTransformers."""

    def __init__(self, config=None):
        """Initialize the embedding service with SentenceTransformers model.

        Args:
            config: Optional configuration object, uses default if not provided
        """
        if config is None:
            config = get_config()

        self.config = config.embedding
        # Use configured model name (defaults handled in config)
        self.model_name = self.config.model_name
        self.model = None
        self.logger = logger

        # Setup local model cache directory (only if caching is enabled)
        if self.config.cache_models_locally:
            self.model_cache_dir = Path("./models")
            self.model_cache_dir.mkdir(exist_ok=True)
        else:
            self.model_cache_dir = None

        # Initialize model
        self._load_model()

        # Initialize tokenizer for token-aware truncation/logging
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        except Exception:
            self.tokenizer = None

    def _quantize_model_int4(self, model_path: str) -> str:
        """Quantize the model to INT4 precision using torch.quantization."""
        try:
            self.logger.info(f"Applying INT4 quantization to model: {model_path}")

            # Load the original model
            model = SentenceTransformer(model_path)

            # Get the underlying transformer model
            transformer_model = model._first_module().auto_model

            # Apply INT4 quantization to linear layers
            quantized_model = torch.quantization.quantize_dynamic(
                transformer_model, {torch.nn.Linear}, dtype=torch.qint4
            )

            # Save quantized model (only if caching is enabled)
            if self.model_cache_dir is None:
                raise RuntimeError("Cannot save quantized model: caching is disabled")
            quantized_path = str(self.model_cache_dir / f"{Path(model_path).name}_int4")
            Path(quantized_path).mkdir(exist_ok=True)

            # Save the quantized transformer model
            quantized_model.save_pretrained(quantized_path)

            # Copy tokenizer and other files
            import shutil

            shutil.copytree(
                f"{model_path}/tokenizer",
                f"{quantized_path}/tokenizer",
                dirs_exist_ok=True,
            )
            shutil.copytree(
                f"{model_path}/config", f"{quantized_path}/config", dirs_exist_ok=True
            )

            self.logger.info(f"INT4 quantized model saved to: {quantized_path}")
            return quantized_path

        except Exception as e:
            self.logger.error(f"INT4 quantization failed: {e}")
            raise

    def _load_quantized_model(self, model_path: str) -> bool:
        """Load a locally cached quantized model."""
        try:
            if self.model_cache_dir is None:
                return False
            quantized_path = str(
                self.model_cache_dir
                / f"{Path(model_path).name.replace('/', '_')}_quantized"
            )

            if Path(quantized_path).exists():
                self.logger.info(
                    f"Loading cached INT8 quantized model: {quantized_path}"
                )
                self.model = SentenceTransformer(quantized_path)
                return True
            return False

        except Exception as e:
            self.logger.error(f"Failed to load quantized model: {e}")
            return False

    def _load_model(self):
        """Load the model with optional quantization support."""
        try:
            # Determine target device and quantization compatibility
            device = self.config.device
            enable_quantization = self.config.enable_int4_quantization

            if device == "auto":
                if torch.cuda.is_available():
                    device = "cuda"
                elif hasattr(torch, "mps") and torch.mps.is_available():
                    device = "mps"
                    # Disable quantization on MPS due to memory constraints
                    enable_quantization = False
                    self.logger.info(
                        "Disabling quantization on MPS due to memory constraints"
                    )
                else:
                    device = "cpu"

            # First try to load cached quantized model if enabled and compatible
            if self.config.cache_models_locally and enable_quantization:
                if self._load_quantized_model(self.model_name):
                    self.logger.info("Successfully loaded cached quantized model")
                    self.model = self.model.to(device)
                    self.logger.info(f"Moved quantized model to {device}")
                    return

            # Load original model
            self.logger.info(f"Loading embedding model: {self.model_name}")
            self.model = SentenceTransformer(self.model_name)

            # Apply quantization if enabled and compatible
            if enable_quantization and device != "mps":
                self.logger.info("Attempting quantization...")
                try:
                    quantized_path = self._quantize_model_int4(self.model_name)

                    # Reload the quantized version
                    self.model = SentenceTransformer(quantized_path)
                    self.logger.info("Successfully applied quantization")

                except Exception as e:
                    self.logger.warning(
                        f"Quantization failed, using original model: {e}"
                    )
            else:
                if device == "mps":
                    self.logger.info(
                        "Using original model (quantization disabled on MPS)"
                    )

            # Move to target device
            self.model = self.model.to(device)

            self.logger.info(f"Successfully loaded embedding model on {device}")
            if enable_quantization and device != "mps":
                self.logger.info("Model optimized with quantization")
            self.logger.info(
                f"Model configuration: max_length={self.config.max_length}, batch_size={self.config.batch_size}"
            )

        except Exception as e:
            self.logger.error(f"Failed to load embedding model: {e}")
            raise

    def _get_embeddings(self, texts: List[str]) -> np.ndarray:
        """Generate embeddings for a list of texts using the embedding model.

        Args:
            texts: List of text strings to embed

        Returns:
            numpy array of embeddings with shape (len(texts), embedding_dim)
        """
        try:
            # Optionally pre-truncate to avoid silent tail loss and log
            if self.tokenizer is not None and isinstance(texts, list):
                prepared_texts = []
                over_limit_count = 0
                for text in texts:
                    enc = self.tokenizer(
                        text,
                        truncation=True,
                        max_length=self.config.max_length,
                        return_tensors=None,
                        add_special_tokens=True,
                    )
                    # If tokenizer returns dict with 'input_ids'
                    input_ids = enc["input_ids"] if isinstance(enc, dict) else enc
                    if (
                        isinstance(input_ids, list)
                        and len(input_ids) > self.config.max_length
                    ):
                        over_limit_count += 1
                    # Decode to plain text so downstream normalization remains consistent
                    if isinstance(enc, dict):
                        truncated_text = self.tokenizer.decode(
                            enc["input_ids"], skip_special_tokens=True
                        )
                    else:
                        truncated_text = text  # Fallback
                    prepared_texts.append(truncated_text)

                if over_limit_count > 0:
                    self.logger.debug(
                        f"Token truncation applied to {over_limit_count} item(s) (max_length={self.config.max_length})"
                    )
                texts_to_encode = prepared_texts
            else:
                texts_to_encode = texts

            # Generate embeddings using Sentence Transformers
            # The model automatically handles batching and normalization
            embeddings = self.model.encode(
                texts_to_encode,
                batch_size=self.config.batch_size,
                max_length=self.config.max_length,
                normalize_embeddings=self.config.normalize_embeddings,
                convert_to_numpy=True,
            )

            return embeddings

        except Exception as e:
            self.logger.error(f"Error generating embeddings: {e}")
            raise

    async def generate_embeddings(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s) asynchronously.

        Args:
            texts: Single text string or list of text strings

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        # Run embedding generation in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        embeddings = await loop.run_in_executor(None, self._get_embeddings, texts)

        return embeddings

    def generate_embeddings_sync(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embeddings for text(s) synchronously.

        Args:
            texts: Single text string or list of text strings

        Returns:
            numpy array of embeddings
        """
        if isinstance(texts, str):
            texts = [texts]

        return self._get_embeddings(texts)

    def get_embedding_dimension(self) -> int:
        """Get the dimension of the embeddings (model-specific)."""
        if self.model is None:
            raise RuntimeError("Model not loaded")

        try:
            return int(self.model.get_sentence_embedding_dimension())
        except Exception:
            # Fallback for models without the helper
            test = self.generate_embeddings_sync("test-dim")
            return int(test.shape[-1])

    def is_ready(self) -> bool:
        """Check if the embedding service is ready.

        Returns:
            True if model is loaded
        """
        return self.model is not None

    def health_check(self) -> dict:
        """Perform a health check on the embedding service.

        Returns:
            Dictionary with health status and model info
        """
        try:
            if not self.is_ready():
                return {
                    "status": "unhealthy",
                    "error": "Model not loaded",
                    "model_name": self.model_name,
                }

            # Test embedding generation
            test_text = "Hello world"
            embedding = self.generate_embeddings_sync(test_text)

            return {
                "status": "healthy",
                "model_name": self.model_name,
                "device": str(self.model.device),
                "embedding_dimension": self.get_embedding_dimension(),
                "test_embedding_shape": embedding.shape,
                "model_loaded": True,
                "config": {
                    "max_length": self.config.max_length,
                    "batch_size": self.config.batch_size,
                    "normalize_embeddings": self.config.normalize_embeddings,
                },
            }

        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e),
                "model_name": self.model_name,
            }

    def similarity(self, embedding1: np.ndarray, embedding2: np.ndarray) -> float:
        """Calculate cosine similarity between two embeddings.

        Args:
            embedding1: First embedding vector
            embedding2: Second embedding vector

        Returns:
            Cosine similarity score between -1 and 1
        """
        try:
            # Ensure embeddings are 2D
            if embedding1.ndim == 1:
                embedding1 = embedding1.reshape(1, -1)
            if embedding2.ndim == 1:
                embedding2 = embedding2.reshape(1, -1)

            # Calculate cosine similarity
            similarity = self.model.similarity(embedding1, embedding2)

            # Return scalar if single comparison
            if similarity.size == 1:
                return float(similarity)
            else:
                return similarity.tolist()

        except Exception as e:
            self.logger.error(f"Error calculating similarity: {e}")
            return 0.0

    async def batch_similarity(
        self, query_embedding: np.ndarray, document_embeddings: np.ndarray
    ) -> List[float]:
        """Calculate similarities between query and multiple documents.

        Args:
            query_embedding: Query embedding vector
            document_embeddings: Document embedding vectors

        Returns:
            List of similarity scores
        """
        try:
            # Ensure query is 2D
            if query_embedding.ndim == 1:
                query_embedding = query_embedding.reshape(1, -1)

            # Calculate similarities
            similarities = self.model.similarity(query_embedding, document_embeddings)

            # Debug logging to understand the structure
            self.logger.debug(f"Similarities type: {type(similarities)}")
            self.logger.debug(
                f"Similarities shape/structure: {getattr(similarities, 'shape', 'no shape')}"
            )

            # Convert to list of floats, handling different tensor types
            if isinstance(similarities, np.ndarray):
                # Convert numpy array to list of floats
                flattened = similarities.flatten()
                self.logger.debug(f"Flattened numpy array: {flattened.shape}")
                return [float(sim) for sim in flattened]
            elif hasattr(similarities, "tolist"):
                # Handle PyTorch tensors
                tensor_list = similarities.tolist()
                self.logger.debug(
                    f"PyTorch tensor converted to list: {type(tensor_list)}"
                )
                # Handle nested lists (common with PyTorch)
                if isinstance(tensor_list, list) and len(tensor_list) > 0:
                    if isinstance(tensor_list[0], list):
                        # Nested list structure, flatten it
                        flattened = [
                            item for sublist in tensor_list for item in sublist
                        ]
                        self.logger.debug(
                            f"Flattened nested list: {len(flattened)} items"
                        )
                        return [float(sim) for sim in flattened]
                    else:
                        return [float(sim) for sim in tensor_list]
                return [float(similarities)]
            elif isinstance(similarities, (list, tuple)):
                # Handle list/tuple of tensors
                result = []
                for sim in similarities:
                    if hasattr(sim, "item"):
                        result.append(float(sim.item()))
                    elif hasattr(sim, "tolist"):
                        # Handle nested structure in individual tensors
                        sim_list = sim.tolist()
                        if isinstance(sim_list, list) and len(sim_list) > 0:
                            if isinstance(sim_list[0], list):
                                # Nested list, flatten
                                flattened = [
                                    item for sublist in sim_list for item in sublist
                                ]
                                result.extend([float(item) for item in flattened])
                            else:
                                result.append(float(sim_list[0]))
                        else:
                            result.append(float(sim_list))
                    else:
                        result.append(float(sim))
                return result
            else:
                # Fallback: try to convert directly
                return [float(similarities)]

        except Exception as e:
            self.logger.error(f"Error calculating batch similarities: {e}")
            self.logger.error(f"Query embedding shape: {query_embedding.shape}")
            self.logger.error(f"Document embeddings shape: {document_embeddings.shape}")
            return [0.0] * len(document_embeddings)
