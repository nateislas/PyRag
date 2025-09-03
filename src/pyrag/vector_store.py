"""Vector store integration for PyRAG."""

import hashlib
from typing import Any, Dict, List, Optional
import json
import chromadb
import numpy as np
from chromadb.config import Settings

from .config import get_config
from .logging import get_logger

logger = get_logger(__name__)


class VectorStore:
    """ChromaDB-based vector store for document storage and retrieval."""

    def __init__(self):
        """Initialize vector store."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing ChromaDB vector store")

        # Get configuration
        config = get_config()

        # Initialize ChromaDB client
        self.client = chromadb.PersistentClient(
            path=config.vector_store.db_path,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True,
            ),
        )

        # Initialize embedding service once (shared instance)
        try:
            from .embeddings import EmbeddingService
            self.embedding_service = EmbeddingService()
            self.logger.info("Embedding service initialized successfully")
        except Exception as e:
            self.logger.warning(f"Failed to initialize embedding service: {e}")
            self.embedding_service = None

        # Get or create collections
        self._setup_collections()

        self.logger.info("Vector store initialized successfully")

    def _setup_collections(self):
        """Set up ChromaDB collections with specialized content type collections for enhanced RAG."""
        # Main collection for all documents
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Main document collection for PyRAG"},
        )

        # Specialized collections for different content types (enhanced RAG collections)
        self.api_reference_collection = self.client.get_or_create_collection(
            name="api_reference",
            metadata={
                "description": (
                    "API reference documentation with class/function details"
                )
            },
        )

        self.tutorials_collection = self.client.get_or_create_collection(
            name="tutorials",
            metadata={"description": "Step-by-step guides and tutorials"},
        )

        self.examples_collection = self.client.get_or_create_collection(
            name="examples",
            metadata={"description": "Code examples and snippets"},
        )

        self.concepts_collection = self.client.get_or_create_collection(
            name="concepts",
            metadata={"description": "Core concepts and theoretical documentation"},
        )

        self.configuration_collection = self.client.get_or_create_collection(
            name="configuration",
            metadata={"description": "Setup, configuration, and options documentation"},
        )

        self.troubleshooting_collection = self.client.get_or_create_collection(
            name="troubleshooting",
            metadata={"description": "FAQ, debugging, and support documentation"},
        )

        self.changelog_collection = self.client.get_or_create_collection(
            name="changelog",
            metadata={"description": "Version history and release notes"},
        )

        self.overview_collection = self.client.get_or_create_collection(
            name="overview",
            metadata={"description": "Library overviews and getting started guides"},
        )

    def _get_collection_by_content_type(self, content_type: str):
        """Get the appropriate collection based on content type for enhanced RAG."""
        content_type_mapping = {
            "api_reference": self.api_reference_collection,
            "tutorial": self.tutorials_collection,
            "tutorials": self.tutorials_collection,
            "examples": self.examples_collection,
            "concepts": self.concepts_collection,
            "configuration": self.configuration_collection,
            "troubleshooting": self.troubleshooting_collection,
            "changelog": self.changelog_collection,
            "overview": self.overview_collection,
            "installation": self.overview_collection,  # Map to overview
            "home": self.overview_collection,  # Map to overview
            "section": self.overview_collection,  # Map to overview
            "subsection": self.overview_collection,  # Map to overview
            "detail": self.documents_collection,  # Map to main documents
        }

        return content_type_mapping.get(content_type, self.documents_collection)

    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique ID for a document chunk."""
        # Create a hash from content and key metadata
        hash_input = (
            f"{content}:{metadata.get('library', '')}:"
            f"{metadata.get('version', '')}:{metadata.get('hierarchy_path', '')}"
        )
        return hashlib.md5(hash_input.encode()).hexdigest()

    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB with enhanced RAG metadata support."""
        prepared = {}
        for key, value in metadata.items():
            # Skip None values as ChromaDB doesn't accept them
            if value is None:
                continue
            elif isinstance(value, list):
                # Convert lists to JSON strings, handling non-serializable items
                try:
                    # First try to serialize as-is
                    json.dumps(value)
                    prepared[key] = value
                except (TypeError, ValueError):
                    # If serialization fails, convert non-serializable items to strings
                    serializable_list = []
                    for item in value:
                        if (hasattr(item, '__dict__') and
                                not isinstance(item, (dict, list, str, int, float, bool))):
                            # Handle objects like ParameterInfo
                            serializable_list.append(str(item))
                        else:
                            serializable_list.append(item)
                    prepared[key] = json.dumps(serializable_list)
            elif isinstance(value, dict):
                # Convert nested dictionaries to JSON strings for complex metadata
                try:
                    # First try to serialize as-is
                    json.dumps(value)
                    prepared[key] = value
                except (TypeError, ValueError):
                    # If serialization fails, convert non-serializable values to strings
                    serializable_dict = {}
                    for k, v in value.items():
                        if (hasattr(v, '__dict__') and
                                not isinstance(v, (dict, list, str, int, float, bool))):
                            serializable_dict[k] = str(v)
                        else:
                            serializable_dict[k] = v
                    prepared[key] = json.dumps(serializable_dict)
            elif (hasattr(value, '__dict__') and
                  not isinstance(value, (str, int, float, bool))):
                # Handle objects like ParameterInfo, convert to string representation
                prepared[key] = str(value)
            else:
                # Handle basic types (str, int, float, bool)
                prepared[key] = value

        return prepared

    def _restore_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore metadata from ChromaDB format with enhanced RAG metadata support."""
        restored = {}
        for key, value in metadata.items():
            # Fields that should be converted from JSON strings back to Python objects
            json_fields = [
                "hierarchy_path",
                "semantic_topics",
                "relationship_metadata",
                "coverage_metadata",
                "children",
                "siblings",
            ]

            if key in json_fields and isinstance(value, str):
                # Convert JSON strings back to Python objects
                try:
                    import json

                    restored[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    restored[key] = value
            else:
                restored[key] = value
        return restored

    async def add_documents(
        self, documents: List[Dict[str, Any]], collection_name: str = "documents"
    ) -> List[str]:
        """Add documents to the vector store with intelligent collection selection."""
        self.logger.info(f"Adding {len(documents)} documents to vector store")

        # If no specific collection is specified, use intelligent selection
        if collection_name == "documents":
            # Group documents by content type for optimal storage
            documents_by_type = (
                self._group_documents_by_content_type(documents)
            )

            all_ids = []
            for content_type, type_docs in documents_by_type.items():
                collection = self._get_collection_by_content_type(content_type)
                type_ids = await self._add_documents_to_collection(
                    type_docs, collection, content_type
                )
                all_ids.extend(type_ids)

            return all_ids
        else:
            # Use specified collection
            collection = self._get_collection(collection_name)
            return await self._add_documents_to_collection(
                documents, collection, collection_name
            )

    def _group_documents_by_content_type(
        self, documents: List[Dict[str, Any]]
    ) -> Dict[str, List[Dict[str, Any]]]:
        """Group documents by their content type for optimal collection storage."""
        grouped = {}

        for doc in documents:
            content_type = doc.get("metadata", {}).get("content_type", "documents")
            if content_type not in grouped:
                grouped[content_type] = []
            grouped[content_type].append(doc)

        return grouped

    async def _add_documents_to_collection(
        self, documents: List[Dict[str, Any]], collection, collection_name: str
    ) -> List[str]:
        """Add documents to a specific collection with enhanced metadata processing."""
        self.logger.info(
            f"Adding {len(documents)} documents to {collection_name} collection"
        )

        # Prepare documents for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embeddings = []

        for doc in documents:
            doc_id = self._generate_id(doc["content"], doc["metadata"])
            ids.append(doc_id)
            texts.append(doc["content"])

            # Prepare metadata for ChromaDB with enhanced RAG support
            prepared_metadata = self._prepare_metadata(doc["metadata"])
            metadatas.append(prepared_metadata)

            # Use provided embedding or generate one
            if "embedding" in doc:
                embeddings.append(doc["embedding"])
            else:
                # TODO: Generate embedding using embedding model
                # For now, use a placeholder
                embeddings.append(np.random.rand(384).tolist())  # Placeholder

        # Add to collection
        if embeddings:
            collection.upsert(
                ids=ids, documents=texts, metadatas=metadatas, embeddings=embeddings
            )
        else:
            collection.upsert(ids=ids, documents=texts, metadatas=metadatas)

        self.logger.info(
            f"Successfully added {len(ids)} documents to {collection_name} collection"
        )
        return ids

    async def search(
        self,
        query: str,
        collection_name: str = "documents",
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None,
    ) -> List[Dict[str, Any]]:
        """Search for documents in the vector store."""
        self.logger.info(f"Searching {collection_name} for: {query[:100]}...")

        collection = self._get_collection(collection_name)

        # Prepare search parameters
        search_kwargs = {
            "n_results": n_results,
        }

        if where:
            search_kwargs["where"] = where

        if embedding:
            search_kwargs["query_embeddings"] = [embedding]
        else:
            # Generate query embedding using our shared embedding service instance
            if self.embedding_service:
                try:
                    query_embedding = self.embedding_service.generate_embeddings_sync(query)
                    # Ensure 1D list of floats
                    if hasattr(query_embedding, "tolist"):
                        query_embedding = query_embedding.tolist()
                    if (
                        isinstance(query_embedding, list)
                        and len(query_embedding) > 0
                        and isinstance(query_embedding[0], list)
                    ):
                        query_embedding = query_embedding[0]
                    search_kwargs["query_embeddings"] = [query_embedding]
                except Exception as e:
                    self.logger.warning(f"Embedding generation failed: {e}")
                    # Fallback to query_texts if embedding generation fails
                    search_kwargs["query_texts"] = [query]
            else:
                # No embedding service available, use query_texts
                search_kwargs["query_texts"] = [query]

        # Perform search
        results = collection.query(**search_kwargs)

        # Format results
        formatted_results = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                # Convert metadata back to proper format
                metadata = self._restore_metadata(results["metadatas"][0][i])
                formatted_results.append(
                    {
                        "id": results["ids"][0][i],
                        "content": results["documents"][0][i],
                        "metadata": metadata,
                        "distance": (
                            results["distances"][0][i]
                            if "distances" in results
                            else None
                        ),
                    }
                )

        self.logger.info(f"Found {len(formatted_results)} results")
        return formatted_results

    async def get_document(
        self, doc_id: str, collection_name: str = "documents"
    ) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        collection = self._get_collection(collection_name)

        results = collection.get(ids=[doc_id])

        if results["documents"]:
            # Convert metadata back to proper format
            metadata = self._restore_metadata(results["metadatas"][0])
            return {
                "id": results["ids"][0],
                "content": results["documents"][0],
                "metadata": metadata,
            }

        return None

    async def delete_documents(
        self, doc_ids: List[str], collection_name: str = "documents"
    ) -> bool:
        """Delete documents from the vector store."""
        self.logger.info(f"Deleting {len(doc_ids)} documents from {collection_name}")

        collection = self._get_collection(collection_name)
        collection.delete(ids=doc_ids)

        self.logger.info(f"Successfully deleted {len(doc_ids)} documents")
        return True

    async def update_document(
        self,
        doc_id: str,
        content: str,
        metadata: Dict[str, Any],
        collection_name: str = "documents",
    ) -> bool:
        """Update a document in the vector store."""
        self.logger.info(f"Updating document {doc_id} in {collection_name}")

        # Delete old document
        await self.delete_documents([doc_id], collection_name)

        # Add updated document
        await self.add_documents(
            [{"content": content, "metadata": metadata, "id": doc_id}], collection_name
        )

        self.logger.info(f"Successfully updated document {doc_id}")
        return True

    def _get_collection(self, collection_name: str):
        """Get a ChromaDB collection by name."""
        if collection_name == "documents":
            return self.documents_collection
        elif collection_name == "api_reference":
            return self.api_reference_collection
        elif collection_name == "examples":
            return self.examples_collection
        elif collection_name == "overview":
            return self.overview_collection
        else:
            # Try to get or create a custom collection
            return self.client.get_or_create_collection(name=collection_name)

    async def get_collection_stats(
        self, collection_name: str = "documents"
    ) -> Dict[str, Any]:
        """Get statistics about a collection."""
        collection = self._get_collection(collection_name)

        # Get collection count
        count = collection.count()

        return {
            "name": collection_name,
            "document_count": count,
            "metadata": collection.metadata,
        }

    async def list_collections(self) -> List[Dict[str, Any]]:
        """List all collections in the vector store."""
        collections = self.client.list_collections()

        stats = []
        for collection in collections:
            stats.append(await self.get_collection_stats(collection.name))

        return stats

    async def reset_collections(self):
        """Reset all collections to handle embedding dimension changes."""
        self.logger.info("Resetting all collections for new embedding dimensions")

        # Delete existing collections
        collections = self.client.list_collections()
        for collection in collections:
            self.client.delete_collection(name=collection.name)
            self.logger.info(f"Deleted collection: {collection.name}")

        # Recreate collections
        self._setup_collections()
        self.logger.info("Collections reset and recreated successfully")
