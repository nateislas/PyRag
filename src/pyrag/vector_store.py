"""Vector store integration for PyRAG."""

import hashlib
from typing import Any, Dict, List, Optional, Tuple
import chromadb
from chromadb.config import Settings
import numpy as np

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
            )
        )
        
        # Get or create collections
        self._setup_collections()
        
        self.logger.info("Vector store initialized successfully")
    
    def _setup_collections(self):
        """Set up ChromaDB collections."""
        # Main collection for all documents
        self.documents_collection = self.client.get_or_create_collection(
            name="documents",
            metadata={"description": "Main document collection for PyRAG"}
        )
        
        # Specialized collections for different content types
        self.api_reference_collection = self.client.get_or_create_collection(
            name="api_reference",
            metadata={"description": "API reference documentation"}
        )
        
        self.examples_collection = self.client.get_or_create_collection(
            name="examples", 
            metadata={"description": "Code examples and tutorials"}
        )
        
        self.overview_collection = self.client.get_or_create_collection(
            name="overview",
            metadata={"description": "Library and module overviews"}
        )
    
    def _generate_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """Generate a unique ID for a document chunk."""
        # Create a hash from content and key metadata
        hash_input = f"{content}:{metadata.get('library', '')}:{metadata.get('version', '')}:{metadata.get('hierarchy_path', '')}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _prepare_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare metadata for ChromaDB (convert lists to strings and filter None values)."""
        prepared = {}
        for key, value in metadata.items():
            # Skip None values as ChromaDB doesn't accept them
            if value is None:
                continue
            elif isinstance(value, list):
                # Convert lists to JSON strings
                import json
                prepared[key] = json.dumps(value)
            else:
                prepared[key] = value
        return prepared
    
    def _restore_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Restore metadata from ChromaDB format (convert JSON strings back to lists)."""
        restored = {}
        for key, value in metadata.items():
            if key in ["hierarchy_path"] and isinstance(value, str):
                # Convert JSON strings back to lists
                try:
                    import json
                    restored[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    restored[key] = value
            else:
                restored[key] = value
        return restored
    
    async def add_documents(
        self,
        documents: List[Dict[str, Any]],
        collection_name: str = "documents"
    ) -> List[str]:
        """Add documents to the vector store."""
        self.logger.info(f"Adding {len(documents)} documents to {collection_name}")
        
        collection = self._get_collection(collection_name)
        
        # Prepare documents for ChromaDB
        ids = []
        texts = []
        metadatas = []
        embeddings = []
        
        for doc in documents:
            doc_id = self._generate_id(doc["content"], doc["metadata"])
            ids.append(doc_id)
            texts.append(doc["content"])
            # Prepare metadata for ChromaDB
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
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas,
                embeddings=embeddings
            )
        else:
            collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )
        
        self.logger.info(f"Successfully added {len(ids)} documents to {collection_name}")
        return ids
    
    async def search(
        self,
        query: str,
        collection_name: str = "documents",
        n_results: int = 10,
        where: Optional[Dict[str, Any]] = None,
        embedding: Optional[List[float]] = None
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
            search_kwargs["query_texts"] = [query]
        
        # Perform search
        results = collection.query(**search_kwargs)
        
        # Format results
        formatted_results = []
        if results["documents"]:
            for i in range(len(results["documents"][0])):
                # Convert metadata back to proper format
                metadata = self._restore_metadata(results["metadatas"][0][i])
                formatted_results.append({
                    "id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": metadata,
                    "distance": results["distances"][0][i] if "distances" in results else None,
                })
        
        self.logger.info(f"Found {len(formatted_results)} results")
        return formatted_results
    
    async def get_document(self, doc_id: str, collection_name: str = "documents") -> Optional[Dict[str, Any]]:
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
        self,
        doc_ids: List[str],
        collection_name: str = "documents"
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
        collection_name: str = "documents"
    ) -> bool:
        """Update a document in the vector store."""
        self.logger.info(f"Updating document {doc_id} in {collection_name}")
        
        # Delete old document
        await self.delete_documents([doc_id], collection_name)
        
        # Add updated document
        await self.add_documents([{
            "content": content,
            "metadata": metadata,
            "id": doc_id
        }], collection_name)
        
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
    
    async def get_collection_stats(self, collection_name: str = "documents") -> Dict[str, Any]:
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
