"""Core PyRAG class for orchestrating the RAG system."""

from typing import Any, Dict, List, Optional

from .config import get_config
from .storage import EmbeddingService
from .logging import get_logger
from .search import SearchEngine
from .storage import VectorStore

logger = get_logger(__name__)


class PyRAG:
    """Main PyRAG class for orchestrating the RAG system."""

    def __init__(self):
        """Initialize PyRAG system."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PyRAG system")

        # Initialize vector store
        self.vector_store = VectorStore()

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize enhanced search engine with all components
        self.search_engine = SearchEngine(
            self.vector_store,
            self.embedding_service,
            llm_client=None,  # Will be set when LLM client is available
        )

        self.logger.info("PyRAG system initialized successfully")

    async def set_llm_client(self, llm_client):
        """Set the LLM client for enhanced search capabilities."""
        self.search_engine.llm_client = llm_client
        self.logger.info("LLM client set for enhanced search")

    async def search_documentation(
        self,
        query: str,
        library: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search for relevant documentation using the enhanced search engine."""
        return await self.search_engine.search(
            query=query,
            library=library,
            max_results=max_results,
        )

    async def search_comprehensive(
        self,
        query: str,
        library: Optional[str] = None,
        max_results: int = 20,
    ) -> Dict[str, Any]:
        """Comprehensive multi-dimensional search for complex queries."""
        return await self.search_engine.search_comprehensive(
            query=query, library=library, max_results=max_results
        )

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        return await self.vector_store.add_documents(documents)

    async def get_collection_info(self, collection_name: str = None) -> Dict[str, Any]:
        """Get information about collections in the vector store."""
        return await self.vector_store.get_collection_info(collection_name)

    async def health_check(self) -> Dict[str, Any]:
        """Check the health of the PyRAG system."""
        try:
            vector_store_health = await self.vector_store.health_check()
            embedding_health = await self.embedding_service.health_check()
            
            return {
                "status": "healthy",
                "vector_store": vector_store_health,
                "embedding_service": embedding_health,
                "search_engine": "initialized",
                "llm_client": "available" if self.search_engine.llm_client else "not_set"
            }
        except Exception as e:
            return {
                "status": "unhealthy",
                "error": str(e)
            }
