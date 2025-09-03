"""Simplified search engine for PyRAG using ChromaDB directly."""

from typing import Any, Dict, List, Optional
from .logging import get_logger

logger = get_logger(__name__)


class SimpleSearchEngine:
    """Simplified search engine that leverages ChromaDB's built-in capabilities."""

    def __init__(self, vector_store, embedding_service):
        """Initialize the simple search engine."""
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_service = embedding_service

    async def search(
        self,
        query: str,
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Simple search using ChromaDB's built-in capabilities."""
        self.logger.info(f"Simple search for query: {query}")
        
        # Determine which collection to search
        collection_name = self._get_collection_name(content_type)
        
        # Build simple where clause
        where_clause = self._build_where_clause(library, version)
        
        try:
            # Use vector store's search method directly
            results = await self.vector_store.search(
                query=query,
                collection_name=collection_name,
                n_results=max_results,
                where=where_clause
            )
            
            # Add collection info to results for transparency
            for result in results:
                result["collection"] = collection_name
                # Ensure score is available (ChromaDB returns distance, convert to score)
                if "distance" in result:
                    result["score"] = 1.0 - result["distance"]
                elif "score" not in result:
                    result["score"] = 1.0
            
            self.logger.info(f"Search returned {len(results)} results")
            return results
            
        except Exception as e:
            self.logger.error(f"Search failed: {e}")
            return []
    
    def _get_collection_name(self, content_type: Optional[str] = None) -> str:
        """Get the appropriate collection name based on content type."""
        if content_type == "api_reference":
            return "api_reference"
        elif content_type == "examples":
            return "examples"
        elif content_type == "overview":
            return "overview"
        elif content_type == "tutorials":
            return "tutorials"
        else:
            # Default to overview collection which has most content
            return "overview"
    
    def _build_where_clause(self, library: Optional[str], version: Optional[str]) -> Optional[Dict[str, Any]]:
        """Build a simple where clause for ChromaDB filtering."""
        if not library and not version:
            return None
        
        where_clause = {}
        
        if library:
            where_clause["library_name"] = str(library)
        
            if version:
                where_clause["version"] = str(version)

        # If we have multiple conditions, use $and
        if len(where_clause) > 1:
            return {"$and": [{"k": v} for k, v in where_clause.items()]}
        
        return where_clause


# Keep the QueryAnalysis dataclass for backward compatibility
class QueryAnalysis:
    """Simple query analysis for backward compatibility."""
    
    def __init__(self, original_query: str):
        self.original_query = original_query
        self.intent = "general"
        self.library_hint = None
        self.version_hint = None
        self.api_path_hint = None
        self.content_type_preference = None
        self.confidence = 1.0


# Keep the QueryAnalyzer class for backward compatibility
class QueryAnalyzer:
    """Simple query analyzer for backward compatibility."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Simple query analysis that returns basic info."""
        return QueryAnalysis(query)


# Keep the IntelligentSearchStrategy class for backward compatibility
class IntelligentSearchStrategy:
    """Simple strategy selector for backward compatibility."""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    def select_strategy(self, query: str, library: str, coverage_data: Dict[str, Any], query_context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Return a simple default strategy."""
        return {"strategy": "simple_search"}


# Keep the RelationshipManager class for backward compatibility
class RelationshipManager:
    """Simple relationship manager for backward compatibility."""
    
    def __init__(self, vector_store):
        self.vector_store = vector_store
    
    async def find_related_documents(self, doc_id: str, max_results: int = 5) -> List[Dict[str, Any]]:
        """Simple related document finder."""
        return []
