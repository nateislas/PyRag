"""Enhanced search capabilities for PyRAG."""

import re
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class QueryAnalysis:
    """Analysis of a search query."""
    original_query: str
    intent: str  # "api_reference", "examples", "tutorial", "general"
    library_hint: Optional[str] = None
    version_hint: Optional[str] = None
    api_path_hint: Optional[str] = None
    content_type_preference: Optional[str] = None
    confidence: float = 0.0


class QueryAnalyzer:
    """Analyzes search queries to determine intent and extract hints."""
    
    def __init__(self):
        """Initialize the query analyzer."""
        self.logger = get_logger(__name__)
        
        # Common library names and their variations
        self.library_patterns = {
            "requests": r"\brequests?\b",
            "pandas": r"\bpandas?\b",
            "numpy": r"\bnumpy\b",
            "fastapi": r"\bfastapi\b",
            "pydantic": r"\bpydantic\b",
            "sqlalchemy": r"\bsqlalchemy\b",
            "django": r"\bdjango\b",
            "flask": r"\bflask\b",
            "httpx": r"\bhttpx\b",
            "aiohttp": r"\baiohttp\b",
        }
        
        # Intent patterns
        self.intent_patterns = {
            "api_reference": [
                r"\b(api|reference|function|method|class|signature)\b",
                r"\b(how to use|how do I use)\b",
                r"\b(parameters|arguments|returns?)\b",
            ],
            "examples": [
                r"\b(example|examples|sample|snippet)\b",
                r"\b(implement|implementation)\b",
            ],
            "tutorial": [
                r"\b(tutorial|guide|walkthrough|getting started)\b",
                r"\b(learn|learning|introduction)\b",
                r"\b(beginner|basic|simple)\b",
            ],
        }
        
        # API path patterns
        self.api_path_pattern = r"\b([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*(\.[a-zA-Z_][a-zA-Z0-9_]*)*)\b"
        
        # Version patterns
        self.version_pattern = r"\b(v?\d+\.\d+(\.\d+)?)\b"
    
    def analyze_query(self, query: str) -> QueryAnalysis:
        """Analyze a search query to extract intent and hints."""
        self.logger.info(f"Analyzing query: {query}")
        
        query_lower = query.lower()
        
        # Extract library hint
        library_hint = self._extract_library_hint(query_lower)
        
        # Extract version hint
        version_hint = self._extract_version_hint(query)
        
        # Extract API path hint
        api_path_hint = self._extract_api_path_hint(query)
        
        # Determine intent
        intent, confidence = self._determine_intent(query_lower)
        
        # Determine content type preference
        content_type_preference = self._determine_content_type_preference(query_lower, intent)
        
        analysis = QueryAnalysis(
            original_query=query,
            intent=intent,
            library_hint=library_hint,
            version_hint=version_hint,
            api_path_hint=api_path_hint,
            content_type_preference=content_type_preference,
            confidence=confidence,
        )
        
        self.logger.info(f"Query analysis: {analysis}")
        return analysis
    
    def _extract_library_hint(self, query_lower: str) -> Optional[str]:
        """Extract library name from query."""
        for library, pattern in self.library_patterns.items():
            if re.search(pattern, query_lower):
                return library
        return None
    
    def _extract_version_hint(self, query: str) -> Optional[str]:
        """Extract version hint from query."""
        match = re.search(self.version_pattern, query)
        return match.group(1) if match else None
    
    def _extract_api_path_hint(self, query: str) -> Optional[str]:
        """Extract API path hint from query."""
        match = re.search(self.api_path_pattern, query)
        return match.group(1) if match else None
    
    def _determine_intent(self, query_lower: str) -> Tuple[str, float]:
        """Determine the intent of the query."""
        scores = {
            "api_reference": 0.0,
            "examples": 0.0,
            "tutorial": 0.0,
            "general": 0.0,
        }
        
        # Score each intent based on patterns
        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    scores[intent] += 1.0
        
        # Default to general if no specific intent detected
        if all(score == 0.0 for score in scores.values()):
            scores["general"] = 1.0
        
        # Find the highest scoring intent
        best_intent = max(scores, key=scores.get)
        confidence = scores[best_intent] / max(sum(scores.values()), 1.0)
        
        return best_intent, confidence
    
    def _determine_content_type_preference(self, query_lower: str, intent: str) -> Optional[str]:
        """Determine content type preference based on intent and query."""
        if intent == "api_reference":
            return "api_reference"
        elif intent == "examples":
            return "examples"
        elif intent == "tutorial":
            return "overview"
        else:
            return None


class EnhancedSearchEngine:
    """Enhanced search engine with multi-index retrieval and reranking."""
    
    def __init__(self, vector_store, embedding_service):
        """Initialize the enhanced search engine."""
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_analyzer = QueryAnalyzer()
    
    async def search(
        self,
        query: str,
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Enhanced search with query analysis and multi-index retrieval."""
        self.logger.info(f"Enhanced search for query: {query}")
        
        # Analyze the query
        analysis = self.query_analyzer.analyze_query(query)
        
        # Use analysis hints if not explicitly provided
        if not library and analysis.library_hint:
            library = analysis.library_hint
        if not version and analysis.version_hint:
            version = analysis.version_hint
        if not content_type and analysis.content_type_preference:
            content_type = analysis.content_type_preference
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embeddings(query)
        
        # Multi-index search
        results = await self._multi_index_search(
            query=query,
            query_embedding=query_embedding,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results * 2,  # Get more results for reranking
        )
        
        # Rerank results
        reranked_results = await self._rerank_results(
            results=results,
            query_analysis=analysis,
            max_results=max_results,
        )
        
        self.logger.info(f"Enhanced search returned {len(reranked_results)} results")
        return reranked_results
    
    async def _multi_index_search(
        self,
        query: str,
        query_embedding: List[float],
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Search across multiple indexes."""
        all_results = []
        
        # Determine which collections to search
        collections = self._get_search_collections(content_type)
        
        # Search each collection
        for collection_name in collections:
            try:
                # Build where clause for filtering
                where_clause = None
                if library:
                    # Ensure library is a string for proper filtering
                    where_clause = {"library": str(library)}
                    if version:
                        where_clause["version"] = str(version)
                
                # Search the collection
                try:
                    collection_results = await self.vector_store.search(
                        query=query,
                        collection_name=collection_name,
                        n_results=max_results // len(collections),
                        where=where_clause,
                        embedding=query_embedding,
                    )
                except Exception as e:
                    self.logger.warning(f"Error searching collection {collection_name}: {e}")
                    # Continue with other collections instead of failing completely
                    continue
                
                # Add collection info to results
                for result in collection_results:
                    result["collection"] = collection_name
                    result["score"] = 1.0 - (result.get("distance", 0.0))
                
                all_results.extend(collection_results)
                
            except Exception as e:
                self.logger.error(f"Error searching collection {collection_name}: {e}")
        
        return all_results
    
    def _get_search_collections(self, content_type: Optional[str] = None) -> List[str]:
        """Get list of collections to search based on content type."""
        if content_type == "api_reference":
            return ["api_reference"]
        elif content_type == "examples":
            return ["examples"]
        elif content_type == "overview":
            return ["overview"]
        else:
            # Search all collections
            return ["documents", "api_reference", "examples", "overview"]
    
    async def _rerank_results(
        self,
        results: List[Dict[str, Any]],
        query_analysis: QueryAnalysis,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Rerank results based on query analysis and additional factors."""
        if not results:
            return []
        
        # Calculate reranking scores
        reranked_results = []
        for result in results:
            # Base score from vector similarity
            base_score = result.get("score", 0.0)
            
            # Boost score based on query analysis
            boost = self._calculate_boost(result, query_analysis)
            
            # Apply boost
            final_score = base_score * boost
            
            # Create reranked result
            reranked_result = result.copy()
            reranked_result["final_score"] = final_score
            reranked_result["boost"] = boost
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        # Return top results
        return reranked_results[:max_results]
    
    def _calculate_boost(self, result: Dict[str, Any], analysis: QueryAnalysis) -> float:
        """Calculate boost factor for a result based on query analysis."""
        boost = 1.0
        
        # Boost based on content type match
        if analysis.content_type_preference:
            result_content_type = result.get("metadata", {}).get("content_type")
            if result_content_type == analysis.content_type_preference:
                boost *= 1.2
        
        # Boost based on library match
        if analysis.library_hint:
            result_library = result.get("metadata", {}).get("library")
            if result_library == analysis.library_hint:
                boost *= 1.1
        
        # Boost based on API path match
        if analysis.api_path_hint:
            result_api_path = result.get("metadata", {}).get("api_path")
            if result_api_path and analysis.api_path_hint in result_api_path:
                boost *= 1.3
        
        # Boost based on recency (newer versions get slight boost)
        result_version = result.get("metadata", {}).get("version")
        if result_version and analysis.version_hint:
            # Simple version comparison (could be more sophisticated)
            if result_version >= analysis.version_hint:
                boost *= 1.05
        
        return boost
