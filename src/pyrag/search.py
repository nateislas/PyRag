"""Enhanced search capabilities for PyRAG."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict

from .logging import get_logger
from .intelligent_strategy import IntelligentSearchStrategy, SearchStrategy
from .relationship_manager import RelationshipManager

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
        content_type_preference = self._determine_content_type_preference(
            query_lower, intent
        )

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

    def _determine_content_type_preference(
        self, query_lower: str, intent: str
    ) -> Optional[str]:
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
    """Enhanced search engine with intelligent strategy selection and multi-index retrieval."""

    def __init__(self, vector_store, embedding_service):
        """Initialize the enhanced search engine."""
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.query_analyzer = QueryAnalyzer()
        self.intelligent_strategy = IntelligentSearchStrategy()
        self.relationship_manager = RelationshipManager(vector_store)

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
                    self.logger.warning(
                        f"Error searching collection {collection_name}: {e}"
                    )
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

    def _calculate_boost(
        self, result: Dict[str, Any], analysis: QueryAnalysis
    ) -> float:
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

    async def intelligent_search(
        self,
        query: str,
        library: str,
        coverage_data: Dict[str, Any],
        version: Optional[str] = None,
        query_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Enhanced search using intelligent strategy selection based on coverage analysis."""
        
        self.logger.info(f"Intelligent search for query: {query} in library: {library}")
        
        # Select optimal search strategy based on coverage and query
        strategy_config = self.intelligent_strategy.select_strategy(
            query, library, coverage_data, query_context
        )
        
        # Analyze the query for additional context
        analysis = self.query_analyzer.analyze_query(query)
        
        # Generate query embedding
        query_embedding = await self.embedding_service.generate_embeddings(query)
        
        # Get ordered collections to search based on strategy
        search_collections = self.intelligent_strategy.get_collection_search_order(strategy_config)
        
        # Multi-index search with strategy-based optimization
        results = await self._intelligent_multi_index_search(
            query=query,
            query_embedding=query_embedding,
            library=library,
            version=version,
            search_collections=search_collections,
            strategy_config=strategy_config,
            max_results=strategy_config.max_results * 2,  # Get more for reranking
        )
        
        # Enhanced reranking using strategy-based metadata boosting
        reranked_results = await self._intelligent_rerank_results(
            results=results,
            query_analysis=analysis,
            strategy_config=strategy_config,
            max_results=strategy_config.max_results,
        )
        
        # Context expansion if enabled
        if strategy_config.expand_context:
            expanded_results = await self._expand_search_context(
                reranked_results, 
                max_expansion=strategy_config.search_depth
            )
            reranked_results = expanded_results[:strategy_config.max_results]
        
        self.logger.info(
            f"Intelligent search returned {len(reranked_results)} results "
            f"using {strategy_config.strategy.value} strategy"
        )
        
        return reranked_results

    async def _intelligent_multi_index_search(
        self,
        query: str,
        query_embedding: List[float],
        library: str,
        version: Optional[str] = None,
        search_collections: List[str] = None,
        strategy_config = None,
        max_results: int = 20,
    ) -> List[Dict[str, Any]]:
        """Multi-index search optimized for intelligent strategy."""
        
        all_results = []
        
        # Search each collection with strategy-based optimization
        for collection_name in search_collections:
            try:
                # Build where clause for filtering
                where_clause = {"library": str(library)}
                if version:
                    where_clause["version"] = str(version)
                
                # Apply quality filtering based on strategy
                # Note: ChromaDB where clause filtering is simplified for now
                # TODO: Implement proper quality filtering when ChromaDB supports it
                pass
                
                # Search the collection
                collection_results = await self.vector_store.search(
                    query=query,
                    collection_name=collection_name,
                    n_results=max_results // len(search_collections),
                    where=where_clause,
                    embedding=query_embedding,
                )
                
                # Add collection info and apply collection weights
                for result in collection_results:
                    result["collection"] = collection_name
                    result["score"] = 1.0 - (result.get("distance", 0.0))
                    # Apply collection weight from strategy
                    collection_weight = strategy_config.collection_weights.get(collection_name, 1.0)
                    result["score"] *= collection_weight
                
                all_results.extend(collection_results)
                
            except Exception as e:
                self.logger.warning(f"Error searching collection {collection_name}: {e}")
                continue
        
        return all_results

    async def _intelligent_rerank_results(
        self,
        results: List[Dict[str, Any]],
        query_analysis: QueryAnalysis,
        strategy_config,
        max_results: int,
    ) -> List[Dict[str, Any]]:
        """Enhanced reranking using intelligent strategy metadata boosting."""
        
        if not results:
            return []
        
        # Calculate reranking scores with strategy-based boosting
        reranked_results = []
        for result in results:
            # Base score from vector similarity
            base_score = result.get("score", 0.0)
            
            # Apply strategy-based metadata boost
            metadata_boost = self.intelligent_strategy.calculate_metadata_boost(
                result.get("metadata", {}), strategy_config
            )
            
            # Apply query analysis boost
            query_boost = self._calculate_boost(result, query_analysis)
            
            # Final score combines all factors
            final_score = base_score * metadata_boost * query_boost
            
            # Create reranked result
            reranked_result = result.copy()
            reranked_result["final_score"] = final_score
            reranked_result["metadata_boost"] = metadata_boost
            reranked_result["query_boost"] = query_boost
            reranked_result["boost_breakdown"] = {
                "metadata": metadata_boost,
                "query": query_boost,
                "total": final_score / base_score if base_score > 0 else 1.0
            }
            
            reranked_results.append(reranked_result)
        
        # Sort by final score
        reranked_results.sort(key=lambda x: x["final_score"], reverse=True)
        
        return reranked_results[:max_results]

    async def _expand_search_context(
        self, 
        primary_results: List[Dict[str, Any]], 
        max_expansion: int = 3
    ) -> List[Dict[str, Any]]:
        """Expand search results using hierarchical relationships."""
        
        try:
            # Use relationship manager for context expansion
            expansion_result = await self.relationship_manager.expand_search_context(
                primary_results, 
                max_expansion=max_expansion
            )
            
            # Combine primary and expanded results
            all_results = expansion_result.primary_results + expansion_result.expanded_results
            
            # Log expansion details
            self.logger.info(
                f"Context expansion: {len(expansion_result.primary_results)} primary + "
                f"{len(expansion_result.expanded_results)} expanded = {len(all_results)} total"
            )
            
            # Log relationship metadata
            if expansion_result.expansion_metadata:
                metadata = expansion_result.expansion_metadata
                self.logger.info(
                    f"Expansion metadata: ratio={metadata.get('expansion_ratio', 0):.2f}, "
                    f"avg_strength={metadata.get('average_relationship_strength', 0):.2f}"
                )
            
            return all_results
            
        except Exception as e:
            self.logger.warning(f"Context expansion failed, returning primary results: {e}")
            return primary_results

    async def discover_semantic_relationships(
        self, 
        documents: List[Dict[str, Any]], 
        similarity_threshold: float = 0.7
    ) -> Dict[str, Any]:
        """Discover semantic relationships between documents."""
        
        try:
            relationships = self.relationship_manager.discover_semantic_relationships(
                documents, similarity_threshold
            )
            
            # Create relationship summary
            summary = self.relationship_manager.create_relationship_summary(relationships)
            
            self.logger.info(f"Discovered {len(relationships)} semantic relationships")
            
            return {
                "relationships": relationships,
                "summary": summary,
                "success": True
            }
            
        except Exception as e:
            self.logger.error(f"Semantic relationship discovery failed: {e}")
            return {
                "relationships": [],
                "summary": {"error": str(e)},
                "success": False
            }

    async def get_relationship_insights(
        self, 
        search_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Get insights about relationships in search results."""
        
        try:
            # Discover relationships
            relationship_result = await self.discover_semantic_relationships(search_results)
            
            if not relationship_result["success"]:
                return relationship_result
            
            # Analyze result clustering
            clustering_insights = self._analyze_result_clustering(search_results)
            
            # Combine insights
            insights = {
                "relationships": relationship_result["summary"],
                "clustering": clustering_insights,
                "content_type_distribution": self._analyze_content_type_distribution(search_results),
                "quality_distribution": self._analyze_quality_distribution(search_results)
            }
            
            return insights
            
        except Exception as e:
            self.logger.error(f"Failed to get relationship insights: {e}")
            return {"error": str(e), "success": False}

    def _analyze_result_clustering(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze how search results cluster together."""
        
        if not results:
            return {"message": "No results to analyze"}
        
        # Group by content type
        content_type_groups = defaultdict(list)
        for result in results:
            content_type = result.get("metadata", {}).get("content_type", "unknown")
            content_type_groups[content_type].append(result)
        
        # Calculate cluster statistics
        cluster_stats = {}
        for content_type, group in content_type_groups.items():
            scores = [r.get("final_score", r.get("score", 0)) for r in group]
            cluster_stats[content_type] = {
                "count": len(group),
                "avg_score": sum(scores) / len(scores) if scores else 0,
                "min_score": min(scores) if scores else 0,
                "max_score": max(scores) if scores else 0
            }
        
        return {
            "total_clusters": len(content_type_groups),
            "cluster_distribution": list(content_type_groups.keys()),
            "cluster_statistics": cluster_stats
        }

    def _analyze_content_type_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the distribution of content types in search results."""
        
        if not results:
            return {"message": "No results to analyze"}
        
        content_type_counts = defaultdict(int)
        for result in results:
            content_type = result.get("metadata", {}).get("content_type", "unknown")
            content_type_counts[content_type] += 1
        
        total = len(results)
        distribution = {
            content_type: {
                "count": count,
                "percentage": (count / total) * 100
            }
            for content_type, count in content_type_counts.items()
        }
        
        return {
            "total_results": total,
            "distribution": distribution,
            "primary_content_type": max(content_type_counts.items(), key=lambda x: x[1])[0] if content_type_counts else None
        }

    def _analyze_quality_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze the quality distribution of search results."""
        
        if not results:
            return {"message": "No results to analyze"}
        
        quality_scores = []
        importance_scores = []
        completeness_scores = []
        
        for result in results:
            metadata = result.get("metadata", {})
            
            if "content_quality_score" in metadata:
                quality_scores.append(metadata["content_quality_score"])
            if "importance_score" in metadata:
                importance_scores.append(metadata["importance_score"])
            if "completeness_score" in metadata:
                completeness_scores.append(metadata["completeness_score"])
        
        def calculate_stats(scores):
            if not scores:
                return {"count": 0, "avg": 0, "min": 0, "max": 0}
            return {
                "count": len(scores),
                "avg": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores)
            }
        
        return {
            "content_quality": calculate_stats(quality_scores),
            "importance": calculate_stats(importance_scores),
            "completeness": calculate_stats(completeness_scores),
            "overall_quality": {
                "high_quality_count": len([s for s in quality_scores if s >= 0.8]),
                "medium_quality_count": len([s for s in quality_scores if 0.6 <= s < 0.8]),
                "low_quality_count": len([s for s in quality_scores if s < 0.6])
            }
        }
