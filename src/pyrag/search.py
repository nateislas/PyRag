"""Search engine for PyRAG using ChromaDB with query expansion and reranking."""

from typing import Any, Dict, List, Optional
from .logging import get_logger

logger = get_logger(__name__)


class SimpleSearchEngine:
    """Search engine with query expansion, reformulation, and reranking."""

    def __init__(self, vector_store, embedding_service, llm_client=None, relationship_manager=None):
        """Initialize the enhanced search engine."""
        self.logger = get_logger(__name__)
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.relationship_manager = relationship_manager

    async def search(
        self,
        query: str,
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search with query expansion and reranking."""
        self.logger.info(f"Search for query: {query}")
        
        # Step 1: Query Analysis and Intent Classification
        query_intent = await self._analyze_query_intent(query, library, content_type)
        self.logger.info(f"Query intent: {query_intent}")
        
        # Step 2: Query Expansion and Reformulation
        expanded_queries = await self._expand_and_reformulate_query(query, query_intent, library)
        self.logger.info(f"Generated {len(expanded_queries)} query variants")
        
        # Step 3: Multi-Query Retrieval
        all_results = await self._multi_query_retrieval(
            expanded_queries, library, version, content_type, max_results
        )
        
        # Step 4: Intelligent Reranking with Context
        reranked_results = await self._intelligent_reranking(
            all_results, query, query_intent, max_results
        )
        
        # Step 5: Context Expansion (if relationship manager available)
        if self.relationship_manager and len(reranked_results) > 0:
            expanded_results = await self.relationship_manager.expand_search_context(
                reranked_results, max_expansion=3
            )
            final_results = expanded_results.primary_results + expanded_results.expanded_results
            self.logger.info(f"Context expansion added {len(expanded_results.expanded_results)} results")
        else:
            final_results = reranked_results
        
        # Ensure we don't exceed max_results
        final_results = final_results[:max_results]
        
        # Add search metadata
        for result in final_results:
            result["search_metadata"] = {
                "query_intent": query_intent,
                "query_variants_used": len(expanded_queries),
                "reranking_applied": True,
                "context_expanded": self.relationship_manager is not None
            }
        
        self.logger.info(f"Search returned {len(final_results)} results")
        return final_results

    async def _analyze_query_intent(self, query: str, library: Optional[str], content_type: Optional[str]) -> Dict[str, Any]:
        """Analyze query intent and classify the type of search needed."""
        intent = {
            "primary_type": "general",
            "content_preference": content_type or "overview",
            "library_focus": library,
            "complexity_level": "intermediate",
            "requires_code_examples": False,
            "requires_api_reference": False,
            "requires_tutorial": False
        }
        
        query_lower = query.lower()
        
        # Detect content type preferences
        if any(term in query_lower for term in ["how to", "tutorial", "guide", "getting started"]):
            intent["primary_type"] = "tutorial"
            intent["requires_tutorial"] = True
        elif any(term in query_lower for term in ["api", "function", "method", "class", "reference"]):
            intent["primary_type"] = "api_reference"
            intent["requires_api_reference"] = True
        elif any(term in query_lower for term in ["example", "sample", "code", "usage"]):
            intent["primary_type"] = "examples"
            intent["requires_code_examples"] = True
        
        # Detect complexity level
        if any(term in query_lower for term in ["basic", "simple", "beginner", "intro"]):
            intent["complexity_level"] = "beginner"
        elif any(term in query_lower for term in ["advanced", "complex", "expert", "detailed"]):
            intent["complexity_level"] = "advanced"
        
        # Use LLM for more sophisticated intent analysis if available
        if self.llm_client:
            try:
                llm_intent = await self._llm_query_intent_analysis(query, library)
                intent.update(llm_intent)
            except Exception as e:
                self.logger.warning(f"LLM intent analysis failed: {e}")
        
        return intent

    async def _llm_query_intent_analysis(self, query: str, library: Optional[str]) -> Dict[str, Any]:
        """Use LLM to analyze query intent more deeply."""
        prompt = f"""Analyze this search query for a Python library documentation system and return JSON with intent classification.

Query: {query}
Library: {library or 'general'}

Return ONLY a JSON object with these fields:
{{
  "primary_type": "[tutorial|api_reference|examples|overview|troubleshooting]",
  "content_preference": "[tutorial|api_reference|examples|overview|troubleshooting]",
  "complexity_level": "[beginner|intermediate|advanced]",
  "requires_code_examples": true/false,
  "requires_api_reference": true/false,
  "requires_tutorial": true/false,
  "key_concepts": ["list", "of", "key", "concepts"],
  "search_strategy": "[comprehensive|focused|quick]"
}}"""

        try:
            response = await self.llm_client.generate(prompt)
            # Extract JSON from response (basic parsing)
            import json
            import re
            
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM intent analysis: {e}")
        
        return {}

    async def _expand_and_reformulate_query(self, query: str, intent: Dict[str, Any], library: Optional[str]) -> List[str]:
        """Expand and reformulate the query based on intent and context."""
        queries = [query]  # Always include original query
        
        # Basic query expansion based on intent
        if intent["requires_code_examples"]:
            queries.append(f"{query} code examples")
            queries.append(f"{query} usage examples")
        
        if intent["requires_api_reference"]:
            queries.append(f"{query} API reference")
            queries.append(f"{query} function signature")
        
        if intent["requires_tutorial"]:
            queries.append(f"{query} tutorial guide")
            queries.append(f"{query} getting started")
        
        # Library-specific reformulation
        if library:
            queries.append(f"{library} {query}")
            queries.append(f"{query} in {library}")
        
        # Complexity-based reformulation
        if intent["complexity_level"] == "beginner":
            queries.append(f"{query} beginner tutorial")
            queries.append(f"{query} simple example")
        elif intent["complexity_level"] == "advanced":
            queries.append(f"{query} advanced usage")
            queries.append(f"{query} detailed implementation")
        
        # Use LLM for semantic query expansion if available
        if self.llm_client:
            try:
                llm_expansions = await self._llm_query_expansion(query, intent, library)
                queries.extend(llm_expansions)
            except Exception as e:
                self.logger.warning(f"LLM query expansion failed: {e}")
        
        # Remove duplicates and limit
        unique_queries = list(dict.fromkeys(queries))  # Preserve order
        return unique_queries[:8]  # Limit to 8 query variants

    async def _llm_query_expansion(self, query: str, intent: Dict[str, Any], library: Optional[str]) -> List[str]:
        """Use LLM to generate semantically related query variants."""
        prompt = f"""Generate 3-5 semantically related search queries for this documentation search.

Original Query: {query}
Intent: {intent['primary_type']}
Library: {library or 'general'}

Generate queries that would help find related or complementary information.
Return ONLY a JSON array of strings: ["query1", "query2", "query3"]"""

        try:
            response = await self.llm_client.generate(prompt)
            import json
            import re
            
            array_match = re.search(r'\[.*\]', response, re.DOTALL)
            if array_match:
                return json.loads(array_match.group(0))
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM query expansion: {e}")
        
        return []

    async def _multi_query_retrieval(
        self, 
        queries: List[str], 
        library: Optional[str], 
        version: Optional[str], 
        content_type: Optional[str], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Retrieve results using multiple query variants."""
        all_results = []
        result_scores = {}  # Track cumulative scores across queries
        
        for i, query in enumerate(queries):
            try:
                # Determine collection and build where clause
                collection_name = self._get_collection_name(content_type)
                where_clause = self._build_where_clause(library, version)
                
                # Search with this query variant
                results = await self.vector_store.search(
                    query=query,
                    collection_name=collection_name,
                    n_results=max_results * 2,  # Get more results for reranking
                    where=where_clause
                )
            
                # Add query variant metadata and adjust scores
                for result in results:
                    result_id = result.get("id", f"result_{i}_{len(all_results)}")
                    
                    # Initialize score tracking
                    if result_id not in result_scores:
                        result_scores[result_id] = {
                            "scores": [],
                            "query_matches": [],
                            "result": result
                        }
                    
                    # Track scores from this query variant
                    score = result.get("score", 0.5)
                    result_scores[result_id]["scores"].append(score)
                    result_scores[result_id]["query_matches"].append(query)
                    
                    # Add query variant info
                    result["query_variant"] = query
                    result["variant_rank"] = i
                
                all_results.extend(results)

            except Exception as e:
                self.logger.warning(f"Query variant {i} failed: {e}")
                continue
        
        # Deduplicate results and prepare for reranking
        unique_results = []
        seen_ids = set()
        
        for result in all_results:
            result_id = result.get("id", "")
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
            elif not result_id:
                # Handle results without IDs
                unique_results.append(result)
        
        return unique_results

    async def _intelligent_reranking(
        self, 
        results: List[Dict[str, Any]], 
        original_query: str, 
        intent: Dict[str, Any], 
        max_results: int
    ) -> List[Dict[str, Any]]:
        """Intelligently rerank results using multiple factors."""
        if not results:
            return []
        
        # Calculate enhanced scores for each result
        enhanced_results = []
        
        for result in results:
            enhanced_score = await self._calculate_score(result, original_query, intent)
            result["enhanced_score"] = enhanced_score
            result["final_score"] = enhanced_score
            enhanced_results.append(result)
        
        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["enhanced_score"], reverse=True)
        
        # Apply content type boosting based on intent
        boosted_results = self._apply_content_type_boosting(enhanced_results, intent)
        
        # Apply metadata-based reranking
        final_results = self._apply_metadata_reranking(boosted_results, intent)
        
        return final_results[:max_results]

    async def _calculate_score(self, result: Dict[str, Any], query: str, intent: Dict[str, Any]) -> float:
        """Calculate score using multiple factors."""
        base_score = result.get("score", 0.5)
        
        # Start with base score
        enhanced_score = base_score
        
        # Query relevance boosting
        query_relevance = self._calculate_query_relevance(result, query)
        enhanced_score += query_relevance * 0.3
        
        # Intent alignment boosting
        intent_alignment = self._calculate_intent_alignment(result, intent)
        enhanced_score += intent_alignment * 0.2
        
        # Metadata quality boosting
        metadata_quality = self._calculate_metadata_quality(result)
        enhanced_score += metadata_quality * 0.15
        
        # Content freshness boosting (if available)
        content_freshness = self._calculate_content_freshness(result)
        enhanced_score += content_freshness * 0.1
        
        # Normalize to 0-1 range
        return max(0.0, min(1.0, enhanced_score))

    def _calculate_query_relevance(self, result: Dict[str, Any], query: str) -> float:
        """Calculate how well the result matches the query."""
        content = result.get("content", "").lower()
        query_terms = query.lower().split()
        
        # Count query term matches
        matches = sum(1 for term in query_terms if term in content)
        total_terms = len(query_terms)
        
        if total_terms == 0:
            return 0.0
        
        return matches / total_terms

    def _calculate_intent_alignment(self, result: Dict[str, Any], intent: Dict[str, Any]) -> float:
        """Calculate how well the result aligns with search intent."""
        metadata = result.get("metadata", {})
        content_type = metadata.get("content_type", "")
        
        # Check content type alignment
        if intent["requires_tutorial"] and content_type in ["tutorial", "guide", "getting-started"]:
            return 0.8
        elif intent["requires_api_reference"] and content_type in ["api_reference", "reference"]:
            return 0.8
        elif intent["requires_code_examples"] and content_type in ["examples", "code"]:
            return 0.8
        
        # Check complexity alignment
        difficulty = metadata.get("difficulty_level", "intermediate")
        if intent["complexity_level"] == difficulty:
            return 0.6
        elif abs(self._complexity_distance(intent["complexity_level"], difficulty)) == 1:
            return 0.4
        
        return 0.2

    def _complexity_distance(self, level1: str, level2: str) -> int:
        """Calculate distance between complexity levels."""
        levels = ["beginner", "intermediate", "advanced"]
        try:
            idx1 = levels.index(level1)
            idx2 = levels.index(level2)
            return idx2 - idx1
        except ValueError:
            return 0

    def _calculate_metadata_quality(self, result: Dict[str, Any]) -> float:
        """Calculate metadata quality score."""
        metadata = result.get("metadata", {})
        
        # Count filled metadata fields
        important_fields = [
            "title", "content_type", "main_topic", "key_concepts", 
            "api_entities", "difficulty_level", "search_keywords"
        ]
        
        filled_fields = sum(1 for field in important_fields if metadata.get(field))
        return filled_fields / len(important_fields)

    def _calculate_content_freshness(self, result: Dict[str, Any]) -> float:
        """Calculate content freshness score."""
        # For now, return neutral score
        # In the future, this could use timestamps, version info, etc.
        return 0.5

    def _apply_content_type_boosting(self, results: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply content type boosting based on search intent."""
        for result in results:
            metadata = result.get("metadata", {})
            content_type = metadata.get("content_type", "")
            
            # Boost based on intent alignment
            if intent["requires_tutorial"] and content_type in ["tutorial", "guide"]:
                result["enhanced_score"] *= 1.2
            elif intent["requires_api_reference"] and content_type in ["api_reference", "reference"]:
                result["enhanced_score"] *= 1.2
            elif intent["requires_code_examples"] and content_type in ["examples", "code"]:
                result["enhanced_score"] *= 1.2
            
            # Recalculate final score
            result["final_score"] = result["enhanced_score"]
        
        return results

    def _apply_metadata_reranking(self, results: List[Dict[str, Any]], intent: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Apply final metadata-based reranking."""
        for result in results:
            metadata = result.get("metadata", {})
            
            # Boost results with rich metadata
            if metadata.get("key_concepts") and len(metadata["key_concepts"]) > 2:
                result["final_score"] *= 1.1
            
            if metadata.get("api_entities") and len(metadata["api_entities"]) > 0:
                result["final_score"] *= 1.05
            
            if metadata.get("code_examples") and metadata["code_examples"] != "No code examples detected":
                result["final_score"] *= 1.1
        
        # Final sort by final score
        results.sort(key=lambda x: x["final_score"], reverse=True)
        return results
    
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
        """Build a where clause for ChromaDB filtering."""
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
