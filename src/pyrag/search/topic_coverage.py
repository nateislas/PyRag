"""Topic coverage engine for ensuring comprehensive responses to complex queries."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set
import json
import re

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class TopicArea:
    """Represents a topic area that should be covered in a comprehensive response."""
    name: str
    keywords: List[str]
    importance: float  # 0.0 to 1.0
    category: str  # "implementation", "deployment", "monitoring", etc.


@dataclass
class CoverageAnalysis:
    """Analysis of topic coverage in search results."""
    required_topics: List[TopicArea]
    covered_topics: Set[str]
    missing_topics: List[str]
    coverage_score: float
    gaps_identified: List[str]


class TopicCoverageEngine:
    """Engine for analyzing and ensuring comprehensive topic coverage."""

    def __init__(self, llm_client=None, vector_store=None):
        """Initialize the topic coverage engine."""
        self.llm_client = llm_client
        self.vector_store = vector_store
        self.logger = get_logger(__name__)
        
        # Predefined topic patterns for common complex queries
        self.topic_patterns = {
            "ai_agent_production": [
                TopicArea("architecture", ["architecture", "design", "structure", "components"], 0.9, "foundation"),
                TopicArea("implementation", ["build", "create", "implement", "code", "develop"], 0.9, "implementation"),
                TopicArea("deployment", ["deploy", "production", "hosting", "docker", "kubernetes"], 0.8, "deployment"),
                TopicArea("monitoring", ["monitor", "logging", "observability", "metrics", "debugging"], 0.7, "operations"),
                TopicArea("scaling", ["scale", "performance", "optimization", "load", "throughput"], 0.7, "operations"),
                TopicArea("security", ["security", "authentication", "authorization", "keys", "access"], 0.6, "operations"),
                TopicArea("testing", ["test", "testing", "validation", "evaluation", "qa"], 0.6, "quality"),
                TopicArea("error_handling", ["error", "exception", "failure", "retry", "fallback"], 0.5, "reliability")
            ],
            "web_framework_production": [
                TopicArea("setup", ["setup", "installation", "getting started", "init"], 0.9, "foundation"),
                TopicArea("api_design", ["api", "endpoint", "route", "handler", "request"], 0.9, "implementation"),
                TopicArea("authentication", ["auth", "login", "security", "token", "session"], 0.8, "security"),
                TopicArea("database", ["database", "db", "model", "orm", "migration"], 0.8, "data"),
                TopicArea("deployment", ["deploy", "production", "server", "hosting", "docker"], 0.7, "deployment"),
                TopicArea("monitoring", ["logging", "metrics", "health", "monitoring", "alerts"], 0.6, "operations"),
                TopicArea("testing", ["test", "testing", "unittest", "integration"], 0.6, "quality")
            ]
        }

    async def extract_required_topics(self, query: str, intent: Dict[str, Any]) -> List[TopicArea]:
        """Extract all topic areas that should be covered for comprehensive answer."""
        self.logger.info(f"Extracting required topics for query: {query}")
        
        # Check for predefined patterns first
        required_topics = self._match_predefined_patterns(query)
        
        # Use LLM for more sophisticated topic extraction if available
        if self.llm_client and (not required_topics or intent.get("is_multi_faceted", False)):
            try:
                llm_topics = await self._llm_extract_topics(query, intent)
                # Merge with predefined topics, preferring LLM insights
                required_topics = self._merge_topic_lists(required_topics, llm_topics)
            except Exception as e:
                self.logger.warning(f"LLM topic extraction failed: {e}")
        
        # Ensure we have at least basic topics for comprehensive queries
        if intent.get("response_depth") == "comprehensive" and not required_topics:
            required_topics = self._get_fallback_topics(query)
        
        self.logger.info(f"Identified {len(required_topics)} required topic areas")
        return required_topics

    def _match_predefined_patterns(self, query: str) -> List[TopicArea]:
        """Match query against predefined topic patterns."""
        query_lower = query.lower()
        
        # AI Agent patterns
        if any(term in query_lower for term in ["ai agent", "llm agent", "chatbot agent"]) and \
           any(term in query_lower for term in ["production", "deploy", "scale", "monitor"]):
            return self.topic_patterns["ai_agent_production"].copy()
        
        # Web framework patterns
        web_frameworks = ["fastapi", "django", "flask", "streamlit"]
        if any(framework in query_lower for framework in web_frameworks) and \
           any(term in query_lower for term in ["production", "deploy", "complete", "full"]):
            return self.topic_patterns["web_framework_production"].copy()
        
        return []

    async def _llm_extract_topics(self, query: str, intent: Dict[str, Any]) -> List[TopicArea]:
        """Use LLM to extract required topics for comprehensive coverage."""
        prompt = f"""Analyze this complex query and identify all the topic areas that should be covered in a comprehensive answer for an AI coding agent.

Query: {query}
Intent: {intent.get('primary_type', 'general')}
Complexity: {intent.get('complexity_level', 'intermediate')}

Consider what an AI coding agent would need to provide a complete, actionable answer. Think about:
- Foundation concepts and architecture
- Step-by-step implementation details  
- Production deployment concerns
- Operational aspects (monitoring, scaling, security)
- Quality assurance and testing
- Common issues and troubleshooting

Return a JSON array of topic objects, each with:
{{
  "name": "topic_name",
  "keywords": ["keyword1", "keyword2", "keyword3"],
  "importance": 0.0-1.0,
  "category": "foundation|implementation|deployment|operations|quality|reliability"
}}

Example for "build production FastAPI app":
[
  {{"name": "project_setup", "keywords": ["setup", "structure", "dependencies"], "importance": 0.9, "category": "foundation"}},
  {{"name": "api_implementation", "keywords": ["endpoints", "routes", "handlers"], "importance": 0.9, "category": "implementation"}},
  {{"name": "authentication", "keywords": ["auth", "security", "tokens"], "importance": 0.8, "category": "security"}},
  {{"name": "deployment", "keywords": ["docker", "production", "server"], "importance": 0.7, "category": "deployment"}}
]"""

        try:
            response = await self.llm_client.generate(prompt)
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                topic_dicts = json.loads(json_match.group(0))
                return [
                    TopicArea(
                        name=topic["name"],
                        keywords=topic["keywords"],
                        importance=topic["importance"],
                        category=topic["category"]
                    )
                    for topic in topic_dicts
                ]
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM topic extraction: {e}")
        
        return []

    def _merge_topic_lists(self, predefined: List[TopicArea], llm_topics: List[TopicArea]) -> List[TopicArea]:
        """Merge predefined and LLM-generated topic lists, avoiding duplicates."""
        merged = predefined.copy()
        predefined_names = {topic.name.lower() for topic in predefined}
        
        for llm_topic in llm_topics:
            if llm_topic.name.lower() not in predefined_names:
                merged.append(llm_topic)
        
        return merged

    def _get_fallback_topics(self, query: str) -> List[TopicArea]:
        """Get basic fallback topics for comprehensive queries."""
        return [
            TopicArea("overview", ["overview", "introduction", "basics"], 0.8, "foundation"),
            TopicArea("implementation", ["implementation", "code", "example"], 0.9, "implementation"),
            TopicArea("best_practices", ["best practices", "patterns", "recommendations"], 0.6, "quality")
        ]

    async def analyze_coverage_gaps(self, results: List[Dict[str, Any]], required_topics: List[TopicArea]) -> CoverageAnalysis:
        """Analyze what topics are missing from current results."""
        self.logger.info(f"Analyzing coverage gaps for {len(required_topics)} required topics")
        
        # Extract covered topics from results
        covered_topics = set()
        for result in results:
            # Check content for topic keywords
            content = result.get("content", "").lower()
            metadata = result.get("metadata", {})
            
            # Check against each required topic
            for topic in required_topics:
                if self._topic_covered_in_content(topic, content, metadata):
                    covered_topics.add(topic.name)
        
        # Identify missing topics
        required_topic_names = {topic.name for topic in required_topics}
        missing_topics = list(required_topic_names - covered_topics)
        
        # Calculate coverage score (weighted by importance)
        total_importance = sum(topic.importance for topic in required_topics)
        covered_importance = sum(
            topic.importance for topic in required_topics 
            if topic.name in covered_topics
        )
        coverage_score = covered_importance / total_importance if total_importance > 0 else 0.0
        
        # Generate specific gap descriptions
        gaps_identified = [
            f"Missing {topic.category} topic: {topic.name}"
            for topic in required_topics
            if topic.name in missing_topics
        ]
        
        self.logger.info(f"Coverage analysis: {len(covered_topics)}/{len(required_topics)} topics covered, score: {coverage_score:.2f}")
        
        return CoverageAnalysis(
            required_topics=required_topics,
            covered_topics=covered_topics,
            missing_topics=missing_topics,
            coverage_score=coverage_score,
            gaps_identified=gaps_identified
        )

    def _topic_covered_in_content(self, topic: TopicArea, content: str, metadata: Dict[str, Any]) -> bool:
        """Check if a topic is adequately covered in content."""
        # Check if any topic keywords appear in content
        keyword_matches = sum(1 for keyword in topic.keywords if keyword in content)
        keyword_coverage = keyword_matches / len(topic.keywords)
        
        # Check metadata for additional signals
        metadata_signals = 0
        if "key_concepts" in metadata:
            key_concepts = metadata.get("key_concepts", [])
            if isinstance(key_concepts, list):
                metadata_signals = sum(
                    1 for concept in key_concepts 
                    if any(keyword in concept.lower() for keyword in topic.keywords)
                )
        
        # Topic is covered if we have good keyword coverage or strong metadata signals
        return keyword_coverage >= 0.3 or metadata_signals >= 2

    async def fill_coverage_gaps(self, gaps: List[str], library: Optional[str], original_query: str) -> List[Dict[str, Any]]:
        """Search specifically for missing topic areas."""
        if not gaps or not self.vector_store:
            return []
        
        self.logger.info(f"Filling {len(gaps)} coverage gaps")
        gap_results = []
        
        for gap in gaps:
            # Extract topic name from gap description
            topic_name = gap.split(": ")[-1] if ": " in gap else gap
            
            # Generate targeted search query
            targeted_query = f"{original_query} {topic_name}"
            if library:
                targeted_query += f" {library}"
            
            try:
                # Search for this specific gap
                results = await self.vector_store.search(
                    query=targeted_query,
                    collection_name="documents",
                    n_results=3,
                    where={"library_name": library} if library else None
                )
                
                # Mark these as gap-filling results
                for result in results:
                    result["gap_fill_for"] = topic_name
                    result["targeted_search"] = True
                
                gap_results.extend(results)
                
            except Exception as e:
                self.logger.warning(f"Failed to fill gap for {topic_name}: {e}")
        
        self.logger.info(f"Retrieved {len(gap_results)} additional results to fill gaps")
        return gap_results

    async def ensure_comprehensive_coverage(self, results: List[Dict[str, Any]], query: str, intent: Dict[str, Any], library: Optional[str] = None) -> List[Dict[str, Any]]:
        """Main method to ensure comprehensive topic coverage."""
        # Extract required topics
        required_topics = await self.extract_required_topics(query, intent)
        
        if not required_topics:
            # No specific coverage requirements, return original results
            return results
        
        # Analyze coverage gaps
        coverage = await self.analyze_coverage_gaps(results, required_topics)
        
        # If coverage is good enough, return original results
        if coverage.coverage_score >= 0.8:
            self.logger.info(f"Coverage sufficient ({coverage.coverage_score:.2f}), no gap filling needed")
            return results
        
        # Fill coverage gaps
        gap_results = await self.fill_coverage_gaps(coverage.gaps_identified, library, query)
        
        # Combine original results with gap-filling results
        enhanced_results = results + gap_results
        
        # Add coverage metadata to all results
        for result in enhanced_results:
            if "search_metadata" not in result:
                result["search_metadata"] = {}
            result["search_metadata"]["coverage_analysis"] = {
                "required_topics": [topic.name for topic in required_topics],
                "coverage_score": coverage.coverage_score,
                "gaps_filled": len(gap_results) > 0
            }
        
        self.logger.info(f"Enhanced results: {len(results)} original + {len(gap_results)} gap-filling = {len(enhanced_results)} total")
        return enhanced_results