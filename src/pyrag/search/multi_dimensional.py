"""Multi-dimensional search for comprehensive query coverage."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple
import json
import re

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SearchDimension:
    """Represents a knowledge dimension for multi-dimensional search."""
    name: str
    description: str
    keywords: List[str]
    search_query: str
    importance: float  # 0.0 to 1.0
    category: str  # "foundation", "implementation", "deployment", etc.
    

@dataclass
class DimensionResult:
    """Results from searching a specific dimension."""
    dimension: SearchDimension
    results: List[Dict[str, Any]]
    search_time: float
    result_count: int
    avg_score: float


@dataclass
class MultiDimensionalResult:
    """Complete multi-dimensional search result."""
    original_query: str
    dimensions_searched: List[SearchDimension]
    dimension_results: List[DimensionResult]
    synthesized_results: List[Dict[str, Any]]
    coverage_score: float
    total_search_time: float
    result_distribution: Dict[str, int]  # dimension -> count


class QueryDimensionDecomposer:
    """Decomposes complex queries into searchable dimensions."""
    
    def __init__(self, llm_client=None):
        """Initialize the query decomposer."""
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        
        # Predefined dimension templates for common query types
        self.dimension_templates = {
            "ai_agent_production": [
                SearchDimension(
                    name="architecture",
                    description="Agent architecture patterns and design principles",
                    keywords=["architecture", "design", "pattern", "structure", "component"],
                    search_query="agent architecture design patterns components structure",
                    importance=0.9,
                    category="foundation"
                ),
                SearchDimension(
                    name="implementation", 
                    description="Step-by-step implementation and code examples",
                    keywords=["implement", "build", "create", "code", "example", "tutorial"],
                    search_query="build create implement agent step by step code examples",
                    importance=0.9,
                    category="implementation"
                ),
                SearchDimension(
                    name="deployment",
                    description="Production deployment and infrastructure",
                    keywords=["deploy", "production", "docker", "kubernetes", "hosting"],
                    search_query="deploy production deployment docker containerization infrastructure",
                    importance=0.8,
                    category="deployment"
                ),
                SearchDimension(
                    name="monitoring",
                    description="Observability, monitoring, and debugging", 
                    keywords=["monitor", "observability", "logging", "metrics", "debug"],
                    search_query="monitoring observability logging metrics debugging tracing",
                    importance=0.7,
                    category="operations"
                ),
                SearchDimension(
                    name="scaling",
                    description="Performance optimization and scaling",
                    keywords=["scale", "performance", "optimization", "load", "throughput"],
                    search_query="scaling performance optimization load balancing throughput",
                    importance=0.7,
                    category="operations"
                ),
                SearchDimension(
                    name="security",
                    description="Security, authentication, and access control",
                    keywords=["security", "auth", "authentication", "access", "keys"],
                    search_query="security authentication authorization access control API keys",
                    importance=0.6,
                    category="security"
                ),
                SearchDimension(
                    name="testing",
                    description="Testing, validation, and quality assurance",
                    keywords=["test", "testing", "validation", "qa", "quality"],
                    search_query="testing validation quality assurance unit tests integration",
                    importance=0.5,
                    category="quality"
                )
            ],
            
            "web_framework_production": [
                SearchDimension(
                    name="setup",
                    description="Project setup and configuration",
                    keywords=["setup", "install", "config", "init", "getting started"],
                    search_query="setup installation configuration getting started project init",
                    importance=0.9,
                    category="foundation"
                ),
                SearchDimension(
                    name="api_design",
                    description="API endpoints and request handling",
                    keywords=["api", "endpoint", "route", "handler", "request", "response"],
                    search_query="API endpoints routes handlers request response REST",
                    importance=0.9,
                    category="implementation"
                ),
                SearchDimension(
                    name="authentication",
                    description="User authentication and authorization",
                    keywords=["auth", "login", "security", "token", "session", "oauth"],
                    search_query="authentication authorization login security tokens OAuth JWT",
                    importance=0.8,
                    category="security"
                ),
                SearchDimension(
                    name="database",
                    description="Database integration and data modeling",
                    keywords=["database", "db", "model", "orm", "sql", "migration"],
                    search_query="database models ORM SQL migrations data persistence",
                    importance=0.8,
                    category="data"
                ),
                SearchDimension(
                    name="deployment",
                    description="Production deployment and hosting",
                    keywords=["deploy", "production", "server", "hosting", "docker"],
                    search_query="deployment production hosting server docker containerization",
                    importance=0.7,
                    category="deployment"
                ),
                SearchDimension(
                    name="testing",
                    description="Testing strategies and frameworks",
                    keywords=["test", "testing", "unittest", "integration", "e2e"],
                    search_query="testing unit tests integration testing end-to-end QA",
                    importance=0.6,
                    category="quality"
                )
            ],
            
            "data_pipeline_production": [
                SearchDimension(
                    name="data_ingestion",
                    description="Data collection and ingestion patterns",
                    keywords=["ingest", "collect", "pipeline", "etl", "streaming"],
                    search_query="data ingestion collection ETL streaming pipeline batch",
                    importance=0.9,
                    category="foundation"
                ),
                SearchDimension(
                    name="processing",
                    description="Data transformation and processing",
                    keywords=["process", "transform", "clean", "validate", "enrich"],
                    search_query="data processing transformation cleaning validation enrichment",
                    importance=0.9,
                    category="implementation"
                ),
                SearchDimension(
                    name="storage",
                    description="Data storage and persistence strategies",
                    keywords=["storage", "database", "warehouse", "lake", "persistence"],
                    search_query="data storage database warehouse data lake persistence",
                    importance=0.8,
                    category="data"
                ),
                SearchDimension(
                    name="monitoring",
                    description="Pipeline monitoring and observability",
                    keywords=["monitor", "observability", "metrics", "alerting", "logging"],
                    search_query="pipeline monitoring observability metrics alerting logging",
                    importance=0.7,
                    category="operations"
                )
            ]
        }

    async def decompose_query(self, query: str, intent: Dict[str, Any], library: Optional[str] = None) -> List[SearchDimension]:
        """Decompose a query into searchable dimensions."""
        self.logger.info(f"Decomposing query into dimensions: {query}")
        
        # Try to match predefined patterns first
        dimensions = self._match_dimension_patterns(query, intent, library)
        
        # Use LLM for custom dimension extraction if available
        if self.llm_client and (not dimensions or intent.get("is_multi_faceted", False)):
            try:
                llm_dimensions = await self._llm_extract_dimensions(query, intent, library)
                # Merge predefined and LLM dimensions
                dimensions = self._merge_dimensions(dimensions, llm_dimensions)
            except Exception as e:
                self.logger.warning(f"LLM dimension extraction failed: {e}")
        
        # Ensure we have at least basic dimensions for comprehensive queries
        if not dimensions and intent.get("response_depth") == "comprehensive":
            dimensions = self._get_fallback_dimensions(query, library)
        
        # Customize dimensions based on library and query specifics
        dimensions = self._customize_dimensions(dimensions, query, library)
        
        self.logger.info(f"Generated {len(dimensions)} search dimensions")
        return dimensions

    def _match_dimension_patterns(self, query: str, intent: Dict[str, Any], library: Optional[str]) -> List[SearchDimension]:
        """Match query against predefined dimension patterns."""
        query_lower = query.lower()
        
        # AI Agent production pattern
        if any(term in query_lower for term in ["ai agent", "llm agent", "agent", "chatbot"]) and \
           any(term in query_lower for term in ["production", "deploy", "scale", "monitor"]):
            self.logger.info("Matched AI agent production pattern")
            return self.dimension_templates["ai_agent_production"].copy()
        
        # Web framework pattern
        web_frameworks = ["fastapi", "django", "flask", "streamlit", "gradio"]
        if any(framework in query_lower for framework in web_frameworks) and \
           any(term in query_lower for term in ["build", "create", "production", "deploy"]):
            self.logger.info("Matched web framework production pattern")
            return self.dimension_templates["web_framework_production"].copy()
        
        # Data pipeline pattern
        if any(term in query_lower for term in ["pipeline", "etl", "data processing", "data ingestion"]) and \
           any(term in query_lower for term in ["production", "deploy", "scale"]):
            self.logger.info("Matched data pipeline production pattern") 
            return self.dimension_templates["data_pipeline_production"].copy()
        
        return []

    async def _llm_extract_dimensions(self, query: str, intent: Dict[str, Any], library: Optional[str]) -> List[SearchDimension]:
        """Use LLM to extract custom dimensions for the query."""
        prompt = f"""Analyze this complex technical query and identify the key knowledge dimensions needed for a comprehensive answer.

Query: {query}
Library: {library or 'general'}
Intent Type: {intent.get('primary_type', 'general')}
Complexity: {intent.get('complexity_level', 'intermediate')}

Break down what an AI coding agent would need to provide a complete answer. Consider:
- Foundation concepts and theory
- Implementation steps and code examples
- Production deployment concerns  
- Operational aspects (monitoring, scaling, security)
- Quality assurance and testing
- Troubleshooting and maintenance

Return a JSON array of dimension objects:
[
  {{
    "name": "dimension_name",
    "description": "What this dimension covers",
    "keywords": ["keyword1", "keyword2", "keyword3"],
    "search_query": "targeted search query for this dimension",
    "importance": 0.0-1.0,
    "category": "foundation|implementation|deployment|operations|security|quality|data"
  }}
]

Example for "build FastAPI app with authentication":
[
  {{"name": "setup", "description": "Project setup and FastAPI basics", "keywords": ["setup", "install", "fastapi"], "search_query": "FastAPI setup installation getting started", "importance": 0.9, "category": "foundation"}},
  {{"name": "authentication", "description": "User authentication implementation", "keywords": ["auth", "login", "security"], "search_query": "FastAPI authentication login security JWT", "importance": 0.8, "category": "security"}}
]"""

        try:
            response = await self.llm_client.generate(prompt)
            # Extract JSON array from response
            json_match = re.search(r'\[.*\]', response, re.DOTALL)
            if json_match:
                dimension_dicts = json.loads(json_match.group(0))
                return [
                    SearchDimension(
                        name=dim["name"],
                        description=dim["description"], 
                        keywords=dim["keywords"],
                        search_query=dim["search_query"],
                        importance=dim["importance"],
                        category=dim["category"]
                    )
                    for dim in dimension_dicts
                ]
        except Exception as e:
            self.logger.warning(f"Failed to parse LLM dimension extraction: {e}")
        
        return []

    def _merge_dimensions(self, predefined: List[SearchDimension], llm_dimensions: List[SearchDimension]) -> List[SearchDimension]:
        """Merge predefined and LLM-generated dimensions, avoiding duplicates."""
        merged = predefined.copy()
        predefined_names = {dim.name.lower() for dim in predefined}
        
        for llm_dim in llm_dimensions:
            if llm_dim.name.lower() not in predefined_names:
                merged.append(llm_dim)
        
        return merged

    def _get_fallback_dimensions(self, query: str, library: Optional[str]) -> List[SearchDimension]:
        """Get basic fallback dimensions for comprehensive queries."""
        return [
            SearchDimension(
                name="overview",
                description="Overview and introduction",
                keywords=["overview", "introduction", "basics", "getting started"],
                search_query=f"{library or ''} overview introduction basics getting started".strip(),
                importance=0.8,
                category="foundation"
            ),
            SearchDimension(
                name="implementation", 
                description="Implementation and examples",
                keywords=["implementation", "example", "code", "tutorial"],
                search_query=f"{library or ''} implementation examples code tutorial".strip(),
                importance=0.9,
                category="implementation"
            ),
            SearchDimension(
                name="production",
                description="Production deployment",
                keywords=["production", "deployment", "best practices"],
                search_query=f"{library or ''} production deployment best practices".strip(),
                importance=0.7,
                category="deployment"
            )
        ]

    def _customize_dimensions(self, dimensions: List[SearchDimension], query: str, library: Optional[str]) -> List[SearchDimension]:
        """Customize dimensions based on specific query and library context."""
        if not library:
            return dimensions
        
        # Add library name to search queries for better targeting
        customized = []
        for dim in dimensions:
            customized_dim = SearchDimension(
                name=dim.name,
                description=dim.description,
                keywords=dim.keywords,
                search_query=f"{library} {dim.search_query}",
                importance=dim.importance,
                category=dim.category
            )
            customized.append(customized_dim)
        
        return customized


class MultiDimensionalSearchEngine:
    """Engine for executing multi-dimensional searches."""
    
    def __init__(self, vector_store, embedding_service, llm_client=None):
        """Initialize the multi-dimensional search engine."""
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        
        # Initialize dimension decomposer
        self.dimension_decomposer = QueryDimensionDecomposer(llm_client=llm_client)

    async def search_multi_dimensional(
        self, 
        query: str, 
        intent: Dict[str, Any], 
        library: Optional[str] = None,
        max_results_per_dimension: int = 5
    ) -> MultiDimensionalResult:
        """Execute multi-dimensional search."""
        self.logger.info(f"Starting multi-dimensional search for: {query}")
        start_time = asyncio.get_event_loop().time()
        
        # Step 1: Decompose query into dimensions
        dimensions = await self.dimension_decomposer.decompose_query(query, intent, library)
        
        if not dimensions:
            self.logger.warning("No dimensions identified, falling back to single search")
            return await self._fallback_single_search(query, intent, library)
        
        # Step 2: Execute parallel searches across dimensions
        self.logger.info(f"Executing parallel searches across {len(dimensions)} dimensions")
        dimension_tasks = [
            self._search_dimension(dim, library, max_results_per_dimension)
            for dim in dimensions
        ]
        
        dimension_results = await asyncio.gather(*dimension_tasks, return_exceptions=True)
        
        # Step 3: Process results and handle any failures
        valid_results = []
        for i, result in enumerate(dimension_results):
            if isinstance(result, Exception):
                self.logger.warning(f"Dimension {dimensions[i].name} search failed: {result}")
            else:
                valid_results.append(result)
        
        # Step 4: Synthesize and deduplicate results
        synthesized_results = self._synthesize_results(valid_results, query)
        
        # Step 5: Calculate metrics
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        coverage_score = self._calculate_coverage_score(valid_results, dimensions)
        result_distribution = {r.dimension.name: r.result_count for r in valid_results}
        
        self.logger.info(
            f"Multi-dimensional search completed in {total_time:.2f}s: "
            f"{len(synthesized_results)} results from {len(valid_results)} dimensions"
        )
        
        return MultiDimensionalResult(
            original_query=query,
            dimensions_searched=dimensions,
            dimension_results=valid_results,
            synthesized_results=synthesized_results,
            coverage_score=coverage_score,
            total_search_time=total_time,
            result_distribution=result_distribution
        )

    async def _search_dimension(self, dimension: SearchDimension, library: Optional[str], max_results: int) -> DimensionResult:
        """Search a specific dimension."""
        self.logger.debug(f"Searching dimension: {dimension.name}")
        start_time = asyncio.get_event_loop().time()
        
        try:
            # Build where clause for library filtering
            where_clause = {"library": library} if library else None
            
            # Execute search for this dimension
            results = await self.vector_store.search(
                query=dimension.search_query,
                collection_name="documents",
                n_results=max_results,
                where=where_clause
            )
            
            # Add dimension metadata to results
            for result in results:
                if "search_metadata" not in result:
                    result["search_metadata"] = {}
                result["search_metadata"]["dimension"] = dimension.name
                result["search_metadata"]["dimension_category"] = dimension.category
                result["search_metadata"]["dimension_importance"] = dimension.importance
            
            end_time = asyncio.get_event_loop().time()
            search_time = end_time - start_time
            
            # Calculate average score
            avg_score = sum(r.get("score", 0) for r in results) / len(results) if results else 0
            
            return DimensionResult(
                dimension=dimension,
                results=results,
                search_time=search_time,
                result_count=len(results),
                avg_score=avg_score
            )
            
        except Exception as e:
            self.logger.error(f"Error searching dimension {dimension.name}: {e}")
            raise

    def _synthesize_results(self, dimension_results: List[DimensionResult], query: str) -> List[Dict[str, Any]]:
        """Synthesize and deduplicate results from all dimensions."""
        self.logger.info("Synthesizing results from all dimensions")
        
        # Collect all results
        all_results = []
        for dim_result in dimension_results:
            all_results.extend(dim_result.results)
        
        # Deduplicate by content similarity (simple approach: by ID or URL)
        seen_ids = set()
        unique_results = []
        
        for result in all_results:
            result_id = result.get("id") or result.get("metadata", {}).get("source_url", "")
            if result_id and result_id not in seen_ids:
                seen_ids.add(result_id)
                unique_results.append(result)
        
        # Sort by importance-weighted score
        def weighted_score(result):
            base_score = result.get("score", 0)
            dimension_importance = result.get("search_metadata", {}).get("dimension_importance", 1.0)
            return base_score * dimension_importance
        
        unique_results.sort(key=weighted_score, reverse=True)
        
        self.logger.info(f"Synthesized {len(unique_results)} unique results from {len(all_results)} total")
        return unique_results

    def _calculate_coverage_score(self, dimension_results: List[DimensionResult], all_dimensions: List[SearchDimension]) -> float:
        """Calculate how well the results cover all dimensions."""
        if not all_dimensions:
            return 1.0
        
        # Weight by dimension importance and result count
        total_importance = sum(dim.importance for dim in all_dimensions)
        covered_importance = 0
        
        for dim_result in dimension_results:
            if dim_result.result_count > 0:
                # Scale coverage by result count (more results = better coverage)
                result_factor = min(1.0, dim_result.result_count / 3)  # Diminishing returns after 3 results
                covered_importance += dim_result.dimension.importance * result_factor
        
        return covered_importance / total_importance if total_importance > 0 else 0.0

    async def _fallback_single_search(self, query: str, intent: Dict[str, Any], library: Optional[str]) -> MultiDimensionalResult:
        """Fallback to single search when dimension decomposition fails."""
        self.logger.info("Falling back to single search")
        start_time = asyncio.get_event_loop().time()
        
        # Create a single "general" dimension
        general_dimension = SearchDimension(
            name="general",
            description="General search results",
            keywords=[],
            search_query=query,
            importance=1.0,
            category="general"
        )
        
        # Execute single search
        dimension_result = await self._search_dimension(general_dimension, library, 10)
        
        end_time = asyncio.get_event_loop().time()
        total_time = end_time - start_time
        
        return MultiDimensionalResult(
            original_query=query,
            dimensions_searched=[general_dimension],
            dimension_results=[dimension_result],
            synthesized_results=dimension_result.results,
            coverage_score=1.0,  # Single dimension always has full coverage
            total_search_time=total_time,
            result_distribution={"general": dimension_result.result_count}
        )