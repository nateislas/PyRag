"""Intelligent search strategy selection and optimization for enhanced RAG."""

import copy
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

from .logging import get_logger

logger = get_logger(__name__)


class SearchStrategy(Enum):
    """Available search strategies for different coverage scenarios."""

    SELECTIVE = "selective"  # High coverage, focused search
    BALANCED = "balanced"  # Medium coverage, balanced approach
    COMPREHENSIVE = "comprehensive"  # Low coverage, broad search
    AGGRESSIVE = "aggressive"  # Very low coverage, exhaustive search


@dataclass
class SearchStrategyConfig:
    """Configuration for a specific search strategy."""

    strategy: SearchStrategy
    max_results: int
    include_low_quality: bool
    expand_context: bool
    search_depth: int
    collection_weights: Dict[str, float]
    metadata_boost_factors: Dict[str, float]


class IntelligentSearchStrategy:
    """Selects optimal search strategy based on coverage analysis and query context."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Strategy configurations for different coverage levels
        self.strategy_configs = {
            SearchStrategy.SELECTIVE: SearchStrategyConfig(
                strategy=SearchStrategy.SELECTIVE,
                max_results=20,
                include_low_quality=False,
                expand_context=False,
                search_depth=2,
                collection_weights={
                    "api_reference": 1.0,
                    "tutorials": 0.8,
                    "examples": 0.7,
                    "concepts": 0.6,
                    "overview": 0.5,
                },
                metadata_boost_factors={
                    "importance_score": 1.3,
                    "content_quality_score": 1.2,
                    "completeness_score": 1.1,
                },
            ),
            SearchStrategy.BALANCED: SearchStrategyConfig(
                strategy=SearchStrategy.BALANCED,
                max_results=30,
                include_low_quality=False,
                expand_context=True,
                search_depth=3,
                collection_weights={
                    "api_reference": 1.0,
                    "tutorials": 0.9,
                    "examples": 0.8,
                    "concepts": 0.7,
                    "overview": 0.6,
                    "configuration": 0.5,
                },
                metadata_boost_factors={
                    "importance_score": 1.2,
                    "content_quality_score": 1.1,
                    "completeness_score": 1.0,
                },
            ),
            SearchStrategy.COMPREHENSIVE: SearchStrategyConfig(
                strategy=SearchStrategy.COMPREHENSIVE,
                max_results=50,
                include_low_quality=True,
                expand_context=True,
                search_depth=4,
                collection_weights={
                    "api_reference": 1.0,
                    "tutorials": 1.0,
                    "examples": 1.0,
                    "concepts": 1.0,
                    "overview": 1.0,
                    "configuration": 0.8,
                    "troubleshooting": 0.7,
                    "changelog": 0.5,
                },
                metadata_boost_factors={
                    "importance_score": 1.0,
                    "content_quality_score": 0.9,
                    "completeness_score": 0.8,
                },
            ),
            SearchStrategy.AGGRESSIVE: SearchStrategyConfig(
                strategy=SearchStrategy.AGGRESSIVE,
                max_results=100,
                include_low_quality=True,
                expand_context=True,
                search_depth=5,
                collection_weights={
                    "api_reference": 1.0,
                    "tutorials": 1.0,
                    "examples": 1.0,
                    "concepts": 1.0,
                    "overview": 1.0,
                    "configuration": 1.0,
                    "troubleshooting": 1.0,
                    "changelog": 1.0,
                },
                metadata_boost_factors={
                    "importance_score": 0.8,
                    "content_quality_score": 0.7,
                    "completeness_score": 0.6,
                },
            ),
        }

    def select_strategy(
        self,
        query: str,
        library: str,
        coverage_data: Dict[str, Any],
        query_context: Optional[Dict[str, Any]] = None,
    ) -> SearchStrategyConfig:
        """Select optimal search strategy based on coverage and query analysis."""

        # Analyze coverage levels
        overall_coverage = coverage_data.get("overall_coverage", 0.0)
        content_type_coverage = coverage_data.get("content_type_coverage", {})
        library_coverage = coverage_data.get("library_coverage", 0.0)

        # Analyze query complexity and intent
        query_analysis = self._analyze_query(query, query_context)

        # Determine base strategy from coverage
        base_strategy = self._determine_base_strategy(
            overall_coverage, library_coverage, content_type_coverage
        )

        # Adjust strategy based on query analysis
        final_strategy = self._adjust_strategy_for_query(base_strategy, query_analysis)

        self.logger.info(
            f"Selected search strategy: {final_strategy.strategy.value} "
            f"(coverage: {overall_coverage:.1%}, query_complexity: {query_analysis['complexity']})"
        )

        return final_strategy

    def _determine_base_strategy(
        self,
        overall_coverage: float,
        library_coverage: float,
        content_type_coverage: Dict[str, float],
    ) -> SearchStrategy:
        """Determine base strategy from coverage analysis."""

        # Very low coverage: aggressive search
        if overall_coverage < 0.2 or library_coverage < 0.15:
            return SearchStrategy.AGGRESSIVE

        # Low coverage: comprehensive search
        elif overall_coverage < 0.4 or library_coverage < 0.3:
            return SearchStrategy.COMPREHENSIVE

        # Medium coverage: balanced approach
        elif overall_coverage < 0.7 or library_coverage < 0.6:
            return SearchStrategy.BALANCED

        # High coverage: selective search
        else:
            return SearchStrategy.SELECTIVE

    def _analyze_query(
        self, query: str, query_context: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Analyze query complexity and intent for strategy adjustment."""

        query_lower = query.lower()

        # Determine complexity
        complexity_indicators = {
            "simple": ["what is", "how to", "example", "basic", "simple"],
            "intermediate": [
                "how does",
                "difference between",
                "compare",
                "when to use",
            ],
            "complex": [
                "why does",
                "performance",
                "optimization",
                "advanced",
                "custom",
            ],
        }

        complexity = "intermediate"  # default
        for level, indicators in complexity_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                complexity = level
                break

        # Determine intent
        intent_indicators = {
            "api_reference": [
                "api",
                "class",
                "method",
                "function",
                "parameter",
                "return",
                "compare",
                "vs",
                "validation",
            ],
            "tutorial": [
                "tutorial",
                "guide",
                "step by step",
                "getting started",
                "learn",
                "how to",
            ],
            "examples": ["example", "code", "sample", "snippet", "demo"],
            "concepts": [
                "concept",
                "theory",
                "explanation",
                "understanding",
                "how it works",
                "what is",
            ],
            "troubleshooting": [
                "error",
                "problem",
                "issue",
                "fix",
                "debug",
                "troubleshoot",
                "why does",
                "fail",
            ],
        }

        intent = "general"
        for intent_type, indicators in intent_indicators.items():
            if any(indicator in query_lower for indicator in indicators):
                intent = intent_type
                break

        return {
            "complexity": complexity,
            "intent": intent,
            "length": len(query.split()),
            "has_code": any(
                char in query for char in ["(", ")", "[", "]", "{", "}", "=", ">", "<"]
            ),
        }

    def _adjust_strategy_for_query(
        self, base_strategy: SearchStrategy, query_analysis: Dict[str, Any]
    ) -> SearchStrategyConfig:
        """Adjust strategy configuration based on query analysis."""

        # Get base configuration and create a new instance
        base_config = self.strategy_configs[base_strategy]
        config = SearchStrategyConfig(
            strategy=base_config.strategy,
            max_results=base_config.max_results,
            include_low_quality=base_config.include_low_quality,
            expand_context=base_config.expand_context,
            search_depth=base_config.search_depth,
            collection_weights=base_config.collection_weights.copy(),
            metadata_boost_factors=base_config.metadata_boost_factors.copy(),
        )

        # Adjust based on query complexity
        if query_analysis["complexity"] == "simple":
            config.max_results = min(config.max_results, 15)
            config.search_depth = max(1, config.search_depth - 1)
        elif query_analysis["complexity"] == "complex":
            config.max_results = min(config.max_results + 10, 100)
            config.search_depth = min(5, config.search_depth + 1)
            config.expand_context = True

        # Adjust based on intent
        if query_analysis["intent"] in ["api_reference", "examples"]:
            # Boost API and examples collections
            config.collection_weights["api_reference"] *= 1.2
            config.collection_weights["examples"] *= 1.2
        elif query_analysis["intent"] == "tutorial":
            # Boost tutorials collection
            config.collection_weights["tutorials"] *= 1.3
        elif query_analysis["intent"] == "troubleshooting":
            # Include troubleshooting content
            config.collection_weights["troubleshooting"] *= 1.2
            config.include_low_quality = True

        return config

    def get_collection_search_order(
        self, strategy_config: SearchStrategyConfig
    ) -> List[str]:
        """Get ordered list of collections to search based on strategy."""

        # Sort collections by weight (highest first)
        sorted_collections = sorted(
            strategy_config.collection_weights.items(), key=lambda x: x[1], reverse=True
        )

        return [collection for collection, weight in sorted_collections if weight > 0.5]

    def calculate_metadata_boost(
        self, metadata: Dict[str, Any], strategy_config: SearchStrategyConfig
    ) -> float:
        """Calculate metadata-based boost factor for result ranking."""

        boost = 1.0

        for field, factor in strategy_config.metadata_boost_factors.items():
            if field in metadata:
                value = metadata[field]
                if isinstance(value, (int, float)) and value > 0:
                    # Apply boost based on field value and strategy factor
                    field_boost = 1.0 + (value * (factor - 1.0))
                    boost *= field_boost

        return boost
