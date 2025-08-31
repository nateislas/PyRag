"""
Query optimization for PyRAG.

This module provides query optimization capabilities including query planning,
complexity analysis, and execution optimization for improved performance.
"""

import asyncio
import time
import re
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class QueryComplexity(Enum):
    """Query complexity levels."""
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    VERY_COMPLEX = "very_complex"


@dataclass
class QueryPlan:
    """Optimized query execution plan."""
    query: str
    complexity: QueryComplexity
    estimated_duration: float
    execution_steps: List[Dict[str, Any]]
    optimization_hints: List[str]
    resource_requirements: Dict[str, Any]
    parallelizable: bool
    cache_key: Optional[str] = None


@dataclass
class OptimizationResult:
    """Result of query optimization."""
    original_query: str
    optimized_query: str
    improvements: List[str]
    estimated_gain: float
    confidence: float
    plan: QueryPlan


class QueryOptimizer:
    """
    Query optimizer for PyRAG.
    
    Provides:
    - Query complexity analysis
    - Execution plan optimization
    - Performance hints and recommendations
    - Resource requirement estimation
    - Parallelization opportunities
    """
    
    def __init__(self):
        """Initialize the query optimizer."""
        self.complexity_patterns = {
            QueryComplexity.SIMPLE: [
                r"how to",
                r"what is",
                r"basic",
                r"simple",
                r"example",
                r"tutorial"
            ],
            QueryComplexity.MEDIUM: [
                r"compare",
                r"difference",
                r"advantages",
                r"disadvantages",
                r"best practices",
                r"configuration"
            ],
            QueryComplexity.COMPLEX: [
                r"build.*using.*and",
                r"integrate.*with",
                r"advanced",
                r"optimization",
                r"performance",
                r"scaling"
            ],
            QueryComplexity.VERY_COMPLEX: [
                r"architecture",
                r"distributed",
                r"microservices",
                r"complex.*system",
                r"enterprise"
            ]
        }
        
        self.optimization_rules = [
            self._optimize_keyword_queries,
            self._optimize_comparison_queries,
            self._optimize_integration_queries,
            self._optimize_performance_queries
        ]
        
        # Query statistics for optimization
        self.query_stats = defaultdict(int)
        self.execution_times = defaultdict(list)
    
    def analyze_complexity(self, query: str) -> QueryComplexity:
        """
        Analyze query complexity.
        
        Args:
            query: Query string to analyze
            
        Returns:
            Query complexity level
        """
        query_lower = query.lower()
        
        # Count complexity indicators
        complexity_scores = defaultdict(int)
        
        for complexity, patterns in self.complexity_patterns.items():
            for pattern in patterns:
                if re.search(pattern, query_lower):
                    complexity_scores[complexity] += 1
        
        # Additional complexity factors
        word_count = len(query.split())
        if word_count > 20:
            complexity_scores[QueryComplexity.VERY_COMPLEX] += 1
        elif word_count > 10:
            complexity_scores[QueryComplexity.COMPLEX] += 1
        
        # Check for technical terms
        technical_terms = [
            "api", "database", "server", "client", "protocol", "framework",
            "library", "dependency", "configuration", "deployment"
        ]
        
        tech_term_count = sum(1 for term in technical_terms if term in query_lower)
        if tech_term_count > 3:
            complexity_scores[QueryComplexity.COMPLEX] += 1
        elif tech_term_count > 1:
            complexity_scores[QueryComplexity.MEDIUM] += 1
        
        # Determine complexity based on scores
        if complexity_scores[QueryComplexity.VERY_COMPLEX] > 0:
            return QueryComplexity.VERY_COMPLEX
        elif complexity_scores[QueryComplexity.COMPLEX] > 1:
            return QueryComplexity.COMPLEX
        elif complexity_scores[QueryComplexity.MEDIUM] > 0:
            return QueryComplexity.MEDIUM
        else:
            return QueryComplexity.SIMPLE
    
    def estimate_duration(self, complexity: QueryComplexity, query: str) -> float:
        """
        Estimate query execution duration.
        
        Args:
            complexity: Query complexity level
            query: Query string
            
        Returns:
            Estimated duration in seconds
        """
        base_durations = {
            QueryComplexity.SIMPLE: 0.1,
            QueryComplexity.MEDIUM: 0.5,
            QueryComplexity.COMPLEX: 2.0,
            QueryComplexity.VERY_COMPLEX: 5.0
        }
        
        base_duration = base_durations[complexity]
        
        # Adjust based on query characteristics
        word_count = len(query.split())
        duration_multiplier = 1.0 + (word_count - 5) * 0.1  # 10% per additional word
        
        # Check for specific patterns that might affect duration
        if "compare" in query.lower():
            duration_multiplier *= 1.5  # Comparisons take longer
        
        if "integration" in query.lower():
            duration_multiplier *= 2.0  # Integration queries are complex
        
        if "performance" in query.lower():
            duration_multiplier *= 1.8  # Performance analysis is detailed
        
        return base_duration * duration_multiplier
    
    def generate_execution_plan(self, query: str) -> QueryPlan:
        """
        Generate optimized execution plan for a query.
        
        Args:
            query: Query string
            
        Returns:
            Optimized execution plan
        """
        complexity = self.analyze_complexity(query)
        estimated_duration = self.estimate_duration(complexity, query)
        
        # Generate execution steps
        execution_steps = self._generate_execution_steps(query, complexity)
        
        # Generate optimization hints
        optimization_hints = self._generate_optimization_hints(query, complexity)
        
        # Estimate resource requirements
        resource_requirements = self._estimate_resource_requirements(complexity, query)
        
        # Determine if query can be parallelized
        parallelizable = self._can_parallelize(query, complexity)
        
        # Generate cache key
        cache_key = self._generate_cache_key(query)
        
        return QueryPlan(
            query=query,
            complexity=complexity,
            estimated_duration=estimated_duration,
            execution_steps=execution_steps,
            optimization_hints=optimization_hints,
            resource_requirements=resource_requirements,
            parallelizable=parallelizable,
            cache_key=cache_key
        )
    
    def _generate_execution_steps(self, query: str, complexity: QueryComplexity) -> List[Dict[str, Any]]:
        """Generate execution steps for the query."""
        steps = []
        
        # Step 1: Query preprocessing
        steps.append({
            "step": 1,
            "name": "query_preprocessing",
            "description": "Preprocess and normalize query",
            "estimated_duration": 0.01,
            "parallelizable": False
        })
        
        # Step 2: Vector search
        steps.append({
            "step": 2,
            "name": "vector_search",
            "description": "Perform vector similarity search",
            "estimated_duration": 0.1,
            "parallelizable": True
        })
        
        # Step 3: Content filtering
        steps.append({
            "step": 3,
            "name": "content_filtering",
            "description": "Filter and rank search results",
            "estimated_duration": 0.05,
            "parallelizable": True
        })
        
        # Additional steps for complex queries
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            steps.append({
                "step": 4,
                "name": "context_analysis",
                "description": "Analyze context and relationships",
                "estimated_duration": 0.2,
                "parallelizable": False
            })
            
            steps.append({
                "step": 5,
                "name": "result_synthesis",
                "description": "Synthesize and format results",
                "estimated_duration": 0.1,
                "parallelizable": False
            })
        
        return steps
    
    def _generate_optimization_hints(self, query: str, complexity: QueryComplexity) -> List[str]:
        """Generate optimization hints for the query."""
        hints = []
        
        # General hints
        if complexity == QueryComplexity.SIMPLE:
            hints.append("Consider caching simple queries for better performance")
        
        if "compare" in query.lower():
            hints.append("Use parallel processing for comparison queries")
        
        if "integration" in query.lower():
            hints.append("Consider breaking integration queries into smaller parts")
        
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            hints.append("Consider using query decomposition for complex queries")
            hints.append("Enable result caching for repeated complex queries")
        
        # Query-specific hints
        word_count = len(query.split())
        if word_count > 15:
            hints.append("Consider simplifying query for better performance")
        
        return hints
    
    def _estimate_resource_requirements(self, complexity: QueryComplexity, query: str) -> Dict[str, Any]:
        """Estimate resource requirements for query execution."""
        base_memory = {
            QueryComplexity.SIMPLE: 50,  # MB
            QueryComplexity.MEDIUM: 100,
            QueryComplexity.COMPLEX: 200,
            QueryComplexity.VERY_COMPLEX: 500
        }
        
        base_cpu = {
            QueryComplexity.SIMPLE: 0.1,  # CPU cores
            QueryComplexity.MEDIUM: 0.2,
            QueryComplexity.COMPLEX: 0.5,
            QueryComplexity.VERY_COMPLEX: 1.0
        }
        
        # Adjust based on query characteristics
        word_count = len(query.split())
        memory_multiplier = 1.0 + (word_count - 5) * 0.05
        cpu_multiplier = 1.0 + (word_count - 5) * 0.02
        
        return {
            "memory_mb": base_memory[complexity] * memory_multiplier,
            "cpu_cores": base_cpu[complexity] * cpu_multiplier,
            "network_requests": self._estimate_network_requests(complexity, query),
            "disk_io_mb": self._estimate_disk_io(complexity, query)
        }
    
    def _estimate_network_requests(self, complexity: QueryComplexity, query: str) -> int:
        """Estimate number of network requests needed."""
        base_requests = {
            QueryComplexity.SIMPLE: 1,
            QueryComplexity.MEDIUM: 2,
            QueryComplexity.COMPLEX: 3,
            QueryComplexity.VERY_COMPLEX: 5
        }
        
        # Additional requests for specific patterns
        additional_requests = 0
        if "compare" in query.lower():
            additional_requests += 1
        if "integration" in query.lower():
            additional_requests += 2
        
        return base_requests[complexity] + additional_requests
    
    def _estimate_disk_io(self, complexity: QueryComplexity, query: str) -> float:
        """Estimate disk I/O in MB."""
        base_io = {
            QueryComplexity.SIMPLE: 1.0,
            QueryComplexity.MEDIUM: 2.0,
            QueryComplexity.COMPLEX: 5.0,
            QueryComplexity.VERY_COMPLEX: 10.0
        }
        
        return base_io[complexity]
    
    def _can_parallelize(self, query: str, complexity: QueryComplexity) -> bool:
        """Determine if query can be parallelized."""
        # Simple queries are usually not worth parallelizing
        if complexity == QueryComplexity.SIMPLE:
            return False
        
        # Queries with comparison or multiple components can be parallelized
        if "compare" in query.lower() or "and" in query.lower():
            return True
        
        # Complex queries can often be broken down
        if complexity in [QueryComplexity.COMPLEX, QueryComplexity.VERY_COMPLEX]:
            return True
        
        return False
    
    def _generate_cache_key(self, query: str) -> str:
        """Generate cache key for the query."""
        # Simple hash-based cache key
        import hashlib
        normalized_query = query.lower().strip()
        return hashlib.md5(normalized_query.encode()).hexdigest()
    
    def optimize_query(self, query: str) -> OptimizationResult:
        """
        Optimize a query for better performance.
        
        Args:
            query: Original query string
            
        Returns:
            Optimization result with improvements
        """
        original_plan = self.generate_execution_plan(query)
        optimized_query = query
        improvements = []
        estimated_gain = 0.0
        
        # Apply optimization rules
        for rule in self.optimization_rules:
            result = rule(query, original_plan)
            if result:
                optimized_query, improvement, gain = result
                improvements.append(improvement)
                estimated_gain += gain
        
        # Generate optimized plan
        optimized_plan = self.generate_execution_plan(optimized_query)
        
        # Calculate confidence based on improvements
        confidence = min(0.9, 0.5 + len(improvements) * 0.1)
        
        return OptimizationResult(
            original_query=query,
            optimized_query=optimized_query,
            improvements=improvements,
            estimated_gain=estimated_gain,
            confidence=confidence,
            plan=optimized_plan
        )
    
    def _optimize_keyword_queries(self, query: str, plan: QueryPlan) -> Optional[Tuple[str, str, float]]:
        """Optimize keyword-based queries."""
        # Add specific keywords for better search results
        if "how to" in query.lower() and "example" not in query.lower():
            optimized = query + " example"
            return optimized, "Added 'example' keyword for better results", 0.1
        
        return None
    
    def _optimize_comparison_queries(self, query: str, plan: QueryPlan) -> Optional[Tuple[str, str, float]]:
        """Optimize comparison queries."""
        if "compare" in query.lower():
            # Ensure comparison queries are well-structured
            if "vs" not in query.lower() and "versus" not in query.lower():
                # Try to identify comparison targets
                words = query.lower().split()
                if len(words) > 3:
                    # Add structure for better comparison
                    optimized = f"compare {words[-2]} vs {words[-1]}"
                    return optimized, "Restructured comparison query", 0.2
        
        return None
    
    def _optimize_integration_queries(self, query: str, plan: QueryPlan) -> Optional[Tuple[str, str, float]]:
        """Optimize integration queries."""
        if "integration" in query.lower() or "integrate" in query.lower():
            # Add specific integration keywords
            if "api" not in query.lower():
                optimized = query + " API integration"
                return optimized, "Added API integration context", 0.15
        
        return None
    
    def _optimize_performance_queries(self, query: str, plan: QueryPlan) -> Optional[Tuple[str, str, float]]:
        """Optimize performance-related queries."""
        if "performance" in query.lower() or "optimization" in query.lower():
            # Add performance-specific keywords
            if "best practices" not in query.lower():
                optimized = query + " best practices"
                return optimized, "Added best practices context", 0.1
        
        return None
    
    def record_execution_time(self, query: str, execution_time: float):
        """
        Record actual execution time for a query.
        
        Args:
            query: Query string
            execution_time: Actual execution time in seconds
        """
        cache_key = self._generate_cache_key(query)
        self.execution_times[cache_key].append(execution_time)
        
        # Keep only recent execution times
        if len(self.execution_times[cache_key]) > 10:
            self.execution_times[cache_key] = self.execution_times[cache_key][-5:]
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics for optimization."""
        stats = {
            "total_queries": sum(self.query_stats.values()),
            "complexity_distribution": {},
            "avg_execution_times": {},
            "optimization_effectiveness": {}
        }
        
        # Calculate complexity distribution
        for complexity in QueryComplexity:
            stats["complexity_distribution"][complexity.value] = 0
        
        # Calculate average execution times
        for cache_key, times in self.execution_times.items():
            if times:
                avg_time = sum(times) / len(times)
                stats["avg_execution_times"][cache_key] = avg_time
        
        return stats
