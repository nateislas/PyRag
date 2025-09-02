"""
Scaling and performance optimization components for PyRAG.

This module provides horizontal scaling capabilities, load balancing,
auto-scaling, and performance monitoring for production deployments.
"""

from .auto_scaler import AutoScaler, ScalingAction, ScalingDecision, ScalingPolicy
from .cache_manager import CacheEntry, CacheManager, CacheStats, QueryCache
from .health_checker import (
    HealthChecker,
    HealthCheckResult,
    HealthMetrics,
    HealthStatus,
)
from .load_balancer import BackendInfo, BackendStatus, LoadBalancer, QueryInfo
from .metrics import (
    MetricsCollector,
    PerformanceMetrics,
    PerformanceMonitor,
    QueryMetrics,
)
from .query_optimizer import (
    OptimizationResult,
    QueryComplexity,
    QueryOptimizer,
    QueryPlan,
)

__all__ = [
    "LoadBalancer",
    "QueryInfo",
    "BackendInfo",
    "BackendStatus",
    "HealthChecker",
    "HealthCheckResult",
    "HealthMetrics",
    "HealthStatus",
    "AutoScaler",
    "ScalingPolicy",
    "ScalingDecision",
    "ScalingAction",
    "PerformanceMetrics",
    "QueryMetrics",
    "MetricsCollector",
    "PerformanceMonitor",
    "QueryOptimizer",
    "QueryPlan",
    "OptimizationResult",
    "QueryComplexity",
    "CacheManager",
    "QueryCache",
    "CacheEntry",
    "CacheStats",
]
