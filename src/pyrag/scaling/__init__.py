"""
Scaling and performance optimization components for PyRAG.

This module provides horizontal scaling capabilities, load balancing,
auto-scaling, and performance monitoring for production deployments.
"""

from .load_balancer import LoadBalancer, QueryInfo, BackendInfo, BackendStatus
from .health_checker import HealthChecker, HealthCheckResult, HealthMetrics, HealthStatus
from .auto_scaler import AutoScaler, ScalingPolicy, ScalingDecision, ScalingAction
from .metrics import PerformanceMetrics, QueryMetrics, MetricsCollector, PerformanceMonitor
from .query_optimizer import QueryOptimizer, QueryPlan, OptimizationResult, QueryComplexity
from .cache_manager import CacheManager, QueryCache, CacheEntry, CacheStats

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
