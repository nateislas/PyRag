"""
Load balancing for horizontal scaling of PyRAG.

This module provides intelligent query routing, load distribution,
and failover handling for production deployments.
"""

import asyncio
import time
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Any
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class BackendStatus(Enum):
    """Backend health status."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class BackendInfo:
    """Information about a backend instance."""
    id: str
    url: str
    status: BackendStatus
    last_health_check: float
    response_time_avg: float
    error_rate: float
    load_factor: float
    max_concurrent_queries: int
    current_queries: int = 0
    total_queries: int = 0
    total_errors: int = 0


@dataclass
class QueryInfo:
    """Information about a query for routing decisions."""
    query: str
    complexity: str  # "simple", "medium", "complex"
    estimated_duration: float
    priority: int = 1
    user_id: Optional[str] = None
    session_id: Optional[str] = None


class LoadBalancer:
    """
    Intelligent load balancer for PyRAG queries.
    
    Provides query routing based on:
    - Backend health and performance
    - Query complexity and resource requirements
    - Load distribution and fairness
    - Failover handling
    """
    
    def __init__(
        self,
        backends: List[Dict[str, Any]],
        health_check_interval: float = 30.0,
        max_retries: int = 3,
        circuit_breaker_threshold: int = 5,
        circuit_breaker_timeout: float = 60.0
    ):
        """
        Initialize the load balancer.
        
        Args:
            backends: List of backend configurations
            health_check_interval: Interval between health checks (seconds)
            max_retries: Maximum retry attempts for failed queries
            circuit_breaker_threshold: Number of failures before circuit breaker opens
            circuit_breaker_timeout: Timeout for circuit breaker reset (seconds)
        """
        self.backends: Dict[str, BackendInfo] = {}
        self.health_check_interval = health_check_interval
        self.max_retries = max_retries
        self.circuit_breaker_threshold = circuit_breaker_threshold
        self.circuit_breaker_timeout = circuit_breaker_timeout
        
        # Initialize backends
        for backend_config in backends:
            backend_id = backend_config["id"]
            self.backends[backend_id] = BackendInfo(
                id=backend_id,
                url=backend_config["url"],
                status=BackendStatus.UNKNOWN,
                last_health_check=time.time(),
                response_time_avg=0.0,
                error_rate=0.0,
                load_factor=1.0,
                max_concurrent_queries=backend_config.get("max_concurrent_queries", 100)
            )
        
        # Performance tracking
        self.query_history: List[Dict[str, Any]] = []
        self.routing_decisions: Dict[str, int] = defaultdict(int)
        
        # Start health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._start_health_checking()
    
    def _start_health_checking(self):
        """Start the health checking background task."""
        if self._health_check_task is None or self._health_check_task.done():
            self._health_check_task = asyncio.create_task(self._health_check_loop())
    
    async def _health_check_loop(self):
        """Background health checking loop."""
        while True:
            try:
                await self._check_all_backends()
                await asyncio.sleep(self.health_check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error
    
    async def _check_all_backends(self):
        """Check health of all backends."""
        tasks = []
        for backend in self.backends.values():
            tasks.append(self._check_backend_health(backend))
        
        await asyncio.gather(*tasks, return_exceptions=True)
    
    async def _check_backend_health(self, backend: BackendInfo):
        """Check health of a single backend."""
        try:
            start_time = time.time()
            
            # Simple health check - could be enhanced with actual API call
            # For now, we'll simulate health checks
            await asyncio.sleep(0.1)  # Simulate network delay
            
            response_time = time.time() - start_time
            
            # Update backend metrics
            backend.last_health_check = time.time()
            backend.response_time_avg = (
                backend.response_time_avg * 0.9 + response_time * 0.1
            )
            
            # Determine status based on metrics
            if backend.error_rate > 0.1:  # >10% error rate
                backend.status = BackendStatus.UNHEALTHY
            elif backend.response_time_avg > 1.0:  # >1s average response time
                backend.status = BackendStatus.DEGRADED
            else:
                backend.status = BackendStatus.HEALTHY
                
        except Exception as e:
            logger.error(f"Health check failed for backend {backend.id}: {e}")
            backend.status = BackendStatus.UNHEALTHY
            backend.error_rate = min(backend.error_rate + 0.1, 1.0)
    
    async def route_query(self, query_info: QueryInfo) -> Optional[str]:
        """
        Route a query to the best available backend.
        
        Args:
            query_info: Information about the query for routing decisions
            
        Returns:
            Backend ID to route the query to, or None if no backends available
        """
        available_backends = self._get_available_backends()
        
        if not available_backends:
            logger.warning("No healthy backends available for query routing")
            return None
        
        # Select backend based on routing strategy
        backend_id = self._select_backend(available_backends, query_info)
        
        if backend_id:
            # Update metrics
            self.backends[backend_id].current_queries += 1
            self.backends[backend_id].total_queries += 1
            self.routing_decisions[backend_id] += 1
            
            # Track query history
            self.query_history.append({
                "timestamp": time.time(),
                "query": query_info.query,
                "backend_id": backend_id,
                "complexity": query_info.complexity,
                "priority": query_info.priority
            })
            
            # Keep only recent history
            if len(self.query_history) > 1000:
                self.query_history = self.query_history[-500:]
        
        return backend_id
    
    def _get_available_backends(self) -> List[BackendInfo]:
        """Get list of available backends."""
        return [
            backend for backend in self.backends.values()
            if backend.status in [BackendStatus.HEALTHY, BackendStatus.DEGRADED]
            and backend.current_queries < backend.max_concurrent_queries
        ]
    
    def _select_backend(
        self, 
        available_backends: List[BackendInfo], 
        query_info: QueryInfo
    ) -> Optional[str]:
        """
        Select the best backend for a query.
        
        Uses weighted round-robin with health and load factors.
        """
        if not available_backends:
            return None
        
        # Calculate weights based on health, load, and performance
        weighted_backends = []
        
        for backend in available_backends:
            # Base weight from health status
            if backend.status == BackendStatus.HEALTHY:
                health_weight = 1.0
            else:  # DEGRADED
                health_weight = 0.5
            
            # Load factor (inverse of current load)
            load_weight = max(0.1, 1.0 - (backend.current_queries / backend.max_concurrent_queries))
            
            # Performance weight (inverse of response time)
            perf_weight = max(0.1, 1.0 / (1.0 + backend.response_time_avg))
            
            # Error rate penalty
            error_penalty = 1.0 - backend.error_rate
            
            # Combined weight
            total_weight = health_weight * load_weight * perf_weight * error_penalty
            
            weighted_backends.append((backend, total_weight))
        
        # Sort by weight (highest first)
        weighted_backends.sort(key=lambda x: x[1], reverse=True)
        
        # Return the best backend
        return weighted_backends[0][0].id
    
    async def execute_query(
        self, 
        query_info: QueryInfo, 
        query_func: callable
    ) -> Any:
        """
        Execute a query through the load balancer.
        
        Args:
            query_info: Information about the query
            query_func: Function to execute the query on the selected backend
            
        Returns:
            Query result
            
        Raises:
            Exception: If query execution fails on all backends
        """
        last_error = None
        
        for attempt in range(self.max_retries):
            backend_id = await self.route_query(query_info)
            
            if not backend_id:
                raise Exception("No healthy backends available")
            
            try:
                # Execute query on selected backend
                result = await query_func(backend_id, query_info.query)
                
                # Update success metrics
                self.backends[backend_id].error_rate = max(
                    0.0, 
                    self.backends[backend_id].error_rate - 0.01
                )
                
                return result
                
            except Exception as e:
                last_error = e
                logger.warning(f"Query failed on backend {backend_id}: {e}")
                
                # Update error metrics
                backend = self.backends[backend_id]
                backend.total_errors += 1
                backend.error_rate = backend.total_errors / max(1, backend.total_queries)
                
                # Mark backend as unhealthy if error rate is too high
                if backend.error_rate > 0.5:
                    backend.status = BackendStatus.UNHEALTHY
                
            finally:
                # Decrement current queries
                if backend_id in self.backends:
                    self.backends[backend_id].current_queries = max(
                        0, 
                        self.backends[backend_id].current_queries - 1
                    )
        
        raise Exception(f"Query failed after {self.max_retries} attempts") from last_error
    
    def get_backend_stats(self) -> Dict[str, Any]:
        """Get statistics about all backends."""
        stats = {
            "backends": {},
            "total_queries": sum(b.total_queries for b in self.backends.values()),
            "total_errors": sum(b.total_errors for b in self.backends.values()),
            "healthy_backends": len([b for b in self.backends.values() if b.status == BackendStatus.HEALTHY]),
            "degraded_backends": len([b for b in self.backends.values() if b.status == BackendStatus.DEGRADED]),
            "unhealthy_backends": len([b for b in self.backends.values() if b.status == BackendStatus.UNHEALTHY]),
        }
        
        for backend_id, backend in self.backends.items():
            stats["backends"][backend_id] = {
                "status": backend.status.value,
                "response_time_avg": backend.response_time_avg,
                "error_rate": backend.error_rate,
                "current_queries": backend.current_queries,
                "total_queries": backend.total_queries,
                "total_errors": backend.total_errors,
                "last_health_check": backend.last_health_check,
            }
        
        return stats
    
    async def shutdown(self):
        """Shutdown the load balancer."""
        if self._health_check_task and not self._health_check_task.done():
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
