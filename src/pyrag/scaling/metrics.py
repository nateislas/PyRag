"""
Performance metrics collection and monitoring for PyRAG.

This module provides comprehensive metrics collection, performance monitoring,
and analytics for production deployments.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Performance metrics for a service or operation."""

    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    memory_used_mb: float
    memory_available_mb: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    throughput_qps: float
    error_rate: float
    active_connections: int
    queue_length: int
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class QueryMetrics:
    """Metrics for individual queries."""

    query_id: str
    query: str
    start_time: float
    end_time: float
    duration: float
    success: bool
    error_message: Optional[str] = None
    backend_id: Optional[str] = None
    user_id: Optional[str] = None
    session_id: Optional[str] = None
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class MetricsCollector:
    """
    Comprehensive metrics collector for PyRAG.

    Provides:
    - System resource monitoring
    - Query performance tracking
    - Custom metrics collection
    - Historical data storage
    - Real-time metrics aggregation
    """

    def __init__(
        self,
        collection_interval: float = 10.0,
        history_retention_hours: int = 24,
        enable_system_metrics: bool = True,
        enable_query_metrics: bool = True,
    ):
        """
        Initialize the metrics collector.

        Args:
            collection_interval: Interval between metric collections (seconds)
            history_retention_hours: Hours to retain historical metrics
            enable_system_metrics: Enable system resource monitoring
            enable_query_metrics: Enable query performance tracking
        """
        self.collection_interval = collection_interval
        self.history_retention_hours = history_retention_hours
        self.enable_system_metrics = enable_system_metrics
        self.enable_query_metrics = enable_query_metrics

        # Metrics storage
        self.performance_history: deque = deque(
            maxlen=10000
        )  # ~2.8 hours at 1s intervals
        self.query_history: deque = deque(maxlen=10000)
        self.custom_metrics: Dict[str, Any] = defaultdict(dict)

        # Aggregation windows
        self.aggregation_windows = {
            "1m": deque(maxlen=60),
            "5m": deque(maxlen=300),
            "15m": deque(maxlen=900),
            "1h": deque(maxlen=3600),
        }

        # Background task
        self._collection_task: Optional[asyncio.Task] = None
        self._running = False

        # Query tracking
        self.active_queries: Dict[str, QueryMetrics] = {}
        self.query_counters = {
            "total": 0,
            "successful": 0,
            "failed": 0,
            "avg_duration": 0.0,
        }

    async def start(self):
        """Start the metrics collection background task."""
        if self._running:
            logger.warning("Metrics collector is already running")
            return

        self._running = True
        self._collection_task = asyncio.create_task(self._collection_loop())
        logger.info("Metrics collector started")

    async def stop(self):
        """Stop the metrics collection background task."""
        self._running = False

        if self._collection_task and not self._collection_task.done():
            self._collection_task.cancel()
            try:
                await self._collection_task
            except asyncio.CancelledError:
                pass

        logger.info("Metrics collector stopped")

    async def _collection_loop(self):
        """Main metrics collection loop."""
        while self._running:
            try:
                if self.enable_system_metrics:
                    await self._collect_system_metrics()

                await self._update_aggregations()
                await self._cleanup_old_metrics()

                await asyncio.sleep(self.collection_interval)

            except Exception as e:
                logger.error(f"Metrics collection error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error

    async def _collect_system_metrics(self):
        """Collect system resource metrics."""
        try:
            # CPU utilization
            cpu_percent = psutil.cpu_percent(interval=1)

            # Memory utilization
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            memory_used_mb = memory.used / (1024 * 1024)
            memory_available_mb = memory.available / (1024 * 1024)

            # Network connections (approximate)
            connections = len(psutil.net_connections())

            # Create performance metrics
            metrics = PerformanceMetrics(
                timestamp=time.time(),
                cpu_utilization=cpu_percent,
                memory_utilization=memory_percent,
                memory_used_mb=memory_used_mb,
                memory_available_mb=memory_available_mb,
                response_time_avg=self._calculate_avg_response_time(),
                response_time_p95=self._calculate_percentile_response_time(95),
                response_time_p99=self._calculate_percentile_response_time(99),
                throughput_qps=self._calculate_throughput(),
                error_rate=self._calculate_error_rate(),
                active_connections=connections,
                queue_length=len(self.active_queries),
            )

            # Store metrics
            self.performance_history.append(metrics)

            # Update aggregations
            for window_name, window_data in self.aggregation_windows.items():
                window_data.append(metrics)

        except Exception as e:
            logger.error(f"System metrics collection error: {e}")

    def _calculate_avg_response_time(self) -> float:
        """Calculate average response time from recent queries."""
        if not self.query_history:
            return 0.0

        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        if not recent_queries:
            return 0.0

        total_duration = sum(query.duration for query in recent_queries)
        return total_duration / len(recent_queries)

    def _calculate_percentile_response_time(self, percentile: int) -> float:
        """Calculate percentile response time."""
        if not self.query_history:
            return 0.0

        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        if not recent_queries:
            return 0.0

        durations = sorted([query.duration for query in recent_queries])
        index = int(len(durations) * percentile / 100)
        return durations[min(index, len(durations) - 1)]

    def _calculate_throughput(self) -> float:
        """Calculate queries per second."""
        if not self.query_history:
            return 0.0

        # Calculate QPS over last minute
        one_minute_ago = time.time() - 60
        recent_queries = [q for q in self.query_history if q.end_time > one_minute_ago]

        if not recent_queries:
            return 0.0

        return len(recent_queries) / 60.0

    def _calculate_error_rate(self) -> float:
        """Calculate error rate from recent queries."""
        if not self.query_history:
            return 0.0

        recent_queries = list(self.query_history)[-100:]  # Last 100 queries
        if not recent_queries:
            return 0.0

        failed_queries = sum(1 for query in recent_queries if not query.success)
        return failed_queries / len(recent_queries)

    async def _update_aggregations(self):
        """Update metric aggregations."""
        # This could be enhanced with more sophisticated aggregation logic
        pass

    async def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        cutoff_time = time.time() - (self.history_retention_hours * 3600)

        # Clean up performance history
        while (
            self.performance_history
            and self.performance_history[0].timestamp < cutoff_time
        ):
            self.performance_history.popleft()

        # Clean up query history
        while self.query_history and self.query_history[0].end_time < cutoff_time:
            self.query_history.popleft()

    def start_query_tracking(self, query_id: str, query: str, **kwargs) -> str:
        """
        Start tracking a query.

        Args:
            query_id: Unique identifier for the query
            query: The actual query string
            **kwargs: Additional query metadata

        Returns:
            Query ID for tracking
        """
        if not self.enable_query_metrics:
            return query_id

        start_time = time.time()

        query_metrics = QueryMetrics(
            query_id=query_id,
            query=query,
            start_time=start_time,
            end_time=0.0,
            duration=0.0,
            success=False,
            **kwargs,
        )

        self.active_queries[query_id] = query_metrics
        self.query_counters["total"] += 1

        return query_id

    def end_query_tracking(
        self, query_id: str, success: bool = True, error_message: Optional[str] = None
    ):
        """
        End tracking a query.

        Args:
            query_id: Query ID to end tracking
            success: Whether the query was successful
            error_message: Error message if query failed
        """
        if not self.enable_query_metrics or query_id not in self.active_queries:
            return

        query_metrics = self.active_queries[query_id]
        end_time = time.time()

        # Update query metrics
        query_metrics.end_time = end_time
        query_metrics.duration = end_time - query_metrics.start_time
        query_metrics.success = success
        query_metrics.error_message = error_message

        # Store in history
        self.query_history.append(query_metrics)

        # Update counters
        if success:
            self.query_counters["successful"] += 1
        else:
            self.query_counters["failed"] += 1

        # Update average duration
        total_queries = (
            self.query_counters["successful"] + self.query_counters["failed"]
        )
        if total_queries > 0:
            self.query_counters["avg_duration"] = (
                self.query_counters["avg_duration"] * (total_queries - 1)
                + query_metrics.duration
            ) / total_queries

        # Remove from active queries
        del self.active_queries[query_id]

    def add_custom_metric(self, category: str, name: str, value: Any):
        """
        Add a custom metric.

        Args:
            category: Metric category (e.g., 'business', 'technical')
            name: Metric name
            value: Metric value
        """
        self.custom_metrics[category][name] = {"value": value, "timestamp": time.time()}

    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current system metrics."""
        if not self.performance_history:
            return {}

        latest = self.performance_history[-1]

        return {
            "timestamp": latest.timestamp,
            "cpu_utilization": latest.cpu_utilization,
            "memory_utilization": latest.memory_utilization,
            "memory_used_mb": latest.memory_used_mb,
            "memory_available_mb": latest.memory_available_mb,
            "response_time_avg": latest.response_time_avg,
            "response_time_p95": latest.response_time_p95,
            "response_time_p99": latest.response_time_p99,
            "throughput_qps": latest.throughput_qps,
            "error_rate": latest.error_rate,
            "active_connections": latest.active_connections,
            "queue_length": latest.queue_length,
            "custom_metrics": dict(self.custom_metrics),
        }

    def get_query_statistics(self) -> Dict[str, Any]:
        """Get query performance statistics."""
        return {
            "total_queries": self.query_counters["total"],
            "successful_queries": self.query_counters["successful"],
            "failed_queries": self.query_counters["failed"],
            "success_rate": (
                self.query_counters["successful"] / max(1, self.query_counters["total"])
            ),
            "average_duration": self.query_counters["avg_duration"],
            "active_queries": len(self.active_queries),
            "recent_queries": len(list(self.query_history)[-100:]),
        }

    def get_metrics_summary(self, window_minutes: int = 60) -> Dict[str, Any]:
        """
        Get metrics summary for a time window.

        Args:
            window_minutes: Time window in minutes

        Returns:
            Metrics summary
        """
        cutoff_time = time.time() - (window_minutes * 60)

        # Filter recent performance metrics
        recent_metrics = [
            m for m in self.performance_history if m.timestamp > cutoff_time
        ]

        if not recent_metrics:
            return {"error": "No metrics available for specified window"}

        # Calculate statistics
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        memory_values = [m.memory_utilization for m in recent_metrics]
        response_times = [m.response_time_avg for m in recent_metrics]
        throughput_values = [m.throughput_qps for m in recent_metrics]

        return {
            "window_minutes": window_minutes,
            "data_points": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values),
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values),
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times),
            },
            "throughput": {
                "avg": sum(throughput_values) / len(throughput_values),
                "min": min(throughput_values),
                "max": max(throughput_values),
            },
        }

    def export_metrics(self, format: str = "json") -> str:
        """
        Export metrics in specified format.

        Args:
            format: Export format ('json' or 'csv')

        Returns:
            Exported metrics as string
        """
        if format.lower() == "json":
            return json.dumps(
                {
                    "current_metrics": self.get_current_metrics(),
                    "query_statistics": self.get_query_statistics(),
                    "custom_metrics": dict(self.custom_metrics),
                    "export_timestamp": time.time(),
                },
                indent=2,
            )

        elif format.lower() == "csv":
            # Simple CSV export of recent performance metrics
            lines = [
                "timestamp,cpu_utilization,memory_utilization,response_time_avg,throughput_qps,error_rate"
            ]

            for metric in list(self.performance_history)[-100:]:  # Last 100 metrics
                lines.append(
                    f"{metric.timestamp},{metric.cpu_utilization},{metric.memory_utilization},{metric.response_time_avg},{metric.throughput_qps},{metric.error_rate}"
                )

            return "\n".join(lines)

        else:
            raise ValueError(f"Unsupported export format: {format}")


class PerformanceMonitor:
    """
    High-level performance monitoring interface.

    Provides easy-to-use methods for monitoring application performance
    and generating alerts based on thresholds.
    """

    def __init__(
        self,
        metrics_collector: MetricsCollector,
        alert_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the performance monitor.

        Args:
            metrics_collector: Metrics collector instance
            alert_callbacks: List of alert callback functions
        """
        self.metrics_collector = metrics_collector
        self.alert_callbacks = alert_callbacks or []
        self.thresholds = {
            "cpu_utilization": 80.0,
            "memory_utilization": 85.0,
            "response_time_avg": 500.0,  # ms
            "error_rate": 0.05,  # 5%
            "throughput_qps": 10.0,  # minimum QPS
        }

    async def check_alerts(self):
        """Check current metrics against thresholds and trigger alerts."""
        current_metrics = self.metrics_collector.get_current_metrics()

        if not current_metrics:
            return

        alerts = []

        # Check CPU utilization
        if current_metrics["cpu_utilization"] > self.thresholds["cpu_utilization"]:
            alerts.append(
                {
                    "type": "high_cpu",
                    "message": f"CPU utilization {current_metrics['cpu_utilization']:.1f}% exceeds threshold {self.thresholds['cpu_utilization']}%",
                    "severity": "warning",
                    "value": current_metrics["cpu_utilization"],
                    "threshold": self.thresholds["cpu_utilization"],
                }
            )

        # Check memory utilization
        if (
            current_metrics["memory_utilization"]
            > self.thresholds["memory_utilization"]
        ):
            alerts.append(
                {
                    "type": "high_memory",
                    "message": f"Memory utilization {current_metrics['memory_utilization']:.1f}% exceeds threshold {self.thresholds['memory_utilization']}%",
                    "severity": "warning",
                    "value": current_metrics["memory_utilization"],
                    "threshold": self.thresholds["memory_utilization"],
                }
            )

        # Check response time
        if current_metrics["response_time_avg"] > self.thresholds["response_time_avg"]:
            alerts.append(
                {
                    "type": "high_response_time",
                    "message": f"Average response time {current_metrics['response_time_avg']:.1f}ms exceeds threshold {self.thresholds['response_time_avg']}ms",
                    "severity": "warning",
                    "value": current_metrics["response_time_avg"],
                    "threshold": self.thresholds["response_time_avg"],
                }
            )

        # Check error rate
        if current_metrics["error_rate"] > self.thresholds["error_rate"]:
            alerts.append(
                {
                    "type": "high_error_rate",
                    "message": f"Error rate {current_metrics['error_rate']:.2%} exceeds threshold {self.thresholds['error_rate']:.2%}",
                    "severity": "critical",
                    "value": current_metrics["error_rate"],
                    "threshold": self.thresholds["error_rate"],
                }
            )

        # Check throughput
        if current_metrics["throughput_qps"] < self.thresholds["throughput_qps"]:
            alerts.append(
                {
                    "type": "low_throughput",
                    "message": f"Throughput {current_metrics['throughput_qps']:.1f} QPS below threshold {self.thresholds['throughput_qps']} QPS",
                    "severity": "warning",
                    "value": current_metrics["throughput_qps"],
                    "threshold": self.thresholds["throughput_qps"],
                }
            )

        # Trigger alerts
        for alert in alerts:
            await self._trigger_alert(alert)

    async def _trigger_alert(self, alert: Dict[str, Any]):
        """Trigger alert callbacks."""
        alert["timestamp"] = time.time()

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert)
                else:
                    callback(alert)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def set_threshold(self, metric: str, value: float):
        """Set a threshold for a metric."""
        if metric not in self.thresholds:
            raise ValueError(f"Unknown metric: {metric}")

        self.thresholds[metric] = value
        logger.info(f"Updated threshold for {metric}: {value}")

    def get_thresholds(self) -> Dict[str, float]:
        """Get current thresholds."""
        return self.thresholds.copy()
