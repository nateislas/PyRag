"""
Prometheus metrics exporter for PyRAG.

This module provides Prometheus metrics export capabilities for
integration with monitoring and alerting systems.
"""

import asyncio
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class PrometheusConfig:
    """Prometheus exporter configuration."""
    enabled: bool = True
    port: int = 9090
    host: str = "0.0.0.0"
    path: str = "/metrics"
    collection_interval: float = 15.0
    include_timestamp: bool = True
    include_help: bool = True


class PrometheusExporter:
    """
    Prometheus metrics exporter for PyRAG.
    
    Provides:
    - System metrics export in Prometheus format
    - Custom metrics support
    - Metric labeling and categorization
    - Integration with Prometheus monitoring stack
    """
    
    def __init__(self, config: PrometheusConfig):
        """
        Initialize the Prometheus exporter.
        
        Args:
            config: Prometheus configuration
        """
        self.config = config
        
        # Metrics storage
        self.metrics: Dict[str, 'PrometheusMetric'] = {}
        self.metric_help: Dict[str, str] = {}
        
        # Background task
        self._exporter_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Web server (placeholder)
        self._server = None
        
        # Initialize default metrics
        self._initialize_default_metrics()
        
        logger.info("Prometheus exporter initialized")
    
    def _initialize_default_metrics(self):
        """Initialize default PyRAG metrics."""
        self.add_metric(
            "pyrag_cpu_utilization",
            "CPU utilization percentage",
            "gauge",
            ["instance", "component"]
        )
        
        self.add_metric(
            "pyrag_memory_utilization",
            "Memory utilization percentage",
            "gauge",
            ["instance", "component"]
        )
        
        self.add_metric(
            "pyrag_disk_usage",
            "Disk usage percentage",
            "gauge",
            ["instance", "mount_point"]
        )
        
        self.add_metric(
            "pyrag_active_connections",
            "Number of active connections",
            "gauge",
            ["instance", "connection_type"]
        )
        
        self.add_metric(
            "pyrag_total_queries",
            "Total number of queries",
            "counter",
            ["instance", "query_type"]
        )
        
        self.add_metric(
            "pyrag_successful_queries",
            "Number of successful queries",
            "counter",
            ["instance", "query_type"]
        )
        
        self.add_metric(
            "pyrag_failed_queries",
            "Number of failed queries",
            "counter",
            ["instance", "query_type"]
        )
        
        self.add_metric(
            "pyrag_query_duration_seconds",
            "Query duration in seconds",
            "histogram",
            ["instance", "query_type"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0]
        )
        
        self.add_metric(
            "pyrag_response_time_seconds",
            "Response time in seconds",
            "histogram",
            ["instance", "endpoint"],
            buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0]
        )
        
        self.add_metric(
            "pyrag_cache_hit_rate",
            "Cache hit rate percentage",
            "gauge",
            ["instance", "cache_type"]
        )
        
        self.add_metric(
            "pyrag_error_rate",
            "Error rate percentage",
            "gauge",
            ["instance", "error_type"]
        )
        
        self.add_metric(
            "pyrag_vector_store_size",
            "Number of documents in vector store",
            "gauge",
            ["instance", "collection"]
        )
        
        self.add_metric(
            "pyrag_embedding_requests",
            "Number of embedding requests",
            "counter",
            ["instance", "model"]
        )
        
        self.add_metric(
            "pyrag_llm_requests",
            "Number of LLM requests",
            "counter",
            ["instance", "model", "operation"]
        )
        
        self.add_metric(
            "pyrag_ingestion_jobs",
            "Number of ingestion jobs",
            "counter",
            ["instance", "status", "library"]
        )
    
    def add_metric(self, name: str, help_text: str, metric_type: str, labels: List[str], buckets: Optional[List[float]] = None):
        """Add a new Prometheus metric."""
        self.metrics[name] = PrometheusMetric(
            name=name,
            metric_type=metric_type,
            labels=labels,
            buckets=buckets or []
        )
        self.metric_help[name] = help_text
        logger.debug(f"Added Prometheus metric: {name}")
    
    def record_metric(self, name: str, value: float, labels: Optional[Dict[str, str]] = None):
        """Record a metric value."""
        if name not in self.metrics:
            logger.warning(f"Unknown metric: {name}")
            return
        
        metric = self.metrics[name]
        metric.record_value(value, labels or {})
    
    async def start(self):
        """Start the Prometheus exporter."""
        if self._running:
            logger.warning("Prometheus exporter is already running")
            return
        
        self._running = True
        
        # Start background task
        self._exporter_task = asyncio.create_task(self._exporter_loop())
        
        # Start web server
        await self._start_web_server()
        
        logger.info("Prometheus exporter started")
    
    async def stop(self):
        """Stop the Prometheus exporter."""
        self._running = False
        
        # Stop background task
        if self._exporter_task and not self._exporter_task.done():
            self._exporter_task.cancel()
            try:
                await self._exporter_task
            except asyncio.CancelledError:
                pass
        
        # Stop web server
        await self._stop_web_server()
        
        logger.info("Prometheus exporter stopped")
    
    async def _exporter_loop(self):
        """Main exporter loop."""
        while self._running:
            try:
                await asyncio.sleep(self.config.collection_interval)
            except Exception as e:
                logger.error(f"Exporter loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def export_metrics(self, system_metrics: 'SystemMetrics'):
        """Export system metrics to Prometheus format."""
        if not system_metrics:
            return
        
        # Record system metrics
        self.record_metric(
            "pyrag_cpu_utilization",
            system_metrics.cpu_utilization,
            {"instance": "pyrag", "component": "system"}
        )
        
        self.record_metric(
            "pyrag_memory_utilization",
            system_metrics.memory_utilization,
            {"instance": "pyrag", "component": "system"}
        )
        
        self.record_metric(
            "pyrag_disk_usage",
            system_metrics.disk_usage,
            {"instance": "pyrag", "mount_point": "/"}
        )
        
        self.record_metric(
            "pyrag_active_connections",
            system_metrics.active_connections,
            {"instance": "pyrag", "connection_type": "http"}
        )
        
        self.record_metric(
            "pyrag_total_queries",
            system_metrics.total_queries,
            {"instance": "pyrag", "query_type": "search"}
        )
        
        self.record_metric(
            "pyrag_successful_queries",
            system_metrics.successful_queries,
            {"instance": "pyrag", "query_type": "search"}
        )
        
        self.record_metric(
            "pyrag_failed_queries",
            system_metrics.failed_queries,
            {"instance": "pyrag", "query_type": "search"}
        )
        
        self.record_metric(
            "pyrag_response_time_seconds",
            system_metrics.avg_response_time / 1000.0,  # Convert ms to seconds
            {"instance": "pyrag", "endpoint": "search"}
        )
        
        self.record_metric(
            "pyrag_cache_hit_rate",
            system_metrics.cache_hit_rate * 100.0,  # Convert to percentage
            {"instance": "pyrag", "cache_type": "query"}
        )
        
        self.record_metric(
            "pyrag_error_rate",
            system_metrics.error_rate * 100.0,  # Convert to percentage
            {"instance": "pyrag", "error_type": "query"}
        )
        
        # Export custom metrics
        for metric_name, metric_data in system_metrics.custom_metrics.items():
            if isinstance(metric_data, dict) and "value" in metric_data:
                value = metric_data["value"]
                tags = metric_data.get("tags", {})
                
                # Create Prometheus metric name
                prometheus_name = f"pyrag_custom_{metric_name}"
                
                # Add metric if it doesn't exist
                if prometheus_name not in self.metrics:
                    self.add_metric(
                        prometheus_name,
                        f"Custom metric: {metric_name}",
                        "gauge",
                        list(tags.keys())
                    )
                
                self.record_metric(prometheus_name, value, tags)
    
    def generate_metrics(self) -> str:
        """Generate Prometheus metrics format string."""
        lines = []
        
        # Add help text if enabled
        if self.config.include_help:
            for name, help_text in self.metric_help.items():
                lines.append(f"# HELP {name} {help_text}")
                lines.append(f"# TYPE {name} {self.metrics[name].metric_type}")
                lines.append("")
        
        # Add metric values
        for name, metric in self.metrics.items():
            for value_data in metric.values:
                # Format labels
                label_str = ""
                if value_data.labels:
                    label_pairs = [f'{k}="{v}"' for k, v in value_data.labels.items()]
                    label_str = "{" + ",".join(label_pairs) + "}"
                
                # Format value
                if self.config.include_timestamp:
                    timestamp = int(value_data.timestamp * 1000)  # Convert to milliseconds
                    lines.append(f"{name}{label_str} {value_data.value} {timestamp}")
                else:
                    lines.append(f"{name}{label_str} {value_data.value}")
        
        return "\n".join(lines)
    
    async def _start_web_server(self):
        """Start the web server for metrics endpoint."""
        # Placeholder for web server implementation
        # In a real implementation, this would start a web server
        # to serve metrics at /metrics endpoint
        logger.info(f"Prometheus metrics endpoint would be available at http://{self.config.host}:{self.config.port}{self.config.path}")
    
    async def _stop_web_server(self):
        """Stop the web server."""
        # Placeholder for web server shutdown
        logger.info("Prometheus web server stopped")
    
    def is_running(self) -> bool:
        """Check if exporter is running."""
        return self._running


class PrometheusMetric:
    """Individual Prometheus metric."""
    
    def __init__(self, name: str, metric_type: str, labels: List[str], buckets: List[float]):
        self.name = name
        self.metric_type = metric_type
        self.labels = labels
        self.buckets = buckets
        self.values: List['MetricValue'] = []
    
    def record_value(self, value: float, labels: Dict[str, str]):
        """Record a metric value with labels."""
        # Validate labels
        for label in labels:
            if label not in self.labels:
                logger.warning(f"Unknown label '{label}' for metric '{self.name}'")
        
        # Create metric value
        metric_value = MetricValue(
            value=value,
            labels=labels,
            timestamp=time.time()
        )
        
        self.values.append(metric_value)
        
        # Keep only recent values (last 100)
        if len(self.values) > 100:
            self.values = self.values[-100:]


class MetricValue:
    """Individual metric value with labels and timestamp."""
    
    def __init__(self, value: float, labels: Dict[str, str], timestamp: float):
        self.value = value
        self.labels = labels
        self.timestamp = timestamp
