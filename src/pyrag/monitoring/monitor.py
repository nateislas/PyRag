"""
Main monitoring system for PyRAG.

This module provides the central monitoring system that orchestrates
all monitoring, alerting, and observability components.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for the monitoring system."""
    enabled: bool = True
    collection_interval: float = 30.0
    retention_days: int = 30
    alerting_enabled: bool = True
    tracing_enabled: bool = True
    logging_enabled: bool = True
    health_check_enabled: bool = True
    prometheus_enabled: bool = True
    grafana_enabled: bool = False
    dashboard_enabled: bool = True


@dataclass
class SystemMetrics:
    """System-wide metrics."""
    timestamp: float
    cpu_utilization: float
    memory_utilization: float
    disk_usage: float
    network_io: Dict[str, float]
    active_connections: int
    total_queries: int
    successful_queries: int
    failed_queries: int
    avg_response_time: float
    p95_response_time: float
    p99_response_time: float
    cache_hit_rate: float
    error_rate: float
    custom_metrics: Dict[str, Any] = field(default_factory=dict)


class MonitoringSystem:
    """
    Central monitoring system for PyRAG.
    
    Orchestrates all monitoring components including:
    - Metrics collection and aggregation
    - Alerting and notification
    - Health monitoring
    - Distributed tracing
    - Structured logging
    - Dashboard and visualization
    """
    
    def __init__(
        self,
        config: MonitoringConfig,
        alert_manager: Optional['AlertManager'] = None,
        health_monitor: Optional['HealthMonitor'] = None,
        trace_collector: Optional['TraceCollector'] = None,
        structured_logger: Optional['StructuredLogger'] = None,
        prometheus_exporter: Optional['PrometheusExporter'] = None,
        grafana_integration: Optional['GrafanaIntegration'] = None,
        dashboard: Optional['MonitoringDashboard'] = None
    ):
        """
        Initialize the monitoring system.
        
        Args:
            config: Monitoring configuration
            alert_manager: Alert manager instance
            health_monitor: Health monitoring instance
            trace_collector: Distributed tracing collector
            structured_logger: Structured logging instance
            prometheus_exporter: Prometheus metrics exporter
            grafana_integration: Grafana integration
            dashboard: Monitoring dashboard
        """
        self.config = config
        self.alert_manager = alert_manager
        self.health_monitor = health_monitor
        self.trace_collector = trace_collector
        self.structured_logger = structured_logger
        self.prometheus_exporter = prometheus_exporter
        self.grafana_integration = grafana_integration
        self.dashboard = dashboard
        
        # Metrics storage
        self.metrics_history: List[SystemMetrics] = []
        self.current_metrics: Optional[SystemMetrics] = None
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Callbacks for metrics collection
        self.metrics_collectors: List[Callable] = []
        self.alert_callbacks: List[Callable] = []
        
        logger.info("Monitoring system initialized")
    
    def add_metrics_collector(self, collector: Callable):
        """Add a metrics collector function."""
        self.metrics_collectors.append(collector)
        logger.info(f"Added metrics collector: {collector.__name__}")
    
    def add_alert_callback(self, callback: Callable):
        """Add an alert callback function."""
        self.alert_callbacks.append(callback)
        logger.info(f"Added alert callback: {callback.__name__}")
    
    async def start(self):
        """Start the monitoring system."""
        if self._running:
            logger.warning("Monitoring system is already running")
            return
        
        self._running = True
        
        # Start background monitoring task
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        # Start individual components
        if self.config.alerting_enabled and self.alert_manager:
            await self.alert_manager.start()
        
        if self.config.health_check_enabled and self.health_monitor:
            await self.health_monitor.start()
        
        if self.config.tracing_enabled and self.trace_collector:
            await self.trace_collector.start()
        
        if self.config.logging_enabled and self.structured_logger:
            await self.structured_logger.start()
        
        if self.config.prometheus_enabled and self.prometheus_exporter:
            await self.prometheus_exporter.start()
        
        if self.config.grafana_enabled and self.grafana_integration:
            await self.grafana_integration.start()
        
        if self.config.dashboard_enabled and self.dashboard:
            await self.dashboard.start()
        
        logger.info("Monitoring system started")
    
    async def stop(self):
        """Stop the monitoring system."""
        self._running = False
        
        # Stop background task
        if self._monitoring_task and not self._monitoring_task.done():
            self._monitoring_task.cancel()
            try:
                await self._monitoring_task
            except asyncio.CancelledError:
                pass
        
        # Stop individual components
        if self.alert_manager:
            await self.alert_manager.stop()
        
        if self.health_monitor:
            await self.health_monitor.stop()
        
        if self.trace_collector:
            await self.trace_collector.stop()
        
        if self.structured_logger:
            await self.structured_logger.stop()
        
        if self.prometheus_exporter:
            await self.prometheus_exporter.stop()
        
        if self.grafana_integration:
            await self.grafana_integration.stop()
        
        if self.dashboard:
            await self.dashboard.stop()
        
        logger.info("Monitoring system stopped")
    
    async def _monitoring_loop(self):
        """Main monitoring loop."""
        while self._running:
            try:
                # Collect system metrics
                await self._collect_system_metrics()
                
                # Process alerts
                if self.config.alerting_enabled:
                    await self._process_alerts()
                
                # Export metrics
                if self.config.prometheus_enabled and self.prometheus_exporter:
                    await self.prometheus_exporter.export_metrics(self.current_metrics)
                
                # Update dashboard
                if self.config.dashboard_enabled and self.dashboard:
                    await self.dashboard.update_metrics(self.current_metrics)
                
                # Cleanup old metrics
                await self._cleanup_old_metrics()
                
                await asyncio.sleep(self.config.collection_interval)
                
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error
    
    async def _collect_system_metrics(self):
        """Collect system-wide metrics."""
        try:
            # Collect metrics from all collectors
            collected_metrics = {}
            for collector in self.metrics_collectors:
                try:
                    if asyncio.iscoroutinefunction(collector):
                        metrics = await collector()
                    else:
                        metrics = collector()
                    collected_metrics.update(metrics)
                except Exception as e:
                    logger.error(f"Metrics collector error: {e}")
            
            # Create system metrics
            self.current_metrics = SystemMetrics(
                timestamp=time.time(),
                cpu_utilization=collected_metrics.get('cpu_utilization', 0.0),
                memory_utilization=collected_metrics.get('memory_utilization', 0.0),
                disk_usage=collected_metrics.get('disk_usage', 0.0),
                network_io=collected_metrics.get('network_io', {}),
                active_connections=collected_metrics.get('active_connections', 0),
                total_queries=collected_metrics.get('total_queries', 0),
                successful_queries=collected_metrics.get('successful_queries', 0),
                failed_queries=collected_metrics.get('failed_queries', 0),
                avg_response_time=collected_metrics.get('avg_response_time', 0.0),
                p95_response_time=collected_metrics.get('p95_response_time', 0.0),
                p99_response_time=collected_metrics.get('p99_response_time', 0.0),
                cache_hit_rate=collected_metrics.get('cache_hit_rate', 0.0),
                error_rate=collected_metrics.get('error_rate', 0.0),
                custom_metrics=collected_metrics.get('custom_metrics', {})
            )
            
            # Store in history
            self.metrics_history.append(self.current_metrics)
            
            # Log metrics
            if self.structured_logger:
                await self.structured_logger.log_metrics(self.current_metrics)
            
        except Exception as e:
            logger.error(f"System metrics collection error: {e}")
    
    async def _process_alerts(self):
        """Process alerts based on current metrics."""
        if not self.current_metrics or not self.alert_manager:
            return
        
        try:
            # Check alert rules
            alerts = await self.alert_manager.check_alerts(self.current_metrics)
            
            # Trigger alert callbacks
            for alert in alerts:
                for callback in self.alert_callbacks:
                    try:
                        if asyncio.iscoroutinefunction(callback):
                            await callback(alert)
                        else:
                            callback(alert)
                    except Exception as e:
                        logger.error(f"Alert callback error: {e}")
            
        except Exception as e:
            logger.error(f"Alert processing error: {e}")
    
    async def _cleanup_old_metrics(self):
        """Clean up old metrics based on retention policy."""
        cutoff_time = time.time() - (self.config.retention_days * 24 * 3600)
        
        # Remove old metrics
        self.metrics_history = [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
        
        logger.debug(f"Cleaned up metrics, retained {len(self.metrics_history)} entries")
    
    def get_current_metrics(self) -> Optional[SystemMetrics]:
        """Get current system metrics."""
        return self.current_metrics
    
    def get_metrics_history(self, hours: int = 24) -> List[SystemMetrics]:
        """Get metrics history for the specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            metrics for metrics in self.metrics_history
            if metrics.timestamp > cutoff_time
        ]
    
    def get_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get metrics summary for the specified hours."""
        recent_metrics = self.get_metrics_history(hours)
        
        if not recent_metrics:
            return {"error": "No metrics available for specified period"}
        
        # Calculate summary statistics
        cpu_values = [m.cpu_utilization for m in recent_metrics]
        memory_values = [m.memory_utilization for m in recent_metrics]
        response_times = [m.avg_response_time for m in recent_metrics]
        error_rates = [m.error_rate for m in recent_metrics]
        cache_hit_rates = [m.cache_hit_rate for m in recent_metrics]
        
        return {
            "period_hours": hours,
            "data_points": len(recent_metrics),
            "cpu": {
                "avg": sum(cpu_values) / len(cpu_values),
                "min": min(cpu_values),
                "max": max(cpu_values)
            },
            "memory": {
                "avg": sum(memory_values) / len(memory_values),
                "min": min(memory_values),
                "max": max(memory_values)
            },
            "response_time": {
                "avg": sum(response_times) / len(response_times),
                "min": min(response_times),
                "max": max(response_times)
            },
            "error_rate": {
                "avg": sum(error_rates) / len(error_rates),
                "min": min(error_rates),
                "max": max(error_rates)
            },
            "cache_hit_rate": {
                "avg": sum(cache_hit_rates) / len(cache_hit_rates),
                "min": min(cache_hit_rates),
                "max": max(cache_hit_rates)
            }
        }
    
    @asynccontextmanager
    async def trace_operation(self, operation_name: str, **kwargs):
        """Context manager for tracing operations."""
        if not self.trace_collector:
            yield
            return
        
        # Generate a trace ID for this operation
        trace_id = str(uuid.uuid4())
        span = self.trace_collector.create_span(
            name=operation_name,
            trace_id=trace_id,
            tags=kwargs
        )
        try:
            yield span
        finally:
            self.trace_collector.end_span(span.span_id)
    
    async def log_event(self, event_type: str, message: str, **kwargs):
        """Log an event with structured logging."""
        if not self.structured_logger:
            logger.info(f"{event_type}: {message}")
            return
        
        await self.structured_logger.log_event(event_type, message, **kwargs)
    
    async def record_metric(self, metric_name: str, value: Any, tags: Optional[Dict[str, str]] = None):
        """Record a custom metric."""
        if not self.current_metrics:
            return
        
        self.current_metrics.custom_metrics[metric_name] = {
            "value": value,
            "timestamp": time.time(),
            "tags": tags or {}
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status of the monitoring system."""
        status = {
            "monitoring_system": "healthy" if self._running else "stopped",
            "components": {}
        }
        
        if self.alert_manager:
            status["components"]["alert_manager"] = "healthy" if self.alert_manager.is_running() else "stopped"
        
        if self.health_monitor:
            status["components"]["health_monitor"] = "healthy" if self.health_monitor.is_running() else "stopped"
        
        if self.trace_collector:
            status["components"]["trace_collector"] = "healthy" if self.trace_collector.is_running() else "stopped"
        
        if self.structured_logger:
            status["components"]["structured_logger"] = "healthy" if self.structured_logger.is_running() else "stopped"
        
        if self.prometheus_exporter:
            status["components"]["prometheus_exporter"] = "healthy" if self.prometheus_exporter.is_running() else "stopped"
        
        return status
