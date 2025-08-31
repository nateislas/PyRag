"""
Monitoring dashboard for PyRAG.

This module provides a real-time monitoring dashboard for visualizing
system metrics, alerts, and performance data.
"""

import asyncio
import time
import json
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


@dataclass
class DashboardConfig:
    """Dashboard configuration."""
    enabled: bool = True
    port: int = 8080
    host: str = "0.0.0.0"
    refresh_interval: float = 5.0
    max_data_points: int = 1000
    theme: str = "dark"
    auto_refresh: bool = True


@dataclass
class DashboardWidget:
    """Dashboard widget configuration."""
    id: str
    title: str
    widget_type: str  # "chart", "metric", "alert", "table"
    data_source: str
    position: Dict[str, int]  # x, y, width, height
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DashboardLayout:
    """Dashboard layout configuration."""
    name: str
    description: str
    widgets: List[DashboardWidget] = field(default_factory=list)
    default: bool = False


class MonitoringDashboard:
    """
    Real-time monitoring dashboard for PyRAG.
    
    Provides:
    - Real-time metrics visualization
    - Configurable widgets and layouts
    - Alert display and management
    - Performance trend analysis
    - System health overview
    """
    
    def __init__(self, config: DashboardConfig):
        """
        Initialize the monitoring dashboard.
        
        Args:
            config: Dashboard configuration
        """
        self.config = config
        
        # Dashboard state
        self.current_layout: Optional[DashboardLayout] = None
        self.layouts: List[DashboardLayout] = []
        self.widget_data: Dict[str, Any] = {}
        
        # Metrics storage for visualization
        self.metrics_history: List[Dict[str, Any]] = []
        self.alerts_history: List[Dict[str, Any]] = []
        
        # Background task
        self._dashboard_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Web server (placeholder for future implementation)
        self._server = None
        
        logger.info("Monitoring dashboard initialized")
    
    def add_layout(self, layout: DashboardLayout):
        """Add a dashboard layout."""
        self.layouts.append(layout)
        if layout.default:
            self.current_layout = layout
        logger.info(f"Added dashboard layout: {layout.name}")
    
    def set_current_layout(self, layout_name: str):
        """Set the current dashboard layout."""
        for layout in self.layouts:
            if layout.name == layout_name:
                self.current_layout = layout
                logger.info(f"Set current layout: {layout_name}")
                return
        
        logger.warning(f"Layout not found: {layout_name}")
    
    async def start(self):
        """Start the monitoring dashboard."""
        if self._running:
            logger.warning("Dashboard is already running")
            return
        
        self._running = True
        
        # Start background task for data updates
        self._dashboard_task = asyncio.create_task(self._dashboard_loop())
        
        # Initialize default layout if none set
        if not self.current_layout and self.layouts:
            default_layout = next((l for l in self.layouts if l.default), self.layouts[0])
            self.current_layout = default_layout
        
        # Start web server (placeholder)
        await self._start_web_server()
        
        logger.info("Monitoring dashboard started")
    
    async def stop(self):
        """Stop the monitoring dashboard."""
        self._running = False
        
        # Stop background task
        if self._dashboard_task and not self._dashboard_task.done():
            self._dashboard_task.cancel()
            try:
                await self._dashboard_task
            except asyncio.CancelledError:
                pass
        
        # Stop web server
        await self._stop_web_server()
        
        logger.info("Monitoring dashboard stopped")
    
    async def _dashboard_loop(self):
        """Main dashboard update loop."""
        while self._running:
            try:
                # Update widget data
                await self._update_widget_data()
                
                # Cleanup old data
                await self._cleanup_old_data()
                
                await asyncio.sleep(self.config.refresh_interval)
                
            except Exception as e:
                logger.error(f"Dashboard loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _update_widget_data(self):
        """Update data for all widgets."""
        if not self.current_layout:
            return
        
        for widget in self.current_layout.widgets:
            try:
                data = await self._get_widget_data(widget)
                self.widget_data[widget.id] = data
            except Exception as e:
                logger.error(f"Error updating widget {widget.id}: {e}")
    
    async def _get_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for a specific widget."""
        if widget.widget_type == "metric":
            return await self._get_metric_widget_data(widget)
        elif widget.widget_type == "chart":
            return await self._get_chart_widget_data(widget)
        elif widget.widget_type == "alert":
            return await self._get_alert_widget_data(widget)
        elif widget.widget_type == "table":
            return await self._get_table_widget_data(widget)
        else:
            return {"error": f"Unknown widget type: {widget.widget_type}"}
    
    async def _get_metric_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for metric widget."""
        metric_name = widget.data_source
        
        # Get current value
        current_value = 0.0
        if self.metrics_history:
            latest_metrics = self.metrics_history[-1]
            current_value = latest_metrics.get(metric_name, 0.0)
        
        # Calculate trend
        trend = "stable"
        if len(self.metrics_history) >= 2:
            previous_value = self.metrics_history[-2].get(metric_name, 0.0)
            if current_value > previous_value * 1.1:
                trend = "increasing"
            elif current_value < previous_value * 0.9:
                trend = "decreasing"
        
        return {
            "type": "metric",
            "value": current_value,
            "trend": trend,
            "unit": widget.config.get("unit", ""),
            "format": widget.config.get("format", "number")
        }
    
    async def _get_chart_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for chart widget."""
        metric_name = widget.data_source
        time_range = widget.config.get("time_range", 3600)  # 1 hour default
        
        # Get data points for the time range
        cutoff_time = time.time() - time_range
        data_points = []
        
        for metrics in self.metrics_history:
            if metrics.get("timestamp", 0) > cutoff_time:
                data_points.append({
                    "timestamp": metrics.get("timestamp", 0),
                    "value": metrics.get(metric_name, 0.0)
                })
        
        return {
            "type": "chart",
            "data": data_points,
            "chart_type": widget.config.get("chart_type", "line"),
            "title": widget.title
        }
    
    async def _get_alert_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for alert widget."""
        time_range = widget.config.get("time_range", 3600)  # 1 hour default
        cutoff_time = time.time() - time_range
        
        # Filter recent alerts
        recent_alerts = [
            alert for alert in self.alerts_history
            if alert.get("timestamp", 0) > cutoff_time
        ]
        
        # Group by severity
        severity_counts = {}
        for alert in recent_alerts:
            severity = alert.get("severity", "unknown")
            severity_counts[severity] = severity_counts.get(severity, 0) + 1
        
        return {
            "type": "alert",
            "alerts": recent_alerts,
            "severity_counts": severity_counts,
            "total_alerts": len(recent_alerts)
        }
    
    async def _get_table_widget_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get data for table widget."""
        table_type = widget.config.get("table_type", "metrics")
        
        if table_type == "metrics":
            return await self._get_metrics_table_data(widget)
        elif table_type == "alerts":
            return await self._get_alerts_table_data(widget)
        else:
            return {"error": f"Unknown table type: {table_type}"}
    
    async def _get_metrics_table_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get metrics table data."""
        if not self.metrics_history:
            return {"type": "table", "data": [], "columns": []}
        
        latest_metrics = self.metrics_history[-1]
        
        # Define columns
        columns = [
            {"key": "metric", "label": "Metric"},
            {"key": "value", "label": "Value"},
            {"key": "unit", "label": "Unit"},
            {"key": "status", "label": "Status"}
        ]
        
        # Define metrics to show
        metrics_to_show = [
            ("cpu_utilization", "CPU Usage", "%", "percentage"),
            ("memory_utilization", "Memory Usage", "%", "percentage"),
            ("avg_response_time", "Avg Response Time", "ms", "time"),
            ("error_rate", "Error Rate", "%", "percentage"),
            ("cache_hit_rate", "Cache Hit Rate", "%", "percentage"),
            ("total_queries", "Total Queries", "", "number")
        ]
        
        data = []
        for metric_key, label, unit, format_type in metrics_to_show:
            value = latest_metrics.get(metric_key, 0.0)
            
            # Determine status
            status = "normal"
            if metric_key == "cpu_utilization" and value > 80:
                status = "warning"
            elif metric_key == "memory_utilization" and value > 85:
                status = "warning"
            elif metric_key == "error_rate" and value > 5:
                status = "critical"
            
            data.append({
                "metric": label,
                "value": value,
                "unit": unit,
                "status": status
            })
        
        return {
            "type": "table",
            "data": data,
            "columns": columns
        }
    
    async def _get_alerts_table_data(self, widget: DashboardWidget) -> Dict[str, Any]:
        """Get alerts table data."""
        columns = [
            {"key": "timestamp", "label": "Time"},
            {"key": "severity", "label": "Severity"},
            {"key": "rule_name", "label": "Rule"},
            {"key": "message", "label": "Message"},
            {"key": "status", "label": "Status"}
        ]
        
        # Get recent alerts
        recent_alerts = self.alerts_history[-10:]  # Last 10 alerts
        
        data = []
        for alert in recent_alerts:
            data.append({
                "timestamp": datetime.fromtimestamp(alert.get("timestamp", 0)).strftime("%H:%M:%S"),
                "severity": alert.get("severity", "unknown"),
                "rule_name": alert.get("rule_name", ""),
                "message": alert.get("message", "")[:50] + "..." if len(alert.get("message", "")) > 50 else alert.get("message", ""),
                "status": "active" if not alert.get("resolved", False) else "resolved"
            })
        
        return {
            "type": "table",
            "data": data,
            "columns": columns
        }
    
    async def update_metrics(self, metrics: 'SystemMetrics'):
        """Update dashboard with new metrics."""
        if not metrics:
            return
        
        # Convert metrics to dict for storage
        metrics_dict = {
            "timestamp": metrics.timestamp,
            "cpu_utilization": metrics.cpu_utilization,
            "memory_utilization": metrics.memory_utilization,
            "disk_usage": metrics.disk_usage,
            "network_io": metrics.network_io,
            "active_connections": metrics.active_connections,
            "total_queries": metrics.total_queries,
            "successful_queries": metrics.successful_queries,
            "failed_queries": metrics.failed_queries,
            "avg_response_time": metrics.avg_response_time,
            "p95_response_time": metrics.p95_response_time,
            "p99_response_time": metrics.p99_response_time,
            "cache_hit_rate": metrics.cache_hit_rate,
            "error_rate": metrics.error_rate,
            "custom_metrics": metrics.custom_metrics
        }
        
        self.metrics_history.append(metrics_dict)
        
        # Limit history size
        if len(self.metrics_history) > self.config.max_data_points:
            self.metrics_history = self.metrics_history[-self.config.max_data_points:]
    
    async def update_alerts(self, alerts: List['Alert']):
        """Update dashboard with new alerts."""
        for alert in alerts:
            alert_dict = {
                "id": alert.id,
                "rule_name": alert.rule_name,
                "severity": alert.severity.value,
                "message": alert.message,
                "timestamp": alert.timestamp,
                "metric_name": alert.metric_name,
                "metric_value": alert.metric_value,
                "threshold": alert.threshold,
                "acknowledged": alert.acknowledged,
                "resolved": alert.resolved
            }
            
            self.alerts_history.append(alert_dict)
    
    async def _cleanup_old_data(self):
        """Clean up old dashboard data."""
        # Keep only recent metrics (last 24 hours)
        cutoff_time = time.time() - (24 * 3600)
        self.metrics_history = [
            metrics for metrics in self.metrics_history
            if metrics.get("timestamp", 0) > cutoff_time
        ]
        
        # Keep only recent alerts (last 7 days)
        cutoff_time = time.time() - (7 * 24 * 3600)
        self.alerts_history = [
            alert for alert in self.alerts_history
            if alert.get("timestamp", 0) > cutoff_time
        ]
    
    async def _start_web_server(self):
        """Start the web server for the dashboard."""
        # Placeholder for web server implementation
        # In a real implementation, this would start a web server
        # (e.g., FastAPI, Flask, or aiohttp) to serve the dashboard
        logger.info(f"Dashboard web server would start on {self.config.host}:{self.config.port}")
    
    async def _stop_web_server(self):
        """Stop the web server."""
        # Placeholder for web server shutdown
        logger.info("Dashboard web server stopped")
    
    def get_dashboard_data(self) -> Dict[str, Any]:
        """Get current dashboard data for API consumption."""
        return {
            "config": {
                "enabled": self.config.enabled,
                "refresh_interval": self.config.refresh_interval,
                "theme": self.config.theme,
                "auto_refresh": self.config.auto_refresh
            },
            "current_layout": self.current_layout.name if self.current_layout else None,
            "available_layouts": [layout.name for layout in self.layouts],
            "widget_data": self.widget_data,
            "metrics_summary": {
                "total_metrics": len(self.metrics_history),
                "total_alerts": len(self.alerts_history),
                "last_update": time.time()
            }
        }
    
    def is_running(self) -> bool:
        """Check if dashboard is running."""
        return self._running


# Predefined dashboard layouts
DEFAULT_DASHBOARD_LAYOUT = DashboardLayout(
    name="default",
    description="Default monitoring dashboard",
    default=True,
    widgets=[
        DashboardWidget(
            id="cpu_metric",
            title="CPU Utilization",
            widget_type="metric",
            data_source="cpu_utilization",
            position={"x": 0, "y": 0, "width": 3, "height": 2},
            config={"unit": "%", "format": "percentage"}
        ),
        DashboardWidget(
            id="memory_metric",
            title="Memory Utilization",
            widget_type="metric",
            data_source="memory_utilization",
            position={"x": 3, "y": 0, "width": 3, "height": 2},
            config={"unit": "%", "format": "percentage"}
        ),
        DashboardWidget(
            id="response_time_chart",
            title="Response Time Trend",
            widget_type="chart",
            data_source="avg_response_time",
            position={"x": 0, "y": 2, "width": 6, "height": 4},
            config={"chart_type": "line", "time_range": 3600}
        ),
        DashboardWidget(
            id="alerts_widget",
            title="Recent Alerts",
            widget_type="alert",
            data_source="alerts",
            position={"x": 6, "y": 0, "width": 6, "height": 6},
            config={"time_range": 3600}
        ),
        DashboardWidget(
            id="metrics_table",
            title="System Metrics",
            widget_type="table",
            data_source="metrics",
            position={"x": 0, "y": 6, "width": 12, "height": 4},
            config={"table_type": "metrics"}
        )
    ]
)

PERFORMANCE_DASHBOARD_LAYOUT = DashboardLayout(
    name="performance",
    description="Performance-focused dashboard",
    widgets=[
        DashboardWidget(
            id="response_time_metric",
            title="Avg Response Time",
            widget_type="metric",
            data_source="avg_response_time",
            position={"x": 0, "y": 0, "width": 3, "height": 2},
            config={"unit": "ms", "format": "time"}
        ),
        DashboardWidget(
            id="throughput_metric",
            title="Queries/Second",
            widget_type="metric",
            data_source="total_queries",
            position={"x": 3, "y": 0, "width": 3, "height": 2},
            config={"unit": "qps", "format": "number"}
        ),
        DashboardWidget(
            id="error_rate_metric",
            title="Error Rate",
            widget_type="metric",
            data_source="error_rate",
            position={"x": 6, "y": 0, "width": 3, "height": 2},
            config={"unit": "%", "format": "percentage"}
        ),
        DashboardWidget(
            id="cache_hit_rate_metric",
            title="Cache Hit Rate",
            widget_type="metric",
            data_source="cache_hit_rate",
            position={"x": 9, "y": 0, "width": 3, "height": 2},
            config={"unit": "%", "format": "percentage"}
        ),
        DashboardWidget(
            id="performance_chart",
            title="Performance Trends",
            widget_type="chart",
            data_source="avg_response_time",
            position={"x": 0, "y": 2, "width": 12, "height": 6},
            config={"chart_type": "line", "time_range": 7200}
        )
    ]
)
