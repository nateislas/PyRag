"""
Monitoring and alerting infrastructure for PyRAG.

This module provides comprehensive monitoring, alerting, and observability
capabilities for production deployments.
"""

from .alerting import (
    DEFAULT_ALERT_RULES,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    ConsoleAlertHandler,
    LogAlertHandler,
    WebhookAlertHandler,
)
from .dashboard import (
    DEFAULT_DASHBOARD_LAYOUT,
    PERFORMANCE_DASHBOARD_LAYOUT,
    DashboardConfig,
    MonitoringDashboard,
)
from .grafana import GrafanaConfig, GrafanaIntegration
from .health import HealthCheck, HealthMonitor
from .logging import LogAggregator, LogConfig, StructuredLogger
from .monitor import MonitoringConfig, MonitoringSystem, SystemMetrics
from .prometheus import PrometheusConfig, PrometheusExporter
from .tracing import SpanContext, TraceCollector

__all__ = [
    "MonitoringSystem",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertChannel",
    "MonitoringDashboard",
    "PrometheusExporter",
    "GrafanaIntegration",
    "StructuredLogger",
    "LogAggregator",
    "TraceCollector",
    "SpanContext",
    "HealthMonitor",
    "HealthCheck",
]
