"""
Alerting system for PyRAG.

This module provides comprehensive alerting capabilities with configurable
rules, multiple notification channels, and alert management.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


class AlertSeverity(Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AlertChannel(Enum):
    """Alert notification channels."""

    EMAIL = "email"
    SLACK = "slack"
    WEBHOOK = "webhook"
    PAGERDUTY = "pagerduty"
    CONSOLE = "console"
    LOG = "log"


@dataclass
class AlertRule:
    """Alert rule configuration."""

    name: str
    description: str
    severity: AlertSeverity
    condition: str  # Metric name and operator (e.g., "cpu_utilization > 80")
    threshold: float
    duration: float = 300.0  # Duration in seconds before alerting
    channels: List[AlertChannel] = field(default_factory=lambda: [AlertChannel.CONSOLE])
    enabled: bool = True
    cooldown: float = 3600.0  # Cooldown period in seconds
    tags: List[str] = field(default_factory=list)


@dataclass
class Alert:
    """Alert instance."""

    id: str
    rule_name: str
    severity: AlertSeverity
    message: str
    timestamp: float
    metric_name: str
    metric_value: float
    threshold: float
    duration: float
    channels: List[AlertChannel]
    tags: List[str]
    acknowledged: bool = False
    resolved: bool = False
    resolved_at: Optional[float] = None
    acknowledged_by: Optional[str] = None
    acknowledged_at: Optional[float] = None


class AlertManager:
    """
    Alert manager for PyRAG.

    Provides:
    - Configurable alert rules
    - Multiple notification channels
    - Alert lifecycle management
    - Alert history and statistics
    """

    def __init__(self, check_interval: float = 30.0):
        """
        Initialize the alert manager.

        Args:
            check_interval: Interval between alert checks (seconds)
        """
        self.check_interval = check_interval

        # Alert rules and state
        self.alert_rules: List[AlertRule] = []
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []

        # Notification channels
        self.channel_handlers: Dict[AlertChannel, Callable] = {}

        # Background task
        self._alert_task: Optional[asyncio.Task] = None
        self._running = False

        # Statistics
        self.stats = {
            "total_alerts": 0,
            "active_alerts": 0,
            "resolved_alerts": 0,
            "acknowledged_alerts": 0,
        }

        logger.info("Alert manager initialized")

    def add_alert_rule(self, rule: AlertRule):
        """Add an alert rule."""
        self.alert_rules.append(rule)
        logger.info(f"Added alert rule: {rule.name}")

    def add_channel_handler(self, channel: AlertChannel, handler: Callable):
        """Add a notification channel handler."""
        self.channel_handlers[channel] = handler
        logger.info(f"Added channel handler for {channel.value}")

    async def start(self):
        """Start the alert manager."""
        if self._running:
            logger.warning("Alert manager is already running")
            return

        self._running = True
        self._alert_task = asyncio.create_task(self._alert_loop())
        logger.info("Alert manager started")

    async def stop(self):
        """Stop the alert manager."""
        self._running = False

        if self._alert_task and not self._alert_task.done():
            self._alert_task.cancel()
            try:
                await self._alert_task
            except asyncio.CancelledError:
                pass

        logger.info("Alert manager stopped")

    async def _alert_loop(self):
        """Main alert checking loop."""
        while self._running:
            try:
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Alert loop error: {e}")
                await asyncio.sleep(5.0)

    async def check_alerts(self, metrics: "SystemMetrics") -> List[Alert]:
        """
        Check metrics against alert rules and generate alerts.

        Args:
            metrics: Current system metrics

        Returns:
            List of new alerts
        """
        new_alerts = []

        for rule in self.alert_rules:
            if not rule.enabled:
                continue

            # Check if alert is in cooldown
            if self._is_alert_in_cooldown(rule):
                continue

            # Evaluate rule condition
            if self._evaluate_condition(rule, metrics):
                # Check if alert already exists
                alert_id = f"{rule.name}_{int(time.time() / rule.duration)}"

                if alert_id not in self.active_alerts:
                    # Create new alert
                    alert = Alert(
                        id=alert_id,
                        rule_name=rule.name,
                        severity=rule.severity,
                        message=self._generate_alert_message(rule, metrics),
                        timestamp=time.time(),
                        metric_name=rule.condition.split()[0],
                        metric_value=self._get_metric_value(
                            rule.condition.split()[0], metrics
                        ),
                        threshold=rule.threshold,
                        duration=rule.duration,
                        channels=rule.channels,
                        tags=rule.tags,
                    )

                    self.active_alerts[alert_id] = alert
                    self.alert_history.append(alert)
                    new_alerts.append(alert)

                    # Update statistics
                    self.stats["total_alerts"] += 1
                    self.stats["active_alerts"] += 1

                    # Send notifications
                    await self._send_notifications(alert)

                    logger.warning(f"Alert triggered: {alert.message}")

        return new_alerts

    def _evaluate_condition(self, rule: AlertRule, metrics: "SystemMetrics") -> bool:
        """Evaluate alert rule condition against metrics."""
        try:
            # Parse condition (e.g., "cpu_utilization > 80")
            parts = rule.condition.split()
            if len(parts) != 3:
                logger.error(f"Invalid alert condition: {rule.condition}")
                return False

            metric_name, operator, threshold_str = parts
            threshold = float(threshold_str)

            # Get metric value
            metric_value = self._get_metric_value(metric_name, metrics)

            # Evaluate condition
            if operator == ">":
                return metric_value > threshold
            elif operator == ">=":
                return metric_value >= threshold
            elif operator == "<":
                return metric_value < threshold
            elif operator == "<=":
                return metric_value <= threshold
            elif operator == "==":
                return metric_value == threshold
            elif operator == "!=":
                return metric_value != threshold
            else:
                logger.error(f"Unknown operator: {operator}")
                return False

        except Exception as e:
            logger.error(f"Error evaluating alert condition: {e}")
            return False

    def _get_metric_value(self, metric_name: str, metrics: "SystemMetrics") -> float:
        """Get metric value from system metrics."""
        if hasattr(metrics, metric_name):
            return getattr(metrics, metric_name)
        elif metric_name in metrics.custom_metrics:
            return metrics.custom_metrics[metric_name].get("value", 0.0)
        else:
            logger.warning(f"Metric not found: {metric_name}")
            return 0.0

    def _generate_alert_message(self, rule: AlertRule, metrics: "SystemMetrics") -> str:
        """Generate alert message."""
        metric_value = self._get_metric_value(rule.condition.split()[0], metrics)

        return (
            f"Alert: {rule.name} - {rule.description}. "
            f"Current value: {metric_value:.2f}, Threshold: {rule.threshold:.2f}"
        )

    def _is_alert_in_cooldown(self, rule: AlertRule) -> bool:
        """Check if alert rule is in cooldown period."""
        # Check if there's a recent alert for this rule
        cutoff_time = time.time() - rule.cooldown

        for alert in self.alert_history:
            if alert.rule_name == rule.name and alert.timestamp > cutoff_time:
                return True

        return False

    async def _send_notifications(self, alert: Alert):
        """Send notifications for an alert."""
        for channel in alert.channels:
            if channel in self.channel_handlers:
                try:
                    handler = self.channel_handlers[channel]
                    if asyncio.iscoroutinefunction(handler):
                        await handler(alert)
                    else:
                        handler(alert)
                except Exception as e:
                    logger.error(f"Error sending notification via {channel.value}: {e}")
            else:
                logger.warning(f"No handler for channel: {channel.value}")

    async def acknowledge_alert(self, alert_id: str, user: str):
        """Acknowledge an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.acknowledged = True
            alert.acknowledged_by = user
            alert.acknowledged_at = time.time()

            self.stats["acknowledged_alerts"] += 1

            logger.info(f"Alert {alert_id} acknowledged by {user}")

    async def resolve_alert(self, alert_id: str, user: Optional[str] = None):
        """Resolve an alert."""
        if alert_id in self.active_alerts:
            alert = self.active_alerts[alert_id]
            alert.resolved = True
            alert.resolved_at = time.time()

            # Move to history
            del self.active_alerts[alert_id]
            self.stats["active_alerts"] -= 1
            self.stats["resolved_alerts"] += 1

            logger.info(f"Alert {alert_id} resolved by {user or 'system'}")

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts."""
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history for the specified hours."""
        cutoff_time = time.time() - (hours * 3600)

        return [alert for alert in self.alert_history if alert.timestamp > cutoff_time]

    def get_alert_statistics(self) -> Dict[str, Any]:
        """Get alert statistics."""
        return {
            "total_alerts": self.stats["total_alerts"],
            "active_alerts": self.stats["active_alerts"],
            "resolved_alerts": self.stats["resolved_alerts"],
            "acknowledged_alerts": self.stats["acknowledged_alerts"],
            "alert_rules": len(self.alert_rules),
            "enabled_rules": len([r for r in self.alert_rules if r.enabled]),
        }

    def is_running(self) -> bool:
        """Check if alert manager is running."""
        return self._running


class ConsoleAlertHandler:
    """Console alert handler for development/testing."""

    async def __call__(self, alert: Alert):
        """Handle alert by printing to console."""
        severity_color = {
            AlertSeverity.INFO: "\033[94m",  # Blue
            AlertSeverity.WARNING: "\033[93m",  # Yellow
            AlertSeverity.CRITICAL: "\033[91m",  # Red
            AlertSeverity.EMERGENCY: "\033[95m",  # Magenta
        }

        color = severity_color.get(alert.severity, "\033[0m")
        reset = "\033[0m"

        print(f"{color}[{alert.severity.value.upper()}] {alert.message}{reset}")
        print(f"  Rule: {alert.rule_name}")
        print(f"  Metric: {alert.metric_name} = {alert.metric_value:.2f}")
        print(f"  Threshold: {alert.threshold:.2f}")
        print(f"  Time: {datetime.fromtimestamp(alert.timestamp)}")
        print()


class LogAlertHandler:
    """Log-based alert handler."""

    def __init__(self, logger_name: str = "alerts"):
        self.logger = logging.getLogger(logger_name)

    async def __call__(self, alert: Alert):
        """Handle alert by logging."""
        log_level = {
            AlertSeverity.INFO: self.logger.info,
            AlertSeverity.WARNING: self.logger.warning,
            AlertSeverity.CRITICAL: self.logger.error,
            AlertSeverity.EMERGENCY: self.logger.critical,
        }

        log_func = log_level.get(alert.severity, self.logger.warning)

        log_func(
            f"Alert: {alert.message} | "
            f"Rule: {alert.rule_name} | "
            f"Metric: {alert.metric_name}={alert.metric_value:.2f} | "
            f"Threshold: {alert.threshold:.2f}"
        )


class WebhookAlertHandler:
    """Webhook-based alert handler."""

    def __init__(self, webhook_url: str, headers: Optional[Dict[str, str]] = None):
        self.webhook_url = webhook_url
        self.headers = headers or {"Content-Type": "application/json"}

    async def __call__(self, alert: Alert):
        """Handle alert by sending webhook."""
        import aiohttp

        payload = {
            "alert_id": alert.id,
            "rule_name": alert.rule_name,
            "severity": alert.severity.value,
            "message": alert.message,
            "timestamp": alert.timestamp,
            "metric_name": alert.metric_name,
            "metric_value": alert.metric_value,
            "threshold": alert.threshold,
            "tags": alert.tags,
        }

        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.webhook_url, json=payload, headers=self.headers
                ) as response:
                    if response.status >= 400:
                        logger.error(f"Webhook failed: {response.status}")
        except Exception as e:
            logger.error(f"Webhook error: {e}")


# Predefined alert rules for common scenarios
DEFAULT_ALERT_RULES = [
    AlertRule(
        name="high_cpu_utilization",
        description="CPU utilization is too high",
        severity=AlertSeverity.WARNING,
        condition="cpu_utilization > 80",
        threshold=80.0,
        duration=300.0,
        channels=[AlertChannel.CONSOLE, AlertChannel.LOG],
    ),
    AlertRule(
        name="high_memory_utilization",
        description="Memory utilization is too high",
        severity=AlertSeverity.WARNING,
        condition="memory_utilization > 85",
        threshold=85.0,
        duration=300.0,
        channels=[AlertChannel.CONSOLE, AlertChannel.LOG],
    ),
    AlertRule(
        name="high_error_rate",
        description="Error rate is too high",
        severity=AlertSeverity.CRITICAL,
        condition="error_rate > 0.05",
        threshold=0.05,
        duration=60.0,
        channels=[AlertChannel.CONSOLE, AlertChannel.LOG, AlertChannel.WEBHOOK],
    ),
    AlertRule(
        name="high_response_time",
        description="Response time is too high",
        severity=AlertSeverity.WARNING,
        condition="avg_response_time > 500",
        threshold=500.0,
        duration=300.0,
        channels=[AlertChannel.CONSOLE, AlertChannel.LOG],
    ),
    AlertRule(
        name="low_cache_hit_rate",
        description="Cache hit rate is too low",
        severity=AlertSeverity.WARNING,
        condition="cache_hit_rate < 0.7",
        threshold=0.7,
        duration=600.0,
        channels=[AlertChannel.CONSOLE, AlertChannel.LOG],
    ),
]
