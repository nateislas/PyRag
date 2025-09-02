"""
Health checking and monitoring for PyRAG backends.

This module provides comprehensive health checking, performance monitoring,
and alerting capabilities for production deployments.
"""

import asyncio
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional

import aiohttp

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheckResult:
    """Result of a health check."""

    status: HealthStatus
    response_time: float
    error_message: Optional[str] = None
    timestamp: float = field(default_factory=time.time)
    details: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HealthMetrics:
    """Health metrics for a service."""

    service_id: str
    status: HealthStatus
    last_check: float
    response_time_avg: float
    response_time_p95: float
    response_time_p99: float
    error_rate: float
    uptime_percentage: float
    consecutive_failures: int
    total_checks: int
    total_failures: int
    last_error: Optional[str] = None


class HealthChecker:
    """
    Comprehensive health checker for PyRAG services.

    Provides:
    - Regular health checks for all services
    - Performance metrics collection
    - Alerting for unhealthy services
    - Historical health data tracking
    """

    def __init__(
        self,
        check_interval: float = 30.0,
        timeout: float = 10.0,
        max_consecutive_failures: int = 3,
        alert_callbacks: Optional[List[Callable]] = None,
    ):
        """
        Initialize the health checker.

        Args:
            check_interval: Interval between health checks (seconds)
            timeout: Timeout for health check requests (seconds)
            max_consecutive_failures: Max failures before marking unhealthy
            alert_callbacks: List of callback functions for alerts
        """
        self.check_interval = check_interval
        self.timeout = timeout
        self.max_consecutive_failures = max_consecutive_failures
        self.alert_callbacks = alert_callbacks or []

        # Service tracking
        self.services: Dict[str, Dict[str, Any]] = {}
        self.health_metrics: Dict[str, HealthMetrics] = {}
        self.health_history: Dict[str, List[HealthCheckResult]] = {}

        # Performance tracking
        self.response_times: Dict[str, List[float]] = {}
        self.error_counts: Dict[str, int] = {}

        # Background task
        self._check_task: Optional[asyncio.Task] = None
        self._running = False

    def add_service(
        self,
        service_id: str,
        health_check_url: str,
        expected_status: int = 200,
        custom_headers: Optional[Dict[str, str]] = None,
        check_method: str = "GET",
        check_body: Optional[str] = None,
    ):
        """
        Add a service for health checking.

        Args:
            service_id: Unique identifier for the service
            health_check_url: URL to check for health
            expected_status: Expected HTTP status code
            custom_headers: Custom headers for the health check
            check_method: HTTP method for health check
            check_body: Request body for health check
        """
        self.services[service_id] = {
            "url": health_check_url,
            "expected_status": expected_status,
            "headers": custom_headers or {},
            "method": check_method,
            "body": check_body,
            "added_at": time.time(),
        }

        # Initialize metrics
        self.health_metrics[service_id] = HealthMetrics(
            service_id=service_id,
            status=HealthStatus.UNKNOWN,
            last_check=0.0,
            response_time_avg=0.0,
            response_time_p95=0.0,
            response_time_p99=0.0,
            error_rate=0.0,
            uptime_percentage=100.0,
            consecutive_failures=0,
            total_checks=0,
            total_failures=0,
        )

        self.health_history[service_id] = []
        self.response_times[service_id] = []
        self.error_counts[service_id] = 0

        logger.info(f"Added service {service_id} for health checking")

    async def start(self):
        """Start the health checking background task."""
        if self._running:
            logger.warning("Health checker is already running")
            return

        self._running = True
        self._check_task = asyncio.create_task(self._health_check_loop())
        logger.info("Health checker started")

    async def stop(self):
        """Stop the health checking background task."""
        self._running = False

        if self._check_task and not self._check_task.done():
            self._check_task.cancel()
            try:
                await self._check_task
            except asyncio.CancelledError:
                pass

        logger.info("Health checker stopped")

    async def _health_check_loop(self):
        """Main health checking loop."""
        while self._running:
            try:
                await self._check_all_services()
                await asyncio.sleep(self.check_interval)
            except Exception as e:
                logger.error(f"Health check loop error: {e}")
                await asyncio.sleep(5.0)  # Short delay on error

    async def _check_all_services(self):
        """Check health of all registered services."""
        if not self.services:
            return

        tasks = []
        for service_id in self.services:
            tasks.append(self._check_service_health(service_id))

        await asyncio.gather(*tasks, return_exceptions=True)

    async def _check_service_health(self, service_id: str) -> HealthCheckResult:
        """Check health of a specific service."""
        service_config = self.services[service_id]
        metrics = self.health_metrics[service_id]

        start_time = time.time()
        result = HealthCheckResult(status=HealthStatus.UNKNOWN)

        try:
            async with aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self.timeout)
            ) as session:
                # Prepare request
                kwargs = {
                    "headers": service_config["headers"],
                    "timeout": aiohttp.ClientTimeout(total=self.timeout),
                }

                if service_config["body"]:
                    kwargs["data"] = service_config["body"]

                # Make request
                async with session.request(
                    service_config["method"], service_config["url"], **kwargs
                ) as response:
                    response_time = time.time() - start_time
                    result.response_time = response_time

                    # Check response
                    if response.status == service_config["expected_status"]:
                        result.status = HealthStatus.HEALTHY
                        result.details["status_code"] = response.status

                        # Check response time for degraded status
                        if response_time > 1.0:  # >1s response time
                            result.status = HealthStatus.DEGRADED
                    else:
                        result.status = HealthStatus.UNHEALTHY
                        result.error_message = (
                            f"Unexpected status code: {response.status}"
                        )
                        result.details["status_code"] = response.status

        except asyncio.TimeoutError:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = "Health check timeout"
            result.response_time = self.timeout

        except Exception as e:
            result.status = HealthStatus.UNHEALTHY
            result.error_message = str(e)
            result.response_time = time.time() - start_time

        # Update metrics
        await self._update_service_metrics(service_id, result)

        # Store in history
        self.health_history[service_id].append(result)

        # Keep only recent history (last 100 checks)
        if len(self.health_history[service_id]) > 100:
            self.health_history[service_id] = self.health_history[service_id][-50:]

        return result

    async def _update_service_metrics(self, service_id: str, result: HealthCheckResult):
        """Update health metrics for a service."""
        metrics = self.health_metrics[service_id]

        # Update basic metrics
        metrics.last_check = result.timestamp
        metrics.total_checks += 1

        # Update response time metrics
        self.response_times[service_id].append(result.response_time)

        # Keep only recent response times (last 100)
        if len(self.response_times[service_id]) > 100:
            self.response_times[service_id] = self.response_times[service_id][-50:]

        # Calculate response time statistics
        response_times = self.response_times[service_id]
        if response_times:
            metrics.response_time_avg = sum(response_times) / len(response_times)

            # Calculate percentiles
            sorted_times = sorted(response_times)
            p95_index = int(len(sorted_times) * 0.95)
            p99_index = int(len(sorted_times) * 0.99)

            metrics.response_time_p95 = (
                sorted_times[p95_index]
                if p95_index < len(sorted_times)
                else sorted_times[-1]
            )
            metrics.response_time_p99 = (
                sorted_times[p99_index]
                if p99_index < len(sorted_times)
                else sorted_times[-1]
            )

        # Update failure metrics
        if result.status == HealthStatus.UNHEALTHY:
            metrics.consecutive_failures += 1
            metrics.total_failures += 1
            metrics.last_error = result.error_message
            self.error_counts[service_id] += 1
        else:
            metrics.consecutive_failures = 0

        # Calculate error rate
        metrics.error_rate = metrics.total_failures / max(1, metrics.total_checks)

        # Calculate uptime percentage
        recent_checks = self.health_history[service_id][-20:]  # Last 20 checks
        if recent_checks:
            healthy_checks = sum(
                1 for check in recent_checks if check.status == HealthStatus.HEALTHY
            )
            metrics.uptime_percentage = (healthy_checks / len(recent_checks)) * 100

        # Determine overall status
        if metrics.consecutive_failures >= self.max_consecutive_failures:
            metrics.status = HealthStatus.UNHEALTHY
        elif result.status == HealthStatus.DEGRADED or metrics.error_rate > 0.1:
            metrics.status = HealthStatus.DEGRADED
        elif result.status == HealthStatus.HEALTHY:
            metrics.status = HealthStatus.HEALTHY

        # Trigger alerts if status changed to unhealthy
        if (
            metrics.status == HealthStatus.UNHEALTHY
            and metrics.consecutive_failures == self.max_consecutive_failures
        ):
            await self._trigger_alert(
                service_id, "Service marked as unhealthy", metrics
            )

    async def _trigger_alert(
        self, service_id: str, message: str, metrics: HealthMetrics
    ):
        """Trigger alert callbacks."""
        alert_data = {
            "service_id": service_id,
            "message": message,
            "timestamp": time.time(),
            "metrics": {
                "status": metrics.status.value,
                "error_rate": metrics.error_rate,
                "response_time_avg": metrics.response_time_avg,
                "consecutive_failures": metrics.consecutive_failures,
                "last_error": metrics.last_error,
            },
        }

        for callback in self.alert_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(alert_data)
                else:
                    callback(alert_data)
            except Exception as e:
                logger.error(f"Alert callback error: {e}")

    def get_service_health(self, service_id: str) -> Optional[HealthMetrics]:
        """Get health metrics for a specific service."""
        return self.health_metrics.get(service_id)

    def get_all_health_metrics(self) -> Dict[str, HealthMetrics]:
        """Get health metrics for all services."""
        return self.health_metrics.copy()

    def get_health_summary(self) -> Dict[str, Any]:
        """Get summary of all service health."""
        summary = {
            "total_services": len(self.services),
            "healthy_services": 0,
            "degraded_services": 0,
            "unhealthy_services": 0,
            "unknown_services": 0,
            "overall_uptime": 0.0,
            "services": {},
        }

        total_uptime = 0.0

        for service_id, metrics in self.health_metrics.items():
            summary["services"][service_id] = {
                "status": metrics.status.value,
                "uptime_percentage": metrics.uptime_percentage,
                "response_time_avg": metrics.response_time_avg,
                "error_rate": metrics.error_rate,
                "last_check": metrics.last_check,
            }

            if metrics.status == HealthStatus.HEALTHY:
                summary["healthy_services"] += 1
            elif metrics.status == HealthStatus.DEGRADED:
                summary["degraded_services"] += 1
            elif metrics.status == HealthStatus.UNHEALTHY:
                summary["unhealthy_services"] += 1
            else:
                summary["unknown_services"] += 1

            total_uptime += metrics.uptime_percentage

        if self.health_metrics:
            summary["overall_uptime"] = total_uptime / len(self.health_metrics)

        return summary

    async def manual_health_check(self, service_id: str) -> HealthCheckResult:
        """Perform a manual health check for a service."""
        if service_id not in self.services:
            raise ValueError(f"Service {service_id} not found")

        return await self._check_service_health(service_id)
