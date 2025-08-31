"""
Health monitoring for PyRAG.

This module provides comprehensive health monitoring capabilities for
checking system health, dependencies, and service status.
"""

import asyncio
import time
import psutil
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
import logging
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status enumeration."""
    HEALTHY = "healthy"
    DEGRADED = "degraded"
    UNHEALTHY = "unhealthy"
    UNKNOWN = "unknown"


@dataclass
class HealthCheck:
    """Health check configuration."""
    name: str
    description: str
    check_function: Callable
    timeout: float = 30.0
    interval: float = 60.0
    critical: bool = True
    tags: List[str] = field(default_factory=list)


@dataclass
class HealthCheckResult:
    """Result of a health check."""
    name: str
    status: HealthStatus
    message: str
    timestamp: float
    duration: float
    details: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


@dataclass
class HealthMetrics:
    """Health-related metrics."""
    cpu_usage: float
    memory_usage: float
    disk_usage: float
    network_io: Dict[str, float]
    process_count: int
    uptime: float
    load_average: List[float]


class HealthMonitor:
    """
    Health monitor for PyRAG.
    
    Provides:
    - System health checks
    - Dependency health monitoring
    - Performance health metrics
    - Health status aggregation
    - Health check scheduling
    """
    
    def __init__(self, check_interval: float = 30.0):
        """
        Initialize the health monitor.
        
        Args:
            check_interval: Interval between health checks (seconds)
        """
        self.check_interval = check_interval
        
        # Health checks
        self.health_checks: List[HealthCheck] = []
        self.check_results: Dict[str, HealthCheckResult] = {}
        
        # Background task
        self._monitor_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Health metrics
        self.health_metrics: Optional[HealthMetrics] = None
        
        # Initialize default health checks
        self._initialize_default_checks()
        
        logger.info("Health monitor initialized")
    
    def _initialize_default_checks(self):
        """Initialize default health checks."""
        # System health checks
        self.add_health_check(HealthCheck(
            name="system_cpu",
            description="Check CPU usage",
            check_function=self._check_cpu_usage,
            timeout=10.0,
            interval=30.0,
            critical=False,
            tags=["system", "cpu"]
        ))
        
        self.add_health_check(HealthCheck(
            name="system_memory",
            description="Check memory usage",
            check_function=self._check_memory_usage,
            timeout=10.0,
            interval=30.0,
            critical=False,
            tags=["system", "memory"]
        ))
        
        self.add_health_check(HealthCheck(
            name="system_disk",
            description="Check disk usage",
            check_function=self._check_disk_usage,
            timeout=10.0,
            interval=60.0,
            critical=False,
            tags=["system", "disk"]
        ))
        
        self.add_health_check(HealthCheck(
            name="system_uptime",
            description="Check system uptime",
            check_function=self._check_uptime,
            timeout=5.0,
            interval=300.0,  # 5 minutes
            critical=False,
            tags=["system", "uptime"]
        ))
    
    def add_health_check(self, health_check: HealthCheck):
        """Add a health check."""
        self.health_checks.append(health_check)
        logger.info(f"Added health check: {health_check.name}")
    
    async def start(self):
        """Start the health monitor."""
        if self._running:
            logger.warning("Health monitor is already running")
            return
        
        self._running = True
        self._monitor_task = asyncio.create_task(self._monitor_loop())
        logger.info("Health monitor started")
    
    async def stop(self):
        """Stop the health monitor."""
        self._running = False
        
        if self._monitor_task and not self._monitor_task.done():
            self._monitor_task.cancel()
            try:
                await self._monitor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Health monitor stopped")
    
    async def _monitor_loop(self):
        """Main health monitoring loop."""
        while self._running:
            try:
                # Run health checks
                await self._run_health_checks()
                
                # Update health metrics
                await self._update_health_metrics()
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                logger.error(f"Health monitor loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _run_health_checks(self):
        """Run all health checks."""
        for health_check in self.health_checks:
            try:
                result = await self._run_single_check(health_check)
                self.check_results[health_check.name] = result
            except Exception as e:
                logger.error(f"Health check error for {health_check.name}: {e}")
                
                # Create error result
                result = HealthCheckResult(
                    name=health_check.name,
                    status=HealthStatus.UNHEALTHY,
                    message=f"Health check failed: {e}",
                    timestamp=time.time(),
                    duration=0.0,
                    error=str(e)
                )
                self.check_results[health_check.name] = result
    
    async def _run_single_check(self, health_check: HealthCheck) -> HealthCheckResult:
        """Run a single health check."""
        start_time = time.time()
        
        try:
            # Run check with timeout
            if asyncio.iscoroutinefunction(health_check.check_function):
                result = await asyncio.wait_for(
                    health_check.check_function(),
                    timeout=health_check.timeout
                )
            else:
                # Run sync function in thread pool
                loop = asyncio.get_event_loop()
                result = await asyncio.wait_for(
                    loop.run_in_executor(None, health_check.check_function),
                    timeout=health_check.timeout
                )
            
            duration = time.time() - start_time
            
            return HealthCheckResult(
                name=health_check.name,
                status=result.get("status", HealthStatus.UNKNOWN),
                message=result.get("message", "Health check completed"),
                timestamp=time.time(),
                duration=duration,
                details=result.get("details", {})
            )
            
        except asyncio.TimeoutError:
            duration = time.time() - start_time
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message="Health check timed out",
                timestamp=time.time(),
                duration=duration,
                error="timeout"
            )
        except Exception as e:
            duration = time.time() - start_time
            return HealthCheckResult(
                name=health_check.name,
                status=HealthStatus.UNHEALTHY,
                message=f"Health check failed: {e}",
                timestamp=time.time(),
                duration=duration,
                error=str(e)
            )
    
    async def _update_health_metrics(self):
        """Update health metrics."""
        try:
            # Get system metrics
            cpu_usage = psutil.cpu_percent(interval=1)
            memory = psutil.virtual_memory()
            disk = psutil.disk_usage('/')
            network = psutil.net_io_counters()
            
            # Get load average (Unix-like systems)
            try:
                load_avg = psutil.getloadavg()
            except AttributeError:
                load_avg = [0.0, 0.0, 0.0]
            
            self.health_metrics = HealthMetrics(
                cpu_usage=cpu_usage,
                memory_usage=memory.percent,
                disk_usage=disk.percent,
                network_io={
                    "bytes_sent": network.bytes_sent,
                    "bytes_recv": network.bytes_recv,
                    "packets_sent": network.packets_sent,
                    "packets_recv": network.packets_recv
                },
                process_count=len(psutil.pids()),
                uptime=time.time() - psutil.boot_time(),
                load_average=list(load_avg)
            )
            
        except Exception as e:
            logger.error(f"Error updating health metrics: {e}")
    
    # Default health check functions
    async def _check_cpu_usage(self) -> Dict[str, Any]:
        """Check CPU usage."""
        cpu_usage = psutil.cpu_percent(interval=1)
        
        if cpu_usage > 90:
            status = HealthStatus.UNHEALTHY
            message = f"CPU usage is very high: {cpu_usage:.1f}%"
        elif cpu_usage > 80:
            status = HealthStatus.DEGRADED
            message = f"CPU usage is high: {cpu_usage:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"CPU usage is normal: {cpu_usage:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {"cpu_usage": cpu_usage}
        }
    
    async def _check_memory_usage(self) -> Dict[str, Any]:
        """Check memory usage."""
        memory = psutil.virtual_memory()
        
        if memory.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Memory usage is critical: {memory.percent:.1f}%"
        elif memory.percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Memory usage is high: {memory.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Memory usage is normal: {memory.percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "memory_usage": memory.percent,
                "memory_available": memory.available,
                "memory_total": memory.total
            }
        }
    
    async def _check_disk_usage(self) -> Dict[str, Any]:
        """Check disk usage."""
        disk = psutil.disk_usage('/')
        
        if disk.percent > 95:
            status = HealthStatus.UNHEALTHY
            message = f"Disk usage is critical: {disk.percent:.1f}%"
        elif disk.percent > 85:
            status = HealthStatus.DEGRADED
            message = f"Disk usage is high: {disk.percent:.1f}%"
        else:
            status = HealthStatus.HEALTHY
            message = f"Disk usage is normal: {disk.percent:.1f}%"
        
        return {
            "status": status,
            "message": message,
            "details": {
                "disk_usage": disk.percent,
                "disk_free": disk.free,
                "disk_total": disk.total
            }
        }
    
    async def _check_uptime(self) -> Dict[str, Any]:
        """Check system uptime."""
        uptime = time.time() - psutil.boot_time()
        uptime_hours = uptime / 3600
        
        if uptime_hours < 1:
            status = HealthStatus.DEGRADED
            message = f"System recently restarted: {uptime_hours:.1f} hours"
        else:
            status = HealthStatus.HEALTHY
            message = f"System uptime: {uptime_hours:.1f} hours"
        
        return {
            "status": status,
            "message": message,
            "details": {"uptime_hours": uptime_hours}
        }
    
    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status."""
        if not self.check_results:
            return {
                "status": HealthStatus.UNKNOWN.value,
                "message": "No health checks have been run",
                "checks": {},
                "timestamp": time.time()
            }
        
        # Determine overall status
        critical_checks = [check for check in self.health_checks if check.critical]
        critical_results = [self.check_results.get(check.name) for check in critical_checks]
        
        # Check if any critical checks are unhealthy
        unhealthy_critical = any(
            result and result.status == HealthStatus.UNHEALTHY
            for result in critical_results
        )
        
        # Check if any checks are degraded
        degraded_checks = any(
            result and result.status == HealthStatus.DEGRADED
            for result in self.check_results.values()
        )
        
        if unhealthy_critical:
            overall_status = HealthStatus.UNHEALTHY
            message = "Critical health checks are failing"
        elif degraded_checks:
            overall_status = HealthStatus.DEGRADED
            message = "Some health checks are degraded"
        else:
            overall_status = HealthStatus.HEALTHY
            message = "All health checks are passing"
        
        # Format check results
        check_results = {}
        for name, result in self.check_results.items():
            check_results[name] = {
                "status": result.status.value,
                "message": result.message,
                "timestamp": result.timestamp,
                "duration": result.duration,
                "details": result.details,
                "error": result.error
            }
        
        return {
            "status": overall_status.value,
            "message": message,
            "checks": check_results,
            "timestamp": time.time(),
            "metrics": self._get_health_metrics_dict()
        }
    
    def _get_health_metrics_dict(self) -> Optional[Dict[str, Any]]:
        """Get health metrics as dictionary."""
        if not self.health_metrics:
            return None
        
        return {
            "cpu_usage": self.health_metrics.cpu_usage,
            "memory_usage": self.health_metrics.memory_usage,
            "disk_usage": self.health_metrics.disk_usage,
            "network_io": self.health_metrics.network_io,
            "process_count": self.health_metrics.process_count,
            "uptime": self.health_metrics.uptime,
            "load_average": self.health_metrics.load_average
        }
    
    def get_health_check_result(self, check_name: str) -> Optional[HealthCheckResult]:
        """Get result of a specific health check."""
        return self.check_results.get(check_name)
    
    def get_health_metrics(self) -> Optional[HealthMetrics]:
        """Get current health metrics."""
        return self.health_metrics
    
    def is_running(self) -> bool:
        """Check if health monitor is running."""
        return self._running
