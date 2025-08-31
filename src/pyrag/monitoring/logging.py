"""
Structured logging and log aggregation for PyRAG.

This module provides structured logging capabilities with log aggregation,
correlation IDs, and integration with monitoring systems.
"""

import asyncio
import time
import json
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
import logging.handlers
from datetime import datetime
import threading
from queue import Queue

logger = logging.getLogger(__name__)


@dataclass
class LogConfig:
    """Logging configuration."""
    enabled: bool = True
    level: str = "INFO"
    format: str = "json"
    output_file: Optional[str] = None
    max_file_size: int = 10 * 1024 * 1024  # 10MB
    backup_count: int = 5
    include_correlation_id: bool = True
    include_timestamp: bool = True
    include_hostname: bool = True
    include_process_id: bool = True


@dataclass
class LogEntry:
    """Structured log entry."""
    timestamp: float
    level: str
    message: str
    correlation_id: Optional[str] = None
    component: Optional[str] = None
    operation: Optional[str] = None
    duration: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    hostname: Optional[str] = None
    process_id: Optional[int] = None


class StructuredLogger:
    """
    Structured logger for PyRAG.
    
    Provides:
    - Structured JSON logging
    - Correlation ID tracking
    - Performance monitoring
    - Error tracking and aggregation
    - Log aggregation capabilities
    """
    
    def __init__(self, config: LogConfig, component: str = "pyrag"):
        """
        Initialize the structured logger.
        
        Args:
            config: Logging configuration
            component: Component name for logging
        """
        self.config = config
        self.component = component
        
        # Logging state
        self.correlation_id: Optional[str] = None
        self.log_queue: Queue = Queue()
        self.log_entries: List[LogEntry] = []
        
        # Background task
        self._logger_task: Optional[asyncio.Task] = None
        self._running = False
        
        # Thread-local storage for correlation IDs
        self._thread_local = threading.local()
        
        # Setup logging handlers
        self._setup_logging()
        
        logger.info("Structured logger initialized")
    
    def _setup_logging(self):
        """Setup logging handlers and formatters."""
        # Create logger
        self.logger = logging.getLogger(f"pyrag.{self.component}")
        self.logger.setLevel(getattr(logging, self.config.level.upper()))
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Create formatter
        if self.config.format == "json":
            formatter = JSONFormatter()
        else:
            formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if configured)
        if self.config.output_file:
            file_handler = logging.handlers.RotatingFileHandler(
                self.config.output_file,
                maxBytes=self.config.max_file_size,
                backupCount=self.config.backup_count
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
    
    def set_correlation_id(self, correlation_id: str):
        """Set correlation ID for current thread."""
        self._thread_local.correlation_id = correlation_id
    
    def get_correlation_id(self) -> Optional[str]:
        """Get correlation ID for current thread."""
        return getattr(self._thread_local, 'correlation_id', None)
    
    def generate_correlation_id(self) -> str:
        """Generate a new correlation ID."""
        return str(uuid.uuid4())
    
    async def start(self):
        """Start the structured logger."""
        if self._running:
            logger.warning("Structured logger is already running")
            return
        
        self._running = True
        self._logger_task = asyncio.create_task(self._logger_loop())
        logger.info("Structured logger started")
    
    async def stop(self):
        """Stop the structured logger."""
        self._running = False
        
        if self._logger_task and not self._logger_task.done():
            self._logger_task.cancel()
            try:
                await self._logger_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Structured logger stopped")
    
    async def _logger_loop(self):
        """Main logging loop for processing log entries."""
        while self._running:
            try:
                # Process log entries from queue
                while not self.log_queue.empty():
                    entry = self.log_queue.get_nowait()
                    await self._process_log_entry(entry)
                
                await asyncio.sleep(0.1)  # Small delay
                
            except Exception as e:
                logger.error(f"Logger loop error: {e}")
                await asyncio.sleep(1.0)
    
    async def _process_log_entry(self, entry: LogEntry):
        """Process a log entry."""
        # Add to history
        self.log_entries.append(entry)
        
        # Keep only recent entries (last 1000)
        if len(self.log_entries) > 1000:
            self.log_entries = self.log_entries[-1000:]
        
        # Log using standard logging
        log_level = getattr(logging, entry.level.upper(), logging.INFO)
        self.logger.log(log_level, entry.message, extra={
            'correlation_id': entry.correlation_id,
            'component': entry.component,
            'operation': entry.operation,
            'duration': entry.duration,
            'error': entry.error,
            'metadata': entry.metadata
        })
    
    async def log_event(self, event_type: str, message: str, **kwargs):
        """Log an event with structured data."""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            message=f"{event_type}: {message}",
            correlation_id=self.get_correlation_id(),
            component=self.component,
            operation=event_type,
            metadata=kwargs
        )
        
        self.log_queue.put(entry)
    
    async def log_metrics(self, metrics: 'SystemMetrics'):
        """Log system metrics."""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            message="System metrics collected",
            correlation_id=self.get_correlation_id(),
            component=self.component,
            operation="metrics_collection",
            metadata={
                "cpu_utilization": metrics.cpu_utilization,
                "memory_utilization": metrics.memory_utilization,
                "total_queries": metrics.total_queries,
                "avg_response_time": metrics.avg_response_time,
                "error_rate": metrics.error_rate,
                "cache_hit_rate": metrics.cache_hit_rate
            }
        )
        
        self.log_queue.put(entry)
    
    async def log_error(self, error: Exception, operation: str, **kwargs):
        """Log an error with structured data."""
        entry = LogEntry(
            timestamp=time.time(),
            level="ERROR",
            message=str(error),
            correlation_id=self.get_correlation_id(),
            component=self.component,
            operation=operation,
            error=type(error).__name__,
            metadata=kwargs
        )
        
        self.log_queue.put(entry)
    
    async def log_performance(self, operation: str, duration: float, **kwargs):
        """Log performance data."""
        entry = LogEntry(
            timestamp=time.time(),
            level="INFO",
            message=f"Performance: {operation} took {duration:.3f}s",
            correlation_id=self.get_correlation_id(),
            component=self.component,
            operation=operation,
            duration=duration,
            metadata=kwargs
        )
        
        self.log_queue.put(entry)
    
    def get_log_entries(self, hours: int = 24) -> List[LogEntry]:
        """Get log entries for the specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            entry for entry in self.log_entries
            if entry.timestamp > cutoff_time
        ]
    
    def get_error_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get error summary for the specified hours."""
        recent_entries = self.get_log_entries(hours)
        error_entries = [entry for entry in recent_entries if entry.level == "ERROR"]
        
        error_counts = {}
        for entry in error_entries:
            error_type = entry.error or "unknown"
            error_counts[error_type] = error_counts.get(error_type, 0) + 1
        
        return {
            "total_errors": len(error_entries),
            "error_types": error_counts,
            "period_hours": hours
        }
    
    def is_running(self) -> bool:
        """Check if logger is running."""
        return self._running


class JSONFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record):
        """Format log record as JSON."""
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage()
        }
        
        # Add extra fields
        if hasattr(record, 'correlation_id') and record.correlation_id:
            log_data["correlation_id"] = record.correlation_id
        
        if hasattr(record, 'component') and record.component:
            log_data["component"] = record.component
        
        if hasattr(record, 'operation') and record.operation:
            log_data["operation"] = record.operation
        
        if hasattr(record, 'duration') and record.duration:
            log_data["duration"] = record.duration
        
        if hasattr(record, 'error') and record.error:
            log_data["error"] = record.error
        
        if hasattr(record, 'metadata') and record.metadata:
            log_data["metadata"] = record.metadata
        
        # Add exception info
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        return json.dumps(log_data)


class LogAggregator:
    """
    Log aggregator for collecting logs from multiple sources.
    
    Provides:
    - Centralized log collection
    - Log filtering and search
    - Log analysis and statistics
    - Integration with monitoring systems
    """
    
    def __init__(self, max_entries: int = 10000):
        """
        Initialize the log aggregator.
        
        Args:
            max_entries: Maximum number of log entries to keep in memory
        """
        self.max_entries = max_entries
        self.log_entries: List[LogEntry] = []
        self.loggers: List[StructuredLogger] = []
        
        logger.info("Log aggregator initialized")
    
    def add_logger(self, logger: StructuredLogger):
        """Add a logger to the aggregator."""
        self.loggers.append(logger)
        logger.info(f"Added logger to aggregator: {logger.component}")
    
    def add_log_entry(self, entry: LogEntry):
        """Add a log entry to the aggregator."""
        self.log_entries.append(entry)
        
        # Keep only recent entries
        if len(self.log_entries) > self.max_entries:
            self.log_entries = self.log_entries[-self.max_entries:]
    
    def search_logs(self, query: str, hours: int = 24) -> List[LogEntry]:
        """Search logs for the specified query."""
        cutoff_time = time.time() - (hours * 3600)
        recent_entries = [
            entry for entry in self.log_entries
            if entry.timestamp > cutoff_time
        ]
        
        # Simple text search
        matching_entries = []
        query_lower = query.lower()
        
        for entry in recent_entries:
            if (query_lower in entry.message.lower() or
                query_lower in entry.component.lower() or
                query_lower in (entry.operation or "").lower()):
                matching_entries.append(entry)
        
        return matching_entries
    
    def get_log_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get log statistics for the specified hours."""
        cutoff_time = time.time() - (hours * 3600)
        recent_entries = [
            entry for entry in self.log_entries
            if entry.timestamp > cutoff_time
        ]
        
        # Count by level
        level_counts = {}
        for entry in recent_entries:
            level_counts[entry.level] = level_counts.get(entry.level, 0) + 1
        
        # Count by component
        component_counts = {}
        for entry in recent_entries:
            component_counts[entry.component] = component_counts.get(entry.component, 0) + 1
        
        # Count by operation
        operation_counts = {}
        for entry in recent_entries:
            if entry.operation:
                operation_counts[entry.operation] = operation_counts.get(entry.operation, 0) + 1
        
        return {
            "total_entries": len(recent_entries),
            "level_counts": level_counts,
            "component_counts": component_counts,
            "operation_counts": operation_counts,
            "period_hours": hours
        }
    
    def get_performance_summary(self, hours: int = 24) -> Dict[str, Any]:
        """Get performance summary from logs."""
        cutoff_time = time.time() - (hours * 3600)
        recent_entries = [
            entry for entry in self.log_entries
            if entry.timestamp > cutoff_time and entry.duration is not None
        ]
        
        if not recent_entries:
            return {"error": "No performance data available"}
        
        durations = [entry.duration for entry in recent_entries]
        
        return {
            "total_operations": len(recent_entries),
            "avg_duration": sum(durations) / len(durations),
            "min_duration": min(durations),
            "max_duration": max(durations),
            "period_hours": hours
        }
    
    def clear_logs(self):
        """Clear all log entries."""
        self.log_entries.clear()
        logger.info("Log aggregator cleared")
