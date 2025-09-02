"""Logging setup and configuration."""

import logging
import sys
import uuid
from typing import Any, Dict, Optional

import structlog
from structlog.stdlib import LoggerFactory

from ..config import get_config


def setup_logging() -> None:
    """Setup structured logging with correlation IDs."""

    # Get configuration
    config = get_config()

    # Configure structlog
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.dev.ConsoleRenderer(),  # Use console renderer for now
        ],
        context_class=dict,
        logger_factory=LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=getattr(logging, config.log_level),
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class CorrelationContext:
    """Context manager for correlation IDs."""

    def __init__(self, correlation_id: Optional[str] = None):
        self.correlation_id = correlation_id or str(uuid.uuid4())
        self._previous_context: Optional[Dict[str, Any]] = None

    def __enter__(self) -> str:
        """Enter correlation context."""
        self._previous_context = structlog.contextvars.get_contextvars().copy()
        structlog.contextvars.clear_contextvars()
        structlog.contextvars.bind_contextvars(correlation_id=self.correlation_id)
        return self.correlation_id

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit correlation context."""
        structlog.contextvars.clear_contextvars()
        if self._previous_context:
            for key, value in self._previous_context.items():
                structlog.contextvars.bind_contextvars(**{key: value})


def log_with_correlation(correlation_id: Optional[str] = None):
    """Decorator to add correlation ID to function calls."""

    def decorator(func):
        def wrapper(*args, **kwargs):
            with CorrelationContext(correlation_id):
                logger = get_logger(func.__module__)
                logger.info(
                    "Function called",
                    function=func.__name__,
                    args_count=len(args),
                    kwargs_keys=list(kwargs.keys()),
                )
                try:
                    result = func(*args, **kwargs)
                    logger.info(
                        "Function completed successfully", function=func.__name__
                    )
                    return result
                except Exception as e:
                    logger.error(
                        "Function failed",
                        function=func.__name__,
                        error=str(e),
                        error_type=type(e).__name__,
                    )
                    raise

        return wrapper

    return decorator
