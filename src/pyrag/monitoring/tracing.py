"""
Distributed tracing for PyRAG.

This module provides distributed tracing capabilities for tracking
request flows across different components and services.
"""

import asyncio
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any
import logging
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


@dataclass
class SpanContext:
    """Span context for distributed tracing."""
    trace_id: str
    span_id: str
    parent_span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Span:
    """Distributed tracing span."""
    span_id: str
    trace_id: str
    name: str
    start_time: float
    end_time: Optional[float] = None
    parent_span_id: Optional[str] = None
    correlation_id: Optional[str] = None
    tags: Dict[str, Any] = field(default_factory=dict)
    events: List[Dict[str, Any]] = field(default_factory=list)
    baggage: Dict[str, str] = field(default_factory=dict)


@dataclass
class Trace:
    """Complete trace with multiple spans."""
    trace_id: str
    start_time: float
    correlation_id: Optional[str] = None
    end_time: Optional[float] = None
    spans: List[Span] = field(default_factory=list)
    tags: Dict[str, Any] = field(default_factory=dict)


class TraceCollector:
    """
    Distributed tracing collector for PyRAG.
    
    Provides:
    - Span creation and management
    - Trace context propagation
    - Performance analysis
    - Request flow visualization
    - Integration with monitoring systems
    """
    
    def __init__(self, max_traces: int = 1000):
        """
        Initialize the trace collector.
        
        Args:
            max_traces: Maximum number of traces to keep in memory
        """
        self.max_traces = max_traces
        self.traces: Dict[str, Trace] = {}
        self.active_spans: Dict[str, Span] = {}
        
        # Background task
        self._collector_task: Optional[asyncio.Task] = None
        self._running = False
        
        logger.info("Trace collector initialized")
    
    async def start(self):
        """Start the trace collector."""
        if self._running:
            logger.warning("Trace collector is already running")
            return
        
        self._running = True
        self._collector_task = asyncio.create_task(self._collector_loop())
        logger.info("Trace collector started")
    
    async def stop(self):
        """Stop the trace collector."""
        self._running = False
        
        if self._collector_task and not self._collector_task.done():
            self._collector_task.cancel()
            try:
                await self._collector_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Trace collector stopped")
    
    async def _collector_loop(self):
        """Main collector loop for processing traces."""
        while self._running:
            try:
                # Cleanup old traces
                await self._cleanup_old_traces()
                
                await asyncio.sleep(30.0)  # Run every 30 seconds
                
            except Exception as e:
                logger.error(f"Collector loop error: {e}")
                await asyncio.sleep(5.0)
    
    async def _cleanup_old_traces(self):
        """Clean up old traces."""
        current_time = time.time()
        cutoff_time = current_time - (24 * 3600)  # Keep traces for 24 hours
        
        traces_to_remove = []
        for trace_id, trace in self.traces.items():
            if trace.start_time < cutoff_time:
                traces_to_remove.append(trace_id)
        
        for trace_id in traces_to_remove:
            del self.traces[trace_id]
        
        if traces_to_remove:
            logger.debug(f"Cleaned up {len(traces_to_remove)} old traces")
    
    def create_trace(self, correlation_id: Optional[str] = None) -> Trace:
        """Create a new trace."""
        trace_id = str(uuid.uuid4())
        trace = Trace(
            trace_id=trace_id,
            correlation_id=correlation_id,
            start_time=time.time()
        )
        
        self.traces[trace_id] = trace
        
        # Limit number of traces
        if len(self.traces) > self.max_traces:
            # Remove oldest trace
            oldest_trace_id = min(self.traces.keys(), key=lambda tid: self.traces[tid].start_time)
            del self.traces[oldest_trace_id]
        
        logger.debug(f"Created trace: {trace_id}")
        return trace
    
    def create_span(
        self,
        name: str,
        trace_id: str,
        parent_span_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        tags: Optional[Dict[str, Any]] = None
    ) -> Span:
        """Create a new span."""
        span_id = str(uuid.uuid4())
        span = Span(
            span_id=span_id,
            trace_id=trace_id,
            name=name,
            start_time=time.time(),
            parent_span_id=parent_span_id,
            correlation_id=correlation_id,
            tags=tags or {}
        )
        
        # Add span to trace
        if trace_id in self.traces:
            self.traces[trace_id].spans.append(span)
        
        # Track active span
        self.active_spans[span_id] = span
        
        logger.debug(f"Created span: {span_id} in trace: {trace_id}")
        return span
    
    def end_span(self, span_id: str, tags: Optional[Dict[str, Any]] = None):
        """End a span."""
        if span_id not in self.active_spans:
            logger.warning(f"Span not found: {span_id}")
            return
        
        span = self.active_spans[span_id]
        span.end_time = time.time()
        
        # Add end tags
        if tags:
            span.tags.update(tags)
        
        # Remove from active spans
        del self.active_spans[span_id]
        
        logger.debug(f"Ended span: {span_id}")
    
    def add_span_event(self, span_id: str, event_name: str, attributes: Optional[Dict[str, Any]] = None):
        """Add an event to a span."""
        if span_id not in self.active_spans:
            logger.warning(f"Span not found: {span_id}")
            return
        
        span = self.active_spans[span_id]
        event = {
            "name": event_name,
            "timestamp": time.time(),
            "attributes": attributes or {}
        }
        span.events.append(event)
        
        logger.debug(f"Added event to span {span_id}: {event_name}")
    
    def add_span_tag(self, span_id: str, key: str, value: Any):
        """Add a tag to a span."""
        if span_id not in self.active_spans:
            logger.warning(f"Span not found: {span_id}")
            return
        
        span = self.active_spans[span_id]
        span.tags[key] = value
        
        logger.debug(f"Added tag to span {span_id}: {key}={value}")
    
    def get_trace(self, trace_id: str) -> Optional[Trace]:
        """Get a trace by ID."""
        return self.traces.get(trace_id)
    
    def get_traces_by_correlation_id(self, correlation_id: str) -> List[Trace]:
        """Get traces by correlation ID."""
        return [
            trace for trace in self.traces.values()
            if trace.correlation_id == correlation_id
        ]
    
    def get_recent_traces(self, hours: int = 1) -> List[Trace]:
        """Get recent traces."""
        cutoff_time = time.time() - (hours * 3600)
        
        return [
            trace for trace in self.traces.values()
            if trace.start_time > cutoff_time
        ]
    
    def get_trace_statistics(self, hours: int = 24) -> Dict[str, Any]:
        """Get trace statistics."""
        cutoff_time = time.time() - (hours * 3600)
        recent_traces = [
            trace for trace in self.traces.values()
            if trace.start_time > cutoff_time
        ]
        
        if not recent_traces:
            return {"error": "No traces available for specified period"}
        
        # Calculate statistics
        total_spans = sum(len(trace.spans) for trace in recent_traces)
        completed_traces = [trace for trace in recent_traces if trace.end_time]
        
        durations = []
        for trace in completed_traces:
            if trace.end_time:
                duration = trace.end_time - trace.start_time
                durations.append(duration)
        
        return {
            "total_traces": len(recent_traces),
            "completed_traces": len(completed_traces),
            "total_spans": total_spans,
            "avg_spans_per_trace": total_spans / len(recent_traces) if recent_traces else 0,
            "avg_trace_duration": sum(durations) / len(durations) if durations else 0,
            "min_trace_duration": min(durations) if durations else 0,
            "max_trace_duration": max(durations) if durations else 0,
            "period_hours": hours
        }
    
    def is_running(self) -> bool:
        """Check if collector is running."""
        return self._running


class TraceContext:
    """Context manager for tracing operations."""
    
    def __init__(self, collector: TraceCollector, name: str, trace_id: str, parent_span_id: Optional[str] = None, **kwargs):
        self.collector = collector
        self.name = name
        self.trace_id = trace_id
        self.parent_span_id = parent_span_id
        self.kwargs = kwargs
        self.span = None
    
    async def __aenter__(self):
        """Enter the trace context."""
        self.span = self.collector.create_span(
            name=self.name,
            trace_id=self.trace_id,
            parent_span_id=self.parent_span_id,
            **self.kwargs
        )
        return self.span
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Exit the trace context."""
        if self.span:
            tags = {}
            if exc_type:
                tags["error"] = True
                tags["error.type"] = exc_type.__name__
                tags["error.message"] = str(exc_val)
            
            self.collector.end_span(self.span.span_id, tags)


# Convenience functions for tracing
def create_trace_context(collector: TraceCollector, name: str, trace_id: str, **kwargs):
    """Create a trace context."""
    return TraceContext(collector, name, trace_id, **kwargs)


def trace_operation(collector: TraceCollector, name: str, trace_id: str, **kwargs):
    """Decorator for tracing operations."""
    def decorator(func):
        async def wrapper(*args, **func_kwargs):
            span = collector.create_span(name, trace_id, **kwargs)
            try:
                result = await func(*args, **func_kwargs)
                collector.end_span(span.span_id)
                return result
            except Exception as e:
                collector.end_span(span.span_id, {"error": True, "error.message": str(e)})
                raise
        return wrapper
    return decorator
