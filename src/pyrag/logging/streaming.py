"""Streaming-specific logging utilities for PyRAG."""

import time
from typing import Any, Dict, Optional
from contextlib import contextmanager

from .setup import get_logger, CorrelationContext

logger = get_logger(__name__)


class StreamingLogger:
    """Enhanced logger for streaming operations."""
    
    def __init__(self, operation_name: str, correlation_id: Optional[str] = None):
        """Initialize streaming logger."""
        self.operation_name = operation_name
        self.correlation_id = correlation_id
        self.logger = get_logger(f"streaming.{operation_name}")
        self.start_time = time.time()
        self.step_times = {}
        self.step_count = 0
        
    def log_start(self, **context):
        """Log the start of a streaming operation."""
        self.logger.info(
            f"🚀 Starting {self.operation_name}",
            operation=self.operation_name,
            correlation_id=self.correlation_id,
            **context
        )
    
    def log_step(self, step_name: str, **context):
        """Log a streaming step with timing."""
        current_time = time.time()
        elapsed = current_time - self.start_time
        
        if self.step_count > 0:
            step_time = current_time - max(self.step_times.values())
        else:
            step_time = elapsed
            
        self.step_times[step_name] = current_time
        self.step_count += 1
        
        self.logger.info(
            f"📊 {step_name}",
            operation=self.operation_name,
            step=step_name,
            step_number=self.step_count,
            step_time=f"{step_time:.3f}s",
            total_elapsed=f"{elapsed:.3f}s",
            correlation_id=self.correlation_id,
            **context
        )
    
    def log_progress(self, current: int, total: int, item_name: str = "items", **context):
        """Log progress with percentage."""
        percentage = (current / total) * 100 if total > 0 else 0
        
        self.logger.info(
            f"📈 Progress: {current}/{total} {item_name} ({percentage:.1f}%)",
            operation=self.operation_name,
            current=current,
            total=total,
            percentage=percentage,
            correlation_id=self.correlation_id,
            **context
        )
    
    def log_result(self, result_count: int, **context):
        """Log streaming result."""
        elapsed = time.time() - self.start_time
        
        self.logger.info(
            f"✅ {self.operation_name} completed",
            operation=self.operation_name,
            result_count=result_count,
            total_time=f"{elapsed:.3f}s",
            throughput=f"{result_count/elapsed:.1f}/s" if elapsed > 0 else "N/A",
            correlation_id=self.correlation_id,
            **context
        )
    
    def log_error(self, error: Exception, **context):
        """Log streaming error."""
        elapsed = time.time() - self.start_time
        
        self.logger.error(
            f"❌ {self.operation_name} failed",
            operation=self.operation_name,
            error=str(error),
            error_type=type(error).__name__,
            elapsed=f"{elapsed:.3f}s",
            correlation_id=self.correlation_id,
            **context
        )


@contextmanager
def streaming_operation(operation_name: str, correlation_id: Optional[str] = None, **start_context):
    """Context manager for streaming operations with automatic logging."""
    stream_logger = StreamingLogger(operation_name, correlation_id)
    
    try:
        stream_logger.log_start(**start_context)
        yield stream_logger
    except Exception as e:
        stream_logger.log_error(e)
        raise


class MultiDimensionalSearchLogger:
    """Specialized logger for multi-dimensional search operations."""
    
    def __init__(self, query: str, correlation_id: Optional[str] = None):
        """Initialize multi-dimensional search logger."""
        self.query = query
        self.correlation_id = correlation_id
        self.logger = get_logger("multidim.search")
        self.start_time = time.time()
        self.dimension_times = {}
        self.dimension_results = {}
        
    def log_query_start(self, library: Optional[str] = None, max_results: int = 10):
        """Log the start of a multi-dimensional query."""
        self.logger.info(
            "🔍 Multi-dimensional search started",
            query=self.query[:100] + "..." if len(self.query) > 100 else self.query,
            library=library,
            max_results=max_results,
            correlation_id=self.correlation_id
        )
    
    def log_intent_analysis(self, intent: Dict[str, Any]):
        """Log query intent analysis results."""
        self.logger.info(
            "🧠 Query intent analyzed",
            response_depth=intent.get("response_depth"),
            is_multi_faceted=intent.get("is_multi_faceted"),
            production_focused=intent.get("production_focused"),
            workflow_query=intent.get("workflow_query"),
            complexity_level=intent.get("complexity_level"),
            correlation_id=self.correlation_id
        )
    
    def log_dimensions_identified(self, dimensions):
        """Log identified search dimensions."""
        self.logger.info(
            "📊 Search dimensions identified",
            dimension_count=len(dimensions),
            dimensions=[{
                "name": dim.name,
                "category": dim.category, 
                "importance": dim.importance
            } for dim in dimensions],
            correlation_id=self.correlation_id
        )
    
    def log_dimension_search_start(self, dimension_name: str, search_query: str):
        """Log the start of a dimension search."""
        self.dimension_times[dimension_name] = time.time()
        self.logger.info(
            f"🔎 Searching dimension: {dimension_name}",
            dimension=dimension_name,
            search_query=search_query,
            correlation_id=self.correlation_id
        )
    
    def log_dimension_search_complete(self, dimension_name: str, result_count: int, avg_score: float):
        """Log completion of a dimension search."""
        if dimension_name in self.dimension_times:
            search_time = time.time() - self.dimension_times[dimension_name]
        else:
            search_time = 0
            
        self.dimension_results[dimension_name] = {
            "result_count": result_count,
            "avg_score": avg_score,
            "search_time": search_time
        }
        
        self.logger.info(
            f"✅ Dimension search completed: {dimension_name}",
            dimension=dimension_name,
            result_count=result_count,
            avg_score=f"{avg_score:.3f}",
            search_time=f"{search_time:.3f}s",
            correlation_id=self.correlation_id
        )
    
    def log_synthesis_start(self, total_results: int):
        """Log start of result synthesis."""
        self.logger.info(
            "🔧 Starting result synthesis",
            total_raw_results=total_results,
            correlation_id=self.correlation_id
        )
    
    def log_coverage_analysis(self, coverage_score: float, missing_topics: list):
        """Log coverage analysis results."""
        self.logger.info(
            "📈 Coverage analysis completed",
            coverage_score=f"{coverage_score:.2f}",
            missing_topics=missing_topics,
            correlation_id=self.correlation_id
        )
    
    def log_search_complete(self, final_result_count: int, coverage_score: float):
        """Log completion of multi-dimensional search."""
        total_time = time.time() - self.start_time
        
        self.logger.info(
            "🎯 Multi-dimensional search completed",
            query=self.query[:50] + "..." if len(self.query) > 50 else self.query,
            final_result_count=final_result_count,
            coverage_score=f"{coverage_score:.2f}",
            total_time=f"{total_time:.3f}s",
            dimension_count=len(self.dimension_results),
            dimension_summary=self.dimension_results,
            correlation_id=self.correlation_id
        )


def log_mcp_request(tool_name: str, args: Dict[str, Any], correlation_id: Optional[str] = None):
    """Log MCP tool request."""
    mcp_logger = get_logger("mcp.request")
    mcp_logger.info(
        f"📞 MCP tool called: {tool_name}",
        tool=tool_name,
        args={k: v for k, v in args.items() if k != "ctx"},  # Exclude context object
        correlation_id=correlation_id
    )


def log_mcp_response(tool_name: str, result_type: str, size_info: Dict[str, Any], 
                     correlation_id: Optional[str] = None):
    """Log MCP tool response."""
    mcp_logger = get_logger("mcp.response")
    mcp_logger.info(
        f"📤 MCP response sent: {tool_name}",
        tool=tool_name,
        result_type=result_type,
        **size_info,
        correlation_id=correlation_id
    )


def log_mcp_streaming_start(tool_name: str, expected_steps: int, 
                           correlation_id: Optional[str] = None):
    """Log start of MCP streaming response."""
    mcp_logger = get_logger("mcp.streaming")
    mcp_logger.info(
        f"📡 Starting streaming response: {tool_name}",
        tool=tool_name,
        expected_steps=expected_steps,
        correlation_id=correlation_id
    )


def log_mcp_streaming_step(tool_name: str, step: int, step_name: str, 
                          data: Dict[str, Any], correlation_id: Optional[str] = None):
    """Log MCP streaming step."""
    mcp_logger = get_logger("mcp.streaming")
    mcp_logger.info(
        f"📊 Streaming step {step}: {step_name}",
        tool=tool_name,
        step=step,
        step_name=step_name,
        data_summary=data,
        correlation_id=correlation_id
    )