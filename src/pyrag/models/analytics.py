"""Analytics models for tracking query metrics and performance data."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Column, String, Integer, Float, DateTime, Index, Text
from sqlalchemy.orm import Mapped, mapped_column

from .base import Base


class QueryMetric(Base):
    """Model for tracking query performance and user satisfaction."""
    
    __tablename__ = "query_metrics"
    
    # Query information
    query_hash: Mapped[str] = mapped_column(String(64), nullable=False)  # Hash of the query for deduplication
    query_text: Mapped[str] = mapped_column(Text, nullable=False)
    query_type: Mapped[str] = mapped_column(String(50), nullable=False)  # search, reference, deprecation_check, etc.
    
    # Performance metrics
    response_time_ms: Mapped[float] = mapped_column(Float, nullable=False)
    result_count: Mapped[int] = mapped_column(Integer, nullable=False)
    cache_hit: Mapped[bool] = mapped_column(default=False, nullable=False)
    
    # User satisfaction (optional)
    user_satisfaction: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)  # 1-5 scale
    user_feedback: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    
    # Context information
    library_filter: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    version_filter: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    content_type_filter: Mapped[Optional[str]] = mapped_column(String(50), nullable=True)
    
    # Timestamp
    query_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self) -> str:
        """String representation of the query metric."""
        return f"<QueryMetric(query_hash='{self.query_hash[:8]}...', response_time={self.response_time_ms}ms, results={self.result_count})>"


class PerformanceMetric(Base):
    """Model for tracking system performance metrics."""
    
    __tablename__ = "performance_metrics"
    
    # Metric information
    metric_type: Mapped[str] = mapped_column(String(100), nullable=False)  # response_time_p95, cache_hit_rate, etc.
    metric_name: Mapped[str] = mapped_column(String(100), nullable=False)  # Specific metric name
    metric_value: Mapped[float] = mapped_column(Float, nullable=False)
    metric_unit: Mapped[str] = mapped_column(String(20), nullable=False)  # ms, %, count, etc.
    
    # Context
    component: Mapped[str] = mapped_column(String(50), nullable=False)  # api, vector_store, database, etc.
    library_name: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    
    # Timing
    measurement_timestamp: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, nullable=False)
    measurement_window: Mapped[Optional[str]] = mapped_column(String(20), nullable=True)  # 1m, 5m, 1h, 1d
    
    # Additional context
    additional_metadata: Mapped[Optional[str]] = mapped_column(Text, nullable=True)  # JSON string with additional context
    
    def __repr__(self) -> str:
        """String representation of the performance metric."""
        return f"<PerformanceMetric(type='{self.metric_type}', value={self.metric_value}{self.metric_unit}, component='{self.component}')>"


# Indexes for performance
Index("idx_query_metrics_hash", QueryMetric.query_hash)
Index("idx_query_metrics_type", QueryMetric.query_type)
Index("idx_query_metrics_timestamp", QueryMetric.query_timestamp)
Index("idx_query_metrics_library", QueryMetric.library_filter)
Index("idx_query_metrics_response_time", QueryMetric.response_time_ms)

Index("idx_performance_metrics_type", PerformanceMetric.metric_type)
Index("idx_performance_metrics_component", PerformanceMetric.component)
Index("idx_performance_metrics_timestamp", PerformanceMetric.measurement_timestamp)
Index("idx_performance_metrics_library", PerformanceMetric.library_name)
