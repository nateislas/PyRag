"""Compliance models for tracking legal compliance and update logs."""

from datetime import datetime
from typing import Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class ComplianceStatus(Base):
    """Model for tracking legal compliance status of libraries."""

    __tablename__ = "compliance_status"

    # Foreign key to library
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), nullable=False)

    # License information
    license_approved: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    license_review_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    license_reviewer: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Maintainer communication
    maintainer_contacted: Mapped[bool] = mapped_column(
        Boolean, default=False, nullable=False
    )
    maintainer_contact_date: Mapped[Optional[datetime]] = mapped_column(
        DateTime, nullable=True
    )
    maintainer_response: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Opt-out status
    opt_out_status: Mapped[str] = mapped_column(
        String(50), default="none", nullable=False
    )  # none, requested, granted, denied
    opt_out_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    opt_out_reason: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Notes and documentation
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Relationships
    library: Mapped["Library"] = relationship(
        "Library", back_populates="compliance_status"
    )

    def __repr__(self) -> str:
        """String representation of the compliance status."""
        return f"<ComplianceStatus(library='{self.library.name}', license_approved={self.license_approved}, opt_out='{self.opt_out_status}')>"


class UpdateLog(Base):
    """Model for tracking update operations and their status."""

    __tablename__ = "update_logs"

    # Foreign key to library version
    library_version_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("library_versions.id"), nullable=True
    )
    library_id: Mapped[Optional[int]] = mapped_column(
        ForeignKey("libraries.id"), nullable=True
    )

    # Update information
    update_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # initial, version_update, content_update, compliance_check
    status: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # pending, in_progress, completed, failed

    # Timing
    started_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Results
    chunks_processed: Mapped[int] = mapped_column(default=0, nullable=False)
    chunks_added: Mapped[int] = mapped_column(default=0, nullable=False)
    chunks_updated: Mapped[int] = mapped_column(default=0, nullable=False)
    chunks_deleted: Mapped[int] = mapped_column(default=0, nullable=False)

    # Error information
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error_details: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # Metadata
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    library_version: Mapped[Optional["LibraryVersion"]] = relationship(
        "LibraryVersion", back_populates="update_logs"
    )

    def __repr__(self) -> str:
        """String representation of the update log."""
        library_name = (
            self.library_version.library.name if self.library_version else "unknown"
        )
        return f"<UpdateLog(library='{library_name}', type='{self.update_type}', status='{self.status}')>"

    @property
    def duration_seconds(self) -> Optional[float]:
        """Calculate duration in seconds if completed."""
        if self.started_at and self.completed_at:
            return (self.completed_at - self.started_at).total_seconds()
        return None


# Indexes for performance
Index("idx_compliance_status_library", ComplianceStatus.library_id)
Index("idx_compliance_status_license", ComplianceStatus.license_approved)
Index("idx_compliance_status_opt_out", ComplianceStatus.opt_out_status)

Index("idx_update_logs_library_version", UpdateLog.library_version_id)
Index("idx_update_logs_library", UpdateLog.library_id)
Index("idx_update_logs_type", UpdateLog.update_type)
Index("idx_update_logs_status", UpdateLog.status)
Index("idx_update_logs_started_at", UpdateLog.started_at)
