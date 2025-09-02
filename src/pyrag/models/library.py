"""Library models for tracking Python libraries and versions."""

from datetime import datetime
from typing import List, Optional

from sqlalchemy import Boolean, Column, DateTime, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class Library(Base):
    """Model for tracking Python libraries."""

    __tablename__ = "libraries"

    # Basic library information
    name: Mapped[str] = mapped_column(
        String(255), unique=True, index=True, nullable=False
    )
    description: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    license: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # URLs
    repository_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    documentation_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    pypi_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)

    # Status and tracking
    is_active: Mapped[bool] = mapped_column(Boolean, default=True, nullable=False)
    last_checked: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)
    indexing_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )

    # Relationships
    versions: Mapped[List["LibraryVersion"]] = relationship(
        "LibraryVersion", back_populates="library", cascade="all, delete-orphan"
    )
    compliance_status: Mapped[Optional["ComplianceStatus"]] = relationship(
        "ComplianceStatus",
        back_populates="library",
        uselist=False,
        cascade="all, delete-orphan",
    )

    def __repr__(self) -> str:
        """String representation of the library."""
        return f"<Library(name='{self.name}', status='{self.indexing_status}')>"


class LibraryVersion(Base):
    """Model for tracking specific versions of libraries."""

    __tablename__ = "library_versions"

    # Foreign key to library
    library_id: Mapped[int] = mapped_column(ForeignKey("libraries.id"), nullable=False)

    # Version information
    version: Mapped[str] = mapped_column(String(50), nullable=False)
    release_date: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Indexing status
    indexing_status: Mapped[str] = mapped_column(
        String(50), default="pending", nullable=False
    )
    indexed_at: Mapped[Optional[datetime]] = mapped_column(DateTime, nullable=True)

    # Content tracking
    chunk_count: Mapped[int] = mapped_column(default=0, nullable=False)
    content_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)

    # Relationships
    library: Mapped["Library"] = relationship("Library", back_populates="versions")
    chunks: Mapped[List["DocumentChunk"]] = relationship(
        "DocumentChunk", back_populates="library_version", cascade="all, delete-orphan"
    )
    update_logs: Mapped[List["UpdateLog"]] = relationship(
        "UpdateLog", back_populates="library_version", cascade="all, delete-orphan"
    )

    def __repr__(self) -> str:
        """String representation of the library version."""
        return f"<LibraryVersion(library='{self.library.name}', version='{self.version}', status='{self.indexing_status}')>"


# Indexes for performance
Index("idx_libraries_name", Library.name)
Index("idx_libraries_status", Library.indexing_status)
Index(
    "idx_library_versions_library_version",
    LibraryVersion.library_id,
    LibraryVersion.version,
)
Index("idx_library_versions_status", LibraryVersion.indexing_status)
