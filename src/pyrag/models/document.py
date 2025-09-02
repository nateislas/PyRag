"""Document chunk model for storing processed documentation content."""

from typing import Optional

from sqlalchemy import JSON, Column, ForeignKey, Index, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from .base import Base


class DocumentChunk(Base):
    """Model for storing document chunks with hierarchical organization."""

    __tablename__ = "document_chunks"

    # Foreign key to library version
    library_version_id: Mapped[int] = mapped_column(
        ForeignKey("library_versions.id"), nullable=False
    )

    # Content information
    content: Mapped[str] = mapped_column(Text, nullable=False)
    content_type: Mapped[str] = mapped_column(
        String(50), nullable=False
    )  # signature, description, example, overview

    # Hierarchical organization
    hierarchy_path: Mapped[str] = mapped_column(
        String(500), nullable=False
    )  # e.g., "requests.auth.HTTPBasicAuth.__call__"
    hierarchy_level: Mapped[int] = mapped_column(
        default=0, nullable=False
    )  # 0=library, 1=module, 2=class, 3=method

    # Metadata
    title: Mapped[Optional[str]] = mapped_column(String(255), nullable=True)
    source_url: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    line_number: Mapped[Optional[int]] = mapped_column(nullable=True)

    # Vector storage
    embedding_id: Mapped[Optional[str]] = mapped_column(
        String(255), nullable=True
    )  # ID in vector store
    embedding_model: Mapped[Optional[str]] = mapped_column(String(100), nullable=True)

    # Additional metadata as JSON
    additional_metadata: Mapped[Optional[dict]] = mapped_column(
        JSON, nullable=True
    )  # Parameters, deprecation info, etc.

    # Relationships
    library_version: Mapped["LibraryVersion"] = relationship(
        "LibraryVersion", back_populates="chunks"
    )

    def __repr__(self) -> str:
        """String representation of the document chunk."""
        return f"<DocumentChunk(library='{self.library_version.library.name}', path='{self.hierarchy_path}', type='{self.content_type}')>"

    @property
    def library_name(self) -> str:
        """Get the library name from the relationship."""
        return self.library_version.library.name

    @property
    def version(self) -> str:
        """Get the version from the relationship."""
        return self.library_version.version


# Indexes for performance
Index("idx_document_chunks_library_version", DocumentChunk.library_version_id)
Index("idx_document_chunks_hierarchy_path", DocumentChunk.hierarchy_path)
Index("idx_document_chunks_content_type", DocumentChunk.content_type)
Index("idx_document_chunks_embedding_id", DocumentChunk.embedding_id)
Index("idx_document_chunks_hierarchy_level", DocumentChunk.hierarchy_level)
