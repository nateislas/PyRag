"""Document chunk model for storing processed documentation content."""

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from .metadata_schema import validate_metadata_schema, get_chroma_cloud_field_names


@dataclass
class DocumentChunk:
    """Model for storing document chunks with hierarchical organization."""

    # Content information
    content: str
    content_type: str  # signature, description, example, overview
    
    # Hierarchical organization
    hierarchy_path: str  # e.g., "requests.auth.HTTPBasicAuth.__call__"
    hierarchy_level: int = field(default=0)  # 0=library, 1=module, 2=class, 3=method
    
    # Metadata
    title: Optional[str] = None
    source_url: Optional[str] = None
    line_number: Optional[int] = None
    
    # Vector storage
    embedding_id: Optional[str] = None  # ID in vector store
    embedding_model: Optional[str] = None
    
    # Additional metadata as JSON
    additional_metadata: Optional[Dict[str, Any]] = None  # Parameters, deprecation info, etc.
    
    # Library information (simplified from SQLAlchemy relationships)
    library_name: Optional[str] = None
    version: Optional[str] = None
    
    # Enhanced metadata for RAG optimization
    document_type: Optional[str] = None
    main_topic: Optional[str] = None
    key_concepts: Optional[list] = None
    api_entities: Optional[list] = None
    code_examples: Optional[str] = None
    prerequisites: Optional[str] = None
    related_topics: Optional[list] = None
    difficulty_level: Optional[str] = None
    search_keywords: Optional[list] = None
    has_code_blocks: Optional[bool] = None
    optimized_processing: Optional[bool] = None

    def __repr__(self) -> str:
        """String representation of the document chunk."""
        return f"<DocumentChunk(library='{self.library_name}', path='{self.hierarchy_path}', type='{self.content_type}')>"

    @property
    def metadata(self) -> Dict[str, Any]:
        """Get metadata as dictionary optimized for Chroma Cloud (16 fields max)."""
        # Build raw metadata from all available fields
        raw_metadata = {
            "content_type": self.content_type,
            "hierarchy_path": self.hierarchy_path,
            "hierarchy_level": self.hierarchy_level,
            "title": self.title,
            "source_url": self.source_url,
            "line_number": self.line_number,
            "embedding_id": self.embedding_id,
            "embedding_model": self.embedding_model,
            "library_name": self.library_name,
            "version": self.version,
            "document_type": self.document_type,
            "main_topic": self.main_topic,
            "key_concepts": self.key_concepts,
            "api_entities": self.api_entities,
            "code_examples": self.code_examples,
            "prerequisites": self.prerequisites,
            "related_topics": self.related_topics,
            "difficulty_level": self.difficulty_level,
            "search_keywords": self.search_keywords,
            "has_code_blocks": self.has_code_blocks,
            "optimized_processing": self.optimized_processing,
        }
        
        # Add additional metadata if present
        if self.additional_metadata:
            raw_metadata.update(self.additional_metadata)
        
        # Validate and optimize for Chroma Cloud schema (16 fields max)
        return validate_metadata_schema(raw_metadata)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "content": self.content,
            "content_type": self.content_type,
            "hierarchy_path": self.hierarchy_path,
            "hierarchy_level": self.hierarchy_level,
            "title": self.title,
            "source_url": self.source_url,
            "line_number": self.line_number,
            "embedding_id": self.embedding_id,
            "embedding_model": self.embedding_model,
            "additional_metadata": self.additional_metadata,
            "library_name": self.library_name,
            "version": self.version,
            "document_type": self.document_type,
            "main_topic": self.main_topic,
            "key_concepts": self.key_concepts,
            "api_entities": self.api_entities,
            "code_examples": self.code_examples,
            "prerequisites": self.prerequisites,
            "related_topics": self.related_topics,
            "difficulty_level": self.difficulty_level,
            "search_keywords": self.search_keywords,
            "has_code_blocks": self.has_code_blocks,
            "optimized_processing": self.optimized_processing,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DocumentChunk":
        """Create DocumentChunk from dictionary."""
        return cls(**data)
