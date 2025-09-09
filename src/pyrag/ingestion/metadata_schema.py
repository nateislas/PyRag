"""
Metadata schema configuration for Chroma Cloud compatibility.

This module defines the optimized 16-field metadata schema that fits within
Chroma Cloud's field limit while maintaining all essential functionality.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


@dataclass
class MetadataField:
    """Represents a metadata field with its properties."""
    name: str
    description: str
    required: bool = True
    max_length: Optional[int] = None
    merge_strategy: Optional[str] = None  # For fields that merge multiple sources


# Chroma Cloud optimized 16-field schema
CHROMA_CLOUD_FIELDS: List[MetadataField] = [
    MetadataField("content_type", "Type of content (text, signature, etc.)", required=True),
    MetadataField("library_name", "Library name for filtering", required=True),
    MetadataField("version", "Library version", required=True),
    MetadataField("title", "Document title", required=True),
    MetadataField("source_url", "Original source URL", required=True),
    MetadataField("document_type", "Document type (tutorial, API, etc.)", required=True),
    MetadataField("main_topic", "Main topic of the document", required=True),
    MetadataField("difficulty_level", "Difficulty level (beginner, intermediate, advanced)", required=True),
    MetadataField("has_code_blocks", "Whether document contains code blocks", required=True),
    MetadataField("hierarchy_path", "Hierarchical path for structure", required=True),
    MetadataField("hierarchy_level", "Level in hierarchy (0=library, 1=module, etc.)", required=True),
    MetadataField("key_concepts", "Key concepts and API entities (merged)", required=True, max_length=1000),
    MetadataField("search_keywords", "Search keywords and related topics (merged)", required=True, max_length=1000),
    MetadataField("chunk_index", "Index of this chunk", required=True),
    MetadataField("total_chunks", "Total number of chunks", required=True),
    MetadataField("content_length", "Length of content in characters", required=True),
]

# Field mapping from old schema to new schema
FIELD_MAPPING: Dict[str, str] = {
    # Direct mappings (no change)
    "content_type": "content_type",
    "library_name": "library_name", 
    "version": "version",
    "title": "title",
    "source_url": "source_url",
    "document_type": "document_type",
    "main_topic": "main_topic",
    "difficulty_level": "difficulty_level",
    "has_code_blocks": "has_code_blocks",
    "hierarchy_path": "hierarchy_path",
    "hierarchy_level": "hierarchy_level",
    "chunk_index": "chunk_index",
    "total_chunks": "total_chunks",
    "content_length": "content_length",
    
    # Merged fields
    "key_concepts": "key_concepts",  # Will merge api_entities and api_references
    "search_keywords": "search_keywords",  # Will merge related_topics
}

# Fields to remove (not included in new schema)
REMOVED_FIELDS: List[str] = [
    "api_entities",           # Merged into key_concepts
    "api_references",         # Merged into key_concepts
    "chunk_focus",           # Redundant with document_type
    "code_examples",         # Too verbose for metadata
    "optimized_processing",  # Internal flag
    "original_chunk_length", # Redundant with content_length
    "prerequisites",         # Too verbose for metadata
    "related_topics",        # Merged into search_keywords
    "source",               # Always "firecrawl"
    "split_index",          # Internal processing info
    "split_total",          # Internal processing info
    "line_number",          # Processing-only field
    "embedding_id",         # Processing-only field
    "embedding_model",      # Processing-only field
]

# Fields that need merging strategies
MERGE_STRATEGIES: Dict[str, Dict[str, Any]] = {
    "key_concepts": {
        "source_fields": ["key_concepts", "api_entities", "api_references"],
        "strategy": "merge_lists_to_string",
        "separator": ", ",
        "max_length": 1000
    },
    "search_keywords": {
        "source_fields": ["search_keywords", "related_topics"],
        "strategy": "merge_lists_to_string", 
        "separator": ", ",
        "max_length": 1000
    }
}


def get_chroma_cloud_field_names() -> List[str]:
    """Get list of field names for Chroma Cloud schema."""
    return [field.name for field in CHROMA_CLOUD_FIELDS]


def get_required_fields() -> List[str]:
    """Get list of required field names."""
    return [field.name for field in CHROMA_CLOUD_FIELDS if field.required]


def validate_metadata_schema(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Validate and ensure metadata conforms to Chroma Cloud schema.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Validated metadata dictionary with exactly 16 fields
    """
    validated = {}
    
    # Process direct mappings
    for old_field, new_field in FIELD_MAPPING.items():
        if old_field in metadata and metadata[old_field] is not None:
            validated[new_field] = metadata[old_field]
    
    # Process merged fields
    for field_name, strategy in MERGE_STRATEGIES.items():
        merged_value = _merge_field_values(metadata, strategy)
        if merged_value:
            validated[field_name] = merged_value
    
    # Ensure all required fields are present
    for field in CHROMA_CLOUD_FIELDS:
        if field.required and field.name not in validated:
            validated[field.name] = _get_default_value(field.name)
    
    # Truncate fields that exceed max length
    for field in CHROMA_CLOUD_FIELDS:
        if field.max_length and field.name in validated:
            value = validated[field.name]
            if isinstance(value, str) and len(value) > field.max_length:
                validated[field.name] = value[:field.max_length-3] + "..."
    
    return validated


def _merge_field_values(metadata: Dict[str, Any], strategy: Dict[str, Any]) -> Optional[str]:
    """Merge multiple field values according to strategy."""
    source_fields = strategy["source_fields"]
    separator = strategy["separator"]
    max_length = strategy.get("max_length", 1000)
    
    values = []
    for field in source_fields:
        if field in metadata and metadata[field] is not None:
            value = metadata[field]
            if isinstance(value, list):
                values.extend([str(item) for item in value if item])
            elif isinstance(value, str) and value.strip():
                values.append(value.strip())
    
    if not values:
        return None
    
    # Remove duplicates while preserving order
    unique_values = []
    seen = set()
    for value in values:
        if value not in seen:
            unique_values.append(value)
            seen.add(value)
    
    result = separator.join(unique_values)
    
    # Truncate if too long
    if len(result) > max_length:
        result = result[:max_length-3] + "..."
    
    return result


def _get_default_value(field_name: str) -> Any:
    """Get default value for required field."""
    defaults = {
        "content_type": "text",
        "library_name": "unknown",
        "version": "latest",
        "title": "Untitled",
        "source_url": "",
        "document_type": "document",
        "main_topic": "General",
        "difficulty_level": "beginner",
        "has_code_blocks": False,
        "hierarchy_path": "",
        "hierarchy_level": 0,
        "key_concepts": "",
        "search_keywords": "",
        "chunk_index": 0,
        "total_chunks": 1,
        "content_length": 0,
    }
    return defaults.get(field_name, "")


def get_schema_info() -> Dict[str, Any]:
    """Get information about the metadata schema."""
    return {
        "total_fields": len(CHROMA_CLOUD_FIELDS),
        "required_fields": len(get_required_fields()),
        "field_names": get_chroma_cloud_field_names(),
        "removed_fields": REMOVED_FIELDS,
        "merge_strategies": list(MERGE_STRATEGIES.keys()),
        "chroma_cloud_compatible": True,
        "max_fields": 16
    }
