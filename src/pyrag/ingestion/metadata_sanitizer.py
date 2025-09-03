"""
Metadata sanitizer for ChromaDB compatibility.

This module provides utilities to sanitize metadata before storing in ChromaDB,
ensuring all metadata values are serializable and compatible with ChromaDB's
requirements.
"""

import json
from typing import Any, Dict, List, Union, Optional
from datetime import datetime
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MetadataSanitizer:
    """
    Sanitizes metadata to ensure ChromaDB compatibility.
    
    ChromaDB has specific requirements for metadata values:
    - Must be JSON serializable
    - Lists must contain only primitive types
    - No complex objects or custom classes
    - No circular references
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Track sanitization statistics
        self.sanitization_stats = {
            "total_metadata_fields": 0,
            "sanitized_fields": 0,
            "removed_fields": 0,
            "converted_lists": 0,
            "converted_objects": 0,
            "truncated_strings": 0,
        }
    
    def sanitize_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Sanitize metadata dictionary for ChromaDB storage.
        
        Args:
            metadata: Raw metadata dictionary
            
        Returns:
            Sanitized metadata dictionary safe for ChromaDB
        """
        if not metadata:
            return {}
        
        self.logger.debug(f"Sanitizing metadata with {len(metadata)} fields")
        self._reset_stats()
        
        sanitized = {}
        
        for key, value in metadata.items():
            try:
                sanitized_value = self._sanitize_value(value)
                if sanitized_value is not None:
                    sanitized[key] = sanitized_value
                    self.sanitization_stats["sanitized_fields"] += 1
                else:
                    self.logger.warning(f"Removed metadata field '{key}' with unsanitizable value: {type(value).__name__}")
                    self.sanitization_stats["removed_fields"] += 1
            except Exception as e:
                self.logger.error(f"Error sanitizing metadata field '{key}': {e}")
                self.sanitization_stats["removed_fields"] += 1
        
        self.sanitization_stats["total_metadata_fields"] = len(metadata)
        
        self.logger.info(f"Metadata sanitization complete: {self.sanitization_stats}")
        return sanitized
    
    def _sanitize_value(self, value: Any) -> Optional[Any]:
        """
        Sanitize a single metadata value.
        
        Args:
            value: Raw metadata value
            
        Returns:
            Sanitized value or None if unsanitizable
        """
        if value is None:
            return None
        
        # Handle primitive types (these are safe)
        if isinstance(value, (str, int, float, bool)):
            return self._sanitize_primitive(value)
        
        # Handle lists (need to sanitize each element)
        elif isinstance(value, list):
            return self._sanitize_list(value)
        
        # Handle dictionaries (need to sanitize recursively)
        elif isinstance(value, dict):
            return self._sanitize_dict(value)
        
        # Handle other types (convert to string or remove)
        else:
            return self._sanitize_object(value)
    
    def _sanitize_primitive(self, value: Union[str, int, float, bool]) -> Union[str, int, float, bool]:
        """Sanitize primitive types."""
        if isinstance(value, str):
            # Truncate very long strings (ChromaDB has limits)
            if len(value) > 1000:
                truncated = value[:1000] + "..."
                self.logger.warning(f"Truncated string metadata from {len(value)} to {len(truncated)} characters")
                self.sanitization_stats["truncated_strings"] += 1
                return truncated
            return value
        
        return value
    
    def _sanitize_list(self, value: List[Any]) -> str:
        """Sanitize list values by converting them to strings for ChromaDB compatibility.
        
        ChromaDB doesn't accept list metadata values, so we convert all lists to strings.
        """
        if not value:
            return ""
        
        try:
            # Convert list to comma-separated string for readability
            # This is better than JSON for simple lists
            if all(isinstance(item, (str, int, float, bool)) for item in value):
                # Simple list - convert to comma-separated string
                string_items = [str(item) for item in value]
                result = ", ".join(string_items)
                
                # Truncate if too long
                if len(result) > 1000:
                    result = result[:1000] + "..."
                    self.sanitization_stats["truncated_strings"] += 1
                
                self.logger.debug(f"Converted simple list to comma-separated string: {type(value).__name__}")
                self.sanitization_stats["converted_lists"] += 1
                return result
            
            else:
                # Complex list - convert to JSON string
                json_str = json.dumps(value, default=str, ensure_ascii=False)
                if len(json_str) > 1000:
                    json_str = json_str[:1000] + "..."
                    self.sanitization_stats["truncated_strings"] += 1
                
                self.logger.debug(f"Converted complex list to JSON string: {type(value).__name__}")
                self.sanitization_stats["converted_lists"] += 1
                return json_str
        
        except (TypeError, ValueError) as e:
            self.logger.warning(f"Failed to convert list to string, using str(): {e}")
            # Fallback to string representation
            str_repr = str(value)
            if len(str_repr) > 1000:
                str_repr = str_repr[:1000] + "..."
                self.sanitization_stats["truncated_strings"] += 1
            
            self.sanitization_stats["converted_lists"] += 1
            return str_repr
    
    def _sanitize_dict(self, value: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Sanitize dictionary values recursively."""
        if not value:
            return {}
        
        sanitized_dict = {}
        
        for key, val in value.items():
            # Ensure key is string
            if not isinstance(key, str):
                key = str(key)
            
            # Sanitize value
            sanitized_val = self._sanitize_value(val)
            if sanitized_val is not None:
                sanitized_dict[key] = sanitized_val
        
        return sanitized_dict if sanitized_dict else None
    
    def _sanitize_object(self, value: Any) -> Optional[str]:
        """Sanitize complex objects by converting to string representation."""
        try:
            # Try to get a meaningful string representation
            if hasattr(value, '__str__'):
                str_repr = str(value)
            elif hasattr(value, '__repr__'):
                str_repr = repr(value)
            else:
                str_repr = f"<{type(value).__name__} object>"
            
            # Truncate if too long
            if len(str_repr) > 1000:
                str_repr = str_repr[:1000] + "..."
                self.sanitization_stats["truncated_strings"] += 1
            
            self.logger.debug(f"Converted object to string: {type(value).__name__}")
            self.sanitization_stats["converted_objects"] += 1
            return str_repr
        
        except Exception as e:
            self.logger.warning(f"Failed to convert object to string: {e}")
            return None
    
    def _reset_stats(self):
        """Reset sanitization statistics."""
        self.sanitization_stats = {
            "total_metadata_fields": 0,
            "sanitized_fields": 0,
            "removed_fields": 0,
            "converted_lists": 0,
            "converted_objects": 0,
            "truncated_strings": 0,
        }
    
    def get_sanitization_report(self) -> Dict[str, Any]:
        """Get a report of sanitization operations."""
        if self.sanitization_stats["total_metadata_fields"] == 0:
            return {"message": "No metadata sanitized yet"}
        
        success_rate = (
            self.sanitization_stats["sanitized_fields"] / 
            self.sanitization_stats["total_metadata_fields"]
        ) * 100
        
        return {
            "total_fields": self.sanitization_stats["total_metadata_fields"],
            "successfully_sanitized": self.sanitization_stats["sanitized_fields"],
            "removed_fields": self.sanitization_stats["removed_fields"],
            "success_rate_percentage": round(success_rate, 2),
            "conversions": {
                "lists_to_strings": self.sanitization_stats["converted_lists"],
                "objects_to_strings": self.sanitization_stats["converted_objects"],
                "truncated_strings": self.sanitization_stats["truncated_strings"],
            }
        }
    
    def validate_metadata(self, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """
        Validate that metadata is ChromaDB compatible without modifying it.
        
        Args:
            metadata: Metadata dictionary to validate
            
        Returns:
            Validation report with any issues found
        """
        if not metadata:
            return {"valid": True, "issues": []}
        
        issues = []
        
        for key, value in metadata.items():
            try:
                # Try to serialize the value
                json.dumps(value, default=str)
            except (TypeError, ValueError) as e:
                issues.append({
                    "field": key,
                    "value_type": type(value).__name__,
                    "error": str(e),
                    "value_preview": str(value)[:100] + "..." if len(str(value)) > 100 else str(value)
                })
        
        return {
            "valid": len(issues) == 0,
            "total_fields": len(metadata),
            "issues": issues,
            "issue_count": len(issues)
        }


# Convenience function for quick metadata sanitization
def sanitize_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick function to sanitize metadata for ChromaDB.
    
    Args:
        metadata: Raw metadata dictionary
        
    Returns:
        Sanitized metadata dictionary
    """
    sanitizer = MetadataSanitizer()
    return sanitizer.sanitize_metadata(metadata)


# Convenience function for metadata validation
def validate_metadata(metadata: Dict[str, Any]) -> Dict[str, Any]:
    """
    Quick function to validate metadata for ChromaDB compatibility.
    
    Args:
        metadata: Metadata dictionary to validate
        
    Returns:
        Validation report
    """
    sanitizer = MetadataSanitizer()
    return sanitizer.validate_metadata(metadata)
