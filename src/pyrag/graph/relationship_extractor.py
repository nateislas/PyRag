"""Extract relationships between API entities from documentation."""

import re
from typing import List, Dict, Any, Set
from dataclasses import dataclass
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class Relationship:
    """Represents a relationship between API entities."""
    source: str
    target: str
    relationship_type: str
    confidence: float
    context: str
    properties: Dict[str, Any]


class RelationshipExtractor:
    """Extract relationships between API entities from documentation."""
    
    def __init__(self):
        """Initialize the relationship extractor."""
        self.logger = get_logger(__name__)
        
        # Common Python import patterns
        self.import_patterns = [
            r'from\s+(\w+(?:\.\w+)*)\s+import\s+(\w+(?:\s*,\s*\w+)*)',
            r'import\s+(\w+(?:\.\w+)*)',
            r'from\s+(\w+(?:\.\w+)*)\s+import\s+\*',
        ]
        
        # Function call patterns
        self.function_call_patterns = [
            r'(\w+(?:\.\w+)*)\.(\w+)\s*\(',
            r'(\w+)\s*=\s*(\w+(?:\.\w+)*)\.(\w+)\s*\(',
        ]
        
        # Inheritance patterns
        self.inheritance_patterns = [
            r'class\s+(\w+)\s*\(\s*(\w+(?:\.\w+)*)\s*\)',
            r'class\s+(\w+)\s*\(\s*(\w+)\s*\)',
        ]
        
        # Usage patterns
        self.usage_patterns = [
            r'(\w+(?:\.\w+)*)\s*\.\s*(\w+)',
            r'(\w+)\s*=\s*(\w+(?:\.\w+)*)',
        ]
    
    def extract_relationships(self, content: str, library: str = None) -> List[Relationship]:
        """Extract relationships from documentation content."""
        relationships = []
        
        try:
            # Extract import relationships
            relationships.extend(self._extract_imports(content, library))
            
            # Extract function call relationships
            relationships.extend(self._extract_function_calls(content, library))
            
            # Extract inheritance relationships
            relationships.extend(self._extract_inheritance(content, library))
            
            # Extract usage relationships
            relationships.extend(self._extract_usage_patterns(content, library))
            
            # Remove duplicates and sort by confidence
            unique_relationships = self._deduplicate_relationships(relationships)
            
            self.logger.info(f"Extracted {len(unique_relationships)} relationships from content")
            return unique_relationships
            
        except Exception as e:
            self.logger.error(f"Error extracting relationships: {e}")
            return []
    
    def _extract_imports(self, content: str, library: str = None) -> List[Relationship]:
        """Extract import relationships."""
        relationships = []
        
        for pattern in self.import_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    source_module = match.group(1)
                    imported_items = match.group(2).split(',')
                    
                    for item in imported_items:
                        item = item.strip()
                        if item and item != '*':
                            relationship = Relationship(
                                source=source_module,
                                target=item,
                                relationship_type="imports",
                                confidence=0.9,
                                context=match.group(0),
                                properties={
                                    "line": content[:match.start()].count('\n') + 1,
                                    "library": library
                                }
                            )
                            relationships.append(relationship)
        
        return relationships
    
    def _extract_function_calls(self, content: str, library: str = None) -> List[Relationship]:
        """Extract function call relationships."""
        relationships = []
        
        for pattern in self.function_call_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    caller = match.group(1)
                    function = match.group(2)
                    
                    relationship = Relationship(
                        source=caller,
                        target=function,
                        relationship_type="calls",
                        confidence=0.8,
                        context=match.group(0),
                        properties={
                            "line": content[:match.start()].count('\n') + 1,
                            "library": library
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_inheritance(self, content: str, library: str = None) -> List[Relationship]:
        """Extract inheritance relationships."""
        relationships = []
        
        for pattern in self.inheritance_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    child_class = match.group(1)
                    parent_class = match.group(2)
                    
                    relationship = Relationship(
                        source=child_class,
                        target=parent_class,
                        relationship_type="inherits_from",
                        confidence=0.95,
                        context=match.group(0),
                        properties={
                            "line": content[:match.start()].count('\n') + 1,
                            "library": library
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _extract_usage_patterns(self, content: str, library: str = None) -> List[Relationship]:
        """Extract usage pattern relationships."""
        relationships = []
        
        for pattern in self.usage_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    user = match.group(1)
                    used_item = match.group(2)
                    
                    relationship = Relationship(
                        source=user,
                        target=used_item,
                        relationship_type="uses",
                        confidence=0.7,
                        context=match.group(0),
                        properties={
                            "line": content[:match.start()].count('\n') + 1,
                            "library": library
                        }
                    )
                    relationships.append(relationship)
        
        return relationships
    
    def _deduplicate_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """Remove duplicate relationships and sort by confidence."""
        seen = set()
        unique_relationships = []
        
        for rel in relationships:
            # Create a unique key for each relationship
            key = (rel.source, rel.target, rel.relationship_type)
            
            if key not in seen:
                seen.add(key)
                unique_relationships.append(rel)
        
        # Sort by confidence (highest first)
        unique_relationships.sort(key=lambda x: x.confidence, reverse=True)
        
        return unique_relationships
    
    def extract_api_paths(self, content: str) -> List[str]:
        """Extract API paths from documentation content."""
        api_paths = set()
        
        # Pattern for API paths like "requests.Session.get"
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        
        matches = re.finditer(api_pattern, content)
        for match in matches:
            path = match.group(0)
            # Filter out simple words and common patterns
            if '.' in path and len(path.split('.')) >= 2:
                api_paths.add(path)
        
        return list(api_paths)
    
    def extract_dependencies(self, content: str) -> List[str]:
        """Extract library dependencies from content."""
        dependencies = set()
        
        # Look for common dependency patterns
        dep_patterns = [
            r'requires\s+(\w+(?:[><=!]+\d+\.\d+)?)',
            r'depends\s+on\s+(\w+(?:[><=!]+\d+\.\d+)?)',
            r'(\w+)\s*[><=!]+\s*\d+\.\d+',
        ]
        
        for pattern in dep_patterns:
            matches = re.finditer(pattern, content, re.IGNORECASE)
            for match in matches:
                dep = match.group(1)
                if dep and len(dep) > 1:
                    dependencies.add(dep)
        
        return list(dependencies)
