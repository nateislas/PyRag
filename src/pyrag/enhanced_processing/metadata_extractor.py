"""Rich metadata extraction for enhanced search capabilities."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from ..logging import get_logger
from .semantic_chunker import SemanticChunk


# Import utility function locally to avoid circular import
def parse_llm_json_response(content: str):
    """Parse LLM JSON response with markdown code block handling."""
    import json
    import re

    # Remove markdown code blocks if present
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        # If all else fails, try to parse as a simple list
        try:
            # Look for array-like content
            array_match = re.search(r"\[.*\]", content, re.DOTALL)
            if array_match:
                return json.loads(array_match.group(0))
        except:
            pass

        raise e


logger = get_logger(__name__)


@dataclass
class ParameterInfo:
    """Information about a function parameter."""

    name: str
    type: Optional[str] = None
    default: Optional[str] = None
    description: Optional[str] = None
    required: bool = True


@dataclass
class EnhancedMetadata:
    """Enhanced metadata for intelligent search and ranking."""

    # Basic Info
    library: str
    version: str
    content_type: str

    # API Information
    api_path: Optional[str] = None  # e.g., "requests.get"
    function_signature: Optional[str] = None  # e.g., "get(url, params=None, **kwargs)"
    parameters: List[ParameterInfo] = None
    return_type: Optional[str] = None
    exceptions: List[str] = None

    # Content Analysis
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    usage_frequency: float = 0.5  # calculated from examples and references
    importance_score: float = 0.5  # calculated relevance

    # Relationships
    related_functions: List[str] = None
    parent_class: Optional[str] = None
    subclasses: List[str] = None
    dependencies: List[str] = None

    # Context
    examples: List[str] = None
    common_use_cases: List[str] = None
    version_compatibility: Dict[str, bool] = None

    def __post_init__(self):
        """Initialize default values for lists."""
        if self.parameters is None:
            self.parameters = []
        if self.exceptions is None:
            self.exceptions = []
        if self.related_functions is None:
            self.related_functions = []
        if self.subclasses is None:
            self.subclasses = []
        if self.dependencies is None:
            self.dependencies = []
        if self.examples is None:
            self.examples = []
        if self.common_use_cases is None:
            self.common_use_cases = []
        if self.version_compatibility is None:
            self.version_compatibility = {}


class MetadataExtractor:
    """Extract comprehensive metadata from semantic chunks."""

    def __init__(self, llm_client: LLMClient):
        """Initialize metadata extractor."""
        self.llm_client = llm_client
        self.logger = get_logger(__name__)

    async def extract_metadata(
        self, chunk: SemanticChunk, base_metadata: Dict[str, Any]
    ) -> EnhancedMetadata:
        """Extract comprehensive metadata from a semantic chunk."""
        self.logger.info(f"Extracting metadata for chunk type: {chunk.chunk_type}")

        # Start with base metadata
        enhanced_metadata = EnhancedMetadata(
            library=base_metadata.get("library", ""),
            version=base_metadata.get("version", ""),
            content_type=chunk.chunk_type,
        )

        # Extract API-specific metadata
        if chunk.chunk_type in ["function", "class"]:
            await self._extract_api_metadata(chunk, enhanced_metadata)

        # Extract content analysis
        await self._extract_content_analysis(chunk, enhanced_metadata)

        # Extract relationships
        await self._extract_relationships(chunk, enhanced_metadata)

        # Extract context information
        await self._extract_context_info(chunk, enhanced_metadata)

        return enhanced_metadata

    async def _extract_api_metadata(
        self, chunk: SemanticChunk, metadata: EnhancedMetadata
    ):
        """Extract API-specific metadata."""
        # Use existing chunk metadata
        metadata.api_path = chunk.api_path
        metadata.function_signature = chunk.function_signature
        metadata.return_type = chunk.return_type

        # Convert parameters to ParameterInfo objects
        if chunk.parameters:
            metadata.parameters = [
                ParameterInfo(
                    name=param.get("name", ""),
                    type=param.get("type"),
                    default=param.get("default"),
                    description=param.get("description"),
                    required=param.get("required", True),
                )
                for param in chunk.parameters
            ]

        # Extract exceptions using LLM
        metadata.exceptions = await self._extract_exceptions(chunk.content)

    async def _extract_exceptions(self, content: str) -> List[str]:
        """Extract possible exceptions from content."""
        prompt = f"""
        Extract possible exceptions that this function/class might raise:

        {content}

        Return a JSON array of exception names (e.g., ["ValueError", "TypeError", "KeyError"]).
        Include only exceptions that are explicitly mentioned or commonly associated with this type of operation.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            if isinstance(result, dict) and "exceptions" in result:
                return result["exceptions"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Exception extraction failed: {e}")
            return []

    async def _extract_content_analysis(
        self, chunk: SemanticChunk, metadata: EnhancedMetadata
    ):
        """Extract content analysis metadata."""
        # Determine complexity level
        metadata.complexity_level = await self._determine_complexity(chunk.content)

        # Calculate usage frequency
        metadata.usage_frequency = self._calculate_usage_frequency(chunk)

        # Calculate importance score
        metadata.importance_score = self._calculate_importance_score(chunk)

    async def _determine_complexity(self, content: str) -> str:
        """Determine content complexity level."""
        prompt = f"""
        Analyze the complexity of this documentation content:

        {content[:500]}...

        Classify as one of:
        - "beginner": Basic concepts, simple examples, introductory content
        - "intermediate": Moderate complexity, some advanced concepts
        - "advanced": Complex topics, advanced patterns, expert-level content

        Consider factors like:
        - Technical depth
        - Prerequisites needed
        - Complexity of examples
        - Target audience

        Respond with only the classification.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )

            complexity = response.choices[0].message.content.strip().lower()

            # Validate response
            if complexity in ["beginner", "intermediate", "advanced"]:
                return complexity
            else:
                return "intermediate"  # Default

        except Exception as e:
            self.logger.warning(f"Complexity determination failed: {e}")
            return "intermediate"

    def _calculate_usage_frequency(self, chunk: SemanticChunk) -> float:
        """Calculate usage frequency based on examples and references."""
        frequency = 0.5  # Base frequency

        # Boost for examples
        if chunk.examples:
            frequency += min(len(chunk.examples) * 0.1, 0.3)

        # Boost for function/class definitions (more commonly used)
        if chunk.chunk_type in ["function", "class"]:
            frequency += 0.2

        # Boost for high importance
        frequency += chunk.importance_score * 0.2

        return min(frequency, 1.0)

    def _calculate_importance_score(self, chunk: SemanticChunk) -> float:
        """Calculate importance score based on various factors."""
        importance = chunk.importance_score  # Start with LLM-determined importance

        # Boost for API definitions
        if chunk.chunk_type in ["function", "class"]:
            importance += 0.2

        # Boost for examples
        if chunk.examples:
            importance += 0.1

        # Boost for parameters (detailed documentation)
        if chunk.parameters:
            importance += 0.1

        return min(importance, 1.0)

    async def _extract_relationships(
        self, chunk: SemanticChunk, metadata: EnhancedMetadata
    ):
        """Extract relationship metadata."""
        # Use existing relationships from chunk
        metadata.related_functions = chunk.relationships

        # Extract parent class information
        metadata.parent_class = await self._extract_parent_class(chunk.content)

        # Extract dependencies
        metadata.dependencies = await self._extract_dependencies(chunk.content)

    async def _extract_parent_class(self, content: str) -> Optional[str]:
        """Extract parent class information."""
        # Look for class inheritance patterns
        inheritance_match = re.search(r"class\s+(\w+)\s*\(([^)]+)\)", content)
        if inheritance_match:
            parent = inheritance_match.group(2).strip()
            # Clean up common patterns
            if parent != "object" and parent != "Exception":
                return parent

        return None

    async def _extract_dependencies(self, content: str) -> List[str]:
        """Extract dependencies from content."""
        prompt = f"""
        Extract Python library dependencies mentioned in this content:

        {content}

        Return a JSON array of dependency names (e.g., ["requests", "pandas", "numpy"]).
        Include only external libraries that this code depends on.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=200,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            if isinstance(result, dict) and "dependencies" in result:
                return result["dependencies"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Dependency extraction failed: {e}")
            return []

    async def _extract_context_info(
        self, chunk: SemanticChunk, metadata: EnhancedMetadata
    ):
        """Extract context information."""
        # Use existing examples
        metadata.examples = chunk.examples or []

        # Extract common use cases
        metadata.common_use_cases = await self._extract_use_cases(chunk.content)

        # Extract version compatibility
        metadata.version_compatibility = await self._extract_version_compatibility(
            chunk.content
        )

    async def _extract_use_cases(self, content: str) -> List[str]:
        """Extract common use cases from content."""
        prompt = f"""
        Extract common use cases or scenarios for this functionality:

        {content}

        Return a JSON array of use case descriptions (e.g., ["Making HTTP requests", "Handling authentication", "Processing JSON data"]).
        Focus on practical, real-world scenarios.
        """

        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=300,
                temperature=0.1,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            result = parse_llm_json_response(content)

            if isinstance(result, dict) and "use_cases" in result:
                return result["use_cases"]
            elif isinstance(result, list):
                return result
            else:
                return []

        except Exception as e:
            self.logger.warning(f"Use case extraction failed: {e}")
            return []

    async def _extract_version_compatibility(self, content: str) -> Dict[str, bool]:
        """Extract version compatibility information."""
        # Look for version mentions
        version_patterns = [
            r"Python\s+(\d+\.\d+)",
            r"version\s+(\d+\.\d+)",
            r"(\d+\.\d+)\+",
            r"Python\s+(\d+)\.(\d+)",
        ]

        compatibility = {}

        for pattern in version_patterns:
            matches = re.findall(pattern, content, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    version = f"{match[0]}.{match[1]}"
                else:
                    version = match
                compatibility[version] = True

        return compatibility
