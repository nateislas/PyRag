"""Semantic document chunking using LLM for intelligent content segmentation."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..llm.client import LLMClient
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SemanticChunk:
    """A semantically meaningful chunk of documentation content."""
    content: str
    chunk_type: str  # function, class, example, explanation, parameter, overview
    semantic_boundaries: List[str]  # logical breakpoints
    context_window: str  # surrounding context
    importance_score: float
    relationships: List[str]  # links to related chunks
    api_path: Optional[str] = None
    function_signature: Optional[str] = None
    parameters: Optional[List[Dict[str, Any]]] = None
    return_type: Optional[str] = None
    examples: Optional[List[str]] = None


class SemanticChunker:
    """Intelligent document chunking that preserves semantic meaning."""
    
    def __init__(self, llm_client: LLMClient, max_chunk_size: int = 2000):
        """Initialize semantic chunker."""
        self.llm_client = llm_client
        self.max_chunk_size = max_chunk_size
        self.logger = get_logger(__name__)
    
    async def chunk_content(
        self, 
        content: str, 
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> List[SemanticChunk]:
        """Chunk content using LLM-guided semantic analysis."""
        self.logger.info(f"Semantic chunking content of length {len(content)}")
        
        # First, use LLM to identify semantic boundaries
        semantic_boundaries = await self._identify_semantic_boundaries(content)
        
        # Then create chunks based on these boundaries
        chunks = await self._create_semantic_chunks(
            content, 
            semantic_boundaries, 
            metadata, 
            hierarchy_path
        )
        
        self.logger.info(f"Created {len(chunks)} semantic chunks")
        return chunks
    
    async def _identify_semantic_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Use LLM to identify semantic boundaries in content."""
        prompt = f"""
        Analyze this documentation content and identify logical semantic boundaries.
        
        Content:
        {content}
        
        For each logical unit, identify:
        1. The type of content (function, class, example, explanation, parameter, overview)
        2. The start and end boundaries
        3. The main topic or API being described
        4. Any related concepts or dependencies
        
        Return a JSON array of objects with this structure:
        {{
            "boundaries": [
                {{
                    "type": "function|class|example|explanation|parameter|overview",
                    "start": "text that starts this section",
                    "end": "text that ends this section", 
                    "topic": "main topic or API name",
                    "importance": 0.0-1.0,
                    "related_concepts": ["concept1", "concept2"],
                    "api_path": "optional.api.path",
                    "function_signature": "optional function signature"
                }}
            ]
        }}
        
        Ensure boundaries preserve complete logical units (don't break functions, classes, or examples).
        Focus on identifying distinct sections like function definitions, code examples, and explanatory text.
        """
        
        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.llm_client.config.max_tokens,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            self.logger.info(f"LLM response: {content}")
            
            from .utils import parse_llm_json_response
            result = parse_llm_json_response(content)
            
            # Extract boundaries from response
            if isinstance(result, dict) and "boundaries" in result:
                boundaries = result["boundaries"]
            elif isinstance(result, list):
                boundaries = result
            elif isinstance(result, dict):
                # Try to find boundaries in the response
                for key, value in result.items():
                    if isinstance(value, list) and value and isinstance(value[0], dict):
                        boundaries = value
                        break
                else:
                    self.logger.warning(f"Unexpected LLM response format: {result}")
                    return self._fallback_boundaries(content)
            else:
                self.logger.warning(f"Unexpected LLM response format: {result}")
                return self._fallback_boundaries(content)
            
            return boundaries
            
        except Exception as e:
            self.logger.error(f"LLM boundary identification failed: {e}")
            return self._fallback_boundaries(content)
    
    def _fallback_boundaries(self, content: str) -> List[Dict[str, Any]]:
        """Fallback boundary identification using basic heuristics."""
        boundaries = []
        
        # Split by common documentation patterns
        sections = re.split(r'\n(?=#{1,6}\s|##\s|###\s|####\s)', content)
        
        for section in sections:
            if not section.strip():
                continue
                
            # Determine section type
            section_type = self._classify_section_type(section)
            
            # Extract topic from heading
            topic = self._extract_topic(section)
            
            boundaries.append({
                "type": section_type,
                "start": section[:100] + "..." if len(section) > 100 else section,
                "end": section[-100:] + "..." if len(section) > 100 else section,
                "topic": topic,
                "importance": 0.5,
                "related_concepts": [],
                "api_path": None,
                "function_signature": None
            })
        
        return boundaries
    
    def _classify_section_type(self, section: str) -> str:
        """Classify section type using basic heuristics."""
        section_lower = section.lower()
        
        # Look for code blocks
        if "```python" in section or "def " in section or "class " in section:
            if "def " in section:
                return "function"
            elif "class " in section:
                return "class"
            else:
                return "example"
        
        # Look for parameter documentation
        if "parameter" in section_lower or "param" in section_lower:
            return "parameter"
        
        # Look for overview content
        if any(word in section_lower for word in ["overview", "introduction", "getting started"]):
            return "overview"
        
        return "explanation"
    
    def _extract_topic(self, section: str) -> str:
        """Extract main topic from section."""
        # Look for headings
        heading_match = re.search(r'^#{1,6}\s+(.+)$', section, re.MULTILINE)
        if heading_match:
            return heading_match.group(1).strip()
        
        # Look for function/class definitions
        func_match = re.search(r'def\s+(\w+)', section)
        if func_match:
            return func_match.group(1)
        
        class_match = re.search(r'class\s+(\w+)', section)
        if class_match:
            return class_match.group(1)
        
        return "Unknown"
    
    async def _create_semantic_chunks(
        self,
        content: str,
        boundaries: List[Dict[str, Any]],
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> List[SemanticChunk]:
        """Create semantic chunks based on identified boundaries."""
        chunks = []
        
        for boundary in boundaries:
            # Extract the actual content for this boundary
            chunk_content = self._extract_boundary_content(content, boundary)
            
            if not chunk_content or len(chunk_content) < 50:
                continue
            
            # Create semantic chunk
            chunk = SemanticChunk(
                content=chunk_content,
                chunk_type=boundary.get("type", "explanation"),
                semantic_boundaries=[boundary.get("start", ""), boundary.get("end", "")],
                context_window=self._get_context_window(content, chunk_content),
                importance_score=boundary.get("importance", 0.5),
                relationships=boundary.get("related_concepts", []),
                api_path=boundary.get("api_path"),
                function_signature=boundary.get("function_signature"),
                parameters=await self._extract_parameters(chunk_content) if boundary.get("type") == "function" else None,
                return_type=await self._extract_return_type(chunk_content) if boundary.get("type") == "function" else None,
                examples=await self._extract_examples(chunk_content)
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def _extract_boundary_content(self, content: str, boundary: Dict[str, Any]) -> str:
        """Extract the actual content for a boundary."""
        start_text = boundary.get("start", "")
        end_text = boundary.get("end", "")
        
        # Find start position
        start_pos = content.find(start_text) if start_text else 0
        
        # Find end position
        end_pos = content.find(end_text, start_pos) if end_text else len(content)
        if end_pos == -1:
            end_pos = len(content)
        
        return content[start_pos:end_pos].strip()
    
    def _get_context_window(self, full_content: str, chunk_content: str) -> str:
        """Get surrounding context for a chunk."""
        # Find chunk position in full content
        chunk_start = full_content.find(chunk_content)
        if chunk_start == -1:
            return ""
        
        # Get context before and after
        context_before = full_content[max(0, chunk_start - 200):chunk_start]
        context_after = full_content[chunk_start + len(chunk_content):chunk_start + len(chunk_content) + 200]
        
        return f"{context_before}...{context_after}"
    
    async def _extract_parameters(self, content: str) -> Optional[List[Dict[str, Any]]]:
        """Extract function parameters using LLM."""
        if "def " not in content:
            return None
        
        prompt = f"""
        Extract function parameters from this content:
        
        {content}
        
        Return a JSON array of parameter objects with:
        {{
            "name": "parameter_name",
            "type": "parameter_type", 
            "default": "default_value",
            "description": "parameter_description",
            "required": true/false
        }}
        
        If no parameters found, return empty array.
        """
        
        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            import json
            result = json.loads(content)
            
            if isinstance(result, dict) and "parameters" in result:
                return result["parameters"]
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Parameter extraction failed: {e}")
            return []
    
    async def _extract_return_type(self, content: str) -> Optional[str]:
        """Extract function return type using LLM."""
        if "def " not in content:
            return None
        
        prompt = f"""
        Extract the return type from this function:
        
        {content}
        
        Return only the return type (e.g., "str", "List[int]", "None", "Optional[Dict]")
        If no return type is specified, return "Any".
        """
        
        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=50,
                temperature=0.1
            )
            
            return response.choices[0].message.content.strip()
            
        except Exception as e:
            self.logger.warning(f"Return type extraction failed: {e}")
            return "Any"
    
    async def _extract_examples(self, content: str) -> Optional[List[str]]:
        """Extract code examples from content."""
        # Basic regex extraction
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        if code_blocks:
            return [block.strip() for block in code_blocks]
        
        # Try LLM extraction for more complex cases
        prompt = f"""
        Extract code examples from this content:
        
        {content}
        
        Return a JSON array of code examples as strings.
        Include only complete, runnable code examples.
        """
        
        try:
            response = await self.llm_client.client.chat.completions.create(
                model=self.llm_client.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=1000,
                temperature=0.1,
                response_format={"type": "json_object"}
            )
            
            content = response.choices[0].message.content
            import json
            result = json.loads(content)
            
            if isinstance(result, dict) and "examples" in result:
                return result["examples"]
            elif isinstance(result, list):
                return result
            else:
                return []
                
        except Exception as e:
            self.logger.warning(f"Example extraction failed: {e}")
            return []
