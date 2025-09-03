"""
Document Structure Analyzer for RAG-optimized metadata extraction.

This module replaces the complex enhanced processing pipeline with a single,
strategic LLM call per document that extracts standardized metadata designed
for developer and AI agent search and retrieval.
"""

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
import logging

from ..llm.client import LLMClient
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentAnalysis:
    """Standardized document analysis result for RAG optimization."""
    
    document_type: str
    main_topic: str
    key_concepts: List[str]
    api_entities: List[str]
    code_examples: str
    prerequisites: str
    related_topics: List[str]
    difficulty_level: str
    search_keywords: List[str]
    
    # Additional metadata for chunking
    content_length: int
    has_code_blocks: bool
    header_structure: List[str]
    
    def to_metadata(self) -> Dict[str, Any]:
        """Convert analysis to metadata format for storage."""
        return self.to_sanitized_metadata()
    
    def to_sanitized_metadata(self) -> Dict[str, Any]:
        """Convert analysis to sanitized metadata format compatible with ChromaDB.
        
        This method converts list metadata values to strings to ensure ChromaDB compatibility.
        """
        def sanitize_list(value):
            """Convert list to comma-separated string for ChromaDB compatibility."""
            if isinstance(value, list):
                if not value:
                    return ""
                # Convert list items to strings and join with commas
                string_items = [str(item) for item in value]
                result = ", ".join(string_items)
                # Truncate if too long (ChromaDB has limits)
                if len(result) > 1000:
                    result = result[:1000] + "..."
                return result
            return value
        
        return {
            "document_type": self.document_type,
            "main_topic": self.main_topic,
            "key_concepts": sanitize_list(self.key_concepts),
            "api_entities": sanitize_list(self.api_entities),
            "code_examples": self.code_examples,
            "prerequisites": self.prerequisites,
            "related_topics": sanitize_list(self.related_topics),
            "difficulty_level": self.difficulty_level,
            "search_keywords": sanitize_list(self.search_keywords),
            "content_length": self.content_length,
            "has_code_blocks": self.has_code_blocks,
            "header_structure": sanitize_list(self.header_structure),
        }


class DocumentStructureAnalyzer:
    """
    Analyzes document structure using a single LLM call to extract
    standardized metadata optimized for RAG systems.
    """
    
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.logger = get_logger(__name__)
        
        # Standardized document types
        self.valid_document_types = [
            "api_reference", "tutorial", "example", "overview", 
            "configuration", "troubleshooting", "installation", "changelog"
        ]
        
        # Standardized difficulty levels
        self.valid_difficulty_levels = ["beginner", "intermediate", "advanced"]
    
    async def analyze_document(
        self, 
        content: str, 
        url: str, 
        title: str = ""
    ) -> DocumentAnalysis:
        """
        Analyze a document with a single LLM call to extract standardized metadata.
        
        Args:
            content: Document content (markdown preferred)
            url: Document URL for context
            title: Document title if available
            
        Returns:
            DocumentAnalysis with standardized metadata
        """
        try:
            self.logger.info(f"Analyzing document structure: {url}")
            
            # Prepare content for analysis (limit to reasonable size)
            analysis_content = self._prepare_content_for_analysis(content)
            
            # Extract basic structural information without LLM
            structural_info = self._extract_structural_info(content, url, title)
            
            # Single LLM call for comprehensive analysis
            llm_analysis = await self._analyze_with_llm(analysis_content, url, title)
            
            # Combine LLM analysis with structural info
            combined_analysis = self._combine_analysis(llm_analysis, structural_info)
            
            # Validate and clean the analysis
            validated_analysis = self._validate_analysis(combined_analysis)
            
            self.logger.info(f"Document analysis complete: {validated_analysis.document_type}")
            return validated_analysis
            
        except Exception as e:
            self.logger.error(f"Error analyzing document {url}: {e}")
            # Return fallback analysis
            return self._create_fallback_analysis(content, url, title)
    
    def _prepare_content_for_analysis(self, content: str) -> str:
        """Prepare content for LLM analysis by limiting size and cleaning."""
        # Limit content to first 3000 characters for LLM efficiency
        # This is usually enough to understand the document structure
        if len(content) > 3000:
            content = content[:3000] + "\n\n[Content truncated for analysis...]"
        
        # Clean up excessive whitespace
        content = re.sub(r'\n{3,}', '\n\n', content)
        
        return content
    
    def _extract_structural_info(self, content: str, url: str, title: str) -> Dict[str, Any]:
        """Extract basic structural information without LLM calls."""
        # Extract markdown headers
        headers = re.findall(r'^#{1,6}\s+(.+)$', content, re.MULTILINE)
        
        # Check for code blocks
        has_code_blocks = bool(re.search(r'```[\w]*\n', content))
        
        # Determine content length
        content_length = len(content)
        
        # Extract basic content type from URL
        url_lower = url.lower()
        if any(pattern in url_lower for pattern in ["/api/", "/reference/"]):
            inferred_type = "api_reference"
        elif any(pattern in url_lower for pattern in ["/tutorial/", "/guide/", "/getting-started/"]):
            inferred_type = "tutorial"
        elif any(pattern in url_lower for pattern in ["/example/", "/examples/"]):
            inferred_type = "example"
        elif any(pattern in url_lower for pattern in ["/config/", "/configuration/"]):
            inferred_type = "configuration"
        elif any(pattern in url_lower for pattern in ["/troubleshooting/", "/faq/", "/help/"]):
            inferred_type = "troubleshooting"
        else:
            inferred_type = "overview"
        
        return {
            "inferred_type": inferred_type,
            "headers": headers,
            "has_code_blocks": has_code_blocks,
            "content_length": content_length,
            "url_patterns": self._extract_url_patterns(url)
        }
    
    def _extract_url_patterns(self, url: str) -> List[str]:
        """Extract meaningful patterns from URL for analysis."""
        patterns = []
        url_lower = url.lower()
        
        # Extract path segments
        path_parts = [part for part in url.split('/') if part and part not in ['http:', 'https:', '']]
        patterns.extend(path_parts)
        
        # Extract common patterns
        if 'api' in url_lower:
            patterns.append('api')
        if 'v' in url_lower and re.search(r'v\d+', url_lower):
            patterns.append('versioned')
        if 'docs' in url_lower:
            patterns.append('documentation')
        
        return patterns
    
    async def _analyze_with_llm(self, content: str, url: str, title: str) -> Dict[str, Any]:
        """Single LLM call for comprehensive document analysis."""
        prompt = f"""You are helping build a RAG (Retrieval-Augmented Generation) system for developers and AI agents to easily find and understand documentation.

Analyze this documentation page and extract the following standardized information that will help developers/AI agents quickly find relevant content:

URL: {url}
Title: {title}

Content Preview:
{content}

Extract and return ONLY a JSON object with these exact fields:

{{
  "DOCUMENT_TYPE": "[api_reference|tutorial|example|overview|configuration|troubleshooting|installation|changelog]",
  "MAIN_TOPIC": "[Brief, specific description of what this page covers - what would a developer search for?]",
  "KEY_CONCEPTS": ["[List of 3-7 main technical concepts, terms, or technologies discussed - these are what developers would search for]"],
  "API_ENTITIES": ["[List of specific classes, functions, methods, endpoints, or modules mentioned - exact names developers would look up]"],
  "CODE_EXAMPLES": "[Brief description of any code examples, including language and purpose]",
  "PREREQUISITES": "[What knowledge or setup the developer needs before reading this]",
  "RELATED_TOPICS": ["[2-4 related technical topics or pages this connects to]"],
  "DIFFICULTY_LEVEL": "[beginner|intermediate|advanced]",
  "SEARCH_KEYWORDS": ["[5-10 specific technical terms developers would search for to find this content]"]
}}

Focus on technical accuracy and searchability for developers and AI agents. Return ONLY the JSON, no other text."""

        try:
            response = await self.llm_client.generate(prompt)
            
            # Try to extract JSON from response
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
                analysis = json.loads(json_str)
                self.logger.info("LLM analysis completed successfully")
                return analysis
            else:
                raise ValueError("No JSON found in LLM response")
                
        except Exception as e:
            self.logger.error(f"LLM analysis failed: {e}")
            # Return empty analysis for fallback
            return {}
    
    def _combine_analysis(
        self, 
        llm_analysis: Dict[str, Any], 
        structural_info: Dict[str, Any]
    ) -> DocumentAnalysis:
        """Combine LLM analysis with structural information."""
        # Use LLM analysis as primary, fall back to structural info
        document_type = llm_analysis.get("DOCUMENT_TYPE") or structural_info["inferred_type"]
        main_topic = llm_analysis.get("MAIN_TOPIC") or f"Documentation page about {structural_info['inferred_type']}"
        
        return DocumentAnalysis(
            document_type=document_type,
            main_topic=main_topic,
            key_concepts=llm_analysis.get("KEY_CONCEPTS", []),
            api_entities=llm_analysis.get("API_ENTITIES", []),
            code_examples=llm_analysis.get("CODE_EXAMPLES", "No code examples detected"),
            prerequisites=llm_analysis.get("PREREQUISITES", "No prerequisites specified"),
            related_topics=llm_analysis.get("RELATED_TOPICS", []),
            difficulty_level=llm_analysis.get("DIFFICULTY_LEVEL", "intermediate"),
            search_keywords=llm_analysis.get("SEARCH_KEYWORDS", []),
            content_length=structural_info["content_length"],
            has_code_blocks=structural_info["has_code_blocks"],
            header_structure=structural_info["headers"]
        )
    
    def _validate_analysis(self, analysis: DocumentAnalysis) -> DocumentAnalysis:
        """Validate and clean the analysis results."""
        # Validate document type
        if analysis.document_type not in self.valid_document_types:
            analysis.document_type = "overview"
            self.logger.warning(f"Invalid document type, defaulting to 'overview'")
        
        # Validate difficulty level
        if analysis.difficulty_level not in self.valid_difficulty_levels:
            analysis.difficulty_level = "intermediate"
            self.logger.warning(f"Invalid difficulty level, defaulting to 'intermediate'")
        
        # Ensure lists are not empty
        if not analysis.key_concepts:
            analysis.key_concepts = ["documentation", "reference"]
        if not analysis.api_entities:
            analysis.api_entities = []
        if not analysis.related_topics:
            analysis.related_topics = ["general documentation"]
        if not analysis.search_keywords:
            analysis.search_keywords = [analysis.main_topic.lower()]
        
        return analysis
    
    def _create_fallback_analysis(self, content: str, url: str, title: str) -> DocumentAnalysis:
        """Create fallback analysis when LLM analysis fails."""
        structural_info = self._extract_structural_info(content, url, title)
        
        return DocumentAnalysis(
            document_type=structural_info["inferred_type"],
            main_topic=f"Documentation page: {title or url}",
            key_concepts=["documentation", "reference"],
            api_entities=[],
            code_examples="No code examples detected",
            prerequisites="No prerequisites specified",
            related_topics=["general documentation"],
            difficulty_level="intermediate",
            search_keywords=[title.lower() if title else "documentation"],
            content_length=structural_info["content_length"],
            has_code_blocks=structural_info["has_code_blocks"],
            header_structure=structural_info["headers"]
        )
    
    def create_smart_chunks(
        self, 
        content: str, 
        analysis: DocumentAnalysis,
        max_chunk_size: int = 1000
    ) -> List[Dict[str, Any]]:
        """
        Create smart chunks based on document analysis.
        
        Args:
            content: Document content
            analysis: Document analysis result
            max_chunk_size: Maximum chunk size in characters
            
        Returns:
            List of chunks with standardized metadata
        """
        chunks = []
        
        # Use markdown headers for natural chunking
        if analysis.header_structure:
            chunks = self._chunk_by_headers(content, analysis)
        else:
            # Fallback to size-based chunking
            chunks = self._chunk_by_size(content, analysis, max_chunk_size)
        
        # Add standardized metadata to each chunk
        for i, chunk in enumerate(chunks):
            chunk["metadata"] = {
                **analysis.to_metadata(),
                "chunk_index": i,
                "total_chunks": len(chunks),
                "chunk_content_type": chunk.get("content_type", "text"),
                "chunk_focus": chunk.get("focus", "general")
            }
        
        return chunks
    
    def _chunk_by_headers(self, content: str, analysis: DocumentAnalysis) -> List[Dict[str, Any]]:
        """Create chunks based on markdown header structure."""
        chunks = []
        
        # Split content by headers
        header_pattern = r'^(#{1,6})\s+(.+)$'
        sections = re.split(header_pattern, content, flags=re.MULTILINE)
        
        current_chunk = ""
        current_header = ""
        
        for i in range(0, len(sections), 3):
            if i + 2 < len(sections):
                header_level = sections[i]
                header_text = sections[i + 1]
                section_content = sections[i + 2] if i + 2 < len(sections) else ""
                
                # Start new chunk for main headers (## and above)
                if header_level.startswith("##"):
                    if current_chunk.strip():
                        chunks.append({
                            "content": current_chunk.strip(),
                            "content_type": "section",
                            "focus": current_header
                        })
                    
                    current_chunk = f"{header_level} {header_text}\n{section_content}"
                    current_header = header_text
                else:
                    current_chunk += f"{header_level} {header_text}\n{section_content}"
        
        # Add final chunk
        if current_chunk.strip():
            chunks.append({
                "content": current_chunk.strip(),
                "content_type": "section",
                "focus": current_header
            })
        
        return chunks
    
    def _chunk_by_size(self, content: str, analysis: DocumentAnalysis, max_size: int) -> List[Dict[str, Any]]:
        """Create chunks based on size when header chunking isn't possible."""
        chunks = []
        
        # Simple size-based chunking with overlap
        overlap = 100
        start = 0
        
        while start < len(content):
            end = start + max_size
            
            # Try to break at sentence boundaries
            if end < len(content):
                # Look for sentence endings near the chunk boundary
                for i in range(end, max(start + max_size - 200, start), -1):
                    if content[i] in '.!?':
                        end = i + 1
                        break
            
            chunk_content = content[start:end].strip()
            if chunk_content:
                chunks.append({
                    "content": chunk_content,
                    "content_type": "text",
                    "focus": "general"
                })
            
            start = end - overlap
        
        return chunks
