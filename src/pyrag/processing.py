"""Document processing utilities for PyRAG."""

import re
from typing import Any, Dict, List, Optional
from dataclasses import dataclass

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentChunk:
    """A chunk of document content."""
    content: str
    metadata: Dict[str, Any]
    hierarchy_path: List[str]
    content_type: str
    api_path: Optional[str] = None
    source_url: Optional[str] = None
    deprecated: bool = False


class DocumentProcessor:
    """Process and chunk documents for vector storage."""
    
    def __init__(self, max_chunk_size: int = 1000, overlap: int = 200):
        """Initialize document processor."""
        self.max_chunk_size = max_chunk_size
        self.overlap = overlap
        self.logger = get_logger(__name__)
    
    def chunk_text(
        self,
        text: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str],
        content_type: str = "documents"
    ) -> List[DocumentChunk]:
        """Chunk text into smaller pieces for vector storage."""
        self.logger.info(f"Chunking text of length {len(text)}")
        
        chunks = []
        
        # Simple text chunking by sentences and size
        sentences = self._split_into_sentences(text)
        
        current_chunk = ""
        current_sentences = []
        
        for sentence in sentences:
            # Check if adding this sentence would exceed chunk size
            if len(current_chunk) + len(sentence) > self.max_chunk_size and current_chunk:
                # Create chunk
                chunk = DocumentChunk(
                    content=current_chunk.strip(),
                    metadata=metadata.copy(),
                    hierarchy_path=hierarchy_path.copy(),
                    content_type=content_type
                )
                chunks.append(chunk)
                
                # Start new chunk with overlap
                if self.overlap > 0 and current_sentences:
                    # Keep last few sentences for overlap
                    overlap_sentences = current_sentences[-2:] if len(current_sentences) >= 2 else current_sentences
                    current_chunk = " ".join(overlap_sentences) + " " + sentence
                    current_sentences = overlap_sentences + [sentence]
                else:
                    current_chunk = sentence
                    current_sentences = [sentence]
            else:
                current_chunk += " " + sentence if current_chunk else sentence
                current_sentences.append(sentence)
        
        # Add final chunk
        if current_chunk:
            chunk = DocumentChunk(
                content=current_chunk.strip(),
                metadata=metadata.copy(),
                hierarchy_path=hierarchy_path.copy(),
                content_type=content_type
            )
            chunks.append(chunk)
        
        self.logger.info(f"Created {len(chunks)} chunks")
        return chunks
    
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences."""
        # Simple sentence splitting - can be improved with NLP libraries
        sentences = re.split(r'[.!?]+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    def process_api_documentation(
        self,
        content: str,
        api_path: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> List[DocumentChunk]:
        """Process API documentation with special handling."""
        self.logger.info(f"Processing API documentation for {api_path}")
        
        # Add API path to metadata
        api_metadata = metadata.copy()
        api_metadata["api_path"] = api_path
        
        # Create chunks
        chunks = self.chunk_text(
            text=content,
            metadata=api_metadata,
            hierarchy_path=hierarchy_path,
            content_type="api_reference"
        )
        
        # Add API path to each chunk
        for chunk in chunks:
            chunk.api_path = api_path
        
        return chunks
    
    def process_examples(
        self,
        content: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> List[DocumentChunk]:
        """Process code examples."""
        self.logger.info("Processing code examples")
        
        # For examples, we might want to keep them as single chunks
        # to preserve code block integrity
        chunks = self.chunk_text(
            text=content,
            metadata=metadata,
            hierarchy_path=hierarchy_path,
            content_type="examples"
        )
        
        return chunks
    
    def process_overview(
        self,
        content: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> List[DocumentChunk]:
        """Process overview/introduction content."""
        self.logger.info("Processing overview content")
        
        chunks = self.chunk_text(
            text=content,
            metadata=metadata,
            hierarchy_path=hierarchy_path,
            content_type="overview"
        )
        
        return chunks
    
    def create_sample_documents(self, library_name: str, version: str) -> List[Dict[str, Any]]:
        """Create sample documents for testing."""
        self.logger.info(f"Creating sample documents for {library_name}")
        
        documents = []
        
        # Sample API reference
        api_docs = [
            {
                "content": f"{library_name} is a powerful library for making HTTP requests. The Session class provides a way to persist certain parameters across requests.",
                "metadata": {
                    "library": library_name,
                    "version": version,
                    "content_type": "api_reference",
                    "api_path": f"{library_name}.Session"
                },
                "hierarchy_path": [library_name, "Session"],
                "content_type": "api_reference"
            },
            {
                "content": f"The get method in {library_name} allows you to make GET requests. It returns a Response object containing the server's response.",
                "metadata": {
                    "library": library_name,
                    "version": version,
                    "content_type": "api_reference",
                    "api_path": f"{library_name}.Session.get"
                },
                "hierarchy_path": [library_name, "Session", "get"],
                "content_type": "api_reference"
            }
        ]
        
        # Sample examples
        example_docs = [
            {
                "content": f"import {library_name}\n\nsession = {library_name}.Session()\nresponse = session.get('https://api.example.com/data')\nprint(response.json())",
                "metadata": {
                    "library": library_name,
                    "version": version,
                    "content_type": "examples"
                },
                "hierarchy_path": [library_name, "examples"],
                "content_type": "examples"
            }
        ]
        
        # Sample overview
        overview_docs = [
            {
                "content": f"{library_name} is a popular Python library for making HTTP requests. It provides a simple and elegant API for sending HTTP requests and handling responses.",
                "metadata": {
                    "library": library_name,
                    "version": version,
                    "content_type": "overview"
                },
                "hierarchy_path": [library_name],
                "content_type": "overview"
            }
        ]
        
        documents.extend(api_docs)
        documents.extend(example_docs)
        documents.extend(overview_docs)
        
        return documents
