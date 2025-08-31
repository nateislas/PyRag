"""Documentation processor for converting scraped content into PyRAG documents."""

import re
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..models.document import DocumentChunk
from ..processing import DocumentProcessor
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ProcessingResult:
    """Result of document processing."""
    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]


class DocumentationProcessor:
    """Process scraped documentation into PyRAG document chunks."""
    
    def __init__(self):
        self.logger = get_logger(__name__)
        self.document_processor = DocumentProcessor()
        self.api_patterns = [
            r'`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))`',  # function calls
            r'`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)`',  # class/module references
            r'([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))',  # function calls without backticks
        ]
    
    def process_scraped_document(
        self, 
        scraped_doc: 'ScrapedDocument', 
        library_name: str,
        version: str = "latest"
    ) -> ProcessingResult:
        """Process a scraped document into PyRAG document chunks."""
        try:
            self.logger.info(f"Processing document: {scraped_doc.url}")
            
            # Extract content (prefer markdown, fallback to text)
            content = scraped_doc.markdown if scraped_doc.markdown else scraped_doc.content
            
            # Extract API references
            api_references = self._extract_api_references(content)
            
            # Determine content type
            content_type = self._determine_content_type(scraped_doc.url, content)
            
            # Extract hierarchy path
            hierarchy_path = self._extract_hierarchy_path(scraped_doc.url, scraped_doc.title)
            
            # Chunk the content
            chunks = self._chunk_content(
                content, 
                library_name, 
                version, 
                content_type, 
                hierarchy_path,
                api_references
            )
            
            # Calculate processing stats
            stats = {
                "total_chunks": len(chunks),
                "content_length": len(content),
                "api_references": len(api_references),
                "content_type": content_type,
                "hierarchy_depth": len(hierarchy_path)
            }
            
            return ProcessingResult(
                chunks=chunks,
                metadata={
                    "source_url": scraped_doc.url,
                    "title": scraped_doc.title,
                    "api_references": api_references,
                    "links": scraped_doc.metadata.get("links", []),
                    "images": scraped_doc.metadata.get("images", []),
                },
                processing_stats=stats
            )
            
        except Exception as e:
            self.logger.error(f"Error processing document {scraped_doc.url}: {e}")
            # Return empty result
            return ProcessingResult(
                chunks=[],
                metadata={},
                processing_stats={"error": str(e)}
            )
    
    def _extract_api_references(self, content: str) -> List[str]:
        """Extract API references from content."""
        api_refs = set()
        
        for pattern in self.api_patterns:
            matches = re.findall(pattern, content)
            api_refs.update(matches)
        
        return list(api_refs)
    
    def _determine_content_type(self, url: str, content: str) -> str:
        """Determine the type of content based on URL and content."""
        url_lower = url.lower()
        content_lower = content.lower()
        
        # Check content for code blocks first (highest priority)
        if '```' in content or 'code' in content_lower:
            return 'examples'
        
        # Check for API reference pages
        if any(keyword in url_lower for keyword in ['api', 'reference']):
            return 'api_reference'
        
        # Check for example pages
        if any(keyword in url_lower for keyword in ['example', 'tutorial', 'guide']):
            return 'examples'
        
        # Check for overview pages
        if any(keyword in url_lower for keyword in ['index', 'home', 'overview']):
            return 'overview'
        
        # Default to overview
        return 'overview'
    
    def _extract_hierarchy_path(self, url: str, title: str) -> List[str]:
        """Extract hierarchy path from URL and title."""
        path_parts = []
        
        # Extract from URL path
        try:
            from urllib.parse import urlparse
            parsed = urlparse(url)
            path_segments = [seg for seg in parsed.path.split('/') if seg]
            
            # Filter out common documentation paths
            filtered_segments = []
            for seg in path_segments:
                if seg not in ['docs', 'documentation', 'api', 'reference', 'guide']:
                    filtered_segments.append(seg)
            
            path_parts.extend(filtered_segments)
        except Exception:
            pass
        
        # Add title if it's meaningful
        if title and title not in path_parts:
            # Clean title
            clean_title = re.sub(r'[^\w\s-]', '', title).strip()
            if clean_title and len(clean_title) > 2:
                path_parts.append(clean_title)
        
        return path_parts
    
    def _chunk_content(
        self, 
        content: str, 
        library_name: str, 
        version: str, 
        content_type: str, 
        hierarchy_path: List[str],
        api_references: List[str]
    ) -> List[DocumentChunk]:
        """Chunk content into DocumentChunk objects."""
        chunks = []
        
        # Create base metadata for chunking
        base_metadata = {
            "library": library_name,
            "version": version,
            "content_type": content_type,
            "hierarchy_path": hierarchy_path,
            "api_references": api_references,
            "source": "firecrawl"
        }
        
        # Use the document processor to chunk the content
        chunks_data = self.document_processor.chunk_text(
            content, 
            base_metadata, 
            hierarchy_path, 
            content_type
        )
        
        for i, chunk_data in enumerate(chunks_data):
            # Create metadata for this chunk
            chunk_metadata = {
                "library": library_name,
                "version": version,
                "content_type": content_type,
                "hierarchy_path": hierarchy_path,
                "chunk_index": i,
                "total_chunks": len(chunks_data),
                "api_references": [ref for ref in api_references if ref in chunk_data.content],
                "source": "firecrawl"
            }
            
            # Update the chunk metadata
            chunk_data.metadata.update(chunk_metadata)
            
            # Convert to our DocumentChunk format
            chunk = DocumentChunk(
                content=chunk_data.content,
                metadata=chunk_data.metadata
            )
            
            chunks.append(chunk)
        
        return chunks
    
    def process_multiple_documents(
        self, 
        scraped_docs: List['ScrapedDocument'], 
        library_name: str,
        version: str = "latest"
    ) -> List[ProcessingResult]:
        """Process multiple scraped documents."""
        results = []
        
        for doc in scraped_docs:
            result = self.process_scraped_document(doc, library_name, version)
            results.append(result)
        
        return results
    
    def get_processing_stats(self, results: List[ProcessingResult]) -> Dict[str, Any]:
        """Get aggregate statistics from processing results."""
        total_chunks = sum(len(r.chunks) for r in results)
        total_content_length = sum(r.processing_stats.get("content_length", 0) for r in results)
        total_api_refs = sum(r.processing_stats.get("api_references", 0) for r in results)
        
        content_types = {}
        for r in results:
            content_type = r.processing_stats.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1
        
        return {
            "total_documents": len(results),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "total_api_references": total_api_refs,
            "content_type_distribution": content_types,
            "average_chunks_per_document": total_chunks / len(results) if results else 0,
        }
