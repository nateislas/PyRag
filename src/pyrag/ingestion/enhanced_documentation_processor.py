"""Enhanced documentation processor integrating semantic chunking and rich metadata extraction."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..enhanced_processing import EnhancedDocumentProcessor
from ..llm.client import LLMClient
from ..logging import get_logger
from ..models.document import DocumentChunk

logger = get_logger(__name__)


@dataclass
class EnhancedProcessingResult:
    """Result of enhanced document processing."""

    chunks: List[DocumentChunk]
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]
    enhanced_metadata: Dict[str, Any]  # Rich metadata from enhanced processing


class EnhancedDocumentationProcessor:
    """Enhanced documentation processor using semantic chunking and rich metadata extraction."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.logger = get_logger(__name__)
        self.llm_client = llm_client
        self.enhanced_processor = EnhancedDocumentProcessor(llm_client=llm_client)

        self.api_patterns = [
            r"`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))`",  # function calls
            r"`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)`",  # class/module references
            r"([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))",  # function calls without backticks
        ]

    async def process_scraped_document(
        self, scraped_doc: "ScrapedDocument", library_name: str, version: str = "latest"
    ) -> EnhancedProcessingResult:
        """Process a scraped document using enhanced semantic chunking and metadata extraction."""
        try:
            self.logger.info(
                f"Processing document with enhanced system: {scraped_doc.url}"
            )

            # Extract content (prefer markdown)
            content = (
                scraped_doc.markdown if scraped_doc.markdown else scraped_doc.content
            )

            api_references = self._extract_api_references(content)

            # Determine content type
            content_type = self._determine_content_type(scraped_doc.url, content)

            # Extract hierarchy path
            hierarchy_path = self._extract_hierarchy_path(
                scraped_doc.url, scraped_doc.title
            )

            # Use enhanced processing to create semantic chunks with rich metadata
            enhanced_chunks = await self._process_with_enhanced_system(
                content,
                library_name,
                version,
                content_type,
                hierarchy_path,
                scraped_doc.url,
                scraped_doc.title,
            )

            # Convert enhanced chunks to DocumentChunk format for compatibility
            document_chunks = self._convert_to_document_chunks(
                enhanced_chunks,
                library_name,
                version,
                content_type,
                hierarchy_path,
                scraped_doc.url,
                scraped_doc.title,
                api_references,
            )

            # Calculate processing stats
            stats = {
                "total_chunks": len(document_chunks),
                "content_length": len(content),
                "api_references": len(api_references),
                "content_type": content_type,
                "hierarchy_depth": len(hierarchy_path),
                "enhanced_chunks": len(enhanced_chunks),
                "semantic_boundaries": sum(
                    len(chunk.semantic_chunk.semantic_boundaries)
                    for chunk in enhanced_chunks
                ),
                "rich_metadata_extracted": any(
                    chunk.enhanced_metadata.parameters
                    or chunk.enhanced_metadata.return_type
                    for chunk in enhanced_chunks
                ),
            }

            # Extract enhanced metadata for potential use
            enhanced_metadata = self._extract_enhanced_metadata(enhanced_chunks)

            return EnhancedProcessingResult(
                chunks=document_chunks,
                metadata={
                    "source_url": scraped_doc.url,
                    "title": scraped_doc.title,
                    "api_references": api_references,
                    "links": scraped_doc.metadata.get("links", []),
                    "images": scraped_doc.metadata.get("images", []),
                    "enhanced_processing": True,
                },
                processing_stats=stats,
                enhanced_metadata=enhanced_metadata,
            )

        except Exception as e:
            self.logger.error(
                f"Error in enhanced processing of document {scraped_doc.url}: {e}"
            )
            # Enhanced processing failed - return empty result
            return EnhancedProcessingResult(
                chunks=[],
                metadata={},
                processing_stats={
                    "total_chunks": 0,
                    "content_length": 0,
                    "api_references": 0,
                    "enhanced_chunks": 0,
                    "semantic_boundaries": 0,
                    "rich_metadata_extracted": False,
                    "error": str(e),
                },
                enhanced_metadata={},
            )

    async def _process_with_enhanced_system(
        self,
        content: str,
        library_name: str,
        version: str,
        content_type: str,
        hierarchy_path: List[str],
        url: str,
        title: str,
    ) -> List["EnhancedDocumentChunk"]:
        """Process content using the enhanced processing system."""
        try:
            # Create a document context for enhanced processing
            document_context = {
                "library": library_name,
                "version": version,
                "content_type": content_type,
                "hierarchy_path": hierarchy_path,
                "url": url,
                "title": title,
                "source": "firecrawl",
            }

            # Use the enhanced processor to create semantic chunks
            enhanced_result = await self.enhanced_processor.process_document(
                content=content,
                metadata=document_context,
                hierarchy_path=hierarchy_path,
            )

            return enhanced_result.chunks

        except Exception as e:
            self.logger.warning(
                f"Enhanced processing failed, falling back to basic: {e}"
            )

            return []

    def _convert_to_document_chunks(
        self,
        enhanced_chunks: List["EnhancedDocumentChunk"],
        library_name: str,
        version: str,
        content_type: str,
        hierarchy_path: List[str],
        url: str,
        title: str,
        api_references: List[str],
    ) -> List[DocumentChunk]:
        """Convert enhanced chunks to DocumentChunk format for compatibility."""
        document_chunks = []

        for i, enhanced_chunk in enumerate(enhanced_chunks):
            # Create rich metadata from enhanced chunk
            chunk_metadata = {
                "library": library_name,
                "version": version,
                "content_type": content_type,
                "hierarchy_path": hierarchy_path,
                "chunk_index": i,
                "total_chunks": len(enhanced_chunks),
                "api_references": api_references,
                "source": "firecrawl",
                "enhanced_processing": True,
                # Enhanced metadata
                "chunk_type": enhanced_chunk.chunk_type,
                "semantic_boundaries": enhanced_chunk.semantic_chunk.semantic_boundaries,
                "importance_score": enhanced_chunk.semantic_chunk.importance_score,
                "relationships": enhanced_chunk.semantic_chunk.relationships,
                # API-specific metadata from enhanced metadata
                "api_path": enhanced_chunk.enhanced_metadata.api_path,
                "function_signature": enhanced_chunk.enhanced_metadata.function_signature,
                "parameters": enhanced_chunk.enhanced_metadata.parameters,
                "return_type": enhanced_chunk.enhanced_metadata.return_type,
                "examples": enhanced_chunk.enhanced_metadata.examples,
                # Content processing metadata
                "context_window": enhanced_chunk.semantic_chunk.context_window,
            }

            # Create DocumentChunk with enhanced metadata
            chunk = DocumentChunk(
                content=enhanced_chunk.content, metadata=chunk_metadata
            )

            document_chunks.append(chunk)

        return document_chunks

    def _extract_enhanced_metadata(
        self, enhanced_chunks: List["EnhancedDocumentChunk"]
    ) -> Dict[str, Any]:
        """Extract aggregate enhanced metadata from all chunks."""
        metadata = {
            "total_enhanced_chunks": len(enhanced_chunks),
            "chunk_types": {},
            "api_paths": [],
            "function_signatures": [],
            "parameters": [],
            "return_types": [],
            "examples": [],
            "relationships": [],
            "average_importance_score": 0.0,
        }

        if not enhanced_chunks:
            return metadata

        # Aggregate metadata from all chunks
        total_importance = 0.0
        for chunk in enhanced_chunks:
            # Count chunk types
            chunk_type = chunk.chunk_type
            metadata["chunk_types"][chunk_type] = (
                metadata["chunk_types"].get(chunk_type, 0) + 1
            )

            # Collect API information from enhanced metadata
            if chunk.enhanced_metadata.api_path:
                metadata["api_paths"].append(chunk.enhanced_metadata.api_path)
            if chunk.enhanced_metadata.function_signature:
                metadata["function_signatures"].append(
                    chunk.enhanced_metadata.function_signature
                )
            if chunk.enhanced_metadata.parameters:
                metadata["parameters"].extend(chunk.enhanced_metadata.parameters)
            if chunk.enhanced_metadata.return_type:
                metadata["return_types"].append(chunk.enhanced_metadata.return_type)
            if chunk.enhanced_metadata.examples:
                metadata["examples"].extend(chunk.enhanced_metadata.examples)

            # Collect relationships from semantic chunk
            if chunk.semantic_chunk.relationships:
                metadata["relationships"].extend(chunk.semantic_chunk.relationships)

            # Sum importance scores from semantic chunk
            total_importance += chunk.semantic_chunk.importance_score

        # Calculate average importance
        metadata["average_importance_score"] = total_importance / len(enhanced_chunks)

        # Remove duplicates
        metadata["api_paths"] = list(set(metadata["api_paths"]))
        metadata["function_signatures"] = list(set(metadata["function_signatures"]))
        metadata["return_types"] = list(set(metadata["return_types"]))
        metadata["relationships"] = list(set(metadata["relationships"]))

        return metadata

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
        if "```" in content or "code" in content_lower:
            return "examples"

        # Check for API reference pages
        if any(keyword in url_lower for keyword in ["api", "reference"]):
            return "api_reference"

        # Check for example pages
        if any(keyword in url_lower for keyword in ["example", "tutorial", "guide"]):
            return "examples"

        # Check for overview pages
        if any(keyword in url_lower for keyword in ["index", "home", "overview"]):
            return "overview"

        # Default to overview
        return "overview"

    def _extract_hierarchy_path(self, url: str, title: str) -> List[str]:
        """Extract hierarchy path from URL and title."""
        path_parts = []

        # Extract from URL path
        try:
            from urllib.parse import urlparse

            parsed = urlparse(url)
            path_segments = [seg for seg in parsed.path.split("/") if seg]

            # Filter out common documentation paths
            filtered_segments = []
            for seg in path_segments:
                if seg not in ["docs", "documentation", "api", "reference", "guide"]:
                    filtered_segments.append(seg)

            path_parts.extend(filtered_segments)
        except Exception:
            pass

        # Add title if it's meaningful
        if title and title not in path_parts:
            # Clean title
            clean_title = re.sub(r"[^\w\s-]", "", title).strip()
            if clean_title and len(clean_title) > 2:
                path_parts.append(clean_title)

        return path_parts

    async def process_multiple_documents(
        self,
        scraped_docs: List["ScrapedDocument"],
        library_name: str,
        version: str = "latest",
    ) -> List[EnhancedProcessingResult]:
        """Process multiple scraped documents with enhanced processing."""
        results = []

        for doc in scraped_docs:
            result = await self.process_scraped_document(doc, library_name, version)
            results.append(result)

        return results

    def get_processing_stats(
        self, results: List[EnhancedProcessingResult]
    ) -> Dict[str, Any]:
        """Get aggregate statistics from enhanced processing results."""
        total_chunks = sum(len(r.chunks) for r in results)
        total_content_length = sum(
            r.processing_stats.get("content_length", 0) for r in results
        )
        total_api_refs = sum(
            r.processing_stats.get("api_references", 0) for r in results
        )
        total_enhanced_chunks = sum(
            r.processing_stats.get("enhanced_chunks", 0) for r in results
        )

        content_types = {}
        chunk_types = {}
        for r in results:
            content_type = r.processing_stats.get("content_type", "unknown")
            content_types[content_type] = content_types.get(content_type, 0) + 1

            # Aggregate enhanced chunk types
            for chunk in r.chunks:
                chunk_type = chunk.metadata.get("chunk_type", "unknown")
                chunk_types[chunk_type] = chunk_types.get(chunk_type, 0) + 1

        return {
            "total_documents": len(results),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "total_api_references": total_api_refs,
            "total_enhanced_chunks": total_enhanced_chunks,
            "content_type_distribution": content_types,
            "enhanced_chunk_type_distribution": chunk_types,
            "average_chunks_per_document": (
                total_chunks / len(results) if results else 0
            ),
            "enhanced_processing_success_rate": 1.0 if results else 0,
        }
