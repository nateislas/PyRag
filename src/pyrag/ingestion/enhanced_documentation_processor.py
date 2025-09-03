"""Optimized documentation processor using single LLM call for RAG-optimized metadata."""

import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from ..llm.client import LLMClient
from ..logging import get_logger
from .document_chunk import DocumentChunk
from .document_structure_analyzer import DocumentStructureAnalyzer, DocumentAnalysis

logger = get_logger(__name__)


@dataclass
class EnhancedProcessingResult:
    """Result of enhanced document processing."""

    chunks: List[DocumentChunk]  # Restored DocumentChunk structure
    metadata: Dict[str, Any]
    processing_stats: Dict[str, Any]
    enhanced_metadata: Dict[str, Any]  # Rich metadata from enhanced processing


class EnhancedDocumentationProcessor:
    """Optimized documentation processor using single LLM call for RAG-optimized metadata."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.logger = get_logger(__name__)
        self.llm_client = llm_client
        
        # Initialize the new document structure analyzer
        self.structure_analyzer = DocumentStructureAnalyzer(llm_client=llm_client)
        
        # Keep API patterns for basic extraction
        self.api_patterns = [
            r"`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))`",  # function calls
            r"`([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*)`",  # class/module references
            r"([a-zA-Z_][a-zA-Z0-9_]*\.[a-zA-Z_][a-zA-Z0-9_]*\([^)]*\))",  # function calls without backticks
        ]

    async def process_scraped_document(
        self, scraped_doc: "ScrapedDocument", library_name: str, version: str = "latest"
    ) -> EnhancedProcessingResult:
        """Process a scraped document using optimized single LLM call analysis."""
        try:
            self.logger.info(
                f"Processing document with optimized system: {scraped_doc.url}"
            )

            # Extract content (prefer markdown)
            content = (
                scraped_doc.markdown if scraped_doc.markdown else scraped_doc.content
            )

            # Extract basic API references using patterns (no LLM needed)
            api_references = self._extract_api_references(content)

            # Use the new document structure analyzer (single LLM call)
            document_analysis = await self.structure_analyzer.analyze_document(
                content=content,
                url=scraped_doc.url,
                title=scraped_doc.title
            )

            # Create smart chunks based on the analysis
            raw_chunks = self.structure_analyzer.create_smart_chunks(
                content=content,
                analysis=document_analysis
            )

            # Convert to DocumentChunk format
            document_chunks = self._convert_raw_chunks_to_document_chunks(
                raw_chunks,
                library_name,
                version,
                scraped_doc.url,
                scraped_doc.title,
                api_references,
                document_analysis
            )

            # Calculate processing stats
            stats = {
                "total_chunks": len(document_chunks),
                "content_length": len(content),
                "api_references": len(api_references),
                "document_type": document_analysis.document_type,
                "difficulty_level": document_analysis.difficulty_level,
                "key_concepts_count": len(document_analysis.key_concepts),
                "api_entities_count": len(document_analysis.api_entities),
                "has_code_blocks": document_analysis.has_code_blocks,
                "processing_method": "optimized_single_llm_call",
            }

            # Create enhanced metadata from the analysis
            enhanced_metadata = {
                "document_analysis": document_analysis.to_metadata(),
                "processing_stats": stats,
                "optimization_level": "high"
            }

            return EnhancedProcessingResult(
                chunks=document_chunks,
                metadata={
                    "source_url": scraped_doc.url,
                    "title": scraped_doc.title,
                    "api_references": api_references,
                    "links": scraped_doc.metadata.get("links", []),
                    "images": scraped_doc.metadata.get("images", []),
                    "optimized_processing": True,
                    "document_type": document_analysis.document_type,
                    "main_topic": document_analysis.main_topic,
                    "difficulty_level": document_analysis.difficulty_level,
                },
                processing_stats=stats,
                enhanced_metadata=enhanced_metadata,
            )

        except Exception as e:
            self.logger.error(
                f"Error in optimized processing of document {scraped_doc.url}: {e}"
            )
            # Processing failed - return empty result
            return EnhancedProcessingResult(
                chunks=[],
                metadata={},
                processing_stats={
                    "total_chunks": 0,
                    "content_length": 0,
                    "api_references": 0,
                    "error": str(e),
                    "processing_method": "failed",
                },
                enhanced_metadata={},
            )



    def _convert_raw_chunks_to_document_chunks(
        self,
        raw_chunks: List[Dict[str, Any]],
        library_name: str,
        version: str,
        url: str,
        title: str,
        api_references: List[str],
        document_analysis: DocumentAnalysis,
    ) -> List[DocumentChunk]:
        """Convert raw chunks to DocumentChunk format with optimized metadata."""
        document_chunks = []

        for i, raw_chunk in enumerate(raw_chunks):
            # Create DocumentChunk with optimized metadata
            chunk = DocumentChunk(
                content=raw_chunk["content"],
                content_type=raw_chunk.get("content_type", "text"),
                hierarchy_path=raw_chunk.get("hierarchy_path", ""),
                hierarchy_level=raw_chunk.get("hierarchy_level", 0),
                title=title,
                source_url=url,
                library_name=library_name,
                version=version,
                
                # Enhanced metadata from document analysis
                document_type=document_analysis.document_type,
                main_topic=document_analysis.main_topic,
                key_concepts=document_analysis.key_concepts,
                api_entities=document_analysis.api_entities,
                code_examples=document_analysis.code_examples,
                prerequisites=document_analysis.prerequisites,
                related_topics=document_analysis.related_topics,
                difficulty_level=document_analysis.difficulty_level,
                search_keywords=document_analysis.search_keywords,
                has_code_blocks=document_analysis.has_code_blocks,
                optimized_processing=True,
                
                # Additional metadata
                additional_metadata={
                    "chunk_index": i,
                    "total_chunks": len(raw_chunks),
                    "chunk_focus": raw_chunk.get("focus", "general"),
                    "api_references": api_references,
                    "content_length": len(raw_chunk["content"]),
                    "source": "firecrawl",
                }
            )

            document_chunks.append(chunk)

        return document_chunks



    def _extract_api_references(self, content: str) -> List[str]:
        """Extract API references from content."""
        api_refs = set()

        for pattern in self.api_patterns:
            matches = re.findall(pattern, content)
            api_refs.update(matches)

        return list(api_refs)



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
        """Get aggregate statistics from optimized processing results."""
        total_chunks = sum(len(r.chunks) for r in results)
        total_content_length = sum(
            r.processing_stats.get("content_length", 0) for r in results
        )
        total_api_refs = sum(
            r.processing_stats.get("api_references", 0) for r in results
        )

        document_types = {}
        difficulty_levels = {}
        for r in results:
            doc_type = r.processing_stats.get("document_type", "unknown")
            document_types[doc_type] = document_types.get(doc_type, 0) + 1

            difficulty = r.processing_stats.get("difficulty_level", "unknown")
            difficulty_levels[difficulty] = difficulty_levels.get(difficulty, 0) + 1

        return {
            "total_documents": len(results),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "total_api_references": total_api_refs,
            "document_type_distribution": document_types,
            "difficulty_level_distribution": difficulty_levels,
            "average_chunks_per_document": (
                total_chunks / len(results) if results else 0
            ),
            "optimized_processing_success_rate": 1.0 if results else 0,
            "processing_method": "single_llm_call_optimized",
        }
