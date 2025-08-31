"""Enhanced document processor integrating all RAG improvement components."""

from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from .semantic_chunker import SemanticChunker, SemanticChunk
from .metadata_extractor import MetadataExtractor, EnhancedMetadata
from .content_processor import ContentTypeProcessor, CodeChunk, APIDocChunk, ExampleChunk, TutorialChunk
from ..llm.client import LLMClient
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class EnhancedDocumentChunk:
    """Enhanced document chunk with comprehensive metadata and processing."""
    # Basic content
    content: str
    chunk_type: str
    
    # Semantic information
    semantic_chunk: SemanticChunk
    enhanced_metadata: EnhancedMetadata
    
    # Content-specific processing
    code_chunk: Optional[CodeChunk] = None
    api_doc_chunk: Optional[APIDocChunk] = None
    example_chunk: Optional[ExampleChunk] = None
    tutorial_chunk: Optional[TutorialChunk] = None
    
    # Storage information
    hierarchy_path: List[str] = None
    source_url: Optional[str] = None
    library: str = ""
    version: str = ""
    
    def __post_init__(self):
        """Initialize default values."""
        if self.hierarchy_path is None:
            self.hierarchy_path = []


@dataclass
class EnhancedProcessingResult:
    """Result of enhanced document processing."""
    chunks: List[EnhancedDocumentChunk]
    processing_stats: Dict[str, Any]
    metadata: Dict[str, Any]


class EnhancedDocumentProcessor:
    """Enhanced document processor with semantic chunking and rich metadata extraction."""
    
    def __init__(self, llm_client: LLMClient):
        """Initialize enhanced document processor."""
        self.llm_client = llm_client
        self.semantic_chunker = SemanticChunker(llm_client)
        self.metadata_extractor = MetadataExtractor(llm_client)
        self.content_processor = ContentTypeProcessor(llm_client)
        self.logger = get_logger(__name__)
    
    async def process_document(
        self,
        content: str,
        metadata: Dict[str, Any],
        hierarchy_path: List[str]
    ) -> EnhancedProcessingResult:
        """Process a document with enhanced semantic chunking and metadata extraction."""
        self.logger.info(f"Enhanced processing document with {len(content)} characters")
        
        # Step 1: Semantic chunking
        semantic_chunks = await self.semantic_chunker.chunk_content(
            content, metadata, hierarchy_path
        )
        
        # Step 2: Process each semantic chunk
        enhanced_chunks = []
        processing_stats = {
            "total_semantic_chunks": len(semantic_chunks),
            "chunk_types": {},
            "content_processing": {},
            "metadata_extraction": {}
        }
        
        for i, semantic_chunk in enumerate(semantic_chunks):
            self.logger.info(f"Processing semantic chunk {i+1}/{len(semantic_chunks)}: {semantic_chunk.chunk_type}")
            
            # Extract enhanced metadata
            enhanced_metadata = await self.metadata_extractor.extract_metadata(
                semantic_chunk, metadata
            )
            
            # Process content based on type
            content_processing = await self._process_content_by_type(semantic_chunk)
            
            # Create enhanced document chunk
            enhanced_chunk = EnhancedDocumentChunk(
                content=semantic_chunk.content,
                chunk_type=semantic_chunk.chunk_type,
                semantic_chunk=semantic_chunk,
                enhanced_metadata=enhanced_metadata,
                hierarchy_path=hierarchy_path,
                source_url=metadata.get("source_url"),
                library=metadata.get("library", ""),
                version=metadata.get("version", ""),
                **content_processing
            )
            
            enhanced_chunks.append(enhanced_chunk)
            
            # Update processing stats
            processing_stats["chunk_types"][semantic_chunk.chunk_type] = processing_stats["chunk_types"].get(semantic_chunk.chunk_type, 0) + 1
        
        # Step 3: Calculate final statistics
        processing_stats["total_enhanced_chunks"] = len(enhanced_chunks)
        processing_stats["average_metadata_completeness"] = self._calculate_metadata_completeness(enhanced_chunks)
        processing_stats["content_type_distribution"] = self._calculate_content_type_distribution(enhanced_chunks)
        
        return EnhancedProcessingResult(
            chunks=enhanced_chunks,
            processing_stats=processing_stats,
            metadata=metadata
        )
    
    async def _process_content_by_type(self, semantic_chunk: SemanticChunk) -> Dict[str, Any]:
        """Process content based on its type."""
        content_processing = {}
        
        if semantic_chunk.chunk_type == "function":
            # Process as API documentation
            api_doc = await self.content_processor.process_api_documentation(semantic_chunk.content)
            content_processing["api_doc_chunk"] = api_doc
            
            # Also process any code examples
            if semantic_chunk.examples:
                example_chunks = []
                for example in semantic_chunk.examples:
                    example_chunk = await self.content_processor.process_example(example)
                    example_chunks.append(example_chunk)
                content_processing["example_chunks"] = example_chunks
        
        elif semantic_chunk.chunk_type == "class":
            # Process as API documentation
            api_doc = await self.content_processor.process_api_documentation(semantic_chunk.content)
            content_processing["api_doc_chunk"] = api_doc
        
        elif semantic_chunk.chunk_type == "example":
            # Process as code example
            example_chunk = await self.content_processor.process_example(semantic_chunk.content)
            content_processing["example_chunk"] = example_chunk
        
        elif semantic_chunk.chunk_type == "overview":
            # Check if it contains tutorial-like content
            if self._is_tutorial_content(semantic_chunk.content):
                tutorial_chunk = await self.content_processor.process_tutorial(semantic_chunk.content)
                content_processing["tutorial_chunk"] = tutorial_chunk
        
        # Process any code blocks found in the content
        code_blocks = self._extract_code_blocks(semantic_chunk.content)
        if code_blocks:
            code_chunks = []
            for code_block in code_blocks:
                code_chunk = await self.content_processor.process_code_block(code_block)
                code_chunks.append(code_chunk)
            content_processing["code_chunk"] = code_chunks[0] if code_chunks else None  # Use first code chunk
        
        return content_processing
    
    def _is_tutorial_content(self, content: str) -> bool:
        """Determine if content is tutorial-like."""
        tutorial_indicators = [
            "step", "tutorial", "guide", "how to", "walkthrough",
            "prerequisites", "requirements", "follow these steps"
        ]
        
        content_lower = content.lower()
        return any(indicator in content_lower for indicator in tutorial_indicators)
    
    def _extract_code_blocks(self, content: str) -> List[str]:
        """Extract code blocks from content."""
        import re
        
        # Look for markdown code blocks
        code_blocks = re.findall(r'```python\n(.*?)\n```', content, re.DOTALL)
        
        # Look for indented code blocks
        lines = content.split('\n')
        current_block = []
        code_blocks_extended = []
        
        for line in lines:
            if line.startswith('    ') or line.startswith('\t'):
                current_block.append(line)
            else:
                if current_block:
                    code_blocks_extended.append('\n'.join(current_block))
                    current_block = []
        
        if current_block:
            code_blocks_extended.append('\n'.join(current_block))
        
        # Combine and deduplicate
        all_blocks = code_blocks + code_blocks_extended
        return [block.strip() for block in all_blocks if len(block.strip()) > 10]
    
    def _calculate_metadata_completeness(self, chunks: List[EnhancedDocumentChunk]) -> float:
        """Calculate average metadata completeness across chunks."""
        if not chunks:
            return 0.0
        
        total_completeness = 0.0
        
        for chunk in chunks:
            metadata = chunk.enhanced_metadata
            completeness = 0.0
            
            # Check basic fields
            if metadata.library:
                completeness += 0.2
            if metadata.version:
                completeness += 0.2
            if metadata.content_type:
                completeness += 0.1
            
            # Check API fields
            if metadata.api_path:
                completeness += 0.1
            if metadata.function_signature:
                completeness += 0.1
            if metadata.parameters:
                completeness += 0.1
            if metadata.return_type:
                completeness += 0.1
            
            # Check analysis fields
            if metadata.complexity_level:
                completeness += 0.05
            if metadata.importance_score > 0:
                completeness += 0.05
            
            total_completeness += completeness
        
        return total_completeness / len(chunks)
    
    def _calculate_content_type_distribution(self, chunks: List[EnhancedDocumentChunk]) -> Dict[str, int]:
        """Calculate distribution of content types."""
        distribution = {}
        
        for chunk in chunks:
            chunk_type = chunk.chunk_type
            distribution[chunk_type] = distribution.get(chunk_type, 0) + 1
        
        return distribution
    
    async def process_multiple_documents(
        self,
        documents: List[Dict[str, Any]]
    ) -> List[EnhancedProcessingResult]:
        """Process multiple documents with enhanced processing."""
        self.logger.info(f"Enhanced processing {len(documents)} documents")
        
        results = []
        
        for i, doc in enumerate(documents):
            self.logger.info(f"Processing document {i+1}/{len(documents)}")
            
            try:
                result = await self.process_document(
                    content=doc["content"],
                    metadata=doc["metadata"],
                    hierarchy_path=doc.get("hierarchy_path", [])
                )
                results.append(result)
                
            except Exception as e:
                self.logger.error(f"Error processing document {i+1}: {e}")
                continue
        
        return results
    
    def to_vector_store_format(self, enhanced_chunk: EnhancedDocumentChunk) -> Dict[str, Any]:
        """Convert enhanced chunk to vector store format."""
        # Create comprehensive metadata for vector storage
        vector_metadata = {
            # Basic information
            "library": enhanced_chunk.library,
            "version": enhanced_chunk.version,
            "content_type": enhanced_chunk.chunk_type,
            "hierarchy_path": enhanced_chunk.hierarchy_path,
            "source_url": enhanced_chunk.source_url,
            
            # Enhanced metadata
            "api_path": enhanced_chunk.enhanced_metadata.api_path,
            "function_signature": enhanced_chunk.enhanced_metadata.function_signature,
            "return_type": enhanced_chunk.enhanced_metadata.return_type,
            "complexity_level": enhanced_chunk.enhanced_metadata.complexity_level,
            "importance_score": enhanced_chunk.enhanced_metadata.importance_score,
            "usage_frequency": enhanced_chunk.enhanced_metadata.usage_frequency,
            
            # Relationships
            "related_functions": enhanced_chunk.enhanced_metadata.related_functions,
            "dependencies": enhanced_chunk.enhanced_metadata.dependencies,
            "parent_class": enhanced_chunk.enhanced_metadata.parent_class,
            
            # Context
            "examples": enhanced_chunk.enhanced_metadata.examples,
            "common_use_cases": enhanced_chunk.enhanced_metadata.common_use_cases,
            "version_compatibility": enhanced_chunk.enhanced_metadata.version_compatibility,
            
            # Semantic information
            "semantic_boundaries": enhanced_chunk.semantic_chunk.semantic_boundaries,
            "context_window": enhanced_chunk.semantic_chunk.context_window,
            "relationships": enhanced_chunk.semantic_chunk.relationships,
            
            # Content-specific information
            "has_code_processing": enhanced_chunk.code_chunk is not None,
            "has_api_processing": enhanced_chunk.api_doc_chunk is not None,
            "has_example_processing": enhanced_chunk.example_chunk is not None,
            "has_tutorial_processing": enhanced_chunk.tutorial_chunk is not None,
        }
        
        # Add parameters if available
        if enhanced_chunk.enhanced_metadata.parameters:
            vector_metadata["parameters"] = [
                {
                    "name": param.name,
                    "type": param.type,
                    "default": param.default,
                    "required": param.required
                }
                for param in enhanced_chunk.enhanced_metadata.parameters
            ]
        
        # Add exceptions if available
        if enhanced_chunk.enhanced_metadata.exceptions:
            vector_metadata["exceptions"] = enhanced_chunk.enhanced_metadata.exceptions
        
        return {
            "content": enhanced_chunk.content,
            "metadata": vector_metadata
        }
    
    def get_search_metadata(self, enhanced_chunk: EnhancedDocumentChunk) -> Dict[str, Any]:
        """Get metadata optimized for search and filtering."""
        return {
            "library": enhanced_chunk.library,
            "version": enhanced_chunk.version,
            "content_type": enhanced_chunk.chunk_type,
            "api_path": enhanced_chunk.enhanced_metadata.api_path,
            "complexity_level": enhanced_chunk.enhanced_metadata.complexity_level,
            "importance_score": enhanced_chunk.enhanced_metadata.importance_score,
            "has_examples": bool(enhanced_chunk.enhanced_metadata.examples),
            "has_parameters": bool(enhanced_chunk.enhanced_metadata.parameters),
            "is_deprecated": enhanced_chunk.api_doc_chunk.deprecation_info.get("is_deprecated", False) if enhanced_chunk.api_doc_chunk else False
        }
