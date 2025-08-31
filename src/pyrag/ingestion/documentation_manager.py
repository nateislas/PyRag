"""Comprehensive documentation manager for two-phase ingestion."""

import asyncio
from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from pathlib import Path
import json
import time

from ..logging import get_logger
from ..llm.client import LLMClient
from .site_crawler import SiteCrawler, CrawlResult
from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .documentation_processor import DocumentationProcessor, ProcessingResult
from ..vector_store import VectorStore
from ..embeddings import EmbeddingService

logger = get_logger(__name__)

@dataclass
class DocumentationJob:
    """Configuration for a documentation ingestion job."""
    library_name: str
    version: str
    base_url: str
    output_dir: str
    max_crawl_depth: int = 3
    max_crawl_pages: int = 100
    max_content_pages: int = 50
    use_llm_filtering: bool = True
    exclude_patterns: Optional[List[str]] = None
    include_patterns: Optional[List[str]] = None

@dataclass
class DocumentationResult:
    """Result of a complete documentation ingestion job."""
    job: DocumentationJob
    crawl_result: CrawlResult
    extraction_stats: Dict[str, Any]
    processing_stats: Dict[str, Any]
    storage_stats: Dict[str, Any]
    errors: List[str]
    success: bool

class DocumentationManager:
    """Manages the complete two-phase documentation ingestion process."""
    
    def __init__(
        self,
        vector_store: VectorStore,
        embedding_service: EmbeddingService,
        llm_client: Optional[LLMClient] = None,
        firecrawl_api_key: Optional[str] = None,
        cache_dir: str = "./cache"
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.firecrawl_api_key = firecrawl_api_key
        self.cache_dir = Path(cache_dir)
        self.processor = DocumentationProcessor()
        self.logger = get_logger(__name__)
        
        # Create cache directory
        self.cache_dir.mkdir(exist_ok=True)
    
    async def ingest_documentation(self, job: DocumentationJob) -> DocumentationResult:
        """Complete documentation ingestion process."""
        
        self.logger.info(f"Starting documentation ingestion for {job.library_name}")
        
        # Initialize result tracking
        errors = []
        extraction_stats = {}
        processing_stats = {}
        storage_stats = {}
        
        try:
            # Phase 1: Link Discovery
            self.logger.info("Phase 1: Discovering documentation links...")
            crawl_result = await self._discover_links(job)
            
            if not crawl_result.relevant_urls:
                error_msg = "No relevant documentation links found"
                self.logger.error(error_msg)
                errors.append(error_msg)
                return DocumentationResult(
                    job=job,
                    crawl_result=crawl_result,
                    extraction_stats={},
                    processing_stats={},
                    storage_stats={},
                    errors=errors,
                    success=False
                )
            
            # Phase 2: Content Extraction
            self.logger.info("Phase 2: Extracting content from discovered links...")
            extraction_result, documents = await self._extract_content(job, crawl_result.relevant_urls)
            extraction_stats = extraction_result
            
            # Phase 3: Process and Store
            self.logger.info("Phase 3: Processing and storing content...")
            processing_result, storage_result = await self._process_and_store(
                job, documents
            )
            processing_stats = processing_result
            storage_stats = storage_result
            
            # Save metadata
            await self._save_metadata(job, crawl_result, extraction_stats, processing_stats)
            
            self.logger.info(f"Documentation ingestion completed successfully for {job.library_name}")
            
            return DocumentationResult(
                job=job,
                crawl_result=crawl_result,
                extraction_stats=extraction_stats,
                processing_stats=processing_stats,
                storage_stats=storage_stats,
                errors=errors,
                success=True
            )
            
        except Exception as e:
            error_msg = f"Error during documentation ingestion: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return DocumentationResult(
                job=job,
                crawl_result=CrawlResult(set(), set(), {}, []),
                extraction_stats={},
                processing_stats={},
                storage_stats={},
                errors=errors,
                success=False
            )
    
    async def _discover_links(self, job: DocumentationJob) -> CrawlResult:
        """Phase 1: Discover all relevant documentation links."""
        
        async with SiteCrawler(
            llm_client=self.llm_client if job.use_llm_filtering else None,
            max_depth=job.max_crawl_depth,
            max_pages=job.max_crawl_pages,
            delay=1.0
        ) as crawler:
            
            return await crawler.crawl_documentation_site(
                base_url=job.base_url,
                library_name=job.library_name,
                exclude_patterns=job.exclude_patterns,
                include_patterns=job.include_patterns
            )
    
    async def _extract_content(
        self, 
        job: DocumentationJob, 
        urls: Set[str]
    ) -> tuple[Dict[str, Any], List[ScrapedDocument]]:
        """Phase 2: Extract content from discovered URLs using Firecrawl."""
        
        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API key required for content extraction")
        
        self.logger.info(f"Extracting content from {len(urls)} URLs")
        
        documents = []
        failed_urls = []
        total_content_length = 0
        
        # Limit number of pages to extract
        urls_to_extract = list(urls)[:job.max_content_pages]
        
        async with FirecrawlClient(api_key=self.firecrawl_api_key) as client:
            for i, url in enumerate(urls_to_extract, 1):
                try:
                    self.logger.info(f"Extracting content ({i}/{len(urls_to_extract)}): {url}")
                    
                    doc = await client.scrape_url(url)
                    documents.append(doc)
                    total_content_length += len(doc.content)
                    
                    # Add delay to be respectful
                    await asyncio.sleep(1)
                    
                except Exception as e:
                    self.logger.warning(f"Failed to extract content from {url}: {e}")
                    failed_urls.append(url)
        
        return {
            "total_urls": len(urls),
            "extracted_urls": len(documents),
            "failed_urls": len(failed_urls),
            "failed_url_list": failed_urls,
            "total_content_length": total_content_length,
            "document_urls": [doc.url for doc in documents]  # Store URLs instead of objects
        }, documents
    
    async def _process_and_store(
        self, 
        job: DocumentationJob, 
        documents: List[ScrapedDocument]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Phase 3: Process documents and store in vector database."""
        
        self.logger.info(f"Processing {len(documents)} documents")
        
        processing_results = []
        total_chunks = 0
        total_content_length = 0
        content_type_distribution = {}
        
        # Process each document
        for doc in documents:
            try:
                result = self.processor.process_scraped_document(
                    scraped_doc=doc,
                    library_name=job.library_name,
                    version=job.version
                )
                processing_results.append(result)
                
                total_chunks += len(result.chunks)
                total_content_length += len(doc.content)
                
                # Track content type distribution
                for chunk in result.chunks:
                    content_type = chunk.metadata.get("content_type", "unknown")
                    content_type_distribution[content_type] = content_type_distribution.get(content_type, 0) + 1
                
            except Exception as e:
                self.logger.error(f"Error processing document {doc.url}: {e}")
        
        # Store in vector database
        stored_chunks = []
        for result in processing_results:
            for chunk in result.chunks:
                try:
                    embedding = await self.embedding_service.embed_text(chunk.content)
                    document_data = {
                        "content": chunk.content,
                        "metadata": chunk.metadata,
                        "embedding": embedding
                    }
                    chunk_ids = await self.vector_store.add_documents([document_data])
                    stored_chunks.extend(chunk_ids)
                except Exception as e:
                    self.logger.error(f"Error storing chunk: {e}")
        
        processing_stats = {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "content_type_distribution": content_type_distribution,
            "average_chunks_per_document": total_chunks / len(documents) if documents else 0
        }
        
        storage_stats = {
            "stored_chunks": len(stored_chunks),
            "storage_success_rate": len(stored_chunks) / total_chunks if total_chunks > 0 else 0
        }
        
        return processing_stats, storage_stats
    
    async def _save_metadata(
        self,
        job: DocumentationJob,
        crawl_result: CrawlResult,
        extraction_stats: Dict[str, Any],
        processing_stats: Dict[str, Any]
    ):
        """Save metadata about the ingestion job."""
        
        metadata = {
            "job": {
                "library_name": job.library_name,
                "version": job.version,
                "base_url": job.base_url,
                "timestamp": time.time(),
                "max_crawl_depth": job.max_crawl_depth,
                "max_crawl_pages": job.max_crawl_pages,
                "max_content_pages": job.max_content_pages
            },
            "crawl_stats": crawl_result.crawl_stats,
            "extraction_stats": extraction_stats,
            "processing_stats": processing_stats
        }
        
        # Save to cache directory
        cache_file = self.cache_dir / f"{job.library_name}_{job.version}_metadata.json"
        with open(cache_file, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        self.logger.info(f"Metadata saved to {cache_file}")
    
    async def get_ingestion_status(self, library_name: str, version: str) -> Optional[Dict[str, Any]]:
        """Get status of a previous ingestion job."""
        
        cache_file = self.cache_dir / f"{library_name}_{version}_metadata.json"
        
        if cache_file.exists():
            with open(cache_file, 'r') as f:
                return json.load(f)
        
        return None
    
    async def list_ingested_libraries(self) -> List[Dict[str, Any]]:
        """List all ingested libraries."""
        
        libraries = []
        
        for cache_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(cache_file, 'r') as f:
                    metadata = json.load(f)
                    libraries.append({
                        "library_name": metadata["job"]["library_name"],
                        "version": metadata["job"]["version"],
                        "timestamp": metadata["job"]["timestamp"],
                        "total_chunks": metadata["processing_stats"]["total_chunks"],
                        "base_url": metadata["job"]["base_url"]
                    })
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")
        
        return sorted(libraries, key=lambda x: x["timestamp"], reverse=True)
