"""Comprehensive documentation manager for two-phase ingestion."""
from .sitemap_analyzer import SitemapEntry
import asyncio
from .crawl4ai_client import Crawl4AIClient
import asyncio
import json
import time
# Use threading for content extraction (better for I/O-bound web scraping)
import concurrent.futures
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp

from ..storage import EmbeddingService
from ..llm.client import LLMClient
from ..logging import get_logger

# Function for threading-based parallel processing
async def process_url_batch_async(url_batch: List[str]) -> List[Dict[str, Any]]:
    """Process a batch of URLs asynchronously - works with threading."""
    from .crawl4ai_client import Crawl4AIClient
    
    client = Crawl4AIClient()
    results = []
    
    async with client:
        for url in url_batch:
            try:
                doc = await client.scrape_url(url)
                results.append({
                    "success": True,
                    "url": url,
                    "content": doc.content,
                    "markdown": doc.markdown,
                    "title": doc.title,
                    "metadata": doc.metadata,
                    "content_length": len(doc.content)
                })
            except Exception as e:
                results.append({
                    "success": False,
                    "url": url,
                    "error": str(e)
                })
    
    return results

# Function for threading-based document processing
async def process_document_async(
    doc: "ScrapedDocument", 
    processor: "DocumentProcessor", 
    library_name: str, 
    version: str,
    doc_index: int,
    total_docs: int
) -> tuple[bool, "ProcessingResult", str]:
    """Process a single document asynchronously - works with threading."""
    try:
        # Check if processor is async (enhanced) or sync (basic)
        if hasattr(processor, "process_scraped_document") and asyncio.iscoroutinefunction(
            processor.process_scraped_document
        ):
            result = await processor.process_scraped_document(
                scraped_doc=doc,
                library_name=library_name,
                version=version,
            )
        else:
            result = processor.process_scraped_document(
                scraped_doc=doc,
                library_name=library_name,
                version=version,
            )
        
        return True, result, f"✅ Processed {doc.url} into {len(result.chunks)} chunks"
    except Exception as e:
        return False, None, f"Error processing document {doc.url}: {e}"
from ..storage import VectorStore
from .document_processor import (
    DocumentProcessor,
    ProcessingResult,
)
from .crawl4ai_client import Crawl4AIClient, ScrapedDocument
from .crawler import CrawlResult, CrawlStrategy, Crawler
from .sitemap_analyzer import SitemapAnalyzer
from .structure_mapper import DocumentationStructureMapper
from .metadata_sanitizer import MetadataSanitizer, sanitize_metadata

logger = get_logger(__name__)

# Global LLM client to avoid recreating for each batch (singleton pattern)
_global_llm_client = None

def get_global_llm_client(llm_client: Optional[LLMClient] = None) -> LLMClient:
    """Get or create the global LLM client (singleton pattern)."""
    global _global_llm_client
    if _global_llm_client is None and llm_client is not None:
        _global_llm_client = llm_client
    return _global_llm_client


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
        cache_dir: str = "./cache",
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = get_global_llm_client(llm_client)
        self.cache_dir = Path(cache_dir)
        self.logger = get_logger(__name__)
        
        # Optimized settings are now the default configuration
        self.llm_filtering_batch_size = 20  # URLs per batch for LLM classification
        self.parallel_processes = 3  # Optimal for MacBook Air
        self.document_processing_threads = 5  # Threads for document processing (LLM calls)

        # Always use enhanced processor for semantic chunking and rich metadata
        if not llm_client:
            raise ValueError(
                "LLM client is required for enhanced documentation processing"
            )

        self.processor = DocumentProcessor(llm_client=llm_client)
        self.logger.info(
            "Using documentation processor with semantic chunking"
        )

        # Initialize metadata sanitizer for ChromaDB compatibility
        self.metadata_sanitizer = MetadataSanitizer()
        self.logger.info("Initialized metadata sanitizer for ChromaDB compatibility")

        # Log client selection
        self.logger.info("🚀 Using Crawl4AI client (local, fast, unlimited)")

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

            if not crawl_result.success:
                error_msg = f"Link discovery failed: {crawl_result.errors}"
                self.logger.error(error_msg)
                errors.append(error_msg)
                return DocumentationResult(
                    job=job,
                    crawl_result=crawl_result,
                    extraction_stats={},
                    processing_stats={},
                    storage_stats={},
                    errors=errors,
                    success=False,
                )

            if not crawl_result.discovered_urls:
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
                    success=False,
                )

            # Phase 2: Content Extraction
            self.logger.info("Phase 2: Extracting content from discovered URLs...")
            extraction_result, documents = await self._extract_content(
                job, crawl_result.discovered_urls
            )
            extraction_stats = extraction_result

            # Phase 3: Process and Store
            self.logger.info("Phase 3: Processing and storing content...")
            processing_result, storage_result = await self._process_and_store(
                job, documents
            )
            processing_stats = processing_result
            storage_stats = storage_result

            # Save metadata
            await self._save_metadata(
                job, crawl_result, extraction_stats, processing_stats
            )

            self.logger.info(
                f"Documentation ingestion completed successfully for {job.library_name}"
            )

            return DocumentationResult(
                job=job,
                crawl_result=crawl_result,
                extraction_stats=extraction_stats,
                processing_stats=processing_stats,
                storage_stats=storage_stats,
                errors=errors,
                success=True,
            )

        except Exception as e:
            error_msg = f"Error during documentation ingestion: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)

            # Create a dummy CrawlResult with the new structure
            dummy_crawl_result = CrawlResult(
                discovered_urls=set(),
                crawled_urls=set(),
                processed_urls=set(),
                content_quality_scores={},
                importance_scores={},
                relationship_data={},
                crawl_statistics={},
                errors=errors,
                warnings=[],
                success=False,
            )

            return DocumentationResult(
                job=job,
                crawl_result=dummy_crawl_result,
                extraction_stats={},
                processing_stats={},
                storage_stats={},
                errors=errors,
                success=False,
            )

    async def _discover_links(self, job: DocumentationJob) -> CrawlResult:
        """Phase 1: Discover all relevant documentation links using optimized pipeline."""

        self.logger.info(f"Starting optimized discovery for {job.library_name}")

        try:
            # Step 1: Analyze sitemap and discover URLs
            self.logger.info("Analyzing documentation sitemap...")

            async with aiohttp.ClientSession() as session:
                sitemap_analyzer = SitemapAnalyzer(session=session)
                sitemap_analysis = await sitemap_analyzer.analyze_documentation_site(
                    job.base_url
                )

            if not sitemap_analysis or not sitemap_analysis.discovered_urls:
                self.logger.warning("No URLs discovered from sitemap analysis")
                return self._create_empty_crawl_result(
                    "No URLs discovered from sitemap"
                )

            self.logger.info(
                f"Discovered {len(sitemap_analysis.discovered_urls)} URLs from sitemap"
            )

            # Step 2: Apply intelligent LLM-based URL filtering (OPTIMIZATION)
            if job.use_llm_filtering:
                self.logger.info("Applying intelligent LLM-based URL filtering...")
                filtered_urls = await self._intelligent_url_filtering(
                    [entry.url for entry in sitemap_analysis.discovered_urls]
                )
                self.logger.info(f"Filtered to {len(filtered_urls)} relevant URLs")
            else:
                filtered_urls = [entry.url for entry in sitemap_analysis.discovered_urls]

            # Step 3: Map documentation structure (using filtered URLs)
            self.logger.info("Mapping documentation structure...")
            structure_mapper = DocumentationStructureMapper()
            structure = structure_mapper.map_documentation_structure(
                filtered_urls, job.base_url
            )

            self.logger.info(
                f"Mapped {len(structure.nodes)} nodes with {len(structure.content_types)} content types"
            )

            # Step 4: Create optimized crawl result (NO REDUNDANT CRAWLING!)
            self.logger.info("Creating optimized crawl result (bypassing redundant crawling)...")
            
            # Convert filtered URLs to SitemapEntry format
            
            filtered_entries = []
            for url in filtered_urls:
                # Find the original entry to preserve metadata
                original_entry = next((entry for entry in sitemap_analysis.discovered_urls if entry.url == url), None)
                if original_entry:
                    filtered_entries.append(original_entry)
                else:
                    # Create a basic entry if not found
                    filtered_entries.append(SitemapEntry(
                        url=url,
                        last_modified=None,
                        change_frequency=None,
                        priority=None,
                        content_type="documentation"
                    ))

            # Create optimized crawl result
            crawl_result = CrawlResult(
                discovered_urls=set(entry.url for entry in filtered_entries),
                crawled_urls=set(),  # No crawling was done
                processed_urls=set(),  # No processing was done yet
                content_quality_scores={},  # No quality scores yet
                importance_scores={},  # No importance scores yet
                relationship_data={},  # No relationship data yet
                crawl_statistics={
                    "total_urls_discovered": len(filtered_entries),
                    "sitemap_urls": sitemap_analysis.sitemap_urls,
                    "filtering_applied": job.use_llm_filtering,
                    "original_urls": len(sitemap_analysis.discovered_urls),
                    "filtered_urls": len(filtered_entries),
                    "filtering_rate": (len(sitemap_analysis.discovered_urls) - len(filtered_entries)) / len(sitemap_analysis.discovered_urls) if sitemap_analysis.discovered_urls else 0
                },
                errors=[],
                warnings=[],
                success=True
            )

            self.logger.info(
                f"Optimized discovery completed: {len(filtered_entries)} URLs ready for extraction"
            )
            return crawl_result

        except Exception as e:
            self.logger.error(f"Discovery failed: {e}")
            return self._create_empty_crawl_result(f"Discovery failed: {e}")

    async def _intelligent_url_filtering(self, urls: List[str]) -> List[str]:
        """Apply intelligent LLM-based URL filtering to remove non-documentation content."""
        
        self.logger.info(f"Applying intelligent URL filtering to {len(urls)} URLs...")
        
        # Process URLs in batches sequentially to avoid resource contention
        batch_size = self.llm_filtering_batch_size
        batches = [urls[i:i+batch_size] for i in range(0, len(urls), batch_size)]
        
        self.logger.info(f"Processing {len(batches)} batches sequentially")
        
        # Combine results
        keep_urls = []
        
        for i, batch in enumerate(batches):
            self.logger.info(f"Processing batch {i + 1}/{len(batches)} ({len(batch)} URLs)")
            
            try:
                classifications = await self._llm_classify_urls_batch(batch)
                
                batch_keep = []
                for url, keep in zip(batch, classifications):
                    if keep:
                        batch_keep.append(url)
                
                keep_urls.extend(batch_keep)
                
                self.logger.info(f"Batch {i + 1}: {len(batch_keep)} keep, {len(batch) - len(batch_keep)} exclude")
                            
            except Exception as e:
                self.logger.warning(f"Batch {i + 1} failed: {e}, keeping all URLs")
                keep_urls.extend(batch)
        
        self.logger.info(f"URL filtering complete: {len(keep_urls)} URLs kept from {len(urls)} original")
        return keep_urls

    async def _llm_classify_urls_batch(self, urls: List[str]) -> List[bool]:
        """Classify multiple URLs in a single LLM call."""
        
        prompt = f"""You are an expert at classifying URLs for a documentation ingestion system. Your task is to determine which URLs contain useful documentation content for developers and users.

CLASSIFY AS DOCUMENTATION (YES) if the URL contains:
- API references, endpoints, and method documentation
- Tutorials, guides, and how-to instructions  
- Code examples, samples, and demos
- Getting started, quickstart, and onboarding content
- Installation, setup, and configuration guides
- Usage instructions and best practices
- Reference materials and specifications
- Troubleshooting and FAQ content
- SDK documentation and code libraries
- Integration guides and examples
- Architecture and design documentation
- Performance optimization guides
- Security and authentication documentation

CLASSIFY AS NON-DOCUMENTATION (NO) if the URL contains:
- Release notes, changelogs, and version history
- Blog posts, news articles, and announcements
- Marketing pages, pricing, and sales content
- Company information, about us, team pages
- Community forums, discussions, and user-generated content
- Support tickets, help desk, and customer service
- Legal pages, privacy policy, terms of service
- Social media links, external platform redirects
- Status pages, monitoring, and system health
- Authentication pages (login, signup, profile)
- File downloads (PDFs, ZIPs, executables)
- Search results, sitemaps, and navigation pages

CONTEXT: These URLs are from documentation websites. Focus on whether the content would be valuable for developers learning to use the technology or API.

URLs to classify:
{chr(10).join(f"{i+1}. {url}" for i, url in enumerate(urls))}

Respond with only "YES" or "NO" for each URL, one per line. YES = documentation, NO = non-documentation."""
        
        try:
            response = await self.llm_client.generate(prompt)
            lines = response.strip().split('\n')
            
            # Parse responses
            classifications = []
            for line in lines:
                line = line.strip().upper()
                if line.startswith('YES'):
                    classifications.append(True)
                elif line.startswith('NO'):
                    classifications.append(False)
                else:
                    # Default to True if unclear
                    classifications.append(True)
            
            # Ensure we have the right number of classifications
            while len(classifications) < len(urls):
                classifications.append(True)
            
            return classifications[:len(urls)]
            
        except Exception as e:
            self.logger.warning(f"LLM classification failed: {e}")
            # Default to keeping all URLs if LLM fails
            return [True] * len(urls)

    def _chunk_urls(self, urls: List[str], num_chunks: int) -> List[List[str]]:
        """Split URLs into simple chunks for parallel processing."""
        chunk_size = len(urls) // num_chunks
        chunks = []
        
        for i in range(num_chunks):
            start = i * chunk_size
            end = start + chunk_size if i < num_chunks - 1 else len(urls)
            chunks.append(urls[start:end])
        
        return chunks



    def _log_crawl_progress(self, progress_data):
        """Log crawling progress updates."""
        # Handle both CrawlProgress objects and dictionaries
        if hasattr(progress_data, "total_crawled"):
            # It's a CrawlProgress object
            total_crawled = progress_data.total_crawled
            total_discovered = progress_data.total_discovered
            elapsed_time = progress_data.elapsed_time
            pages_per_minute = progress_data.pages_per_minute
            completion_percentage = progress_data.completion_percentage
        else:
            # It's a dictionary (fallback)
            total_crawled = progress_data.get("total_crawled", 0)
            total_discovered = progress_data.get("total_discovered", 0)
            elapsed_time = progress_data.get("elapsed_time", 0)
            pages_per_minute = progress_data.get("pages_per_minute", 0)
            completion_percentage = progress_data.get("completion_percentage", 0)

        # Calculate remaining time estimate
        if pages_per_minute > 0:
            remaining_urls = total_discovered - total_crawled
            remaining_minutes = remaining_urls / pages_per_minute
            time_estimate = f" (~{remaining_minutes:.1f} min remaining)"
        else:
            time_estimate = ""

        self.logger.info(
            f"🔄 Crawl Progress: {total_crawled}/{total_discovered} URLs processed "
            f"({completion_percentage:.1f}%) - "
            f"{pages_per_minute:.1f} pages/min{time_estimate}"
        )

    def _create_crawl_strategy(
        self, job: DocumentationJob, structure
    ) -> CrawlStrategy:
        """Create crawling strategy based on structure analysis."""

        # Determine strategy based on structure complexity
        total_nodes = len(structure.nodes)
        content_types = len(structure.content_types)

        if total_nodes > 200 or content_types > 8:
            strategy_name = "aggressive"
            max_concurrent = 10
            max_depth = 6
        elif total_nodes > 100 or content_types > 5:
            strategy_name = "comprehensive"
            max_concurrent = 8
            max_depth = 5
        elif total_nodes > 50 or content_types > 3:
            strategy_name = "balanced"
            max_concurrent = 5
            max_depth = 4
        else:
            strategy_name = "selective"
            max_concurrent = 3
            max_depth = 3

        self.logger.info(
            f"Selected {strategy_name} strategy: {max_concurrent} concurrent, depth {max_depth}"
        )

        return CrawlStrategy(
            name=strategy_name,
            max_concurrent_requests=min(
                max_concurrent, 3
            ),  # Cap at 3 for rate limit respect
            request_delay=2.0,  # Increased delay to be more respectful
            max_depth=max_depth,
            content_quality_threshold=0.0,
            importance_threshold=0.0,
            adaptive_depth=True,
            content_based_filtering=True,
            relationship_tracking=True,
        )

    def _create_empty_crawl_result(self, error_message: str) -> CrawlResult:
        """Create an empty crawl result with error information."""

        return CrawlResult(
            discovered_urls=set(),
            crawled_urls=set(),
            processed_urls=set(),
            content_quality_scores={},
            importance_scores={},
            relationship_data={},
            crawl_statistics={
                "total_discovered": 0,
                "total_crawled": 0,
                "total_processed": 0,
                "elapsed_time": 0.0,
                "pages_per_minute": 0.0,
                "completion_percentage": 0.0,
                "strategy_used": "none",
                "max_depth_reached": 0,
            },
            errors=[error_message],
            warnings=[],
            success=False,
        )

    async def _extract_content(
        self, job: DocumentationJob, urls: Set[str]
    ) -> tuple[Dict[str, Any], List[ScrapedDocument]]:
        """Phase 2: Extract content from discovered URLs using parallel processing."""

        self.logger.info("🚀 Using Crawl4AI client with parallel processing for content extraction")

        # Limit number of pages to extract (<=0 means no cap)
        all_urls = list(urls)
        urls_to_extract = (
            all_urls if getattr(job, "max_content_pages", 0) <= 0 else all_urls[: job.max_content_pages]
        )

        self.logger.info(f"Extracting content from {len(urls_to_extract)} URLs using parallel processing")


        
        start_time = time.time()
        
        # Threading configuration (better for I/O-bound web scraping)
        NUM_THREADS = self.parallel_processes
        self.logger.info(f"🚀 Using {NUM_THREADS} threads for parallel processing")
        
        # Split URLs into chunks
        url_chunks = self._chunk_urls(urls_to_extract, NUM_THREADS)
        
        self.logger.info(f"📦 Split into {NUM_THREADS} chunks:")
        for i, chunk in enumerate(url_chunks):
            self.logger.info(f"   Thread {i}: {len(chunk)} URLs")
        
        # Process in parallel using threading
        self.logger.info(f"🔄 Starting threaded parallel processing...")
        chunk_results = []
        
        # Use asyncio.gather to run all chunks concurrently
        tasks = [process_url_batch_async(chunk) for chunk in url_chunks]
        chunk_results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions that occurred
        for i, result in enumerate(chunk_results):
            if isinstance(result, Exception):
                self.logger.error(f"Thread {i} failed: {result}")
                chunk_results[i] = []  # Replace exception with empty list
        
        # Flatten results
        all_results = []
        for chunk_result in chunk_results:
            all_results.extend(chunk_result)
        
        # Calculate statistics
        documents = []
        failed_urls = []
        total_content_length = 0
        
        for result_item in all_results:
            if result_item.get("success", False):
                # Recreate ScrapedDocument from result
                doc = ScrapedDocument(
                    url=result_item["url"],
                    content=result_item.get("content", ""),
                    markdown=result_item.get("markdown", ""),
                    title=result_item.get("title", ""),
                    metadata=result_item.get("metadata", {}),
                    screenshot_url=None
                )
                documents.append(doc)
                total_content_length += result_item.get("content_length", 0)
            else:
                failed_urls.append(result_item["url"])
        
        end_time = time.time()
        total_time = end_time - start_time
        
        self.logger.info(f"✅ Parallel content extraction completed:")
        self.logger.info(f"   • Total URLs: {len(urls_to_extract)}")
        self.logger.info(f"   • Successful: {len(documents)} ({len(documents)/len(urls_to_extract)*100:.1f}%)")
        self.logger.info(f"   • Failed: {len(failed_urls)} ({len(failed_urls)/len(urls_to_extract)*100:.1f}%)")
        self.logger.info(f"   • Total content: {total_content_length:,} characters")
        if len(documents) > 0:
            avg_content = total_content_length // len(documents)
            self.logger.info(f"   • Average per page: {avg_content:,} chars")
        self.logger.info(f"   • Total time: {total_time/60:.1f} minutes")
        self.logger.info(f"   • Throughput: {len(urls_to_extract)/total_time*60:.1f} URLs/minute")
        self.logger.info(f"   • Speed improvement: ~{NUM_THREADS}x faster than sequential (threading)")

        return {
            "total_urls": len(urls),
            "extracted_urls": len(documents),
            "failed_urls": len(failed_urls),
            "failed_url_list": failed_urls,
            "total_content_length": total_content_length,
            "total_time_seconds": total_time,
            "parallel_threads": NUM_THREADS,
            "throughput_urls_per_minute": len(urls_to_extract)/total_time*60,
            "document_urls": [doc.url for doc in documents],
        }, documents

    async def _process_and_store(
        self, job: DocumentationJob, documents: List[ScrapedDocument]
    ) -> tuple[Dict[str, Any], Dict[str, Any]]:
        """Phase 3: Process documents and store in vector database."""

        self.logger.info(f"Processing {len(documents)} documents")

        processing_results = []
        total_chunks = 0
        total_content_length = 0
        content_type_distribution = {}
        # Token-aware splitting stats
        token_over_limit_chunks = 0
        token_split_segments = 0

        # Process documents in parallel using threading (better for I/O-bound LLM calls)
        self.logger.info(f"🚀 Processing {len(documents)} documents with {self.document_processing_threads} threads...")
        
        # Create semaphore to limit concurrent LLM calls
        semaphore = asyncio.Semaphore(self.document_processing_threads)
        
        async def process_document_with_semaphore(doc, i):
            async with semaphore:
                return await process_document_async(
                    doc=doc,
                    processor=self.processor,
                    library_name=job.library_name,
                    version=job.version,
                    doc_index=i,
                    total_docs=len(documents)
                )
        
        # Create tasks for parallel processing
        tasks = []
        for i, doc in enumerate(documents, 1):
            task = process_document_with_semaphore(doc, i)
            tasks.append(task)
        
        # Process all documents concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                self.logger.error(f"Document {i+1} processing failed: {result}")
                continue
                
            success, processing_result, message = result
            
            if success and processing_result:
                processing_results.append(processing_result)
                total_chunks += len(processing_result.chunks)
                total_content_length += len(documents[i].content)
                
                self.logger.info(message)
                
                # Track content type distribution
                for chunk in processing_result.chunks:
                    content_type = chunk.metadata.get("content_type", "unknown")
                    content_type_distribution[content_type] = (
                        content_type_distribution.get(content_type, 0) + 1
                    )
            else:
                self.logger.error(message)

        # Store in vector database
        self.logger.info(f"💾 Storing {total_chunks} chunks in vector database...")
        stored_chunks = []
        enhanced_metadata_aggregated = {}

        # Build a list of chunk payloads to enable batched embeddings, with token-aware splitting
        chunks_to_store = []  # List[tuple[str, Dict[str, Any]]]: (content, metadata)

        # Use tokenizer from embedding service if available
        tokenizer = getattr(self.embedding_service, "tokenizer", None)
        max_len = getattr(self.embedding_service.config, "max_length", 256)
        # Choose safe chunk window below max_len to account for specials
        safe_window = max(16, min(240, max_len - 16))
        overlap = min(32, max(0, int(safe_window * 0.15)))

        def _split_text_by_tokens(text: str) -> List[str]:
            if tokenizer is None:
                return [text]
            try:
                enc = tokenizer(
                    text,
                    add_special_tokens=False,
                    return_tensors=None,
                    return_attention_mask=False,
                )
                input_ids = enc.get("input_ids", [])
                if not isinstance(input_ids, list):
                    return [text]
                if len(input_ids) <= safe_window:
                    return [text]

                # Sliding windows with overlap
                nonlocal token_over_limit_chunks, token_split_segments
                token_over_limit_chunks += 1
                windows = []
                stride = max(1, safe_window - overlap)
                for start in range(0, len(input_ids), stride):
                    end = start + safe_window
                    window_ids = input_ids[start:end]
                    if not window_ids:
                        break
                    windows.append(
                        tokenizer.decode(window_ids, skip_special_tokens=True)
                    )
                    if end >= len(input_ids):
                        break
                token_split_segments += max(0, len(windows) - 1)
                return windows if windows else [text]
            except Exception:
                return [text]

        for result in processing_results:
            for chunk in result.chunks:
                # Enhanced metadata for RAG optimization
                enhanced_metadata = chunk.metadata.copy()

                # Add structure metadata if available
                if hasattr(chunk, "structure_metadata"):
                    enhanced_metadata.update(chunk.structure_metadata)

                # Add semantic metadata if available
                if hasattr(chunk, "semantic_metadata"):
                    enhanced_metadata.update(chunk.semantic_metadata)

                # Add coverage metadata if available
                if hasattr(chunk, "coverage_metadata"):
                    enhanced_metadata.update(chunk.coverage_metadata)

                # Ensure content type is set for proper collection selection
                if "content_type" not in enhanced_metadata:
                    enhanced_metadata["content_type"] = "documents"

                # Token-aware split if needed
                pieces = _split_text_by_tokens(chunk.content)
                if len(pieces) == 1:
                    chunks_to_store.append((pieces[0], enhanced_metadata))
                else:
                    for idx, piece in enumerate(pieces):
                        piece_meta = enhanced_metadata.copy()
                        piece_meta["split_index"] = idx
                        piece_meta["split_total"] = len(pieces)
                        piece_meta["original_chunk_length"] = len(chunk.content)
                        chunks_to_store.append((piece, piece_meta))

        # Generate embeddings in batches to speed up processing
        batch_size = 32
        for i in range(0, len(chunks_to_store), batch_size):
            batch = chunks_to_store[i : i + batch_size]
            texts = [item[0] for item in batch]

            try:
                embeddings = await self.embedding_service.generate_embeddings(texts)

                # Convert numpy array to list for ChromaDB compatibility
                if hasattr(embeddings, "tolist"):
                    embeddings = embeddings.tolist()

                # Ensure we have a list of vectors (each a flat list of floats)
                documents_payload = []
                for (content, metadata), embedding in zip(batch, embeddings):
                    # Some backends may return nested lists for single vectors
                    if (
                        isinstance(embedding, list)
                        and embedding
                        and isinstance(embedding[0], list)
                    ):
                        embedding = embedding[0]

                    # Sanitize metadata for ChromaDB compatibility
                    sanitized_metadata = self.metadata_sanitizer.sanitize_metadata(metadata)
                    
                    # Log any metadata sanitization issues
                    if sanitized_metadata != metadata:
                        self.logger.info(f"Metadata sanitized for ChromaDB compatibility")
                        sanitization_report = self.metadata_sanitizer.get_sanitization_report()
                        self.logger.debug(f"Sanitization report: {sanitization_report}")

                    documents_payload.append(
                        {
                            "content": content,
                            "metadata": sanitized_metadata,
                            "embedding": embedding,
                        }
                    )

                chunk_ids = await self.vector_store.add_documents(documents_payload)
                stored_chunks.extend(chunk_ids)

                if len(stored_chunks) % 10 == 0:
                    self.logger.info(f"💾 Stored {len(stored_chunks)} chunks so far...")

            except Exception as e:
                self.logger.error(f"Error storing batch starting at index {i}: {e}")

            # Aggregate enhanced metadata if available
            if hasattr(result, "enhanced_metadata") and result.enhanced_metadata:
                for key, value in result.enhanced_metadata.items():
                    if key not in enhanced_metadata_aggregated:
                        enhanced_metadata_aggregated[key] = []
                    if isinstance(value, list):
                        enhanced_metadata_aggregated[key].extend(value)
                    else:
                        enhanced_metadata_aggregated[key].append(value)

        # Remove duplicates from aggregated metadata (handle non-hashable types)
        for key in enhanced_metadata_aggregated:
            if isinstance(enhanced_metadata_aggregated[key], list):
                # Convert to strings for deduplication if items are not hashable
                try:
                    enhanced_metadata_aggregated[key] = list(
                        set(enhanced_metadata_aggregated[key])
                    )
                except TypeError:
                    # If items are not hashable (like dicts), use a different approach
                    seen = set()
                    unique_items = []
                    for item in enhanced_metadata_aggregated[key]:
                        if isinstance(item, dict):
                            # Convert dict to string for comparison
                            item_str = str(sorted(item.items()))
                            if item_str not in seen:
                                seen.add(item_str)
                                unique_items.append(item)
                        else:
                            if item not in seen:
                                seen.add(item)
                                unique_items.append(item)
                    enhanced_metadata_aggregated[key] = unique_items

        processing_stats = {
            "total_documents": len(documents),
            "total_chunks": total_chunks,
            "total_content_length": total_content_length,
            "content_type_distribution": content_type_distribution,
            "average_chunks_per_document": (
                total_chunks / len(documents) if documents else 0
            ),
            "token_over_limit_chunks": token_over_limit_chunks,
            "token_split_segments": token_split_segments,
            "embedding_max_length": max_len,
            "enhanced_metadata": (
                enhanced_metadata_aggregated if enhanced_metadata_aggregated else None
            ),
        }

        storage_stats = {
            "stored_chunks": len(stored_chunks),
            "storage_success_rate": (
                len(stored_chunks) / total_chunks if total_chunks > 0 else 0
            ),
        }

        self.logger.info(
            f"🎉 Ingestion complete! Stored {len(stored_chunks)} chunks with {storage_stats['storage_success_rate']:.1%} success rate"
        )

        return processing_stats, storage_stats

    async def _save_metadata(
        self,
        job: DocumentationJob,
        crawl_result: CrawlResult,
        extraction_stats: Dict[str, Any],
        processing_stats: Dict[str, Any],
    ):
        """Save ingestion metadata for future reference."""

        metadata = {
            "library_name": job.library_name,
            "version": job.version,
            "base_url": job.base_url,
            "ingestion_timestamp": time.time(),
            "crawl_stats": crawl_result.crawl_statistics,
            "extraction_stats": extraction_stats,
            "processing_stats": processing_stats,
        }

        # Save to cache directory
        cache_file = self.cache_dir / f"{job.library_name}_{job.version}_metadata.json"
        with open(cache_file, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved to {cache_file}")

    async def get_ingestion_status(
        self, library_name: str, version: str
    ) -> Optional[Dict[str, Any]]:
        """Get status of a previous ingestion job."""

        cache_file = self.cache_dir / f"{library_name}_{version}_metadata.json"

        if cache_file.exists():
            with open(cache_file, "r") as f:
                return json.load(f)

        return None

    async def list_ingested_libraries(self) -> List[Dict[str, Any]]:
        """List all ingested libraries."""

        libraries = []

        for cache_file in self.cache_dir.glob("*_metadata.json"):
            try:
                with open(cache_file, "r") as f:
                    metadata = json.load(f)
                    libraries.append(
                        {
                            "library_name": metadata["job"]["library_name"],
                            "version": metadata["job"]["version"],
                            "timestamp": metadata["job"]["timestamp"],
                            "total_chunks": metadata["processing_stats"][
                                "total_chunks"
                            ],
                            "base_url": metadata["job"]["base_url"],
                        }
                    )
            except Exception as e:
                self.logger.warning(f"Error reading cache file {cache_file}: {e}")

        return sorted(libraries, key=lambda x: x["timestamp"], reverse=True)
