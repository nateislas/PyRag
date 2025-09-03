"""Comprehensive documentation manager for two-phase ingestion."""

import asyncio
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import aiohttp

from ..embeddings import EmbeddingService
from ..llm.client import LLMClient
from ..logging import get_logger
from ..vector_store import VectorStore
from .enhanced_documentation_processor import (
    EnhancedDocumentationProcessor,
    EnhancedProcessingResult,
)
from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .crawl4ai_client import Crawl4AIClient
from .intelligent_crawler import CrawlResult, CrawlStrategy, IntelligentCrawler
from .sitemap_analyzer import SitemapAnalyzer
from .structure_mapper import DocumentationStructureMapper
from .metadata_sanitizer import MetadataSanitizer, sanitize_metadata

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
        use_crawl4ai: bool = False,
        cache_dir: str = "./cache",
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.firecrawl_api_key = firecrawl_api_key
        self.use_crawl4ai = use_crawl4ai
        self.cache_dir = Path(cache_dir)
        self.logger = get_logger(__name__)

        # Always use enhanced processor for semantic chunking and rich metadata
        if not llm_client:
            raise ValueError(
                "LLM client is required for enhanced documentation processing"
            )

        self.processor = EnhancedDocumentationProcessor(llm_client=llm_client)
        self.logger.info(
            "Using enhanced documentation processor with semantic chunking"
        )

        # Initialize metadata sanitizer for ChromaDB compatibility
        self.metadata_sanitizer = MetadataSanitizer()
        self.logger.info("Initialized metadata sanitizer for ChromaDB compatibility")

        # Log client selection
        if self.use_crawl4ai:
            self.logger.info("🚀 Using Crawl4AI client (local, fast, unlimited)")
        else:
            self.logger.info("🌐 Using Firecrawl client (external API)")

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
        """Phase 1: Discover all relevant documentation links using enhanced pipeline."""

        self.logger.info(f"Starting enhanced discovery for {job.library_name}")

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

            # Step 2: Map documentation structure
            self.logger.info("Mapping documentation structure...")
            structure_mapper = DocumentationStructureMapper()
            structure = structure_mapper.map_documentation_structure(
                [entry.url for entry in sitemap_analysis.discovered_urls], job.base_url
            )

            self.logger.info(
                f"Mapped {len(structure.nodes)} nodes with {len(structure.content_types)} content types"
            )

            # Step 3: Create intelligent crawling strategy based on structure
            strategy = self._create_enhanced_crawl_strategy(job, structure)

            # Step 4: Execute enhanced crawling with our discovered URLs
            self.logger.info("Executing enhanced crawling with discovered URLs...")

            crawler = IntelligentCrawler(
                strategy=strategy, progress_callback=self._log_crawl_progress
            )

            # Use our enhanced analysis results directly
            crawl_result = await crawler.crawl_documentation_site(
                base_url=job.base_url,
                sitemap_analysis=sitemap_analysis,  # Pass our enhanced analysis
                structure=structure,  # Pass our enhanced structure
                custom_strategy=strategy,  # Pass our enhanced strategy
            )

            self.logger.info(
                f"Enhanced crawling completed: {len(crawl_result.discovered_urls)} URLs discovered"
            )
            return crawl_result

        except Exception as e:
            self.logger.error(f"Enhanced discovery failed: {e}")
            return self._create_empty_crawl_result(f"Discovery failed: {e}")

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

    def _create_enhanced_crawl_strategy(
        self, job: DocumentationJob, structure
    ) -> CrawlStrategy:
        """Create enhanced crawling strategy based on structure analysis."""

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
            content_quality_threshold=0.6,
            importance_threshold=0.5,
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
        """Phase 2: Extract content from discovered URLs using selected client."""

        # Choose client based on configuration
        if self.use_crawl4ai:
            self.logger.info("🚀 Using Crawl4AI client for content extraction")
            client = Crawl4AIClient()
        else:
            if not self.firecrawl_api_key:
                raise ValueError("Firecrawl API key required for content extraction")
            self.logger.info("🌐 Using Firecrawl client for content extraction")
            client = FirecrawlClient(api_key=self.firecrawl_api_key)

        self.logger.info(f"Extracting content from {len(urls)} URLs")

        documents = []
        failed_urls = []
        total_content_length = 0

        # Limit number of pages to extract (<=0 means no cap)
        all_urls = list(urls)
        urls_to_extract = (
            all_urls if getattr(job, "max_content_pages", 0) <= 0 else all_urls[: job.max_content_pages]
        )

        async with client:
            for i, url in enumerate(urls_to_extract, 1):
                try:
                    self.logger.info(
                        f"🌐 Extracting content ({i}/{len(urls_to_extract)}): {url}"
                    )

                    doc = await client.scrape_url(url)
                    documents.append(doc)
                    total_content_length += len(doc.content)

                    # Enhanced caching feedback (only for Firecrawl)
                    if not self.use_crawl4ai and hasattr(client, 'is_cached_result'):
                        if client.is_cached_result(doc):
                            cache_age = client.get_cache_age_hours(doc)
                            self.logger.info(
                                f"⚡ CACHE HIT for {url} (age: {cache_age:.1f}h) - Instant response!"
                            )
                        else:
                            self.logger.info(f"🆕 Fresh scrape for {url} - Stored in cache for future use")
                    else:
                        self.logger.info(f"✅ Successfully extracted: {url}")

                    # Add delay to be respectful and avoid rate limits (only for Firecrawl)
                    if not self.use_crawl4ai:
                        await asyncio.sleep(2)

                except Exception as e:
                    self.logger.warning(f"Failed to extract content from {url}: {e}")
                    failed_urls.append(url)

        return {
            "total_urls": len(urls),
            "extracted_urls": len(documents),
            "failed_urls": len(failed_urls),
            "failed_url_list": failed_urls,
            "total_content_length": total_content_length,
            "document_urls": [
                doc.url for doc in documents
            ],  # Store URLs instead of objects
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

        # Process each document
        for i, doc in enumerate(documents, 1):
            try:
                self.logger.info(
                    f"📄 Processing document ({i}/{len(documents)}): {doc.url}"
                )

                # Check if processor is async (enhanced) or sync (basic)
                if hasattr(
                    self.processor, "process_scraped_document"
                ) and asyncio.iscoroutinefunction(
                    self.processor.process_scraped_document
                ):
                    result = await self.processor.process_scraped_document(
                        scraped_doc=doc,
                        library_name=job.library_name,
                        version=job.version,
                    )
                else:
                    result = self.processor.process_scraped_document(
                        scraped_doc=doc,
                        library_name=job.library_name,
                        version=job.version,
                    )
                processing_results.append(result)

                total_chunks += len(result.chunks)
                total_content_length += len(doc.content)

                self.logger.info(
                    f"✅ Processed {doc.url} into {len(result.chunks)} chunks"
                )

                # Track content type distribution
                for chunk in result.chunks:
                    content_type = chunk.metadata.get("content_type", "unknown")
                    content_type_distribution[content_type] = (
                        content_type_distribution.get(content_type, 0) + 1
                    )

            except Exception as e:
                self.logger.error(f"Error processing document {doc.url}: {e}")

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
