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
from .intelligent_crawler import CrawlResult, CrawlStrategy, IntelligentCrawler
from .sitemap_analyzer import SitemapAnalyzer
from .structure_mapper import DocumentationStructureMapper

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
        cache_dir: str = "./cache",
    ):
        self.vector_store = vector_store
        self.embedding_service = embedding_service
        self.llm_client = llm_client
        self.firecrawl_api_key = firecrawl_api_key
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

    def _log_crawl_progress(self, progress_data: Dict[str, Any]):
        """Log crawling progress updates."""
        self.logger.info(
            f"Crawl progress: {progress_data.get('total_crawled', 0)}/"
            f"{progress_data.get('total_discovered', 0)} URLs processed"
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
            max_concurrent_requests=max_concurrent,
            request_delay=1.0,
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
        """Phase 2: Extract content from discovered URLs using Firecrawl."""

        if not self.firecrawl_api_key:
            raise ValueError("Firecrawl API key required for content extraction")

        self.logger.info(f"Extracting content from {len(urls)} URLs")

        documents = []
        failed_urls = []
        total_content_length = 0

        # Limit number of pages to extract
        urls_to_extract = list(urls)[: job.max_content_pages]

        async with FirecrawlClient(api_key=self.firecrawl_api_key) as client:
            for i, url in enumerate(urls_to_extract, 1):
                try:
                    self.logger.info(
                        f"Extracting content ({i}/{len(urls_to_extract)}): {url}"
                    )

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

        # Process each document
        for doc in documents:
            try:
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

                # Track content type distribution
                for chunk in result.chunks:
                    content_type = chunk.metadata.get("content_type", "unknown")
                    content_type_distribution[content_type] = (
                        content_type_distribution.get(content_type, 0) + 1
                    )

            except Exception as e:
                self.logger.error(f"Error processing document {doc.url}: {e}")

        # Store in vector database
        stored_chunks = []
        enhanced_metadata_aggregated = {}

        for result in processing_results:
            for chunk in result.chunks:
                try:
                    embedding = await self.embedding_service.embed_text(chunk.content)

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

                    document_data = {
                        "content": chunk.content,
                        "metadata": enhanced_metadata,
                        "embedding": embedding,
                    }
                    chunk_ids = await self.vector_store.add_documents([document_data])
                    stored_chunks.extend(chunk_ids)
                except Exception as e:
                    self.logger.error(f"Error storing chunk: {e}")

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
            "average_chunks_per_document": total_chunks / len(documents)
            if documents
            else 0,
            "enhanced_metadata": enhanced_metadata_aggregated
            if enhanced_metadata_aggregated
            else None,
        }

        storage_stats = {
            "stored_chunks": len(stored_chunks),
            "storage_success_rate": len(stored_chunks) / total_chunks
            if total_chunks > 0
            else 0,
        }

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
