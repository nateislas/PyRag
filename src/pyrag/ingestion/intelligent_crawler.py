"""Intelligent crawler for adaptive documentation discovery and crawling."""

import asyncio
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

from ..logging import get_logger
from .sitemap_analyzer import SitemapAnalysis, SitemapAnalyzer
from .structure_mapper import DocumentationStructure, DocumentationStructureMapper

logger = get_logger(__name__)


@dataclass
class CrawlStrategy:
    """Configuration for intelligent crawling strategy."""

    name: str  # "aggressive", "balanced", "selective", "comprehensive"
    max_concurrent_requests: int = 3  # Reduced for better rate limit handling
    request_delay: float = 2.0  # Increased delay to be more respectful
    max_depth: int = 5
    content_quality_threshold: float = 0.6
    importance_threshold: float = 0.5
    max_pages_per_type: Optional[Dict[str, int]] = None
    adaptive_depth: bool = True
    content_based_filtering: bool = True
    relationship_tracking: bool = True


@dataclass
class CrawlProgress:
    """Tracks crawling progress and statistics."""

    total_discovered: int = 0
    total_crawled: int = 0
    total_processed: int = 0
    current_depth: int = 0
    current_batch: int = 0
    start_time: float = field(default_factory=time.time)
    last_update: float = field(default_factory=time.time)
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def elapsed_time(self) -> float:
        return time.time() - self.start_time

    @property
    def pages_per_minute(self) -> float:
        elapsed = self.elapsed_time
        if elapsed > 0:
            return (self.total_crawled / elapsed) * 60
        return 0.0

    @property
    def completion_percentage(self) -> float:
        if self.total_discovered > 0:
            return (self.total_crawled / self.total_discovered) * 100
        return 0.0


@dataclass
class CrawlResult:
    """Result of intelligent crawling operation."""

    discovered_urls: Set[str]
    crawled_urls: Set[str]
    processed_urls: Set[str]
    content_quality_scores: Dict[str, float]
    importance_scores: Dict[str, float]
    relationship_data: Dict[str, Dict[str, Any]]
    crawl_statistics: Dict[str, Any]
    errors: List[str]
    warnings: List[str]
    success: bool


class IntelligentCrawler:
    """Intelligent crawler that adapts to documentation structure and content quality."""

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        strategy: Optional[CrawlStrategy] = None,
        content_analyzer: Optional[Callable] = None,
        progress_callback: Optional[Callable] = None,
    ):
        self.session = session
        self.strategy = strategy or CrawlStrategy("balanced")
        self.content_analyzer = content_analyzer
        self.progress_callback = progress_callback
        self.logger = get_logger(__name__)

        # Initialize components
        self.structure_mapper = DocumentationStructureMapper()

        # Crawling state
        self.crawled_urls: Set[str] = set()
        self.discovered_urls: Set[str] = set()
        self.processing_queue: asyncio.Queue = asyncio.Queue()
        self.progress = CrawlProgress()

        # Content quality tracking
        self.content_quality_scores: Dict[str, float] = {}
        self.importance_scores: Dict[str, float] = {}
        self.relationship_data: Dict[str, Dict[str, Any]] = {}

    async def crawl_documentation_site(
        self,
        base_url: str,
        sitemap_analysis: Optional[SitemapAnalysis] = None,
        structure: Optional[DocumentationStructure] = None,
        custom_strategy: Optional[CrawlStrategy] = None,
    ) -> CrawlResult:
        """Discover and analyze documentation URLs using intelligent, adaptive strategies (no content fetching)."""
        self.logger.info(f"Starting intelligent URL discovery for: {base_url}")

        # Use custom strategy if provided
        if custom_strategy:
            self.strategy = custom_strategy

        # Initialize or use provided analysis
        if not sitemap_analysis:
            sitemap_analyzer = SitemapAnalyzer(session=self.session)
            sitemap_analysis = await sitemap_analyzer.analyze_documentation_site(
                base_url
            )

        if not structure:
            urls = [entry.url for entry in sitemap_analysis.discovered_urls]
            structure = self.structure_mapper.map_documentation_structure(
                urls, base_url
            )

        # Initialize crawling state
        self._initialize_crawl_state(sitemap_analysis, structure)

        # Get prioritized URLs for crawling
        priority_urls = self._get_crawl_priorities(structure, sitemap_analysis)

        # Start crawling with adaptive strategy
        try:
            await self._execute_crawl_strategy(priority_urls, structure)

            # Compile results
            result = self._compile_crawl_result()

            self.logger.info(
                f"URL discovery complete: {len(self.crawled_urls)} URLs analyzed, "
                f"{len(self.discovered_urls)} total discovered"
            )

            return result

        except Exception as e:
            self.logger.error(f"Crawl failed: {e}")
            self.progress.errors.append(str(e))
            return self._compile_crawl_result()

    def _initialize_crawl_state(
        self, sitemap_analysis: SitemapAnalysis, structure: DocumentationStructure
    ):
        """Initialize the crawling state with discovered URLs and structure."""
        # Add discovered URLs to the set
        for entry in sitemap_analysis.discovered_urls:
            self.discovered_urls.add(entry.url)

        # Initialize progress
        self.progress.total_discovered = len(self.discovered_urls)
        self.progress.start_time = time.time()
        self.progress.last_update = time.time()

        # Initialize content quality and importance scores
        for entry in sitemap_analysis.discovered_urls:
            url = entry.url
            # Use structure analysis for scores if available
            if url in structure.nodes:
                node = structure.nodes[url]
                self.importance_scores[url] = node.importance_score
                self.content_quality_scores[url] = node.completeness_score
            else:
                # Default scores
                self.importance_scores[url] = 0.5
                self.content_quality_scores[url] = 0.5

    def _get_crawl_priorities(
        self, structure: DocumentationStructure, sitemap_analysis: SitemapAnalysis
    ) -> List[str]:
        """Get prioritized URLs for crawling based on structure and analysis."""
        priorities = []

        # Use structure mapper priorities if available
        if structure.nodes:
            priorities = self.structure_mapper.get_crawl_priorities(structure)
        else:
            # Fallback to sitemap analysis priorities
            priorities = [entry.url for entry in sitemap_analysis.discovered_urls]

        # Apply strategy-specific filtering
        if self.strategy.name == "selective":
            # Only crawl high-importance URLs
            priorities = [
                url
                for url in priorities
                if self.importance_scores.get(url, 0)
                > self.strategy.importance_threshold
            ]  # No artificial limits - process all high-importance URLs

        elif self.strategy.name == "aggressive":
            # Include all URLs but prioritize by importance
            pass  # No filtering for aggressive strategy

        elif self.strategy.name == "comprehensive":
            # Ensure we cover all content types
            priorities = self._ensure_content_type_coverage(priorities, structure)

        self.logger.info(f"Prioritized {len(priorities)} URLs for crawling")
        return priorities

    def _ensure_content_type_coverage(
        self, priorities: List[str], structure: DocumentationStructure
    ) -> List[str]:
        """Ensure comprehensive coverage across all content types."""
        if not structure.content_types:
            return priorities

        # Start with all high-priority URLs - no artificial limits
        covered_priorities = priorities

        # Add representatives from each content type
        for content_type, urls in structure.content_types.items():
            if content_type not in ["other", "detail"]:  # Skip generic types
                # Add top URLs from this content type
                type_priorities = sorted(
                    urls, key=lambda u: self.importance_scores.get(u, 0), reverse=True
                )[:10]

                for url in type_priorities:
                    if url not in covered_priorities:
                        covered_priorities.append(url)

        return covered_priorities

    async def _execute_crawl_strategy(
        self, priority_urls: List[str], structure: DocumentationStructure
    ):
        """Execute the discovery strategy with adaptive behavior (no content fetching)."""
        # Create URL analysis tasks
        tasks = []
        semaphore = asyncio.Semaphore(self.strategy.max_concurrent_requests)

        for url in priority_urls:
            if url not in self.crawled_urls:
                task = self._crawl_url_with_semaphore(url, semaphore, structure)
                tasks.append(task)

        # Execute tasks with progress tracking
        if tasks:
            await self._execute_tasks_with_progress(tasks)

        # Adaptive depth analysis if enabled
        if self.strategy.adaptive_depth and self.strategy.max_depth > 1:
            await self._adaptive_depth_crawling(structure)

    async def _crawl_url_with_semaphore(
        self, url: str, semaphore: asyncio.Semaphore, structure: DocumentationStructure
    ):
        """Analyze a URL with semaphore control for concurrency (discovery-only)."""
        async with semaphore:
            try:
                await self._crawl_single_url(url, structure)
                await asyncio.sleep(self.strategy.request_delay)
            except Exception as e:
                self.logger.error(f"Error analyzing {url}: {e}")
                self.progress.errors.append(f"{url}: {e}")

    async def _crawl_single_url(self, url: str, structure: DocumentationStructure):
        """Analyze a single URL for structure and importance (no content fetching)."""
        if url in self.crawled_urls:
            return

        try:
            self.logger.info(
                f"🔍 Analyzing URL {self.progress.total_crawled + 1}/{len(self.discovered_urls)}: {url}"
            )

            # For discovery-only mode, we don't fetch content
            # Just analyze the URL structure and mark as analyzed
            self.logger.info(f"📊 URL analysis complete for {url}")

            # Mark as analyzed (not crawled, since we're not fetching content)
            self.crawled_urls.add(url)
            self.progress.total_crawled += 1

            self.logger.info(f"✅ Successfully analyzed {url}")

            # Update progress
            self._update_progress()

        except Exception as e:
            self.logger.error(f"❌ Error analyzing {url}: {e}")
            self.progress.errors.append(f"{url}: {e}")

    # Content fetching removed - this is now handled by DocumentationManager with Firecrawl
    # The IntelligentCrawler is discovery-only and doesn't fetch actual content

    # Content quality analysis removed - this is now handled by DocumentationManager with Firecrawl
    # The IntelligentCrawler is discovery-only and doesn't analyze actual content

    async def _extract_relationships_and_urls(
        self, url: str, content_data: Dict[str, Any], structure: DocumentationStructure
    ) -> Set[str]:
        """Extract relationships and discover new URLs from content."""
        new_urls = set()

        # Extract links from content
        links = content_data.get("links", [])
        base_domain = urlparse(url).netloc

        for link in links:
            try:
                link_domain = urlparse(link).netloc

                # Only include same-domain links
                if link_domain == base_domain or link_domain.endswith(
                    f".{base_domain}"
                ):
                    # Check if it's a new documentation URL
                    if (
                        self._is_documentation_url(link)
                        and link not in self.discovered_urls
                    ):
                        new_urls.add(link)

            except Exception as e:
                self.logger.debug(f"Error processing link {link}: {e}")

        # Store relationship data
        if new_urls:
            self.relationship_data[url] = {
                "discovered_urls": list(new_urls),
                "content_quality": self.content_quality_scores.get(url, 0.5),
                "timestamp": time.time(),
            }

        return new_urls

    def _is_documentation_url(self, url: str) -> bool:
        """Check if a URL appears to be documentation content."""
        url_lower = url.lower()

        # Exclude common non-documentation patterns
        exclude_patterns = [
            "blog",
            "news",
            "changelog",
            "download",
            "github.com",
            "twitter.com",
            "linkedin.com",
            "facebook.com",
            "youtube.com",
            "discord.com",
            "mailto:",
            "javascript:",
            "tel:",
            "#",
            ".pdf",
            ".zip",
            ".tar.gz",
        ]

        for pattern in exclude_patterns:
            if pattern in url_lower:
                return False

        # Include if it has a meaningful path
        parsed = urlparse(url)
        if parsed.path and len(parsed.path) > 1:
            return True

        return False

    def _add_newly_discovered_urls(self, new_urls: Set[str]):
        """Add newly discovered URLs to the tracking sets."""
        for url in new_urls:
            if url not in self.discovered_urls:
                self.discovered_urls.add(url)
                self.progress.total_discovered += 1

                # Initialize scores for new URLs
                if url not in self.importance_scores:
                    self.importance_scores[url] = 0.5
                if url not in self.content_quality_scores:
                    self.content_quality_scores[url] = 0.5

    async def _adaptive_depth_crawling(self, structure: DocumentationStructure):
        """Perform adaptive depth crawling based on content quality and importance."""
        if not structure.nodes:
            return

        # Group URLs by depth
        depth_groups = {}
        for url, node in structure.nodes.items():
            if url not in self.crawled_urls:
                depth = node.depth
                if depth not in depth_groups:
                    depth_groups[depth] = []
                depth_groups[depth].append(url)

        # Crawl deeper levels if content quality is good
        for depth in sorted(depth_groups.keys()):
            if depth > self.strategy.max_depth:
                break

            urls_at_depth = depth_groups[depth]

            # Check if we should continue at this depth
            if not await self._should_continue_at_depth(depth, urls_at_depth):
                break

            # Crawl URLs at this depth
            await self._crawl_depth_level(urls_at_depth, structure)

    async def _should_continue_at_depth(self, depth: int, urls: List[str]) -> bool:
        """Determine if we should continue crawling at a given depth."""
        if not urls:
            return False

        # Check content quality at this depth
        quality_scores = [self.content_quality_scores.get(url, 0.5) for url in urls]
        avg_quality = sum(quality_scores) / len(quality_scores)

        # Check importance at this depth
        importance_scores = [self.importance_scores.get(url, 0.5) for url in urls]
        avg_importance = sum(importance_scores) / len(importance_scores)

        # Continue if quality and importance are above thresholds
        should_continue = (
            avg_quality >= self.strategy.content_quality_threshold
            and avg_importance >= self.strategy.importance_threshold
        )

        self.logger.debug(
            f"Depth {depth}: avg_quality={avg_quality:.2f}, "
            f"avg_importance={avg_importance:.2f}, continue={should_continue}"
        )

        return should_continue

    async def _crawl_depth_level(
        self, urls: List[str], structure: DocumentationStructure
    ):
        """Crawl all URLs at a specific depth level."""
        # Prioritize URLs by importance and quality
        prioritized_urls = sorted(
            urls,
            key=lambda u: (
                self.importance_scores.get(u, 0.5),
                self.content_quality_scores.get(u, 0.5),
            ),
            reverse=True,
        )

        # Limit based on strategy
        if self.strategy.max_pages_per_type:
            max_pages = self.strategy.max_pages_per_type.get("detail", 50)
            prioritized_urls = prioritized_urls[:max_pages]

        # Create crawling tasks
        tasks = []
        semaphore = asyncio.Semaphore(self.strategy.max_concurrent_requests)

        for url in prioritized_urls:
            if url not in self.crawled_urls:
                task = self._crawl_url_with_semaphore(url, semaphore, structure)
                tasks.append(task)

        # Execute tasks
        if tasks:
            await self._execute_tasks_with_progress(tasks)

    async def _execute_tasks_with_progress(self, tasks: List[asyncio.Task]):
        """Execute tasks while updating progress."""
        # Execute in batches for progress tracking
        batch_size = 10
        for i in range(0, len(tasks), batch_size):
            batch = tasks[i : i + batch_size]
            self.progress.current_batch += 1

            # Execute batch
            await asyncio.gather(*batch, return_exceptions=True)

            # Update progress
            self._update_progress()

            # Small delay between batches
            await asyncio.sleep(0.1)

    def _update_progress(self):
        """Update progress and call callback if provided."""
        self.progress.last_update = time.time()

        if self.progress_callback:
            try:
                self.progress_callback(self.progress)
            except Exception as e:
                self.logger.warning(f"Progress callback failed: {e}")

    def _compile_crawl_result(self) -> CrawlResult:
        """Compile the final crawl result."""
        # Calculate statistics
        crawl_stats = {
            "total_discovered": len(self.discovered_urls),
            "total_crawled": len(self.crawled_urls),
            "total_processed": len(self.crawled_urls),  # For now, crawled = processed
            "elapsed_time": self.progress.elapsed_time,
            "pages_per_minute": self.progress.pages_per_minute,
            "completion_percentage": self.progress.completion_percentage,
            "strategy_used": self.strategy.name,
            "max_depth_reached": max(
                (len(urlparse(url).path.split("/")) for url in self.crawled_urls),
                default=0,
            ),
        }

        # Calculate quality statistics
        if self.content_quality_scores:
            quality_values = list(self.content_quality_scores.values())
            crawl_stats["quality_stats"] = {
                "average_quality": sum(quality_values) / len(quality_values),
                "high_quality_pages": sum(1 for q in quality_values if q > 0.8),
                "low_quality_pages": sum(1 for q in quality_values if q < 0.3),
            }

        # Calculate importance statistics
        if self.importance_scores:
            importance_values = list(self.importance_scores.values())
            crawl_stats["importance_stats"] = {
                "average_importance": sum(importance_values) / len(importance_values),
                "high_importance_pages": sum(1 for i in importance_values if i > 0.8),
                "low_importance_pages": sum(1 for i in importance_values if i < 0.3),
            }

        return CrawlResult(
            discovered_urls=self.discovered_urls,
            crawled_urls=self.crawled_urls,
            processed_urls=self.crawled_urls,
            content_quality_scores=self.content_quality_scores,
            importance_scores=self.importance_scores,
            relationship_data=self.relationship_data,
            crawl_statistics=crawl_stats,
            errors=self.progress.errors,
            warnings=self.progress.warnings,
            success=len(self.progress.errors)
            < 10,  # Consider successful if < 10 errors
        )

    def get_crawl_recommendations(self) -> Dict[str, Any]:
        """Get recommendations for future crawling based on current results."""
        recommendations = {
            "strategy_adjustments": [],
            "content_improvements": [],
            "coverage_gaps": [],
            "performance_optimizations": [],
        }

        # Strategy adjustments
        if self.progress.completion_percentage < 50:
            recommendations["strategy_adjustments"].append(
                "Consider more aggressive crawling strategy for better coverage"
            )

        if self.progress.pages_per_minute < 10:
            recommendations["strategy_adjustments"].append(
                "Increase concurrent requests or reduce delays for better performance"
            )

        # Content improvements
        if self.content_quality_scores:
            avg_quality = sum(self.content_quality_scores.values()) / len(
                self.content_quality_scores
            )
            if avg_quality < 0.6:
                recommendations["content_improvements"].append(
                    "Content quality is below threshold - consider content filtering"
                )

        # Coverage gaps
        if len(self.discovered_urls) - len(self.crawled_urls) > 100:
            recommendations["coverage_gaps"].append(
                f"Large number of undiscovered URLs ({len(self.discovered_urls) - len(self.crawled_urls)}) - consider deeper crawling"
            )

        # Performance optimizations
        if self.progress.elapsed_time > 300:  # 5 minutes
            recommendations["performance_optimizations"].append(
                "Crawl time is high - consider parallel processing or content filtering"
            )

        return recommendations
