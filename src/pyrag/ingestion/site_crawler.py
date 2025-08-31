"""Site crawler for discovering documentation links recursively."""

import asyncio
import re
import aiohttp
from typing import List, Set, Dict, Any, Optional
from urllib.parse import urljoin, urlparse, urlunparse
from dataclasses import dataclass
from ..logging import get_logger
from ..llm.client import LLMClient

logger = get_logger(__name__)

@dataclass
class CrawlResult:
    """Result of a site crawling operation."""
    discovered_urls: Set[str]
    relevant_urls: Set[str]
    crawl_stats: Dict[str, Any]
    errors: List[str]

class SiteCrawler:
    """Recursive site crawler for discovering documentation links."""
    
    def __init__(
        self,
        llm_client: Optional[LLMClient] = None,
        max_depth: int = 3,
        max_pages: int = 100,
        delay: float = 1.0,
        timeout: int = 30
    ):
        self.llm_client = llm_client
        self.max_depth = max_depth
        self.max_pages = max_pages
        self.delay = delay
        self.timeout = timeout
        self.logger = get_logger(__name__)
        
        # Track crawled URLs to avoid duplicates
        self.crawled_urls: Set[str] = set()
        self.session: Optional[aiohttp.ClientSession] = None
    
    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.timeout),
            headers={
                "User-Agent": "PyRAG Documentation Crawler/1.0"
            }
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()
    
    async def crawl_documentation_site(
        self,
        base_url: str,
        library_name: str,
        exclude_patterns: Optional[List[str]] = None,
        include_patterns: Optional[List[str]] = None
    ) -> CrawlResult:
        """Crawl a documentation site to discover all relevant links."""
        
        if not self.session:
            raise RuntimeError("Crawler not initialized. Use async context manager.")
        
        self.logger.info(f"Starting documentation site crawl: {base_url}")
        
        # Initialize tracking
        self.crawled_urls.clear()
        discovered_urls: Set[str] = set()
        relevant_urls: Set[str] = set()
        errors: List[str] = []
        
        # Default patterns
        if exclude_patterns is None:
            exclude_patterns = [
                "linkedin.com", "twitter.com", "github.com", "facebook.com",
                "youtube.com", "discord.com", "slack.com", "zapier.com",
                "mailto:", "javascript:", "#", "tel:"
            ]
        
        if include_patterns is None:
            include_patterns = [
                "/docs/", "/guide/", "/tutorial/", "/api/", "/reference/",
                "/introduction", "/quickstart", "/examples", "/learn/"
            ]
        
        try:
            # Start recursive crawling from base URL
            await self._crawl_recursive(
                url=base_url,
                depth=0,
                base_domain=urlparse(base_url).netloc,
                discovered_urls=discovered_urls,
                exclude_patterns=exclude_patterns,
                include_patterns=include_patterns,
                library_name=library_name
            )
            
            # Filter discovered URLs for relevance
            relevant_urls = await self._filter_relevant_urls(
                discovered_urls, base_url, library_name
            )
            
            # Calculate stats
            crawl_stats = {
                "total_discovered": len(discovered_urls),
                "total_relevant": len(relevant_urls),
                "max_depth_reached": self.max_depth,
                "pages_crawled": len(self.crawled_urls),
                "excluded_patterns": exclude_patterns,
                "included_patterns": include_patterns
            }
            
            self.logger.info(f"Crawl completed: {len(discovered_urls)} discovered, {len(relevant_urls)} relevant")
            
            return CrawlResult(
                discovered_urls=discovered_urls,
                relevant_urls=relevant_urls,
                crawl_stats=crawl_stats,
                errors=errors
            )
            
        except Exception as e:
            error_msg = f"Error during site crawl: {e}"
            self.logger.error(error_msg)
            errors.append(error_msg)
            
            return CrawlResult(
                discovered_urls=discovered_urls,
                relevant_urls=relevant_urls,
                crawl_stats={"error": str(e)},
                errors=errors
            )
    
    async def _crawl_recursive(
        self,
        url: str,
        depth: int,
        base_domain: str,
        discovered_urls: Set[str],
        exclude_patterns: List[str],
        include_patterns: List[str],
        library_name: str
    ):
        """Recursively crawl URLs to discover documentation links."""
        
        # Check limits
        if depth > self.max_depth or len(self.crawled_urls) >= self.max_pages:
            return
        
        # Skip if already crawled
        if url in self.crawled_urls:
            return
        
        # Skip if not same domain
        parsed_url = urlparse(url)
        if parsed_url.netloc != base_domain:
            return
        
        # Skip excluded patterns
        if any(pattern in url.lower() for pattern in exclude_patterns):
            return
        
        try:
            self.logger.info(f"Crawling (depth {depth}): {url}")
            self.crawled_urls.add(url)
            
            # Fetch the page
            async with self.session.get(url) as response:
                if response.status != 200:
                    return
                
                content = await response.text()
                
                # Extract links from HTML
                links = self._extract_links_from_html(content, url)
                
                # Add to discovered URLs
                for link in links:
                    discovered_urls.add(link)
                
                # Recursively crawl relevant links
                if depth < self.max_depth:
                    for link in links:
                        # Check if link is relevant for further crawling
                        if self._is_relevant_for_crawling(link, include_patterns):
                            await asyncio.sleep(self.delay)  # Be respectful
                            await self._crawl_recursive(
                                url=link,
                                depth=depth + 1,
                                base_domain=base_domain,
                                discovered_urls=discovered_urls,
                                exclude_patterns=exclude_patterns,
                                include_patterns=include_patterns,
                                library_name=library_name
                            )
                
        except Exception as e:
            self.logger.warning(f"Error crawling {url}: {e}")
    
    def _extract_links_from_html(self, html_content: str, base_url: str) -> List[str]:
        """Extract all links from HTML content."""
        links = []
        
        # Pattern for HTML anchor tags
        link_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
        
        for match in re.finditer(link_pattern, html_content, re.IGNORECASE):
            link_url = match.group(1)
            
            # Skip anchors and invalid URLs
            if (link_url.startswith('#') or 
                link_url.startswith('mailto:') or 
                link_url.startswith('javascript:') or
                not link_url.strip()):
                continue
            
            # Convert relative URLs to absolute
            if link_url.startswith('/'):
                link_url = urljoin(base_url, link_url)
            elif not link_url.startswith('http'):
                link_url = urljoin(base_url, link_url)
            
            # Clean up URL
            clean_url = link_url.split('#')[0].split('?')[0]
            if clean_url:
                links.append(clean_url)
        
        return list(set(links))  # Remove duplicates
    
    def _is_relevant_for_crawling(self, url: str, include_patterns: List[str]) -> bool:
        """Check if a URL is relevant for further crawling."""
        return any(pattern in url.lower() for pattern in include_patterns)
    
    async def _filter_relevant_urls(
        self,
        discovered_urls: Set[str],
        base_url: str,
        library_name: str
    ) -> Set[str]:
        """Filter discovered URLs for documentation relevance."""
        
        if not self.llm_client:
            # Fallback to basic filtering
            return self._basic_filter_urls(discovered_urls, base_url)
        
        try:
            # Use LLM to filter URLs
            filtered_urls = await self.llm_client.filter_links(
                base_url=base_url,
                all_links=list(discovered_urls),
                library_name=library_name
            )
            
            return set(filtered_urls)
            
        except Exception as e:
            self.logger.warning(f"LLM filtering failed, using basic filtering: {e}")
            return self._basic_filter_urls(discovered_urls, base_url)
    
    def _basic_filter_urls(self, discovered_urls: Set[str], base_url: str) -> Set[str]:
        """Basic URL filtering fallback."""
        base_parsed = urlparse(base_url)
        relevant_urls = set()
        
        exclude_patterns = [
            "linkedin.com", "twitter.com", "github.com", "facebook.com",
            "youtube.com", "discord.com", "slack.com", "zapier.com"
        ]
        
        include_patterns = [
            "/docs/", "/guide/", "/tutorial/", "/api/", "/reference/",
            "/introduction", "/quickstart", "/examples", "/learn/"
        ]
        
        for url in discovered_urls:
            parsed = urlparse(url)
            
            # Must be same domain
            if parsed.netloc != base_parsed.netloc:
                continue
            
            # Check exclude patterns
            if any(pattern in url.lower() for pattern in exclude_patterns):
                continue
            
            # Check include patterns
            if any(pattern in url.lower() for pattern in include_patterns):
                relevant_urls.add(url)
        
        return relevant_urls
