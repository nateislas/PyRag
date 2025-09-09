"""Crawl4AI client wrapper for documentation scraping.

This client provides the same interface as FirecrawlClient but uses Crawl4AI
for local, fast, and unlimited web scraping without external API dependencies.
"""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional
from pathlib import Path

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document from Crawl4AI.
    
    This matches the interface of FirecrawlClient's ScrapedDocument.
    """
    url: str
    title: str
    content: str
    markdown: str
    metadata: Dict[str, Any]
    screenshot_url: Optional[str] = None


class Crawl4AIClient:
    """Client for interacting with Crawl4AI for documentation scraping.
    
    This client provides the same interface as FirecrawlClient but uses
    Crawl4AI for local, fast, and unlimited web scraping.
    """
    
    def __init__(self, api_key: Optional[str] = None, base_url: str = ""):
        """Initialize Crawl4AI client.
        
        Args:
            api_key: Not used for Crawl4AI (kept for interface compatibility)
            base_url: Not used for Crawl4AI (kept for interface compatibility)
        """
        try:
            from crawl4ai import AsyncWebCrawler
            self.crawler = AsyncWebCrawler()
            self.session = None  # Not needed for Crawl4AI
            logger.info("✅ Crawl4AI client initialized successfully")
        except ImportError:
            logger.error("❌ Crawl4AI not installed. Install with: pip install crawl4ai")
            raise ImportError("Crawl4AI is required. Install with: pip install crawl4ai")
        
        # Set default headers (not used but kept for interface compatibility)
        self.headers = {"Content-Type": "application/json"}
        self.api_key = api_key  # Not used but kept for interface compatibility
        self.base_url = base_url  # Not used but kept for interface compatibility

    async def __aenter__(self):
        """Async context manager entry."""
        # Crawl4AI doesn't need session management
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        # Crawl4AI doesn't need session cleanup
        pass

    async def scrape_url(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ScrapedDocument:
        """Scrape a single URL using Crawl4AI.
        
        Args:
            url: The URL to scrape
            options: Optional configuration (kept for interface compatibility)
            
        Returns:
            ScrapedDocument with the scraped content
        """
        try:
            logger.info(f"🌐 Scraping {url} with Crawl4AI (local, fast, unlimited)")
            
            # Configure crawling options - OPTIMIZED for documentation scraping
            crawl_options = {
                "max_pages": 1,  # Single page for this method
                "extract_markdown": True,  # ✅ We need this
                "extract_html": False,     # ❌ Remove HTML extraction
                "extract_links": False,    # ❌ Remove link extraction
                "extract_forms": False,    # ❌ Remove form extraction
                "extract_tables": False,   # ❌ Remove table extraction
                "screenshot": False,       # ❌ Remove screenshot capture
                "memory_ceiling_mb": 10,   # ✅ Much lower memory limit
                "batch_size": 1,           # Single page batch
                "wait_for": "networkidle", # ✅ Wait for content to load
                "timeout": 30000,          # ✅ 30 second timeout
                "user_agent": "Mozilla/5.0 (compatible; PyRAG/1.0)",  # ✅ Proper UA
            }
            
            # Override with any provided options
            if options:
                crawl_options.update(options)
            
            # Scrape the single URL
            result = await self.crawler.arun(
                url=url,
                **crawl_options
            )
            
            # Extract content from the result - OPTIMIZED
            if hasattr(result, 'markdown') and result.markdown:
                content = result.markdown
                markdown = result.markdown
            else:
                # Fallback: if markdown extraction failed, try HTML
                if hasattr(result, 'html') and result.html:
                    content = self._clean_html_content(result.html)
                    markdown = self._html_to_markdown(result.html)
                else:
                    content = "No content extracted"
                    markdown = "# No Content\n\nNo content could be extracted from this page."
            
            # Extract title
            title = getattr(result, 'title', url) or url
            
            # Build metadata - OPTIMIZED (minimal metadata)
            metadata = {
                "url": url,
                "title": title,
                "content_length": len(content),
                "scraped_at": asyncio.get_event_loop().time(),
                "cache_hit": False,  # Crawl4AI doesn't have caching like Firecrawl
                "cache_age_hours": None,
                "source": "crawl4ai",
                "extractor": "local_optimized"
            }
            
            # Add any additional metadata from the result
            if hasattr(result, 'metadata'):
                metadata.update(result.metadata)
            
            logger.info(f"✅ Successfully scraped {url}")
            logger.info(f"   - Title: {title}")
            logger.info(f"   - Content length: {len(content)} chars")
            logger.info(f"   - Markdown length: {len(markdown)} chars")
            
            return ScrapedDocument(
                url=url,
                title=title,
                content=content,
                markdown=markdown,
                metadata=metadata,
                screenshot_url=None  # ❌ No screenshots
            )
            
        except Exception as e:
            logger.error(f"❌ Error scraping {url} with Crawl4AI: {e}")
            # Return error document instead of raising (matches FirecrawlClient behavior)
            return ScrapedDocument(
                url=url,
                title="Error",
                content=f"Failed to scrape content: {e}",
                markdown=f"# Error\n\nFailed to scrape content: {e}",
                metadata={
                    "error": str(e),
                    "cache_hit": False,
                    "source": "crawl4ai"
                },
                screenshot_url=None,  # ❌ No screenshots
            )

    async def crawl_site(
        self, start_url: str, options: Optional[Dict[str, Any]] = None
    ) -> List[ScrapedDocument]:
        """Crawl an entire site using Crawl4AI.
        
        Args:
            start_url: The starting URL for crawling
            options: Optional configuration for crawling
            
        Returns:
            List of ScrapedDocument objects
        """
        try:
            logger.info(f"🕷️  Crawling site {start_url} with Crawl4AI")
            
            # Configure crawling options - OPTIMIZED for documentation scraping
            crawl_options = {
                "max_pages": options.get("max_pages", 100) if options else 100,
                "extract_markdown": True,  # ✅ We need this
                "extract_html": False,     # ❌ Remove HTML extraction
                "extract_links": False,    # ❌ Remove link extraction
                "extract_forms": False,    # ❌ Remove form extraction
                "extract_tables": False,   # ❌ Remove table extraction
                "screenshot": False,       # ❌ Remove screenshot capture
                "memory_ceiling_mb": options.get("memory_ceiling_mb", 10) if options else 10,  # ✅ Optimized for memory efficiency
                "batch_size": options.get("batch_size", 50) if options else 50,
                "wait_for": "networkidle", # ✅ Wait for content to load
                "timeout": 30000,          # ✅ 30 second timeout
                "user_agent": "Mozilla/5.0 (compatible; PyRAG/1.0)",  # ✅ Proper UA
            }
            
            # Override with any provided options
            if options:
                crawl_options.update(options)
            
            logger.info(f"   - Max pages: {crawl_options['max_pages']}")
            logger.info(f"   - Memory limit: {crawl_options['memory_ceiling_mb']}MB")
            logger.info(f"   - Batch size: {crawl_options['batch_size']}")
            
            # Use async iterator for memory efficiency
            documents = []
            async for result in self.crawler.arun_many(
                urls=[f"{start_url}/*"],
                **crawl_options
            ):
                # Extract content - OPTIMIZED
                if hasattr(result, 'markdown') and result.markdown:
                    content = result.markdown
                    markdown = result.markdown
                else:
                    # Fallback: if markdown extraction failed, try HTML
                    if hasattr(result, 'html') and result.html:
                        content = self._clean_html_content(result.html)
                        markdown = self._html_to_markdown(result.html)
                    else:
                        content = "No content extracted"
                        markdown = "# No Content\n\nNo content could be extracted from this page."
                
                # Extract title
                title = getattr(result, 'title', result.url) or result.url
                
                # Build metadata - OPTIMIZED (minimal metadata)
                metadata = {
                    "url": result.url,
                    "title": title,
                    "content_length": len(content),
                    "scraped_at": asyncio.get_event_loop().time(),
                    "cache_hit": False,
                    "cache_age_hours": None,
                    "source": "crawl4ai",
                    "extractor": "local_optimized"
                }
                
                # Add any additional metadata from the result
                if hasattr(result, 'metadata'):
                    metadata.update(result.metadata)
                
                document = ScrapedDocument(
                    url=result.url,
                    title=title,
                    content=content,
                    markdown=markdown,
                    metadata=metadata,
                    screenshot_url=None  # ❌ No screenshots
                )
                
                documents.append(document)
                logger.info(f"   ✅ Scraped: {result.url} ({len(content)} chars)")
            
            logger.info(f"✅ Site crawling completed: {len(documents)} pages scraped")
            return documents
            
        except Exception as e:
            logger.error(f"❌ Error crawling site {start_url}: {e}")
            return []

    def _clean_html_content(self, html: str) -> str:
        """Clean HTML content using BeautifulSoup.
        
        Args:
            html: Raw HTML content
            
        Returns:
            Cleaned text content
        """
        try:
            from bs4 import BeautifulSoup
            
            # Parse HTML
            soup = BeautifulSoup(html, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style", "nav", "footer", "header"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up whitespace
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
            
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML cleaning")
            # Basic HTML tag removal
            import re
            text = re.sub(r'<[^>]+>', '', html)
            text = re.sub(r'\s+', ' ', text).strip()
            return text

    def _html_to_markdown(self, html: str) -> str:
        """Convert HTML to markdown.
        
        Args:
            html: HTML content
            
        Returns:
            Markdown content
        """
        try:
            from bs4 import BeautifulSoup
            
            soup = BeautifulSoup(html, 'html.parser')
            
            # Simple markdown conversion
            markdown = ""
            
            # Handle headings
            for i in range(1, 7):
                for heading in soup.find_all(f'h{i}'):
                    markdown += f"{'#' * i} {heading.get_text().strip()}\n\n"
            
            # Handle paragraphs
            for p in soup.find_all('p'):
                markdown += f"{p.get_text().strip()}\n\n"
            
            # Handle lists
            for ul in soup.find_all('ul'):
                for li in ul.find_all('li'):
                    markdown += f"- {li.get_text().strip()}\n"
                markdown += "\n"
            
            for ol in soup.find_all('ol'):
                for i, li in enumerate(ol.find_all('li'), 1):
                    markdown += f"{i}. {li.get_text().strip()}\n"
                markdown += "\n"
            
            # Handle code blocks
            for code in soup.find_all('code'):
                if code.parent.name == 'pre':
                    markdown += f"```\n{code.get_text().strip()}\n```\n\n"
                else:
                    markdown += f"`{code.get_text().strip()}`"
            
            # Handle links
            for a in soup.find_all('a'):
                text = a.get_text().strip()
                href = a.get('href', '')
                if text and href:
                    markdown += f"[{text}]({href})\n\n"
            
            return markdown.strip()
            
        except ImportError:
            logger.warning("BeautifulSoup not available, using basic HTML to markdown")
            # Basic conversion
            import re
            # Remove HTML tags
            text = re.sub(r'<[^>]+>', '', html)
            # Clean up whitespace
            text = re.sub(r'\s+', ' ', text).strip()
            return f"# Content\n\n{text}"

    def is_cached_result(self, document: ScrapedDocument) -> bool:
        """Check if a document result was cached.
        
        Crawl4AI doesn't have caching like Firecrawl, so this always returns False.
        """
        return False

    def get_cache_age_hours(self, document: ScrapedDocument) -> Optional[float]:
        """Get the age of cached content in hours.
        
        Crawl4AI doesn't have caching like Firecrawl, so this always returns None.
        """
        return None

    def get_cache_info(self, document: ScrapedDocument) -> Dict[str, Any]:
        """Get comprehensive cache information for a document.
        
        Crawl4AI doesn't have caching like Firecrawl, so this returns static info.
        """
        return {
            "is_cached": False,
            "cache_age_hours": None,
            "cache_freshness": "fresh",
            "url": document.url,
            "source": "crawl4ai"
        }

    async def health_check(self) -> bool:
        """Check if Crawl4AI is working properly."""
        try:
            # Try to create a simple crawler instance
            from crawl4ai import AsyncWebCrawler
            test_crawler = AsyncWebCrawler()
            logger.info("✅ Crawl4AI health check passed")
            return True
        except Exception as e:
            logger.error(f"❌ Crawl4AI health check failed: {e}")
            return False
