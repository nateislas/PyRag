"""Firecrawl client for documentation scraping."""

import asyncio
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import aiohttp

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class ScrapedDocument:
    """Represents a scraped document from Firecrawl."""

    url: str
    title: str
    content: str
    markdown: str
    metadata: Dict[str, Any]
    screenshot_url: Optional[str] = None


class FirecrawlClient:
    """Client for interacting with Firecrawl API for documentation scraping."""

    def __init__(
        self, api_key: Optional[str] = None, base_url: str = "https://api.firecrawl.dev"
    ):
        self.api_key = api_key
        self.base_url = base_url
        self.session: Optional[aiohttp.ClientSession] = None
        logger.info("Initialized Firecrawl client")

        # Set default headers
        self.headers = {"Content-Type": "application/json"}
        if self.api_key:
            # Try different authorization formats
            self.headers["Authorization"] = f"Bearer {self.api_key}"
            # Also try without "Bearer" prefix
            self.headers["X-API-Key"] = self.api_key

        # API endpoints - using the correct Firecrawl API
        self.extract_endpoint = "/v2/extract"
        self.scrape_endpoint = "/v2/scrape"  # Add scrape endpoint for caching support

    async def __aenter__(self):
        """Async context manager entry."""
        self.session = aiohttp.ClientSession()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self.session:
            await self.session.close()

    async def scrape_url(
        self, url: str, options: Optional[Dict[str, Any]] = None
    ) -> ScrapedDocument:
        """Scrape a single URL using Firecrawl scrape API with caching support."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Enhanced caching strategy for documentation
        # Use 1 week cache for documentation (604800000 ms) - much more aggressive
        # This provides 500% faster response times for most documentation
        default_max_age = 604800000  # 1 week in milliseconds
        max_age = options.get("maxAge", default_max_age) if options else default_max_age
        
        # Add cache optimization for documentation sites
        cache_options = {
            "maxAge": max_age,
            "storeInCache": True,
        }

        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            **cache_options
        }

        # Log caching strategy
        cache_duration_hours = max_age / (1000 * 60 * 60)
        logger.info(f"🌐 Scraping {url} with {cache_duration_hours:.1f}h cache (Firecrawl will return cached if available)")

        # Retries with exponential backoff and jitter for resilience
        max_attempts = 5
        base_delay = 2.0
        import random, asyncio
        for attempt in range(1, max_attempts + 1):
            try:
                logger.info(f"Scraping content from URL: {url}")
                async with self.session.post(
                    f"{self.base_url}{self.scrape_endpoint}",
                    json=payload,
                    headers=self.headers,
                ) as response:
                    if response.status == 200:
                        data = await response.json()
                        return self._parse_scrape_response(data, url)
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Firecrawl API error: {response.status} - {error_text}"
                        )
                        # Handle 429 with small wait; Firecrawl returns a reset ETA in text
                        if response.status in (429, 502, 503, 504):
                            # Backoff with jitter
                            delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                            # Cap delay to a reasonable bound
                            delay = min(delay, 30.0)
                            logger.info(f"Retrying {url} in {delay:.1f}s (attempt {attempt}/{max_attempts})")
                            await asyncio.sleep(delay)
                            continue
                        raise Exception(f"Firecrawl API error: {response.status}")
            except Exception as e:
                if attempt >= max_attempts:
                    logger.error(f"Error extracting content from {url} after {attempt} attempts: {e}")
                    raise
                delay = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
                delay = min(delay, 30.0)
                logger.warning(f"Error scraping {url}: {e}. Retrying in {delay:.1f}s (attempt {attempt}/{max_attempts})")
                await asyncio.sleep(delay)

    async def crawl_site(
        self, start_url: str, options: Optional[Dict[str, Any]] = None
    ) -> List[ScrapedDocument]:
        """Crawl an entire site using Firecrawl extract API."""
        if not self.session:
            raise RuntimeError("Client not initialized. Use async context manager.")

        # Create a schema for extracting documentation content from multiple pages
        schema = {
            "type": "object",
            "properties": {
                "pages": {
                    "type": "array",
                    "items": {
                        "type": "object",
                        "properties": {
                            "url": {"type": "string"},
                            "title": {"type": "string"},
                            "content": {"type": "string"},
                            "markdown": {"type": "string"},
                            "links": {"type": "array", "items": {"type": "string"}},
                            "images": {"type": "array", "items": {"type": "string"}},
                        },
                        "required": ["url", "title", "content"],
                    },
                }
            },
            "required": ["pages"],
        }

        # Create prompt for documentation extraction
        prompt = f"Extract documentation content from all pages on this site. For each page, include the URL, title, main content as plain text, content as markdown, and any relevant links and images."

        payload = {
            "urls": [f"{start_url}/*"],  # Use wildcard to crawl entire site
            "prompt": prompt,
            "schema": schema,
        }

        try:
            logger.info(f"Extracting content from site: {start_url}")
            async with self.session.post(
                f"{self.base_url}{self.extract_endpoint}",
                json=payload,
                headers=self.headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()
                    return self._parse_extract_site_response(data)
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Firecrawl API error: {response.status} - {error_text}"
                    )
                    raise Exception(f"Firecrawl API error: {response.status}")

        except Exception as e:
            logger.error(f"Error extracting content from {start_url}: {e}")
            raise

    def _parse_extract_response(
        self, data: Dict[str, Any], url: str
    ) -> ScrapedDocument:
        """Parse Firecrawl extract response for single page."""
        try:
            # Handle extract API response format
            if "data" in data:
                extracted_data = data["data"]
            else:
                extracted_data = data

            return ScrapedDocument(
                url=url,
                title=extracted_data.get("title", ""),
                content=extracted_data.get("content", ""),
                markdown=extracted_data.get("markdown", ""),
                metadata={
                    "links": extracted_data.get("links", []),
                    "images": extracted_data.get("images", []),
                    "metadata": extracted_data.get("metadata", {}),
                },
                screenshot_url=None,  # Extract API doesn't provide screenshots
            )
        except Exception as e:
            logger.error(f"Error parsing extract response: {e}")
            # Return minimal document
            return ScrapedDocument(
                url=url, title="", content="", markdown="", metadata={}
            )

    def _parse_extract_site_response(
        self, data: Dict[str, Any]
    ) -> List[ScrapedDocument]:
        """Parse Firecrawl extract response for multiple pages."""
        documents = []

        try:
            # Handle extract API response format
            if "data" in data:
                extracted_data = data["data"]
            else:
                extracted_data = data

            # Extract pages from the response
            pages = extracted_data.get("pages", [])

            for page in pages:
                doc = ScrapedDocument(
                    url=page.get("url", ""),
                    title=page.get("title", ""),
                    content=page.get("content", ""),
                    markdown=page.get("markdown", ""),
                    metadata={
                        "links": page.get("links", []),
                        "images": page.get("images", []),
                        "metadata": page.get("metadata", {}),
                    },
                    screenshot_url=None,
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing extract site response: {e}")

        return documents

    def _parse_crawl_response(self, data: Dict[str, Any]) -> List[ScrapedDocument]:
        """Parse Firecrawl crawl response."""
        documents = []

        try:
            # Handle different response formats
            if "data" in data:
                pages = data["data"]
            else:
                pages = data

            for page in pages:
                doc = ScrapedDocument(
                    url=page.get("url", ""),
                    title=page.get("title", ""),
                    content=page.get("text", ""),
                    markdown=page.get("markdown", ""),
                    metadata={
                        "links": page.get("links", []),
                        "images": page.get("images", []),
                        "metadata": page.get("metadata", {}),
                    },
                    screenshot_url=page.get("screenshot", None),
                )
                documents.append(doc)

        except Exception as e:
            logger.error(f"Error parsing crawl response: {e}")

        return documents

    async def _poll_job_status(
        self, job_id: str, url: str, max_attempts: int = 60, delay: float = 3.0
    ) -> ScrapedDocument:
        """Poll job status until completion."""
        for attempt in range(max_attempts):
            try:
                async with self.session.get(
                    f"{self.base_url}/v2/extract/{job_id}", headers=self.headers
                ) as response:
                    if response.status == 200:
                        data = await response.json()

                        if data.get("status") == "completed":
                            logger.info(f"Job {job_id} completed successfully")
                            return self._parse_extract_response(data, url)
                        elif data.get("status") == "failed":
                            error_msg = data.get("error", "Unknown error")
                            logger.error(f"Job {job_id} failed: {error_msg}")
                            raise Exception(f"Extraction job failed: {error_msg}")
                        elif data.get("status") == "cancelled":
                            logger.error(f"Job {job_id} was cancelled")
                            raise Exception("Extraction job was cancelled")
                        else:
                            # Still processing
                            logger.info(
                                f"Job {job_id} still processing (attempt {attempt + 1}/{max_attempts})"
                            )
                            await asyncio.sleep(delay)
                    else:
                        error_text = await response.text()
                        logger.error(
                            f"Error checking job status: {response.status} - {error_text}"
                        )
                        raise Exception(f"Job status check failed: {response.status}")

            except Exception as e:
                if attempt == max_attempts - 1:
                    raise e
                logger.warning(f"Error polling job status (attempt {attempt + 1}): {e}")
                await asyncio.sleep(delay)

        raise Exception(
            f"Job {job_id} did not complete within {max_attempts * delay} seconds"
        )

    def _parse_scrape_response(self, data: Dict[str, Any], url: str) -> ScrapedDocument:
        """Parse the response from Firecrawl scrape API."""
        try:
            # Check if this was a cache hit
            is_cached = data.get("cached", False)
            cache_age = data.get("cacheAge", 0)
            
            if is_cached:
                cache_age_hours = cache_age / (1000 * 60 * 60)
                logger.info(f"⚡ CACHE HIT for {url} (age: {cache_age_hours:.1f}h) - Instant response!")
            else:
                logger.info(f"🆕 Fresh scrape for {url} - Stored in cache for future use")
            
            # Extract content - handle v2 API response format
            content = data.get("markdown", "") or data.get("html", "")
            title = data.get("title", url)
            
            # Extract metadata
            metadata = data.get("metadata", {})
            if not metadata:
                metadata = {
                    "url": url,
                    "title": title,
                    "content_length": len(content),
                    "scraped_at": data.get("timestamp"),
                    "cache_hit": is_cached,
                    "cache_age_hours": cache_age_hours if is_cached else None
                }
            
            return ScrapedDocument(
                url=url,
                title=title,
                content=content,
                markdown=data.get("markdown", ""),
                metadata=metadata,
                screenshot_url=data.get("screenshot")
            )
            
        except Exception as e:
            logger.error(f"Error parsing Firecrawl response for {url}: {e}")
            # Return a minimal document on error instead of raising
            return ScrapedDocument(
                url=url,
                title="Error",
                content=f"Failed to parse content: {e}",
                markdown=f"# Error\n\nFailed to parse content: {e}",
                metadata={"error": str(e), "cache_hit": False},
                screenshot_url=None,
            )

    def is_cached_result(self, document: ScrapedDocument) -> bool:
        """Check if a document result was cached."""
        return document.metadata.get("cache_hit", False)
    
    def get_cache_age_hours(self, document: ScrapedDocument) -> Optional[float]:
        """Get the age of cached content in hours."""
        return document.metadata.get("cache_age_hours")
    
    def get_cache_info(self, document: ScrapedDocument) -> Dict[str, Any]:
        """Get comprehensive cache information for a document."""
        return {
            "is_cached": self.is_cached_result(document),
            "cache_age_hours": self.get_cache_age_hours(document),
            "cache_freshness": "fresh" if not self.is_cached_result(document) else "cached",
            "url": document.url
        }

    async def health_check(self) -> bool:
        """Check if Firecrawl API is accessible."""
        try:
            if not self.session:
                return False

            # Test with a simple extract request
            test_payload = {
                "urls": ["https://example.com"],
                "prompt": "Extract the page title",
                "schema": {
                    "type": "object",
                    "properties": {"title": {"type": "string"}},
                    "required": ["title"],
                },
            }

            async with self.session.post(
                f"{self.base_url}{self.extract_endpoint}",
                json=test_payload,
                headers=self.headers,
            ) as response:
                # Even if it fails, if we get a proper error response, the API is accessible
                return response.status in [200, 400, 401, 422]  # Valid API responses
        except Exception as e:
            logger.error(f"Firecrawl health check failed: {e}")
            return False
