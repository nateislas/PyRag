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

        # Use scrape API for better caching support
        # Default to 1 day cache for documentation (86400000 ms)
        max_age = options.get("maxAge", 86400000) if options else 86400000

        payload = {
            "url": url,
            "formats": ["markdown", "html"],
            "maxAge": max_age,  # Use cached data if less than 1 day old
            "storeInCache": True,  # Store results for future use
        }

        try:
            logger.info(f"Scraping content from URL: {url}")
            async with self.session.post(
                f"{self.base_url}{self.scrape_endpoint}",
                json=payload,
                headers=self.headers,
            ) as response:
                if response.status == 200:
                    data = await response.json()

                    # Parse scrape API response format
                    return self._parse_scrape_response(data, url)
                else:
                    error_text = await response.text()
                    logger.error(
                        f"Firecrawl API error: {response.status} - {error_text}"
                    )
                    raise Exception(f"Firecrawl API error: {response.status}")

        except Exception as e:
            logger.error(f"Error extracting content from {url}: {e}")
            raise

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
        """Parse Firecrawl scrape API response format."""
        try:
            # Handle scrape API response format
            if "data" in data:
                scraped_data = data["data"]
            else:
                scraped_data = data

            # Extract content from the response
            title = scraped_data.get("title", "")
            content = scraped_data.get("text", "") or scraped_data.get("content", "")
            markdown = scraped_data.get("markdown", "")

            # Extract links and images from HTML if available
            links = []
            images = []
            if "html" in scraped_data:
                # Simple HTML parsing for links and images
                html_content = scraped_data["html"]
                # Extract links (basic regex for href attributes)
                import re

                link_matches = re.findall(r'href=["\']([^"\']+)["\']', html_content)
                links = [link for link in link_matches if link.startswith("http")]

                # Extract images (basic regex for src attributes)
                img_matches = re.findall(r'src=["\']([^"\']+)["\']', html_content)
                images = [img for img in img_matches if img.startswith("http")]

            return ScrapedDocument(
                url=url,
                title=title,
                content=content,
                markdown=markdown,
                metadata={
                    "links": links,
                    "images": images,
                    "metadata": scraped_data.get("metadata", {}),
                    "cached": scraped_data.get(
                        "cached", False
                    ),  # Track if result was cached
                },
                screenshot_url=scraped_data.get("screenshot", None),
            )

        except Exception as e:
            logger.error(f"Error parsing scrape response: {e}")
            # Return a minimal document on error
            return ScrapedDocument(
                url=url,
                title="Error",
                content=f"Failed to parse content: {e}",
                markdown=f"# Error\n\nFailed to parse content: {e}",
                metadata={"error": str(e)},
                screenshot_url=None,
            )

    def is_cached_result(self, document: ScrapedDocument) -> bool:
        """Check if a document result was served from cache."""
        return document.metadata.get("cached", False)

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
