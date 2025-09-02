"""LLM client for intelligent link filtering and content analysis."""

import asyncio
import os
from typing import Any, Dict, List, Optional

from openai import AsyncOpenAI

from ..config import LLMConfig
from ..logging import get_logger


# Import utility function locally to avoid circular import
def parse_llm_json_response(content: str):
    """Parse LLM JSON response with markdown code block handling."""
    import json
    import re

    # Remove markdown code blocks if present
    content = re.sub(r"```json\s*", "", content)
    content = re.sub(r"```\s*$", "", content)
    content = content.strip()

    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        # Try to extract JSON from the response
        try:
            # Look for JSON-like content
            json_match = re.search(r"\{.*\}", content, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(0))
        except:
            pass

        # If all else fails, try to parse as a simple list
        try:
            # Look for array-like content
            array_match = re.search(r"\[.*\]", content, re.DOTALL)
            if array_match:
                return json.loads(array_match.group(0))
        except:
            pass

        raise e


logger = get_logger(__name__)


class LLMClient:
    """LLM client for intelligent operations."""

    def __init__(self, config: LLMConfig):
        self.config = config
        self.client = AsyncOpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
        )
        self.logger = get_logger(__name__)

    async def filter_links(
        self, base_url: str, all_links: List[str], library_name: str
    ) -> List[str]:
        """Use LLM to filter relevant documentation links."""
        if not all_links:
            return []

        prompt = f"""
        You are an expert at identifying relevant documentation links for Python libraries.

        Base documentation URL: {base_url}
        Library name: {library_name}

        Given these discovered links, return ONLY the URLs that are:
        1. Documentation pages (guides, tutorials, API reference, examples)
        2. Same domain as the base URL
        3. NOT social media, external integrations, or non-documentation pages
        4. Relevant for developers using this library

        Links to evaluate:
        {chr(10).join(f"- {link}" for link in all_links)}

        Respond with ONLY a JSON array of relevant URLs, like:
        ["https://docs.example.com/page1", "https://docs.example.com/page2"]

        If no links are relevant, return an empty array: []
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature,
                response_format={"type": "json_object"},
            )

            content = response.choices[0].message.content
            self.logger.info(f"LLM response: {content}")

            # Parse JSON response using utility function
            result = parse_llm_json_response(content)

            # Extract URLs from response
            if isinstance(result, dict) and "urls" in result:
                filtered_links = result["urls"]
            elif isinstance(result, list):
                filtered_links = result
            else:
                self.logger.warning(f"Unexpected LLM response format: {result}")
                return []

            self.logger.info(
                f"LLM filtered {len(all_links)} links to {len(filtered_links)} relevant links"
            )
            return filtered_links

        except Exception as e:
            self.logger.error(f"LLM filtering failed: {e}")
            # Fallback to basic filtering
            return self._basic_filter_links(base_url, all_links)

    async def validate_link(self, link: str, base_url: str, library_name: str) -> bool:
        """Use LLM to validate if a single link is documentation-relevant."""

        prompt = f"""
        Given this documentation base URL: {base_url}
        Library name: {library_name}
        And this specific link: {link}

        Is this link likely to contain documentation content (guides, tutorials, API reference, etc.)?

        Consider:
        1. Is it a documentation page, not social media or external service?
        2. Is it related to the library's documentation?
        3. Would it be useful for developers using this library?

        Respond with only "yes" or "no".
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip().lower()
            return "yes" in content

        except Exception as e:
            self.logger.warning(f"LLM validation failed for {link}: {e}")
            return True  # Default to True on error

    async def analyze_content_type(
        self, url: str, title: str, content_preview: str
    ) -> str:
        """Use LLM to analyze the content type of a page."""

        prompt = f"""
        Analyze this documentation page and classify its content type:

        URL: {url}
        Title: {title}
        Content Preview: {content_preview[:200]}...

        Classify as one of:
        - "overview": Introduction, getting started, overview pages
        - "api_reference": API documentation, function/class references
        - "tutorial": Step-by-step guides, tutorials, how-to pages
        - "examples": Code examples, sample code, demonstrations
        - "guide": Detailed guides, best practices, advanced topics

        Respond with only the classification (e.g., "api_reference").
        """

        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=20,
                temperature=0.1,
            )

            content_type = response.choices[0].message.content.strip().lower()
            return content_type

        except Exception as e:
            self.logger.warning(f"LLM content type analysis failed: {e}")
            return "overview"  # Default fallback

    def _basic_filter_links(self, base_url: str, all_links: List[str]) -> List[str]:
        """Basic filtering fallback when LLM is unavailable."""
        from urllib.parse import urlparse

        base_parsed = urlparse(base_url)
        relevant_links = []

        exclude_patterns = [
            "linkedin.com",
            "twitter.com",
            "github.com",
            "facebook.com",
            "youtube.com",
            "discord.com",
            "slack.com",
            "zapier.com",
        ]

        include_patterns = [
            "/docs/",
            "/guide/",
            "/tutorial/",
            "/api/",
            "/reference/",
            "/introduction",
            "/quickstart",
            "/examples",
            "/learn/",
        ]

        for link in all_links:
            parsed = urlparse(link)

            # Must be same domain
            if parsed.netloc != base_parsed.netloc:
                continue

            # Check exclude patterns
            if any(pattern in link.lower() for pattern in exclude_patterns):
                continue

            # Check include patterns
            if any(pattern in link.lower() for pattern in include_patterns):
                relevant_links.append(link)

        return relevant_links

    async def health_check(self) -> bool:
        """Check if LLM client is working."""
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            return True
        except Exception as e:
            self.logger.error(f"LLM health check failed: {e}")
            return False
