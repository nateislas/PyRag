"""Enhanced sitemap analyzer for comprehensive documentation discovery."""

import asyncio
import re
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse, urlunparse

import aiohttp
from bs4 import BeautifulSoup

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class SitemapEntry:
    """Represents a single sitemap entry."""

    url: str
    last_modified: Optional[str] = None
    change_frequency: Optional[str] = None
    priority: Optional[float] = None
    content_type: Optional[str] = None
    estimated_size: Optional[int] = None


@dataclass
class SitemapAnalysis:
    """Result of sitemap analysis."""

    base_url: str
    sitemap_urls: List[str]
    discovered_urls: List[SitemapEntry]
    documentation_urls: List[SitemapEntry]
    api_urls: List[SitemapEntry]
    tutorial_urls: List[SitemapEntry]
    example_urls: List[SitemapEntry]
    other_urls: List[SitemapEntry]
    total_urls: int
    coverage_estimate: float
    structure_insights: Dict[str, Any]


class SitemapAnalyzer:
    """Comprehensive sitemap analyzer for documentation sites."""

    def __init__(
        self,
        session: Optional[aiohttp.ClientSession] = None,
        timeout: int = 30,
        max_redirects: int = 5,
    ):
        self.session = session
        self.timeout = timeout
        self.max_redirects = max_redirects
        self.logger = get_logger(__name__)

        # Common sitemap locations
        self.sitemap_paths = [
            "/sitemap.xml",
            "/sitemap_index.xml",
            "/sitemap/sitemap.xml",
            "/docs/sitemap.xml",
            "/documentation/sitemap.xml",
            "/api/sitemap.xml",
        ]

        # Documentation URL patterns
        self.doc_patterns = {
            "api": [
                r"/api/",
                r"/reference/",
                r"/docs/api/",
                r"/documentation/api/",
                r"/v\d+/",
                r"/latest/api/",
            ],
            "tutorial": [
                r"/tutorial/",
                r"/guide/",
                r"/getting-started/",
                r"/docs/tutorial/",
                r"/documentation/guide/",
            ],
            "example": [
                r"/example/",
                r"/examples/",
                r"/sample/",
                r"/docs/example/",
                r"/documentation/examples/",
            ],
            "documentation": [
                r"/docs/",
                r"/documentation/",
                r"/help/",
                r"/manual/",
                r"/guide/",
            ],
        }

    async def analyze_documentation_site(self, base_url: str) -> SitemapAnalysis:
        """Analyze a documentation site for comprehensive coverage."""
        self.logger.info(f"Starting comprehensive sitemap analysis for: {base_url}")

        # Discover sitemaps
        sitemap_urls = await self._discover_sitemaps(base_url)
        self.logger.info(f"Discovered {len(sitemap_urls)} sitemap(s)")

        # Parse all sitemaps
        all_entries = []
        for sitemap_url in sitemap_urls:
            entries = await self._parse_sitemap(sitemap_url)
            all_entries.extend(entries)
            self.logger.info(f"Parsed {len(entries)} URLs from {sitemap_url}")

        # Remove duplicates and filter to documentation domain
        unique_entries = self._deduplicate_entries(all_entries)
        doc_entries = self._filter_documentation_urls(unique_entries, base_url)

        # Classify URLs by type
        classified_entries = self._classify_urls(doc_entries)

        # Analyze structure and coverage
        structure_insights = self._analyze_structure(classified_entries, base_url)
        coverage_estimate = self._estimate_coverage(classified_entries, base_url)

        analysis = SitemapAnalysis(
            base_url=base_url,
            sitemap_urls=sitemap_urls,
            discovered_urls=doc_entries,
            documentation_urls=classified_entries.get("documentation", []),
            api_urls=classified_entries.get("api", []),
            tutorial_urls=classified_entries.get("tutorial", []),
            example_urls=classified_entries.get("example", []),
            other_urls=classified_entries.get("other", []),
            total_urls=len(doc_entries),
            coverage_estimate=coverage_estimate,
            structure_insights=structure_insights,
        )

        self.logger.info(
            f"Analysis complete: {analysis.total_urls} URLs, "
            f"{coverage_estimate:.1%} coverage estimate"
        )

        return analysis

    async def _discover_sitemaps(self, base_url: str) -> List[str]:
        """Discover sitemap URLs for a given base URL."""
        discovered_sitemaps = []

        # Check common sitemap locations
        for path in self.sitemap_paths:
            sitemap_url = urljoin(base_url, path)
            if await self._check_url_exists(sitemap_url):
                discovered_sitemaps.append(sitemap_url)
                self.logger.debug(f"Found sitemap at: {sitemap_url}")

        # Check robots.txt for sitemap references
        robots_url = urljoin(base_url, "/robots.txt")
        robots_sitemaps = await self._extract_sitemaps_from_robots(robots_url)
        discovered_sitemaps.extend(robots_sitemaps)

        # Check HTML for sitemap links
        html_sitemaps = await self._extract_sitemaps_from_html(base_url)
        discovered_sitemaps.extend(html_sitemaps)

        # Remove duplicates
        discovered_sitemaps = list(set(discovered_sitemaps))

        # Expand sitemap indexes recursively
        expanded_sitemaps = []
        for sitemap_url in discovered_sitemaps:
            try:
                # Check if this is a sitemap index
                async with self.session.get(sitemap_url) as response:
                    if response.status == 200:
                        content = await response.text()
                        if "<sitemapindex" in content:
                            # This is a sitemap index, extract nested sitemaps
                            nested_sitemaps = self._extract_sitemap_index_urls(content)
                            expanded_sitemaps.extend(nested_sitemaps)
                            self.logger.debug(
                                f"Expanded sitemap index {sitemap_url} to {len(nested_sitemaps)} sitemaps"
                            )
                        else:
                            # This is a regular sitemap
                            expanded_sitemaps.append(sitemap_url)
            except Exception as e:
                self.logger.debug(
                    f"Failed to check sitemap type for {sitemap_url}: {e}"
                )
                # Assume it's a regular sitemap if we can't determine
                expanded_sitemaps.append(sitemap_url)

        return expanded_sitemaps

    async def _check_url_exists(self, url: str) -> bool:
        """Check if a URL exists and is accessible."""
        if not self.session:
            return False

        try:
            async with self.session.head(url, allow_redirects=True) as response:
                return response.status == 200
        except Exception as e:
            self.logger.debug(f"Failed to check URL {url}: {e}")
            return False

    async def _extract_sitemaps_from_robots(self, robots_url: str) -> List[str]:
        """Extract sitemap URLs from robots.txt file."""
        if not self.session:
            return []

        try:
            async with self.session.get(robots_url) as response:
                if response.status == 200:
                    content = await response.text()
                    sitemaps = []
                    for line in content.split("\n"):
                        if line.lower().startswith("sitemap:"):
                            sitemap_url = line.split(":", 1)[1].strip()
                            sitemaps.append(sitemap_url)
                    return sitemaps
        except Exception as e:
            self.logger.debug(f"Failed to parse robots.txt at {robots_url}: {e}")

        return []

    async def _extract_sitemaps_from_html(self, base_url: str) -> List[str]:
        """Extract sitemap URLs from HTML head section."""
        if not self.session:
            return []

        try:
            async with self.session.get(base_url) as response:
                if response.status == 200:
                    content = await response.text()
                    soup = BeautifulSoup(content, "html.parser")

                    sitemaps = []
                    # Look for sitemap links in head
                    for link in soup.find_all("link", rel="sitemap"):
                        href = link.get("href")
                        if href:
                            sitemap_url = urljoin(base_url, href)
                            sitemaps.append(sitemap_url)

                    # Also check for sitemap meta tags
                    for meta in soup.find_all("meta", attrs={"name": "sitemap"}):
                        content = meta.get("content")
                        if content:
                            sitemap_url = urljoin(base_url, content)
                            sitemaps.append(sitemap_url)

                    return sitemaps
        except Exception as e:
            self.logger.debug(f"Failed to parse HTML at {base_url}: {e}")

        return []

    async def _parse_sitemap(self, sitemap_url: str) -> List[SitemapEntry]:
        """Parse a sitemap XML file."""
        if not self.session:
            return []

        try:
            async with self.session.get(sitemap_url) as response:
                if response.status == 200:
                    content = await response.text()
                    return self._parse_sitemap_xml(content, sitemap_url)
        except Exception as e:
            self.logger.error(f"Failed to parse sitemap {sitemap_url}: {e}")

        return []

    def _parse_sitemap_xml(
        self, xml_content: str, sitemap_url: str
    ) -> List[SitemapEntry]:
        """Parse XML content and extract sitemap entries."""
        entries = []

        try:
            root = ET.fromstring(xml_content)

            # Handle both sitemap index and regular sitemaps
            if "sitemapindex" in root.tag:
                # This is a sitemap index
                for sitemap in root.findall(".//{*}sitemap"):
                    loc = sitemap.find("{*}loc")
                    if loc is not None and loc.text:
                        # Recursively parse nested sitemaps
                        # Note: This would need async handling in practice
                        pass
            else:
                # Regular sitemap
                for url in root.findall(".//{*}url"):
                    entry = self._extract_sitemap_entry(url, sitemap_url)
                    if entry:
                        entries.append(entry)

        except ET.ParseError as e:
            self.logger.error(f"Failed to parse XML from {sitemap_url}: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing sitemap {sitemap_url}: {e}")

        return entries

    def _extract_sitemap_index_urls(self, xml_content: str) -> List[str]:
        """Extract sitemap URLs from a sitemap index."""
        sitemap_urls = []

        try:
            root = ET.fromstring(xml_content)

            # Look for sitemap elements in sitemap index
            for sitemap in root.findall(".//{*}sitemap"):
                loc = sitemap.find("{*}loc")
                if loc is not None and loc.text:
                    sitemap_urls.append(loc.text.strip())

        except ET.ParseError as e:
            self.logger.error(f"Failed to parse sitemap index XML: {e}")
        except Exception as e:
            self.logger.error(f"Unexpected error parsing sitemap index: {e}")

        return sitemap_urls

    def _extract_sitemap_entry(
        self, url_elem, sitemap_url: str
    ) -> Optional[SitemapEntry]:
        """Extract a single sitemap entry from XML."""
        try:
            loc = url_elem.find("{*}loc")
            if loc is None or not loc.text:
                return None

            url = loc.text.strip()

            # Extract optional fields
            last_modified = None
            lastmod_elem = url_elem.find("{*}lastmod")
            if lastmod_elem is not None and lastmod_elem.text:
                last_modified = lastmod_elem.text.strip()

            change_frequency = None
            changefreq_elem = url_elem.find("{*}changefreq")
            if changefreq_elem is not None and changefreq_elem.text:
                change_frequency = changefreq_elem.text.strip()

            priority = None
            priority_elem = url_elem.find("{*}priority")
            if priority_elem is not None and priority_elem.text:
                try:
                    priority = float(priority_elem.text.strip())
                except ValueError:
                    pass

            return SitemapEntry(
                url=url,
                last_modified=last_modified,
                change_frequency=change_frequency,
                priority=priority,
            )

        except Exception as e:
            self.logger.debug(f"Failed to extract sitemap entry: {e}")
            return None

    def _deduplicate_entries(self, entries: List[SitemapEntry]) -> List[SitemapEntry]:
        """Remove duplicate URLs from sitemap entries."""
        seen_urls = set()
        unique_entries = []

        for entry in entries:
            if entry.url not in seen_urls:
                seen_urls.add(entry.url)
                unique_entries.append(entry)

        return unique_entries

    def _filter_documentation_urls(
        self, entries: List[SitemapEntry], base_url: str
    ) -> List[SitemapEntry]:
        """Filter URLs to only include documentation-related content."""
        base_domain = urlparse(base_url).netloc
        filtered_entries = []

        for entry in entries:
            entry_domain = urlparse(entry.url).netloc

            # Include if same domain or subdomain
            if entry_domain == base_domain or entry_domain.endswith(f".{base_domain}"):
                # Check if it looks like documentation
                if self._is_documentation_url(entry.url):
                    filtered_entries.append(entry)

        return filtered_entries

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

        # Include if it matches documentation patterns
        for doc_type, patterns in self.doc_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return True

        # Also include if it's on the main domain and has a path
        parsed = urlparse(url)
        if parsed.path and len(parsed.path) > 1:
            return True

        return False

    def _classify_urls(
        self, entries: List[SitemapEntry]
    ) -> Dict[str, List[SitemapEntry]]:
        """Classify URLs by their content type."""
        classified = {
            "api": [],
            "tutorial": [],
            "example": [],
            "documentation": [],
            "other": [],
        }

        for entry in entries:
            url_lower = entry.url.lower()
            classified_flag = False

            # Classify by URL patterns
            for doc_type, patterns in self.doc_patterns.items():
                for pattern in patterns:
                    if re.search(pattern, url_lower):
                        classified[doc_type].append(entry)
                        classified_flag = True
                        break
                if classified_flag:
                    break

            if not classified_flag:
                classified["other"].append(entry)

        return classified

    def _analyze_structure(
        self, classified_entries: Dict[str, List[SitemapEntry]], base_url: str
    ) -> Dict[str, Any]:
        """Analyze the documentation structure and provide insights."""
        insights = {
            "total_pages": sum(len(entries) for entries in classified_entries.values()),
            "by_type": {
                doc_type: len(entries)
                for doc_type, entries in classified_entries.items()
            },
            "depth_analysis": self._analyze_depth_distribution(
                classified_entries, base_url
            ),
            "coverage_gaps": self._identify_coverage_gaps(classified_entries, base_url),
            "recommendations": [],
        }

        # Generate recommendations
        if insights["by_type"]["api"] < 10:
            insights["recommendations"].append(
                "Low API documentation coverage - consider manual review"
            )

        if insights["by_type"]["tutorial"] < 5:
            insights["recommendations"].append(
                "Limited tutorial content - may need additional discovery"
            )

        if insights["by_type"]["example"] < 3:
            insights["recommendations"].append(
                "Few examples found - consider manual example discovery"
            )

        return insights

    def _analyze_depth_distribution(
        self, classified_entries: Dict[str, List[SitemapEntry]], base_url: str
    ) -> Dict[str, int]:
        """Analyze the depth distribution of documentation URLs."""
        depth_counts = {"shallow": 0, "medium": 0, "deep": 0}

        for entries in classified_entries.values():
            for entry in entries:
                parsed = urlparse(entry.url)
                path_parts = [p for p in parsed.path.split("/") if p]

                if len(path_parts) <= 2:
                    depth_counts["shallow"] += 1
                elif len(path_parts) <= 4:
                    depth_counts["medium"] += 1
                else:
                    depth_counts["deep"] += 1

        return depth_counts

    def _identify_coverage_gaps(
        self, classified_entries: Dict[str, List[SitemapEntry]], base_url: str
    ) -> List[str]:
        """Identify potential coverage gaps in the documentation."""
        gaps = []

        # Check for missing common documentation sections
        if not classified_entries["api"]:
            gaps.append("No API reference documentation found")

        if not classified_entries["tutorial"]:
            gaps.append("No tutorial or getting started content found")

        if not classified_entries["example"]:
            gaps.append("No code examples found")

        # Check depth distribution
        depth_analysis = self._analyze_depth_distribution(classified_entries, base_url)
        if depth_analysis["deep"] == 0:
            gaps.append(
                "No deep documentation content found - may be missing detailed guides"
            )

        if (
            depth_analysis["shallow"]
            > depth_analysis["medium"] + depth_analysis["deep"]
        ):
            gaps.append(
                "Heavy concentration of shallow content - may be missing detailed documentation"
            )

        return gaps

    def _estimate_coverage(
        self, classified_entries: Dict[str, List[SitemapEntry]], base_url: str
    ) -> float:
        """Estimate the documentation coverage based on discovered content."""
        total_pages = sum(len(entries) for entries in classified_entries.values())

        if total_pages == 0:
            return 0.0

        # Weight different content types
        weights = {
            "api": 0.4,  # API docs are most important
            "tutorial": 0.3,  # Tutorials provide context
            "example": 0.2,  # Examples show usage
            "documentation": 0.1,  # General docs
            "other": 0.05,  # Other content
        }

        weighted_score = 0.0
        for doc_type, entries in classified_entries.items():
            weight = weights.get(doc_type, 0.05)
            weighted_score += len(entries) * weight

        # Normalize to 0-1 range (assuming 100+ pages is comprehensive)
        max_expected = 100
        coverage = min(weighted_score / max_expected, 1.0)

        return coverage

    async def get_crawl_recommendations(
        self, analysis: SitemapAnalysis
    ) -> Dict[str, Any]:
        """Generate crawl recommendations based on sitemap analysis."""
        recommendations = {
            "crawl_strategy": "comprehensive",
            "priority_urls": [],
            "depth_limits": {},
            "content_filters": {},
            "estimated_duration": 0,
            "resource_requirements": {},
        }

        # Set crawl strategy based on coverage
        if analysis.coverage_estimate < 0.3:
            recommendations["crawl_strategy"] = "aggressive"
        elif analysis.coverage_estimate < 0.7:
            recommendations["crawl_strategy"] = "balanced"
        else:
            recommendations["crawl_strategy"] = "selective"

        # Prioritize URLs by type
        if analysis.api_urls:
            recommendations["priority_urls"].extend(
                [url.url for url in analysis.api_urls[:20]]
            )

        if analysis.tutorial_urls:
            recommendations["priority_urls"].extend(
                [url.url for url in analysis.tutorial_urls[:10]]
            )

        # Set depth limits based on structure
        depth_analysis = analysis.structure_insights["depth_analysis"]
        if depth_analysis["deep"] > 0:
            recommendations["depth_limits"]["max_depth"] = 5
        elif depth_analysis["medium"] > 0:
            recommendations["depth_limits"]["max_depth"] = 4
        else:
            recommendations["depth_limits"]["max_depth"] = 3

        # Estimate duration (rough calculation)
        total_pages = analysis.total_urls
        if recommendations["crawl_strategy"] == "aggressive":
            recommendations["estimated_duration"] = (
                total_pages * 2
            )  # 2 seconds per page
        elif recommendations["crawl_strategy"] == "balanced":
            recommendations["estimated_duration"] = (
                total_pages * 3
            )  # 3 seconds per page
        else:
            recommendations["estimated_duration"] = (
                total_pages * 5
            )  # 5 seconds per page

        # Resource requirements
        recommendations["resource_requirements"] = {
            "concurrent_requests": 10
            if recommendations["crawl_strategy"] == "aggressive"
            else 5,
            "memory_usage_mb": total_pages * 0.5,  # Rough estimate
            "storage_gb": total_pages * 0.001,  # Rough estimate
        }

        return recommendations
