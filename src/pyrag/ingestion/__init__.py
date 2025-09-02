"""Enhanced documentation ingestion system for comprehensive coverage."""

from .documentation_manager import (
    DocumentationJob,
    DocumentationManager,
    DocumentationResult,
)
from .enhanced_documentation_processor import (
    EnhancedDocumentationProcessor,
    EnhancedProcessingResult,
)
from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .intelligent_crawler import CrawlProgress
from .intelligent_crawler import CrawlResult as IntelligentCrawlResult
from .intelligent_crawler import CrawlStrategy, IntelligentCrawler
from .sitemap_analyzer import SitemapAnalysis, SitemapAnalyzer, SitemapEntry
from .structure_mapper import (
    DocumentationNode,
    DocumentationStructure,
    DocumentationStructureMapper,
)

__all__ = [
    # Core components
    "DocumentationManager",
    "DocumentationJob",
    "DocumentationResult",
    # Enhanced processing
    "EnhancedDocumentationProcessor",
    "EnhancedProcessingResult",
    # Content extraction
    "FirecrawlClient",
    "ScrapedDocument",
    # Structure analysis
    "SitemapAnalyzer",
    "SitemapAnalysis",
    "SitemapEntry",
    "DocumentationStructureMapper",
    "DocumentationStructure",
    "DocumentationNode",
    # Intelligent crawling
    "IntelligentCrawler",
    "CrawlStrategy",
    "CrawlProgress",
    "IntelligentCrawlResult",
]
