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
from .crawl4ai_client import Crawl4AIClient, ScrapedDocument
from .intelligent_crawler import (
    CrawlProgress,
)
from .intelligent_crawler import CrawlResult as IntelligentCrawlResult
from .intelligent_crawler import (
    CrawlStrategy,
    IntelligentCrawler,
)
from .sitemap_analyzer import SitemapAnalysis, SitemapAnalyzer, SitemapEntry
from .structure_mapper import (
    DocumentationNode,
    DocumentationStructure,
    DocumentationStructureMapper,
)
from .metadata_sanitizer import (
    MetadataSanitizer,
    sanitize_metadata,
    validate_metadata,
)
from .document_structure_analyzer import (
    DocumentStructureAnalyzer,
    DocumentAnalysis,
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
    "Crawl4AIClient",
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
    # Metadata sanitization
    "MetadataSanitizer",
    "sanitize_metadata",
    "validate_metadata",
    # Document structure analysis
    "DocumentStructureAnalyzer",
    "DocumentAnalysis",
]
