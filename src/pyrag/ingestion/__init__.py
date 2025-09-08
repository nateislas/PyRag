"""Documentation ingestion system for comprehensive coverage."""

from .documentation_manager import (
    DocumentationJob,
    DocumentationManager,
    DocumentationResult,
)
from .document_processor import (
    DocumentProcessor,
    ProcessingResult,
)
from .crawl4ai_client import Crawl4AIClient, ScrapedDocument
from .crawler import (
    CrawlProgress,
    CrawlResult,
    CrawlStrategy,
    Crawler,
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
    # Document processing
    "DocumentProcessor",
    "ProcessingResult",
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
    # Web crawling
    "Crawler",
    "CrawlStrategy", 
    "CrawlProgress",
    "CrawlResult",
    # Metadata sanitization
    "MetadataSanitizer",
    "sanitize_metadata",
    "validate_metadata",
    # Document structure analysis
    "DocumentStructureAnalyzer",
    "DocumentAnalysis",
]
