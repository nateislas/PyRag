"""Enhanced documentation ingestion system for comprehensive coverage."""

from .documentation_manager import (
    DocumentationManager,
    DocumentationJob,
    DocumentationResult,
)
from .enhanced_documentation_processor import (
    EnhancedDocumentationProcessor,
    EnhancedProcessingResult,
)
from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .sitemap_analyzer import SitemapAnalyzer, SitemapAnalysis, SitemapEntry
from .structure_mapper import (
    DocumentationStructureMapper,
    DocumentationStructure,
    DocumentationNode,
)
from .intelligent_crawler import (
    IntelligentCrawler,
    CrawlStrategy,
    CrawlProgress,
    CrawlResult as IntelligentCrawlResult,
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
