"""Documentation ingestion module for automated library documentation processing."""

from .firecrawl_client import FirecrawlClient, ScrapedDocument
from .enhanced_documentation_processor import EnhancedDocumentationProcessor, EnhancedProcessingResult
from .documentation_manager import DocumentationManager, DocumentationJob, DocumentationResult

__all__ = [
    "FirecrawlClient",
    "ScrapedDocument",
    "EnhancedDocumentationProcessor",
    "EnhancedProcessingResult",
    "DocumentationManager",
    "DocumentationJob", 
    "DocumentationResult",
]
