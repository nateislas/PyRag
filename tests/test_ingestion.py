"""Tests for PyRAG ingestion functionality."""

import pytest
from unittest.mock import Mock, patch

from pyrag.ingestion import (
    DocumentationManager,
    DocumentationJob,
    DocumentationResult,
    FirecrawlClient,
    ScrapedDocument,
    IntelligentCrawler,
    CrawlStrategy,
    CrawlProgress,
    IntelligentCrawlResult,
    SitemapAnalyzer,
    SitemapAnalysis,
    SitemapEntry,
    DocumentationStructureMapper,
    DocumentationStructure,
    DocumentationNode,
    MetadataSanitizer,
    sanitize_metadata,
    validate_metadata,
    DocumentStructureAnalyzer,
    DocumentAnalysis,
)


class TestIngestionComponents:
    """Test cases for ingestion components."""

    def test_documentation_job_creation(self):
        """Test creating a documentation job."""
        job = DocumentationJob(
            library_name="test-library",
            version="1.0.0",
            documentation_url="https://test.com/docs",
            strategy="comprehensive"
        )
        
        assert job.library_name == "test-library"
        assert job.version == "1.0.0"
        assert job.documentation_url == "https://test.com/docs"
        assert job.strategy == "comprehensive"

    def test_scraped_document_creation(self):
        """Test creating a scraped document."""
        doc = ScrapedDocument(
            url="https://test.com/page",
            title="Test Page",
            content="Test content",
            metadata={"test": "data"}
        )
        
        assert doc.url == "https://test.com/page"
        assert doc.title == "Test Page"
        assert doc.content == "Test content"
        assert doc.metadata["test"] == "data"

    def test_crawl_strategy_enum(self):
        """Test crawl strategy enum values."""
        assert CrawlStrategy.COMPREHENSIVE == "comprehensive"
        assert CrawlStrategy.TARGETED == "targeted"
        assert CrawlStrategy.QUICK == "quick"

    def test_metadata_sanitization(self):
        """Test metadata sanitization functions."""
        # Test sanitize_metadata
        raw_metadata = {
            "title": "Test Title",
            "description": "Test description",
            "tags": ["tag1", "tag2"]
        }
        
        sanitized = sanitize_metadata(raw_metadata)
        assert sanitized["title"] == "Test Title"
        assert sanitized["description"] == "Test description"
        assert sanitized["tags"] == ["tag1", "tag2"]

    def test_metadata_validation(self):
        """Test metadata validation functions."""
        # Test valid metadata
        valid_metadata = {
            "title": "Test",
            "content_type": "api_reference"
        }
        
        assert validate_metadata(valid_metadata) is True

    def test_document_analysis_creation(self):
        """Test creating document analysis."""
        analysis = DocumentAnalysis(
            document_type="api_reference",
            complexity_score=0.8,
            topics=["api", "reference"],
            summary="Test summary"
        )
        
        assert analysis.document_type == "api_reference"
        assert analysis.complexity_score == 0.8
        assert analysis.topics == ["api", "reference"]
        assert analysis.summary == "Test summary"
