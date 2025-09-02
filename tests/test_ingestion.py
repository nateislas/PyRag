"""Tests for documentation ingestion pipeline."""

from unittest.mock import AsyncMock, Mock, patch

import pytest

from pyrag.ingestion import (
    DocumentationIngestionPipeline,
    DocumentationProcessor,
    FirecrawlClient,
    IngestionConfig,
    ScrapedDocument,
)
from pyrag.models.document import DocumentChunk


class TestFirecrawlClient:
    """Test Firecrawl client functionality."""

    @pytest.fixture
    def mock_session(self):
        """Mock aiohttp session."""
        return AsyncMock()

    @pytest.mark.asyncio
    async def test_firecrawl_client_initialization(self):
        """Test Firecrawl client initialization."""
        client = FirecrawlClient()
        assert client.api_key is None
        assert client.base_url == "https://api.firecrawl.dev"

    @pytest.mark.asyncio
    async def test_firecrawl_client_context_manager(self):
        """Test Firecrawl client async context manager."""
        async with FirecrawlClient() as client:
            assert client.session is not None

    @pytest.mark.asyncio
    async def test_parse_scrape_response(self):
        """Test parsing Firecrawl scrape response."""
        client = FirecrawlClient()

        # Mock response data
        response_data = {
            "data": {
                "title": "Test Page",
                "text": "This is test content",
                "markdown": "# Test Page\n\nThis is test content",
                "links": ["https://example.com"],
                "images": ["https://example.com/image.png"],
                "metadata": {"description": "Test page"},
            }
        }

        doc = client._parse_scrape_response(response_data, "https://example.com")

        assert doc.url == "https://example.com"
        assert doc.title == "Test Page"
        assert doc.content == "This is test content"
        assert doc.markdown == "# Test Page\n\nThis is test content"
        assert "links" in doc.metadata


class TestDocumentationProcessor:
    """Test documentation processor functionality."""

    @pytest.fixture
    def processor(self):
        """Documentation processor instance."""
        return DocumentationProcessor()

    @pytest.fixture
    def sample_scraped_doc(self):
        """Sample scraped document."""
        return ScrapedDocument(
            url="https://docs.python-requests.org/en/latest/user/quickstart/",
            title="Quickstart - Requests 2.31.0 documentation",
            content="import requests\nr = requests.get('https://api.github.com/events')",
            markdown="# Quickstart\n\n```python\nimport requests\nr = requests.get('https://api.github.com/events')\n```",
            metadata={"links": [], "images": []},
        )

    def test_processor_initialization(self, processor):
        """Test documentation processor initialization."""
        assert processor is not None
        assert len(processor.api_patterns) > 0

    def test_extract_api_references(self, processor):
        """Test API reference extraction."""
        content = """
        Use `requests.get()` to make HTTP requests.
        The `requests.Session` class provides session management.
        Call `requests.post()` for POST requests.
        """

        api_refs = processor._extract_api_references(content)

        assert "requests.get()" in api_refs
        assert "requests.Session" in api_refs
        assert "requests.post()" in api_refs

    def test_determine_content_type(self, processor):
        """Test content type determination."""
        # API reference page
        url = "https://docs.python-requests.org/en/latest/api/"
        content = "API reference documentation"
        content_type = processor._determine_content_type(url, content)
        assert content_type == "api_reference"

        # Example page
        url = "https://docs.python-requests.org/en/latest/user/quickstart/"
        content = "```python\nimport requests\n```"
        content_type = processor._determine_content_type(url, content)
        assert content_type == "examples"

    def test_extract_hierarchy_path(self, processor):
        """Test hierarchy path extraction."""
        url = "https://docs.python-requests.org/en/latest/user/quickstart/"
        title = "Quickstart Guide"

        path = processor._extract_hierarchy_path(url, title)

        assert "user" in path
        assert "quickstart" in path

    def test_process_scraped_document(self, processor, sample_scraped_doc):
        """Test processing a scraped document."""
        result = processor.process_scraped_document(
            sample_scraped_doc, "requests", "2.31.0"
        )

        assert result.chunks is not None
        assert len(result.chunks) > 0
        assert result.processing_stats["total_chunks"] > 0
        assert result.processing_stats["content_type"] in [
            "api_reference",
            "examples",
            "overview",
        ]


class TestDocumentationIngestionPipeline:
    """Test documentation ingestion pipeline."""

    @pytest.fixture
    def mock_components(self):
        """Mock pipeline components."""
        vector_store = Mock()
        vector_store.add_document = AsyncMock(return_value="chunk_id_123")
        vector_store.search_documents = AsyncMock(return_value=[])
        vector_store.health_check = AsyncMock(return_value=True)

        embedding_service = Mock()
        embedding_service.embed_text = AsyncMock(return_value=[0.1, 0.2, 0.3])
        embedding_service.health_check = AsyncMock(return_value=True)

        library_manager = Mock()
        library_manager.update_library_status = AsyncMock()

        return vector_store, embedding_service, library_manager

    @pytest.fixture
    def pipeline(self, mock_components):
        """Documentation ingestion pipeline instance."""
        vector_store, embedding_service, library_manager = mock_components

        return DocumentationIngestionPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            library_manager=library_manager,
        )

    @pytest.fixture
    def sample_config(self):
        """Sample ingestion config."""
        return IngestionConfig(
            library_name="requests",
            docs_url="https://docs.python-requests.org/en/latest/user/quickstart/",
            version="2.31.0",
        )

    @pytest.mark.asyncio
    async def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.vector_store is not None
        assert pipeline.embedding_service is not None
        assert pipeline.processor is not None

    @pytest.mark.asyncio
    async def test_health_check(self, pipeline):
        """Test pipeline health check."""
        health_status = await pipeline.health_check()

        assert "vector_store" in health_status
        assert "embedding_service" in health_status
        assert "firecrawl" in health_status

    @pytest.mark.asyncio
    async def test_ingestion_workflow(self, pipeline, sample_config):
        """Test complete ingestion workflow."""
        # Mock the scraping step
        with patch.object(pipeline, "_scrape_documentation") as mock_scrape:
            mock_scrape.return_value = [
                ScrapedDocument(
                    url=sample_config.docs_url,
                    title="Test Page",
                    content="import requests\nr = requests.get('https://api.github.com/events')",
                    markdown="# Test\n\n```python\nimport requests\n```",
                    metadata={"links": [], "images": []},
                )
            ]

            # Run ingestion
            result = await pipeline.ingest_library_documentation(sample_config)

            # Verify result
            assert result.library_name == "requests"
            assert result.version == "2.31.0"
            assert result.total_documents == 1
            assert result.total_chunks > 0
            assert result.success is True

    @pytest.mark.asyncio
    async def test_ingestion_with_errors(self, pipeline, sample_config):
        """Test ingestion with errors."""
        # Mock scraping to fail
        with patch.object(pipeline, "_scrape_documentation") as mock_scrape:
            mock_scrape.side_effect = Exception("Scraping failed")

            # Run ingestion
            result = await pipeline.ingest_library_documentation(sample_config)

            # Verify error handling
            assert result.success is False
            assert len(result.errors) > 0
            assert "Scraping failed" in result.errors[0]

    @pytest.mark.asyncio
    async def test_get_ingestion_status(self, pipeline):
        """Test getting ingestion status."""
        status = await pipeline.get_ingestion_status("requests")

        assert "library" in status
        assert "total_documents" in status
        assert "status" in status


class TestIngestionIntegration:
    """Integration tests for ingestion pipeline."""

    @pytest.mark.asyncio
    async def test_end_to_end_ingestion(self):
        """Test end-to-end ingestion with real components."""
        # This test would require actual Firecrawl API access
        # For now, we'll test the components work together

        from pyrag.embeddings import EmbeddingService
        from pyrag.vector_store import VectorStore

        # Initialize real components
        vector_store = VectorStore()
        embedding_service = EmbeddingService()

        # Initialize pipeline
        pipeline = DocumentationIngestionPipeline(
            vector_store=vector_store, embedding_service=embedding_service
        )

        # Test that components can be initialized
        assert pipeline.vector_store is not None
        assert pipeline.embedding_service is not None
        assert pipeline.processor is not None

        # Test health check
        health_status = await pipeline.health_check()
        assert isinstance(health_status, dict)
        assert "vector_store" in health_status
        assert "embedding_service" in health_status
