"""Test configuration and fixtures for PyRag."""

import pytest
from unittest.mock import Mock


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = Mock()
    config.llm.api_key = "test_key"
    config.llm.base_url = "https://test.api.com"
    config.llm.model = "test-model"
    config.firecrawl.api_key = "test_firecrawl_key"
    config.vector_store.db_path = "./test_chroma_db"
    return config


@pytest.fixture
def sample_library_data():
    """Sample library data for testing."""
    return {
        "name": "test-library",
        "description": "A test library for testing purposes",
        "repository_url": "https://github.com/test/test-library",
        "documentation_url": "https://test-library.readthedocs.io",
        "license": "MIT",
    }


@pytest.fixture
def sample_document_data():
    """Sample document data for testing."""
    return {
        "content": "This is a test document content for testing purposes.",
        "metadata": {
            "library": "test-library",
            "version": "1.0.0",
            "content_type": "api_reference",
            "hierarchy_path": ["test", "function"],
            "api_path": "test.function",
        },
    }
