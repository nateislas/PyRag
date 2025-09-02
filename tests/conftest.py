"""Pytest configuration and fixtures."""

import pytest
from typing import Generator
from unittest.mock import Mock, patch

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session

from pyrag.config import get_config
from pyrag.models import Base
from pyrag.database import get_session


@pytest.fixture(scope="session")
def test_engine():
    """Create test database engine."""
    # Use in-memory SQLite for testing
    engine = create_engine("sqlite:///:memory:", echo=False)
    Base.metadata.create_all(bind=engine)
    yield engine
    Base.metadata.drop_all(bind=engine)


@pytest.fixture
def test_session(test_engine) -> Generator[Session, None, None]:
    """Create test database session."""
    TestingSessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=test_engine)
    session = TestingSessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def mock_settings():
    """Mock settings for testing."""
    with patch("pyrag.config.get_config") as mock:
        # Configure mock settings
        mock.return_value.environment = "testing"
        mock.return_value.vector_store.db_path = "./test_chroma_db"
        mock.return_value.embedding.device = "cpu"
        mock.return_value.log_level = "DEBUG"
        yield mock


@pytest.fixture
def mock_logger():
    """Mock logger for testing."""
    with patch("pyrag.logging.get_logger") as mock:
        logger = Mock()
        mock.return_value = logger
        yield logger


@pytest.fixture
def sample_library_data():
    """Sample library data for testing."""
    return {
        "name": "requests",
        "description": "Python HTTP library",
        "repository_url": "https://github.com/psf/requests",
        "documentation_url": "https://docs.python-requests.org/",
        "license": "Apache-2.0",
    }


@pytest.fixture
def sample_document_chunk_data():
    """Sample document chunk data for testing."""
    return {
        "content": "Make a GET request to the specified URL",
        "content_type": "method_description",
        "hierarchy_path": "requests.get",
        "hierarchy_level": 3,
        "title": "requests.get",
        "source_url": "https://docs.python-requests.org/en/latest/api/#requests.get",
    }
