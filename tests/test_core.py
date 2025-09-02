"""Tests for core PyRAG functionality."""

from unittest.mock import Mock, patch

import pytest

from pyrag.core import PyRAG
from pyrag.models import Library


class TestPyRAG:
    """Test cases for PyRAG core functionality."""

    @pytest.fixture
    def pyrag(self):
        """Create PyRAG instance for testing."""
        return PyRAG()

    @pytest.mark.asyncio
    async def test_search_documentation_placeholder(self, pyrag):
        """Test that search_documentation returns results (system is working)."""
        results = await pyrag.search_documentation("test query")
        assert isinstance(results, list)
        assert len(results) > 0  # System is working and returning results

    @pytest.mark.asyncio
    async def test_get_api_reference_placeholder(self, pyrag):
        """Test that get_api_reference returns None (placeholder)."""
        result = await pyrag.get_api_reference("requests", "requests.get")
        assert result is None

    @pytest.mark.asyncio
    async def test_check_deprecation_placeholder(self, pyrag):
        """Test that check_deprecation returns expected structure."""
        result = await pyrag.check_deprecation("requests", ["requests.get"])
        assert isinstance(result, dict)
        assert "library" in result
        assert "deprecated_apis" in result
        assert "replacement_suggestions" in result
        assert result["library"] == "requests"
        assert isinstance(result["deprecated_apis"], list)
        assert isinstance(result["replacement_suggestions"], dict)

    @pytest.mark.asyncio
    async def test_find_similar_patterns_placeholder(self, pyrag):
        """Test that find_similar_patterns returns empty list (placeholder)."""
        results = await pyrag.find_similar_patterns("import requests")
        assert isinstance(results, list)
        assert len(results) == 0

    @pytest.mark.asyncio
    async def test_add_library_new(self, pyrag, test_session, sample_library_data):
        """Test adding a new library."""
        with patch("pyrag.core.get_session") as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = test_session

            library = await pyrag.add_library(**sample_library_data)

            assert isinstance(library, Library)
            assert library.name == sample_library_data["name"]
            assert library.description == sample_library_data["description"]
            assert library.repository_url == sample_library_data["repository_url"]
            assert library.documentation_url == sample_library_data["documentation_url"]
            assert library.license == sample_library_data["license"]
            assert library.indexing_status == "pending"

    @pytest.mark.asyncio
    async def test_add_library_existing(self, pyrag, test_session, sample_library_data):
        """Test adding a library that already exists."""
        # First, add the library
        with patch("pyrag.core.get_session") as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = test_session

            library1 = await pyrag.add_library(**sample_library_data)

            # Try to add the same library again
            library2 = await pyrag.add_library(**sample_library_data)

            assert library1.id == library2.id
            assert library1.name == library2.name

    @pytest.mark.asyncio
    async def test_get_library_status_not_found(self, pyrag, test_session):
        """Test getting status for non-existent library."""
        with patch("pyrag.core.get_session") as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = test_session

            result = await pyrag.get_library_status("non_existent_library")
            assert result is None

    @pytest.mark.asyncio
    async def test_list_libraries_empty(self, pyrag, test_session):
        """Test listing libraries when none exist."""
        # Clear the database first
        test_session.query(Library).delete()
        test_session.commit()

        with patch("pyrag.core.get_session") as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = test_session

            libraries = await pyrag.list_libraries()
            assert isinstance(libraries, list)
            assert len(libraries) == 0

    @pytest.mark.asyncio
    async def test_list_libraries_with_data(
        self, pyrag, test_session, sample_library_data
    ):
        """Test listing libraries when some exist."""
        # Add a library first
        with patch("pyrag.core.get_session") as mock_get_session:
            mock_get_session.return_value.__enter__.return_value = test_session

            await pyrag.add_library(**sample_library_data)

            # List libraries
            libraries = await pyrag.list_libraries()
            assert isinstance(libraries, list)
            assert len(libraries) == 1
            assert libraries[0]["name"] == sample_library_data["name"]
            assert libraries[0]["description"] == sample_library_data["description"]
            assert libraries[0]["status"] == "pending"
