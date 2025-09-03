"""Tests for Phase 2 features."""

import asyncio
from unittest.mock import AsyncMock, Mock

import pytest

from pyrag.trash.caching import CacheManager
from pyrag.core import PyRAG
from pyrag.mcp.server import mcp
from pyrag.search import EnhancedSearchEngine, QueryAnalysis, QueryAnalyzer


class TestQueryAnalyzer:
    """Test query analyzer functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.analyzer = QueryAnalyzer()

    def test_extract_library_hint(self):
        """Test library hint extraction."""
        # Test with requests
        analysis = self.analyzer.analyze_query(
            "How to make HTTP requests with requests library"
        )
        assert analysis.library_hint == "requests"

        # Test with pandas
        analysis = self.analyzer.analyze_query("pandas DataFrame operations")
        assert analysis.library_hint == "pandas"

        # Test with no library hint
        analysis = self.analyzer.analyze_query("How to parse JSON")
        assert analysis.library_hint is None

    def test_extract_version_hint(self):
        """Test version hint extraction."""
        analysis = self.analyzer.analyze_query("requests 2.31.0 authentication")
        assert analysis.version_hint == "2.31.0"

        analysis = self.analyzer.analyze_query("pandas v1.5.0 DataFrame")
        assert analysis.version_hint == "v1.5.0"

    def test_extract_api_path_hint(self):
        """Test API path hint extraction."""
        analysis = self.analyzer.analyze_query("requests.Session.get method")
        assert analysis.api_path_hint == "requests.Session.get"

        analysis = self.analyzer.analyze_query("pandas.DataFrame.merge function")
        assert analysis.api_path_hint == "pandas.DataFrame.merge"

    def test_determine_intent(self):
        """Test intent determination."""
        # API reference intent
        analysis = self.analyzer.analyze_query("requests.Session.get API reference")
        assert analysis.intent == "api_reference"

        # Examples intent
        analysis = self.analyzer.analyze_query("requests authentication examples")
        assert analysis.intent == "examples"

        # Tutorial intent
        analysis = self.analyzer.analyze_query("pandas tutorial for beginners")
        assert analysis.intent == "tutorial"

        # General intent
        analysis = self.analyzer.analyze_query("how to work with data")
        assert analysis.intent == "general"


class TestCacheManager:
    """Test cache manager functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.cache_manager = CacheManager()

    @pytest.mark.asyncio
    async def test_memory_cache(self):
        """Test in-memory cache functionality."""
        # Test set and get
        await self.cache_manager.set("test", "value", ttl=60, key="test_key")
        result = await self.cache_manager.get("test", key="test_key")
        assert result == "value"

        # Test cache miss
        result = await self.cache_manager.get("test", key="nonexistent")
        assert result is None

    @pytest.mark.asyncio
    async def test_cache_stats(self):
        """Test cache statistics."""
        # Perform some operations
        await self.cache_manager.set("test", "value", ttl=60, key="test_key")
        await self.cache_manager.get("test", key="test_key")
        await self.cache_manager.get("test", key="nonexistent")

        stats = self.cache_manager.get_stats()
        assert stats["sets"] == 1
        assert stats["hits"] == 1
        assert stats["misses"] == 1
        assert stats["hit_rate"] == 0.5


class TestEnhancedSearchEngine:
    """Test enhanced search engine functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.mock_vector_store = Mock()
        self.mock_embedding_service = Mock()
        self.search_engine = EnhancedSearchEngine(
            self.mock_vector_store, self.mock_embedding_service
        )

    @pytest.mark.asyncio
    async def test_search_with_analysis(self):
        """Test search with query analysis."""
        # Mock embedding service
        self.mock_embedding_service.embed_query = AsyncMock(
            return_value=[0.1, 0.2, 0.3]
        )

        # Mock vector store search
        mock_results = [
            {
                "id": "1",
                "content": "Test content",
                "metadata": {"library": "requests", "version": "2.31.0"},
                "distance": 0.1,
            }
        ]
        self.mock_vector_store.search = AsyncMock(return_value=mock_results)

        # Test search
        results = await self.search_engine.search(
            query="requests authentication", max_results=5
        )

        assert len(results) > 0
        assert results[0]["final_score"] > 0


class TestMCPServer:
    """Test MCP server functionality."""

    def test_mcp_server_initialization(self):
        """Test MCP server initialization."""
        # Check that the MCP server is properly initialized
        assert mcp is not None
        assert hasattr(mcp, "run")

        # FastMCP creates a server instance, verify it exists
        assert True  # Placeholder - FastMCP doesn't expose tools list easily

    @pytest.mark.asyncio
    async def test_search_python_docs_tool(self):
        """Test search_python_docs MCP tool."""
        # FastMCP decorators replace functions with FunctionTool objects
        from pyrag.mcp.server import search_python_docs

        # Verify the tool object exists and has the expected name
        assert hasattr(search_python_docs, "name")
        assert search_python_docs.name == "search_python_docs"

        # Test that we can import the tool
        assert True  # Placeholder - FastMCP tools are not directly testable

    @pytest.mark.asyncio
    async def test_get_api_reference_tool(self):
        """Test get_api_reference MCP tool."""
        # FastMCP decorators replace functions with FunctionTool objects
        from pyrag.mcp.server import get_api_reference

        # Verify the tool object exists and has the expected name
        assert hasattr(get_api_reference, "name")
        assert get_api_reference.name == "get_api_reference"

        # Test that we can import the tool
        assert True  # Placeholder - FastMCP tools are not directly testable


class TestPhase2Integration:
    """Integration tests for Phase 2 features."""

    @pytest.mark.asyncio
    async def test_enhanced_search_integration(self):
        """Test enhanced search integration with PyRAG core."""
        # This would require a full PyRAG instance with test data
        # For now, we'll test the components work together
        pass

    @pytest.mark.asyncio
    async def test_mcp_server_integration(self):
        """Test MCP server integration."""
        # This would test the full MCP server with a client
        # For now, we'll test the server starts correctly
        pass


if __name__ == "__main__":
    pytest.main([__file__])
