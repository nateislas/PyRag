"""MCP-specific pytest configuration and fixtures."""

import pytest
import asyncio
import os
from unittest.mock import Mock, patch
from typing import Generator

# Import MCP testing utilities
from .utils.mcp_test_utils import (
    MockMCPClient,
    MockPyRAG,
    MCPPerformanceMonitor,
    MCPTestDataGenerator
)


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_mcp_client():
    """Mock MCP client for testing."""
    return MockMCPClient()


@pytest.fixture
def mock_pyrag_mcp():
    """Mock PyRAG system for MCP testing."""
    return MockPyRAG()


@pytest.fixture
def mcp_performance_monitor():
    """MCP performance monitor for testing."""
    return MCPPerformanceMonitor()


@pytest.fixture
def mcp_test_data_generator():
    """MCP test data generator."""
    return MCPTestDataGenerator()


@pytest.fixture
def sample_search_queries():
    """Sample search queries for testing."""
    return MCPTestDataGenerator.generate_search_queries(5)


@pytest.fixture
def sample_api_paths():
    """Sample API paths for testing."""
    return MCPTestDataGenerator.generate_api_paths(5)


@pytest.fixture
def sample_library_names():
    """Sample library names for testing."""
    return MCPTestDataGenerator.generate_library_names(5)


@pytest.fixture
def mcp_environment_vars():
    """MCP environment variables for testing."""
    return {
        "MCP_TRANSPORT": "stdio",
        "MCP_ENABLE_RATE_LIMIT": "true",
        "MCP_RATE_LIMIT_REQUESTS": "100",
        "MCP_RATE_LIMIT_WINDOW": "3600",
        "MCP_ENABLE_API_KEYS": "false",
        "MCP_MAX_REQUEST_SIZE": "1048576",
        "MCP_ENABLE_IP_WHITELIST": "false"
    }


@pytest.fixture
def mcp_security_config():
    """MCP security configuration for testing."""
    with patch.dict('os.environ', {
        'MCP_ENABLE_RATE_LIMIT': 'true',
        'MCP_RATE_LIMIT_REQUESTS': '10',
        'MCP_RATE_LIMIT_WINDOW': '60',
        'MCP_ENABLE_API_KEYS': 'false',
        'MCP_MAX_REQUEST_SIZE': '1024',
        'MCP_ENABLE_IP_WHITELIST': 'false'
    }, clear=True):
        from pyrag.mcp.server import SecurityConfig
        return SecurityConfig()


@pytest.fixture
def mcp_server_instance():
    """MCP server instance for testing."""
    from pyrag.mcp.server import mcp
    return mcp


@pytest.fixture
def mock_mcp_context():
    """Mock MCP context for testing."""
    class MockContext:
        def __init__(self, client_ip: str = "127.0.0.1"):
            self.client_ip = client_ip
            self.info_calls = []
            self.error_calls = []
            self.progress_calls = []
        
        async def info(self, message: str):
            self.info_calls.append(message)
        
        async def error(self, message: str):
            self.error_calls.append(message)
        
        async def progress(self, message: str, progress: float):
            self.progress_calls.append((message, progress))
    
    return MockContext()


@pytest.fixture
def rate_limit_store_clean():
    """Clean rate limit store for testing."""
    from pyrag.mcp.server import rate_limit_store
    # Clear the store before each test
    rate_limit_store.clear()
    yield rate_limit_store
    # Clean up after test
    rate_limit_store.clear()


@pytest.fixture
def mock_pyrag_singleton():
    """Mock PyRAG singleton for testing."""
    with patch('pyrag.mcp.server._pyrag_instance', None):
        from pyrag.mcp.server import get_pyrag
        yield get_pyrag


# Performance testing fixtures
@pytest.fixture
def performance_test_config():
    """Configuration for performance testing."""
    return {
        "max_response_time": 5.0,  # seconds
        "max_memory_usage": 512,   # MB
        "min_success_rate": 0.95,  # 95%
        "max_concurrent_requests": 10
    }


@pytest.fixture
def load_test_data():
    """Data for load testing."""
    return {
        "concurrent_users": 5,
        "requests_per_user": 20,
        "think_time": 1.0,  # seconds between requests
        "test_duration": 60  # seconds
    }


# Security testing fixtures
@pytest.fixture
def malicious_inputs():
    """Malicious inputs for security testing."""
    return [
        "<script>alert('xss')</script>",
        "'; DROP TABLE users; --",
        "../../../etc/passwd",
        "javascript:alert('xss')",
        "data:text/html,<script>alert('xss')</script>",
        "x" * 10000,  # Very long input
        "",  # Empty input
        None,  # Null input
        {"key": "value"},  # Non-string input
        ["list", "input"]  # List input
    ]


@pytest.fixture
def valid_api_keys():
    """Valid API keys for testing."""
    return [
        "valid_key_1",
        "valid_key_2",
        "valid_key_3"
    ]


@pytest.fixture
def invalid_api_keys():
    """Invalid API keys for testing."""
    return [
        "",
        None,
        "invalid_key",
        "key_with_special_chars!@#$%",
        "x" * 1000  # Very long key
    ]


# Integration testing fixtures
@pytest.fixture
def mcp_server_ready():
    """Check if MCP server is ready for testing."""
    try:
        from pyrag.mcp.server import mcp
        return mcp is not None
    except ImportError:
        return False


@pytest.fixture
def pyrag_core_ready():
    """Check if PyRAG core is ready for testing."""
    try:
        from pyrag.core import PyRAG
        return True
    except ImportError:
        return False


# Test data fixtures
@pytest.fixture
def fastapi_test_data():
    """FastAPI-specific test data."""
    return {
        "search_queries": [
            "How do I create a FastAPI application?",
            "What is the best way to handle errors in FastAPI?",
            "How to implement authentication in FastAPI?",
            "How to use FastAPI with databases?",
            "What are FastAPI dependencies?"
        ],
        "api_paths": [
            "FastAPI",
            "FastAPI.get",
            "FastAPI.post",
            "FastAPI.put",
            "FastAPI.delete"
        ],
        "examples": [
            "app = FastAPI()",
            "app.get('/')",
            "app.post('/items')",
            "app.put('/items/{item_id}')",
            "app.delete('/items/{item_id}')"
        ]
    }


@pytest.fixture
def pandas_test_data():
    """Pandas-specific test data."""
    return {
        "search_queries": [
            "How do I create a pandas DataFrame?",
            "What is the best way to handle missing data in pandas?",
            "How to merge DataFrames in pandas?",
            "How to group data in pandas?",
            "How to create visualizations with pandas?"
        ],
        "api_paths": [
            "pandas.DataFrame",
            "pandas.DataFrame.merge",
            "pandas.DataFrame.groupby",
            "pandas.DataFrame.fillna",
            "pandas.DataFrame.plot"
        ],
        "examples": [
            "df = pd.DataFrame(data)",
            "df.merge(other_df, on='key')",
            "df.groupby('column').agg('mean')",
            "df.fillna(0)",
            "df.plot(kind='bar')"
        ]
    }


# Error scenario fixtures
@pytest.fixture
def error_scenarios():
    """Common error scenarios for testing."""
    return {
        "network_timeout": Exception("Network timeout"),
        "database_error": Exception("Database connection failed"),
        "authentication_error": Exception("Invalid credentials"),
        "rate_limit_exceeded": Exception("Rate limit exceeded"),
        "invalid_input": ValueError("Invalid input provided"),
        "resource_not_found": FileNotFoundError("Resource not found"),
        "permission_denied": PermissionError("Permission denied"),
        "memory_error": MemoryError("Out of memory")
    }


# Cleanup fixtures
@pytest.fixture(autouse=True)
def cleanup_after_test():
    """Clean up after each test."""
    yield
    # Clean up any global state that might have been modified
    try:
        from pyrag.mcp.server import rate_limit_store
        rate_limit_store.clear()
    except ImportError:
        pass


# Skip conditions
def pytest_configure(config):
    """Configure pytest for MCP testing."""
    config.addinivalue_line(
        "markers", "mcp: mark test as MCP-specific"
    )
    config.addinivalue_line(
        "markers", "mcp_security: mark test as MCP security test"
    )
    config.addinivalue_line(
        "markers", "mcp_performance: mark test as MCP performance test"
    )
    config.addinivalue_line(
        "markers", "mcp_integration: mark test as MCP integration test"
    )


def pytest_collection_modifyitems(config, items):
    """Modify test collection for MCP testing."""
    for item in items:
        # Mark tests based on their names
        if "test_mcp" in item.name:
            item.add_marker(pytest.mark.mcp)
        if "security" in item.name:
            item.add_marker(pytest.mark.mcp_security)
        if "performance" in item.name:
            item.add_marker(pytest.mark.mcp_performance)
        if "integration" in item.name:
            item.add_marker(pytest.mark.mcp_integration)
