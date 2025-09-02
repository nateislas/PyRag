"""Utilities for testing PyRAG MCP Server."""

import asyncio
import json
import time
from typing import Dict, Any, List, Optional, Union
from unittest.mock import Mock, AsyncMock
from datetime import datetime, timedelta


class MockMCPClient:
    """Mock MCP client for testing server responses."""
    
    def __init__(self):
        self.requests = []
        self.responses = []
        self.errors = []
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Mock tool call."""
        request = {
            "tool": tool_name,
            "params": kwargs,
            "timestamp": time.time()
        }
        self.requests.append(request)
        
        # Simulate response based on tool
        if tool_name == "search_python_docs":
            response = await self._mock_search_response(kwargs)
        elif tool_name == "get_api_reference":
            response = await self._mock_api_reference_response(kwargs)
        elif tool_name == "check_deprecation":
            response = await self._mock_deprecation_response(kwargs)
        elif tool_name == "find_similar_patterns":
            response = await self._mock_patterns_response(kwargs)
        elif tool_name == "list_available_libraries":
            response = await self._mock_libraries_response()
        elif tool_name == "get_library_status":
            response = await self._mock_library_status_response(kwargs)
        else:
            response = {"error": f"Unknown tool: {tool_name}"}
        
        self.responses.append(response)
        return response
    
    async def _mock_search_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock search response."""
        query = params.get("query", "")
        library = params.get("library", "unknown")
        
        return {
            "success": True,
            "results": [
                {
                    "score": 0.95,
                    "library": library,
                    "version": "0.104.0",
                    "content_type": "examples",
                    "content": f"Mock response for query: {query}",
                    "source": f"https://docs.{library}.org/"
                }
            ],
            "total_results": 1,
            "query_time_ms": 150
        }
    
    async def _mock_api_reference_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock API reference response."""
        api_path = params.get("api_path", "unknown")
        library = params.get("library", "unknown")
        
        return {
            "success": True,
            "api_path": api_path,
            "library": library,
            "description": f"Mock description for {api_path}",
            "examples": [
                f"# Example usage of {api_path}",
                f"result = {api_path}()"
            ],
            "related_apis": [
                {"name": "related_api", "description": "Related functionality"}
            ]
        }
    
    async def _mock_deprecation_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock deprecation check response."""
        library = params.get("library", "unknown")
        apis = params.get("apis", [])
        
        return {
            "success": True,
            "library": library,
            "deprecated_apis": apis[:1] if apis else [],  # Mock first API as deprecated
            "replacement_suggestions": {
                apis[0]: f"new_{apis[0]}" if apis else "new_api"
            },
            "check_time": datetime.utcnow().isoformat()
        }
    
    async def _mock_patterns_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock similar patterns response."""
        code_snippet = params.get("code_snippet", "")
        
        return {
            "success": True,
            "patterns": [
                {
                    "library": "fastapi",
                    "score": 0.89,
                    "code": code_snippet,
                    "description": f"Mock pattern for: {code_snippet}",
                    "usage_count": 42
                }
            ],
            "total_patterns": 1
        }
    
    async def _mock_libraries_response(self) -> Dict[str, Any]:
        """Mock libraries list response."""
        return {
            "success": True,
            "libraries": [
                {
                    "name": "fastapi",
                    "description": "Fast web framework",
                    "status": "indexed",
                    "latest_version": "0.104.0",
                    "chunk_count": 150,
                    "last_updated": "2024-01-01T00:00:00Z"
                },
                {
                    "name": "pandas",
                    "description": "Data analysis library",
                    "status": "indexed",
                    "latest_version": "2.1.0",
                    "chunk_count": 300,
                    "last_updated": "2024-01-01T00:00:00Z"
                }
            ],
            "total_libraries": 2
        }
    
    async def _mock_library_status_response(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Mock library status response."""
        library_name = params.get("library_name", "unknown")
        
        return {
            "success": True,
            "name": library_name,
            "status": "indexed",
            "latest_version": "0.104.0",
            "chunk_count": 150,
            "last_checked": "2024-01-01T00:00:00Z",
            "indexing_progress": 100,
            "health_status": "healthy"
        }


class MCPPerformanceMonitor:
    """Monitor MCP server performance during testing."""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.requests = []
        self.response_times = []
        self.memory_usage = []
    
    def start_monitoring(self):
        """Start performance monitoring."""
        self.start_time = time.time()
        self.requests = []
        self.response_times = []
        self.memory_usage = []
    
    def record_request(self, tool_name: str, params: Dict[str, Any]):
        """Record a request."""
        self.requests.append({
            "tool": tool_name,
            "params": params,
            "timestamp": time.time()
        })
    
    def record_response(self, response_time: float, memory_usage: Optional[float] = None):
        """Record response time and memory usage."""
        self.response_times.append(response_time)
        if memory_usage is not None:
            self.memory_usage.append(memory_usage)
    
    def stop_monitoring(self):
        """Stop performance monitoring."""
        self.end_time = time.time()
    
    def get_summary(self) -> Dict[str, Any]:
        """Get performance summary."""
        if not self.response_times:
            return {"error": "No performance data collected"}
        
        total_time = self.end_time - self.start_time if self.end_time else 0
        avg_response_time = sum(self.response_times) / len(self.response_times)
        min_response_time = min(self.response_times)
        max_response_time = max(self.response_times)
        
        return {
            "total_monitoring_time": total_time,
            "total_requests": len(self.requests),
            "avg_response_time": avg_response_time,
            "min_response_time": min_response_time,
            "max_response_time": max_response_time,
            "requests_per_second": len(self.requests) / total_time if total_time > 0 else 0,
            "memory_usage": {
                "avg": sum(self.memory_usage) / len(self.memory_usage) if self.memory_usage else 0,
                "min": min(self.memory_usage) if self.memory_usage else 0,
                "max": max(self.memory_usage) if self.memory_usage else 0
            } if self.memory_usage else None
        }


class MCPTestDataGenerator:
    """Generate test data for MCP testing."""
    
    @staticmethod
    def generate_search_queries(count: int = 10) -> List[Dict[str, Any]]:
        """Generate test search queries."""
        queries = [
            "How do I create a FastAPI application?",
            "What is the best way to handle errors in FastAPI?",
            "How to implement authentication in FastAPI?",
            "What are the differences between FastAPI and Flask?",
            "How to deploy a FastAPI application?",
            "How do I use pandas for data analysis?",
            "What are the best practices for pandas performance?",
            "How to handle missing data in pandas?",
            "How to create visualizations with matplotlib?",
            "What is the difference between numpy and pandas?"
        ]
        
        libraries = ["fastapi", "pandas", "numpy", "matplotlib", "requests"]
        content_types = ["examples", "tutorials", "reference", "overview"]
        
        test_data = []
        for i in range(min(count, len(queries))):
            test_data.append({
                "query": queries[i],
                "library": libraries[i % len(libraries)],
                "version": f"0.{100 + i}.0",
                "content_type": content_types[i % len(content_types)]
            })
        
        return test_data
    
    @staticmethod
    def generate_api_paths(count: int = 10) -> List[Dict[str, Any]]:
        """Generate test API paths."""
        api_paths = [
            "FastAPI",
            "FastAPI.get",
            "FastAPI.post",
            "pandas.DataFrame",
            "pandas.DataFrame.merge",
            "numpy.array",
            "numpy.array.reshape",
            "matplotlib.pyplot.plot",
            "requests.get",
            "requests.post"
        ]
        
        libraries = ["fastapi", "pandas", "numpy", "matplotlib", "requests"]
        
        test_data = []
        for i in range(min(count, len(api_paths))):
            test_data.append({
                "library": libraries[i % len(libraries)],
                "api_path": api_paths[i],
                "include_examples": i % 2 == 0
            })
        
        return test_data
    
    @staticmethod
    def generate_library_names(count: int = 10) -> List[str]:
        """Generate test library names."""
        libraries = [
            "fastapi", "pandas", "numpy", "matplotlib", "seaborn",
            "requests", "beautifulsoup4", "sqlalchemy", "alembic", "pytest"
        ]
        return libraries[:count]


class MCPTestValidator:
    """Validate MCP test results."""
    
    @staticmethod
    def validate_search_response(response: Dict[str, Any]) -> bool:
        """Validate search response format."""
        required_fields = ["success", "results", "total_results"]
        if not all(field in response for field in required_fields):
            return False
        
        if not isinstance(response["results"], list):
            return False
        
        if response["results"]:
            result = response["results"][0]
            required_result_fields = ["score", "library", "content"]
            if not all(field in result for field in required_result_fields):
                return False
        
        return True
    
    @staticmethod
    def validate_api_reference_response(response: Dict[str, Any]) -> bool:
        """Validate API reference response format."""
        required_fields = ["success", "api_path", "library", "description"]
        return all(field in response for field in required_fields)
    
    @staticmethod
    def validate_libraries_response(response: Dict[str, Any]) -> bool:
        """Validate libraries response format."""
        required_fields = ["success", "libraries", "total_libraries"]
        if not all(field in response for field in required_fields):
            return False
        
        if not isinstance(response["libraries"], list):
            return False
        
        if response["libraries"]:
            library = response["libraries"][0]
            required_library_fields = ["name", "status", "latest_version"]
            if not all(field in library for field in required_library_fields):
                return False
        
        return True
    
    @staticmethod
    def validate_response_time(response_time: float, max_time: float = 5.0) -> bool:
        """Validate response time is within acceptable limits."""
        return response_time <= max_time
    
    @staticmethod
    def validate_error_response(response: Dict[str, Any]) -> bool:
        """Validate error response format."""
        return "error" in response or "success" in response and not response["success"]


# Convenience functions for common test scenarios
async def run_mcp_tool_test(
    tool_function,
    params: Dict[str, Any],
    expected_content: List[str],
    max_response_time: float = 5.0
) -> Dict[str, Any]:
    """Run a single MCP tool test."""
    start_time = time.time()
    
    try:
        result = await tool_function(**params)
        response_time = time.time() - start_time
        
        # Validate response time
        time_valid = MCPTestValidator.validate_response_time(response_time, max_response_time)
        
        # Validate response content
        content_valid = all(content in str(result) for content in expected_content)
        
        return {
            "success": True,
            "result": result,
            "response_time": response_time,
            "time_valid": time_valid,
            "content_valid": content_valid,
            "error": None
        }
        
    except Exception as e:
        response_time = time.time() - start_time
        return {
            "success": False,
            "result": None,
            "response_time": response_time,
            "time_valid": False,
            "content_valid": False,
            "error": str(e)
        }


async def run_mcp_performance_test(
    tool_function,
    test_data: List[Dict[str, Any]],
    max_response_time: float = 5.0
) -> Dict[str, Any]:
    """Run performance test for an MCP tool."""
    monitor = MCPPerformanceMonitor()
    monitor.start_monitoring()
    
    results = []
    for data in test_data:
        monitor.record_request(tool_function.__name__, data)
        
        start_time = time.time()
        try:
            result = await tool_function(**data)
            response_time = time.time() - start_time
            
            monitor.record_response(response_time)
            
            results.append({
                "success": True,
                "data": data,
                "result": result,
                "response_time": response_time,
                "time_valid": response_time <= max_response_time
            })
            
        except Exception as e:
            response_time = time.time() - start_time
            monitor.record_response(response_time)
            
            results.append({
                "success": False,
                "data": data,
                "error": str(e),
                "response_time": response_time,
                "time_valid": False
            })
    
    monitor.stop_monitoring()
    
    return {
        "performance_summary": monitor.get_summary(),
        "test_results": results,
        "success_rate": len([r for r in results if r["success"]]) / len(results) if results else 0
    }
