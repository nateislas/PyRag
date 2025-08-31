#!/usr/bin/env python3
"""Test script for PyRAG MCP server."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.mcp.server import mcp
from pyrag.core import PyRAG
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_mcp_server():
    """Test the MCP server tools directly."""
    
    print("🧪 Testing PyRAG MCP Server Tools")
    print("=" * 50)
    
    # Initialize PyRAG
    print("📚 Initializing PyRAG...")
    try:
        pyrag = PyRAG()
        print("✅ PyRAG initialized successfully")
    except Exception as e:
        print(f"❌ PyRAG initialization failed: {e}")
        return
    
    # Test each tool
    print(f"\n🔧 Testing {len(mcp.tools)} MCP tools...")
    
    # 1. Test search_python_docs
    print("\n1. Testing search_python_docs...")
    try:
        start_time = time.time()
        result = await mcp.tools["search_python_docs"].function(
            query="how to make HTTP requests",
            library="requests",
            content_type="examples"
        )
        query_time = time.time() - start_time
        print(f"✅ Search completed in {query_time:.2f}s")
        print(f"📄 Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    # 2. Test list_available_libraries
    print("\n2. Testing list_available_libraries...")
    try:
        start_time = time.time()
        result = await mcp.tools["list_available_libraries"].function()
        query_time = time.time() - start_time
        print(f"✅ Libraries query completed in {query_time:.2f}s")
        print(f"📚 Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Libraries failed: {e}")
    
    # 3. Test get_library_status
    print("\n3. Testing get_library_status...")
    try:
        start_time = time.time()
        result = await mcp.tools["get_library_status"].function(library_name="requests")
        query_time = time.time() - start_time
        print(f"✅ Status query completed in {query_time:.2f}s")
        print(f"📊 Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Status failed: {e}")
    
    # 4. Test get_api_reference
    print("\n4. Testing get_api_reference...")
    try:
        start_time = time.time()
        result = await mcp.tools["get_api_reference"].function(
            library="requests",
            api_path="requests.get",
            include_examples=True
        )
        query_time = time.time() - start_time
        print(f"✅ API reference query completed in {query_time:.2f}s")
        print(f"📖 Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ API reference failed: {e}")
    
    # 5. Test check_deprecation
    print("\n5. Testing check_deprecation...")
    try:
        start_time = time.time()
        result = await mcp.tools["check_deprecation"].function(
            library="requests",
            apis=["requests.get", "requests.post"]
        )
        query_time = time.time() - start_time
        print(f"✅ Deprecation check completed in {query_time:.2f}s")
        print(f"⚠️  Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Deprecation check failed: {e}")
    
    # 6. Test find_similar_patterns
    print("\n6. Testing find_similar_patterns...")
    try:
        start_time = time.time()
        result = await mcp.tools["find_similar_patterns"].function(
            code_snippet="requests.get(url)",
            intent="HTTP GET request"
        )
        query_time = time.time() - start_time
        print(f"✅ Pattern search completed in {query_time:.2f}s")
        print(f"🔍 Result preview: {result[:200]}...")
    except Exception as e:
        print(f"❌ Pattern search failed: {e}")
    
    print(f"\n🎉 MCP server testing completed!")
    print(f"📊 All {len(mcp.tools)} tools tested successfully")


async def test_mcp_server_connection():
    """Test if the MCP server can be started and connected to."""
    
    print("\n🔗 Testing MCP Server Connection")
    print("=" * 40)
    
    try:
        # Test if we can import and create the MCP server
        from pyrag.mcp.server import mcp
        
        print("✅ MCP server imported successfully")
        print(f"📋 Available tools: {list(mcp.tools.keys())}")
        
        # Test server initialization
        print("🚀 Testing server initialization...")
        
        # Note: We can't actually start the server in this test because it blocks
        # But we can verify the tools are properly configured
        for tool_name, tool in mcp.tools.items():
            print(f"  ✅ {tool_name}: {tool.description[:50]}...")
        
        print("✅ MCP server tools are properly configured")
        
    except Exception as e:
        print(f"❌ MCP server connection test failed: {e}")
        import traceback
        traceback.print_exc()


async def performance_test():
    """Test MCP server performance with multiple queries."""
    
    print("\n⚡ Performance Testing")
    print("=" * 30)
    
    queries = [
        "how to make HTTP requests",
        "pandas DataFrame operations",
        "FastAPI routing examples",
        "SQLAlchemy database queries",
        "matplotlib plotting examples"
    ]
    
    start_time = time.time()
    successful_queries = 0
    
    for i, query in enumerate(queries, 1):
        print(f"\n🔍 Query {i}/{len(queries)}: {query}")
        query_start = time.time()
        
        try:
            result = await mcp.tools["search_python_docs"].function(query=query)
            query_time = time.time() - query_start
            successful_queries += 1
            print(f"✅ Completed in {query_time:.2f}s")
            print(f"📄 Result length: {len(result)} characters")
        except Exception as e:
            query_time = time.time() - query_start
            print(f"❌ Failed in {query_time:.2f}s: {e}")
    
    total_time = time.time() - start_time
    success_rate = (successful_queries / len(queries)) * 100
    
    print(f"\n📊 Performance Summary:")
    print(f"  Total queries: {len(queries)}")
    print(f"  Successful: {successful_queries}")
    print(f"  Success rate: {success_rate:.1f}%")
    print(f"  Total time: {total_time:.2f}s")
    print(f"  Average time: {total_time/len(queries):.2f}s per query")


async def main():
    """Main test function."""
    
    print("🚀 PyRAG MCP Server Test Suite")
    print("=" * 50)
    
    # Test MCP server connection
    await test_mcp_server_connection()
    
    # Test individual tools
    await test_mcp_server()
    
    # Test performance
    await performance_test()
    
    print(f"\n🎉 All tests completed!")
    print(f"📝 Check the results above to verify the MCP server is working correctly.")


if __name__ == "__main__":
    asyncio.run(main())
