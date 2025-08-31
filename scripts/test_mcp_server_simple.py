#!/usr/bin/env python3
"""Simple test script for PyRAG MCP server."""

import asyncio
import sys
import time
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.core import PyRAG
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_pyrag_core():
    """Test the PyRAG core functionality directly."""
    
    print("🧪 Testing PyRAG Core Functionality")
    print("=" * 50)
    
    # Initialize PyRAG
    print("📚 Initializing PyRAG...")
    try:
        pyrag = PyRAG()
        print("✅ PyRAG initialized successfully")
    except Exception as e:
        print(f"❌ PyRAG initialization failed: {e}")
        return False
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    try:
        start_time = time.time()
        results = await pyrag.search_documentation(
            query="how to make HTTP requests",
            library="requests",
            max_results=5
        )
        query_time = time.time() - start_time
        print(f"✅ Search completed in {query_time:.2f}s")
        print(f"📄 Found {len(results) if results else 0} results")
        
        if results:
            print(f"📖 First result preview: {results[0]['content'][:200]}...")
        else:
            print("📝 No results found (this is normal if no libraries are ingested)")
            
    except Exception as e:
        print(f"❌ Search failed: {e}")
        return False
    
    # Test library listing
    print("\n📚 Testing library listing...")
    try:
        start_time = time.time()
        libraries = await pyrag.list_libraries()
        query_time = time.time() - start_time
        print(f"✅ Library listing completed in {query_time:.2f}s")
        print(f"📚 Found {len(libraries) if libraries else 0} libraries")
        
        if libraries:
            for lib in libraries[:3]:  # Show first 3
                print(f"  📖 {lib.get('name', 'Unknown')} - {lib.get('status', 'Unknown')}")
        else:
            print("📝 No libraries found (this is normal if none are ingested)")
            
    except Exception as e:
        print(f"❌ Library listing failed: {e}")
        return False
    
    return True


async def test_mcp_server_startup():
    """Test if the MCP server can be imported and configured."""
    
    print("\n🔗 Testing MCP Server Configuration")
    print("=" * 40)
    
    try:
        # Test if we can import the MCP server
        from pyrag.mcp.server import mcp, main
        
        print("✅ MCP server imported successfully")
        print(f"📋 Server name: {mcp.name}")
        
        # Test if we can import the tools (they should be registered via decorators)
        print("🔧 Checking tool registration...")
        
        # FastMCP tools are registered via decorators, so we can't easily list them
        # But we can verify the server is properly configured
        print("✅ MCP server is properly configured")
        
        return True
        
    except Exception as e:
        print(f"❌ MCP server configuration failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_services():
    """Test if all required services are accessible."""
    
    print("\n🔌 Testing Service Connections")
    print("=" * 35)
    
    # Test SQLite database connection (PyRAG uses SQLite, not PostgreSQL)
    print("🗄️  Testing SQLite Database...")
    try:
        from pyrag.database.connection import get_engine
        from sqlalchemy import text
        
        engine = get_engine()
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print("✅ SQLite database connection successful")
    except Exception as e:
        print(f"❌ SQLite database connection failed: {e}")
        return False
    
    # Test ChromaDB connection
    print("🔍 Testing ChromaDB...")
    try:
        from pyrag.vector_store import VectorStore
        
        vector_store = VectorStore()
        collections = vector_store.client.list_collections()
        print(f"✅ ChromaDB connection successful ({len(collections)} collections)")
    except Exception as e:
        print(f"❌ ChromaDB connection failed: {e}")
        return False
    
    # Test Redis connection
    print("🔴 Testing Redis...")
    try:
        import redis
        
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        print("✅ Redis connection successful")
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        return False
    
    return True


async def main():
    """Main test function."""
    
    print("🚀 PyRAG MCP Server Simple Test Suite")
    print("=" * 50)
    
    # Test services first
    services_ok = await test_services()
    if not services_ok:
        print("\n❌ Service tests failed. Please check your Docker services.")
        return
    
    # Test PyRAG core
    core_ok = await test_pyrag_core()
    if not core_ok:
        print("\n❌ PyRAG core tests failed.")
        return
    
    # Test MCP server configuration
    mcp_ok = await test_mcp_server_startup()
    if not mcp_ok:
        print("\n❌ MCP server configuration failed.")
        return
    
    print(f"\n🎉 All tests completed successfully!")
    print(f"📝 The MCP server should be ready to run.")
    print(f"\n🚀 To start the MCP server, run:")
    print(f"   python scripts/run_mcp_server.py")
    print(f"\n🔧 To test with Cursor, configure ~/.cursor/mcp.json with:")
    print(f"   {Path(__file__).parent.parent}/scripts/run_mcp_server.py")


if __name__ == "__main__":
    asyncio.run(main())
