# PyRAG MCP Server Setup & Testing Guide

## Overview

The PyRAG MCP (Model Context Protocol) server provides AI coding assistants with access to current Python library documentation through natural language queries. This guide will help you set up and test the MCP server.

## Prerequisites

- Python 3.11+
- PyRAG development environment set up
- Essential services running (PostgreSQL, Redis, ChromaDB)

## Quick Setup

### 1. Install Dependencies

The MCP server uses FastMCP, which is already included in the project dependencies:

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies (if not already done)
pip install -e .
```

### 2. Start Essential Services

```bash
# Start the required services
docker-compose -f docker-compose.simple.yml up -d

# Verify services are running
docker-compose -f docker-compose.simple.yml ps
```

### 3. Run the MCP Server

```bash
# Run the MCP server directly
python scripts/run_mcp_server.py

# Or run it as a module
python -m pyrag.mcp.server
```

## Testing the MCP Server

### Method 1: Direct Testing with Python

Create a test script to verify the MCP server is working:

```python
#!/usr/bin/env python3
"""Test script for PyRAG MCP server."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.mcp.server import mcp
from pyrag.core import PyRAG

async def test_mcp_server():
    """Test the MCP server tools directly."""
    
    print("🧪 Testing PyRAG MCP Server Tools")
    print("=" * 50)
    
    # Initialize PyRAG
    pyrag = PyRAG()
    
    # Test each tool
    print("\n1. Testing search_python_docs...")
    try:
        result = await mcp.tools["search_python_docs"].function(
            query="how to make HTTP requests",
            library="requests",
            content_type="examples"
        )
        print(f"✅ Search result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Search failed: {e}")
    
    print("\n2. Testing list_available_libraries...")
    try:
        result = await mcp.tools["list_available_libraries"].function()
        print(f"✅ Libraries result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Libraries failed: {e}")
    
    print("\n3. Testing get_library_status...")
    try:
        result = await mcp.tools["get_library_status"].function(library_name="requests")
        print(f"✅ Status result: {result[:200]}...")
    except Exception as e:
        print(f"❌ Status failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_mcp_server())
```

### Method 2: Test with MCP Client

Install an MCP client to test the server:

```bash
# Install MCP client (if needed)
pip install mcp

# Test the server
mcp test pyrag
```

### Method 3: Test with Cursor Integration

1. **Configure Cursor** (if using Cursor IDE):
   
   Create or edit `~/.cursor/mcp.json`:
   ```json
   {
     "mcpServers": {
       "pyrag": {
         "command": "python",
         "args": ["/path/to/your/pyrag/scripts/run_mcp_server.py"],
         "env": {
           "PYTHONPATH": "/path/to/your/pyrag/src"
         }
       }
     }
   }
   ```

2. **Test in Cursor**:
   - Restart Cursor
   - Open a Python file
   - Try asking: "How do I make HTTP requests with the requests library?"
   - Cursor should use the PyRAG MCP server to provide current documentation

## Available MCP Tools

The PyRAG MCP server provides 6 tools:

### 1. `search_python_docs`
Search Python documentation with semantic understanding.

**Parameters:**
- `query` (str): Natural language query
- `library` (str, optional): Specific library to search
- `version` (str, optional): Version constraint
- `content_type` (str, optional): "examples", "reference", "tutorials", "all"

**Example:**
```python
result = await search_python_docs(
    query="how to make authenticated HTTP requests",
    library="requests",
    content_type="examples"
)
```

### 2. `get_api_reference`
Get detailed API reference for specific functions/classes.

**Parameters:**
- `library` (str): Library name
- `api_path` (str): API path (e.g., "pandas.DataFrame.merge")
- `include_examples` (bool): Whether to include examples

**Example:**
```python
result = await get_api_reference(
    library="pandas",
    api_path="DataFrame.merge",
    include_examples=True
)
```

### 3. `check_deprecation`
Check if APIs are deprecated and get replacement suggestions.

**Parameters:**
- `library` (str): Library name
- `apis` (list): List of API names to check

**Example:**
```python
result = await check_deprecation(
    library="requests",
    APIs=["requests.get", "requests.post"]
)
```

### 4. `find_similar_patterns`
Find similar usage patterns or alternative approaches.

**Parameters:**
- `code_snippet` (str): Code snippet to find patterns for
- `intent` (str, optional): Intent description

**Example:**
```python
result = await find_similar_patterns(
    code_snippet="requests.get(url)",
    intent="HTTP GET request"
)
```

### 5. `list_available_libraries`
List all available libraries in the system.

**Example:**
```python
result = await list_available_libraries()
```

### 6. `get_library_status`
Get detailed status of a specific library.

**Parameters:**
- `library_name` (str): Name of the library

**Example:**
```python
result = await get_library_status(library_name="requests")
```

## Troubleshooting

### Common Issues

1. **Import Errors**:
   ```bash
   # Make sure you're in the project root
   cd /path/to/pyrag
   
   # Set PYTHONPATH
   export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
   ```

2. **Service Connection Errors**:
   ```bash
   # Check if services are running
   docker-compose -f docker-compose.simple.yml ps
   
   # Restart services if needed
   docker-compose -f docker-compose.simple.yml restart
   ```

3. **MCP Server Not Starting**:
   ```bash
   # Check logs
   python scripts/run_mcp_server.py 2>&1 | tee mcp.log
   
   # Verify dependencies
   pip list | grep fastmcp
   ```

4. **No Documentation Found**:
   - Ensure you've ingested some libraries first
   - Check the vector store is populated
   - Verify the embedding service is working

### Debug Mode

Run the MCP server with debug logging:

```bash
# Set debug environment variable
export LOG_LEVEL=DEBUG

# Run with debug output
python scripts/run_mcp_server.py
```

## Integration Examples

### With Cursor IDE

1. **Setup**:
   ```json
   // ~/.cursor/mcp.json
   {
     "mcpServers": {
       "pyrag": {
         "command": "python",
         "args": ["/absolute/path/to/pyrag/scripts/run_mcp_server.py"],
         "env": {
           "PYTHONPATH": "/absolute/path/to/pyrag/src",
           "LOG_LEVEL": "INFO"
         }
       }
     }
   }
   ```

2. **Usage**:
   - Ask: "How do I use pandas DataFrame?"
   - Ask: "Show me examples of FastAPI routing"
   - Ask: "What's the latest way to make HTTP requests?"

### With Other MCP Clients

```bash
# Test with mcp-cli
pip install mcp-cli
mcp-cli --server pyrag --query "how to use requests library"
```

## Performance Testing

Test the MCP server performance:

```python
import asyncio
import time
from pyrag.mcp.server import mcp

async def performance_test():
    """Test MCP server performance."""
    
    queries = [
        "how to make HTTP requests",
        "pandas DataFrame operations",
        "FastAPI routing examples",
        "SQLAlchemy database queries"
    ]
    
    start_time = time.time()
    
    for query in queries:
        query_start = time.time()
        try:
            result = await mcp.tools["search_python_docs"].function(query=query)
            query_time = time.time() - query_start
            print(f"✅ {query}: {query_time:.2f}s")
        except Exception as e:
            print(f"❌ {query}: {e}")
    
    total_time = time.time() - start_time
    print(f"\n📊 Total time: {total_time:.2f}s")
    print(f"📊 Average time: {total_time/len(queries):.2f}s per query")

if __name__ == "__main__":
    asyncio.run(performance_test())
```

## Next Steps

1. **Ingest Libraries**: Use the ingestion scripts to add Python libraries
2. **Test Tools**: Try all 6 MCP tools with different queries
3. **Integrate**: Connect with your preferred AI coding assistant
4. **Monitor**: Check logs and performance metrics
5. **Scale**: Add more libraries and optimize performance

## Support

- Check the logs for detailed error messages
- Verify all services are running correctly
- Ensure you have the latest version of FastMCP
- Test with simple queries first before complex ones

The PyRAG MCP server should now be ready to provide current Python documentation to your AI coding assistant! 🚀
