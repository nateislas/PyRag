#!/usr/bin/env python3
"""
Standalone FastMCP server entrypoint for PyRAG.
This file can be run directly without relative imports.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add the src directory to Python path so we can import pyrag modules
project_root = Path(__file__).parent
src_path = project_root / "src"
sys.path.insert(0, str(src_path))

# Now we can import pyrag modules
from pyrag.mcp.server import main

if __name__ == "__main__":
    # Set default environment variables for FastMCP Cloud
    os.environ.setdefault("MCP_TRANSPORT", "http")
    os.environ.setdefault("MCP_HOST", "0.0.0.0")
    os.environ.setdefault("MCP_PORT", "8000")
    
    # Run the MCP server
    asyncio.run(main())
