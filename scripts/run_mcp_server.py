#!/usr/bin/env python3
"""Standalone script to run the PyRAG MCP server."""

import sys
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from pyrag.mcp.server import main

if __name__ == "__main__":
    import asyncio
    asyncio.run(main())
