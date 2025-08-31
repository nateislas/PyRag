"""PyRAG CLI for running the MCP server."""

import asyncio
import sys
from pathlib import Path

from .mcp.server import main as mcp_main
from .logging import get_logger

logger = get_logger(__name__)


def main():
    """Main CLI entry point."""
    try:
        logger.info("Starting PyRAG MCP Server via CLI")
        asyncio.run(mcp_main())
    except KeyboardInterrupt:
        logger.info("PyRAG MCP Server stopped by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Error running PyRAG MCP Server: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
