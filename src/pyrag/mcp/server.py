"""PyRAG MCP Server using FastMCP - optimized for FastMCP Cloud deployment."""

import logging
import os
import sys
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP

# Configure lightweight logging for FastMCP Cloud
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

logger = logging.getLogger(__name__)
logger.info("PyRAG MCP Server starting...")

# Create FastMCP server instance
mcp = FastMCP("PyRAG 🐍")
logger.info("FastMCP server instance created")

# Lazy initialization of PyRAG to avoid heavy import-time work
_pyrag_instance = None

def get_pyrag():
    """Get PyRAG instance with lazy initialization."""
    global _pyrag_instance
    if _pyrag_instance is None:
        try:
            logger.info("Initializing PyRAG system...")
            from pyrag.core import PyRAG
            _pyrag_instance = PyRAG()
            logger.info("PyRAG system initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize PyRAG: {e}")
            raise
    return _pyrag_instance

@mcp.tool
async def search_python_docs(
    query: str,
    library: Optional[str] = None,
    version: Optional[str] = None,
    content_type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Search Python library documentation with semantic understanding.

    This is the main RAG tool that searches through indexed Python library documentation
    to find relevant information based on your query.

    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional, e.g., "fastapi", "pandas")
        version: Specific version constraint (optional)
        content_type: Type of documentation to prioritize (examples, reference, tutorials, all)
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Searching Python docs for: {query}")
        if library:
            await ctx.info(f"Filtering by library: {library}")

    try:
        # Get PyRAG instance (lazy initialization)
        pyrag = get_pyrag()
        
        # Map content_type to our internal format
        mapped_content_type = None
        if content_type:
            if content_type in ["examples", "tutorials"]:
                mapped_content_type = "examples"
            elif content_type in ["reference", "api_reference"]:
                mapped_content_type = "api_reference"
            elif content_type == "overview":
                mapped_content_type = "overview"

        # Search documentation
        results = await pyrag.search_documentation(
            query=query,
            library=library,
            version=version,
            content_type=mapped_content_type,
            max_results=10,
        )

        if not results:
            return "No relevant documentation found for your query."

        # Format results
        response_parts = []
        for i, result in enumerate(results[:5], 1):  # Limit to top 5 results
            response_parts.append(f"**Result {i}** (Score: {result['score']:.2f})")
            response_parts.append(f"Library: {result['library']} v{result['version']}")
            response_parts.append(f"Type: {result['content_type']}")
            response_parts.append("")
            response_parts.append(
                result["content"][:500] + "..."
                if len(result["content"]) > 500
                else result["content"]
            )
            response_parts.append("")

        return "\n".join(response_parts)

    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
        if ctx:
            await ctx.error(f"Error searching documentation: {e}")
        return f"Error searching documentation: {e}"

logger.info("PyRAG MCP Server ready with 1 tool: search_python_docs")