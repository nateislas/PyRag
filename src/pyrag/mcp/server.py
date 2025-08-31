"""PyRAG MCP Server using FastMCP."""

import asyncio
from typing import Any, Dict, List, Optional

from fastmcp import FastMCP, Context

from ..core import PyRAG
from ..logging import get_logger

logger = get_logger(__name__)

# Create FastMCP server instance
mcp = FastMCP("PyRAG 🐍")

# Initialize PyRAG core system
pyrag = PyRAG()


@mcp.tool
async def search_python_docs(
    query: str,
    library: Optional[str] = None,
    version: Optional[str] = None,
    content_type: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Search current Python library documentation with semantic understanding.
    
    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional)
        version: Specific version constraint (optional)
        content_type: Type of documentation to prioritize (examples, reference, tutorials, all)
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Searching Python docs for: {query}")
        if library:
            await ctx.info(f"Filtering by library: {library}")
    
    try:
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
            response_parts.append(result['content'][:500] + "..." if len(result['content']) > 500 else result['content'])
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error searching documentation: {e}")
        if ctx:
            await ctx.error(f"Error searching documentation: {e}")
        return f"Error searching documentation: {e}"


@mcp.tool
async def get_api_reference(
    library: str,
    api_path: str,
    include_examples: bool = True,
    ctx: Context = None,
) -> str:
    """Get detailed API reference for specific functions/classes.
    
    Args:
        library: Library name (e.g., "requests", "pandas")
        api_path: API path (e.g., "pandas.DataFrame.merge")
        include_examples: Whether to include usage examples
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Getting API reference for {api_path} in {library}")
    
    try:
        result = await pyrag.get_api_reference(
            library=library,
            api_path=api_path,
            include_examples=include_examples,
        )
        
        if not result:
            return f"No API reference found for {api_path} in {library}"
        
        # Format response
        response_parts = []
        response_parts.append(f"# {api_path}")
        response_parts.append(f"**Library:** {result['library']}")
        response_parts.append(f"**Score:** {result['score']:.2f}")
        response_parts.append("")
        response_parts.append("## Description")
        response_parts.append(result['content'])
        response_parts.append("")
        
        if result['examples']:
            response_parts.append("## Examples")
            for i, example in enumerate(result['examples'], 1):
                response_parts.append(f"### Example {i}")
                response_parts.append("```python")
                response_parts.append(example)
                response_parts.append("```")
                response_parts.append("")
        
        if result['related_apis']:
            response_parts.append("## Related APIs")
            for related in result['related_apis']:
                response_parts.append(f"- **{related['api_path']}** (Score: {related['score']:.2f})")
                response_parts.append(f"  {related['content']}")
                response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error getting API reference: {e}")
        if ctx:
            await ctx.error(f"Error getting API reference: {e}")
        return f"Error getting API reference: {e}"


@mcp.tool
async def check_deprecation(
    library: str,
    apis: List[str],
    ctx: Context = None,
) -> str:
    """Check if APIs are deprecated and get replacement suggestions.
    
    Args:
        library: Library name to check
        apis: List of API paths to check for deprecation
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Checking deprecation status for {len(apis)} APIs in {library}")
    
    try:
        result = await pyrag.check_deprecation(library=library, apis=apis)
        
        if not result['deprecated_apis']:
            return f"✅ All checked APIs in {library} are current and not deprecated."
        
        # Format response
        response_parts = []
        response_parts.append(f"# Deprecation Check for {library}")
        response_parts.append("")
        
        for api in result['deprecated_apis']:
            response_parts.append(f"## ⚠️ {api}")
            if api in result['replacement_suggestions']:
                replacement = result['replacement_suggestions'][api]
                response_parts.append(f"**Replacement:** {replacement}")
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error checking deprecation: {e}")
        if ctx:
            await ctx.error(f"Error checking deprecation: {e}")
        return f"Error checking deprecation: {e}"


@mcp.tool
async def find_similar_patterns(
    code_snippet: str,
    intent: Optional[str] = None,
    ctx: Context = None,
) -> str:
    """Find similar usage patterns or alternative approaches.
    
    Args:
        code_snippet: Python code snippet to find patterns for
        intent: Optional description of what the code is trying to achieve
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Finding similar patterns for code snippet")
        if intent:
            await ctx.info(f"Intent: {intent}")
    
    try:
        results = await pyrag.find_similar_patterns(
            code_snippet=code_snippet,
            intent=intent,
        )
        
        if not results:
            return "No similar patterns found for the provided code snippet."
        
        # Format response
        response_parts = []
        response_parts.append("# Similar Usage Patterns")
        response_parts.append("")
        
        for i, pattern in enumerate(results[:5], 1):  # Limit to top 5
            response_parts.append(f"## Pattern {i}")
            response_parts.append(f"**Library:** {pattern.get('library', 'Unknown')}")
            response_parts.append(f"**Score:** {pattern.get('score', 0):.2f}")
            response_parts.append("")
            response_parts.append("**Code:**")
            response_parts.append("```python")
            response_parts.append(pattern.get('code', ''))
            response_parts.append("```")
            response_parts.append("")
            response_parts.append("**Description:**")
            response_parts.append(pattern.get('description', ''))
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error finding similar patterns: {e}")
        if ctx:
            await ctx.error(f"Error finding similar patterns: {e}")
        return f"Error finding similar patterns: {e}"


@mcp.tool
async def list_available_libraries(ctx: Context = None) -> str:
    """List all available libraries in the PyRAG system.
    
    Args:
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info("Listing available libraries")
    
    try:
        libraries = await pyrag.list_libraries()
        
        if not libraries:
            return "No libraries are currently indexed in the system."
        
        # Format response
        response_parts = []
        response_parts.append("# Available Libraries")
        response_parts.append("")
        
        for lib in libraries:
            status_emoji = "✅" if lib['status'] == 'indexed' else "⏳" if lib['status'] == 'indexing' else "❌"
            response_parts.append(f"## {status_emoji} {lib['name']}")
            if lib['description']:
                response_parts.append(f"**Description:** {lib['description']}")
            response_parts.append(f"**Status:** {lib['status']}")
            if lib['latest_version']:
                response_parts.append(f"**Latest Version:** {lib['latest_version']}")
            if lib['chunk_count']:
                response_parts.append(f"**Documentation Chunks:** {lib['chunk_count']}")
            response_parts.append("")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error listing libraries: {e}")
        if ctx:
            await ctx.error(f"Error listing libraries: {e}")
        return f"Error listing libraries: {e}"


@mcp.tool
async def get_library_status(
    library_name: str,
    ctx: Context = None,
) -> str:
    """Get detailed status of a specific library.
    
    Args:
        library_name: Name of the library to check
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Getting status for library: {library_name}")
    
    try:
        status = await pyrag.get_library_status(library_name)
        
        if not status:
            return f"Library '{library_name}' not found in the system."
        
        # Format response
        response_parts = []
        response_parts.append(f"# Library Status: {status['name']}")
        response_parts.append("")
        
        status_emoji = "✅" if status['status'] == 'indexed' else "⏳" if status['status'] == 'indexing' else "❌"
        response_parts.append(f"**Status:** {status_emoji} {status['status']}")
        
        if status['latest_version']:
            response_parts.append(f"**Latest Version:** {status['latest_version']}")
        
        if status['chunk_count']:
            response_parts.append(f"**Documentation Chunks:** {status['chunk_count']}")
        
        if status['last_checked']:
            response_parts.append(f"**Last Checked:** {status['last_checked']}")
        
        return "\n".join(response_parts)
        
    except Exception as e:
        logger.error(f"Error getting library status: {e}")
        if ctx:
            await ctx.error(f"Error getting library status: {e}")
        return f"Error getting library status: {e}"


async def main():
    """Main entry point for the MCP server."""
    logger.info("Starting PyRAG MCP Server")
    
    # Run the server
    await mcp.run()


if __name__ == "__main__":
    asyncio.run(main())
