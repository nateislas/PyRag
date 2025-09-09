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
    ctx: Context = None,
) -> str:
    """Search Python library documentation with semantic understanding.

    This is the main RAG tool that searches through indexed Python library documentation
    to find relevant information based on your query.

    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional, e.g., "fastapi", "pandas")
        ctx: MCP context for logging and progress reporting
    """
    if ctx:
        await ctx.info(f"Searching Python docs for: {query}")
        if library:
            await ctx.info(f"Filtering by library: {library}")

    try:
        # Get PyRAG instance (lazy initialization)
        pyrag = get_pyrag()

        # Search documentation
        results = await pyrag.search_documentation(
            query=query,
            library=library,
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

@mcp.tool
async def search_python_docs_comprehensive(
    query: str,
    library: Optional[str] = None,
    response_format: str = "comprehensive",
    max_results: int = 20,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Comprehensive Python documentation search optimized for AI coding agents.
    
    This tool provides structured, comprehensive responses perfect for AI agents that need
    to synthesize complete answers to complex development questions.
    
    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional)
        response_format: "quick"|"standard"|"comprehensive" (default: comprehensive)
        max_results: Maximum number of results to consider (default: 20)
        ctx: MCP context for logging and progress reporting
        
    Returns:
        Dict with structured comprehensive response including sections, metadata, and coverage info
    """
    if ctx:
        await ctx.info(f"Comprehensive search for: {query}")
        if library:
            await ctx.info(f"Library focus: {library}")
        await ctx.info(f"Response format: {response_format}")

    try:
        # Get PyRAG instance
        pyrag = get_pyrag()

        # Use comprehensive search for better results
        if response_format == "comprehensive":
            # Use new multi-dimensional search
            comprehensive_result = await pyrag.search_comprehensive(
                query=query,
                library=library,
                max_results=max_results,
            )
            results = comprehensive_result["results"]
            search_metadata = comprehensive_result.get("multi_dimensional_metadata")
        else:
            # Use standard search
            results = await pyrag.search_documentation(
                query=query,
                library=library,
                max_results=max_results,
            )
            search_metadata = None

        if not results:
            return {
                "query": query,
                "library": library,
                "response_format": response_format,
                "comprehensive_answer": "No relevant documentation found for your query.",
                "sections": {},
                "metadata": {
                    "total_results": 0,
                    "completeness_score": 0.0,
                    "response_length": 0,
                    "coverage_areas": []
                }
            }

        # Build comprehensive response based on format
        if response_format == "comprehensive":
            response = await _build_comprehensive_response(results, query, library)
            # Add multi-dimensional metadata if available
            if search_metadata:
                response["metadata"]["multi_dimensional_search"] = search_metadata
            return response
        elif response_format == "standard":
            return await _build_standard_response(results, query, library)
        else:  # quick
            return await _build_quick_response(results, query, library)

    except Exception as e:
        logger.error(f"Error in comprehensive search: {e}")
        if ctx:
            await ctx.error(f"Error in comprehensive search: {e}")
        return {
            "query": query,
            "library": library,
            "error": str(e),
            "comprehensive_answer": f"Error searching documentation: {e}",
            "sections": {},
            "metadata": {"total_results": 0, "completeness_score": 0.0}
        }

async def _build_comprehensive_response(results: List[Dict[str, Any]], query: str, library: Optional[str]) -> Dict[str, Any]:
    """Build a comprehensive structured response for AI agents."""
    
    # Group results by content type and topic
    sections = {}
    total_content_length = 0
    coverage_areas = set()
    
    for result in results:
        content_type = result.get("metadata", {}).get("content_type", "general")
        main_topic = result.get("metadata", {}).get("main_topic", "Documentation")
        
        # Create section key
        section_key = f"{main_topic} ({content_type.title()})"
        
        if section_key not in sections:
            sections[section_key] = {
                "content_items": [],
                "total_length": 0,
                "source_count": 0
            }
        
        # Add full content (no truncation for comprehensive responses)
        content = result["content"]
        sections[section_key]["content_items"].append({
            "content": content,
            "source_url": result.get("metadata", {}).get("source_url", ""),
            "title": result.get("metadata", {}).get("title", ""),
            "score": result.get("score", 0.0),
            "library": result.get("metadata", {}).get("library_name", library or "unknown"),
            "version": result.get("metadata", {}).get("version", "latest")
        })
        
        sections[section_key]["total_length"] += len(content)
        sections[section_key]["source_count"] += 1
        total_content_length += len(content)
        
        # Track coverage areas
        if "key_concepts" in result.get("metadata", {}):
            key_concepts = result["metadata"]["key_concepts"]
            if isinstance(key_concepts, list):
                coverage_areas.update(key_concepts)
    
    # Build comprehensive answer by combining sections
    comprehensive_parts = [f"# {_generate_response_title(query)}"]
    
    for section_name, section_data in sections.items():
        comprehensive_parts.append(f"\n## {section_name}")
        
        for item in section_data["content_items"]:
            if item["title"]:
                comprehensive_parts.append(f"\n### {item['title']}")
            comprehensive_parts.append(f"\n{item['content']}")
            
            if item["source_url"]:
                comprehensive_parts.append(f"\n*Source: {item['source_url']}*")
        
        comprehensive_parts.append("")  # Add spacing between sections
    
    comprehensive_answer = "\n".join(comprehensive_parts)
    
    # Calculate completeness score (simple heuristic)
    completeness_score = min(1.0, len(sections) * 0.2 + min(total_content_length / 10000, 0.6))
    
    return {
        "query": query,
        "library": library,
        "response_format": "comprehensive",
        "comprehensive_answer": comprehensive_answer,
        "sections": sections,
        "metadata": {
            "total_results": len(results),
            "total_sections": len(sections),
            "total_content_length": total_content_length,
            "completeness_score": round(completeness_score, 2),
            "coverage_areas": list(coverage_areas),
            "response_type": "comprehensive_structured"
        }
    }

async def _build_standard_response(results: List[Dict[str, Any]], query: str, library: Optional[str]) -> Dict[str, Any]:
    """Build a standard structured response."""
    # Similar to comprehensive but with some content truncation
    sections = {}
    total_content_length = 0
    
    for result in results[:10]:  # Limit to top 10 for standard
        content_type = result.get("metadata", {}).get("content_type", "general")
        
        if content_type not in sections:
            sections[content_type] = {"items": []}
        
        # Truncate content for standard response
        content = result["content"]
        if len(content) > 1500:
            content = content[:1500] + "..."
        
        sections[content_type]["items"].append({
            "content": content,
            "title": result.get("metadata", {}).get("title", ""),
            "score": result.get("score", 0.0)
        })
        total_content_length += len(content)
    
    comprehensive_answer = f"# {_generate_response_title(query)}\n\n"
    for section_name, section_data in sections.items():
        comprehensive_answer += f"## {section_name.title()}\n\n"
        for item in section_data["items"]:
            if item["title"]:
                comprehensive_answer += f"### {item['title']}\n"
            comprehensive_answer += f"{item['content']}\n\n"
    
    return {
        "query": query,
        "library": library,
        "response_format": "standard",
        "comprehensive_answer": comprehensive_answer,
        "sections": sections,
        "metadata": {
            "total_results": len(results),
            "total_content_length": total_content_length,
            "response_type": "standard_structured"
        }
    }

async def _build_quick_response(results: List[Dict[str, Any]], query: str, library: Optional[str]) -> Dict[str, Any]:
    """Build a quick structured response."""
    # Take top 3 results, truncate heavily
    quick_items = []
    total_content_length = 0
    
    for result in results[:3]:
        content = result["content"]
        if len(content) > 800:
            content = content[:800] + "..."
            
        quick_items.append({
            "content": content,
            "title": result.get("metadata", {}).get("title", ""),
            "score": result.get("score", 0.0)
        })
        total_content_length += len(content)
    
    comprehensive_answer = f"# {_generate_response_title(query)}\n\n"
    for i, item in enumerate(quick_items, 1):
        if item["title"]:
            comprehensive_answer += f"## {i}. {item['title']}\n"
        comprehensive_answer += f"{item['content']}\n\n"
    
    return {
        "query": query,
        "library": library,
        "response_format": "quick",
        "comprehensive_answer": comprehensive_answer,
        "sections": {"quick_results": {"items": quick_items}},
        "metadata": {
            "total_results": len(results),
            "total_content_length": total_content_length,
            "response_type": "quick_structured"
        }
    }

def _generate_response_title(query: str) -> str:
    """Generate a helpful title for the response."""
    query_words = query.split()
    if len(query_words) > 8:
        return " ".join(query_words[:8]).title() + "..."
    return query.title()

logger.info("PyRAG MCP Server ready with 2 tools: search_python_docs, search_python_docs_comprehensive")