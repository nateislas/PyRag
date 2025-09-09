"""PyRAG MCP Server using FastMCP - optimized for FastMCP Cloud deployment."""

import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP

# Import streaming capabilities
from .streaming import create_streaming_response
from ..logging import (
    log_mcp_request, 
    log_mcp_response, 
    CorrelationContext,
    MultiDimensionalSearchLogger
)

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
    # Generate correlation ID for this request
    correlation_id = str(uuid.uuid4())
    
    # Log the MCP request
    log_mcp_request("search_python_docs_comprehensive", {
        "query": query,
        "library": library,
        "response_format": response_format,
        "max_results": max_results
    }, correlation_id)

    if ctx:
        await ctx.info(f"🔍 Comprehensive search started (ID: {correlation_id})")
        await ctx.info(f"Query: {query}")
        if library:
            await ctx.info(f"Library focus: {library}")
        await ctx.info(f"Response format: {response_format}")

    try:
        with CorrelationContext(correlation_id):
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
                
            # Log successful response
            log_mcp_response("search_python_docs_comprehensive", "comprehensive", {
                "total_sections": response.get("metadata", {}).get("total_sections", 0),
                "total_content_length": response.get("metadata", {}).get("total_content_length", 0),
                "result_count": response.get("metadata", {}).get("total_results", 0),
                "multi_dimensional": search_metadata is not None
            }, correlation_id)
            
            if ctx:
                await ctx.info(f"✅ Comprehensive search completed")
                await ctx.info(f"Results: {response.get('metadata', {}).get('total_results', 0)} items")
                
            return response
        elif response_format == "standard":
            response = await _build_standard_response(results, query, library)
            log_mcp_response("search_python_docs_comprehensive", "standard", {
                "result_count": response.get("metadata", {}).get("total_results", 0)
            }, correlation_id)
            return response
        else:  # quick
            response = await _build_quick_response(results, query, library)
            log_mcp_response("search_python_docs_comprehensive", "quick", {
                "result_count": len(response.get("sections", {}).get("quick_results", {}).get("items", []))
            }, correlation_id)
            return response

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

@mcp.tool
async def search_python_docs_streaming(
    query: str,
    library: Optional[str] = None,
    max_results: int = 20,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Streaming Python documentation search with real-time progress updates.
    
    This tool provides streaming comprehensive responses with progress notifications,
    perfect for complex queries that benefit from real-time feedback.
    
    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional)
        max_results: Maximum number of results to consider (default: 20)
        ctx: MCP context for progress reporting and logging
        
    Returns:
        Dict with comprehensive streaming response including progress metadata
    """
    # Generate correlation ID for this request
    correlation_id = str(uuid.uuid4())
    
    # Log the MCP request
    log_mcp_request("search_python_docs_streaming", {
        "query": query,
        "library": library,
        "max_results": max_results
    }, correlation_id)
    
    if ctx:
        await ctx.info(f"🔍 Starting streaming search with correlation ID: {correlation_id}")
        await ctx.info(f"Query: {query}")
        if library:
            await ctx.info(f"Library filter: {library}")

    try:
        with CorrelationContext(correlation_id):
            # Get PyRAG instance
            pyrag = get_pyrag()
            
            # Create streaming response
            response = await create_streaming_response(
                pyrag=pyrag,
                query=query,
                library=library, 
                max_results=max_results,
                ctx=ctx,
                correlation_id=correlation_id
            )
            
            # Log the response
            log_mcp_response("search_python_docs_streaming", "streaming_comprehensive", {
                "total_sections": response.get("metadata", {}).get("total_sections", 0),
                "total_content_length": response.get("metadata", {}).get("total_content_length", 0),
                "result_count": response.get("metadata", {}).get("total_results", 0),
                "streaming_steps": response.get("streaming_metadata", {}).get("total_steps", 0)
            }, correlation_id)
            
            if ctx:
                await ctx.info(f"✅ Streaming search completed")
                await ctx.info(f"Results: {response.get('metadata', {}).get('total_results', 0)} items")
                await ctx.info(f"Sections: {response.get('metadata', {}).get('total_sections', 0)}")
            
            return response

    except Exception as e:
        logger.error(f"Error in streaming search: {e}")
        if ctx:
            await ctx.error(f"Streaming search failed: {e}")
        return {
            "query": query,
            "library": library,
            "error": str(e),
            "comprehensive_answer": f"Error in streaming search: {e}",
            "sections": {},
            "metadata": {"total_results": 0, "completeness_score": 0.0},
            "streaming_metadata": {"error": True, "correlation_id": correlation_id}
        }

@mcp.tool
async def get_search_status(
    correlation_id: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Get status of a search operation by correlation ID.
    
    This tool allows checking the status of long-running search operations.
    
    Args:
        correlation_id: Correlation ID from a previous search request
        ctx: MCP context for logging
        
    Returns:
        Dict with search status information
    """
    if ctx:
        await ctx.info(f"Checking status for correlation ID: {correlation_id}")
    
    # This is a placeholder - in a real implementation, you'd track operation status
    return {
        "correlation_id": correlation_id,
        "status": "completed",  # completed | running | failed
        "message": "Status tracking not yet implemented",
        "timestamp": "2024-01-01T00:00:00Z"
    }

@mcp.tool
async def health_check(ctx: Context = None) -> Dict[str, Any]:
    """Health check endpoint to verify server connectivity and basic functionality.
    
    This tool helps diagnose server issues by checking:
    - Server responsiveness
    - Basic imports
    - Configuration status
    
    Returns:
        Dict with health status information
    """
    if ctx:
        await ctx.info("Running health check...")
    
    try:
        import datetime
        health_status = {
            "status": "healthy",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "server": "PyRAG MCP Server",
            "checks": {}
        }
        
        # Check basic imports
        try:
            import sys
            import os
            health_status["checks"]["imports"] = "ok"
            health_status["checks"]["python_version"] = sys.version
        except Exception as e:
            health_status["checks"]["imports"] = f"error: {e}"
            health_status["status"] = "unhealthy"
        
        # Check if we can access the PyRAG module
        try:
            from pyrag.core import PyRAG
            health_status["checks"]["pyrag_import"] = "ok"
        except Exception as e:
            health_status["checks"]["pyrag_import"] = f"error: {e}"
            health_status["status"] = "unhealthy"
        
        # Check environment variables
        try:
            import os
            env_vars = {}
            for key in ["CHROMA_DB_PATH", "OPENAI_API_KEY", "CRAWL4AI_API_KEY"]:
                env_vars[key] = "set" if os.getenv(key) else "not_set"
            health_status["checks"]["environment"] = env_vars
        except Exception as e:
            health_status["checks"]["environment"] = f"error: {e}"
        
        if ctx:
            await ctx.info(f"Health check completed: {health_status['status']}")
        
        return health_status
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        if ctx:
            await ctx.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@mcp.tool
async def test_chromadb_connection(ctx: Context = None) -> Dict[str, Any]:
    """Test ChromaDB connectivity and basic operations.
    
    This tool helps diagnose database connectivity issues by:
    - Testing ChromaDB connection
    - Checking collection status
    - Verifying data availability
    
    Returns:
        Dict with ChromaDB connection status and metadata
    """
    if ctx:
        await ctx.info("Testing ChromaDB connection...")
    
    try:
        import datetime
        # Get PyRAG instance
        pyrag = get_pyrag()
        
        # Test ChromaDB connection through PyRAG
        db_status = {
            "status": "connected",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "checks": {}
        }
        
        # Check if we can access the storage engine
        try:
            if hasattr(pyrag, 'storage_engine'):
                storage = pyrag.storage_engine
                db_status["checks"]["storage_engine"] = "available"
                
                # Try to get collection info
                if hasattr(storage, 'get_collection_info'):
                    collection_info = storage.get_collection_info()
                    db_status["checks"]["collection_info"] = collection_info
                else:
                    db_status["checks"]["collection_info"] = "method_not_available"
            else:
                db_status["checks"]["storage_engine"] = "not_available"
                db_status["status"] = "error"
        except Exception as e:
            db_status["checks"]["storage_engine"] = f"error: {e}"
            db_status["status"] = "error"
        
        # Check if we can perform a simple search
        try:
            if ctx:
                await ctx.info("Testing basic search functionality...")
            
            # Try a very simple search
            test_results = await pyrag.search_documentation(
                query="test",
                max_results=1
            )
            
            db_status["checks"]["basic_search"] = "ok"
            db_status["checks"]["search_results_count"] = len(test_results) if test_results else 0
            
        except Exception as e:
            db_status["checks"]["basic_search"] = f"error: {e}"
            db_status["status"] = "error"
        
        if ctx:
            await ctx.info(f"ChromaDB test completed: {db_status['status']}")
        
        return db_status
        
    except Exception as e:
        logger.error(f"ChromaDB connection test failed: {e}")
        if ctx:
            await ctx.error(f"ChromaDB connection test failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

@mcp.tool
async def get_server_status(ctx: Context = None) -> Dict[str, Any]:
    """Get comprehensive server status including initialization state.
    
    This tool provides detailed information about:
    - Server initialization status
    - PyRAG instance state
    - Available tools
    - System resources
    
    Returns:
        Dict with comprehensive server status
    """
    if ctx:
        await ctx.info("Getting server status...")
    
    try:
        import datetime
        status = {
            "status": "running",
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "server_info": {},
            "pyrag_status": {},
            "tools": []
        }
        
        # Server info
        import sys
        import os
        status["server_info"] = {
            "python_version": sys.version,
            "working_directory": os.getcwd(),
            "environment": os.environ.get("ENVIRONMENT", "unknown")
        }
        
        # PyRAG status
        try:
            pyrag = get_pyrag()
            status["pyrag_status"]["initialized"] = True
            status["pyrag_status"]["instance_id"] = id(pyrag)
            
            # Check if PyRAG has required components
            components = {}
            for attr in ["storage_engine", "search_engine", "llm_client"]:
                components[attr] = hasattr(pyrag, attr)
            status["pyrag_status"]["components"] = components
            
        except Exception as e:
            status["pyrag_status"]["initialized"] = False
            status["pyrag_status"]["error"] = str(e)
            status["status"] = "error"
        
        # Available tools
        status["tools"] = [
            "search_python_docs",
            "search_python_docs_comprehensive", 
            "search_python_docs_streaming",
            "get_search_status",
            "health_check",
            "test_chromadb_connection",
            "get_server_status"
        ]
        
        if ctx:
            await ctx.info(f"Server status: {status['status']}")
        
        return status
        
    except Exception as e:
        logger.error(f"Server status check failed: {e}")
        if ctx:
            await ctx.error(f"Server status check failed: {e}")
        return {
            "status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }

logger.info("PyRAG MCP Server ready with 7 tools: search_python_docs, search_python_docs_comprehensive, search_python_docs_streaming, get_search_status, health_check, test_chromadb_connection, get_server_status")