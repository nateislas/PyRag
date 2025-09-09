"""PyRAG MCP Server using FastMCP - Consolidated tool version for optimal deployment."""

import logging
import os
import sys
import uuid
from typing import Any, Dict, List, Optional

from fastmcp import Context, FastMCP

# Import streaming capabilities
from pyrag.mcp.streaming import create_streaming_response
from pyrag.logging import (
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
logger.info("PyRAG MCP Server (Consolidated) starting...")

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
async def search_docs(
    query: str,
    library: Optional[str] = None,
    format: str = "standard",
    streaming: bool = False,
    max_results: int = None,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Unified Python documentation search with flexible response formats.

    This is the main RAG tool that searches through indexed Python library documentation
    with support for different response formats and optional streaming for complex queries.

    Args:
        query: Natural language query about Python functionality
        library: Specific library to search within (optional, e.g., "fastapi", "pandas")
        format: Response format - "quick" (top 3, truncated) | "standard" (top 10, structured) | "comprehensive" (top 20+, full content)
        streaming: Enable real-time progress updates for complex searches (recommended for comprehensive format)
        max_results: Override default result limits (quick: 3, standard: 10, comprehensive: 20)
        ctx: MCP context for logging and progress reporting

    Returns:
        Dict with structured response including comprehensive_answer, sections, and metadata
    """
    # Validate format parameter
    valid_formats = ["quick", "standard", "comprehensive"]
    if format not in valid_formats:
        format = "standard"
    
    # Set intelligent defaults for max_results based on format
    if max_results is None:
        max_results = {"quick": 3, "standard": 10, "comprehensive": 20}[format]
    
    # Generate correlation ID for this request
    correlation_id = str(uuid.uuid4())
    
    # Log the MCP request
    log_mcp_request("search_docs", {
        "query": query,
        "library": library,
        "format": format,
        "streaming": streaming,
        "max_results": max_results
    }, correlation_id)

    if ctx:
        if streaming:
            await ctx.info(f"🔍 Starting streaming search (ID: {correlation_id})")
        else:
            await ctx.info(f"🔍 Searching Python docs (format: {format})")
        await ctx.info(f"Query: {query}")
        if library:
            await ctx.info(f"Library filter: {library}")

    try:
        with CorrelationContext(correlation_id):
            # Get PyRAG instance
            pyrag = get_pyrag()

            # Route to streaming or standard search based on parameters
            if streaming and format == "comprehensive":
                # Use streaming for comprehensive searches
                response = await create_streaming_response(
                    pyrag=pyrag,
                    query=query,
                    library=library,
                    max_results=max_results,
                    ctx=ctx,
                    correlation_id=correlation_id
                )
                
                # Log streaming response
                log_mcp_response("search_docs", "streaming_comprehensive", {
                    "total_sections": response.get("metadata", {}).get("total_sections", 0),
                    "total_content_length": response.get("metadata", {}).get("total_content_length", 0),
                    "result_count": response.get("metadata", {}).get("total_results", 0),
                    "streaming_steps": response.get("streaming_metadata", {}).get("total_steps", 0)
                }, correlation_id)
                
            else:
                # Use standard search for non-streaming requests
                if format == "comprehensive":
                    # Use comprehensive search for better results
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
                        "format": format,
                        "comprehensive_answer": "No relevant documentation found for your query.",
                        "sections": {},
                        "metadata": {
                            "total_results": 0,
                            "completeness_score": 0.0,
                            "response_length": 0,
                            "coverage_areas": []
                        }
                    }

                # Build response based on format
                if format == "comprehensive":
                    response = await _build_comprehensive_response(results, query, library)
                    # Add multi-dimensional metadata if available
                    if search_metadata:
                        response["metadata"]["multi_dimensional_search"] = search_metadata
                elif format == "standard":
                    response = await _build_standard_response(results, query, library)
                else:  # quick
                    response = await _build_quick_response(results, query, library)
                
                # Log standard response
                log_mcp_response("search_docs", format, {
                    "total_sections": response.get("metadata", {}).get("total_sections", 0),
                    "total_content_length": response.get("metadata", {}).get("total_content_length", 0),
                    "result_count": response.get("metadata", {}).get("total_results", 0),
                    "multi_dimensional": search_metadata is not None
                }, correlation_id)

            if ctx:
                await ctx.info(f"✅ Search completed")
                await ctx.info(f"Results: {response.get('metadata', {}).get('total_results', 0)} items")
                if response.get('metadata', {}).get('total_sections'):
                    await ctx.info(f"Sections: {response.get('metadata', {}).get('total_sections', 0)}")
                
            return response

    except Exception as e:
        logger.error(f"Error in search_docs: {e}")
        if ctx:
            await ctx.error(f"Search failed: {e}")
        return {
            "query": query,
            "library": library,
            "format": format,
            "error": str(e),
            "comprehensive_answer": f"Error searching documentation: {e}",
            "sections": {},
            "metadata": {"total_results": 0, "completeness_score": 0.0}
        }


@mcp.tool
async def diagnose(
    check_type: str = "all",
    ctx: Context = None,
) -> Dict[str, Any]:
    """Comprehensive diagnostic tool for PyRAG server health and connectivity.

    This tool runs various diagnostic checks to help troubleshoot issues:
    - health: Basic server health (imports, Python version, environment)
    - database: ChromaDB connectivity and data availability
    - server: Complete server status including PyRAG initialization
    - all: Run all diagnostic checks (default)

    Args:
        check_type: Type of diagnostic to run - "health" | "database" | "server" | "all"
        ctx: MCP context for progress reporting

    Returns:
        Dict with comprehensive diagnostic results
    """
    # Validate check_type
    valid_types = ["health", "database", "server", "all"]
    if check_type not in valid_types:
        check_type = "all"

    if ctx:
        await ctx.info(f"🔧 Running diagnostic checks: {check_type}")

    try:
        import datetime
        
        diagnostic_result = {
            "check_type": check_type,
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "overall_status": "healthy",
            "checks": {}
        }

        # Health checks
        if check_type in ["health", "all"]:
            if ctx:
                await ctx.info("Checking server health...")
            
            health_checks = {}
            
            # Basic imports
            try:
                import sys
                import os
                health_checks["imports"] = "ok"
                health_checks["python_version"] = sys.version
            except Exception as e:
                health_checks["imports"] = f"error: {e}"
                diagnostic_result["overall_status"] = "unhealthy"
            
            # PyRAG module import
            try:
                from pyrag.core import PyRAG
                health_checks["pyrag_import"] = "ok"
            except Exception as e:
                health_checks["pyrag_import"] = f"error: {e}"
                diagnostic_result["overall_status"] = "unhealthy"
            
            # Environment variables
            try:
                env_vars = {}
                for key in ["CHROMA_CLOUD", "CHROMA_CLOUD_API_KEY", "CHROMA_TENANT_ID", "CHROMA_DATABASE", "LLAMA_API_KEY", "FIRECRAWL_API_KEY"]:
                    env_vars[key] = "set" if os.getenv(key) else "not_set"
                health_checks["environment"] = env_vars
            except Exception as e:
                health_checks["environment"] = f"error: {e}"
            
            diagnostic_result["checks"]["health"] = health_checks

        # Database checks
        if check_type in ["database", "all"]:
            if ctx:
                await ctx.info("Testing ChromaDB connection...")
            
            db_checks = {}
            
            try:
                # Get PyRAG instance
                pyrag = get_pyrag()
                db_checks["pyrag_initialization"] = "ok"
                
                # Check storage engine
                if hasattr(pyrag, 'storage_engine'):
                    db_checks["storage_engine"] = "available"
                    
                    # Try to get collection info
                    if hasattr(pyrag.storage_engine, 'get_collection_info'):
                        collection_info = pyrag.storage_engine.get_collection_info()
                        db_checks["collection_info"] = collection_info
                    else:
                        db_checks["collection_info"] = "method_not_available"
                else:
                    db_checks["storage_engine"] = "not_available"
                    diagnostic_result["overall_status"] = "unhealthy"
                
                # Test basic search functionality
                try:
                    test_results = await pyrag.search_documentation(
                        query="test",
                        max_results=1
                    )
                    db_checks["basic_search"] = "ok"
                    db_checks["search_results_count"] = len(test_results) if test_results else 0
                except Exception as e:
                    db_checks["basic_search"] = f"error: {e}"
                    diagnostic_result["overall_status"] = "unhealthy"
                    
            except Exception as e:
                db_checks["connection"] = f"error: {e}"
                diagnostic_result["overall_status"] = "unhealthy"
            
            diagnostic_result["checks"]["database"] = db_checks

        # Server status checks
        if check_type in ["server", "all"]:
            if ctx:
                await ctx.info("Checking server status...")
            
            server_checks = {}
            
            # Server info
            import sys
            import os
            server_checks["server_info"] = {
                "python_version": sys.version,
                "working_directory": os.getcwd(),
                "environment": os.environ.get("ENVIRONMENT", "unknown")
            }
            
            # PyRAG status
            try:
                pyrag = get_pyrag()
                pyrag_status = {
                    "initialized": True,
                    "instance_id": id(pyrag)
                }
                
                # Check PyRAG components
                components = {}
                for attr in ["storage_engine", "search_engine", "llm_client"]:
                    components[attr] = hasattr(pyrag, attr)
                pyrag_status["components"] = components
                
                server_checks["pyrag_status"] = pyrag_status
                
            except Exception as e:
                server_checks["pyrag_status"] = {
                    "initialized": False,
                    "error": str(e)
                }
                diagnostic_result["overall_status"] = "unhealthy"
            
            # Available tools
            server_checks["available_tools"] = [
                "search_docs",
                "diagnose", 
                "get_status"
            ]
            
            diagnostic_result["checks"]["server"] = server_checks

        if ctx:
            await ctx.info(f"✅ Diagnostic completed: {diagnostic_result['overall_status']}")

        return diagnostic_result

    except Exception as e:
        logger.error(f"Diagnostic check failed: {e}")
        if ctx:
            await ctx.error(f"Diagnostic failed: {e}")
        return {
            "check_type": check_type,
            "overall_status": "error",
            "error": str(e),
            "timestamp": datetime.datetime.utcnow().isoformat()
        }


@mcp.tool
async def get_status(
    correlation_id: str,
    ctx: Context = None,
) -> Dict[str, Any]:
    """Get status of a search operation by correlation ID.
    
    This tool allows checking the status and results of search operations,
    particularly useful for tracking long-running comprehensive searches.
    
    Args:
        correlation_id: Correlation ID from a previous search request
        ctx: MCP context for logging
        
    Returns:
        Dict with operation status information
    """
    if ctx:
        await ctx.info(f"📊 Checking status for correlation ID: {correlation_id}")
    
    try:
        import datetime
        
        # This is a simplified status checker
        # In a full implementation, you'd track operation states in memory/database
        status_result = {
            "correlation_id": correlation_id,
            "status": "completed",  # completed | running | failed | not_found
            "timestamp": datetime.datetime.utcnow().isoformat(),
            "message": "Status tracking operational - search operations complete immediately in current implementation"
        }
        
        # Validate correlation ID format (UUID)
        try:
            uuid.UUID(correlation_id)
            status_result["correlation_id_valid"] = True
        except ValueError:
            status_result["correlation_id_valid"] = False
            status_result["status"] = "invalid_id"
            status_result["message"] = "Invalid correlation ID format"
        
        if ctx:
            await ctx.info(f"Status check completed: {status_result['status']}")
        
        return status_result
        
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        if ctx:
            await ctx.error(f"Status check failed: {e}")
        return {
            "correlation_id": correlation_id,
            "status": "error",
            "error": str(e),
            "timestamp": "2024-01-01T00:00:00Z"
        }


# Helper functions for response building (consolidated from original server)

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
        "format": "comprehensive",
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
            "score": result.get("score", 0.0),
            "source_url": result.get("metadata", {}).get("source_url", "")
        })
        total_content_length += len(content)
    
    comprehensive_answer = f"# {_generate_response_title(query)}\n\n"
    for section_name, section_data in sections.items():
        comprehensive_answer += f"## {section_name.title()}\n\n"
        for item in section_data["items"]:
            if item["title"]:
                comprehensive_answer += f"### {item['title']}\n"
            comprehensive_answer += f"{item['content']}\n\n"
            if item["source_url"]:
                comprehensive_answer += f"*Source: {item['source_url']}*\n\n"
    
    return {
        "query": query,
        "library": library,
        "format": "standard",
        "comprehensive_answer": comprehensive_answer,
        "sections": sections,
        "metadata": {
            "total_results": len(results),
            "total_sections": len(sections),
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
            "score": result.get("score", 0.0),
            "source_url": result.get("metadata", {}).get("source_url", "")
        })
        total_content_length += len(content)
    
    comprehensive_answer = f"# {_generate_response_title(query)}\n\n"
    for i, item in enumerate(quick_items, 1):
        if item["title"]:
            comprehensive_answer += f"## {i}. {item['title']}\n"
        comprehensive_answer += f"{item['content']}\n\n"
        if item["source_url"]:
            comprehensive_answer += f"*Source: {item['source_url']}*\n\n"
    
    return {
        "query": query,
        "library": library,
        "format": "quick",
        "comprehensive_answer": comprehensive_answer,
        "sections": {"quick_results": {"items": quick_items}},
        "metadata": {
            "total_results": len(results),
            "total_sections": 1,
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

logger.info("PyRAG MCP Server ready with 3 consolidated tools: search_docs, diagnose, get_status")