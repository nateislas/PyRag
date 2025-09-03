"""Main FastAPI application for PyRAG."""

import asyncio
import os
from contextlib import asynccontextmanager
from typing import Any, Dict

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from ..config import settings
from ..logging import get_logger, setup_logging

# Setup logging
setup_logging()
logger = get_logger(__name__)

# Global variable to store MCP server task
_mcp_server_task = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    global _mcp_server_task

    # Startup
    logger.info("Starting PyRAG API server")
    try:

        # Start MCP server if configured for HTTP transport
        if os.getenv("MCP_TRANSPORT", "stdio").lower() == "http":
            logger.info("Starting MCP server in HTTP mode")
            import os
            from pathlib import Path

            from ..mcp.server import mcp

            host = os.getenv("MCP_HOST", "0.0.0.0")
            port = int(os.getenv("MCP_PORT", "8002"))
            enable_https = os.getenv("MCP_ENABLE_HTTPS", "false").lower() == "true"

            if enable_https:
                logger.info("Starting MCP server with HTTPS")
                # Use uvicorn with SSL configuration
                import uvicorn

                cert_path = Path("certs/mcp_server.crt")
                key_path = Path("certs/mcp_server.key")

                if not cert_path.exists() or not key_path.exists():
                    logger.error(
                        "SSL certificates not found. Please run tools/generate_ssl_certs.py first"
                    )
                    raise RuntimeError("SSL certificates not found")

                # Create FastAPI app from MCP server
                app = mcp.http_app()

                # Run with HTTPS
                uvicorn_config = {
                    "ssl_keyfile": str(key_path),
                    "ssl_certfile": str(cert_path),
                    "ssl_version": "TLSv1_2",
                }

                logger.info(f"Starting MCP server on {host}:{port} with HTTPS")
                _mcp_server_task = asyncio.create_task(
                    uvicorn.run(app, host=host, port=port, **uvicorn_config)
                )
            else:
                logger.info(f"Starting MCP server on {host}:{port} with HTTP")
                _mcp_server_task = asyncio.create_task(mcp.run(host=host, port=port))

            logger.info("MCP server started successfully")

    except Exception as e:
        logger.error(f"Failed to start services: {e}")
        raise

    yield

    # Shutdown
    logger.info("Shutting down PyRAG API server")

    # Stop MCP server if running
    if _mcp_server_task and not _mcp_server_task.done():
        logger.info("Stopping MCP server")
        _mcp_server_task.cancel()
        try:
            await _mcp_server_task
        except asyncio.CancelledError:
            pass
        logger.info("MCP server stopped")


# Create FastAPI application
app = FastAPI(
    title="PyRAG API",
    description="Python Documentation RAG System for AI Coding Assistants",
    version="0.1.0",
    docs_url="/docs" if settings.is_development else None,
    redoc_url="/redoc" if settings.is_development else None,
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.api.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def root() -> Dict[str, Any]:
    """Root endpoint."""
    return {
        "message": "PyRAG API",
        "version": "0.1.0",
        "status": "running",
        "environment": settings.environment,
    }


@app.get("/health")
async def health_check() -> Dict[str, Any]:
    """Health check endpoint."""
    try:
        from ..core import PyRAG

        # Initialize PyRAG to test services
        pyrag = PyRAG()

        # Test vector store
        vector_store_healthy = True
        try:
            stats = await pyrag.vector_store.list_collections()
        except Exception as e:
            logger.error(f"Vector store health check failed: {e}")
            vector_store_healthy = False

        # Test embedding service
        embedding_healthy = True
        try:
            embedding_healthy = await pyrag.embedding_service.health_check()
        except Exception as e:
            logger.error(f"Embedding service health check failed: {e}")
            embedding_healthy = False

        # Determine overall status
        overall_healthy = vector_store_healthy and embedding_healthy

        return {
            "status": "healthy" if overall_healthy else "unhealthy",
            "environment": settings.environment,
            "version": "0.1.0",
            "services": {
                "api": "healthy",
                "database": "healthy",  # TODO: Add actual database health check
                "vector_store": "healthy" if vector_store_healthy else "unhealthy",
                "embedding_service": "healthy" if embedding_healthy else "unhealthy",
            },
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        raise HTTPException(status_code=503, detail="Service unhealthy")


@app.get("/info")
async def info() -> Dict[str, Any]:
    """API information endpoint."""
    return {
        "name": "PyRAG",
        "description": "Python Documentation RAG System for AI Coding Assistants",
        "version": "0.1.0",
        "environment": settings.environment,
        "features": {
            "documentation_search": True,
            "api_reference": True,
            "deprecation_checking": True,
            "mcp_integration": True,
        },
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler."""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"},
    )


# Import and include routers
from .routers import libraries, search

app.include_router(search.router, prefix="/api/v1/search", tags=["search"])
app.include_router(libraries.router, prefix="/api/v1/libraries", tags=["libraries"])
