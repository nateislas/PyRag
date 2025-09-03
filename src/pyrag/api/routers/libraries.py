"""Library management API endpoints."""

from typing import Any, Dict, Optional

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import JSONResponse

from ...core import PyRAG
from ...logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/libraries", tags=["libraries"])


def get_pyrag() -> PyRAG:
    """Get PyRAG instance."""
    return PyRAG()


@router.get("/")
async def list_libraries() -> Dict[str, Any]:
    """List all libraries in the system."""
    try:
        logger.info("Listing libraries")
        
        # Since we're not using a database, return empty list for now
        # This could be enhanced to scan ChromaDB collections for library metadata
        return {
            "libraries": [],
            "total": 0,
            "message": "Library listing not implemented (ChromaDB-only mode)"
        }

    except Exception as e:
        logger.error(f"Library listing error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Library listing failed: {str(e)}"
        )


@router.get("/{library_name}/search")
async def search_library(
    library_name: str,
    query: str = Query(..., description="Search query"),
    version: Optional[str] = Query(None, description="Specific version to search"),
    content_type: Optional[str] = Query(None, description="Type of content to search"),
    max_results: int = Query(10, description="Maximum number of results"),
    pyrag: PyRAG = Depends(get_pyrag),
) -> Dict[str, Any]:
    """Search within a specific library."""
    try:
        logger.info(f"Searching library {library_name}: {query}")

        results = await pyrag.search_documentation(
            query=query,
            library=library_name,
            version=version,
            content_type=content_type,
            max_results=max_results,
        )

        return {
            "library": library_name,
            "query": query,
            "results": results,
            "total_results": len(results),
            "version": version,
            "content_type": content_type,
        }

    except Exception as e:
        logger.error(f"Library search error: {e}")
        raise HTTPException(status_code=500, detail=f"Library search failed: {str(e)}")


@router.get("/{library_name}/api-reference/{api_path:path}")
async def get_api_reference(
    library_name: str,
    api_path: str,
    include_examples: bool = Query(True, description="Include usage examples"),
    pyrag: PyRAG = Depends(get_pyrag),
) -> Dict[str, Any]:
    """Get API reference for a specific function/class."""
    try:
        logger.info(f"Getting API reference for {api_path} in {library_name}")

        api_ref = await pyrag.get_api_reference(
            library=library_name,
            api_path=api_path,
            include_examples=include_examples,
        )

        if not api_ref:
            raise HTTPException(
                status_code=404, 
                detail=f"API reference not found for {api_path} in {library_name}"
            )

        return api_ref

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API reference error: {e}")
        raise HTTPException(
            status_code=500, detail=f"API reference retrieval failed: {str(e)}"
        )


@router.post("/{library_name}/check-deprecation")
async def check_deprecation(
    library_name: str,
    api_paths: list[str],
    pyrag: PyRAG = Depends(get_pyrag),
) -> Dict[str, Any]:
    """Check if APIs are deprecated."""
    try:
        logger.info(f"Checking deprecation for {len(api_paths)} APIs in {library_name}")

        result = await pyrag.check_deprecation(
            library=library_name,
            api_paths=api_paths,
        )

        return result

    except Exception as e:
        logger.error(f"Deprecation check error: {e}")
        raise HTTPException(
            status_code=500, detail=f"Deprecation check failed: {str(e)}"
        )
