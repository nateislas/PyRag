"""Search API router for PyRAG."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...core import PyRAG
from ...logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class SearchRequest(BaseModel):
    """Search request model."""
    query: str = Field(..., description="Search query")
    library: Optional[str] = Field(None, description="Specific library to search")
    version: Optional[str] = Field(None, description="Specific version to search")
    content_type: Optional[str] = Field(None, description="Type of content to search")
    max_results: int = Field(10, description="Maximum number of results")


class SearchResponse(BaseModel):
    """Search response model."""
    query: str
    results: List[Dict[str, Any]]
    total_results: int
    library: Optional[str] = None
    version: Optional[str] = None
    content_type: Optional[str] = None


class APIReferenceRequest(BaseModel):
    """API reference request model."""
    library: str = Field(..., description="Library name")
    api_path: str = Field(..., description="API path (e.g., 'requests.Session.get')")
    include_examples: bool = Field(True, description="Include code examples")


class APIReferenceResponse(BaseModel):
    """API reference response model."""
    library: str
    api_path: str
    content: str
    metadata: Dict[str, Any]
    score: float
    examples: List[str]
    related_apis: List[Dict[str, Any]]


def get_pyrag() -> PyRAG:
    """Dependency to get PyRAG instance."""
    return PyRAG()


@router.post("/search", response_model=SearchResponse)
async def search_documentation(
    request: SearchRequest,
    pyrag: PyRAG = Depends(get_pyrag)
) -> SearchResponse:
    """Search documentation using semantic search."""
    try:
        logger.info(f"Search request: {request.query}")
        
        results = await pyrag.search_documentation(
            query=request.query,
            library=request.library,
            version=request.version,
            content_type=request.content_type,
            max_results=request.max_results
        )
        
        return SearchResponse(
            query=request.query,
            results=results,
            total_results=len(results),
            library=request.library,
            version=request.version,
            content_type=request.content_type
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.get("/search", response_model=SearchResponse)
async def search_documentation_get(
    query: str = Query(..., description="Search query"),
    library: Optional[str] = Query(None, description="Specific library to search"),
    version: Optional[str] = Query(None, description="Specific version to search"),
    content_type: Optional[str] = Query(None, description="Type of content to search"),
    max_results: int = Query(10, description="Maximum number of results"),
    pyrag: PyRAG = Depends(get_pyrag)
) -> SearchResponse:
    """Search documentation using semantic search (GET endpoint)."""
    try:
        logger.info(f"Search request (GET): {query}")
        
        results = await pyrag.search_documentation(
            query=query,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results
        )
        
        return SearchResponse(
            query=query,
            results=results,
            total_results=len(results),
            library=library,
            version=version,
            content_type=content_type
        )
        
    except Exception as e:
        logger.error(f"Search error: {e}")
        raise HTTPException(status_code=500, detail=f"Search failed: {str(e)}")


@router.post("/api-reference", response_model=APIReferenceResponse)
async def get_api_reference(
    request: APIReferenceRequest,
    pyrag: PyRAG = Depends(get_pyrag)
) -> APIReferenceResponse:
    """Get detailed API reference for specific function/class."""
    try:
        logger.info(f"API reference request: {request.library}.{request.api_path}")
        
        result = await pyrag.get_api_reference(
            library=request.library,
            api_path=request.api_path,
            include_examples=request.include_examples
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"API reference not found for {request.library}.{request.api_path}"
            )
        
        return APIReferenceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API reference error: {e}")
        raise HTTPException(status_code=500, detail=f"API reference lookup failed: {str(e)}")


@router.get("/api-reference/{library}/{api_path:path}", response_model=APIReferenceResponse)
async def get_api_reference_get(
    library: str,
    api_path: str,
    include_examples: bool = Query(True, description="Include code examples"),
    pyrag: PyRAG = Depends(get_pyrag)
) -> APIReferenceResponse:
    """Get detailed API reference for specific function/class (GET endpoint)."""
    try:
        logger.info(f"API reference request (GET): {library}.{api_path}")
        
        result = await pyrag.get_api_reference(
            library=library,
            api_path=api_path,
            include_examples=include_examples
        )
        
        if not result:
            raise HTTPException(
                status_code=404,
                detail=f"API reference not found for {library}.{api_path}"
            )
        
        return APIReferenceResponse(**result)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"API reference error: {e}")
        raise HTTPException(status_code=500, detail=f"API reference lookup failed: {str(e)}")


@router.get("/health")
async def search_health_check() -> Dict[str, Any]:
    """Health check for search functionality."""
    try:
        pyrag = PyRAG()
        
        # Test basic functionality
        test_results = await pyrag.search_documentation(
            query="test query",
            max_results=1
        )
        
        return {
            "status": "healthy",
            "service": "search",
            "test_query_successful": True,
            "vector_store_available": True,
            "embedding_service_available": True
        }
        
    except Exception as e:
        logger.error(f"Search health check failed: {e}")
        return {
            "status": "unhealthy",
            "service": "search",
            "error": str(e)
        }
