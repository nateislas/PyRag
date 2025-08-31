"""Library management API router for PyRAG."""

from typing import Any, Dict, List, Optional
from fastapi import APIRouter, HTTPException, Query, Depends
from pydantic import BaseModel, Field

from ...core import PyRAG
from ...logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class LibraryCreateRequest(BaseModel):
    """Library creation request model."""
    name: str = Field(..., description="Library name")
    description: Optional[str] = Field(None, description="Library description")
    repository_url: Optional[str] = Field(None, description="Repository URL")
    documentation_url: Optional[str] = Field(None, description="Documentation URL")
    license: Optional[str] = Field(None, description="License information")


class LibraryResponse(BaseModel):
    """Library response model."""
    name: str
    description: Optional[str] = None
    repository_url: Optional[str] = None
    documentation_url: Optional[str] = None
    license: Optional[str] = None
    status: str
    latest_version: Optional[str] = None
    chunk_count: int = 0


class DocumentAddRequest(BaseModel):
    """Document addition request model."""
    library_name: str = Field(..., description="Library name")
    version: str = Field(..., description="Library version")
    documents: List[Dict[str, Any]] = Field(..., description="Documents to add")
    content_type: str = Field("documents", description="Type of content")


class DocumentAddResponse(BaseModel):
    """Document addition response model."""
    library: str
    version: str
    documents_added: int
    content_type: str
    status: str


def get_pyrag() -> PyRAG:
    """Dependency to get PyRAG instance."""
    return PyRAG()


@router.post("/", response_model=LibraryResponse)
async def create_library(
    request: LibraryCreateRequest,
    pyrag: PyRAG = Depends(get_pyrag)
) -> LibraryResponse:
    """Create a new library."""
    try:
        logger.info(f"Creating library: {request.name}")
        
        library = await pyrag.add_library(
            name=request.name,
            description=request.description,
            repository_url=request.repository_url,
            documentation_url=request.documentation_url,
            license=request.license
        )
        
        return LibraryResponse(
            name=library.name,
            description=library.description,
            repository_url=library.repository_url,
            documentation_url=library.documentation_url,
            license=library.license,
            status=library.indexing_status,
            latest_version=None,
            chunk_count=0
        )
        
    except Exception as e:
        logger.error(f"Library creation error: {e}")
        raise HTTPException(status_code=500, detail=f"Library creation failed: {str(e)}")


@router.get("/", response_model=List[LibraryResponse])
async def list_libraries(
    pyrag: PyRAG = Depends(get_pyrag)
) -> List[LibraryResponse]:
    """List all libraries."""
    try:
        logger.info("Listing libraries")
        
        libraries = await pyrag.list_libraries()
        
        return [
            LibraryResponse(
                name=lib["name"],
                description=lib["description"],
                repository_url=None,  # Not included in list response
                documentation_url=None,  # Not included in list response
                license=None,  # Not included in list response
                status=lib["status"],
                latest_version=lib["latest_version"],
                chunk_count=lib["chunk_count"]
            )
            for lib in libraries
        ]
        
    except Exception as e:
        logger.error(f"Library listing error: {e}")
        raise HTTPException(status_code=500, detail=f"Library listing failed: {str(e)}")


@router.get("/{library_name}", response_model=LibraryResponse)
async def get_library(
    library_name: str,
    pyrag: PyRAG = Depends(get_pyrag)
) -> LibraryResponse:
    """Get library details."""
    try:
        logger.info(f"Getting library: {library_name}")
        
        status = await pyrag.get_library_status(library_name)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Library not found: {library_name}"
            )
        
        return LibraryResponse(
            name=status["name"],
            description=None,  # Not included in status response
            repository_url=None,  # Not included in status response
            documentation_url=None,  # Not included in status response
            license=None,  # Not included in status response
            status=status["status"],
            latest_version=status["latest_version"],
            chunk_count=status["chunk_count"]
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Library retrieval error: {e}")
        raise HTTPException(status_code=500, detail=f"Library retrieval failed: {str(e)}")


@router.post("/{library_name}/documents", response_model=DocumentAddResponse)
async def add_documents(
    library_name: str,
    request: DocumentAddRequest,
    pyrag: PyRAG = Depends(get_pyrag)
) -> DocumentAddResponse:
    """Add documents to a library."""
    try:
        logger.info(f"Adding documents to library: {library_name}")
        
        result = await pyrag.add_documents(
            library_name=library_name,
            version=request.version,
            documents=request.documents,
            content_type=request.content_type
        )
        
        return DocumentAddResponse(**result)
        
    except Exception as e:
        logger.error(f"Document addition error: {e}")
        raise HTTPException(status_code=500, detail=f"Document addition failed: {str(e)}")


@router.get("/{library_name}/status")
async def get_library_status(
    library_name: str,
    pyrag: PyRAG = Depends(get_pyrag)
) -> Dict[str, Any]:
    """Get detailed library status."""
    try:
        logger.info(f"Getting library status: {library_name}")
        
        status = await pyrag.get_library_status(library_name)
        
        if not status:
            raise HTTPException(
                status_code=404,
                detail=f"Library not found: {library_name}"
            )
        
        return status
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Library status error: {e}")
        raise HTTPException(status_code=500, detail=f"Library status retrieval failed: {str(e)}")


@router.get("/{library_name}/search")
async def search_library(
    library_name: str,
    query: str = Query(..., description="Search query"),
    version: Optional[str] = Query(None, description="Specific version to search"),
    content_type: Optional[str] = Query(None, description="Type of content to search"),
    max_results: int = Query(10, description="Maximum number of results"),
    pyrag: PyRAG = Depends(get_pyrag)
) -> Dict[str, Any]:
    """Search within a specific library."""
    try:
        logger.info(f"Searching library {library_name}: {query}")
        
        results = await pyrag.search_documentation(
            query=query,
            library=library_name,
            version=version,
            content_type=content_type,
            max_results=max_results
        )
        
        return {
            "library": library_name,
            "query": query,
            "results": results,
            "total_results": len(results),
            "version": version,
            "content_type": content_type
        }
        
    except Exception as e:
        logger.error(f"Library search error: {e}")
        raise HTTPException(status_code=500, detail=f"Library search failed: {str(e)}")
