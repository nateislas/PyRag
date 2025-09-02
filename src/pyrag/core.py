"""Core PyRAG class for orchestrating the RAG system."""

from typing import Any, Dict, List, Optional

from .config import get_config
from .database import get_session
from .embeddings import EmbeddingService
from .logging import get_logger
from .models import DocumentChunk, Library, LibraryVersion
from .search import EnhancedSearchEngine
from .vector_store import VectorStore

logger = get_logger(__name__)


class PyRAG:
    """Main PyRAG class for orchestrating the RAG system."""

    def __init__(self):
        """Initialize PyRAG system."""
        self.logger = get_logger(__name__)
        self.logger.info("Initializing PyRAG system")

        # Initialize vector store
        self.vector_store = VectorStore()

        # Initialize embedding service
        self.embedding_service = EmbeddingService()

        # Initialize enhanced search engine
        self.search_engine = EnhancedSearchEngine(
            self.vector_store, self.embedding_service
        )

        # TODO: Initialize caching layer

    async def search_documentation(
        self,
        query: str,
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 10,
    ) -> List[Dict[str, Any]]:
        """Search documentation using enhanced semantic search."""
        self.logger.info(
            "Searching documentation",
            query=query,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results,
        )

        try:
            # Use enhanced search engine
            results = await self.search_engine.search(
                query=query,
                library=library,
                version=version,
                content_type=content_type,
                max_results=max_results,
            )

            # Format results for backward compatibility
            formatted_results = []
            for result in results:
                formatted_results.append(
                    {
                        "id": result["id"],
                        "content": result["content"],
                        "metadata": result["metadata"],
                        "score": result.get("final_score", result.get("score", 0.0)),
                        "library": result["metadata"].get("library"),
                        "version": result["metadata"].get("version"),
                        "content_type": result["metadata"].get("content_type"),
                        "hierarchy_path": result["metadata"].get("hierarchy_path", []),
                    }
                )

            self.logger.info(f"Found {len(formatted_results)} results for query")
            return formatted_results

        except Exception as e:
            self.logger.error(f"Error searching documentation: {e}")
            return []

    async def get_api_reference(
        self,
        library: str,
        api_path: str,
        include_examples: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Get detailed API reference for specific function/class."""
        self.logger.info(
            "Getting API reference",
            library=library,
            api_path=api_path,
            include_examples=include_examples,
        )

        try:
            # Parse API path (e.g., "requests.Session.get")
            path_parts = api_path.split(".")

            # Search for the specific API reference
            query = f"{api_path} {library}"
            results = await self.search_documentation(
                query=query,
                library=library,
                content_type="api_reference",
                max_results=5,
            )

            if not results:
                return None

            # Get the best match
            best_result = results[0]

            # Build response
            response = {
                "library": library,
                "api_path": api_path,
                "content": best_result["content"],
                "metadata": best_result["metadata"],
                "score": best_result["score"],
                "examples": [],
                "related_apis": [],
            }

            # Get examples if requested
            if include_examples:
                example_results = await self.search_documentation(
                    query=f"{api_path} example usage",
                    library=library,
                    content_type="examples",
                    max_results=3,
                )
                response["examples"] = [r["content"] for r in example_results]

            # Get related APIs
            related_results = await self.search_documentation(
                query=f"{'.'.join(path_parts[:-1])} {library}",
                library=library,
                content_type="api_reference",
                max_results=5,
            )
            response["related_apis"] = [
                {
                    "api_path": r["metadata"].get("api_path"),
                    "content": r["content"][:200] + "..."
                    if len(r["content"]) > 200
                    else r["content"],
                    "score": r["score"],
                }
                for r in related_results[:3]
            ]

            return response

        except Exception as e:
            self.logger.error(f"Error getting API reference: {e}")
            return None

    async def check_deprecation(
        self,
        library: str,
        apis: List[str],
    ) -> Dict[str, Any]:
        """Check if APIs are deprecated and get replacement suggestions."""
        self.logger.info(
            "Checking deprecation status",
            library=library,
            apis=apis,
        )

        # TODO: Implement deprecation checking
        # 1. Look up APIs in database
        # 2. Check deprecation status
        # 3. Find replacement suggestions
        # 4. Return structured response

        return {
            "library": library,
            "deprecated_apis": [],
            "replacement_suggestions": {},
        }

    async def find_similar_patterns(
        self,
        code_snippet: str,
        intent: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        """Find similar usage patterns or alternative approaches."""
        self.logger.info(
            "Finding similar patterns",
            code_snippet=code_snippet[:100] + "..."
            if len(code_snippet) > 100
            else code_snippet,
            intent=intent,
        )

        # TODO: Implement pattern matching
        # 1. Extract patterns from code snippet
        # 2. Search for similar examples
        # 3. Rank by similarity
        # 4. Return structured results

        return []

    async def add_library(
        self,
        name: str,
        description: Optional[str] = None,
        repository_url: Optional[str] = None,
        documentation_url: Optional[str] = None,
        license: Optional[str] = None,
    ) -> Library:
        """Add a new library to the system."""
        self.logger.info("Adding library", name=name)

        with get_session() as session:
            # Check if library already exists
            existing = session.query(Library).filter(Library.name == name).first()
            if existing:
                self.logger.warning("Library already exists", name=name)
                return existing

            # Create new library
            library = Library(
                name=name,
                description=description,
                repository_url=repository_url,
                documentation_url=documentation_url,
                license=license,
                indexing_status="pending",
            )

            session.add(library)
            session.commit()
            session.refresh(library)

            self.logger.info("Library added successfully", name=name, id=library.id)
            return library

    async def get_library_status(self, library_name: str) -> Optional[Dict[str, Any]]:
        """Get the status of a library."""
        self.logger.info("Getting library status", name=library_name)

        with get_session() as session:
            library = (
                session.query(Library).filter(Library.name == library_name).first()
            )
            if not library:
                return None

            # Get latest version
            latest_version = (
                session.query(LibraryVersion)
                .filter(LibraryVersion.library_id == library.id)
                .order_by(LibraryVersion.created_at.desc())
                .first()
            )

            return {
                "name": library.name,
                "status": library.indexing_status,
                "last_checked": library.last_checked,
                "latest_version": latest_version.version if latest_version else None,
                "chunk_count": latest_version.chunk_count if latest_version else 0,
            }

    async def list_libraries(self) -> List[Dict[str, Any]]:
        """List all libraries in the system."""
        self.logger.info("Listing libraries")

        with get_session() as session:
            libraries = session.query(Library).all()

            result = []
            for library in libraries:
                # Get latest version info
                latest_version = (
                    session.query(LibraryVersion)
                    .filter(LibraryVersion.library_id == library.id)
                    .order_by(LibraryVersion.created_at.desc())
                    .first()
                )

                result.append(
                    {
                        "name": library.name,
                        "description": library.description,
                        "status": library.indexing_status,
                        "latest_version": latest_version.version
                        if latest_version
                        else None,
                        "chunk_count": latest_version.chunk_count
                        if latest_version
                        else 0,
                    }
                )

            return result

    async def add_documents(
        self,
        library_name: str,
        version: str,
        documents: List[Dict[str, Any]],
        content_type: str = "documents",
    ) -> Dict[str, Any]:
        """Add documents to the vector store for a library."""
        self.logger.info(
            "Adding documents",
            library=library_name,
            version=version,
            count=len(documents),
            content_type=content_type,
        )

        try:
            # Get or create library and get its ID within session context
            with get_session() as session:
                # Check if library already exists
                library = (
                    session.query(Library).filter(Library.name == library_name).first()
                )
                if not library:
                    self.logger.info("Adding library", name=library_name)
                    library = Library(
                        name=library_name,
                        description=f"Documentation for {library_name}",
                        indexing_status="pending",
                    )
                    session.add(library)
                    session.commit()
                    session.refresh(library)
                else:
                    self.logger.warning("Library already exists", name=library_name)

                library_id = (
                    library.id
                )  # Get ID while library is still in session context

            # Generate embeddings for documents
            texts = [doc["content"] for doc in documents]
            embeddings = await self.embedding_service.embed_texts(texts)

            # Prepare documents with embeddings
            prepared_docs = []
            for i, doc in enumerate(documents):
                prepared_doc = {
                    "content": doc["content"],
                    "metadata": {
                        "library": library_name,
                        "version": version,
                        "content_type": content_type,
                        "hierarchy_path": doc.get("hierarchy_path", []),
                        "api_path": doc.get("api_path"),
                        "source_url": doc.get("source_url"),
                        "deprecated": doc.get("deprecated", False),
                        **doc.get("metadata", {}),
                    },
                    "embedding": embeddings[i],
                }
                prepared_docs.append(prepared_doc)

            # Add to vector store
            doc_ids = await self.vector_store.add_documents(
                documents=prepared_docs, collection_name=content_type
            )

            # Update library version in database
            with get_session() as session:
                # Create or update library version
                library_version = (
                    session.query(LibraryVersion)
                    .filter(
                        LibraryVersion.library_id == library_id,
                        LibraryVersion.version == version,
                    )
                    .first()
                )

                if not library_version:
                    library_version = LibraryVersion(
                        library_id=library_id,
                        version=version,
                        chunk_count=len(doc_ids),
                        indexing_status="completed",
                    )
                    session.add(library_version)
                else:
                    library_version.chunk_count = len(doc_ids)
                    library_version.indexing_status = "completed"

                # Update library status
                library_obj = (
                    session.query(Library).filter(Library.id == library_id).first()
                )
                if library_obj:
                    library_obj.indexing_status = "completed"
                    library_obj.last_checked = (
                        None  # Will be updated by monitoring system
                    )

                session.commit()

            self.logger.info(
                "Successfully added documents",
                library=library_name,
                version=version,
                count=len(doc_ids),
            )

            return {
                "library": library_name,
                "version": version,
                "documents_added": len(doc_ids),
                "content_type": content_type,
                "status": "completed",
            }

        except Exception as e:
            self.logger.error(f"Error adding documents: {e}")
            raise
