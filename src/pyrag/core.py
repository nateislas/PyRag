"""Core PyRAG class for orchestrating the RAG system."""

from typing import Any, Dict, List, Optional

from .config import get_config
from .embeddings import EmbeddingService
from .logging import get_logger
from .search import SimpleSearchEngine
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
        self.search_engine = SimpleSearchEngine(
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
            # Search for API reference content
            query = f"API reference for {api_path}"
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
            api_reference = {
                "library": library,
                "api_path": api_path,
                "content": best_result["content"],
                "metadata": best_result["metadata"],
                "score": best_result["score"],
            }

            # Include examples if requested
            if include_examples:
                examples = await self.search_documentation(
                    query=f"examples for {api_path}",
                    library=library,
                    content_type="examples",
                    max_results=3,
                )
                api_reference["examples"] = [
                    {
                        "content": ex["content"],
                        "metadata": ex["metadata"],
                        "score": ex["score"],
                    }
                    for ex in examples
                ]

            return api_reference

        except Exception as e:
            self.logger.error(f"Error getting API reference: {e}")
            return None

    async def check_deprecation(
        self, library: str, api_paths: List[str]
    ) -> Dict[str, Any]:
        """Check if APIs are deprecated and suggest replacements."""
        self.logger.info("Checking deprecation", library=library, paths=api_paths)

        try:
            deprecated_apis = []
            replacement_suggestions = {}

            for api_path in api_paths:
                # Search for deprecation information
                query = f"deprecated {api_path}"
                results = await self.search_documentation(
                    query=query,
                    library=library,
                    max_results=3,
                )

                # Check if API is deprecated
                is_deprecated = any(
                    "deprecated" in result["content"].lower()
                    or "deprecation" in result["content"].lower()
                    for result in results
                )

                if is_deprecated:
                    deprecated_apis.append(api_path)

                    # Look for replacement suggestions
                    replacement_query = f"replacement alternative {api_path}"
                    replacement_results = await self.search_documentation(
                        query=replacement_query,
                        library=library,
                        max_results=2,
                    )

                    if replacement_results:
                        replacement_suggestions[api_path] = [
                            {
                                "suggestion": result["content"][:200],
                                "score": result["score"],
                            }
                            for result in replacement_results
                        ]

            return {
                "library": library,
                "deprecated_apis": deprecated_apis,
                "replacement_suggestions": replacement_suggestions,
            }

        except Exception as e:
            self.logger.error(f"Error checking deprecation: {e}")
            return {
                "library": library,
                "deprecated_apis": [],
                "replacement_suggestions": {},
            }

    async def find_similar_patterns(self, code_snippet: str) -> List[Dict[str, Any]]:
        """Find similar code patterns in documentation."""
        self.logger.info("Finding similar patterns", snippet_length=len(code_snippet))

        try:
            # Analyze the code snippet to extract patterns
            # This is a placeholder implementation
            intent = "code_pattern_search"

            # TODO: Implement pattern matching
            # 1. Extract patterns from code snippet
            # 2. Search for similar examples
            # 3. Rank by similarity
            # 4. Return structured results

            return []

        except Exception as e:
            self.logger.error(f"Error finding similar patterns: {e}")
            return []

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
            # Generate embeddings for documents
            texts = [doc["content"] for doc in documents]
            embeddings = await self.embedding_service.generate_embeddings(texts)

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
