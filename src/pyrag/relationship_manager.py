"""Document relationship manager for enhanced RAG context expansion."""

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

from .logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentRelationship:
    """Represents a relationship between documents."""

    source_id: str
    target_id: str
    relationship_type: str  # "parent", "child", "sibling", "related"
    strength: float  # 0.0 to 1.0
    metadata: Dict[str, Any]


@dataclass
class ContextExpansionResult:
    """Result of context expansion operation."""

    primary_results: List[Dict[str, Any]]
    expanded_results: List[Dict[str, Any]]
    relationship_graph: Dict[str, List[DocumentRelationship]]
    expansion_metadata: Dict[str, Any]


class RelationshipManager:
    """Manages document relationships and context expansion for enhanced RAG."""

    def __init__(self, vector_store):
        """Initialize the relationship manager."""
        self.vector_store = vector_store
        self.logger = get_logger(__name__)

        # Relationship type weights for context expansion
        self.relationship_weights = {
            "parent": 0.9,  # Parent documents are highly relevant
            "child": 0.7,  # Child documents provide detail
            "sibling": 0.8,  # Sibling documents are closely related
            "related": 0.6,  # General related documents
            "context": 0.5,  # Contextual background
        }

    async def expand_search_context(
        self,
        primary_results: List[Dict[str, Any]],
        max_expansion: int = 5,
        relationship_threshold: float = 0.5,
    ) -> ContextExpansionResult:
        """Expand search results using hierarchical relationships."""

        self.logger.info(
            f"Expanding context for {len(primary_results)} primary results"
        )

        # Build relationship graph from primary results
        relationship_graph = self._build_relationship_graph(primary_results)

        # Find related documents
        expanded_results = await self._find_related_documents(
            primary_results, relationship_graph, max_expansion, relationship_threshold
        )

        # Group and rank expanded results
        grouped_results = self._group_expanded_results(
            primary_results, expanded_results
        )

        # Calculate expansion metadata
        expansion_metadata = self._calculate_expansion_metadata(
            primary_results, expanded_results, relationship_graph
        )

        self.logger.info(
            f"Context expansion added {len(expanded_results)} related documents"
        )

        return ContextExpansionResult(
            primary_results=primary_results,
            expanded_results=expanded_results,
            relationship_graph=relationship_graph,
            expansion_metadata=expansion_metadata,
        )

    def _build_relationship_graph(
        self, results: List[Dict[str, Any]]
    ) -> Dict[str, List[DocumentRelationship]]:
        """Build a graph of document relationships from search results."""

        relationship_graph = defaultdict(list)

        for result in results:
            doc_id = result.get("id", "")
            metadata = result.get("metadata", {})

            # Extract relationship metadata
            relationship_metadata = metadata.get("relationship_metadata", {})

            # Add parent relationship
            if "parent" in relationship_metadata:
                parent_id = relationship_metadata["parent"]
                relationship_graph[doc_id].append(
                    DocumentRelationship(
                        source_id=doc_id,
                        target_id=parent_id,
                        relationship_type="parent",
                        strength=0.9,
                        metadata={"source": "metadata"},
                    )
                )

            # Add child relationships
            children = relationship_metadata.get("children", [])
            for child_id in children:
                relationship_graph[doc_id].append(
                    DocumentRelationship(
                        source_id=doc_id,
                        target_id=child_id,
                        relationship_type="child",
                        strength=0.7,
                        metadata={"source": "metadata"},
                    )
                )

            # Add sibling relationships
            siblings = relationship_metadata.get("siblings", [])
            for sibling_id in siblings:
                relationship_graph[doc_id].append(
                    DocumentRelationship(
                        source_id=doc_id,
                        target_id=sibling_id,
                        relationship_type="sibling",
                        strength=0.8,
                        metadata={"source": "metadata"},
                    )
                )

        return dict(relationship_graph)

    async def _find_related_documents(
        self,
        primary_results: List[Dict[str, Any]],
        relationship_graph: Dict[str, List[DocumentRelationship]],
        max_expansion: int,
        relationship_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Find related documents based on relationship graph."""

        related_docs = []
        processed_ids = set()

        # Collect all related document IDs
        related_ids = set()
        for relationships in relationship_graph.values():
            for rel in relationships:
                if rel.strength >= relationship_threshold:
                    related_ids.add(rel.target_id)

        # Remove IDs that are already in primary results
        primary_ids = {result.get("id", "") for result in primary_results}
        related_ids -= primary_ids

        # Limit expansion
        related_ids = list(related_ids)[:max_expansion]

        # Fetch related documents from vector store
        for doc_id in related_ids:
            try:
                # TODO: Implement document retrieval by ID
                # For now, we'll create placeholder documents
                related_doc = {
                    "id": doc_id,
                    "content": f"Related document {doc_id}",
                    "metadata": {
                        "relationship_type": "context_expansion",
                        "relationship_to": list(primary_ids)[0] if primary_ids else "",
                        "content_type": "related",
                    },
                    "score": 0.5,  # Lower score for expanded results
                    "final_score": 0.5,
                }
                related_docs.append(related_doc)
                processed_ids.add(doc_id)

            except Exception as e:
                self.logger.warning(f"Failed to fetch related document {doc_id}: {e}")
                continue

        return related_docs

    def _group_expanded_results(
        self,
        primary_results: List[Dict[str, Any]],
        expanded_results: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Group and organize expanded results for better presentation."""

        # Create groups based on content type and relationships
        grouped = defaultdict(list)

        # Group primary results
        for result in primary_results:
            content_type = result.get("metadata", {}).get("content_type", "unknown")
            grouped[f"primary_{content_type}"].append(result)

        # Group expanded results
        for result in expanded_results:
            content_type = result.get("metadata", {}).get("content_type", "related")
            grouped[f"expanded_{content_type}"].append(result)

        # Flatten groups while maintaining order
        final_results = []

        # Add primary results first
        for group_key in sorted(grouped.keys()):
            if group_key.startswith("primary_"):
                final_results.extend(grouped[group_key])

        # Add expanded results
        for group_key in sorted(grouped.keys()):
            if group_key.startswith("expanded_"):
                final_results.extend(grouped[group_key])

        return final_results

    def _calculate_expansion_metadata(
        self,
        primary_results: List[Dict[str, Any]],
        expanded_results: List[Dict[str, Any]],
        relationship_graph: Dict[str, List[DocumentRelationship]],
    ) -> Dict[str, Any]:
        """Calculate metadata about the context expansion."""

        # Count relationship types
        relationship_counts = defaultdict(int)
        for relationships in relationship_graph.values():
            for rel in relationships:
                relationship_counts[rel.relationship_type] += 1

        # Calculate expansion ratios
        total_primary = len(primary_results)
        total_expanded = len(expanded_results)
        expansion_ratio = total_expanded / total_primary if total_primary > 0 else 0

        # Calculate average relationship strength
        total_strength = 0
        total_relationships = 0
        for relationships in relationship_graph.values():
            for rel in relationships:
                total_strength += rel.strength
                total_relationships += 1

        avg_strength = (
            total_strength / total_relationships if total_relationships > 0 else 0
        )

        return {
            "total_primary_results": total_primary,
            "total_expanded_results": total_expanded,
            "expansion_ratio": expansion_ratio,
            "relationship_counts": dict(relationship_counts),
            "average_relationship_strength": avg_strength,
            "total_relationships": total_relationships,
        }

    def discover_semantic_relationships(
        self, documents: List[Dict[str, Any]], similarity_threshold: float = 0.7
    ) -> List[DocumentRelationship]:
        """Discover semantic relationships between documents based on content similarity."""

        self.logger.info(
            f"Discovering semantic relationships between {len(documents)} documents"
        )

        relationships = []

        # Simple semantic relationship discovery based on content overlap
        # In a production system, this would use embeddings and semantic similarity
        for i, doc1 in enumerate(documents):
            for j, doc2 in enumerate(documents[i + 1 :], i + 1):
                # Calculate simple content similarity (word overlap)
                similarity = self._calculate_content_similarity(doc1, doc2)

                if similarity >= similarity_threshold:
                    # Determine relationship type based on similarity and metadata
                    relationship_type = self._determine_relationship_type(
                        doc1, doc2, similarity
                    )

                    relationship = DocumentRelationship(
                        source_id=doc1.get("id", ""),
                        target_id=doc2.get("id", ""),
                        relationship_type=relationship_type,
                        strength=similarity,
                        metadata={
                            "similarity_score": similarity,
                            "discovery_method": "semantic_analysis",
                        },
                    )
                    relationships.append(relationship)

        self.logger.info(f"Discovered {len(relationships)} semantic relationships")
        return relationships

    def _calculate_content_similarity(
        self, doc1: Dict[str, Any], doc2: Dict[str, Any]
    ) -> float:
        """Calculate simple content similarity between two documents."""

        content1 = doc1.get("content", "").lower().split()
        content2 = doc2.get("content", "").lower().split()

        if not content1 or not content2:
            return 0.0

        # Calculate Jaccard similarity
        set1 = set(content1)
        set2 = set(content2)

        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))

        return intersection / union if union > 0 else 0.0

    def _determine_relationship_type(
        self, doc1: Dict[str, Any], doc2: Dict[str, Any], similarity: float
    ) -> str:
        """Determine the type of relationship between two documents."""

        metadata1 = doc1.get("metadata", {})
        metadata2 = doc2.get("metadata", {})

        # Check if they're the same content type
        content_type1 = metadata1.get("content_type", "")
        content_type2 = metadata2.get("content_type", "")

        if content_type1 == content_type2:
            if similarity > 0.8:
                return "sibling"
            else:
                return "related"

        # Check for hierarchical relationships based on content type
        if content_type1 == "api_reference" and content_type2 in [
            "examples",
            "tutorials",
        ]:
            return "parent"  # API reference is parent to examples/tutorials
        elif (
            content_type1 in ["examples", "tutorials"]
            and content_type2 == "api_reference"
        ):
            return "child"  # Examples/tutorials are children of API reference

        # Default to related
        return "related"

    def create_relationship_summary(
        self, relationships: List[DocumentRelationship]
    ) -> Dict[str, Any]:
        """Create a summary of document relationships."""

        if not relationships:
            return {"message": "No relationships found"}

        # Group by relationship type
        by_type = defaultdict(list)
        for rel in relationships:
            by_type[rel.relationship_type].append(rel)

        # Calculate statistics
        total_relationships = len(relationships)
        avg_strength = sum(rel.strength for rel in relationships) / total_relationships

        # Find strongest relationships
        strongest = sorted(relationships, key=lambda x: x.strength, reverse=True)[:5]

        return {
            "total_relationships": total_relationships,
            "relationship_types": dict(by_type),
            "average_strength": avg_strength,
            "strongest_relationships": [
                {
                    "source": rel.source_id,
                    "target": rel.target_id,
                    "type": rel.relationship_type,
                    "strength": rel.strength,
                }
                for rel in strongest
            ],
        }
