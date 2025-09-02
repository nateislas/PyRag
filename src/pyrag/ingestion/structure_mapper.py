"""Documentation structure mapper for analyzing documentation hierarchy and relationships."""

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class DocumentationNode:
    """Represents a node in the documentation structure with enhanced metadata for RAG optimization."""

    url: str
    title: str
    content_type: str
    depth: int
    parent_url: Optional[str] = None
    children: List[str] = field(default_factory=list)
    siblings: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance_score: float = 0.0
    completeness_score: float = 0.0

    # Enhanced RAG metadata
    content_quality_score: float = 0.0
    semantic_topics: List[str] = field(default_factory=list)
    code_examples: bool = False
    api_signature: Optional[str] = None
    deprecated: bool = False
    complexity_level: str = "intermediate"  # beginner, intermediate, advanced
    last_updated: Optional[str] = None

    # Structure relationship metadata
    relationship_metadata: Dict[str, Any] = field(default_factory=dict)

    # Coverage and gap analysis metadata
    coverage_metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DocumentationStructure:
    """Complete documentation structure mapping."""

    base_url: str
    root_nodes: List[str]
    nodes: Dict[str, DocumentationNode]
    content_types: Dict[str, List[str]]
    depth_levels: Dict[int, List[str]]
    relationships: Dict[str, Dict[str, List[str]]]
    structure_metrics: Dict[str, Any]
    coverage_analysis: Dict[str, Any]


class DocumentationStructureMapper:
    """Maps and analyzes documentation structure hierarchy."""

    def __init__(self):
        self.logger = get_logger(__name__)

        # Content type patterns for classification
        self.content_patterns = {
            "api_reference": [
                r"/api/",
                r"/reference/",
                r"/docs/api/",
                r"/documentation/api/",
                r"/v\d+/",
                r"/latest/api/",
                r"/modules/",
                r"/classes/",
                r"/functions/",
            ],
            "tutorial": [
                r"/tutorial/",
                r"/guide/",
                r"/getting-started/",
                r"/docs/tutorial/",
                r"/documentation/guide/",
                r"/learn/",
                r"/examples/",
            ],
            "installation": [
                r"/install/",
                r"/installation/",
                r"/setup/",
                r"/getting-started/",
                r"/quickstart/",
            ],
            "configuration": [
                r"/config/",
                r"/configuration/",
                r"/settings/",
                r"/options/",
            ],
            "examples": [
                r"/example/",
                r"/examples/",
                r"/sample/",
                r"/demo/",
                r"/snippets/",
            ],
            "troubleshooting": [
                r"/troubleshooting/",
                r"/faq/",
                r"/help/",
                r"/support/",
                r"/debug/",
            ],
            "changelog": [
                r"/changelog/",
                r"/release-notes/",
                r"/version/",
                r"/updates/",
            ],
        }

        # URL structure patterns for depth analysis
        self.depth_patterns = {
            "shallow": [r"^/[^/]+/?$", r"^/[^/]+/[^/]+/?$"],
            "medium": [r"^/[^/]+/[^/]+/[^/]+/?$", r"^/[^/]+/[^/]+/[^/]+/[^/]+/?$"],
            "deep": [
                r"^/[^/]+/[^/]+/[^/]+/[^/]+/[^/]+",
                r"^/[^/]+/[^/]+/[^/]+/[^/]+/[^/]+/[^/]+",
            ],
        }

    def map_documentation_structure(
        self,
        urls: List[str],
        base_url: str,
        titles: Optional[Dict[str, str]] = None,
        content_types: Optional[Dict[str, str]] = None,
    ) -> DocumentationStructure:
        """Map the complete documentation structure from a list of URLs."""
        self.logger.info(f"Mapping documentation structure for {len(urls)} URLs")

        # Initialize structure
        nodes: Dict[str, DocumentationNode] = {}
        root_nodes: List[str] = []

        # Process each URL
        for url in urls:
            node = self._create_node(url, base_url, titles, content_types)
            if node:
                nodes[url] = node

                # Track root nodes
                if node.depth <= 2:
                    root_nodes.append(url)

        # Build relationships
        self._build_relationships(nodes, base_url)

        # Analyze structure
        structure_metrics = self._analyze_structure_metrics(nodes, base_url)
        coverage_analysis = self._analyze_coverage(nodes, base_url)

        # Organize by content type and depth
        content_types = self._organize_by_content_type(nodes)
        depth_levels = self._organize_by_depth(nodes)
        relationships = self._extract_relationships(nodes)

        structure = DocumentationStructure(
            base_url=base_url,
            root_nodes=root_nodes,
            nodes=nodes,
            content_types=content_types,
            depth_levels=depth_levels,
            relationships=relationships,
            structure_metrics=structure_metrics,
            coverage_analysis=coverage_analysis,
        )

        self.logger.info(f"Structure mapping complete: {len(nodes)} nodes mapped")
        return structure

    def _create_node(
        self,
        url: str,
        base_url: str,
        titles: Optional[Dict[str, str]] = None,
        content_types: Optional[Dict[str, str]] = None,
    ) -> Optional[DocumentationNode]:
        """Create a documentation node from a URL."""
        try:
            parsed = urlparse(url)
            path_parts = [p for p in parsed.path.split("/") if p]
            depth = len(path_parts)

            # Determine content type
            content_type = self._determine_content_type(url, content_types)

            # Get title
            title = (
                titles.get(url, self._extract_title_from_url(url))
                if titles
                else self._extract_title_from_url(url)
            )

            # Calculate importance score
            importance_score = self._calculate_importance_score(
                url, depth, content_type
            )

            # Calculate completeness score
            completeness_score = self._calculate_completeness_score(url, content_type)

            node = DocumentationNode(
                url=url,
                title=title,
                content_type=content_type,
                depth=depth,
                importance_score=importance_score,
                completeness_score=completeness_score,
                metadata={
                    "path_parts": path_parts,
                    "domain": parsed.netloc,
                    "query": parsed.query,
                    "fragment": parsed.fragment,
                },
            )

            return node

        except Exception as e:
            self.logger.error(f"Failed to create node for {url}: {e}")
            return None

    def _determine_content_type(
        self, url: str, content_types: Optional[Dict[str, str]] = None
    ) -> str:
        """Determine the content type of a URL."""
        if content_types and url in content_types:
            return content_types[url]

        url_lower = url.lower()

        # Check against patterns
        for content_type, patterns in self.content_patterns.items():
            for pattern in patterns:
                if re.search(pattern, url_lower):
                    return content_type

        # Default classification based on URL structure
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]

        if len(path_parts) == 0:
            return "home"
        elif len(path_parts) == 1:
            return "section"
        elif len(path_parts) == 2:
            return "subsection"
        else:
            return "detail"

    def _extract_title_from_url(self, url: str) -> str:
        """Extract a human-readable title from a URL."""
        parsed = urlparse(url)
        path_parts = [p for p in parsed.path.split("/") if p]

        if not path_parts:
            return "Home"

        # Convert path parts to title
        title_parts = []
        for part in path_parts:
            # Convert kebab-case, snake_case, or camelCase to Title Case
            if "-" in part:
                words = part.split("-")
            elif "_" in part:
                words = part.split("_")
            else:
                # Handle camelCase
                words = re.findall(
                    r"[A-Z]?[a-z]+|[A-Z]{2,}(?=[A-Z][a-z]|\d|\W|$)|\d+", part
                )

            title_parts.extend([word.title() for word in words if word])

        if not title_parts:
            return "Documentation"

        return " ".join(title_parts)

    def _calculate_importance_score(
        self, url: str, depth: int, content_type: str
    ) -> float:
        """Calculate the importance score for a documentation node."""
        score = 0.0

        # Content type importance
        type_weights = {
            "api_reference": 1.0,
            "tutorial": 0.9,
            "installation": 0.8,
            "examples": 0.7,
            "configuration": 0.6,
            "troubleshooting": 0.5,
            "changelog": 0.3,
            "home": 0.8,
            "section": 0.7,
            "subsection": 0.6,
            "detail": 0.4,
        }

        score += type_weights.get(content_type, 0.5)

        # Depth importance (shallow pages are more important)
        if depth <= 2:
            score += 0.3
        elif depth <= 4:
            score += 0.1
        else:
            score += 0.0

        # URL pattern importance
        if any(pattern in url.lower() for pattern in ["index", "main", "overview"]):
            score += 0.2

        # Normalize to 0-1 range
        return min(score, 1.0)

    def _calculate_completeness_score(self, url: str, content_type: str) -> float:
        """Calculate the completeness score for a documentation node."""
        # This is a placeholder - in practice, this would analyze actual content
        # For now, we'll use heuristics based on content type and URL structure

        base_score = 0.5

        # Content type completeness expectations
        type_completeness = {
            "api_reference": 0.9,
            "tutorial": 0.8,
            "installation": 0.7,
            "examples": 0.6,
            "configuration": 0.7,
            "troubleshooting": 0.6,
            "changelog": 0.4,
            "home": 0.6,
            "section": 0.7,
            "subsection": 0.6,
            "detail": 0.5,
        }

        return type_completeness.get(content_type, base_score)

    def _build_relationships(self, nodes: Dict[str, DocumentationNode], base_url: str):
        """Build parent-child and sibling relationships between nodes."""
        # Group nodes by depth
        nodes_by_depth = {}
        for url, node in nodes.items():
            if node.depth not in nodes_by_depth:
                nodes_by_depth[node.depth] = []
            nodes_by_depth[node.depth].append(url)

        # Build parent-child relationships
        for depth in sorted(nodes_by_depth.keys()):
            if depth == 0:
                continue

            for url in nodes_by_depth[depth]:
                node = nodes[url]
                parent_url = self._find_parent_url(
                    node, nodes_by_depth.get(depth - 1, [])
                )
                if parent_url:
                    node.parent_url = parent_url
                    nodes[parent_url].children.append(url)

        # Build sibling relationships
        for depth, urls in nodes_by_depth.items():
            if len(urls) > 1:
                for url in urls:
                    node = nodes[url]
                    siblings = [u for u in urls if u != url]
                    node.siblings = siblings

    def _find_parent_url(
        self, node: DocumentationNode, potential_parents: List[str]
    ) -> Optional[str]:
        """Find the parent URL for a given node."""
        node_path = node.metadata["path_parts"]

        for parent_url in potential_parents:
            parent_node = node.metadata.get("parent_node")
            if parent_node:
                parent_path = parent_node.metadata["path_parts"]
                if len(parent_path) == len(node_path) - 1:
                    # Check if parent path is a prefix of node path
                    if parent_path == node_path[:-1]:
                        return parent_url

        # Fallback: find by path similarity
        for parent_url in potential_parents:
            parent_parsed = urlparse(parent_url)
            parent_path = [p for p in parent_parsed.path.split("/") if p]

            if len(parent_path) == len(node_path) - 1:
                # Check if parent path is a prefix of node path
                if parent_path == node_path[:-1]:
                    return parent_url

        return None

    def _analyze_structure_metrics(
        self, nodes: Dict[str, DocumentationNode], base_url: str
    ) -> Dict[str, Any]:
        """Analyze the overall structure metrics."""
        total_nodes = len(nodes)
        if total_nodes == 0:
            return {}

        # Depth distribution
        depth_counts = {}
        for node in nodes.values():
            depth = node.depth
            depth_counts[depth] = depth_counts.get(depth, 0) + 1

        # Content type distribution
        content_type_counts = {}
        for node in nodes.values():
            content_type = node.content_type
            content_type_counts[content_type] = (
                content_type_counts.get(content_type, 0) + 1
            )

        # Relationship analysis
        nodes_with_parents = sum(1 for node in nodes.values() if node.parent_url)
        nodes_with_children = sum(1 for node in nodes.values() if node.children)
        nodes_with_siblings = sum(1 for node in nodes.values() if node.siblings)

        # Importance and completeness analysis
        importance_scores = [node.importance_score for node in nodes.values()]
        completeness_scores = [node.completeness_score for node in nodes.values()]

        metrics = {
            "total_nodes": total_nodes,
            "depth_distribution": depth_counts,
            "content_type_distribution": content_type_counts,
            "relationship_coverage": {
                "with_parents": nodes_with_parents,
                "with_children": nodes_with_children,
                "with_siblings": nodes_with_siblings,
                "parent_coverage": nodes_with_parents / total_nodes
                if total_nodes > 0
                else 0,
                "children_coverage": nodes_with_children / total_nodes
                if total_nodes > 0
                else 0,
                "sibling_coverage": nodes_with_siblings / total_nodes
                if total_nodes > 0
                else 0,
            },
            "importance_analysis": {
                "average_importance": sum(importance_scores) / len(importance_scores)
                if importance_scores
                else 0,
                "high_importance_nodes": sum(
                    1 for score in importance_scores if score > 0.8
                ),
                "low_importance_nodes": sum(
                    1 for score in importance_scores if score < 0.3
                ),
            },
            "completeness_analysis": {
                "average_completeness": sum(completeness_scores)
                / len(completeness_scores)
                if completeness_scores
                else 0,
                "high_completeness_nodes": sum(
                    1 for score in completeness_scores if score > 0.8
                ),
                "low_completeness_nodes": sum(
                    1 for score in completeness_scores if score < 0.3
                ),
            },
        }

        return metrics

    def _analyze_coverage(
        self, nodes: Dict[str, DocumentationNode], base_url: str
    ) -> Dict[str, Any]:
        """Analyze documentation coverage and identify gaps."""
        coverage = {
            "content_type_coverage": {},
            "depth_coverage": {},
            "identified_gaps": [],
            "recommendations": [],
        }

        # Analyze content type coverage
        expected_content_types = set(self.content_patterns.keys())
        actual_content_types = set(node.content_type for node in nodes.values())

        missing_types = expected_content_types - actual_content_types
        coverage["content_type_coverage"] = {
            "covered": list(actual_content_types),
            "missing": list(missing_types),
            "coverage_percentage": len(actual_content_types)
            / len(expected_content_types)
            if expected_content_types
            else 0,
        }

        # Analyze depth coverage
        max_depth = max((node.depth for node in nodes.values()), default=0)
        depth_coverage = {}
        for depth in range(max_depth + 1):
            depth_nodes = [node for node in nodes.values() if node.depth == depth]
            depth_coverage[depth] = {
                "count": len(depth_nodes),
                "types": list(set(node.content_type for node in depth_nodes)),
            }
        coverage["depth_coverage"] = depth_coverage

        # Identify gaps
        if missing_types:
            coverage["identified_gaps"].extend(
                [
                    f"Missing {content_type} documentation"
                    for content_type in missing_types
                ]
            )

        # Check for shallow documentation
        shallow_nodes = [node for node in nodes.values() if node.depth <= 2]
        if len(shallow_nodes) < 5:
            coverage["identified_gaps"].append(
                "Limited shallow documentation - may be missing overview pages"
            )

        # Check for deep documentation
        deep_nodes = [node for node in nodes.values() if node.depth >= 4]
        if len(deep_nodes) < 10:
            coverage["identified_gaps"].append(
                "Limited deep documentation - may be missing detailed guides"
            )

        # Generate recommendations
        if missing_types:
            coverage["recommendations"].append(
                f"Focus on adding missing content types: {', '.join(missing_types)}"
            )

        if len(shallow_nodes) < 5:
            coverage["recommendations"].append(
                "Add more overview and getting started pages"
            )

        if len(deep_nodes) < 10:
            coverage["recommendations"].append(
                "Expand detailed documentation and examples"
            )

        return coverage

    def _organize_by_content_type(
        self, nodes: Dict[str, DocumentationNode]
    ) -> Dict[str, List[str]]:
        """Organize nodes by content type."""
        organized = {}
        for url, node in nodes.items():
            content_type = node.content_type
            if content_type not in organized:
                organized[content_type] = []
            organized[content_type].append(url)
        return organized

    def _organize_by_depth(
        self, nodes: Dict[str, DocumentationNode]
    ) -> Dict[int, List[str]]:
        """Organize nodes by depth level."""
        organized = {}
        for url, node in nodes.items():
            depth = node.depth
            if depth not in organized:
                organized[depth] = []
            organized[depth].append(url)
        return organized

    def _extract_relationships(
        self, nodes: Dict[str, DocumentationNode]
    ) -> Dict[str, Dict[str, List[str]]]:
        """Extract relationship information from nodes."""
        relationships = {
            "parent_child": {},
            "siblings": {},
            "content_type_groups": {},
        }

        # Parent-child relationships
        for url, node in nodes.items():
            if node.parent_url:
                if node.parent_url not in relationships["parent_child"]:
                    relationships["parent_child"][node.parent_url] = []
                relationships["parent_child"][node.parent_url].append(url)

        # Sibling relationships
        for url, node in nodes.items():
            if node.siblings:
                relationships["siblings"][url] = node.siblings

        # Content type groupings
        for url, node in nodes.items():
            content_type = node.content_type
            if content_type not in relationships["content_type_groups"]:
                relationships["content_type_groups"][content_type] = []
            relationships["content_type_groups"][content_type].append(url)

        return relationships

    def get_crawl_priorities(self, structure: DocumentationStructure) -> List[str]:
        """Get prioritized URLs for crawling based on structure analysis."""
        priorities = []

        # High importance nodes first
        high_importance = [
            url for url, node in structure.nodes.items() if node.importance_score > 0.8
        ]
        priorities.extend(
            sorted(
                high_importance,
                key=lambda u: structure.nodes[u].importance_score,
                reverse=True,
            )
        )

        # Root nodes (overview pages)
        priorities.extend(
            [url for url in structure.root_nodes if url not in priorities]
        )

        # API reference pages
        api_urls = structure.content_types.get("api_reference", [])
        priorities.extend([url for url in api_urls if url not in priorities])

        # Tutorial pages
        tutorial_urls = structure.content_types.get("tutorial", [])
        priorities.extend([url for url in tutorial_urls if url not in priorities])

        # Remaining nodes by importance
        remaining = [url for url in structure.nodes.keys() if url not in priorities]
        priorities.extend(
            sorted(
                remaining,
                key=lambda u: structure.nodes[u].importance_score,
                reverse=True,
            )
        )

        return priorities

    def export_structure_visualization(
        self, structure: DocumentationStructure, output_path: str
    ):
        """Export the documentation structure for visualization."""
        try:
            import json

            # Prepare data for visualization
            viz_data = {
                "base_url": structure.base_url,
                "nodes": [
                    {
                        "id": url,
                        "title": node.title,
                        "content_type": node.content_type,
                        "depth": node.depth,
                        "importance": node.importance_score,
                        "completeness": node.completeness_score,
                        "parent": node.parent_url,
                        "children": node.children,
                        "siblings": node.siblings,
                    }
                    for url, node in structure.nodes.items()
                ],
                "metrics": structure.structure_metrics,
                "coverage": structure.coverage_analysis,
            }

            # Write to file
            with open(output_path, "w") as f:
                json.dump(viz_data, f, indent=2)

            self.logger.info(f"Structure visualization exported to: {output_path}")

        except ImportError:
            self.logger.warning("JSON module not available for structure export")
        except Exception as e:
            self.logger.error(f"Failed to export structure visualization: {e}")
