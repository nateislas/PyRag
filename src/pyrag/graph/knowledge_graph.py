"""Knowledge graph for relationship mapping and multi-hop reasoning."""

from typing import List, Dict, Any, Optional, Set
from dataclasses import dataclass
from ..logging import get_logger
from ..vector_store import VectorStore
from .graph_db import GraphDatabase, GraphNode, GraphRelationship
from .relationship_extractor import RelationshipExtractor, Relationship

logger = get_logger(__name__)


@dataclass
class MultiHopResult:
    """Result of a multi-hop reasoning query."""
    query: str
    reasoning_steps: List[Dict[str, Any]]
    final_answer: str
    confidence: float
    supporting_evidence: List[Dict[str, Any]]
    execution_time: float


class KnowledgeGraph:
    """Knowledge graph for relationship mapping and multi-hop reasoning."""
    
    def __init__(self, vector_store: VectorStore, graph_db: Optional[GraphDatabase] = None):
        """Initialize the knowledge graph."""
        self.vector_store = vector_store
        self.graph_db = graph_db or GraphDatabase()
        self.relationship_extractor = RelationshipExtractor()
        self.logger = get_logger(__name__)
        
        # Cache for frequently accessed relationships
        self.relationship_cache: Dict[str, List[Relationship]] = {}
        
    async def initialize(self):
        """Initialize the knowledge graph."""
        try:
            await self.graph_db.connect()
            self.logger.info("Knowledge graph initialized successfully")
        except Exception as e:
            self.logger.error(f"Failed to initialize knowledge graph: {e}")
            raise
    
    async def build_graph_from_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Build knowledge graph from document chunks."""
        try:
            self.logger.info(f"Building knowledge graph from {len(documents)} documents")
            
            total_relationships = 0
            
            for doc in documents:
                # Extract relationships from document content
                relationships = self.relationship_extractor.extract_relationships(
                    doc.get("content", ""),
                    doc.get("metadata", {}).get("library")
                )
                
                # Create graph nodes and relationships
                await self._add_document_to_graph(doc, relationships)
                total_relationships += len(relationships)
            
            self.logger.info(f"Built knowledge graph with {total_relationships} relationships")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to build knowledge graph: {e}")
            return False
    
    async def _add_document_to_graph(self, doc: Dict[str, Any], relationships: List[Relationship]):
        """Add a document and its relationships to the knowledge graph."""
        try:
            # Create document node
            doc_node = GraphNode(
                id=doc.get("id", ""),
                label="Document",
                properties={
                    "content": doc.get("content", ""),
                    "library": doc.get("metadata", {}).get("library"),
                    "version": doc.get("metadata", {}).get("version"),
                    "content_type": doc.get("metadata", {}).get("content_type"),
                    "hierarchy_path": doc.get("metadata", {}).get("hierarchy_path", []),
                },
                node_type="document"
            )
            
            await self.graph_db.create_node(doc_node)
            
            # Create relationship nodes and edges
            for rel in relationships:
                # Create source node if it doesn't exist
                source_node = GraphNode(
                    id=rel.source,
                    label="API",
                    properties={
                        "name": rel.source,
                        "library": rel.properties.get("library"),
                        "type": "api_entity"
                    },
                    node_type="api"
                )
                
                # Create target node if it doesn't exist
                target_node = GraphNode(
                    id=rel.target,
                    label="API",
                    properties={
                        "name": rel.target,
                        "library": rel.properties.get("library"),
                        "type": "api_entity"
                    },
                    node_type="api"
                )
                
                await self.graph_db.create_node(source_node)
                await self.graph_db.create_node(target_node)
                
                # Create relationship
                graph_rel = GraphRelationship(
                    id=f"{rel.source}_{rel.relationship_type}_{rel.target}",
                    source_id=rel.source,
                    target_id=rel.target,
                    relationship_type=rel.relationship_type,
                    properties={
                        "confidence": rel.confidence,
                        "context": rel.context,
                        "library": rel.properties.get("library"),
                        **rel.properties
                    }
                )
                
                await self.graph_db.create_relationship(graph_rel)
                
        except Exception as e:
            self.logger.error(f"Failed to add document to graph: {e}")
    
    async def multi_hop_query(self, query: str, max_hops: int = 3) -> MultiHopResult:
        """Execute multi-hop reasoning queries."""
        try:
            self.logger.info(f"Executing multi-hop query: {query} (max {max_hops} hops)")
            
            # Parse query into reasoning steps
            reasoning_steps = await self._plan_reasoning_steps(query, max_hops)
            
            # Execute each step
            results = []
            for step in reasoning_steps:
                step_result = await self._execute_reasoning_step(step)
                results.append(step_result)
            
            # Combine results
            final_answer = await self._combine_reasoning_results(results)
            
            # Calculate confidence
            confidence = self._calculate_confidence(results)
            
            return MultiHopResult(
                query=query,
                reasoning_steps=results,
                final_answer=final_answer,
                confidence=confidence,
                supporting_evidence=results,
                execution_time=0.0  # TODO: Add timing
            )
            
        except Exception as e:
            self.logger.error(f"Failed to execute multi-hop query: {e}")
            return MultiHopResult(
                query=query,
                reasoning_steps=[],
                final_answer="Unable to process query",
                confidence=0.0,
                supporting_evidence=[],
                execution_time=0.0
            )
    
    async def _plan_reasoning_steps(self, query: str, max_hops: int) -> List[Dict[str, Any]]:
        """Plan the reasoning steps for a complex query."""
        steps = []
        
        # Simple heuristic-based planning for now
        # In production, this would use more sophisticated NLP
        
        # Step 1: Extract key entities
        entities = self._extract_entities(query)
        steps.append({
            "step": 1,
            "type": "entity_extraction",
            "entities": entities,
            "description": f"Extract key entities: {', '.join(entities)}"
        })
        
        # Step 2: Find relationships
        if len(entities) >= 2:
            steps.append({
                "step": 2,
                "type": "relationship_search",
                "entities": entities,
                "description": f"Find relationships between {entities[0]} and {entities[1]}"
            })
        
        # Step 3: Synthesize answer
        steps.append({
            "step": 3,
            "type": "synthesis",
            "description": "Combine information to answer the query"
        })
        
        return steps[:max_hops]
    
    async def _execute_reasoning_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Execute a single reasoning step."""
        step_type = step.get("type")
        
        if step_type == "entity_extraction":
            return await self._extract_entities_step(step)
        elif step_type == "relationship_search":
            return await self._search_relationships_step(step)
        elif step_type == "synthesis":
            return await self._synthesize_step(step)
        else:
            return {"error": f"Unknown step type: {step_type}"}
    
    async def _extract_entities_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Extract entities from the query."""
        entities = step.get("entities", [])
        
        # Search for entities in the knowledge graph
        found_entities = []
        for entity in entities:
            nodes = await self.graph_db.find_nodes("API", {"name": entity})
            if nodes:
                found_entities.append({
                    "name": entity,
                    "nodes": [node.id for node in nodes]
                })
        
        return {
            "step_type": "entity_extraction",
            "entities": found_entities,
            "success": len(found_entities) > 0
        }
    
    async def _search_relationships_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Search for relationships between entities."""
        entities = step.get("entities", [])
        
        if len(entities) < 2:
            return {"error": "Need at least 2 entities for relationship search"}
        
        # Find paths between entities
        paths = await self.graph_db.find_paths(
            entities[0],
            entities[1],
            max_hops=3
        )
        
        return {
            "step_type": "relationship_search",
            "paths": paths,
            "success": len(paths) > 0
        }
    
    async def _synthesize_step(self, step: Dict[str, Any]) -> Dict[str, Any]:
        """Synthesize information from previous steps."""
        # This would combine information from previous steps
        # For now, return a simple synthesis
        return {
            "step_type": "synthesis",
            "result": "Synthesized answer based on previous steps",
            "success": True
        }
    
    async def _combine_reasoning_results(self, results: List[Dict[str, Any]]) -> str:
        """Combine results from multiple reasoning steps."""
        # Simple combination for now
        # In production, this would use more sophisticated reasoning
        
        successful_steps = [r for r in results if r.get("success", False)]
        
        if not successful_steps:
            return "Unable to find relevant information"
        
        # Extract key information from successful steps
        key_info = []
        for step in successful_steps:
            if step.get("step_type") == "entity_extraction":
                entities = step.get("entities", [])
                key_info.append(f"Found entities: {', '.join([e['name'] for e in entities])}")
            elif step.get("step_type") == "relationship_search":
                paths = step.get("paths", [])
                key_info.append(f"Found {len(paths)} relationship paths")
        
        return f"Based on the analysis: {'; '.join(key_info)}"
    
    def _calculate_confidence(self, results: List[Dict[str, Any]]) -> float:
        """Calculate confidence score for the reasoning result."""
        if not results:
            return 0.0
        
        successful_steps = sum(1 for r in results if r.get("success", False))
        total_steps = len(results)
        
        return successful_steps / total_steps if total_steps > 0 else 0.0
    
    def _extract_entities(self, query: str) -> List[str]:
        """Extract key entities from a query."""
        # Simple entity extraction for now
        # In production, this would use NER or more sophisticated NLP
        
        # Look for API-like patterns
        import re
        api_pattern = r'\b\w+(?:\.\w+)+\b'
        entities = re.findall(api_pattern, query)
        
        # Also look for library names
        library_pattern = r'\b(requests|pandas|numpy|fastapi|django|sqlalchemy)\b'
        libraries = re.findall(library_pattern, query, re.IGNORECASE)
        
        return list(set(entities + libraries))
    
    async def find_related_apis(self, api_path: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Find APIs related to a given API path."""
        try:
            # Find nodes connected to this API
            related_nodes = await self.graph_db.find_nodes("API", {"name": api_path})
            
            if not related_nodes:
                return []
            
            # Find relationships involving this API
            # This would use graph traversal in production
            related_apis = []
            
            # For now, return a simple mock result
            related_apis.append({
                "api_path": f"{api_path}.related",
                "relationship_type": "similar_functionality",
                "confidence": 0.8,
                "description": "Related API with similar functionality"
            })
            
            return related_apis[:max_results]
            
        except Exception as e:
            self.logger.error(f"Failed to find related APIs: {e}")
            return []
    
    async def health_check(self) -> bool:
        """Check if the knowledge graph is healthy."""
        try:
            return await self.graph_db.health_check()
        except Exception as e:
            self.logger.error(f"Knowledge graph health check failed: {e}")
            return False
