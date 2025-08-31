"""Simple tests for Phase 3 components."""

import pytest
from unittest.mock import Mock, patch, AsyncMock
from pyrag.graph import (
    KnowledgeGraph, 
    RelationshipExtractor, 
    QueryPlanner, 
    ReasoningEngine
)
from pyrag.vector_store import VectorStore


class TestGraphRAGSimple:
    """Simple tests for GraphRAG components."""
    
    @pytest.fixture
    def mock_vector_store(self):
        """Mock vector store for testing."""
        return Mock(spec=VectorStore)
    
    @pytest.fixture
    def knowledge_graph(self, mock_vector_store):
        """Knowledge graph instance for testing."""
        return KnowledgeGraph(mock_vector_store)
    
    @pytest.fixture
    def relationship_extractor(self):
        """Relationship extractor instance for testing."""
        return RelationshipExtractor()
    
    @pytest.fixture
    def query_planner(self):
        """Query planner instance for testing."""
        return QueryPlanner()
    
    @pytest.fixture
    def reasoning_engine(self):
        """Reasoning engine instance for testing."""
        return ReasoningEngine()
    
    def test_relationship_extractor_creation(self, relationship_extractor):
        """Test that relationship extractor can be created."""
        assert relationship_extractor is not None
        assert hasattr(relationship_extractor, 'extract_relationships')
    
    def test_api_path_extraction(self, relationship_extractor):
        """Test API path extraction."""
        content = "Use requests.Session.get() to make HTTP requests"
        
        api_paths = relationship_extractor.extract_api_paths(content)
        
        assert "requests.Session.get" in api_paths
    
    def test_dependency_extraction(self, relationship_extractor):
        """Test dependency extraction."""
        content = "This library requires requests>=2.25.0 and depends on urllib3"
        
        dependencies = relationship_extractor.extract_dependencies(content)
        
        assert len(dependencies) > 0
    
    @pytest.mark.asyncio
    async def test_query_planner_creation(self, query_planner):
        """Test that query planner can be created and used."""
        assert query_planner is not None
        assert hasattr(query_planner, 'plan_query')
        
        # Test simple query planning
        query = "How do I make HTTP requests with requests?"
        plan = await query_planner.plan_query(query)
        
        assert plan.query == query
        assert len(plan.steps) > 0
        assert plan.total_estimated_cost > 0
    
    @pytest.mark.asyncio
    async def test_reasoning_engine_creation(self, reasoning_engine):
        """Test that reasoning engine can be created and used."""
        assert reasoning_engine is not None
        assert hasattr(reasoning_engine, 'reason')
        
        # Test simple reasoning
        query = "Compare requests vs httpx for HTTP requests"
        context = {"library": "requests"}
        
        result = await reasoning_engine.reason(query, context)
        
        assert result.query == query
        assert result.confidence >= 0.0
        assert len(result.steps) >= 0
    
    @pytest.mark.asyncio
    async def test_knowledge_graph_creation(self, knowledge_graph):
        """Test that knowledge graph can be created."""
        assert knowledge_graph is not None
        assert hasattr(knowledge_graph, 'multi_hop_query')
        
        # Mock the graph database methods
        knowledge_graph.graph_db.connect = AsyncMock()
        knowledge_graph.graph_db.find_nodes = AsyncMock(return_value=[])
        knowledge_graph.graph_db.find_paths = AsyncMock(return_value=[])
        
        # Test initialization
        await knowledge_graph.initialize()
        knowledge_graph.graph_db.connect.assert_called_once()
    
    def test_complexity_analysis(self, query_planner):
        """Test query complexity analysis."""
        # Simple query
        simple_query = "How do I use requests?"
        simple_complexity = query_planner._analyze_complexity(simple_query)
        
        # Complex query
        complex_query = "Compare requests and httpx libraries and show me how to integrate them with FastAPI"
        complex_complexity = query_planner._analyze_complexity(complex_query)
        
        assert simple_complexity < complex_complexity
        assert 0.0 <= simple_complexity <= 1.0
        assert 0.0 <= complex_complexity <= 1.0
    
    def test_reasoning_type_detection(self, reasoning_engine):
        """Test reasoning type detection."""
        # Comparison query
        comparison_query = "Compare pandas and numpy"
        comparison_type = reasoning_engine._determine_reasoning_type(comparison_query)
        assert comparison_type == "comparison"
        
        # Composition query
        composition_query = "How to build a web API using FastAPI and SQLAlchemy"
        composition_type = reasoning_engine._determine_reasoning_type(composition_query)
        assert composition_type == "composition"
        
        # Synthesis query
        synthesis_query = "What is the best way to handle HTTP requests?"
        synthesis_type = reasoning_engine._determine_reasoning_type(synthesis_query)
        assert synthesis_type == "synthesis"


class TestLibraryManagementSimple:
    """Simple tests for library management components."""
    
    def test_library_discovery_creation(self):
        """Test that library discovery can be created."""
        # Import here to avoid aiohttp dependency issues
        try:
            from pyrag.libraries import LibraryDiscovery
            discovery = LibraryDiscovery()
            assert discovery is not None
            assert hasattr(discovery, 'discover_popular_libraries')
        except ImportError:
            pytest.skip("Library discovery not available")
    
    def test_library_categories(self):
        """Test library categories."""
        try:
            from pyrag.libraries import LibraryDiscovery
            discovery = LibraryDiscovery()
            
            # Test that categories exist
            assert hasattr(discovery, 'categories')
            assert 'web_frameworks' in discovery.categories
            assert 'data_science' in discovery.categories
            
            # Test that categories contain libraries
            web_frameworks = discovery.categories['web_frameworks']
            assert 'fastapi' in web_frameworks
            assert 'django' in web_frameworks
            
        except ImportError:
            pytest.skip("Library discovery not available")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
