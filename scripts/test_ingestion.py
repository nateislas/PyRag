#!/usr/bin/env python3
"""Test script for documentation ingestion pipeline."""

import asyncio
import sys
import os
from pathlib import Path

# Add the src directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.ingestion import DocumentationIngestionPipeline, IngestionConfig
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.graph import KnowledgeGraph
from pyrag.libraries import LibraryManager
from pyrag.logging import setup_logging

# Setup logging
setup_logging()


async def test_ingestion_pipeline():
    """Test the documentation ingestion pipeline with real documentation."""
    
    print("🚀 Testing Documentation Ingestion Pipeline")
    print("=" * 50)
    
    # Initialize components
    print("📦 Initializing components...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    knowledge_graph = KnowledgeGraph(vector_store)
    library_manager = LibraryManager()
    
    # Initialize ingestion pipeline
    pipeline = DocumentationIngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        knowledge_graph=knowledge_graph,
        library_manager=library_manager
    )
    
    # Test health check
    print("🏥 Running health check...")
    health_status = await pipeline.health_check()
    for component, status in health_status.items():
        status_icon = "✅" if status else "❌"
        print(f"  {status_icon} {component}: {'Healthy' if status else 'Unhealthy'}")
    
    # Define test libraries with their documentation URLs
    test_libraries = [
        {
            "name": "requests",
            "url": "https://docs.python-requests.org/en/latest/",
            "version": "2.31.0"
        },
        {
            "name": "fastapi",
            "url": "https://fastapi.tiangolo.com/",
            "version": "0.104.1"
        },
        {
            "name": "pandas",
            "url": "https://pandas.pydata.org/docs/",
            "version": "2.1.4"
        }
    ]
    
    # Create ingestion configs
    configs = []
    for lib in test_libraries:
        config = IngestionConfig(
            library_name=lib["name"],
            docs_url=lib["url"],
            version=lib["version"],
            crawl_options={
                "crawl_entire_site": False,  # Start with single page for testing
                "pageOptions": {
                    "onlyMainContent": True,
                    "includeMarkdown": True,
                    "waitFor": 3000,  # Wait for dynamic content
                }
            }
        )
        configs.append(config)
    
    # Test ingestion for each library
    print("\n📚 Testing documentation ingestion...")
    for config in configs:
        print(f"\n🔍 Ingesting {config.library_name} documentation...")
        print(f"   URL: {config.docs_url}")
        
        try:
            result = await pipeline.ingest_library_documentation(config)
            
            if result.success:
                print(f"   ✅ Success! {result.total_documents} documents, {result.total_chunks} chunks")
                print(f"   📊 Stats: {result.processing_stats}")
            else:
                print(f"   ❌ Failed: {result.errors}")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Test search functionality
    print("\n🔍 Testing search functionality...")
    try:
        # Search for requests documentation
        results = await vector_store.search_documents(
            query="HTTP requests authentication",
            metadata_filter={"library": "requests"},
            limit=5
        )
        
        print(f"   Found {len(results)} results for 'HTTP requests authentication'")
        for i, result in enumerate(results[:3]):
            print(f"   {i+1}. {result['content'][:100]}...")
            
    except Exception as e:
        print(f"   ❌ Search error: {e}")
    
    # Test knowledge graph functionality
    print("\n🧠 Testing knowledge graph functionality...")
    try:
        # Test multi-hop query
        result = await knowledge_graph.multi_hop_query(
            "How to make authenticated HTTP requests with requests library?",
            max_hops=2
        )
        
        print(f"   Multi-hop query result: {result.final_answer[:200]}...")
        
    except Exception as e:
        print(f"   ❌ Knowledge graph error: {e}")
    
    print("\n🎉 Ingestion pipeline test completed!")


async def test_single_library():
    """Test ingestion for a single library (for debugging)."""
    
    print("🧪 Testing single library ingestion...")
    
    # Initialize components
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Initialize ingestion pipeline (without knowledge graph for simplicity)
    pipeline = DocumentationIngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service
    )
    
    # Test with a simple library
    config = IngestionConfig(
        library_name="requests",
        docs_url="https://docs.python-requests.org/en/latest/user/quickstart/",
        version="2.31.0",
        crawl_options={
            "crawl_entire_site": False,
            "pageOptions": {
                "onlyMainContent": True,
                "includeMarkdown": True,
                "waitFor": 2000,
            }
        }
    )
    
    # Initialize Firecrawl client with API key
    from pyrag.ingestion import FirecrawlClient
    firecrawl_client = FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9")
    
    # Test Firecrawl connection first
    print("🔗 Testing Firecrawl connection...")
    async with firecrawl_client as client:
        health = await client.health_check()
        print(f"   Firecrawl health: {'✅' if health else '❌'}")
        
        if health:
            print("   Testing single page scrape...")
            try:
                doc = await client.scrape_url("https://docs.python-requests.org/en/latest/user/quickstart/")
                print(f"   ✅ Successfully scraped: {doc.title}")
                print(f"   📄 Content length: {len(doc.content)} characters")
                print(f"   📝 Markdown length: {len(doc.markdown)} characters")
            except Exception as e:
                print(f"   ❌ Scrape failed: {e}")
    
    # Now test the full pipeline
    print("\n🚀 Testing full ingestion pipeline...")
    
    print(f"📚 Ingesting {config.library_name} from {config.docs_url}")
    
    try:
        result = await pipeline.ingest_library_documentation(config)
        
        if result.success:
            print(f"✅ Success! {result.total_documents} documents, {result.total_chunks} chunks")
            print(f"📊 Processing stats: {result.processing_stats}")
        else:
            print(f"❌ Failed: {result.errors}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    # Check if we want to test a single library
    if len(sys.argv) > 1 and sys.argv[1] == "--single":
        asyncio.run(test_single_library())
    else:
        asyncio.run(test_ingestion_pipeline())
