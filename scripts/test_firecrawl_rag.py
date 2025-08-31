#!/usr/bin/env python3
"""Test Firecrawl documentation ingestion into PyRAG system."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.ingestion import FirecrawlClient, DocumentationIngestionPipeline
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService

async def test_firecrawl_rag():
    """Test complete Firecrawl documentation ingestion and RAG pipeline."""
    
    print("🔥 Testing Firecrawl Documentation RAG Pipeline")
    print("=" * 55)
    
    # Initialize components
    print("📚 Initializing PyRAG components...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Test Firecrawl client
    print("\n🔗 Testing Firecrawl connection...")
    async with FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9") as client:
        health = await client.health_check()
        print(f"   Firecrawl health: {'✅' if health else '❌'}")
        
        if not health:
            print("❌ Firecrawl not accessible")
            return
        
        # Test single page extraction first
        print("\n📄 Testing Firecrawl intro page extraction...")
        try:
            intro_url = "https://docs.firecrawl.dev/introduction"
            doc = await client.scrape_url(intro_url)
            
            print(f"   ✅ Successfully extracted intro page")
            print(f"   📄 Title: {doc.title[:100]}...")
            print(f"   📝 Content length: {len(doc.content)} characters")
            print(f"   🔗 Links found: {len(doc.metadata.get('links', []))}")
            
            if doc.content:
                preview = doc.content[:200].replace('\n', ' ').strip()
                print(f"   📖 Content preview: {preview}...")
            
        except Exception as e:
            print(f"   ❌ Failed to extract intro page: {e}")
            return
    
    # Test full ingestion pipeline
    print("\n🚀 Testing full Firecrawl documentation ingestion...")
    try:
        pipeline = DocumentationIngestionPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service,
            firecrawl_api_key="fc-efb6ba5edf62402a9845440bff5c03a9"
        )
        
        # Create ingestion config for Firecrawl docs
        from pyrag.ingestion import IngestionConfig
        
        config = IngestionConfig(
            library_name="firecrawl",
            version="2.0.0",
            docs_url="https://docs.firecrawl.dev/introduction",
            crawl_options={
                "crawl_entire_site": False,  # Just scrape the intro page for now
                "max_pages": 1
            }
        )
        
        print(f"   📚 Ingesting Firecrawl documentation...")
        result = await pipeline.ingest_library_documentation(config)
        
        if result.success:
            print(f"   ✅ Successfully ingested Firecrawl documentation")
            print(f"   📊 Documents processed: {result.total_documents}")
            print(f"   📄 Chunks created: {result.total_chunks}")
            print(f"   📈 Processing stats: {result.processing_stats}")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            
            # Test various search queries
            search_queries = [
                "how to scrape a website",
                "firecrawl API features",
                "extract structured data",
                "crawl documentation"
            ]
            
            for query in search_queries:
                print(f"\n   🔍 Searching for: '{query}'")
                try:
                    search_results = await vector_store.search(query, n_results=3)
                    print(f"   📊 Found {len(search_results)} results")
                    
                    for i, result in enumerate(search_results[:2]):
                        title = result["metadata"].get('title', 'No title')
                        content_preview = result["content"][:100].replace('\n', ' ').strip()
                        print(f"   {i+1}. {title[:60]}...")
                        print(f"      {content_preview}...")
                        
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
            
            # Test specific API reference search
            print(f"\n   🔍 Searching for API reference...")
            try:
                api_results = await vector_store.search("scrape method API", n_results=2)
                print(f"   📊 Found {len(api_results)} API reference results")
                
                for i, result in enumerate(api_results):
                    title = result["metadata"].get('title', 'No title')
                    print(f"   {i+1}. {title[:80]}...")
                    
            except Exception as e:
                print(f"   ❌ API search failed: {e}")
                
        else:
            print(f"   ❌ Ingestion failed: {result.errors}")
            
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_firecrawl_rag())
