#!/usr/bin/env python3
"""Test PyTorch documentation ingestion with Firecrawl."""

import asyncio
import sys
import os
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.ingestion import FirecrawlClient, ScrapedDocument, DocumentationIngestionPipeline
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.libraries import LibraryManager

async def test_pytorch_ingestion():
    """Test PyTorch documentation ingestion."""
    
    print("🔥 Testing PyTorch Documentation Ingestion")
    print("=" * 50)
    
    # Initialize components
    print("📚 Initializing components...")
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
        
        # Test single page extraction
        print("\n📄 Testing PyTorch main page extraction...")
        try:
            pytorch_url = "https://docs.pytorch.org/docs/stable/index.html"
            doc = await client.scrape_url(pytorch_url)
            
            print(f"   ✅ Successfully extracted PyTorch main page")
            print(f"   📄 Title: {doc.title[:100]}...")
            print(f"   📝 Content length: {len(doc.content)} characters")
            print(f"   🔗 Links found: {len(doc.metadata.get('links', []))}")
            
            # Show a snippet of the content
            if doc.content:
                print(f"   📖 Content preview: {doc.content[:200]}...")
            
        except Exception as e:
            print(f"   ❌ Failed to extract PyTorch main page: {e}")
            return
        
        # Test API reference page
        print("\n🔧 Testing PyTorch API reference extraction...")
        try:
            api_url = "https://docs.pytorch.org/docs/stable/torch.html"
            api_doc = await client.scrape_url(api_url)
            
            print(f"   ✅ Successfully extracted PyTorch API reference")
            print(f"   📄 Title: {api_doc.title[:100]}...")
            print(f"   📝 Content length: {len(api_doc.content)} characters")
            
            # Check if we got API-related content
            if "tensor" in api_doc.content.lower() or "torch" in api_doc.content.lower():
                print(f"   ✅ Content appears to be API-related")
            else:
                print(f"   ⚠️  Content may not be API-related")
                
        except Exception as e:
            print(f"   ❌ Failed to extract API reference: {e}")
    
    # Test full ingestion pipeline
    print("\n🚀 Testing full PyTorch ingestion pipeline...")
    try:
        pipeline = DocumentationIngestionPipeline(
            vector_store=vector_store,
            embedding_service=embedding_service
        )
        
        # Create ingestion config for PyTorch
        config = {
            "library_name": "pytorch",
            "version": "2.1.0",
            "documentation_url": "https://docs.pytorch.org/docs/stable/",
            "api_reference_url": "https://docs.pytorch.org/docs/stable/torch.html",
            "tutorials_url": "https://docs.pytorch.org/docs/stable/tutorials/index.html",
            "max_pages": 5  # Limit for testing
        }
        
        result = await pipeline.ingest_library(config)
        
        if result.success:
            print(f"   ✅ Successfully ingested PyTorch documentation")
            print(f"   📊 Pages processed: {result.pages_processed}")
            print(f"   📄 Chunks created: {result.chunks_created}")
            print(f"   ⏱️  Processing time: {result.processing_time:.2f}s")
            
            # Test search
            print("\n🔍 Testing search functionality...")
            search_results = await vector_store.search("tensor operations", limit=3)
            print(f"   📊 Found {len(search_results)} results for 'tensor operations'")
            
            for i, result in enumerate(search_results[:2]):
                print(f"   {i+1}. {result.metadata.get('title', 'No title')[:80]}...")
                
        else:
            print(f"   ❌ Ingestion failed: {result.errors}")
            
    except Exception as e:
        print(f"   ❌ Pipeline test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_pytorch_ingestion())
