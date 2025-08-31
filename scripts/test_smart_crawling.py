#!/usr/bin/env python3
"""Test smart crawling with LLM-guided link filtering for documentation ingestion."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.ingestion.ingestion_pipeline import DocumentationIngestionPipeline, IngestionConfig
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_smart_crawling():
    """Test smart crawling with comprehensive documentation ingestion."""
    
    print("🕷️  Testing Smart Crawling with LLM-Guided Link Filtering")
    print("=" * 60)
    
    # Initialize PyRAG components
    print("📚 Initializing PyRAG components...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Test configuration for Firecrawl documentation
    config = IngestionConfig(
        library_name="firecrawl",
        version="2.0.0",
        docs_url="https://docs.firecrawl.dev/introduction",
        crawl_options={
            "crawl_entire_site": True,
            "max_pages": 10,  # Limit for testing
            "use_llm_filtering": False,  # Disable LLM for now (no client)
            "exclude_patterns": [
                "linkedin.com", "twitter.com", "github.com",
                "zapier.com", "pabbly.com", "facebook.com",
                "youtube.com", "discord.com", "slack.com"
            ],
            "include_patterns": [
                "/docs/", "/guide/", "/tutorial/", "/api/", "/reference/",
                "/introduction", "/quickstart", "/examples", "/learn/"
            ]
        }
    )
    
    print(f"🎯 Target: {config.docs_url}")
    print(f"📋 Crawl options: {config.crawl_options}")
    
    # Initialize pipeline
    pipeline = DocumentationIngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        firecrawl_api_key="fc-efb6ba5edf62402a9845440bff5c03a9"
    )
    
    # Test smart crawling
    print("\n🚀 Starting smart crawling...")
    try:
        result = await pipeline.ingest_library_documentation(config)
        
        if result.success:
            print(f"✅ Smart crawling completed successfully!")
            print(f"📊 Documents processed: {result.total_documents}")
            print(f"📄 Chunks created: {result.total_chunks}")
            print(f"📈 Processing stats: {result.processing_stats}")
            
            if result.crawled_urls:
                print(f"🔗 Crawled URLs ({len(result.crawled_urls)}):")
                for i, url in enumerate(result.crawled_urls[:5], 1):
                    print(f"   {i}. {url}")
                if len(result.crawled_urls) > 5:
                    print(f"   ... and {len(result.crawled_urls) - 5} more")
            
            # Test search functionality
            print("\n🔍 Testing search functionality...")
            search_queries = [
                "how to scrape a website",
                "firecrawl API features", 
                "crawl documentation",
                "extract structured data",
                "search web content"
            ]
            
            for query in search_queries:
                print(f"\n   🔍 Searching for: '{query}'")
                try:
                    search_results = await vector_store.search(query, n_results=3)
                    print(f"   📊 Found {len(search_results)} results")
                    
                    for i, result in enumerate(search_results[:2]):
                        title = result["metadata"].get('title', 'No title')
                        url = result["metadata"].get('url', 'No URL')
                        content_preview = result["content"][:100].replace('\n', ' ').strip()
                        print(f"   {i+1}. {title[:60]}...")
                        print(f"      URL: {url}")
                        print(f"      {content_preview}...")
                        
                except Exception as e:
                    print(f"   ❌ Search failed: {e}")
            
            # Test collection statistics
            print("\n📊 Collection Statistics:")
            try:
                stats = await vector_store.get_collection_stats()
                print(f"   📚 Total documents in collection: {stats['document_count']}")
            except Exception as e:
                print(f"   ❌ Failed to get stats: {e}")
                
        else:
            print(f"❌ Smart crawling failed: {result.errors}")
            
    except Exception as e:
        print(f"❌ Smart crawling test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_link_filtering():
    """Test the link filtering logic specifically."""
    
    print("\n🔗 Testing Link Filtering Logic")
    print("=" * 40)
    
    # Initialize pipeline
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    pipeline = DocumentationIngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        firecrawl_api_key="fc-efb6ba5edf62402a9845440bff5c03a9"
    )
    
    # Test configuration
    config = IngestionConfig(
        library_name="test",
        version="1.0.0",
        docs_url="https://docs.firecrawl.dev/introduction"
    )
    
    # Test URLs
    test_urls = [
        "https://docs.firecrawl.dev/introduction",  # ✅ Should include
        "https://docs.firecrawl.dev/quickstart",    # ✅ Should include
        "https://docs.firecrawl.dev/api/scrape",    # ✅ Should include
        "https://docs.firecrawl.dev/learn/rag",     # ✅ Should include
        "https://linkedin.com/company/firecrawl",   # ❌ Should exclude
        "https://github.com/firecrawl/firecrawl",   # ❌ Should exclude
        "https://twitter.com/firecrawl",            # ❌ Should exclude
        "https://zapier.com/apps/firecrawl",        # ❌ Should exclude
        "https://docs.firecrawl.dev/external-link", # ❌ Different domain
    ]
    
    print("🔍 Testing URL filtering:")
    for url in test_urls:
        is_relevant = pipeline._is_relevant_link(url, config)
        status = "✅ Include" if is_relevant else "❌ Exclude"
        print(f"   {status}: {url}")
    
    # Test link extraction from markdown
    print("\n📝 Testing link extraction from markdown:")
    sample_markdown = """
    # Firecrawl Documentation
    
    Welcome to [Firecrawl](https://docs.firecrawl.dev/introduction)!
    
    Check out our [Quickstart Guide](https://docs.firecrawl.dev/quickstart) and 
    [API Reference](https://docs.firecrawl.dev/api/scrape).
    
    Follow us on [LinkedIn](https://linkedin.com/company/firecrawl) and 
    [GitHub](https://github.com/firecrawl/firecrawl).
    
    Integrate with [Zapier](https://zapier.com/apps/firecrawl).
    """
    
    base_url = "https://docs.firecrawl.dev/introduction"
    extracted_links = pipeline._extract_links_from_content(sample_markdown, base_url)
    
    print(f"   📋 Extracted {len(extracted_links)} links:")
    for link in extracted_links:
        print(f"      - {link}")

if __name__ == "__main__":
    print("🧪 Smart Crawling Test Suite")
    print("=" * 50)
    
    # Run tests
    asyncio.run(test_link_filtering())
    asyncio.run(test_smart_crawling())
    
    print("\n🎉 Smart crawling test suite completed!")
