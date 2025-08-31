#!/usr/bin/env python3
"""Test LLM-guided crawling with environment variable configuration."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyrag.config import get_config, validate_config
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.llm.client import LLMClient
from pyrag.ingestion.ingestion_pipeline import DocumentationIngestionPipeline, IngestionConfig
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_llm_guided_crawling():
    """Test LLM-guided crawling with comprehensive documentation ingestion."""
    
    print("🧠 Testing LLM-Guided Smart Crawling")
    print("=" * 50)
    
    # Load configuration
    print("📋 Loading configuration...")
    config = get_config()
    
    if not validate_config(config):
        print("❌ Configuration validation failed. Please set up your environment variables.")
        return
    
    print("✅ Configuration loaded successfully")
    
    # Initialize PyRAG components
    print("📚 Initializing PyRAG components...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    
    # Initialize LLM client
    print("🤖 Initializing LLM client...")
    llm_client = LLMClient(config.llm)
    
    # Test LLM health
    print("🔍 Testing LLM connection...")
    llm_healthy = await llm_client.health_check()
    if llm_healthy:
        print("✅ LLM client is healthy")
    else:
        print("❌ LLM client health check failed")
        return
    
    # Test configuration for Firecrawl documentation
    ingestion_config = IngestionConfig(
        library_name="firecrawl",
        version="2.0.0",
        docs_url="https://docs.firecrawl.dev/introduction",
        crawl_options={
            "crawl_entire_site": True,
            "max_pages": 5,  # Limit for testing
            "use_llm_filtering": True,  # Enable LLM filtering
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
    
    print(f"🎯 Target: {ingestion_config.docs_url}")
    print(f"📋 Crawl options: {ingestion_config.crawl_options}")
    
    # Initialize pipeline with LLM client
    pipeline = DocumentationIngestionPipeline(
        vector_store=vector_store,
        embedding_service=embedding_service,
        firecrawl_api_key=config.firecrawl.api_key,
        llm_client=llm_client
    )
    
    # Test LLM link filtering
    print("\n🔗 Testing LLM link filtering...")
    test_links = [
        "https://docs.firecrawl.dev/introduction",
        "https://docs.firecrawl.dev/quickstart", 
        "https://docs.firecrawl.dev/api/scrape",
        "https://linkedin.com/company/firecrawl",
        "https://github.com/firecrawl/firecrawl",
        "https://docs.firecrawl.dev/learn/rag"
    ]
    
    try:
        filtered_links = await llm_client.filter_links(
            base_url=ingestion_config.docs_url,
            all_links=test_links,
            library_name=ingestion_config.library_name
        )
        
        print(f"📊 LLM filtered {len(test_links)} links to {len(filtered_links)} relevant links:")
        for link in filtered_links:
            print(f"   ✅ {link}")
            
    except Exception as e:
        print(f"❌ LLM link filtering failed: {e}")
        return
    
    # Test smart crawling
    print("\n🚀 Starting LLM-guided smart crawling...")
    try:
        result = await pipeline.ingest_library_documentation(ingestion_config)
        
        if result.success:
            print(f"✅ LLM-guided crawling completed successfully!")
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
                "extract structured data"
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
                    
        else:
            print(f"❌ LLM-guided crawling failed: {result.errors}")
            
    except Exception as e:
        print(f"❌ LLM-guided crawling test failed: {e}")
        import traceback
        traceback.print_exc()

async def test_llm_content_analysis():
    """Test LLM content type analysis."""
    
    print("\n📝 Testing LLM Content Type Analysis")
    print("=" * 40)
    
    config = get_config()
    llm_client = LLMClient(config.llm)
    
    test_pages = [
        {
            "url": "https://docs.firecrawl.dev/introduction",
            "title": "Quickstart | Firecrawl",
            "content": "Firecrawl allows you to turn entire websites into LLM-ready markdown..."
        },
        {
            "url": "https://docs.firecrawl.dev/api/scrape", 
            "title": "Scrape API | Firecrawl",
            "content": "The scrape endpoint allows you to extract content from a single URL..."
        },
        {
            "url": "https://docs.firecrawl.dev/examples",
            "title": "Examples | Firecrawl", 
            "content": "Here are some examples of how to use Firecrawl in different scenarios..."
        }
    ]
    
    for page in test_pages:
        try:
            content_type = await llm_client.analyze_content_type(
                url=page["url"],
                title=page["title"], 
                content_preview=page["content"]
            )
            print(f"📄 {page['title']} → {content_type}")
        except Exception as e:
            print(f"❌ Content analysis failed for {page['title']}: {e}")

if __name__ == "__main__":
    print("🧪 LLM-Guided Crawling Test Suite")
    print("=" * 50)
    
    # Check for required environment variables
    if not os.getenv("LLAMA_API_KEY"):
        print("❌ LLAMA_API_KEY environment variable is required")
        print("📝 Please set it with: export LLAMA_API_KEY=your_key_here")
        sys.exit(1)
    
    if not os.getenv("FIRECRAWL_API_KEY"):
        print("❌ FIRECRAWL_API_KEY environment variable is required") 
        print("📝 Please set it with: export FIRECRAWL_API_KEY=your_key_here")
        sys.exit(1)
    
    # Run tests
    asyncio.run(test_llm_content_analysis())
    asyncio.run(test_llm_guided_crawling())
    
    print("\n🎉 LLM-guided crawling test suite completed!")
