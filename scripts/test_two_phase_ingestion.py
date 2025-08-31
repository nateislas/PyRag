#!/usr/bin/env python3
"""Test the complete two-phase documentation ingestion approach."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyrag.config import get_config, validate_config
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.llm.client import LLMClient
from pyrag.ingestion.documentation_manager import DocumentationManager, DocumentationJob
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def test_two_phase_ingestion():
    """Test the complete two-phase documentation ingestion."""
    
    print("🚀 Testing Two-Phase Documentation Ingestion")
    print("=" * 60)
    
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
    
    # Initialize documentation manager
    print("📖 Initializing documentation manager...")
    doc_manager = DocumentationManager(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,
        firecrawl_api_key=config.firecrawl.api_key,
        cache_dir="./cache"
    )
    
    # Create documentation job for Firecrawl
    job = DocumentationJob(
        library_name="firecrawl",
        version="2.0.0",
        base_url="https://docs.firecrawl.dev/introduction",
        output_dir="./output",
        max_crawl_depth=2,  # Limit for testing
        max_crawl_pages=20,  # Limit for testing
        max_content_pages=10,  # Limit for testing
        use_llm_filtering=True,
        exclude_patterns=[
            "linkedin.com", "twitter.com", "github.com", "facebook.com",
            "youtube.com", "discord.com", "slack.com", "zapier.com"
        ],
        include_patterns=[
            "/docs/", "/guide/", "/tutorial/", "/api/", "/reference/",
            "/introduction", "/quickstart", "/examples", "/learn/"
        ]
    )
    
    print(f"🎯 Job Configuration:")
    print(f"   Library: {job.library_name} v{job.version}")
    print(f"   Base URL: {job.base_url}")
    print(f"   Max crawl depth: {job.max_crawl_depth}")
    print(f"   Max crawl pages: {job.max_crawl_pages}")
    print(f"   Max content pages: {job.max_content_pages}")
    print(f"   LLM filtering: {job.use_llm_filtering}")
    
    # Execute the complete ingestion process
    print("\n🚀 Starting complete documentation ingestion...")
    try:
        result = await doc_manager.ingest_documentation(job)
        
        if result.success:
            print(f"\n✅ Documentation ingestion completed successfully!")
            
            # Display results
            print(f"\n📊 Crawl Results:")
            print(f"   Total discovered URLs: {result.crawl_result.crawl_stats['total_discovered']}")
            print(f"   Relevant URLs: {result.crawl_result.crawl_stats['total_relevant']}")
            print(f"   Pages crawled: {result.crawl_result.crawl_stats['pages_crawled']}")
            
            print(f"\n📄 Extraction Results:")
            print(f"   Total URLs: {result.extraction_stats['total_urls']}")
            print(f"   Extracted URLs: {result.extraction_stats['extracted_urls']}")
            print(f"   Failed URLs: {result.extraction_stats['failed_urls']}")
            print(f"   Total content length: {result.extraction_stats['total_content_length']}")
            
            print(f"\n🔧 Processing Results:")
            print(f"   Total documents: {result.processing_stats['total_documents']}")
            print(f"   Total chunks: {result.processing_stats['total_chunks']}")
            print(f"   Content type distribution: {result.processing_stats['content_type_distribution']}")
            
            print(f"\n💾 Storage Results:")
            print(f"   Stored chunks: {result.storage_stats['stored_chunks']}")
            print(f"   Storage success rate: {result.storage_stats['storage_success_rate']:.2%}")
            
            # Show some discovered URLs
            if result.crawl_result.relevant_urls:
                print(f"\n🔗 Sample Discovered URLs:")
                for i, url in enumerate(list(result.crawl_result.relevant_urls)[:10], 1):
                    print(f"   {i:2d}. {url}")
                if len(result.crawl_result.relevant_urls) > 10:
                    print(f"   ... and {len(result.crawl_result.relevant_urls) - 10} more")
            
            # Test search functionality
            print(f"\n🔍 Testing search functionality...")
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
            
            # List ingested libraries
            print(f"\n📚 Ingested Libraries:")
            libraries = await doc_manager.list_ingested_libraries()
            for lib in libraries:
                print(f"   📖 {lib['library_name']} v{lib['version']} - {lib['total_chunks']} chunks")
                
        else:
            print(f"\n❌ Documentation ingestion failed:")
            for error in result.errors:
                print(f"   ❌ {error}")
            
    except Exception as e:
        print(f"❌ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()

async def test_site_crawler_only():
    """Test just the site crawler (Phase 1) to see link discovery."""
    
    print("\n🕷️  Testing Site Crawler (Phase 1 Only)")
    print("=" * 50)
    
    config = get_config()
    llm_client = LLMClient(config.llm)
    
    from pyrag.ingestion.site_crawler import SiteCrawler
    
    async with SiteCrawler(
        llm_client=llm_client,
        max_depth=2,
        max_pages=10,
        delay=1.0
    ) as crawler:
        
        result = await crawler.crawl_documentation_site(
            base_url="https://docs.firecrawl.dev/introduction",
            library_name="firecrawl"
        )
        
        print(f"📊 Crawl Results:")
        print(f"   Total discovered URLs: {result.crawl_stats['total_discovered']}")
        print(f"   Relevant URLs: {result.crawl_stats['total_relevant']}")
        print(f"   Pages crawled: {result.crawl_stats['pages_crawled']}")
        
        if result.relevant_urls:
            print(f"\n🔗 Discovered Documentation URLs:")
            for i, url in enumerate(list(result.relevant_urls)[:15], 1):
                print(f"   {i:2d}. {url}")
            if len(result.relevant_urls) > 15:
                print(f"   ... and {len(result.relevant_urls) - 15} more")

if __name__ == "__main__":
    print("🧪 Two-Phase Documentation Ingestion Test Suite")
    print("=" * 60)
    
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
    asyncio.run(test_site_crawler_only())
    asyncio.run(test_two_phase_ingestion())
    
    print("\n🎉 Two-phase ingestion test suite completed!")
