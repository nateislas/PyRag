#!/usr/bin/env python3
"""Test Firecrawl documentation extraction."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.ingestion import FirecrawlClient

async def test_firecrawl_docs():
    """Test extraction from Firecrawl documentation."""
    
    print("🔥 Testing Firecrawl Documentation Extraction")
    print("=" * 50)
    
    # Test URLs from Firecrawl docs
    test_urls = [
        "https://docs.firecrawl.dev/introduction",  # Main intro page
        "https://docs.firecrawl.dev/standard-features/scrape",  # Scrape feature docs
        "https://docs.firecrawl.dev/agentic-features/extract"   # Extract feature docs
    ]
    
    async with FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9") as client:
        health = await client.health_check()
        print(f"🔗 Firecrawl health: {'✅' if health else '❌'}")
        
        if not health:
            print("❌ Firecrawl not accessible")
            return
        
        for i, url in enumerate(test_urls, 1):
            print(f"\n📄 Test {i}: Extracting from {url}")
            try:
                doc = await client.scrape_url(url)
                
                print(f"   ✅ Successfully extracted")
                print(f"   📄 Title: {doc.title[:100]}...")
                print(f"   📝 Content length: {len(doc.content)} characters")
                print(f"   🔗 Links found: {len(doc.metadata.get('links', []))}")
                
                if doc.content:
                    # Show a more detailed preview
                    preview = doc.content[:300].replace('\n', ' ').strip()
                    print(f"   📖 Content preview: {preview}...")
                    
                    # Check for key Firecrawl terms
                    key_terms = ['firecrawl', 'scrape', 'crawl', 'extract', 'api']
                    found_terms = [term for term in key_terms if term.lower() in doc.content.lower()]
                    if found_terms:
                        print(f"   🔍 Found key terms: {', '.join(found_terms)}")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")
                import traceback
                traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_firecrawl_docs())
