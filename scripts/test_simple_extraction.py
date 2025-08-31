#!/usr/bin/env python3
"""Simple test for Firecrawl extraction with a smaller target."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.ingestion import FirecrawlClient

async def test_simple_extraction():
    """Test simple extraction with a smaller target."""
    
    print("🧪 Testing Simple Firecrawl Extraction")
    print("=" * 40)
    
    # Test with a simpler, smaller page
    test_urls = [
        "https://example.com",  # Simple static page
        "https://httpbin.org/json",  # Simple JSON API
        "https://jsonplaceholder.typicode.com/posts/1"  # Simple API response
    ]
    
    async with FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9") as client:
        health = await client.health_check()
        print(f"🔗 Firecrawl health: {'✅' if health else '❌'}")
        
        if not health:
            print("❌ Firecrawl not accessible")
            return
        
        for url in test_urls:
            print(f"\n📄 Testing extraction from: {url}")
            try:
                doc = await client.scrape_url(url)
                
                print(f"   ✅ Successfully extracted")
                print(f"   📄 Title: {doc.title[:100]}...")
                print(f"   📝 Content length: {len(doc.content)} characters")
                print(f"   🔗 Links found: {len(doc.metadata.get('links', []))}")
                
                if doc.content:
                    print(f"   📖 Content preview: {doc.content[:200]}...")
                
            except Exception as e:
                print(f"   ❌ Failed: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_extraction())
