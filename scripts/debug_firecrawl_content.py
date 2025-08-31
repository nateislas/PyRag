#!/usr/bin/env python3
"""Debug script to examine Firecrawl content types."""

import asyncio
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyrag.ingestion.firecrawl_client import FirecrawlClient
from pyrag.logging import get_logger

logger = get_logger(__name__)

async def debug_firecrawl_content():
    """Debug what content Firecrawl returns."""
    
    print("🔍 Debugging Firecrawl Content Types")
    print("=" * 50)
    
    # Initialize Firecrawl client
    client = FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9")
    
    try:
        async with client:
            # Scrape with different options
            print("📄 Scraping Firecrawl introduction page...")
            
            # Try to get HTML content
            doc = await client.scrape_url("https://docs.firecrawl.dev/introduction")
            
            print(f"✅ Scraped successfully!")
            print(f"📝 Title: {doc.title}")
            print(f"🔗 URL: {doc.url}")
            print(f"📄 Content length: {len(doc.content)}")
            print(f"📝 Markdown length: {len(doc.markdown) if doc.markdown else 0}")
            print(f"🌐 HTML length: {len(doc.html) if doc.html else 0}")
            
            # Show content preview
            print(f"\n📄 Content Preview (first 500 chars):")
            print("-" * 50)
            print(doc.content[:500])
            print("-" * 50)
            
            if doc.markdown:
                print(f"\n📝 Markdown Preview (first 500 chars):")
                print("-" * 50)
                print(doc.markdown[:500])
                print("-" * 50)
            
            if doc.html:
                print(f"\n🌐 HTML Preview (first 500 chars):")
                print("-" * 50)
                print(doc.html[:500])
                print("-" * 50)
            
            # Check metadata
            print(f"\n📊 Metadata:")
            for key, value in doc.metadata.items():
                if isinstance(value, list):
                    print(f"   {key}: {len(value)} items")
                    if key == "links" and value:
                        print(f"      Links: {value[:5]}...")
                else:
                    print(f"   {key}: {value}")
            
            # Try to extract links from HTML if available
            if doc.html:
                print(f"\n🔍 Extracting links from HTML...")
                import re
                
                # Look for navigation links in HTML
                nav_patterns = [
                    r'href=["\']([^"\']*/(?:docs|guide|tutorial|api|reference|learn|examples)[^"\']*)["\']',
                    r'href=["\']([^"\']*/(?:introduction|quickstart|overview)[^"\']*)["\']',
                    r'href=["\']([^"\']*/(?:scrape|crawl|search|map)[^"\']*)["\']',
                ]
                
                all_links = []
                for pattern in nav_patterns:
                    matches = re.finditer(pattern, doc.html, re.IGNORECASE)
                    for match in matches:
                        link = match.group(1)
                        if link not in all_links:
                            all_links.append(link)
                
                print(f"   Found {len(all_links)} navigation links in HTML:")
                for i, link in enumerate(all_links[:10], 1):
                    print(f"   {i:2d}. {link}")
                if len(all_links) > 10:
                    print(f"   ... and {len(all_links) - 10} more")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(debug_firecrawl_content())
