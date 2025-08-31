#!/usr/bin/env python3
"""Debug script to test link extraction from documentation content."""

import asyncio
import sys
import os
import re
from urllib.parse import urljoin

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from pyrag.ingestion.firecrawl_client import FirecrawlClient
from pyrag.ingestion.ingestion_pipeline import DocumentationIngestionPipeline
from pyrag.logging import get_logger

logger = get_logger(__name__)

def extract_links_debug(content: str, base_url: str) -> list:
    """Debug version of link extraction with detailed logging."""
    links = []
    
    print(f"🔍 Analyzing content of length: {len(content)}")
    print(f"🎯 Base URL: {base_url}")
    
    # Pattern 1: Markdown links [text](url)
    markdown_pattern = r'\[([^\]]+)\]\(([^)]+)\)'
    markdown_matches = list(re.finditer(markdown_pattern, content))
    print(f"📝 Markdown links found: {len(markdown_matches)}")
    for match in markdown_matches:
        link_url = match.group(2)
        link_text = match.group(1)
        print(f"   📄 [{link_text}]({link_url})")
        links.append(link_url)
    
    # Pattern 2: HTML anchor tags <a href="url">text</a>
    html_pattern = r'<a[^>]+href=["\']([^"\']+)["\'][^>]*>'
    html_matches = list(re.finditer(html_pattern, content, re.IGNORECASE))
    print(f"🌐 HTML links found: {len(html_matches)}")
    for match in html_matches[:10]:  # Show first 10
        link_url = match.group(1)
        print(f"   🔗 <a href=\"{link_url}\">")
        links.append(link_url)
    if len(html_matches) > 10:
        print(f"   ... and {len(html_matches) - 10} more HTML links")
    
    # Pattern 3: Plain URLs (http/https)
    url_pattern = r'https?://[^\s\)\]\>\"]+'
    url_matches = list(re.finditer(url_pattern, content))
    print(f"🔗 Plain URLs found: {len(url_matches)}")
    for match in url_matches[:5]:  # Show first 5
        link_url = match.group(0)
        print(f"   🌍 {link_url}")
        links.append(link_url)
    if len(url_matches) > 5:
        print(f"   ... and {len(url_matches) - 5} more plain URLs")
    
    # Pattern 4: Relative URLs starting with /
    relative_pattern = r'["\'](/[^"\']+)["\']'
    relative_matches = list(re.finditer(relative_pattern, content))
    print(f"📂 Relative URLs found: {len(relative_matches)}")
    for match in relative_matches[:10]:  # Show first 10
        link_url = match.group(1)
        print(f"   📁 {link_url}")
        links.append(link_url)
    if len(relative_matches) > 10:
        print(f"   ... and {len(relative_matches) - 10} more relative URLs")
    
    # Pattern 5: Navigation links in various formats
    nav_patterns = [
        r'href=["\']([^"\']*/(?:docs|guide|tutorial|api|reference|learn|examples)[^"\']*)["\']',
        r'href=["\']([^"\']*/(?:introduction|quickstart|overview)[^"\']*)["\']',
    ]
    nav_links = []
    for pattern in nav_patterns:
        matches = list(re.finditer(pattern, content, re.IGNORECASE))
        nav_links.extend([match.group(1) for match in matches])
    print(f"🧭 Navigation links found: {len(nav_links)}")
    for link_url in nav_links[:10]:  # Show first 10
        print(f"   🧭 {link_url}")
        links.append(link_url)
    if len(nav_links) > 10:
        print(f"   ... and {len(nav_links) - 10} more navigation links")
    
    # Convert relative URLs to absolute and clean up
    absolute_links = []
    for link_url in links:
        # Skip anchors, fragments, and invalid URLs
        if (link_url.startswith('#') or 
            link_url.startswith('mailto:') or 
            link_url.startswith('javascript:') or
            not link_url.strip()):
            continue
            
        # Convert relative URLs to absolute
        if link_url.startswith('/'):
            link_url = urljoin(base_url, link_url)
        elif not link_url.startswith('http'):
            link_url = urljoin(base_url, link_url)
        
        # Clean up URL (remove fragments, normalize)
        clean_url = link_url.split('#')[0].split('?')[0]
        if clean_url:
            absolute_links.append(clean_url)
    
    # Remove duplicates and show final results
    unique_links = list(set(absolute_links))
    print(f"\n📊 Final Results:")
    print(f"   Total raw links found: {len(links)}")
    print(f"   Unique absolute links: {len(unique_links)}")
    
    return unique_links

async def test_link_extraction():
    """Test link extraction with actual Firecrawl documentation."""
    
    print("🔍 Testing Link Extraction from Firecrawl Documentation")
    print("=" * 60)
    
    # Initialize Firecrawl client
    client = FirecrawlClient(api_key="fc-efb6ba5edf62402a9845440bff5c03a9")
    
    try:
        async with client:
            # Scrape the introduction page
            print("📄 Scraping Firecrawl introduction page...")
            doc = await client.scrape_url("https://docs.firecrawl.dev/introduction")
            
            print(f"✅ Scraped content length: {len(doc.content)}")
            print(f"📝 Title: {doc.title}")
            
            # Test our link extraction
            print("\n🔍 Extracting links...")
            links = extract_links_debug(doc.content, "https://docs.firecrawl.dev/introduction")
            
            print(f"\n🎯 Final unique links found: {len(links)}")
            for i, link in enumerate(links[:20], 1):  # Show first 20
                print(f"   {i:2d}. {link}")
            if len(links) > 20:
                print(f"   ... and {len(links) - 20} more links")
            
            # Test LLM filtering
            print(f"\n🧠 Testing LLM filtering on {len(links)} links...")
            from pyrag.llm.client import LLMClient
            from pyrag.config import get_config
            
            config = get_config()
            llm_client = LLMClient(config.llm)
            
            filtered_links = await llm_client.filter_links(
                base_url="https://docs.firecrawl.dev/introduction",
                all_links=links,
                library_name="firecrawl"
            )
            
            print(f"✅ LLM filtered to {len(filtered_links)} relevant links:")
            for link in filtered_links:
                print(f"   ✅ {link}")
                
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_link_extraction())
