#!/usr/bin/env python3
"""Simple test script to verify Firecrawl API."""

import asyncio
import aiohttp
import json

API_KEY = "fc-efb6ba5edf62402a9845440bff5c03a9"

async def test_firecrawl_api():
    """Test Firecrawl API directly."""
    
    print("🔗 Testing Firecrawl API directly...")
    
    # Test different API endpoints
    endpoints = [
        "https://api.firecrawl.dev/scrape",
        "https://api.firecrawl.dev/crawl", 
        "https://api.firecrawl.dev/v1/scrape",
        "https://api.firecrawl.dev/v1/crawl",
        "https://api.firecrawl.dev/api/scrape",
        "https://api.firecrawl.dev/api/crawl"
    ]
    
    test_payload = {
        "url": "https://docs.python-requests.org/en/latest/user/quickstart/",
        "apiKey": API_KEY,
        "pageOptions": {
            "onlyMainContent": True,
            "includeMarkdown": True
        }
    }
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    
    async with aiohttp.ClientSession() as session:
        for endpoint in endpoints:
            print(f"\n🔍 Testing endpoint: {endpoint}")
            
            try:
                # Try with API key in payload
                async with session.post(endpoint, json=test_payload) as response:
                    print(f"   Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Success! Response keys: {list(data.keys())}")
                        return endpoint, "payload"
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error: {error_text[:200]}...")
                
                # Try with API key in headers
                payload_without_key = {k: v for k, v in test_payload.items() if k != "apiKey"}
                async with session.post(endpoint, json=payload_without_key, headers=headers) as response:
                    print(f"   Status (headers): {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Success! Response keys: {list(data.keys())}")
                        return endpoint, "headers"
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error: {error_text[:200]}...")
                        
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    print("\n❌ No working endpoint found")
    return None, None

async def test_simple_request():
    """Test a simple GET request to see if the API is accessible."""
    
    print("\n🔍 Testing simple API access...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get("https://api.firecrawl.dev/") as response:
                print(f"   Base URL status: {response.status}")
                if response.status == 200:
                    content = await response.text()
                    print(f"   ✅ Base URL accessible")
                    print(f"   Content preview: {content[:200]}...")
                else:
                    print(f"   ❌ Base URL not accessible")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_request())
    asyncio.run(test_firecrawl_api())
