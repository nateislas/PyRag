#!/usr/bin/env python3
"""Debug script to test Firecrawl API key formats."""

import asyncio
import aiohttp
import json

API_KEY = "fc-efb6ba5edf62402a9845440bff5c03a9"

async def test_api_key_formats():
    """Test different API key formats."""
    
    print("🔍 Testing different API key formats...")
    
    # Test different header formats
    header_formats = [
        {"Authorization": f"Bearer {API_KEY}"},
        {"Authorization": API_KEY},
        {"X-API-Key": API_KEY},
        {"api-key": API_KEY},
        {"x-api-key": API_KEY},
    ]
    
    test_payload = {
        "urls": ["https://example.com"],
        "prompt": "Extract the page title",
        "schema": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"]
        }
    }
    
    async with aiohttp.ClientSession() as session:
        for i, headers in enumerate(header_formats):
            print(f"\n🔍 Test {i+1}: {list(headers.keys())}")
            
            # Add Content-Type
            headers["Content-Type"] = "application/json"
            
            try:
                async with session.post(
                    "https://api.firecrawl.dev/v2/extract",
                    json=test_payload,
                    headers=headers
                ) as response:
                    print(f"   Status: {response.status}")
                    if response.status == 200:
                        data = await response.json()
                        print(f"   ✅ Success! Response: {data}")
                        return headers
                    else:
                        error_text = await response.text()
                        print(f"   ❌ Error: {error_text[:200]}...")
                        
            except Exception as e:
                print(f"   ❌ Exception: {e}")
    
    print("\n❌ No working API key format found")
    return None

async def test_simple_request():
    """Test a simple request to see what the API expects."""
    
    print("\n🔍 Testing simple request without auth...")
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.post(
                "https://api.firecrawl.dev/v2/extract",
                json={"urls": ["https://example.com"]}
            ) as response:
                print(f"   Status: {response.status}")
                error_text = await response.text()
                print(f"   Response: {error_text[:300]}...")
        except Exception as e:
            print(f"   ❌ Exception: {e}")

if __name__ == "__main__":
    asyncio.run(test_simple_request())
    asyncio.run(test_api_key_formats())
