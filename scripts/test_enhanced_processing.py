#!/usr/bin/env python3
"""Test script for enhanced processing system."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.config import get_config, validate_config
from pyrag.llm.client import LLMClient
from pyrag.enhanced_processing import EnhancedDocumentProcessor

# Sample documentation content for testing
SAMPLE_DOCUMENTATION = """
# Requests Library Documentation

## Overview
The requests library is a popular HTTP library for Python. It provides a simple interface for making HTTP requests.

## Basic Usage

### Making GET Requests

The `requests.get()` function is used to make GET requests to web servers.

```python
import requests

# Basic GET request
response = requests.get('https://api.github.com/users/octocat')
print(response.status_code)  # 200
print(response.json())  # {'login': 'octocat', 'id': 583231, ...}
```

### Making POST Requests

The `requests.post()` function is used to make POST requests.

```python
import requests

# POST request with data
data = {'username': 'john', 'password': 'secret'}
response = requests.post('https://httpbin.org/post', data=data)
print(response.status_code)  # 200
```

## Advanced Features

### Session Objects

Sessions allow you to persist certain parameters across requests.

```python
import requests

with requests.Session() as session:
    session.headers.update({'User-Agent': 'MyApp/1.0'})
    response1 = session.get('https://httpbin.org/get')
    response2 = session.get('https://httpbin.org/get')
```

### Authentication

Requests supports various authentication methods.

```python
import requests
from requests.auth import HTTPBasicAuth

# Basic authentication
response = requests.get('https://api.github.com/user', 
                       auth=HTTPBasicAuth('username', 'password'))

# Token authentication
headers = {'Authorization': 'token YOUR_TOKEN'}
response = requests.get('https://api.github.com/user', headers=headers)
```

## Error Handling

Always handle potential errors when making requests.

```python
import requests

try:
    response = requests.get('https://httpbin.org/status/404')
    response.raise_for_status()  # Raises an HTTPError for bad responses
except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")
```

## Parameters

### requests.get()

- **url** (str): The URL to send the request to
- **params** (dict, optional): Query parameters to add to the URL
- **headers** (dict, optional): HTTP headers to send with the request
- **timeout** (int, optional): Timeout in seconds

### requests.post()

- **url** (str): The URL to send the request to
- **data** (dict, optional): Data to send in the request body
- **json** (dict, optional): JSON data to send in the request body
- **headers** (dict, optional): HTTP headers to send with the request
"""

async def test_enhanced_processing():
    """Test the enhanced processing system."""
    
    print("🧪 Testing Enhanced Processing System")
    print("=" * 50)
    
    # Load configuration
    try:
        config = get_config()
        if not validate_config(config):
            print("❌ Configuration validation failed")
            return
        
        print("✅ Configuration loaded successfully")
    except Exception as e:
        print(f"❌ Configuration error: {e}")
        return
    
    # Initialize LLM client
    try:
        llm_client = LLMClient(config.llm)
        health_check = await llm_client.health_check()
        if not health_check:
            print("❌ LLM client health check failed")
            return
        
        print("✅ LLM client initialized successfully")
    except Exception as e:
        print(f"❌ LLM client error: {e}")
        return
    
    # Initialize enhanced processor
    try:
        enhanced_processor = EnhancedDocumentProcessor(llm_client)
        print("✅ Enhanced processor initialized successfully")
    except Exception as e:
        print(f"❌ Enhanced processor error: {e}")
        return
    
    # Test document processing
    print("\n📄 Testing Document Processing")
    print("-" * 30)
    
    try:
        # Create test document
        test_document = {
            "content": SAMPLE_DOCUMENTATION,
            "metadata": {
                "library": "requests",
                "version": "2.31.0",
                "source_url": "https://docs.python-requests.org/",
                "title": "Requests Library Documentation"
            },
            "hierarchy_path": ["requests", "documentation", "overview"]
        }
        
        # Process the document
        result = await enhanced_processor.process_document(
            content=test_document["content"],
            metadata=test_document["metadata"],
            hierarchy_path=test_document["hierarchy_path"]
        )
        
        print(f"✅ Document processed successfully")
        print(f"📊 Processing stats:")
        print(f"   - Total semantic chunks: {result.processing_stats['total_semantic_chunks']}")
        print(f"   - Total enhanced chunks: {result.processing_stats['total_enhanced_chunks']}")
        print(f"   - Metadata completeness: {result.processing_stats['average_metadata_completeness']:.2f}")
        print(f"   - Chunk types: {result.processing_stats['chunk_types']}")
        
        # Analyze chunks
        print(f"\n🔍 Chunk Analysis:")
        print("-" * 20)
        
        for i, chunk in enumerate(result.chunks):
            print(f"\nChunk {i+1}:")
            print(f"  Type: {chunk.chunk_type}")
            print(f"  Content length: {len(chunk.content)} chars")
            print(f"  API path: {chunk.enhanced_metadata.api_path}")
            print(f"  Complexity: {chunk.enhanced_metadata.complexity_level}")
            print(f"  Importance: {chunk.enhanced_metadata.importance_score:.2f}")
            print(f"  Has examples: {bool(chunk.enhanced_metadata.examples)}")
            print(f"  Has parameters: {bool(chunk.enhanced_metadata.parameters)}")
            
            if chunk.api_doc_chunk:
                print(f"  Function signature: {chunk.api_doc_chunk.function_signature}")
                print(f"  Return type: {chunk.api_doc_chunk.return_type}")
                print(f"  Parameters: {len(chunk.api_doc_chunk.parameters)}")
            
            if chunk.example_chunk:
                print(f"  Example use case: {chunk.example_chunk.use_case}")
                print(f"  Example complexity: {chunk.example_chunk.complexity_level}")
        
        # Test vector store format conversion
        print(f"\n💾 Testing Vector Store Format")
        print("-" * 30)
        
        if result.chunks:
            vector_format = enhanced_processor.to_vector_store_format(result.chunks[0])
            print(f"✅ Vector format conversion successful")
            print(f"   - Content length: {len(vector_format['content'])}")
            print(f"   - Metadata fields: {len(vector_format['metadata'])}")
            print(f"   - Key metadata: {list(vector_format['metadata'].keys())[:5]}...")
        
        # Test search metadata
        print(f"\n🔍 Testing Search Metadata")
        print("-" * 30)
        
        if result.chunks:
            search_metadata = enhanced_processor.get_search_metadata(result.chunks[0])
            print(f"✅ Search metadata extraction successful")
            print(f"   - Search fields: {list(search_metadata.keys())}")
            print(f"   - Library: {search_metadata['library']}")
            print(f"   - Content type: {search_metadata['content_type']}")
            print(f"   - Complexity: {search_metadata['complexity_level']}")
        
        print(f"\n🎉 Enhanced Processing Test Completed Successfully!")
        
    except Exception as e:
        print(f"❌ Document processing error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_enhanced_processing())
