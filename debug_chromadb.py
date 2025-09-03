#!/usr/bin/env python3
"""Debug script to test ChromaDB search directly."""

import asyncio
import chromadb
from chromadb.config import Settings
from src.pyrag.config import get_config
from src.pyrag.logging import get_logger

logger = get_logger(__name__)

async def debug_chromadb():
    """Debug ChromaDB search to identify the array ambiguity error."""
    print("🔍 Debugging ChromaDB Search...")
    
    # Get configuration
    config = get_config()
    
    # Initialize ChromaDB client
    client = chromadb.PersistentClient(
        path=config.vector_store.db_path,
        settings=Settings(
            anonymized_telemetry=False,
            allow_reset=True,
        ),
    )
    
    print(f"📁 ChromaDB path: {config.vector_store.db_path}")
    
    # List all collections
    collections = client.list_collections()
    print(f"📚 Collections found: {[col.name for col in collections]}")
    
    # Test each collection
    for collection in collections:
        print(f"\n🔍 Testing collection: {collection.name}")
        
        # Get collection info
        try:
            count = collection.count()
            print(f"   📊 Document count: {count}")
            
            if count > 0:
                # Try to get a sample document
                sample = collection.get(limit=1)
                print(f"   📄 Sample document keys: {list(sample.keys())}")
                
                if 'metadatas' in sample and sample['metadatas']:
                    metadata = sample['metadatas'][0]
                    print(f"   🏷️  Sample metadata keys: {list(metadata.keys())}")
                    
                    # Show key metadata values
                    key_fields = ['library_name', 'library', 'version', 'content_type', 'source']
                    for field in key_fields:
                        if field in metadata:
                            value = metadata[field]
                            print(f"      📋 {field}: {value} (type: {type(value)})")
                    
                    # Check for array fields
                    for key, value in metadata.items():
                        if isinstance(value, list):
                            print(f"      ⚠️  Array field '{key}': {type(value)} with {len(value)} items")
                            if len(value) > 0:
                                print(f"         First item type: {type(value[0])}")
                
                # Try a simple search without filters
                try:
                    print(f"   🔍 Testing simple search...")
                    results = collection.query(
                        query_texts=["fastapi"],
                        n_results=1
                    )
                    print(f"   ✅ Simple search successful: {len(results['documents'][0])} results")
                except Exception as e:
                    print(f"   ❌ Simple search failed: {e}")
                
                # Try search with library_name filter
                try:
                    print(f"   🔍 Testing search with library_name filter...")
                    results = collection.query(
                        query_texts=["fastapi"],
                        n_results=1,
                        where={"library_name": "fastapi"}
                    )
                    print(f"   ✅ library_name filter search successful: {len(results['documents'][0])} results")
                except Exception as e:
                    print(f"   ❌ library_name filter search failed: {e}")
                
                # Try search with library filter
                try:
                    print(f"   🔍 Testing search with library filter...")
                    results = collection.query(
                        query_texts=["fastapi"],
                        n_results=1,
                        where={"library": "fastapi"}
                    )
                    print(f"   ✅ library filter search successful: {len(results['documents'][0])} results")
                except Exception as e:
                    print(f"   ❌ library filter search failed: {e}")
                
                # Try search with no filters but different query
                try:
                    print(f"   🔍 Testing search with 'api' query...")
                    results = collection.query(
                        query_texts=["api"],
                        n_results=1
                    )
                    print(f"   ✅ 'api' query search successful: {len(results['documents'][0])} results")
                except Exception as e:
                    print(f"   ❌ 'api' query search failed: {e}")
                
        except Exception as e:
            print(f"   ❌ Error accessing collection: {e}")

if __name__ == "__main__":
    asyncio.run(debug_chromadb())
