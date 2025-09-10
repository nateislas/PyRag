#!/usr/bin/env python3
"""Quick script to check unique libraries in ChromaDB collection."""

import chromadb
from collections import Counter
import sys
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def check_libraries():
    """Check unique libraries and document counts in the documents collection."""
    try:
        # Initialize ChromaDB client for cloud
        print("🔍 Connecting to ChromaDB Cloud...")
        
        # Get cloud configuration from environment
        api_key = os.getenv("CHROMA_CLOUD_API_KEY")
        tenant_id = os.getenv("CHROMA_TENANT_ID") 
        database = os.getenv("CHROMA_DATABASE")
        
        if not all([api_key, tenant_id, database]):
            print("❌ Missing ChromaDB cloud configuration in .env file")
            return
        
        print(f"  Tenant ID: {tenant_id}")
        print(f"  Database: {database}")
        
        # Connect to Chroma Cloud
        client = chromadb.CloudClient(
            api_key=api_key,
            tenant=tenant_id,
            database=database
        )
        
        # First, list all collections to see what exists
        print("📚 Listing all collections...")
        collections = client.list_collections()
        print(f"Found {len(collections)} collections:")
        for col in collections:
            print(f"  - {col.name}")
        
        if not collections:
            print("❌ No collections found! ChromaDB is empty.")
            return
        
        # Try to find the main collection (might be named differently)
        collection_name = None
        for col in collections:
            if col.name.lower() in ['documents', 'docs', 'pyrag', 'main']:
                collection_name = col.name
                break
        
        if not collection_name:
            collection_name = collections[0].name  # Use first available
        
        print(f"📚 Using collection: {collection_name}")
        collection = client.get_collection(collection_name)
        
        # Get total document count
        total_docs = collection.count()
        print(f"📊 Total documents in collection: {total_docs:,}")
        
        # Get all metadata in batches (ChromaDB has limits)
        print("🔄 Fetching metadata in batches...")
        all_metadatas = []
        batch_size = 300  # ChromaDB Cloud quota limit
        offset = 0
        
        while True:
            try:
                # Get batch of documents
                batch_data = collection.get(
                    include=["metadatas"],
                    limit=batch_size,
                    offset=offset
                )
                batch_metadatas = batch_data.get("metadatas", [])
                
                if not batch_metadatas:
                    break
                    
                all_metadatas.extend(batch_metadatas)
                offset += batch_size
                
                print(f"  Fetched {len(all_metadatas):,} documents so far...")
                
                # Safety limit to avoid infinite loops
                if offset > 60000:
                    print("  Reached safety limit of 60,000 documents")
                    break
                    
            except Exception as e:
                print(f"  Error fetching batch at offset {offset}: {e}")
                break
        print(f"✅ Retrieved metadata for {len(all_metadatas):,} documents")
        
        if not all_metadatas:
            print("❌ No metadata found!")
            return
        
        # Extract unique library values and count documents per library
        library_counts = Counter()
        libraries_with_no_metadata = 0
        
        for metadata in all_metadatas:
            library_name = None
            if metadata:
                # Check both library and library_name fields
                library_name = metadata.get("library") or metadata.get("library_name")
            
            if library_name:
                library_counts[library_name] += 1
            else:
                libraries_with_no_metadata += 1
        
        # Get unique libraries
        unique_libraries = sorted(library_counts.keys())
        
        print(f"\n🎯 RESULTS:")
        print("=" * 50)
        print(f"📚 Total unique libraries: {len(unique_libraries)}")
        print(f"📄 Documents with library metadata: {sum(library_counts.values()):,}")
        if libraries_with_no_metadata > 0:
            print(f"⚠️  Documents without library metadata: {libraries_with_no_metadata:,}")
        
        print(f"\n📖 LIBRARIES & DOCUMENT COUNTS:")
        print("=" * 50)
        
        for library in unique_libraries:
            count = library_counts[library]
            print(f"  {library:<25} {count:>8,} docs")
        
        # Summary stats
        total_with_metadata = sum(library_counts.values())
        avg_docs_per_library = total_with_metadata / len(unique_libraries) if unique_libraries else 0
        
        print(f"\n📈 STATISTICS:")
        print("=" * 50)
        print(f"  Average docs per library: {avg_docs_per_library:.1f}")
        print(f"  Largest library: {max(library_counts, key=library_counts.get)} ({max(library_counts.values()):,} docs)")
        print(f"  Smallest library: {min(library_counts, key=library_counts.get)} ({min(library_counts.values()):,} docs)")
        
        return unique_libraries, library_counts
        
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None, None

if __name__ == "__main__":
    print("🚀 PyRAG Library Checker")
    print("=" * 50)
    
    # Check if we're in the right directory
    if not os.path.exists("src/pyrag"):
        print("⚠️  Warning: Not in PyRAG project root directory")
        print(f"Current directory: {os.getcwd()}")
    
    libraries, counts = check_libraries()
    
    if libraries:
        print(f"\n✅ Found {len(libraries)} libraries in your ChromaDB!")
    else:
        print("\n❌ Failed to retrieve library information")