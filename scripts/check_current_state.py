#!/usr/bin/env python3
"""Check current state and run ingestion pipeline."""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from pyrag.config import get_config, validate_config
from pyrag.vector_store import VectorStore
from pyrag.embeddings import EmbeddingService
from pyrag.llm.client import LLMClient
from pyrag.ingestion.documentation_manager import DocumentationManager, DocumentationJob


async def check_current_state():
    """Check current state of the vector store."""
    print("🔍 Checking current state...")
    
    try:
        vector_store = VectorStore()
        collections = await vector_store.list_collections()
        
        print(f"📚 Current collections: {len(collections)}")
        
        total_chunks = 0
        for collection_info in collections:
            collection_name = collection_info["name"]
            count = collection_info["document_count"]
            total_chunks += count
            print(f"   {collection_name}: {count} chunks")
        
        print(f"📊 Total chunks: {total_chunks}")
        
        if total_chunks == 0:
            print("⚠️  No documentation found - need to run ingestion pipeline")
            return False
        else:
            print("✅ Documentation found - ready for evaluation")
            return True
            
    except Exception as e:
        print(f"❌ Error checking state: {e}")
        return False


async def run_ingestion_pipeline():
    """Run the automated documentation ingestion pipeline."""
    print("\n🚀 Starting automated documentation ingestion...")
    
    try:
        # Load configuration
        print("📋 Loading configuration...")
        config = get_config()
        
        if not validate_config(config):
            print("❌ Configuration validation failed")
            return False
        
        # Initialize components
        print("🔧 Initializing components...")
        vector_store = VectorStore()
        embedding_service = EmbeddingService()
        llm_client = LLMClient(config.llm)
        
        # Initialize documentation manager
        doc_manager = DocumentationManager(
            vector_store=vector_store,
            embedding_service=embedding_service,
            llm_client=llm_client,
            firecrawl_api_key=config.firecrawl.api_key
        )
        
        # Define libraries to ingest
        libraries_to_ingest = [
            {
                "name": "requests",
                "version": "2.31.0",
                "base_url": "https://docs.python-requests.org/en/latest/",
                "description": "HTTP library for Python"
            },
            {
                "name": "pandas",
                "version": "2.1.0",
                "base_url": "https://pandas.pydata.org/docs/",
                "description": "Data manipulation and analysis library"
            },
            {
                "name": "fastapi",
                "version": "0.104.0",
                "base_url": "https://fastapi.tiangolo.com/",
                "description": "Modern web framework for building APIs"
            },
            {
                "name": "sqlalchemy",
                "version": "2.0.0",
                "base_url": "https://docs.sqlalchemy.org/en/20/",
                "description": "SQL toolkit and Object Relational Mapper"
            },
            {
                "name": "pydantic",
                "version": "2.5.0",
                "base_url": "https://docs.pydantic.dev/",
                "description": "Data validation using Python type annotations"
            }
        ]
        
        print(f"📚 Ingesting {len(libraries_to_ingest)} libraries...")
        
        for i, lib_info in enumerate(libraries_to_ingest, 1):
            print(f"\n🔄 [{i}/{len(libraries_to_ingest)}] Ingesting {lib_info['name']}...")
            
            # Create documentation job with required output_dir
            job = DocumentationJob(
                library_name=lib_info["name"],
                version=lib_info["version"],
                base_url=lib_info["base_url"],
                output_dir=f"./cache/{lib_info['name']}",  # Add required output_dir
                max_crawl_depth=2,
                max_crawl_pages=30,
                max_content_pages=20,
                use_llm_filtering=True
            )
            
            try:
                # Execute ingestion
                result = await doc_manager.ingest_documentation(job)
                
                if result.success:
                    print(f"✅ {lib_info['name']} ingested successfully!")
                    print(f"   📄 Documents: {result.processing_stats.get('total_documents', 0)}")
                    print(f"   🧩 Chunks: {result.processing_stats.get('total_chunks', 0)}")
                    print(f"   ⏱️  Time: {result.processing_stats.get('total_time_seconds', 0):.1f}s")
                else:
                    print(f"❌ {lib_info['name']} ingestion failed: {result.errors}")
                    
            except Exception as e:
                print(f"❌ Error ingesting {lib_info['name']}: {e}")
                continue
        
        print("\n🎉 Ingestion pipeline completed!")
        return True
        
    except Exception as e:
        print(f"❌ Ingestion pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    print("🚀 PyRAG Documentation Ingestion Pipeline")
    print("=" * 60)
    
    # Check current state
    has_docs = await check_current_state()
    
    if not has_docs:
        print("\n📥 No documentation found - running ingestion pipeline...")
        success = await run_ingestion_pipeline()
        
        if success:
            print("\n🔍 Re-checking state after ingestion...")
            await check_current_state()
        else:
            print("\n❌ Ingestion failed - cannot proceed")
            return 1
    else:
        print("\n✅ Documentation already available - ready for evaluation")
        
        # Ask if user wants to re-ingest
        response = input("\n🤔 Do you want to re-run ingestion anyway? (y/N): ")
        if response.lower() in ['y', 'yes']:
            print("\n🔄 Re-running ingestion pipeline...")
            success = await run_ingestion_pipeline()
            
            if success:
                print("\n🔍 Re-checking state after ingestion...")
                await check_current_state()
            else:
                print("\n❌ Re-ingestion failed")
                return 1
    
    print("\n🎯 Ready for RAG evaluation!")
    return 0


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
