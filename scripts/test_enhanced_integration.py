#!/usr/bin/env python3
"""Test script for enhanced processing integration into two-phase pipeline."""

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

async def test_enhanced_integration():
    """Test the integration of enhanced processing into the two-phase pipeline."""
    
    print("🧪 Testing Enhanced Processing Integration")
    print("=" * 50)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = get_config()
    if not validate_config(config):
        print("❌ Configuration validation failed")
        return
    
    print("✅ Configuration loaded successfully")
    
    # Initialize components
    print("\n2. Initializing components...")
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    llm_client = LLMClient(config.llm)
    
    print("✅ Components initialized")
    
    # Test LLM client
    print("\n3. Testing LLM client...")
    try:
        is_healthy = await llm_client.health_check()
        print(f"✅ LLM client health check: {is_healthy}")
    except Exception as e:
        print(f"❌ LLM client health check failed: {e}")
        return
    
    # Initialize documentation manager with enhanced processing
    print("\n4. Initializing documentation manager with enhanced processing...")
    doc_manager = DocumentationManager(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,  # This will enable enhanced processing
        firecrawl_api_key=config.firecrawl.api_key
    )
    
    print("✅ Documentation manager initialized with enhanced processing")
    
    # Create a test job
    print("\n5. Creating test documentation job...")
    job = DocumentationJob(
        library_name="test_enhanced",
        version="1.0.0",
        base_url="https://docs.firecrawl.dev/introduction",
        output_dir="./test_output",
        max_crawl_depth=1,
        max_crawl_pages=3,
        max_content_pages=2,
        use_llm_filtering=True
    )
    
    print("✅ Test job created")
    
    # Execute the enhanced ingestion
    print("\n6. Executing enhanced documentation ingestion...")
    try:
        result = await doc_manager.ingest_documentation(job)
        
        if result.success:
            print("✅ Enhanced ingestion completed successfully!")
            
            # Display results
            print(f"\n📊 Crawl Results:")
            print(f"   - Relevant URLs: {len(result.crawl_result.relevant_urls)}")
            print(f"   - Total URLs crawled: {result.crawl_result.crawl_stats.get('total_urls', 0)}")
            
            print(f"\n📊 Extraction Results:")
            print(f"   - Documents extracted: {result.extraction_stats.get('extracted_urls', 0)}")
            print(f"   - Failed extractions: {result.extraction_stats.get('failed_urls', 0)}")
            
            print(f"\n📊 Processing Results:")
            print(f"   - Total chunks: {result.processing_stats.get('total_chunks', 0)}")
            print(f"   - Content types: {result.processing_stats.get('content_type_distribution', {})}")
            
            # Check for enhanced metadata
            enhanced_metadata = result.processing_stats.get('enhanced_metadata')
            if enhanced_metadata:
                print(f"\n🚀 Enhanced Processing Results:")
                print(f"   - Enhanced chunks: {enhanced_metadata.get('total_enhanced_chunks', 0)}")
                print(f"   - API paths found: {len(enhanced_metadata.get('api_paths', []))}")
                print(f"   - Function signatures: {len(enhanced_metadata.get('function_signatures', []))}")
                print(f"   - Parameters extracted: {len(enhanced_metadata.get('parameters', []))}")
                print(f"   - Examples found: {len(enhanced_metadata.get('examples', []))}")
                avg_score = enhanced_metadata.get('average_importance_score', 0)
                if isinstance(avg_score, (int, float)):
                    print(f"   - Average importance score: {avg_score:.2f}")
                else:
                    print(f"   - Average importance score: {avg_score}")
                
                # Show some examples
                if enhanced_metadata.get('api_paths'):
                    print(f"\n📝 Sample API paths:")
                    for path in enhanced_metadata['api_paths'][:3]:
                        print(f"   - {path}")
                
                if enhanced_metadata.get('function_signatures'):
                    print(f"\n📝 Sample function signatures:")
                    for sig in enhanced_metadata['function_signatures'][:3]:
                        print(f"   - {sig}")
            else:
                print(f"\n⚠️  No enhanced metadata found - may have used fallback processing")
            
            print(f"\n📊 Storage Results:")
            print(f"   - Stored chunks: {result.storage_stats.get('stored_chunks', 0)}")
            print(f"   - Storage success rate: {result.storage_stats.get('storage_success_rate', 0):.2%}")
            
        else:
            print("❌ Enhanced ingestion failed!")
            print(f"Errors: {result.errors}")
            
    except Exception as e:
        print(f"❌ Error during enhanced ingestion: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n🎉 Enhanced integration test completed!")

if __name__ == "__main__":
    asyncio.run(test_enhanced_integration())
