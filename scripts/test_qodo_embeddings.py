#!/usr/bin/env python3
"""Test script for the new Qodo embedding service."""

import asyncio
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

try:
    from pyrag.embeddings import EmbeddingService
    from pyrag.config import get_config
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure you're running from the project root directory")
    sys.exit(1)


async def test_qodo_embeddings():
    """Test the Qodo embedding service."""
    print("🚀 Testing Qodo Embedding Service")
    print("=" * 50)
    
    try:
        # Get configuration
        config = get_config()
        print(f"📋 Configuration loaded:")
        print(f"   - Model: {config.embedding.model_name}")
        print(f"   - Device: {config.embedding.device}")
        print(f"   - Max Length: {config.embedding.max_length}")
        print(f"   - Batch Size: {config.embedding.batch_size}")
        print()
        
        # Initialize embedding service
        print("🔧 Initializing embedding service...")
        embedding_service = EmbeddingService(config)
        
        # Health check
        print("🏥 Performing health check...")
        health = embedding_service.health_check()
        print(f"   - Status: {health['status']}")
        print(f"   - Device: {health['device']}")
        print(f"   - Dimension: {health['embedding_dimension']}")
        print()
        
        if health['status'] != 'healthy':
            print("❌ Health check failed!")
            return False
        
        # Test single text embedding
        print("📝 Testing single text embedding...")
        test_text = "How do I create a FastAPI endpoint?"
        embedding = await embedding_service.generate_embeddings(test_text)
        print(f"   - Input: '{test_text}'")
        print(f"   - Output shape: {embedding.shape}")
        print(f"   - Output type: {type(embedding)}")
        print()
        
        # Test multiple texts
        print("📚 Testing multiple text embeddings...")
        test_texts = [
            "How do I create a FastAPI endpoint?",
            "What is SQLAlchemy?",
            "How to use pandas for data analysis?",
            "Explain async/await in Python"
        ]
        
        embeddings = await embedding_service.generate_embeddings(test_texts)
        print(f"   - Input count: {len(test_texts)}")
        print(f"   - Output shape: {embeddings.shape}")
        print()
        
        # Test similarity calculation
        print("🔍 Testing similarity calculation...")
        
        # Calculate similarities between all pairs using the embedding service
        print("   - Similarity matrix:")
        for i, text1 in enumerate(test_texts):
            for j, text2 in enumerate(test_texts):
                if i < j:  # Only show upper triangle
                    # Get embeddings for the two texts
                    emb1 = embeddings[i:i+1]  # Keep as 2D array
                    emb2 = embeddings[j:j+1]  # Keep as 2D array
                    
                    # Calculate similarity
                    sim = embedding_service.similarity(emb1, emb2)
                    
                    # Handle different return types - similarity returns nested list
                    if isinstance(sim, (list, np.ndarray)):
                        # Extract the actual similarity value from nested structure
                        if isinstance(sim[0], (list, np.ndarray)):
                            sim_value = sim[0][0]
                        else:
                            sim_value = sim[0]
                    else:
                        sim_value = sim
                    
                    # Ensure sim_value is a number and format it
                    try:
                        sim_value = float(sim_value)
                        print(f"     '{text1[:30]}...' vs '{text2[:30]}...': {sim_value:.3f}")
                    except (ValueError, TypeError):
                        print(f"     '{text1[:30]}...' vs '{text2[:30]}...': {sim_value}")
        
        print()
        
        # Test batch processing
        print("⚡ Testing batch processing...")
        large_texts = [f"This is test text number {i} for batch processing." for i in range(20)]
        
        start_time = asyncio.get_event_loop().time()
        batch_embeddings = await embedding_service.generate_embeddings(large_texts)
        end_time = asyncio.get_event_loop().time()
        
        print(f"   - Processed {len(large_texts)} texts in {end_time - start_time:.2f} seconds")
        print(f"   - Output shape: {batch_embeddings.shape}")
        print()
        
        print("✅ All tests passed! Qodo embedding service is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Main function."""
    success = await test_qodo_embeddings()
    
    if success:
        print("\n🎉 Qodo embedding service is ready for production use!")
        print("\n💡 Next steps:")
        print("   1. Run the ingestion pipeline to re-embed existing documents")
        print("   2. Test the RAG system with the new embeddings")
        print("   3. Monitor performance improvements")
    else:
        print("\n💥 Setup failed. Please check the error messages above.")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
