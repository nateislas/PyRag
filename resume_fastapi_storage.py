#!/usr/bin/env python3
"""Resume FastAPI storage from already processed chunks."""

import asyncio
import json
from pathlib import Path

from src.pyrag.ingestion.documentation_manager import DocumentationManager
from src.pyrag.vector_store import VectorStore
from src.pyrag.embeddings import EmbeddingService
from src.pyrag.llm.client import LLMClient
from src.pyrag.config import get_config

async def resume_fastapi_storage():
    """Resume FastAPI storage from already processed chunks."""
    print("🚀 Resuming FastAPI storage from processed chunks...")
    
    # Initialize services
    config = get_config()
    vector_store = VectorStore()
    embedding_service = EmbeddingService()
    llm_client = LLMClient(config)
    
    # Reset collections to handle new embedding dimensions
    print("🔄 Resetting collections for new embedding dimensions...")
    await vector_store.reset_collections()
    
    # Create documentation manager
    manager = DocumentationManager(
        vector_store=vector_store,
        embedding_service=embedding_service,
        llm_client=llm_client,
        firecrawl_api_key=config.firecrawl.api_key
    )
    
    # Check if we have cached processed chunks
    cache_dir = Path("./cache/fastapi")
    if cache_dir.exists():
        print(f"📁 Found cache directory: {cache_dir}")
        
        # Look for processed chunks
        chunk_files = list(cache_dir.glob("*_chunks.json"))
        if chunk_files:
            print(f"📄 Found {len(chunk_files)} chunk files")
            
            # Load and store chunks
            total_stored = 0
            for chunk_file in chunk_files:
                try:
                    with open(chunk_file, 'r') as f:
                        chunks_data = json.load(f)
                    
                    print(f"💾 Storing chunks from {chunk_file.name}...")
                    
                    # Store chunks
                    for chunk_data in chunks_data:
                        try:
                            # Generate embedding
                            embedding = await embedding_service.generate_embeddings(chunk_data['content'])
                            
                            # Convert to list format
                            if hasattr(embedding, 'tolist'):
                                embedding = embedding.tolist()
                            
                            # Ensure flat list
                            if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                                embedding = embedding[0]
                            
                            # Prepare document
                            doc = {
                                'content': chunk_data['content'],
                                'metadata': chunk_data['metadata'],
                                'embedding': embedding
                            }
                            
                            # Store in vector store
                            chunk_ids = await vector_store.add_documents([doc])
                            total_stored += len(chunk_ids)
                            
                        except Exception as e:
                            print(f"❌ Error storing chunk: {e}")
                            continue
                    
                    print(f"✅ Stored chunks from {chunk_file.name}")
                    
                except Exception as e:
                    print(f"❌ Error processing {chunk_file}: {e}")
                    continue
            
            print(f"🎉 Successfully stored {total_stored} chunks!")
            
        else:
            print("⚠️ No chunk files found in cache")
    else:
        print("⚠️ No cache directory found")
    
    print("🏁 Resume complete!")

if __name__ == "__main__":
    asyncio.run(resume_fastapi_storage())
