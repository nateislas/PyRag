# Qodo Embedding Model Upgrade

## 🎯 **Overview**

PyRAG has been upgraded to use the **Qodo-Embed-1-1.5B** model, a state-of-the-art code embedding model specifically designed for software development and retrieval tasks.

## 🚀 **Key Benefits**

### **Performance Improvements**
- **4x Higher Dimensionality**: 1536 vs 384 dimensions = richer semantic representations
- **Code-Specific Understanding**: Built specifically for programming languages and APIs
- **Multi-Language Support**: Python, C++, C#, Go, Java, JavaScript, PHP, Ruby, TypeScript
- **Higher Token Limit**: Supports up to 32,768 tokens (vs previous 512 limit)

### **RAG System Improvements**
- **Better Code Retrieval**: Superior understanding of function signatures, parameters, and code structure
- **Improved Semantic Search**: More accurate matching between queries and code documentation
- **Enhanced Context Understanding**: Better preservation of semantic relationships in code chunks

## 📊 **Model Comparison**

| Feature | Previous Model | New Qodo Model | Improvement |
|---------|----------------|----------------|-------------|
| **Model** | `all-MiniLM-L6-v2` | `Qodo-Embed-1-1.5B` | - |
| **Dimensions** | 384 | 1536 | **4x increase** |
| **Max Tokens** | 512 | 32,768 | **64x increase** |
| **Code Understanding** | General purpose | **Code-optimized** | **Specialized** |
| **Languages** | General | **10+ programming languages** | **Multi-language** |
| **Performance** | Baseline | **State-of-the-art** | **Best-in-class** |

## 🔧 **Installation**

### **1. Run the Setup Script**
```bash
# Activate your virtual environment first
source venv/bin/activate

# Run the automated setup
python scripts/setup_qodo_embeddings.py
```

### **2. Manual Installation (Alternative)**
```bash
# Install required dependencies
pip install sentence-transformers>=2.2.0
pip install flash-attn>=2.5.6  # Optional but recommended for performance

# Verify installation
python -c "from sentence_transformers import SentenceTransformer; model = SentenceTransformer('Qodo/Qodo-Embed-1-1.5B'); print('✅ Success!')"
```

## ⚙️ **Configuration**

### **Environment Variables**
```bash
# .env file
EMBEDDING_MODEL=Qodo/Qodo-Embed-1-1.5B
EMBEDDING_DEVICE=auto          # auto, cuda, or cpu
EMBEDDING_MAX_LENGTH=32768     # Max tokens (up to 32k)
EMBEDDING_BATCH_SIZE=8         # Adjust based on GPU memory
EMBEDDING_NORMALIZE=true       # Normalize embeddings
```

### **Configuration Options**
- **`EMBEDDING_DEVICE`**: 
  - `auto`: Automatically detect best device
  - `cuda`: Force GPU usage (if available)
  - `cpu`: Force CPU usage
- **`EMBEDDING_BATCH_SIZE`**: 
  - Start with 8, increase if you have more GPU memory
  - Decrease if you encounter out-of-memory errors
- **`EMBEDDING_MAX_LENGTH`**: 
  - Default: 32,768 (maximum supported)
  - Can be reduced for memory optimization

## 🧪 **Testing**

### **Run the Test Script**
```bash
python scripts/test_qodo_embeddings.py
```

This will:
- ✅ Load the Qodo model
- ✅ Generate test embeddings
- ✅ Calculate similarities
- ✅ Verify batch processing
- ✅ Check performance metrics

### **Expected Output**
```
🚀 Testing Qodo Embedding Service
==================================================
📋 Configuration loaded:
   - Model: Qodo/Qodo-Embed-1-1.5B
   - Device: cuda
   - Max Length: 32768
   - Batch Size: 8

🔧 Initializing embedding service...
🏥 Performing health check...
   - Status: healthy
   - Device: cuda:0
   - Dimension: 1536

✅ All tests passed! Qodo embedding service is working correctly.
```

## 🔄 **Migration Guide**

### **What Changed**
1. **Model**: From `sentence-transformers/all-MiniLM-L6-v2` to `Qodo/Qodo-Embed-1-1.5B`
2. **Dimensions**: From 384 to 1536
3. **Token Limit**: From 512 to 32,768
4. **Performance**: Significantly improved code understanding

### **Breaking Changes**
- **Embedding Dimensions**: Existing vector databases will need to be recreated
- **API**: No breaking changes in the Python API
- **Configuration**: New environment variables available

### **Migration Steps**

#### **1. Backup Existing Data**
```bash
# Backup your current vector database
cp -r chroma_db chroma_db_backup
```

#### **2. Install New Dependencies**
```bash
python scripts/setup_qodo_embeddings.py
```

#### **3. Test the New System**
```bash
python scripts/test_qodo_embeddings.py
```

#### **4. Re-embed Existing Documents**
```bash
# Run the ingestion pipeline to re-embed all documents
python scripts/check_current_state.py
```

#### **5. Verify Migration**
```bash
# Check that new embeddings are 1536 dimensions
python -c "
from pyrag.embeddings import EmbeddingService
service = EmbeddingService()
print(f'Embedding dimension: {service.get_embedding_dimension()}')
# Should print: Embedding dimension: 1536
"
```

## 📈 **Performance Optimization**

### **GPU Memory Management**
```bash
# Monitor GPU memory usage
nvidia-smi

# Adjust batch size based on available memory
export EMBEDDING_BATCH_SIZE=4  # Reduce if out of memory
export EMBEDDING_BATCH_SIZE=16 # Increase if you have more memory
```

### **Batch Size Guidelines**
- **GPU Memory < 8GB**: Use batch size 4-8
- **GPU Memory 8-16GB**: Use batch size 8-16
- **GPU Memory > 16GB**: Use batch size 16-32

### **Flash Attention**
The `flash-attn` package provides significant performance improvements:
- **Faster inference**: 2-3x speedup
- **Lower memory usage**: More efficient attention computation
- **Better GPU utilization**: Optimized CUDA kernels

## 🐛 **Troubleshooting**

### **Common Issues**

#### **1. Out of Memory Errors**
```bash
# Reduce batch size
export EMBEDDING_BATCH_SIZE=4

# Force CPU usage
export EMBEDDING_DEVICE=cpu
```

#### **2. Model Download Issues**
```bash
# Clear cache and retry
rm -rf ~/.cache/huggingface/
python scripts/setup_qodo_embeddings.py
```

#### **3. Flash Attention Installation Fails**
```bash
# Install without flash attention (slower but functional)
pip install sentence-transformers>=2.2.0
# Skip flash-attn installation
```

#### **4. Performance Issues**
```bash
# Check device usage
python -c "
from pyrag.embeddings import EmbeddingService
service = EmbeddingService()
print(f'Device: {service.model.device}')
print(f'Model loaded: {service.is_ready()}')
"
```

### **Debug Mode**
```bash
# Enable debug logging
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/test_qodo_embeddings.py
```

## 📚 **API Reference**

### **EmbeddingService Class**

#### **Methods**
- `generate_embeddings(texts)`: Generate embeddings for text(s)
- `similarity(emb1, emb2)`: Calculate cosine similarity
- `batch_similarity(query, documents)`: Calculate similarities between query and documents
- `health_check()`: Check service health and configuration
- `get_embedding_dimension()`: Get embedding dimension (1536)

#### **Configuration**
- `max_length`: Maximum token length (up to 32,768)
- `batch_size`: Processing batch size
- `device`: Processing device (auto/cuda/cpu)
- `normalize_embeddings`: Whether to normalize embeddings

## 🎯 **Next Steps**

### **Immediate Actions**
1. ✅ Install new dependencies
2. ✅ Test the embedding service
3. ✅ Re-embed existing documents
4. ✅ Monitor performance improvements

### **Future Enhancements**
1. **Fine-tuning**: Consider fine-tuning on your specific codebase
2. **Model Variants**: Evaluate larger models (7B parameter version)
3. **Performance Monitoring**: Track embedding quality metrics
4. **A/B Testing**: Compare old vs new embedding quality

## 📞 **Support**

### **Getting Help**
- **Documentation**: Check this guide and the main PyRAG docs
- **Issues**: Report problems in the GitHub issues
- **Discussions**: Join community discussions

### **Useful Commands**
```bash
# Check model status
python -c "from pyrag.embeddings import EmbeddingService; print(EmbeddingService().health_check())"

# Test embedding generation
python -c "
from pyrag.embeddings import EmbeddingService
import asyncio
service = EmbeddingService()
result = asyncio.run(service.generate_embeddings('Hello world'))
print(f'Shape: {result.shape}, Type: {type(result)}')
"
```

---

**🎉 Congratulations!** You've successfully upgraded to the state-of-the-art Qodo embedding model. Your RAG system now has significantly improved code understanding and retrieval capabilities.
