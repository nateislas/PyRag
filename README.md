# PyRAG - Python Documentation RAG System

> **Intelligent Python Documentation Search & Retrieval via MCP Server**

PyRAG is a powerful Retrieval-Augmented Generation (RAG) system that provides intelligent access to Python library documentation through a Model Context Protocol (MCP) server. It enables AI assistants to search, retrieve, and understand Python documentation with high relevance and context awareness.

## 🚀 **Features**

### **Intelligent Documentation Search**
- **Multi-Library Support**: Comprehensive documentation for top Python libraries
- **Semantic Search**: Find relevant information even with natural language queries
- **Context-Aware Results**: Results tailored to your specific use case and skill level
- **Real-Time Updates**: Always access the latest documentation

### **Advanced RAG Capabilities**
- **Two-Phase Ingestion**: Intelligent crawling and content extraction
- **LLM-Guided Filtering**: Smart identification of relevant documentation
- **Rich Metadata**: Detailed API information, examples, and relationships
- **Hybrid Search**: Combines vector similarity with keyword matching

### **MCP Server Integration**
- **Seamless AI Integration**: Works with any MCP-compatible AI assistant
- **Rich Context**: Provides detailed, relevant documentation snippets
- **Query Understanding**: Understands complex documentation requests
- **Multi-Modal Results**: Returns code examples, explanations, and API references

## 🎯 **Use Cases**

### **For Developers**
- **Quick API Reference**: "How do I use requests.get() with authentication?"
- **Code Examples**: "Show me examples of FastAPI dependency injection"
- **Troubleshooting**: "What causes 'ModuleNotFoundError' in pandas?"
- **Best Practices**: "What's the recommended way to handle async operations in aiohttp?"

### **For AI Assistants**
- **Accurate Code Generation**: Access to current, relevant documentation
- **Context-Aware Responses**: Understand library-specific patterns and conventions
- **Multi-Library Support**: Handle projects using multiple Python libraries
- **Real-Time Information**: Always provide up-to-date documentation

## 🛠️ **Setup & Installation**

### **Prerequisites**
- Python 3.11+
- Git
- Cursor IDE (recommended) or any MCP-compatible client

### **Quick Start**

1. **Clone the Repository**
   ```bash
   git clone https://github.com/nateislas/PyRag.git
   cd PyRag
   ```

2. **Set Up Environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -e .
   ```

3. **Configure API Keys**
   ```bash
   # Create .env file
   cp .env.example .env
   
   # Add your API keys
   echo "LLAMA_API_KEY=your_llama_api_key" >> .env
   echo "FIRECRAWL_API_KEY=your_firecrawl_api_key" >> .env
   ```

4. **Ingest Documentation** (Optional)
   ```bash
   # Ingest top Python libraries
   python scripts/automated_library_ingestion.py
   ```

5. **Start MCP Server**
   ```bash
   python -m pyrag.mcp.server
   ```

### **Cursor IDE Integration**

1. **Install MCP Extension**
   - Open Cursor IDE
   - Go to Extensions → Search for "MCP"
   - Install the MCP extension

2. **Configure PyRAG Server**
   ```json
   // In Cursor settings.json
   {
     "mcp.servers": {
       "pyrag": {
         "command": "python",
         "args": ["-m", "pyrag.mcp.server"],
         "env": {
           "PYTHONPATH": "/path/to/PyRag/src"
         }
       }
     }
   }
   ```

3. **Test Integration**
   - Open a Python file in Cursor
   - Ask your AI assistant: "How do I use pandas.read_csv()?"
   - The assistant will now have access to comprehensive pandas documentation

## 📚 **Supported Libraries**

PyRAG currently provides comprehensive documentation for:

- **LangChain** - Framework for developing applications with LLMs
- **FastAPI** - Modern web framework for building APIs
- **Transformers** - State-of-the-art Natural Language Processing
- **Pydantic** - Data validation using Python type annotations
- **OpenAI** - OpenAI API client library

*More libraries are being added regularly. Check our [library configuration](config/libraries.json) for the complete list.*

## 🔍 **Usage Examples**

### **Basic Queries**
```
"How do I make HTTP requests with authentication?"
"Show me examples of FastAPI dependency injection"
"What's the difference between pandas merge and join?"
"How do I handle async operations in aiohttp?"
```

### **Advanced Queries**
```
"Compare the performance of different pandas data loading methods"
"Show me best practices for error handling in FastAPI"
"What are the common pitfalls when using LangChain agents?"
"How do I optimize memory usage in large pandas operations?"
```

### **Code-Specific Queries**
```
"Explain the parameters of requests.Session()"
"What does the 'inplace' parameter do in pandas operations?"
"How do I configure logging in FastAPI applications?"
"What are the return types for Transformers pipeline outputs?"
```

## 🏗️ **Architecture**

PyRAG uses a sophisticated two-phase ingestion system:

1. **Discovery Phase**: LLM-guided crawling identifies relevant documentation pages
2. **Extraction Phase**: High-quality content extraction with metadata preservation

The system then provides intelligent search through:
- **Vector Similarity**: Semantic understanding of queries
- **Metadata Filtering**: Precise filtering by library, version, content type
- **Hybrid Search**: Combines multiple search strategies for optimal results
- **Context Awareness**: Considers user's current context and preferences

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### **Development Setup**
```bash
# Install development dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run comprehensive system test
python scripts/test_complete_system.py
```

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Firecrawl** for high-quality web scraping capabilities
- **ChromaDB** for efficient vector storage
- **LangChain** for LLM integration patterns
- **MCP Community** for the Model Context Protocol specification

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/nateislas/PyRag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nateislas/PyRag/discussions)
- **Documentation**: [Implementation Guide](docs/implementation_guide.md)

---

**Ready to supercharge your Python development with intelligent documentation search?** 🚀

Get started with PyRAG today and experience the future of AI-assisted Python development!
