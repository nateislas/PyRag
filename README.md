# PyRAG - Production-Ready AI Documentation Assistant

> **Transform your AI coding experience with comprehensive, real-time Python documentation**

<div align="center">

## 🚀 **NOW LIVE & PRODUCTION READY** 🚀

**Server URL**: https://PyRAG-MCP.fastmcp.app/mcp  
**11 comprehensive libraries** • **Multi-dimensional search** • **Real-time streaming**

</div>

**The Problem**: LLMs have outdated knowledge. When you're coding with AI assistants like Cursor, they often give you wrong, outdated, or incomplete information about Python libraries. This causes:
- ❌ **Frustrating errors** and failed code generation
- ❌ **Wasted time** fixing outdated examples
- ❌ **Increased costs** from repeated queries trying to get it right
- ❌ **Development slowdown** when your AI assistant hits a wall

**The Solution**: PyRAG is a production MCP (Model Context Protocol) server that gives your AI assistant access to **current, comprehensive Python documentation** with advanced multi-dimensional search capabilities. No more outdated information, no more getting stuck.

## 🚀 **What This Means for You**

### **Advanced Search Capabilities**
- **Multi-Dimensional Search**: Parallel searches across architecture, implementation, deployment, monitoring, security, and testing dimensions
- **AI-Optimized Responses**: 10,000+ character comprehensive responses with complete topic coverage
- **Real-Time Streaming**: Live progress updates for complex queries
- **Intelligent Query Analysis**: Automatically detects simple vs comprehensive query intent

### **Production-Ready Features**
- **11 Major Libraries**: FastAPI, Django, LangChain, Grafana, Pydantic, LlamaIndex, Streamlit, CrewAI, LangSmith, Ragas, Flask
- **Comprehensive Coverage**: API references, tutorials, examples, and best practices
- **Live Server**: Deployed and ready at https://PyRAG-MCP.fastmcp.app/mcp
- **Seamless Integration**: Works automatically with Cursor IDE and other MCP-compatible assistants

### **Real-World Examples**

**Before PyRAG**: Your AI assistant gives outdated information that causes errors:

```
You: "How do I handle authentication in FastAPI?"

AI Assistant: "Use the old security approach:
from fastapi.security import OAuth2PasswordBearer
# ... outdated code that doesn't work with current FastAPI versions
```

**With PyRAG**: Your AI assistant provides current, working information:

```
You: "How do I handle authentication in FastAPI?"

AI Assistant: "Here's the current best practice using FastAPI's built-in security features:

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from jose import JWTError, jwt

app = FastAPI()
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

@app.get("/protected")
async def protected_route(current_user = Depends(get_current_user)):
    return {"message": "Access granted", "user": current_user}
```

## 📚 **11 Production Libraries**

Your AI assistant now has comprehensive documentation for **11 major Python libraries**:

### **🏆 Production Library Coverage**

| Library | Documents | Focus Area |
|---------|-----------|------------|
| **FastAPI** | 14,135 | Modern web APIs & async frameworks |
| **Grafana** | 9,358 | Data visualization & monitoring |
| **LangChain** | 9,044 | LLM application development |
| **Django** | 7,785 | Full-stack web development |
| **Pydantic** | 5,664 | Data validation & settings management |
| **LlamaIndex** | 3,537 | RAG & document retrieval systems |
| **Streamlit** | 1,807 | Rapid AI application prototyping |
| **CrewAI** | 1,395 | Multi-agent AI systems |
| **LangSmith** | 1,290 | LLM observability & debugging |
| **Ragas** | 492 | RAG evaluation & metrics |
| **Flask** | 55 | Lightweight web applications |

### **📊 Documentation Coverage**

- **Content Types**: API references, tutorials, examples, guides, changelogs
- **Real-Time Updates**: Continuous ingestion ensures current information
- **Multi-Dimensional Coverage**: Architecture, implementation, deployment, monitoring, security, testing

## 🎯 **How It Works**

1. **You ask your AI assistant** about Python libraries (just like normal)
2. **Your AI assistant connects** to the PyRAG MCP server via HTTPS
3. **PyRAG searches** comprehensive, up-to-date documentation
4. **You get better answers** with current information and examples

**Simple setup**: Just configure your MCP client to connect to the PyRAG server, then start asking questions!

> **Note**: Claude requires HTTPS connections for security. See [MCP HTTPS Setup](docs/mcp_https_setup.md) for configuration details.

## 🔍 **What You Can Ask**

Just ask your AI assistant normally about Python libraries:

### **API Questions**
- "How do I use pandas.read_csv() with custom delimiters?"
- "What are all the parameters for requests.Session()?"
- "How do I create a FastAPI endpoint with query parameters?"

### **Code Examples**
- "Show me examples of async/await in aiohttp"
- "How do I implement caching in FastAPI?"
- "Give me examples of pandas data manipulation"

### **Troubleshooting**
- "Why am I getting a ModuleNotFoundError with pandas?"
- "How do I handle memory issues with large datasets?"
- "What's the best way to structure a FastAPI project?"

### **Best Practices**
- "What are the recommended patterns for error handling in async code?"
- "How do I optimize performance in data processing?"
- "What are common pitfalls when using LangChain?"

## 🛠️ **Getting Started**

### **For Cursor IDE Users**

**Step 1: Configure MCP in Cursor**
1. Open Cursor IDE
2. Go to **Settings** → **Extensions** → **MCP**
3. Add PyRAG server configuration:
```json
{
  "mcp.servers": {
    "pyrag": {
      "command": "curl",
      "args": ["-X", "POST", "https://your-pyrag-server.com/mcp"],
      "env": {}
    }
  }
}
```

**Step 2: Start Using It**
- Open a Python file in Cursor
- Ask your AI assistant: "How do I use pandas.read_csv()?"
- Your assistant will automatically use PyRAG for current documentation!

### **For Other MCP-Compatible AI Assistants**
Configure your MCP client to connect to: `https://PyRAG-MCP.fastmcp.app/mcp`

## 🏗️ **Production Architecture**

PyRAG features a sophisticated multi-dimensional search system:
- **Intelligent Query Analysis**: LLM-powered intent detection and query expansion
- **Parallel Search Execution**: Simultaneous searches across 4-7 knowledge dimensions  
- **Topic Coverage Engine**: Ensures comprehensive responses with gap detection
- **Real-Time Streaming**: FastMCP streaming with live progress updates
- **ChromaDB Cloud**: Production vector storage with comprehensive documentation
- **Crawl4AI Integration**: Unlimited local web scraping for current documentation

## 🤝 **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### **For Developers**
- **Data Ingestion Pipeline**: [Pipeline Overview](docs/data-ingestion-pipeline-overview.md)
- **RAG Architecture**: [RAG Pipeline Overview](docs/rag-pipeline-overview.md)
- **Multi-Dimensional Search**: [Search Engine Implementation](src/pyrag/search/)

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 **Acknowledgments**

- **Crawl4AI** for unlimited local web scraping capabilities
- **ChromaDB Cloud** for production-grade vector storage
- **FastMCP** for streaming MCP server implementation
- **MCP Community** for the Model Context Protocol specification

## 📞 **Support**

- **Issues**: [GitHub Issues](https://github.com/nateislas/PyRag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/nateislas/PyRag/discussions)
- **Documentation**: [Technical Documentation](docs/DOCUMENTATION.md)

---

**Ready to supercharge your AI coding experience?** 🚀

Connect to https://PyRAG-MCP.fastmcp.app/mcp and experience multi-dimensional search with comprehensive, production-ready responses that actually help you build better software faster!
