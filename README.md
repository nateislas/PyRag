# PyRAG - Smarter Python Documentation for AI Assistants

> **Stop your AI assistant from getting stuck with outdated Python documentation**

<div align="center">

# 🚧 **CURRENTLY IN DEVELOPMENT** 🚧

**PyRAG is actively being developed and is not yet ready for production use.**
**The server will be available soon!**

</div>

**The Problem**: LLMs have outdated knowledge. When you're coding with AI assistants like Cursor, they often give you wrong, outdated, or incomplete information about Python libraries. This causes:
- ❌ **Frustrating errors** and failed code generation
- ❌ **Wasted time** fixing outdated examples
- ❌ **Increased costs** from repeated queries trying to get it right
- ❌ **Development slowdown** when your AI assistant hits a wall

**The Solution**: PyRAG is an MCP (Model Context Protocol) server that gives your AI assistant access to **current, comprehensive Python documentation**. No more outdated information, no more getting stuck.

## 🚀 **What This Means for You**

### **Stop Getting Stuck**
- **Current Information**: Your AI assistant always has the latest Python documentation
- **No More Outdated Examples**: Get working code that actually runs
- **Comprehensive Coverage**: Access to detailed API references, examples, and best practices
- **Context-Aware Answers**: Responses tailored to your specific use case and skill level

### **Faster Development**
- **Fewer Errors**: Your AI assistant won't suggest deprecated methods or wrong syntax
- **Reduced Query Costs**: Get it right the first time, not after multiple attempts
- **Seamless Experience**: Works automatically with MCP-compatible AI assistants
- **Instant Access**: No need to search documentation manually

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

## 📚 **Supported Libraries**

Your AI assistant now has comprehensive documentation for:

- **LangChain** - Framework for developing applications with LLMs
- **FastAPI** - Modern web framework for building APIs
- **Transformers** - State-of-the-art Natural Language Processing
- **Pydantic** - Data validation using Python type annotations
- **OpenAI** - OpenAI API client library

*More libraries are being added regularly.*

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
Configure your MCP client to connect to: `https://your-pyrag-server.com/mcp`

### **Server Status**
- **Server URL**: `https://your-pyrag-server.com/mcp`
- **Status**: Coming soon! (Currently in development)
- **Supported Libraries**: LangChain, FastAPI, Transformers, Pydantic, OpenAI
- **Coverage**: Comprehensive documentation with examples and best practices

## 🏗️ **Behind the Scenes**

PyRAG uses advanced AI to:
- **Intelligently crawl** Python library documentation
- **Extract relevant content** with context and examples
- **Provide semantic search** that understands your questions
- **Keep documentation current** with regular updates

## 🤝 **Contributing**

We welcome contributions! See our [Contributing Guide](CONTRIBUTING.md) for details.

### **For Developers**
- **Architecture Overview**: [Implementation Guide](docs/implementation_guide.md)
- **Current Status**: [Phase 3 Completion](docs/phase3_completion_summary.md)
- **Future Roadmap**: [RAG Improvement Plan](docs/rag_improvement_plan.md)

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
- **Documentation**: [Technical Documentation](docs/DOCUMENTATION.md)

---

**Tired of your AI assistant getting stuck with outdated information?** 🚀

Just ask your AI assistant about Python libraries - it will automatically use PyRAG to give you current, working information that actually helps you code faster!
