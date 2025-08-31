# PyRAG: Python Documentation RAG System

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

PyRAG is an open-source RAG (Retrieval-Augmented Generation) system specifically designed to keep AI coding assistants current with the latest Python library documentation. By providing real-time access to up-to-date API references, examples, and best practices through the Model Context Protocol (MCP), PyRAG solves the critical problem of AI assistants suggesting outdated or deprecated code patterns.

## 🚀 Features

- **Real-time Documentation Access**: Keep AI assistants current with the latest Python library documentation
- **Semantic Search**: Advanced search capabilities across documentation with context awareness
- **MCP Integration**: Seamless integration with coding assistants via Model Context Protocol
- **Multi-format Support**: Handles Sphinx, MkDocs, GitHub, and other documentation formats
- **Hierarchical Chunking**: Maintains context and relationships between documentation sections
- **Legal Compliance**: Built-in compliance tracking and opt-out mechanisms
- **Performance Optimized**: Sub-200ms query response times with intelligent caching

## 🏗️ Architecture

PyRAG is built with a modern, scalable architecture:

- **FastAPI**: High-performance web framework for the API layer
- **PostgreSQL + pgvector**: Relational database with vector similarity search
- **ChromaDB/Weaviate**: Vector database for semantic search
- **Redis**: Caching and task queuing
- **Celery**: Background task processing
- **Docker**: Containerized development and deployment

## 📋 Requirements

- Python 3.11+
- Docker and Docker Compose
- 8GB+ RAM (16GB recommended for embedding generation)
- 10GB+ free disk space

## 🛠️ Quick Start

### Development Setup

1. **Clone the repository**
   ```bash
   git clone https://github.com/pyrag/pyrag.git
   cd pyrag
   ```

2. **Start the development environment**
   ```bash
   docker-compose up -d
   ```

3. **Install dependencies**
   ```bash
   pip install -e .[dev]
   ```

4. **Run tests**
   ```bash
   pytest
   ```

5. **Start the API server**
   ```bash
   uvicorn pyrag.api.main:app --reload
   ```

The API will be available at `http://localhost:8000` with interactive documentation at `http://localhost:8000/docs`.

### Production Deployment

For production deployment, see the [Deployment Guide](docs/deployment.md).

## 📚 Documentation

- [Implementation Guide](docs/implementation_guide.md) - Comprehensive technical documentation
- [Implementation Plan](docs/implementation_plan.md) - Strategic roadmap and execution plan
- [API Reference](docs/api.md) - Complete API documentation
- [MCP Integration](docs/mcp-integration.md) - Guide for integrating with coding assistants

## 🔧 Configuration

PyRAG uses environment variables for configuration. Create a `.env` file in the project root:

```env
# Environment
ENVIRONMENT=development

# Database
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pyrag
DB_USER=pyrag
DB_PASSWORD=pyrag

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8000

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=INFO
LOG_FORMAT=json
```

## 🧪 Testing

Run the test suite:

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=pyrag

# Run specific test categories
pytest -m unit
pytest -m integration
pytest -m slow
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and add tests
4. Run the test suite: `pytest`
5. Format code: `black . && isort .`
6. Commit your changes: `git commit -m 'Add amazing feature'`
7. Push to the branch: `git push origin feature/amazing-feature`
8. Open a Pull Request

### Code Quality

We use several tools to maintain code quality:

- **Black**: Code formatting
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking
- **pytest**: Testing

Run all quality checks:

```bash
pre-commit run --all-files
```

## 📊 Project Status

This project is currently in **Phase 0: Foundation & Architecture**. See our [Implementation Plan](docs/implementation_plan.md) for detailed progress and roadmap.

### Current Phase Goals

- [x] Project structure and architecture setup
- [x] Database schema design
- [x] Configuration management
- [x] Logging infrastructure
- [x] Basic FastAPI application
- [x] Docker development environment
- [ ] Vector store integration
- [ ] Document processing pipeline
- [ ] MCP server implementation

## 🏛️ Architecture Overview

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   MCP Client    │    │   FastAPI API   │    │   Background    │
│   (Cursor)      │◄──►│   Server        │◄──►│   Workers       │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                                │
                                ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   PostgreSQL    │    │   ChromaDB/     │    │   Redis         │
│   + pgvector    │    │   Weaviate      │    │   Cache/Queue   │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## 📈 Roadmap

### Phase 1: Core RAG System (Weeks 2-4)
- Document processing pipeline for 5 initial libraries
- Hierarchical chunking system
- Basic semantic search and retrieval
- Vector store integration

### Phase 2: MCP Integration (Weeks 5-6)
- MCP server implementation
- Cursor integration
- Enhanced retrieval capabilities
- Production deployment preparation

### Phase 3: Scale & Advanced Features (Weeks 9-12)
- Support for 25+ Python libraries
- Automated update pipeline
- Advanced retrieval with GraphRAG
- Performance optimization

### Phase 4: Community & Sustainability (Weeks 13-16)
- Multi-platform integrations
- Community-driven development
- Enterprise features
- Sustainable development model

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- The Python community for creating amazing libraries
- The MCP (Model Context Protocol) team for the integration standard
- All contributors and supporters of this project

## 📞 Support

- **Issues**: [GitHub Issues](https://github.com/pyrag/pyrag/issues)
- **Discussions**: [GitHub Discussions](https://github.com/pyrag/pyrag/discussions)
- **Documentation**: [Project Wiki](https://github.com/pyrag/pyrag/wiki)

---

**PyRAG** - Keeping AI coding assistants current with Python documentation.
