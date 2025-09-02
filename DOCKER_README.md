# Docker Setup for PyRAG

This project uses Docker Compose to run the **complete PyRAG system**.

## 🚀 **Quick Start**

### **Start the Full System**
```bash
# Start all services
docker-compose up -d

# Check status
docker-compose ps

# View logs
docker-compose logs -f
```

### **What Gets Started**
- **PostgreSQL** (port 5432) - Database with pgvector extension
- **Redis** (port 6379) - Caching and task queuing
- **ChromaDB** (port 8001) - Vector storage
- **PyRAG API** (port 8000) - Main API + MCP server
- **Celery Worker** - Background task processing

## 🧹 **Cleanup**
```bash
# Stop and remove containers
docker-compose down

# Remove volumes too
docker-compose down -v
```

## 📍 **Ports**
- **8000**: PyRAG API + MCP Server
- **8001**: ChromaDB
- **5432**: PostgreSQL
- **6379**: Redis

## 🔍 **Testing Your MCP Server**

1. **Start the system:**
   ```bash
   docker-compose up -d
   ```

2. **Check if it's running:**
   ```bash
   docker-compose logs pyrag-api
   ```

3. **Test with MCP Inspector** or your test script

## 🐛 **Troubleshooting**

- **Port conflicts**: Make sure ports 8000, 8001, 5432, 6379 are free
- **Build issues**: Run `docker-compose build` first
- **Memory issues**: Ensure you have enough RAM for all services

## 🔧 **Development**

```bash
# Rebuild and restart
docker-compose up -d --build

# View specific service logs
docker-compose logs -f pyrag-api
docker-compose logs -f postgres
```
