#!/bin/bash

# 🚀 PyRAG MCP Server Startup Script
# This script starts your PyRAG system with Docker

echo "🐍 Starting PyRAG MCP Server..."
echo "=================================="

# Check if Docker is running
if ! docker info > /dev/null 2>&1; then
    echo "❌ Docker is not running. Please start Docker Desktop first."
    exit 1
fi

# Navigate to project directory
cd "$(dirname "$0")"

echo "📁 Project directory: $(pwd)"

# Check if docker-compose.yml exists
if [ ! -f "docker-compose.yml" ]; then
    echo "❌ docker-compose.yml not found. Please run this script from the PyRag project root."
    exit 1
fi

echo "🔧 Starting PyRAG services..."
echo "   - PostgreSQL with pgvector"
echo "   - Redis for caching"
echo "   - ChromaDB for vector storage"
echo "   - PyRAG API (includes MCP server with HTTPS)"
echo "   - Celery Worker for background tasks"

# Generate SSL certificates if they don't exist
if [ ! -f "certs/mcp_server.crt" ] || [ ! -f "certs/mcp_server.key" ]; then
    echo "🔐 Generating SSL certificates for MCP server..."
    python tools/generate_ssl_certs.py
    if [ $? -ne 0 ]; then
        echo "❌ Failed to generate SSL certificates. Please check OpenSSL installation."
        exit 1
    fi
fi

# Start services
docker-compose up -d

# Wait a moment for services to start
echo "⏳ Waiting for services to start..."
sleep 5

# Check service status
echo "📊 Checking service status..."
docker-compose ps

echo ""
echo "🎯 PyRAG MCP Server is starting up!"
echo ""
echo "📋 Access Points:"
echo "   - FastAPI: http://localhost:8000"
echo "   - Health Check: http://localhost:8000/health"
echo "   - MCP Server: https://localhost:8002 (HTTPS)"
echo ""
echo "📖 View logs: docker-compose logs -f pyrag-api"
echo "🛑 Stop services: docker-compose down"
echo ""
echo "⏱️  Initialization takes about 30 seconds for ML models to load..."
echo "✅ Your MCP server will be ready when you see 'PyRAG system initialized and ready'"
echo ""
echo "🚀 Ready to connect with Cursor/Claude!"
echo ""
echo "🔐 IMPORTANT: Use https://localhost:8002 in Claude's MCP connector"
echo "   (Claude requires HTTPS for security reasons)"
