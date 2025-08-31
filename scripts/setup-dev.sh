#!/bin/bash

# PyRAG Development Setup Script
set -e

echo "🚀 Setting up PyRAG development environment..."

# Check if Python 3.11+ is installed
python_version=$(python3 --version 2>&1 | cut -d' ' -f2 | cut -d'.' -f1,2)
required_version="3.11"

if [ "$(printf '%s\n' "$required_version" "$python_version" | sort -V | head -n1)" != "$required_version" ]; then
    echo "❌ Python 3.11+ is required. Found: $python_version"
    echo "Please install Python 3.11+ and try again."
    exit 1
fi

echo "✅ Python version: $python_version"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "📦 Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install dependencies
echo "📚 Installing dependencies..."
pip install -e .[dev]

# Create .env file if it doesn't exist
if [ ! -f ".env" ]; then
    echo "⚙️  Creating .env file..."
    cat > .env << EOF
# Environment
ENVIRONMENT=development

# Database (using SQLite for local development)
DB_HOST=localhost
DB_PORT=5432
DB_NAME=pyrag
DB_USER=pyrag
DB_PASSWORD=pyrag

# Vector Store
VECTOR_STORE_TYPE=chroma
CHROMA_HOST=localhost
CHROMA_PORT=8001

# API
API_HOST=0.0.0.0
API_PORT=8000
DEBUG=true

# Logging
LOG_LEVEL=DEBUG
LOG_FORMAT=json

# Redis (optional for local development)
REDIS_HOST=localhost
REDIS_PORT=6379
EOF
    echo "✅ Created .env file"
fi

# Install pre-commit hooks
echo "🔧 Installing pre-commit hooks..."
pre-commit install

echo "✅ Development environment setup complete!"
echo ""
echo "Next steps:"
echo "1. Start the API server: uvicorn pyrag.api.main:app --reload"
echo "2. Run tests: pytest"
echo "3. Format code: black . && isort ."
echo ""
echo "Note: For full functionality, you'll need PostgreSQL, Redis, and ChromaDB running."
echo "You can start them with Docker when ready: docker-compose up -d"
