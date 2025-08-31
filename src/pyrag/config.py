"""Configuration management for PyRAG."""

import os
from typing import Optional
from dataclasses import dataclass
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

@dataclass
class LLMConfig:
    """LLM configuration."""
    api_key: str
    base_url: str = "https://api.llama.com/compat/v1/"
    model: str = "Llama-3.3-70B-Instruct"
    max_tokens: int = 4000
    temperature: float = 0.1

@dataclass
class FirecrawlConfig:
    """Firecrawl configuration."""
    api_key: str
    base_url: str = "https://api.firecrawl.dev"

@dataclass
class VectorStoreConfig:
    """Vector store configuration."""
    db_path: str = "./chroma_db"
    collection_name: str = "documents"

@dataclass
class PyRAGConfig:
    """Main PyRAG configuration."""
    llm: LLMConfig
    firecrawl: FirecrawlConfig
    vector_store: VectorStoreConfig
    log_level: str = "INFO"

def get_config() -> PyRAGConfig:
    """Get PyRAG configuration from environment variables."""
    
    # LLM Configuration
    llm_config = LLMConfig(
        api_key=os.getenv("LLAMA_API_KEY", ""),
        base_url=os.getenv("LLAMA_BASE_URL", "https://api.llama.com/compat/v1/"),
        model=os.getenv("LLAMA_MODEL", "Llama-3.3-70B-Instruct"),
        max_tokens=int(os.getenv("LLAMA_MAX_TOKENS", "4000")),
        temperature=float(os.getenv("LLAMA_TEMPERATURE", "0.1"))
    )
    
    # Firecrawl Configuration
    firecrawl_config = FirecrawlConfig(
        api_key=os.getenv("FIRECRAWL_API_KEY", ""),
        base_url=os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev")
    )
    
    # Vector Store Configuration
    vector_store_config = VectorStoreConfig(
        db_path=os.getenv("CHROMA_DB_PATH", "./chroma_db"),
        collection_name=os.getenv("CHROMA_COLLECTION", "documents")
    )
    
    return PyRAGConfig(
        llm=llm_config,
        firecrawl=firecrawl_config,
        vector_store=vector_store_config,
        log_level=os.getenv("LOG_LEVEL", "INFO")
    )

def validate_config(config: PyRAGConfig) -> bool:
    """Validate that required configuration is present."""
    errors = []
    
    if not config.llm.api_key:
        errors.append("LLAMA_API_KEY is required")
    
    if not config.firecrawl.api_key:
        errors.append("FIRECRAWL_API_KEY is required")
    
    if errors:
        print("❌ Configuration errors:")
        for error in errors:
            print(f"   - {error}")
        print("\n📝 Please set the required environment variables:")
        print("   export LLAMA_API_KEY=your_key_here")
        print("   export FIRECRAWL_API_KEY=your_key_here")
        print("\n   Or create a .env file with these variables.")
        return False
    
    return True
