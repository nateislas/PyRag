"""Configuration management for PyRAG."""

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic_settings import BaseSettings
from pydantic import Field, validator


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    host: str = Field(default="localhost", env="DB_HOST")
    port: int = Field(default=5432, env="DB_PORT")
    database: str = Field(default="pyrag", env="DB_NAME")
    username: str = Field(default="pyrag", env="DB_USER")
    password: str = Field(default="pyrag", env="DB_PASSWORD")
    pool_size: int = Field(default=10, env="DB_POOL_SIZE")
    max_overflow: int = Field(default=20, env="DB_MAX_OVERFLOW")
    echo: bool = Field(default=False, env="DB_ECHO")

    @property
    def url(self) -> str:
        """Get database URL."""
        return f"postgresql://{self.username}:{self.password}@{self.host}:{self.port}/{self.database}"

    class Config:
        env_prefix = "DB_"
        extra = "ignore"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings."""

    # ChromaDB settings
    path: str = Field(default="./chroma_db", env="CHROMA_PATH")
    host: str = Field(default="localhost", env="CHROMA_HOST")
    port: int = Field(default=8000, env="CHROMA_PORT")
    
    # Weaviate settings (for production)
    weaviate_url: Optional[str] = Field(default=None, env="WEAVIATE_URL")
    weaviate_api_key: Optional[str] = Field(default=None, env="WEAVIATE_API_KEY")
    
    # Vector store selection
    type: str = Field(default="chroma", env="VECTOR_STORE_TYPE")
    
    @validator("type")
    def validate_vector_store_type(cls, v: str) -> str:
        """Validate vector store type."""
        if v not in ["chroma", "weaviate"]:
            raise ValueError("vector_store_type must be 'chroma' or 'weaviate'")
        return v

    class Config:
        env_prefix = "VECTOR_"
        extra = "ignore"


class EmbeddingSettings(BaseSettings):
    """Embedding configuration settings."""

    model_name: str = Field(default="sentence-transformers/all-MiniLM-L6-v2", env="EMBEDDING_MODEL")
    dimension: int = Field(default=384, env="EMBEDDING_DIMENSION")
    batch_size: int = Field(default=32, env="EMBEDDING_BATCH_SIZE")
    device: str = Field(default="cpu", env="EMBEDDING_DEVICE")
    
    # OpenAI settings (for production)
    openai_api_key: Optional[str] = Field(default=None, env="OPENAI_API_KEY")
    openai_model: str = Field(default="text-embedding-3-large", env="OPENAI_EMBEDDING_MODEL")
    
    @validator("device")
    def validate_device(cls, v: str) -> str:
        """Validate device."""
        if v not in ["cpu", "cuda", "mps"]:
            raise ValueError("device must be 'cpu', 'cuda', or 'mps'")
        return v

    class Config:
        env_prefix = "EMBEDDING_"
        extra = "ignore"


class RedisSettings(BaseSettings):
    """Redis configuration settings."""

    host: str = Field(default="localhost", env="REDIS_HOST")
    port: int = Field(default=6379, env="REDIS_PORT")
    password: Optional[str] = Field(default=None, env="REDIS_PASSWORD")
    db: int = Field(default=0, env="REDIS_DB")
    decode_responses: bool = Field(default=True, env="REDIS_DECODE_RESPONSES")

    @property
    def url(self) -> str:
        """Get Redis URL."""
        if self.password:
            return f"redis://:{self.password}@{self.host}:{self.port}/{self.db}"
        return f"redis://{self.host}:{self.port}/{self.db}"

    class Config:
        env_prefix = "REDIS_"
        extra = "ignore"


class APISettings(BaseSettings):
    """API configuration settings."""

    host: str = Field(default="0.0.0.0", env="API_HOST")
    port: int = Field(default=8000, env="API_PORT")
    debug: bool = Field(default=False, env="DEBUG")
    secret_key: str = Field(default="your-secret-key-change-in-production", env="SECRET_KEY")
    cors_origins: List[str] = Field(default=["http://localhost:3000"], env="CORS_ORIGINS")
    rate_limit_per_minute: int = Field(default=100, env="RATE_LIMIT_PER_MINUTE")

    class Config:
        env_prefix = "API_"
        extra = "ignore"


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    level: str = Field(default="INFO", env="LOG_LEVEL")
    format: str = Field(default="json", env="LOG_FORMAT")
    include_timestamp: bool = Field(default=True, env="LOG_INCLUDE_TIMESTAMP")
    include_correlation_id: bool = Field(default=True, env="LOG_INCLUDE_CORRELATION_ID")

    @validator("level")
    def validate_log_level(cls, v: str) -> str:
        """Validate log level."""
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"log_level must be one of {valid_levels}")
        return v.upper()

    class Config:
        env_prefix = "LOG_"
        extra = "ignore"


class MCPSettings(BaseSettings):
    """MCP server configuration settings."""

    server_name: str = Field(default="pyrag", env="MCP_SERVER_NAME")
    server_version: str = Field(default="0.1.0", env="MCP_SERVER_VERSION")
    server_description: str = Field(default="Python Documentation RAG System", env="MCP_SERVER_DESCRIPTION")
    max_results: int = Field(default=10, env="MCP_MAX_RESULTS")
    cache_ttl: int = Field(default=3600, env="MCP_CACHE_TTL")

    class Config:
        env_prefix = "MCP_"
        extra = "ignore"


class Settings(BaseSettings):
    """Main application settings."""

    # Environment
    environment: str = Field(default="development", env="ENVIRONMENT")
    base_dir: Path = Field(default=Path(__file__).parent.parent.parent, env="BASE_DIR")
    
    # Component settings
    database: DatabaseSettings = DatabaseSettings()
    vector_store: VectorStoreSettings = VectorStoreSettings()
    embeddings: EmbeddingSettings = EmbeddingSettings()
    redis: RedisSettings = RedisSettings()
    api: APISettings = APISettings()
    logging: LoggingSettings = LoggingSettings()
    mcp: MCPSettings = MCPSettings()
    
    # Feature flags
    enable_caching: bool = Field(default=True, env="ENABLE_CACHING")
    enable_analytics: bool = Field(default=True, env="ENABLE_ANALYTICS")
    enable_rate_limiting: bool = Field(default=True, env="ENABLE_RATE_LIMITING")
    
    @validator("environment")
    def validate_environment(cls, v: str) -> str:
        """Validate environment."""
        if v not in ["development", "testing", "staging", "production"]:
            raise ValueError("environment must be one of: development, testing, staging, production")
        return v

    @property
    def is_development(self) -> bool:
        """Check if running in development mode."""
        return self.environment == "development"

    @property
    def is_production(self) -> bool:
        """Check if running in production mode."""
        return self.environment == "production"

    @property
    def is_testing(self) -> bool:
        """Check if running in testing mode."""
        return self.environment == "testing"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        extra = "ignore"


# Global settings instance
settings = Settings()
