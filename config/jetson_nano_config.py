"""Jetson Nano specific configuration for PyRAG."""

import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class JetsonNanoConfig:
    """Configuration optimized for Jetson Nano hardware."""

    # Hardware-specific settings
    device: str = "cpu"  # Use CPU for better compatibility
    batch_size: int = 4  # Smaller batch size for limited memory
    max_length: int = 512  # Reduced max length for memory efficiency

    # Performance tuning
    normalize_embeddings: bool = True
    use_half_precision: bool = False  # Jetson Nano doesn't support FP16 well

    # Memory management
    max_memory_usage: str = "2GB"  # Conservative memory limit
    enable_memory_optimization: bool = True

    # Network settings
    mcp_host: str = "0.0.0.0"  # Bind to all interfaces
    mcp_port: int = 8000

    # Service settings
    enable_health_check: bool = True
    health_check_interval: int = 30

    # Logging
    log_level: str = "INFO"
    log_format: str = "json"

    # Database settings (optimized for Jetson Nano)
    db_pool_size: int = 5  # Smaller connection pool
    db_max_overflow: int = 10

    # Vector store settings
    chroma_persist_directory: str = "/opt/pyrag/data/chroma"
    enable_persistence: bool = True

    # Cache settings
    redis_max_connections: int = 10
    cache_ttl: int = 3600  # 1 hour default

    @classmethod
    def from_env(cls) -> "JetsonNanoConfig":
        """Create configuration from environment variables."""
        return cls(
            device=os.getenv("JETSON_DEVICE", "cpu"),
            batch_size=int(os.getenv("JETSON_BATCH_SIZE", "4")),
            max_length=int(os.getenv("JETSON_MAX_LENGTH", "512")),
            normalize_embeddings=os.getenv("JETSON_NORMALIZE", "true").lower()
            == "true",
            use_half_precision=os.getenv("JETSON_HALF_PRECISION", "false").lower()
            == "true",
            max_memory_usage=os.getenv("JETSON_MAX_MEMORY", "2GB"),
            enable_memory_optimization=os.getenv("JETSON_MEMORY_OPT", "true").lower()
            == "true",
            mcp_host=os.getenv("MCP_HOST", "0.0.0.0"),
            mcp_port=int(os.getenv("MCP_PORT", "8000")),
            enable_health_check=os.getenv("JETSON_HEALTH_CHECK", "true").lower()
            == "true",
            health_check_interval=int(os.getenv("JETSON_HEALTH_INTERVAL", "30")),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            log_format=os.getenv("LOG_FORMAT", "json"),
            db_pool_size=int(os.getenv("JETSON_DB_POOL_SIZE", "5")),
            db_max_overflow=int(os.getenv("JETSON_DB_MAX_OVERFLOW", "10")),
            chroma_persist_directory=os.getenv(
                "JETSON_CHROMA_DIR", "/opt/pyrag/data/chroma"
            ),
            enable_persistence=os.getenv("JETSON_ENABLE_PERSISTENCE", "true").lower()
            == "true",
            redis_max_connections=int(os.getenv("JETSON_REDIS_MAX_CONN", "10")),
            cache_ttl=int(os.getenv("JETSON_CACHE_TTL", "3600")),
        )

    def to_dict(self) -> dict:
        """Convert configuration to dictionary."""
        return {
            "device": self.device,
            "batch_size": self.batch_size,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "use_half_precision": self.use_half_precision,
            "max_memory_usage": self.max_memory_usage,
            "enable_memory_optimization": self.enable_memory_optimization,
            "mcp_host": self.mcp_host,
            "mcp_port": self.mcp_port,
            "enable_health_check": self.enable_health_check,
            "health_check_interval": self.health_check_interval,
            "log_level": self.log_level,
            "log_format": self.log_format,
            "db_pool_size": self.db_pool_size,
            "db_max_overflow": self.db_max_overflow,
            "chroma_persist_directory": self.chroma_persist_directory,
            "enable_persistence": self.enable_persistence,
            "redis_max_connections": self.redis_max_connections,
            "cache_ttl": self.cache_ttl,
        }


# Default Jetson Nano configuration
jetson_config = JetsonNanoConfig.from_env()
