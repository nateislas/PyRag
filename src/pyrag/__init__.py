"""PyRAG: Python Documentation RAG System for AI Coding Assistants."""

__version__ = "0.1.0"
__author__ = "PyRAG Contributors"
__email__ = "contributors@pyrag.dev"

from .config import get_config, validate_config
from .core import PyRAG

__all__ = ["PyRAG", "get_config", "validate_config"]
