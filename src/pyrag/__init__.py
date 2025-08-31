"""PyRAG: Python Documentation RAG System for AI Coding Assistants."""

__version__ = "0.1.0"
__author__ = "PyRAG Contributors"
__email__ = "contributors@pyrag.dev"

from .core import PyRAG
from .config import get_config, validate_config

__all__ = ["PyRAG", "get_config", "validate_config"]
