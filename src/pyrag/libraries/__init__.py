"""Library management module for automated discovery and updates."""

from .discovery import LibraryDiscovery, LibraryInfo
from .manager import LibraryManager

__all__ = [
    "LibraryDiscovery",
    "LibraryInfo",
    "LibraryManager",
]
