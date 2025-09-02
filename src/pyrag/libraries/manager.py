"""Library manager for automated discovery and management."""

from typing import Any, Dict, List, Optional

from ..logging import get_logger
from .discovery import LibraryDiscovery, LibraryInfo

logger = get_logger(__name__)


class LibraryManager:
    """Automated library discovery and management."""

    def __init__(self):
        """Initialize the library manager."""
        self.logger = get_logger(__name__)
        self.discovery = LibraryDiscovery()

        # Track managed libraries
        self.managed_libraries: Dict[str, LibraryInfo] = {}

    async def discover_popular_libraries(self, limit: int = 50) -> List[LibraryInfo]:
        """Discover popular Python libraries for indexing."""
        try:
            self.logger.info(f"Discovering popular libraries (limit: {limit})")

            libraries = await self.discovery.discover_popular_libraries(limit)

            # Store discovered libraries
            for library in libraries:
                self.managed_libraries[library.name] = library

            self.logger.info(f"Discovered {len(libraries)} popular libraries")
            return libraries

        except Exception as e:
            self.logger.error(f"Failed to discover libraries: {e}")
            return []

    async def add_library_automated(self, library_name: str) -> bool:
        """Automatically add a library to the system."""
        try:
            self.logger.info(f"Automatically adding library: {library_name}")

            # Check if library is available
            availability = await self.discovery.check_library_availability(library_name)

            if not availability["available"]:
                self.logger.warning(
                    f"Library {library_name} is not available: {availability['reason']}"
                )
                return False

            # Get library details
            library_info = await self.discovery._get_library_details(library_name)

            if not library_info:
                self.logger.error(f"Failed to get details for library {library_name}")
                return False

            # Store library info
            self.managed_libraries[library_name] = library_info

            self.logger.info(f"Successfully added library {library_name}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to add library {library_name}: {e}")
            return False

    async def get_managed_libraries(self) -> List[LibraryInfo]:
        """Get list of currently managed libraries."""
        return list(self.managed_libraries.values())

    async def get_library_info(self, library_name: str) -> Optional[LibraryInfo]:
        """Get information about a specific library."""
        return self.managed_libraries.get(library_name)

    async def remove_library(self, library_name: str) -> bool:
        """Remove a library from management."""
        try:
            if library_name in self.managed_libraries:
                del self.managed_libraries[library_name]
                self.logger.info(f"Removed library {library_name} from management")
                return True
            else:
                self.logger.warning(
                    f"Library {library_name} not found in managed libraries"
                )
                return False

        except Exception as e:
            self.logger.error(f"Failed to remove library {library_name}: {e}")
            return False

    async def update_library_info(self, library_name: str) -> bool:
        """Update information for a specific library."""
        try:
            self.logger.info(f"Updating library info for {library_name}")

            # Get updated library details
            library_info = await self.discovery._get_library_details(library_name)

            if library_info:
                self.managed_libraries[library_name] = library_info
                self.logger.info(
                    f"Successfully updated library info for {library_name}"
                )
                return True
            else:
                self.logger.error(
                    f"Failed to get updated info for library {library_name}"
                )
                return False

        except Exception as e:
            self.logger.error(f"Failed to update library info for {library_name}: {e}")
            return False

    async def get_library_suggestions(
        self, category: Optional[str] = None
    ) -> List[str]:
        """Get library suggestions for a specific category."""
        return await self.discovery.get_library_suggestions(category)

    async def check_library_availability(self, library_name: str) -> Dict[str, Any]:
        """Check if a library is available and has good documentation."""
        return await self.discovery.check_library_availability(library_name)

    async def get_library_stats(self) -> Dict[str, Any]:
        """Get statistics about managed libraries."""
        total_libraries = len(self.managed_libraries)

        # Calculate average quality and popularity scores
        if total_libraries > 0:
            avg_quality = (
                sum(lib.quality_score for lib in self.managed_libraries.values())
                / total_libraries
            )
            avg_popularity = (
                sum(lib.popularity_score for lib in self.managed_libraries.values())
                / total_libraries
            )
        else:
            avg_quality = 0.0
            avg_popularity = 0.0

        # Count libraries by quality tier
        high_quality = sum(
            1 for lib in self.managed_libraries.values() if lib.quality_score > 0.7
        )
        medium_quality = sum(
            1
            for lib in self.managed_libraries.values()
            if 0.4 <= lib.quality_score <= 0.7
        )
        low_quality = sum(
            1 for lib in self.managed_libraries.values() if lib.quality_score < 0.4
        )

        return {
            "total_libraries": total_libraries,
            "average_quality_score": avg_quality,
            "average_popularity_score": avg_popularity,
            "high_quality_libraries": high_quality,
            "medium_quality_libraries": medium_quality,
            "low_quality_libraries": low_quality,
            "libraries": [lib.name for lib in self.managed_libraries.values()],
        }
