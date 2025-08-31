"""Library discovery system for finding popular Python libraries."""

import asyncio
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
from ..logging import get_logger

logger = get_logger(__name__)


@dataclass
class LibraryInfo:
    """Information about a Python library."""
    name: str
    description: str
    version: str
    download_count: int
    github_stars: int
    github_forks: int
    last_updated: str
    license: str
    documentation_url: Optional[str]
    repository_url: Optional[str]
    quality_score: float
    popularity_score: float


class LibraryDiscovery:
    """Discover popular Python libraries for indexing."""
    
    def __init__(self):
        """Initialize the library discovery system."""
        self.logger = get_logger(__name__)
        
        # PyPI API endpoints
        self.pypi_base_url = "https://pypi.org/pypi"
        self.pypi_stats_url = "https://pypi.org/stats"
        
        # GitHub API endpoints
        self.github_api_url = "https://api.github.com"
        
        # Popular library categories
        self.categories = {
            "web_frameworks": ["fastapi", "django", "flask", "tornado", "aiohttp"],
            "data_science": ["pandas", "numpy", "scipy", "matplotlib", "seaborn"],
            "http_clients": ["requests", "httpx", "aiohttp", "urllib3"],
            "databases": ["sqlalchemy", "psycopg2", "pymongo", "redis"],
            "testing": ["pytest", "unittest", "nose", "coverage"],
            "machine_learning": ["scikit-learn", "tensorflow", "pytorch", "keras"],
            "async": ["asyncio", "aiofiles", "aiohttp", "asyncpg"],
            "validation": ["pydantic", "marshmallow", "cerberus"],
        }
    
    async def discover_popular_libraries(self, limit: int = 50) -> List[LibraryInfo]:
        """Discover popular Python libraries for indexing."""
        try:
            self.logger.info(f"Discovering popular libraries (limit: {limit})")
            
            # Get PyPI download statistics
            pypi_libraries = await self._get_pypi_popular_libraries(limit)
            
            # Get GitHub activity data
            github_libraries = await self._get_github_active_libraries(limit)
            
            # Combine and rank libraries
            combined_libraries = self._combine_library_data(pypi_libraries, github_libraries)
            
            # Filter and rank by quality
            ranked_libraries = await self._rank_libraries_by_quality(combined_libraries)
            
            # Take top libraries
            top_libraries = ranked_libraries[:limit]
            
            self.logger.info(f"Discovered {len(top_libraries)} popular libraries")
            return top_libraries
            
        except Exception as e:
            self.logger.error(f"Failed to discover libraries: {e}")
            return []
    
    async def _get_pypi_popular_libraries(self, limit: int) -> List[Dict[str, Any]]:
        """Get popular libraries from PyPI download statistics."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get PyPI download statistics
                async with session.get(f"{self.pypi_stats_url}/packages") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        # Extract top packages by download count
                        packages = []
                        for package in data.get("packages", [])[:limit * 2]:  # Get more to filter
                            package_info = {
                                "name": package.get("package_name"),
                                "download_count": package.get("download_count", 0),
                                "source": "pypi"
                            }
                            packages.append(package_info)
                        
                        return packages
                    else:
                        self.logger.warning(f"Failed to get PyPI stats: {response.status}")
                        return []
                        
        except Exception as e:
            self.logger.error(f"Error fetching PyPI data: {e}")
            return []
    
    async def _get_github_active_libraries(self, limit: int) -> List[Dict[str, Any]]:
        """Get active libraries from GitHub."""
        try:
            # For now, return a curated list of popular libraries
            # In production, this would use GitHub API to find trending Python repos
            
            popular_libraries = [
                {"name": "requests", "github_stars": 50000, "source": "github"},
                {"name": "pandas", "github_stars": 38000, "source": "github"},
                {"name": "numpy", "github_stars": 25000, "source": "github"},
                {"name": "fastapi", "github_stars": 70000, "source": "github"},
                {"name": "django", "github_stars": 72000, "source": "github"},
                {"name": "flask", "github_stars": 65000, "source": "github"},
                {"name": "sqlalchemy", "github_stars": 7000, "source": "github"},
                {"name": "pydantic", "github_stars": 15000, "source": "github"},
                {"name": "pytest", "github_stars": 10000, "source": "github"},
                {"name": "scikit-learn", "github_stars": 55000, "source": "github"},
            ]
            
            return popular_libraries[:limit]
            
        except Exception as e:
            self.logger.error(f"Error fetching GitHub data: {e}")
            return []
    
    def _combine_library_data(self, pypi_libraries: List[Dict], github_libraries: List[Dict]) -> List[Dict[str, Any]]:
        """Combine PyPI and GitHub data for libraries."""
        combined = {}
        
        # Add PyPI data
        for lib in pypi_libraries:
            name = lib.get("name")
            if name:
                combined[name] = {
                    "name": name,
                    "download_count": lib.get("download_count", 0),
                    "github_stars": 0,
                    "source": "pypi"
                }
        
        # Add GitHub data
        for lib in github_libraries:
            name = lib.get("name")
            if name:
                if name in combined:
                    combined[name]["github_stars"] = lib.get("github_stars", 0)
                    combined[name]["source"] = "both"
                else:
                    combined[name] = {
                        "name": name,
                        "download_count": 0,
                        "github_stars": lib.get("github_stars", 0),
                        "source": "github"
                    }
        
        return list(combined.values())
    
    async def _rank_libraries_by_quality(self, libraries: List[Dict[str, Any]]) -> List[LibraryInfo]:
        """Rank libraries by quality and popularity."""
        ranked_libraries = []
        
        for lib in libraries:
            try:
                # Get detailed information for each library
                library_info = await self._get_library_details(lib["name"])
                
                if library_info:
                    # Calculate quality and popularity scores
                    quality_score = self._calculate_quality_score(library_info)
                    popularity_score = self._calculate_popularity_score(lib)
                    
                    library_info.quality_score = quality_score
                    library_info.popularity_score = popularity_score
                    
                    ranked_libraries.append(library_info)
                    
            except Exception as e:
                self.logger.warning(f"Failed to get details for {lib['name']}: {e}")
                continue
        
        # Sort by combined score (quality + popularity)
        ranked_libraries.sort(
            key=lambda x: x.quality_score + x.popularity_score,
            reverse=True
        )
        
        return ranked_libraries
    
    async def _get_library_details(self, library_name: str) -> Optional[LibraryInfo]:
        """Get detailed information for a library."""
        try:
            async with aiohttp.ClientSession() as session:
                # Get PyPI package info
                async with session.get(f"{self.pypi_base_url}/{library_name}/json") as response:
                    if response.status == 200:
                        data = await response.json()
                        
                        info = data.get("info", {})
                        releases = data.get("releases", {})
                        
                        # Get latest version
                        latest_version = info.get("version", "unknown")
                        
                        # Get download count (would need separate API call in production)
                        download_count = 1000000  # Mock value
                        
                        # Get GitHub info (would need separate API call in production)
                        github_stars = 10000  # Mock value
                        github_forks = 1000   # Mock value
                        
                        return LibraryInfo(
                            name=library_name,
                            description=info.get("summary", ""),
                            version=latest_version,
                            download_count=download_count,
                            github_stars=github_stars,
                            github_forks=github_forks,
                            last_updated=info.get("upload_time", ""),
                            license=info.get("license", ""),
                            documentation_url=info.get("project_url", ""),
                            repository_url=info.get("project_urls", {}).get("Repository", ""),
                            quality_score=0.0,  # Will be calculated later
                            popularity_score=0.0  # Will be calculated later
                        )
                    else:
                        self.logger.warning(f"Failed to get PyPI info for {library_name}: {response.status}")
                        return None
                        
        except Exception as e:
            self.logger.error(f"Error getting library details for {library_name}: {e}")
            return None
    
    def _calculate_quality_score(self, library_info: LibraryInfo) -> float:
        """Calculate quality score for a library."""
        score = 0.0
        
        # Documentation quality
        if library_info.documentation_url:
            score += 0.3
        
        # Repository availability
        if library_info.repository_url:
            score += 0.2
        
        # License information
        if library_info.license:
            score += 0.1
        
        # Description quality
        if len(library_info.description) > 50:
            score += 0.2
        
        # Version stability (simple heuristic)
        if library_info.version and library_info.version != "unknown":
            score += 0.2
        
        return min(score, 1.0)
    
    def _calculate_popularity_score(self, library_data: Dict[str, Any]) -> float:
        """Calculate popularity score for a library."""
        score = 0.0
        
        # Download count (normalized)
        download_count = library_data.get("download_count", 0)
        if download_count > 1000000:
            score += 0.5
        elif download_count > 100000:
            score += 0.3
        elif download_count > 10000:
            score += 0.1
        
        # GitHub stars (normalized)
        github_stars = library_data.get("github_stars", 0)
        if github_stars > 10000:
            score += 0.5
        elif github_stars > 1000:
            score += 0.3
        elif github_stars > 100:
            score += 0.1
        
        return min(score, 1.0)
    
    async def get_library_suggestions(self, category: Optional[str] = None) -> List[str]:
        """Get library suggestions for a specific category."""
        if category and category in self.categories:
            return self.categories[category]
        
        # Return all categories flattened
        all_libraries = []
        for libraries in self.categories.values():
            all_libraries.extend(libraries)
        
        return all_libraries
    
    async def check_library_availability(self, library_name: str) -> Dict[str, Any]:
        """Check if a library is available and has good documentation."""
        try:
            library_info = await self._get_library_details(library_name)
            
            if not library_info:
                return {
                    "available": False,
                    "reason": "Library not found on PyPI"
                }
            
            # Check documentation availability
            has_documentation = bool(library_info.documentation_url)
            has_repository = bool(library_info.repository_url)
            
            return {
                "available": True,
                "has_documentation": has_documentation,
                "has_repository": has_repository,
                "quality_score": library_info.quality_score,
                "recommended": library_info.quality_score > 0.5
            }
            
        except Exception as e:
            self.logger.error(f"Error checking library availability for {library_name}: {e}")
            return {
                "available": False,
                "reason": str(e)
            }
