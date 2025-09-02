"""Caching layer for PyRAG to improve performance."""

import asyncio
import hashlib
import json
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union

from .logging import get_logger

logger = get_logger(__name__)


class CacheManager:
    """Cache manager for PyRAG with Redis and in-memory fallback."""

    def __init__(self, redis_client=None, default_ttl: int = 3600):
        """Initialize cache manager.

        Args:
            redis_client: Redis client instance (optional)
            default_ttl: Default TTL in seconds for cached items
        """
        self.logger = get_logger(__name__)
        self.redis_client = redis_client
        self.default_ttl = default_ttl

        # In-memory cache as fallback
        self.memory_cache: Dict[str, Dict[str, Any]] = {}

        # Cache statistics
        self.stats = {
            "hits": 0,
            "misses": 0,
            "sets": 0,
            "deletes": 0,
        }

    def _generate_key(self, prefix: str, *args, **kwargs) -> str:
        """Generate a cache key from prefix and arguments."""
        # Create a string representation of arguments
        key_parts = [prefix]

        # Add positional arguments
        for arg in args:
            key_parts.append(str(arg))

        # Add keyword arguments (sorted for consistency)
        for key, value in sorted(kwargs.items()):
            if value is not None:  # Skip None values
                key_parts.append(f"{key}:{value}")

        # Join and hash
        key_string = "|".join(key_parts)
        return f"pyrag:{hashlib.md5(key_string.encode()).hexdigest()}"

    async def get(self, prefix: str, *args, **kwargs) -> Optional[Any]:
        """Get a value from cache.

        Args:
            prefix: Cache key prefix
            *args: Positional arguments for key generation
            **kwargs: Keyword arguments for key generation

        Returns:
            Cached value or None if not found
        """
        key = self._generate_key(prefix, *args, **kwargs)

        try:
            # Try Redis first
            if self.redis_client:
                cached_value = await self.redis_client.get(key)
                if cached_value:
                    self.stats["hits"] += 1
                    self.logger.debug(f"Cache hit (Redis): {key}")
                    return json.loads(cached_value)

            # Fallback to memory cache
            if key in self.memory_cache:
                cache_entry = self.memory_cache[key]
                if cache_entry["expires_at"] > datetime.now():
                    self.stats["hits"] += 1
                    self.logger.debug(f"Cache hit (memory): {key}")
                    return cache_entry["value"]
                else:
                    # Expired entry, remove it
                    del self.memory_cache[key]

            self.stats["misses"] += 1
            self.logger.debug(f"Cache miss: {key}")
            return None

        except Exception as e:
            self.logger.error(f"Error getting from cache: {e}")
            return None

    async def set(
        self, prefix: str, value: Any, ttl: Optional[int] = None, *args, **kwargs
    ) -> bool:
        """Set a value in cache.

        Args:
            prefix: Cache key prefix
            value: Value to cache
            ttl: TTL in seconds (uses default if None)
            *args: Positional arguments for key generation
            **kwargs: Keyword arguments for key generation

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(prefix, *args, **kwargs)
        ttl = ttl or self.default_ttl

        try:
            # Try Redis first
            if self.redis_client:
                await self.redis_client.setex(key, ttl, json.dumps(value))
                self.stats["sets"] += 1
                self.logger.debug(f"Cache set (Redis): {key}, TTL: {ttl}s")
                return True

            # Fallback to memory cache
            expires_at = datetime.now() + timedelta(seconds=ttl)
            self.memory_cache[key] = {
                "value": value,
                "expires_at": expires_at,
            }
            self.stats["sets"] += 1
            self.logger.debug(f"Cache set (memory): {key}, TTL: {ttl}s")
            return True

        except Exception as e:
            self.logger.error(f"Error setting cache: {e}")
            return False

    async def delete(self, prefix: str, *args, **kwargs) -> bool:
        """Delete a value from cache.

        Args:
            prefix: Cache key prefix
            *args: Positional arguments for key generation
            **kwargs: Keyword arguments for key generation

        Returns:
            True if successful, False otherwise
        """
        key = self._generate_key(prefix, *args, **kwargs)

        try:
            # Try Redis first
            if self.redis_client:
                await self.redis_client.delete(key)

            # Also remove from memory cache
            if key in self.memory_cache:
                del self.memory_cache[key]

            self.stats["deletes"] += 1
            self.logger.debug(f"Cache delete: {key}")
            return True

        except Exception as e:
            self.logger.error(f"Error deleting from cache: {e}")
            return False

    async def clear(self, pattern: Optional[str] = None) -> bool:
        """Clear cache entries.

        Args:
            pattern: Pattern to match keys (optional)

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clear Redis cache
            if self.redis_client:
                if pattern:
                    # Get keys matching pattern
                    keys = await self.redis_client.keys(f"pyrag:{pattern}")
                    if keys:
                        await self.redis_client.delete(*keys)
                else:
                    # Clear all pyrag keys
                    keys = await self.redis_client.keys("pyrag:*")
                    if keys:
                        await self.redis_client.delete(*keys)

            # Clear memory cache
            if pattern:
                # Remove keys matching pattern
                keys_to_remove = [
                    key for key in self.memory_cache.keys() if pattern in key
                ]
                for key in keys_to_remove:
                    del self.memory_cache[key]
            else:
                self.memory_cache.clear()

            self.logger.info(f"Cache cleared, pattern: {pattern or 'all'}")
            return True

        except Exception as e:
            self.logger.error(f"Error clearing cache: {e}")
            return False

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "memory_cache_size": len(self.memory_cache),
        }


class CachedSearchEngine:
    """Cached wrapper for search engine to improve performance."""

    def __init__(self, search_engine, cache_manager: CacheManager):
        """Initialize cached search engine.

        Args:
            search_engine: The underlying search engine
            cache_manager: Cache manager instance
        """
        self.logger = get_logger(__name__)
        self.search_engine = search_engine
        self.cache_manager = cache_manager

    async def search(
        self,
        query: str,
        library: Optional[str] = None,
        version: Optional[str] = None,
        content_type: Optional[str] = None,
        max_results: int = 10,
        use_cache: bool = True,
    ) -> List[Dict[str, Any]]:
        """Search with caching support.

        Args:
            query: Search query
            library: Library filter
            version: Version filter
            content_type: Content type filter
            max_results: Maximum number of results
            use_cache: Whether to use cache

        Returns:
            Search results
        """
        if not use_cache:
            return await self.search_engine.search(
                query=query,
                library=library,
                version=version,
                content_type=content_type,
                max_results=max_results,
            )

        # Try to get from cache
        cached_results = await self.cache_manager.get(
            "search",
            query=query,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results,
        )

        if cached_results is not None:
            self.logger.info("Search results retrieved from cache")
            return cached_results

        # Perform search
        results = await self.search_engine.search(
            query=query,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results,
        )

        # Cache results (shorter TTL for search results)
        await self.cache_manager.set(
            "search",
            results,
            ttl=1800,  # 30 minutes
            query=query,
            library=library,
            version=version,
            content_type=content_type,
            max_results=max_results,
        )

        return results


def cache_decorator(prefix: str, ttl: Optional[int] = None):
    """Decorator for caching function results.

    Args:
        prefix: Cache key prefix
        ttl: TTL in seconds (optional)
    """

    def decorator(func):
        async def wrapper(self, *args, **kwargs):
            # Get cache manager from self
            cache_manager = getattr(self, "cache_manager", None)
            if not cache_manager:
                return await func(self, *args, **kwargs)

            # Try to get from cache
            cached_result = await cache_manager.get(prefix, *args, **kwargs)
            if cached_result is not None:
                return cached_result

            # Execute function
            result = await func(self, *args, **kwargs)

            # Cache result
            await cache_manager.set(prefix, result, ttl=ttl, *args, **kwargs)

            return result

        return wrapper

    return decorator
