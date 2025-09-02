"""
Cache management for PyRAG.

This module provides intelligent caching capabilities including query result caching,
metadata caching, and adaptive cache management for improved performance.
"""

import asyncio
import hashlib
import json
import logging
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with metadata."""

    key: str
    value: Any
    created_at: float
    accessed_at: float
    access_count: int
    size_bytes: int
    ttl: Optional[float] = None
    tags: List[str] = field(default_factory=list)


@dataclass
class CacheStats:
    """Cache statistics."""

    total_entries: int
    total_size_bytes: int
    hit_count: int
    miss_count: int
    hit_rate: float
    avg_access_count: float
    oldest_entry: float
    newest_entry: float


class CacheManager:
    """
    Intelligent cache manager for PyRAG.

    Provides:
    - Query result caching
    - Metadata caching
    - Adaptive TTL management
    - Cache eviction policies
    - Performance monitoring
    """

    def __init__(
        self,
        max_size_mb: int = 100,
        max_entries: int = 1000,
        default_ttl: float = 3600,  # 1 hour
        eviction_policy: str = "lru",
    ):
        """
        Initialize the cache manager.

        Args:
            max_size_mb: Maximum cache size in MB
            max_entries: Maximum number of cache entries
            default_ttl: Default time-to-live in seconds
            eviction_policy: Cache eviction policy ('lru', 'lfu', 'fifo')
        """
        self.max_size_bytes = max_size_mb * 1024 * 1024
        self.max_entries = max_entries
        self.default_ttl = default_ttl
        self.eviction_policy = eviction_policy

        # Cache storage
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()

        # Statistics
        self.stats = {"hits": 0, "misses": 0, "evictions": 0, "total_size": 0}

        # Background cleanup task
        self._cleanup_task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, cleanup_interval: float = 300.0):
        """
        Start the cache manager with background cleanup.

        Args:
            cleanup_interval: Interval between cleanup runs (seconds)
        """
        if self._running:
            logger.warning("Cache manager is already running")
            return

        self._running = True
        self._cleanup_task = asyncio.create_task(self._cleanup_loop(cleanup_interval))
        logger.info("Cache manager started")

    async def stop(self):
        """Stop the cache manager."""
        self._running = False

        if self._cleanup_task and not self._cleanup_task.done():
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass

        logger.info("Cache manager stopped")

    async def _cleanup_loop(self, interval: float):
        """Background cleanup loop."""
        while self._running:
            try:
                await self._cleanup_expired_entries()
                await asyncio.sleep(interval)
            except Exception as e:
                logger.error(f"Cache cleanup error: {e}")
                await asyncio.sleep(60.0)  # Short delay on error

    def _generate_key(self, *args, **kwargs) -> str:
        """Generate cache key from arguments."""
        # Create a deterministic string representation
        key_data = {"args": args, "kwargs": sorted(kwargs.items())}

        key_string = json.dumps(key_data, sort_keys=True)
        return hashlib.md5(key_string.encode()).hexdigest()

    def _estimate_size(self, value: Any) -> int:
        """Estimate size of a value in bytes."""
        try:
            # Try to serialize and measure
            serialized = pickle.dumps(value)
            return len(serialized)
        except Exception:
            # Fallback to string representation
            return len(str(value).encode("utf-8"))

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from cache.

        Args:
            key: Cache key
            default: Default value if key not found

        Returns:
            Cached value or default
        """
        if key not in self.cache:
            self.stats["misses"] += 1
            return default

        entry = self.cache[key]

        # Check if entry is expired
        if entry.ttl and (time.time() - entry.created_at) > entry.ttl:
            del self.cache[key]
            self.stats["misses"] += 1
            self.stats["total_size"] -= entry.size_bytes
            return default

        # Update access statistics
        entry.accessed_at = time.time()
        entry.access_count += 1

        # Update LRU order
        self.cache.move_to_end(key)

        self.stats["hits"] += 1
        return entry.value

    def set(
        self,
        key: str,
        value: Any,
        ttl: Optional[float] = None,
        tags: Optional[List[str]] = None,
    ) -> bool:
        """
        Set a value in cache.

        Args:
            key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (None for default)
            tags: Optional tags for the entry

        Returns:
            True if successfully cached, False otherwise
        """
        # Estimate size
        size_bytes = self._estimate_size(value)

        # Check if we need to evict entries
        while (
            len(self.cache) >= self.max_entries
            or self.stats["total_size"] + size_bytes > self.max_size_bytes
        ) and len(self.cache) > 0:
            self._evict_entry()

        # Create cache entry
        entry = CacheEntry(
            key=key,
            value=value,
            created_at=time.time(),
            accessed_at=time.time(),
            access_count=1,
            size_bytes=size_bytes,
            ttl=ttl or self.default_ttl,
            tags=tags or [],
        )

        # Add to cache
        self.cache[key] = entry
        self.stats["total_size"] += size_bytes

        return True

    def delete(self, key: str) -> bool:
        """
        Delete a cache entry.

        Args:
            key: Cache key to delete

        Returns:
            True if entry was deleted, False if not found
        """
        if key not in self.cache:
            return False

        entry = self.cache[key]
        del self.cache[key]
        self.stats["total_size"] -= entry.size_bytes

        return True

    def delete_by_tags(self, tags: List[str]) -> int:
        """
        Delete cache entries by tags.

        Args:
            tags: List of tags to match

        Returns:
            Number of entries deleted
        """
        deleted_count = 0
        keys_to_delete = []

        for key, entry in self.cache.items():
            if any(tag in entry.tags for tag in tags):
                keys_to_delete.append(key)

        for key in keys_to_delete:
            if self.delete(key):
                deleted_count += 1

        return deleted_count

    def clear(self):
        """Clear all cache entries."""
        self.cache.clear()
        self.stats["total_size"] = 0
        logger.info("Cache cleared")

    def _evict_entry(self):
        """Evict an entry based on the eviction policy."""
        if not self.cache:
            return

        if self.eviction_policy == "lru":
            # Remove least recently used
            key = next(iter(self.cache))
        elif self.eviction_policy == "lfu":
            # Remove least frequently used
            key = min(self.cache.keys(), key=lambda k: self.cache[k].access_count)
        elif self.eviction_policy == "fifo":
            # Remove first in (oldest)
            key = next(iter(self.cache))
        else:
            # Default to LRU
            key = next(iter(self.cache))

        entry = self.cache[key]
        del self.cache[key]
        self.stats["total_size"] -= entry.size_bytes
        self.stats["evictions"] += 1

    async def _cleanup_expired_entries(self):
        """Remove expired entries from cache."""
        current_time = time.time()
        expired_keys = []

        for key, entry in self.cache.items():
            if entry.ttl and (current_time - entry.created_at) > entry.ttl:
                expired_keys.append(key)

        for key in expired_keys:
            self.delete(key)

        if expired_keys:
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")

    def get_stats(self) -> CacheStats:
        """Get cache statistics."""
        total_entries = len(self.cache)
        total_hits = self.stats["hits"]
        total_misses = self.stats["misses"]
        total_requests = total_hits + total_misses

        hit_rate = total_hits / total_requests if total_requests > 0 else 0.0

        avg_access_count = 0.0
        if total_entries > 0:
            total_access_count = sum(
                entry.access_count for entry in self.cache.values()
            )
            avg_access_count = total_access_count / total_entries

        oldest_entry = (
            min(entry.created_at for entry in self.cache.values())
            if self.cache
            else 0.0
        )
        newest_entry = (
            max(entry.created_at for entry in self.cache.values())
            if self.cache
            else 0.0
        )

        return CacheStats(
            total_entries=total_entries,
            total_size_bytes=self.stats["total_size"],
            hit_count=total_hits,
            miss_count=total_misses,
            hit_rate=hit_rate,
            avg_access_count=avg_access_count,
            oldest_entry=oldest_entry,
            newest_entry=newest_entry,
        )

    def get_entries_by_tag(self, tag: str) -> List[Dict[str, Any]]:
        """Get cache entries by tag."""
        entries = []

        for key, entry in self.cache.items():
            if tag in entry.tags:
                entries.append(
                    {
                        "key": key,
                        "created_at": entry.created_at,
                        "accessed_at": entry.accessed_at,
                        "access_count": entry.access_count,
                        "size_bytes": entry.size_bytes,
                        "ttl": entry.ttl,
                        "tags": entry.tags,
                    }
                )

        return entries

    def warm_cache(self, warmup_data: List[Dict[str, Any]]):
        """
        Warm up cache with predefined data.

        Args:
            warmup_data: List of cache entries to preload
        """
        for item in warmup_data:
            key = item.get("key")
            value = item.get("value")
            ttl = item.get("ttl")
            tags = item.get("tags", [])

            if key and value is not None:
                self.set(key, value, ttl, tags)

        logger.info(f"Warmed cache with {len(warmup_data)} entries")


class QueryCache:
    """
    Specialized cache for query results.

    Provides intelligent caching for search queries with
    automatic key generation and result optimization.
    """

    def __init__(self, cache_manager: CacheManager):
        """
        Initialize query cache.

        Args:
            cache_manager: Underlying cache manager
        """
        self.cache_manager = cache_manager
        self.query_stats = {"cached_queries": 0, "cache_hits": 0, "cache_misses": 0}

    def _generate_query_key(self, query: str, **kwargs) -> str:
        """Generate cache key for a query."""
        # Normalize query
        normalized_query = query.lower().strip()

        # Include additional parameters
        params = sorted(kwargs.items())

        key_data = {"query": normalized_query, "params": params}

        key_string = json.dumps(key_data, sort_keys=True)
        return f"query:{hashlib.md5(key_string.encode()).hexdigest()}"

    def get_cached_result(self, query: str, **kwargs) -> Optional[Any]:
        """
        Get cached result for a query.

        Args:
            query: Search query
            **kwargs: Additional query parameters

        Returns:
            Cached result or None
        """
        key = self._generate_query_key(query, **kwargs)
        result = self.cache_manager.get(key)

        if result is not None:
            self.query_stats["cache_hits"] += 1
            return result

        self.query_stats["cache_misses"] += 1
        return None

    def cache_result(
        self, query: str, result: Any, ttl: Optional[float] = None, **kwargs
    ) -> bool:
        """
        Cache a query result.

        Args:
            query: Search query
            result: Query result to cache
            ttl: Time-to-live for the cache entry
            **kwargs: Additional query parameters

        Returns:
            True if successfully cached
        """
        key = self._generate_query_key(query, **kwargs)

        # Determine TTL based on query complexity
        if ttl is None:
            ttl = self._determine_query_ttl(query)

        # Add query-specific tags
        tags = ["query_result", f"query_type:{self._classify_query(query)}"]

        success = self.cache_manager.set(key, result, ttl, tags)

        if success:
            self.query_stats["cached_queries"] += 1

        return success

    def _determine_query_ttl(self, query: str) -> float:
        """Determine TTL based on query characteristics."""
        query_lower = query.lower()

        # Short TTL for time-sensitive queries
        if any(word in query_lower for word in ["latest", "recent", "new", "update"]):
            return 300  # 5 minutes

        # Medium TTL for general queries
        if any(word in query_lower for word in ["how to", "what is", "example"]):
            return 3600  # 1 hour

        # Long TTL for reference queries
        if any(word in query_lower for word in ["reference", "api", "documentation"]):
            return 86400  # 24 hours

        # Default TTL
        return 1800  # 30 minutes

    def _classify_query(self, query: str) -> str:
        """Classify query type for tagging."""
        query_lower = query.lower()

        if "how to" in query_lower:
            return "how_to"
        elif "what is" in query_lower:
            return "definition"
        elif "compare" in query_lower:
            return "comparison"
        elif "example" in query_lower:
            return "example"
        elif "api" in query_lower:
            return "api_reference"
        else:
            return "general"

    def get_query_stats(self) -> Dict[str, Any]:
        """Get query cache statistics."""
        cache_stats = self.cache_manager.get_stats()

        return {
            "cache_stats": cache_stats,
            "query_stats": self.query_stats.copy(),
            "hit_rate": (
                self.query_stats["cache_hits"]
                / max(
                    1, self.query_stats["cache_hits"] + self.query_stats["cache_misses"]
                )
            ),
        }

    def clear_query_cache(self, query_type: Optional[str] = None):
        """
        Clear query cache.

        Args:
            query_type: Optional query type to clear (e.g., "how_to", "api_reference")
        """
        if query_type:
            tags = [f"query_type:{query_type}"]
            deleted_count = self.cache_manager.delete_by_tags(tags)
            logger.info(f"Cleared {deleted_count} {query_type} query cache entries")
        else:
            self.cache_manager.delete_by_tags(["query_result"])
            logger.info("Cleared all query cache entries")
