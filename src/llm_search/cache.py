"""TTL cache for search results.

Caches by (query + num_results) hash to avoid redundant API calls
when the LLM re-searches similar queries in a conversation.
"""

import hashlib
import time
from threading import Lock
from typing import Optional

from cachetools import TTLCache

from .config import settings


class SearchCache:
    """Thread-safe TTL cache for search results."""

    def __init__(self, ttl_seconds: Optional[int] = None):
        ttl = ttl_seconds if ttl_seconds is not None else settings.search_cache_ttl_seconds
        self._cache: TTLCache = TTLCache(maxsize=1000, ttl=ttl)
        self._lock = Lock()
        self._hits: int = 0
        self._misses: int = 0

    @staticmethod
    def _key(query: str, num_results: int) -> str:
        raw = f"{query.lower().strip()}|{num_results}"
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def get(self, query: str, num_results: int) -> Optional[list[dict]]:
        key = self._key(query, num_results)
        with self._lock:
            result = self._cache.get(key)
            if result is not None:
                self._hits += 1
                return result
            self._misses += 1
            return None

    def set(self, query: str, num_results: int, results: list[dict]) -> None:
        key = self._key(query, num_results)
        with self._lock:
            self._cache[key] = results

    @property
    def hits(self) -> int:
        return self._hits

    @property
    def misses(self) -> int:
        return self._misses

    @property
    def hit_rate(self) -> float:
        total = self._hits + self._misses
        if total == 0:
            return 0.0
        return self._hits / total

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0


# Module-level singleton
cache = SearchCache()
