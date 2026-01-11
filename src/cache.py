"""
Simple TTL cache for financial data.

Provides in-memory caching with configurable TTL to reduce API calls
and improve response times. Cache is thread-safe for concurrent access.
"""

import time
import threading
import logging
from typing import Optional, TypeVar, Generic
from dataclasses import dataclass

logger = logging.getLogger(__name__)

T = TypeVar('T')


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with value and expiration time."""
    value: T
    expires_at: float
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at


class TTLCache(Generic[T]):
    """
    Thread-safe TTL cache for any value type.
    
    Usage:
        cache = TTLCache[Quote](ttl_seconds=300)
        cache.set("AAPL", quote)
        quote = cache.get("AAPL")  # Returns None if expired
    """
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for entries (default 5 minutes)
            max_size: Maximum entries before cleanup (default 1000)
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: dict[str, CacheEntry[T]] = {}
        self._lock = threading.RLock()
        self._hits = 0
        self._misses = 0
    
    def get(self, key: str) -> Optional[T]:
        """Get value if exists and not expired."""
        with self._lock:
            entry = self._cache.get(key)
            
            if entry is None:
                self._misses += 1
                return None
            
            if entry.is_expired():
                del self._cache[key]
                self._misses += 1
                return None
            
            self._hits += 1
            return entry.value
    
    def set(self, key: str, value: T, ttl: Optional[int] = None) -> None:
        """Set value with optional custom TTL."""
        with self._lock:
            # Cleanup if at max size
            if len(self._cache) >= self.max_size:
                self._cleanup_expired()
            
            expires_at = time.time() + (ttl or self.ttl_seconds)
            self._cache[key] = CacheEntry(value=value, expires_at=expires_at)
    
    def get_multi(self, keys: list[str]) -> dict[str, T]:
        """Get multiple values, returning only non-expired hits."""
        results = {}
        with self._lock:
            for key in keys:
                value = self.get(key)
                if value is not None:
                    results[key] = value
        return results
    
    def set_multi(self, items: dict[str, T], ttl: Optional[int] = None) -> None:
        """Set multiple values."""
        with self._lock:
            for key, value in items.items():
                self.set(key, value, ttl)
    
    def invalidate(self, key: str) -> None:
        """Remove a specific key."""
        with self._lock:
            self._cache.pop(key, None)
    
    def clear(self) -> None:
        """Clear all entries."""
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0
    
    def _cleanup_expired(self) -> int:
        """Remove expired entries. Returns count removed."""
        now = time.time()
        expired_keys = [
            k for k, v in self._cache.items() 
            if v.expires_at < now
        ]
        for key in expired_keys:
            del self._cache[key]
        
        if expired_keys:
            logger.debug(f"Cache cleanup: removed {len(expired_keys)} expired entries")
        
        return len(expired_keys)
    
    @property
    def stats(self) -> dict:
        """Get cache statistics."""
        with self._lock:
            total = self._hits + self._misses
            hit_rate = (self._hits / total * 100) if total > 0 else 0
            return {
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": f"{hit_rate:.1f}%",
            }


# Global cache instances
_quote_cache: Optional[TTLCache] = None


def get_quote_cache(ttl_seconds: int = 300) -> TTLCache:
    """Get or create the global quote cache."""
    global _quote_cache
    if _quote_cache is None:
        _quote_cache = TTLCache(ttl_seconds=ttl_seconds)
        logger.info(f"Initialized quote cache with {ttl_seconds}s TTL")
    return _quote_cache
