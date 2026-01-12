"""
Advanced caching and metrics system for the stock bot.

Provides:
- TTL-based caching with data-type-specific expiration
- Provider metrics tracking (latency, errors, success rate)
- Rate limiting with exponential backoff
- Circuit breaker pattern for failing providers
"""

import time
import asyncio
import threading
import logging
from typing import Optional, TypeVar, Generic, Dict, Any
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque

logger = logging.getLogger(__name__)

T = TypeVar('T')


# TTL values in seconds
class CacheTTL:
    """Cache TTL values by data type."""
    INTRADAY_QUOTE = 60      # 1 minute
    DAILY_QUOTE = 300        # 5 minutes
    FUNDAMENTALS = 3600      # 1 hour
    CHART = 300              # 5 minutes
    HISTORICAL = 86400       # 24 hours
    NEWS = 600               # 10 minutes
    EARNINGS = 3600          # 1 hour


@dataclass
class CacheEntry(Generic[T]):
    """Single cache entry with value and expiration time."""
    value: T
    expires_at: float
    created_at: float = field(default_factory=time.time)
    
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
    
    def __init__(self, ttl_seconds: int = 300, max_size: int = 1000, name: str = "cache"):
        """
        Initialize cache.
        
        Args:
            ttl_seconds: Time-to-live for entries (default 5 minutes)
            max_size: Maximum entries before cleanup (default 1000)
            name: Cache name for logging
        """
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self.name = name
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
                "name": self.name,
                "size": len(self._cache),
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": hit_rate,
                "ttl_seconds": self.ttl_seconds,
            }


@dataclass
class ProviderMetrics:
    """Metrics for a single provider."""
    name: str
    requests: int = 0
    successes: int = 0
    errors: int = 0
    total_latency_ms: float = 0.0
    last_error_time: Optional[float] = None
    last_error_message: Optional[str] = None
    circuit_open: bool = False
    circuit_open_until: Optional[float] = None
    
    # Recent request latencies for percentile calculation
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.requests += 1
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.recent_latencies.append(latency_ms)
    
    def record_error(self, error_msg: str):
        """Record a failed request."""
        self.requests += 1
        self.errors += 1
        self.last_error_time = time.time()
        self.last_error_message = error_msg
    
    @property
    def success_rate(self) -> float:
        """Success rate as percentage."""
        if self.requests == 0:
            return 100.0
        return (self.successes / self.requests) * 100
    
    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds."""
        if self.successes == 0:
            return 0.0
        return self.total_latency_ms / self.successes
    
    @property
    def p95_latency_ms(self) -> float:
        """95th percentile latency."""
        if not self.recent_latencies:
            return 0.0
        sorted_latencies = sorted(self.recent_latencies)
        idx = int(len(sorted_latencies) * 0.95)
        return sorted_latencies[min(idx, len(sorted_latencies) - 1)]
    
    def is_healthy(self) -> bool:
        """Check if provider is healthy (circuit closed)."""
        if not self.circuit_open:
            return True
        
        # Check if circuit should be half-open (allow retry)
        if self.circuit_open_until and time.time() > self.circuit_open_until:
            return True
        
        return False
    
    def open_circuit(self, duration_seconds: int = 60):
        """Open the circuit breaker."""
        self.circuit_open = True
        self.circuit_open_until = time.time() + duration_seconds
        logger.warning(f"Circuit opened for {self.name} for {duration_seconds}s")
    
    def close_circuit(self):
        """Close the circuit breaker."""
        self.circuit_open = False
        self.circuit_open_until = None
        logger.info(f"Circuit closed for {self.name}")
    
    def to_dict(self) -> dict:
        """Convert to dictionary for display."""
        return {
            "name": self.name,
            "requests": self.requests,
            "successes": self.successes,
            "errors": self.errors,
            "success_rate": f"{self.success_rate:.1f}%",
            "avg_latency_ms": f"{self.avg_latency_ms:.0f}ms",
            "p95_latency_ms": f"{self.p95_latency_ms:.0f}ms",
            "healthy": self.is_healthy(),
            "circuit_open": self.circuit_open,
        }


class MetricsCollector:
    """
    Global metrics collector for the application.
    
    Tracks:
    - Cache statistics
    - Provider metrics
    - Request rates
    """
    
    _instance: Optional['MetricsCollector'] = None
    _lock = threading.Lock()
    
    def __new__(cls):
        """Singleton pattern."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._start_time = time.time()
        self._providers: Dict[str, ProviderMetrics] = {}
        self._caches: Dict[str, TTLCache] = {}
        self._request_times: deque = deque(maxlen=1000)  # Last 1000 request timestamps
        self._lock = threading.RLock()
    
    def register_cache(self, name: str, cache: TTLCache):
        """Register a cache for metrics tracking."""
        with self._lock:
            self._caches[name] = cache
    
    def get_provider_metrics(self, name: str) -> ProviderMetrics:
        """Get or create provider metrics."""
        with self._lock:
            if name not in self._providers:
                self._providers[name] = ProviderMetrics(name=name)
            return self._providers[name]
    
    def record_request(self):
        """Record a request timestamp for rate calculation."""
        with self._lock:
            self._request_times.append(time.time())
    
    @property
    def requests_per_minute(self) -> float:
        """Calculate requests per minute over the last minute."""
        with self._lock:
            now = time.time()
            minute_ago = now - 60
            recent = [t for t in self._request_times if t > minute_ago]
            return len(recent)
    
    @property
    def uptime_seconds(self) -> float:
        """Application uptime in seconds."""
        return time.time() - self._start_time
    
    def get_all_stats(self) -> dict:
        """Get all metrics as a dictionary."""
        with self._lock:
            cache_stats = {
                name: cache.stats 
                for name, cache in self._caches.items()
            }
            
            provider_stats = {
                name: metrics.to_dict() 
                for name, metrics in self._providers.items()
            }
            
            # Calculate aggregate cache hit rate
            total_hits = sum(c.stats["hits"] for c in self._caches.values())
            total_misses = sum(c.stats["misses"] for c in self._caches.values())
            total = total_hits + total_misses
            overall_hit_rate = (total_hits / total * 100) if total > 0 else 0
            
            return {
                "uptime_seconds": self.uptime_seconds,
                "requests_per_minute": self.requests_per_minute,
                "cache": {
                    "overall_hit_rate": f"{overall_hit_rate:.1f}%",
                    "caches": cache_stats,
                },
                "providers": provider_stats,
            }


# Global cache instances
class CacheManager:
    """Manages all application caches with appropriate TTLs."""
    
    _instance: Optional['CacheManager'] = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        
        self._initialized = True
        self._metrics = MetricsCollector()
        
        # Create typed caches with appropriate TTLs
        self.quotes = TTLCache(ttl_seconds=CacheTTL.DAILY_QUOTE, name="quotes")
        self.intraday = TTLCache(ttl_seconds=CacheTTL.INTRADAY_QUOTE, name="intraday")
        self.fundamentals = TTLCache(ttl_seconds=CacheTTL.FUNDAMENTALS, name="fundamentals")
        self.charts = TTLCache(ttl_seconds=CacheTTL.CHART, name="charts")
        self.historical = TTLCache(ttl_seconds=CacheTTL.HISTORICAL, name="historical")
        self.news = TTLCache(ttl_seconds=CacheTTL.NEWS, name="news")
        self.earnings = TTLCache(ttl_seconds=CacheTTL.EARNINGS, name="earnings")
        
        # Register all caches with metrics
        for name in ["quotes", "intraday", "fundamentals", "charts", "historical", "news", "earnings"]:
            self._metrics.register_cache(name, getattr(self, name))
        
        logger.info("CacheManager initialized with data-type-specific caches")
    
    def get_all_stats(self) -> dict:
        """Get statistics for all caches."""
        return {
            "quotes": self.quotes.stats,
            "intraday": self.intraday.stats,
            "fundamentals": self.fundamentals.stats,
            "charts": self.charts.stats,
            "historical": self.historical.stats,
            "news": self.news.stats,
            "earnings": self.earnings.stats,
        }
    
    def clear_all(self):
        """Clear all caches."""
        self.quotes.clear()
        self.intraday.clear()
        self.fundamentals.clear()
        self.charts.clear()
        self.historical.clear()
        self.news.clear()
        self.earnings.clear()


def get_cache_manager() -> CacheManager:
    """Get the global cache manager."""
    return CacheManager()


def get_metrics() -> MetricsCollector:
    """Get the global metrics collector."""
    return MetricsCollector()


class RequestDeduplicator:
    """
    Coalesces identical concurrent requests to avoid duplicate API calls.
    
    If multiple requests for the same key arrive within the window,
    only the first actually executes - others await the same result.
    
    Usage:
        dedup = RequestDeduplicator()
        result = await dedup.execute("AAPL:quote", fetch_quote, "AAPL")
    """
    
    def __init__(self, window_ms: int = 100):
        self.window_ms = window_ms
        self._pending: dict[str, asyncio.Future] = {}
        self._lock = asyncio.Lock()
    
    async def execute(self, key: str, func, *args, **kwargs):
        """
        Execute func if no pending request for key, else return pending result.
        
        Args:
            key: Unique identifier for this request type
            func: Async function to call
            *args, **kwargs: Arguments for func
        """
        async with self._lock:
            # If there's already a pending request for this key, wait for it
            if key in self._pending:
                logger.debug(f"Dedup hit for {key}")
                return await self._pending[key]
            
            # Create a new future for this request
            future = asyncio.get_event_loop().create_future()
            self._pending[key] = future
        
        try:
            # Execute the actual request
            result = await func(*args, **kwargs)
            future.set_result(result)
            return result
        except Exception as e:
            future.set_exception(e)
            raise
        finally:
            # Clean up after a short delay to catch near-simultaneous requests
            async def cleanup():
                await asyncio.sleep(self.window_ms / 1000)
                async with self._lock:
                    if key in self._pending and self._pending[key] is future:
                        del self._pending[key]
            
            asyncio.create_task(cleanup())


# Global deduplicator instance
_deduplicator: Optional[RequestDeduplicator] = None

def get_deduplicator() -> RequestDeduplicator:
    """Get the global request deduplicator."""
    global _deduplicator
    if _deduplicator is None:
        _deduplicator = RequestDeduplicator()
    return _deduplicator

