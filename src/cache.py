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
from collections import OrderedDict, deque

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
    consecutive_errors: int = 0
    
    # Recent request latencies for percentile calculation
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_success(self, latency_ms: float):
        """Record a successful request."""
        self.requests += 1
        self.successes += 1
        self.total_latency_ms += latency_ms
        self.recent_latencies.append(latency_ms)
        self.consecutive_errors = 0
    
    def record_error(self, error_msg: str, *, consecutive: bool = True):
        """Record a failed request."""
        self.requests += 1
        self.errors += 1
        self.last_error_time = time.time()
        self.last_error_message = error_msg
        if consecutive:
            self.consecutive_errors += 1
    
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
            "consecutive_errors": self.consecutive_errors,
        }


@dataclass
class LLMMetrics:
    """Aggregated LLM call counters since process start."""
    calls: int = 0
    successes: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    total_latency_ms: float = 0.0
    last_call_at: Optional[float] = None
    last_error_at: Optional[float] = None
    last_error_msg: Optional[str] = None
    by_purpose: dict = field(default_factory=dict)   # purpose -> count
    by_model: dict = field(default_factory=dict)     # model   -> count
    retries: int = 0
    circuit_rejections: int = 0
    prompt_observations: int = 0
    system_fingerprint_changes: int = 0
    tool_fingerprint_changes: int = 0
    stable_block_changes: int = 0
    unexpected_cache_misses: int = 0
    recent_prompt_cache: deque = field(default_factory=lambda: deque(maxlen=20))

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.successes) if self.successes else 0.0

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return (self.cache_hit_tokens / total) if total else 0.0


@dataclass
class DeepThinkMetrics:
    """Counters for the deep_think tool — separate from main LLM since it
    points at a different (more expensive) model and we want isolated
    cost/latency visibility on the dashboard."""
    calls: int = 0
    successes: int = 0
    errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    cache_hit_tokens: int = 0
    cache_miss_tokens: int = 0
    total_latency_ms: float = 0.0
    last_call_at: Optional[float] = None
    last_error_at: Optional[float] = None
    last_error_msg: Optional[str] = None
    by_model: dict = field(default_factory=dict)     # model -> count

    @property
    def avg_latency_ms(self) -> float:
        return (self.total_latency_ms / self.successes) if self.successes else 0.0

    @property
    def cache_hit_ratio(self) -> float:
        total = self.cache_hit_tokens + self.cache_miss_tokens
        return (self.cache_hit_tokens / total) if total else 0.0


@dataclass
class ReactorMetrics:
    """Counters for the emoji reactor."""
    evaluations: int = 0          # times maybe_react was actually invoked (post-skip-checks)
    reactions_sent: int = 0       # tool was called and Signal API accepted the react
    skipped_disabled: int = 0     # globally or per-context off
    skipped_cooldown: int = 0
    skipped_short: int = 0
    skipped_no_tool: int = 0      # LLM declined (no tool call returned)
    responses_triggered: int = 0  # should_respond tool was invoked (natural-response feature)
    errors: int = 0
    by_emoji: dict = field(default_factory=dict)
    last_reaction_at: Optional[float] = None


class MetricsCollector:
    """
    Global metrics collector for the application.

    Tracks:
    - Cache statistics
    - Provider metrics
    - Request rates
    - LLM call metrics (calls / tokens / latency / by purpose+model)
    - Reactor metrics (evaluations / reactions / skip reasons / top emojis)
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
        self._llm = LLMMetrics()
        self._deep_think = DeepThinkMetrics()
        self._reactor = ReactorMetrics()
        # Last prompt fingerprints are bounded so a bot that sees many
        # one-off DMs cannot grow telemetry memory forever.
        self._prompt_fingerprints: OrderedDict[tuple, dict] = OrderedDict()
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
        _persist("request")
    
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
    
    # ── LLM metrics ────────────────────────────────────────────────────

    def record_llm_success(
        self,
        *,
        purpose: str,
        model: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None:
        with self._lock:
            m = self._llm
            m.calls += 1
            m.successes += 1
            m.tokens_in += int(tokens_in or 0)
            m.tokens_out += int(tokens_out or 0)
            m.cache_hit_tokens += int(cache_hit_tokens or 0)
            m.cache_miss_tokens += int(cache_miss_tokens or 0)
            m.total_latency_ms += latency_ms
            m.last_call_at = time.time()
            m.by_purpose[purpose] = m.by_purpose.get(purpose, 0) + 1
            if model:
                m.by_model[model] = m.by_model.get(model, 0) + 1
        _persist(
            "llm_success",
            purpose=purpose, model=model, latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
        )

    def record_llm_error(self, *, purpose: str, model: str, error_msg: str) -> None:
        with self._lock:
            m = self._llm
            m.calls += 1
            m.errors += 1
            m.last_error_at = time.time()
            m.last_error_msg = (error_msg or "")[:200]
            m.by_purpose[purpose] = m.by_purpose.get(purpose, 0) + 1
            if model:
                m.by_model[model] = m.by_model.get(model, 0) + 1
        _persist(
            "llm_error",
            purpose=purpose, model=model, error_msg=(error_msg or "")[:200],
        )

    def record_llm_retry(self, *, reason: str, purpose: str, model: str) -> None:
        with self._lock:
            self._llm.retries += 1
        logger.info(
            "LLM retry purpose=%s model=%s reason=%s",
            purpose, model, reason,
        )

    def record_llm_circuit_rejection(self, *, purpose: str, model: str) -> None:
        with self._lock:
            self._llm.circuit_rejections += 1
        logger.warning(
            "LLM circuit rejected request purpose=%s model=%s",
            purpose, model,
        )

    def record_prompt_cache_observation(
        self,
        manifest: dict,
        *,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
        latency_ms: float = 0.0,
    ) -> dict:
        """Compare one privacy-safe prompt fingerprint with its predecessor."""

        key = (
            manifest.get("context_id"),
            manifest.get("bot_id"),
            manifest.get("purpose"),
        )
        with self._lock:
            previous = self._prompt_fingerprints.get(key)
            system_changed = bool(
                previous
                and previous.get("system_hash") != manifest.get("system_hash")
            )
            tools_changed = bool(
                previous
                and previous.get("tools_hash") != manifest.get("tools_hash")
            )
            stable_changed: list[str] = []
            if previous:
                old_blocks = {
                    row.get("name"): row.get("hash")
                    for row in previous.get("stable_blocks", [])
                }
                new_blocks = {
                    row.get("name"): row.get("hash")
                    for row in manifest.get("stable_blocks", [])
                }
                stable_changed = sorted(
                    name for name in (old_blocks.keys() | new_blocks.keys())
                    if old_blocks.get(name) != new_blocks.get(name)
                )

            unexpected_miss = bool(
                previous
                and not system_changed
                and not tools_changed
                and int(cache_hit_tokens or 0) == 0
                and int(cache_miss_tokens or 0) >= 1024
            )
            event = {
                "ts": time.time(),
                "context_id": manifest.get("context_id"),
                "bot_id": manifest.get("bot_id"),
                "purpose": manifest.get("purpose"),
                "system_hash": str(manifest.get("system_hash") or "")[:12],
                "tools_hash": str(manifest.get("tools_hash") or "")[:12],
                "system_changed": system_changed,
                "tools_changed": tools_changed,
                "stable_changed": stable_changed,
                "unexpected_miss": unexpected_miss,
                "cache_hit_tokens": int(cache_hit_tokens or 0),
                "cache_miss_tokens": int(cache_miss_tokens or 0),
                "latency_ms": int(latency_ms or 0),
                "system_chars": int(manifest.get("system_chars") or 0),
                "tool_schema_chars": int(manifest.get("tool_schema_chars") or 0),
                "tool_count": int(manifest.get("tool_count") or 0),
            }
            m = self._llm
            m.prompt_observations += 1
            m.system_fingerprint_changes += int(system_changed)
            m.tool_fingerprint_changes += int(tools_changed)
            m.stable_block_changes += len(stable_changed)
            m.unexpected_cache_misses += int(unexpected_miss)
            m.recent_prompt_cache.appendleft(event)
            self._prompt_fingerprints[key] = manifest
            self._prompt_fingerprints.move_to_end(key)
            while len(self._prompt_fingerprints) > 256:
                self._prompt_fingerprints.popitem(last=False)

        if stable_changed:
            logger.warning(
                "Stable prompt blocks changed context=%s bot=%s: %s",
                manifest.get("context_id"), manifest.get("bot_id"),
                ", ".join(stable_changed),
            )
        if unexpected_miss:
            logger.warning(
                "Prompt cache miss with unchanged system/tools context=%s "
                "bot=%s miss_tokens=%s",
                manifest.get("context_id"), manifest.get("bot_id"),
                cache_miss_tokens,
            )
        return event

    # ── Deep think metrics ────────────────────────────────────────────

    def record_deep_think_success(
        self,
        *,
        model: str,
        latency_ms: float,
        tokens_in: int = 0,
        tokens_out: int = 0,
        cache_hit_tokens: int = 0,
        cache_miss_tokens: int = 0,
    ) -> None:
        with self._lock:
            m = self._deep_think
            m.calls += 1
            m.successes += 1
            m.tokens_in += int(tokens_in or 0)
            m.tokens_out += int(tokens_out or 0)
            m.cache_hit_tokens += int(cache_hit_tokens or 0)
            m.cache_miss_tokens += int(cache_miss_tokens or 0)
            m.total_latency_ms += latency_ms
            m.last_call_at = time.time()
            if model:
                m.by_model[model] = m.by_model.get(model, 0) + 1
        _persist(
            "dt_success",
            model=model, latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_hit_tokens=cache_hit_tokens,
            cache_miss_tokens=cache_miss_tokens,
        )

    def record_deep_think_error(self, *, model: str, error_msg: str) -> None:
        with self._lock:
            m = self._deep_think
            m.calls += 1
            m.errors += 1
            m.last_error_at = time.time()
            m.last_error_msg = (error_msg or "")[:200]
            if model:
                m.by_model[model] = m.by_model.get(model, 0) + 1
        _persist(
            "dt_error", model=model, error_msg=(error_msg or "")[:200],
        )

    # ── Reactor metrics ────────────────────────────────────────────────

    def record_reactor_skip(self, reason: str) -> None:
        with self._lock:
            r = self._reactor
            if reason == "disabled":
                r.skipped_disabled += 1
            elif reason == "cooldown":
                r.skipped_cooldown += 1
            elif reason == "short":
                r.skipped_short += 1
            elif reason == "no_tool":
                r.skipped_no_tool += 1
        _persist("reactor_skip", skip_reason=reason)

    def record_reactor_evaluation(self) -> None:
        with self._lock:
            self._reactor.evaluations += 1
        _persist("reactor_eval")

    def record_reactor_response(self) -> None:
        """The reactor's should_respond tool fired (natural-response feature)."""
        with self._lock:
            self._reactor.responses_triggered += 1
        _persist("reactor_response")

    def record_reactor_reaction(self, emoji: str) -> None:
        with self._lock:
            r = self._reactor
            r.reactions_sent += 1
            r.last_reaction_at = time.time()
            if emoji:
                r.by_emoji[emoji] = r.by_emoji.get(emoji, 0) + 1
        _persist("reactor_react", emoji=emoji)

    def record_reactor_error(self) -> None:
        with self._lock:
            self._reactor.errors += 1
        _persist("reactor_error")

    # ── Aggregate snapshot ─────────────────────────────────────────────

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

            total_hits = sum(c.stats["hits"] for c in self._caches.values())
            total_misses = sum(c.stats["misses"] for c in self._caches.values())
            total = total_hits + total_misses
            overall_hit_rate = (total_hits / total * 100) if total > 0 else 0

            llm = self._llm
            top_emojis = sorted(
                self._reactor.by_emoji.items(), key=lambda kv: kv[1], reverse=True
            )[:8]

            return {
                "uptime_seconds": self.uptime_seconds,
                "requests_per_minute": self.requests_per_minute,
                "cache": {
                    "overall_hit_rate": f"{overall_hit_rate:.1f}%",
                    "caches": cache_stats,
                },
                "providers": provider_stats,
                "llm": {
                    "calls": llm.calls,
                    "successes": llm.successes,
                    "errors": llm.errors,
                    "success_rate": (
                        f"{(llm.successes / llm.calls * 100):.1f}%"
                        if llm.calls else "—"
                    ),
                    "tokens_in": llm.tokens_in,
                    "tokens_out": llm.tokens_out,
                    "cache_hit_tokens": llm.cache_hit_tokens,
                    "cache_miss_tokens": llm.cache_miss_tokens,
                    "cache_hit_ratio": llm.cache_hit_ratio,
                    "avg_latency_ms": f"{llm.avg_latency_ms:.0f}",
                    "last_call_at": llm.last_call_at,
                    "last_error_at": llm.last_error_at,
                    "last_error_msg": llm.last_error_msg,
                    "by_purpose": dict(llm.by_purpose),
                    "by_model": dict(llm.by_model),
                    "retries": llm.retries,
                    "circuit_rejections": llm.circuit_rejections,
                    "prompt_observations": llm.prompt_observations,
                    "system_fingerprint_changes": llm.system_fingerprint_changes,
                    "tool_fingerprint_changes": llm.tool_fingerprint_changes,
                    "stable_block_changes": llm.stable_block_changes,
                    "unexpected_cache_misses": llm.unexpected_cache_misses,
                    "recent_prompt_cache": list(llm.recent_prompt_cache),
                },
                "reactor": {
                    "evaluations": self._reactor.evaluations,
                    "reactions_sent": self._reactor.reactions_sent,
                    "skipped_disabled": self._reactor.skipped_disabled,
                    "skipped_cooldown": self._reactor.skipped_cooldown,
                    "skipped_short": self._reactor.skipped_short,
                    "skipped_no_tool": self._reactor.skipped_no_tool,
                    "errors": self._reactor.errors,
                    "top_emojis": top_emojis,
                    "last_reaction_at": self._reactor.last_reaction_at,
                },
                "deep_think": {
                    "calls": self._deep_think.calls,
                    "successes": self._deep_think.successes,
                    "errors": self._deep_think.errors,
                    "success_rate": (
                        f"{(self._deep_think.successes / self._deep_think.calls * 100):.1f}%"
                        if self._deep_think.calls else "—"
                    ),
                    "tokens_in": self._deep_think.tokens_in,
                    "tokens_out": self._deep_think.tokens_out,
                    "cache_hit_tokens": self._deep_think.cache_hit_tokens,
                    "cache_miss_tokens": self._deep_think.cache_miss_tokens,
                    "cache_hit_ratio": self._deep_think.cache_hit_ratio,
                    "avg_latency_ms": f"{self._deep_think.avg_latency_ms:.0f}",
                    "last_call_at": self._deep_think.last_call_at,
                    "last_error_at": self._deep_think.last_error_at,
                    "last_error_msg": self._deep_think.last_error_msg,
                    "by_model": dict(self._deep_think.by_model),
                },
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


def _persist(kind: str, **fields) -> None:
    """Append one metric event to the persistent log so the dashboard
    can show windowed views (24h / 7d / 30d) that survive restarts.
    Lazy-imported to avoid a top-level cycle (metrics_log itself imports
    from .database, which other stores import from). Failure is silent —
    metrics persistence must never affect the hot path it's measuring.
    """
    try:
        from .metrics_log import get_metrics_log
        get_metrics_log().record(kind, **fields)
    except Exception:
        pass


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
            
            # Create a new future for this request. get_running_loop, not
            # the deprecated get_event_loop — the latter raises on 3.12+
            # when no loop is set on the thread.
            future = asyncio.get_running_loop().create_future()
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

