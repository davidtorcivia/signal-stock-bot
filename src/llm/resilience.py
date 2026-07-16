"""Bounded retry and circuit-breaking for OpenAI-compatible HTTP calls."""

from __future__ import annotations

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import timezone
from email.utils import parsedate_to_datetime
from typing import Awaitable, Callable, Optional

import aiohttp


TRANSIENT_STATUSES = frozenset({408, 425, 429, 500, 502, 503, 504, 520, 522, 524})
DEFAULT_RETRIES = 2
DEFAULT_BASE_DELAY = 0.5
DEFAULT_MAX_DELAY = 5.0
DEFAULT_CIRCUIT_FAILURES = 5
DEFAULT_CIRCUIT_COOLDOWN = 60


@dataclass
class LLMHTTPFailure(Exception):
    status: int
    body: str
    retry_after: Optional[float] = None

    @property
    def transient(self) -> bool:
        return self.status in TRANSIENT_STATUSES

    def __str__(self) -> str:
        return f"LLM HTTP {self.status}: {self.body[:200].replace(chr(10), ' ')}"


class LLMTransportFailure(RuntimeError):
    """A network/stream timeout remained after all bounded retries."""


def parse_retry_after(value: Optional[str], *, now: Optional[float] = None) -> Optional[float]:
    if not value:
        return None
    raw = value.strip()
    try:
        return max(0.0, float(raw))
    except ValueError:
        pass
    try:
        when = parsedate_to_datetime(raw)
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        current = time.time() if now is None else now
        return max(0.0, when.timestamp() - current)
    except (TypeError, ValueError, OverflowError):
        return None


def retry_delay(
    retry_index: int,
    retry_after: Optional[float],
    *,
    base_delay: float = DEFAULT_BASE_DELAY,
    max_delay: float = DEFAULT_MAX_DELAY,
    jitter: Callable[[], float] = random.random,
) -> float:
    if retry_after is not None:
        return min(max_delay, max(0.0, retry_after))
    exponential = max(0.0, base_delay) * (2 ** max(0, retry_index))
    return min(max_delay, exponential + (max(0.0, base_delay) * jitter()))


async def resilient_chat_post(
    *,
    session,
    url: str,
    payload: dict,
    headers: dict,
    request_timeout: float,
    hard_timeout: float,
    provider_metrics,
    retry_attempts: int = DEFAULT_RETRIES,
    should_retry_unpinned: Optional[Callable[[dict, str], bool]] = None,
    on_retry: Optional[Callable[[str, int, float], None]] = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
) -> dict:
    """POST one completion within a single end-to-end deadline.

    Transient retries and the optional pinned-provider fallback share the same
    deadline, so resilience cannot silently multiply an administrator's timeout.
    """

    if provider_metrics is not None and not provider_metrics.is_healthy():
        raise RuntimeError("LLM provider circuit is open; try again shortly")

    deadline = time.monotonic() + max(0.1, hard_timeout)
    transient_used = 0
    unpinned_used = False
    attempt = 0

    async def _one(remaining: float) -> dict:
        timeout = aiohttp.ClientTimeout(
            total=min(max(0.1, request_timeout), remaining),
            connect=min(10.0, remaining),
        )
        async with session.post(url, json=payload, headers=headers, timeout=timeout) as resp:
            body = await resp.text()
            if resp.status >= 400:
                raise LLMHTTPFailure(
                    status=resp.status,
                    body=body,
                    retry_after=parse_retry_after(resp.headers.get("Retry-After")),
                )
            return await resp.json(content_type=None)

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise asyncio.TimeoutError
        attempt += 1
        started = time.monotonic()
        try:
            data = await asyncio.wait_for(_one(remaining), timeout=remaining)
        except LLMHTTPFailure as exc:
            if provider_metrics is not None:
                provider_metrics.record_error(
                    str(exc), consecutive=exc.transient,
                )
            if (
                should_retry_unpinned is not None
                and not unpinned_used
                and should_retry_unpinned(payload, str(exc))
            ):
                unpinned_used = True
                payload.pop("provider", None)
                if on_retry is not None:
                    on_retry("unpinned_provider", attempt, 0.0)
                continue
            if not exc.transient or transient_used >= max(0, retry_attempts):
                if (
                    provider_metrics is not None
                    and exc.transient
                    and provider_metrics.consecutive_errors >= DEFAULT_CIRCUIT_FAILURES
                ):
                    provider_metrics.open_circuit(DEFAULT_CIRCUIT_COOLDOWN)
                raise
            delay = retry_delay(transient_used, exc.retry_after)
            transient_used += 1
            if on_retry is not None:
                on_retry(f"http_{exc.status}", attempt, delay)
            if delay:
                remaining = deadline - time.monotonic()
                if delay >= remaining:
                    raise exc
                await sleep(delay)
        except (
            asyncio.TimeoutError,
            aiohttp.ClientConnectionError,
            aiohttp.ServerTimeoutError,
        ) as exc:
            if provider_metrics is not None:
                provider_metrics.record_error(str(exc))
            if transient_used >= max(0, retry_attempts):
                if (
                    provider_metrics is not None
                    and provider_metrics.consecutive_errors >= DEFAULT_CIRCUIT_FAILURES
                ):
                    provider_metrics.open_circuit(DEFAULT_CIRCUIT_COOLDOWN)
                if isinstance(exc, asyncio.TimeoutError):
                    raise LLMTransportFailure(
                        "LLM network timeout after bounded retries"
                    ) from exc
                raise LLMTransportFailure(
                    f"LLM network error after bounded retries: {exc}"
                ) from exc
            delay = retry_delay(transient_used, None)
            transient_used += 1
            if on_retry is not None:
                reason = "timeout" if isinstance(exc, asyncio.TimeoutError) else "network"
                on_retry(reason, attempt, delay)
            if delay:
                remaining = deadline - time.monotonic()
                if delay >= remaining:
                    raise LLMTransportFailure(
                        "LLM network retry budget exhausted"
                    ) from exc
                await sleep(delay)
        else:
            if provider_metrics is not None:
                provider_metrics.record_success((time.monotonic() - started) * 1000.0)
                if provider_metrics.circuit_open:
                    provider_metrics.close_circuit()
            return data
