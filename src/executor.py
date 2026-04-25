"""
Shared bounded thread pool for blocking I/O and CPU-bound work.

yfinance, matplotlib, and similar libraries are synchronous. Running them via
the default asyncio executor uses an unbounded pool and provides no timeout,
so a single stuck call can exhaust the pool or hang the bot indefinitely.
"""

import asyncio
import concurrent.futures
import functools
import logging
from typing import Any, Callable, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")

DEFAULT_TIMEOUT = 20.0
RENDER_TIMEOUT = 30.0

_BLOCKING_POOL = concurrent.futures.ThreadPoolExecutor(
    max_workers=8,
    thread_name_prefix="bot-blocking",
)


async def run_blocking(
    func: Callable[..., T],
    *args: Any,
    timeout: float = DEFAULT_TIMEOUT,
    **kwargs: Any,
) -> T:
    """Run a sync callable in the shared pool with a hard timeout."""
    loop = asyncio.get_running_loop()
    call = functools.partial(func, *args, **kwargs)
    future = loop.run_in_executor(_BLOCKING_POOL, call)
    try:
        return await asyncio.wait_for(future, timeout=timeout)
    except asyncio.TimeoutError:
        logger.warning(
            f"Blocking call to {getattr(func, '__name__', func)!r} timed out after {timeout}s"
        )
        raise


def shutdown(wait: bool = False) -> None:
    """Shut down the pool (only useful at process exit)."""
    _BLOCKING_POOL.shutdown(wait=wait)
