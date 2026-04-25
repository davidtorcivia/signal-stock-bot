"""
In-memory pub/sub event bus for the admin live viewer.

The dispatcher and handler publish small structured events here as they
process messages; the SSE endpoint subscribes and streams them to the
admin browser. The bus is bounded — slow subscribers drop oldest events
rather than back-pressuring the main loop.

Events are intentionally lightweight (a dict, not an object) so they
serialize trivially to SSE `data:` lines without ceremony.

Cross-thread bridging: the bot's async work runs on a dedicated loop in
its own thread; Flask requests run on gunicorn workers. `publish()` is
called from the bot loop; `subscribe()` is consumed from the Flask
request thread. We use a thread-safe `queue.Queue` per subscriber so
nothing has to know about asyncio across the boundary.
"""

import logging
import threading
import time
from collections import deque
from queue import Empty, Full, Queue
from typing import Any

logger = logging.getLogger(__name__)

MAX_QUEUE_PER_SUBSCRIBER = 200
HISTORY_KEEP = 100  # last N events replayed to new subscribers


class AdminEventBus:
    def __init__(self):
        self._subscribers: list[Queue] = []
        self._history: deque = deque(maxlen=HISTORY_KEEP)
        self._lock = threading.Lock()

    def publish(self, event_type: str, **fields: Any) -> None:
        """Add an event and fan it out to current subscribers.

        Safe to call from any thread (the asyncio bot loop OR a Flask
        worker). Slow subscribers drop oldest events on overflow.
        """
        event = {"type": event_type, "ts": time.time(), **fields}
        with self._lock:
            self._history.append(event)
            for q in self._subscribers:
                try:
                    q.put_nowait(event)
                except Full:
                    # Drop oldest from this subscriber's queue, then push.
                    try:
                        q.get_nowait()
                        q.put_nowait(event)
                    except Empty:
                        pass

    def subscribe(self) -> Queue:
        """Register a subscriber and immediately replay recent history."""
        q: Queue = Queue(maxsize=MAX_QUEUE_PER_SUBSCRIBER)
        with self._lock:
            for past in self._history:
                try:
                    q.put_nowait(past)
                except Full:
                    break
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: Queue) -> None:
        with self._lock:
            try:
                self._subscribers.remove(q)
            except ValueError:
                pass

    def history(self) -> list[dict]:
        with self._lock:
            return list(self._history)


# Module-level singleton — dispatcher/handler import this directly so we
# don't have to thread the bus through every constructor signature.
_BUS: AdminEventBus | None = None


def get_bus() -> AdminEventBus:
    global _BUS
    if _BUS is None:
        _BUS = AdminEventBus()
    return _BUS
