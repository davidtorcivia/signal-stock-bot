"""
Persistent metrics event log.

The in-process `MetricsCollector` (cache.py) is fast but loses everything on
restart and only ever shows since-boot totals. The dashboard wants windowed
views — last 24h, 7d, 30d — that survive deploys. This module persists each
metric event to SQLite so those windows can be queried at any time.

Design:
  * Sync `record(kind, **fields)` — appends to an in-process queue. Cheap
    (one append + one lock). Safe to call from anywhere, async or sync.
  * Async `flush()` — drains the queue to SQLite in a single executemany().
    Called periodically by a background worker (every ~10s) so dashboard
    queries see fresh data without paying a per-event commit cost.
  * Async `query_window(seconds)` — aggregates events into the same shape
    the in-memory collector returns, so dashboard rendering doesn't care
    whether it's looking at a window or the live counters.
  * Async `prune(retention_days)` — drops rows older than the retention
    setting. Default 30d, capped at the largest selectable window.

Why not write through aiosqlite directly from `record`? The collector is
called from the LLM client, reactor, and dispatcher — sometimes from sync
threads (Flask), sometimes from coroutines. A queue + flush keeps the
record path single-line and decouples write latency from message
handling.
"""

from __future__ import annotations

import asyncio
import collections
import logging
import threading
import time
from pathlib import Path
from typing import Any, Optional

import aiosqlite

from .database import db_session

logger = logging.getLogger(__name__)


# Bounded queue — if the flusher falls more than this far behind, oldest
# events are dropped first. Set high enough that a 60s flush gap during a
# busy minute doesn't lose anything; capped so a stuck flusher can't OOM.
_QUEUE_MAX = 50_000

# Event kinds. Keep these stable — they're persisted into SQL.
KIND_LLM_OK = "llm_success"
KIND_LLM_ERR = "llm_error"
KIND_DT_OK = "dt_success"
KIND_DT_ERR = "dt_error"
KIND_REACTOR_EVAL = "reactor_eval"
KIND_REACTOR_REACT = "reactor_react"
KIND_REACTOR_SKIP = "reactor_skip"
KIND_REACTOR_RESPONSE = "reactor_response"
KIND_REACTOR_ERROR = "reactor_error"
KIND_REQUEST = "request"

# Reactor skip reasons that get a pre-seeded zero in the windowed aggregate,
# so the dashboard renders "budget: 0" rather than a blank when nothing has
# been skipped for that reason yet. This list is cosmetic ONLY — the
# aggregate accumulates whatever reasons it actually finds, so a reason
# added to the reactor without being added here still shows up in the data.
# Keep it in sync with MetricsCollector._REACTOR_SKIP_FIELDS; a test
# enforces that.
REACTOR_SKIP_REASONS = (
    "disabled",
    "cooldown",
    "short",
    "no_tool",
    "will_reply",
    "budget",
    "repeat",
    "low_score",
    "no_tools",
)


class MetricsLog:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        self._queue: collections.deque = collections.deque(maxlen=_QUEUE_MAX)
        self._queue_lock = threading.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                from .database import apply_db_pragmas
                await apply_db_pragmas(db)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS metric_events (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        occurred_at REAL NOT NULL,
                        kind TEXT NOT NULL,
                        purpose TEXT,
                        model TEXT,
                        latency_ms REAL,
                        tokens_in INTEGER,
                        tokens_out INTEGER,
                        cache_hit_tokens INTEGER,
                        cache_miss_tokens INTEGER,
                        emoji TEXT,
                        skip_reason TEXT,
                        error_msg TEXT,
                        bot_id INTEGER,
                        score INTEGER
                    )
                    """
                )
                # Pre-existing installs may not yet have newer columns; add
                # them in-place. Each ALTER is guarded so re-running on a
                # fresh install is a no-op.
                cursor = await db.execute("PRAGMA table_info(metric_events)")
                cols = {r[1] for r in await cursor.fetchall()}
                if "bot_id" not in cols:
                    await db.execute(
                        "ALTER TABLE metric_events ADD COLUMN bot_id INTEGER"
                    )
                if "cache_hit_tokens" not in cols:
                    await db.execute(
                        "ALTER TABLE metric_events ADD COLUMN cache_hit_tokens INTEGER"
                    )
                if "cache_miss_tokens" not in cols:
                    await db.execute(
                        "ALTER TABLE metric_events ADD COLUMN cache_miss_tokens INTEGER"
                    )
                if "score" not in cols:
                    await db.execute(
                        "ALTER TABLE metric_events ADD COLUMN score INTEGER"
                    )
                # Two indexes: one for kind-scoped windowed queries (the
                # common dashboard shape), one for time-only prunes.
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metric_kind_time "
                    "ON metric_events(kind, occurred_at)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_metric_time "
                    "ON metric_events(occurred_at)"
                )
                await db.commit()
            self._initialized = True

    # ── Sync recording — fast, lock-protected queue append ─────────────

    def record(self, kind: str, **fields: Any) -> None:
        """Buffer one event for the next flush. Never blocks on I/O."""
        ev = (
            time.time(),
            kind,
            fields.get("purpose"),
            fields.get("model"),
            fields.get("latency_ms"),
            fields.get("tokens_in"),
            fields.get("tokens_out"),
            fields.get("cache_hit_tokens"),
            fields.get("cache_miss_tokens"),
            fields.get("emoji"),
            fields.get("skip_reason"),
            fields.get("error_msg"),
            fields.get("score"),
            fields.get("bot_id"),
        )
        with self._queue_lock:
            self._queue.append(ev)

    # ── Async flush — drains queue to disk in one batch insert ─────────

    async def flush(self) -> int:
        """Drain the queue to SQLite. Returns the number of events written.

        executemany() so a backlog of N events is one round-trip instead
        of N. Failure isn't retried — the events stay in the queue if the
        flusher crashes, but the queue is bounded so a permanent failure
        can't grow without limit.
        """
        with self._queue_lock:
            if not self._queue:
                return 0
            batch = list(self._queue)
            self._queue.clear()
        await self._ensure_initialized()
        try:
            async with db_session(self) as db:
                await db.executemany(
                    """INSERT INTO metric_events
                       (occurred_at, kind, purpose, model, latency_ms,
                        tokens_in, tokens_out, cache_hit_tokens,
                        cache_miss_tokens, emoji, skip_reason, error_msg,
                        score, bot_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    batch,
                )
                await db.commit()
            return len(batch)
        except Exception as e:
            logger.error(f"MetricsLog flush failed: {e}")
            return 0

    # ── Async windowed aggregation for the dashboard ───────────────────

    async def query_window(self, seconds: float) -> dict:
        """Aggregate events from the last `seconds` into the same dict
        shape as `MetricsCollector.get_all_stats()` consumers. Falls back
        to zeros on schema-empty windows so the template is identical.
        """
        await self._ensure_initialized()
        cutoff = time.time() - max(0.0, float(seconds))

        async with db_session(self) as db:
            llm = await self._aggregate_llm(db, cutoff, KIND_LLM_OK, KIND_LLM_ERR)
            dt = await self._aggregate_llm(db, cutoff, KIND_DT_OK, KIND_DT_ERR)
            reactor = await self._aggregate_reactor(db, cutoff)
            request_count = await self._count(db, cutoff, KIND_REQUEST)

        return {
            "window_seconds": seconds,
            "llm": llm,
            "deep_think": dt,
            "reactor": reactor,
            "request_count": request_count,
        }

    async def _aggregate_llm(
        self, db, cutoff: float, ok_kind: str, err_kind: str,
    ) -> dict:
        cursor = await db.execute(
            f"""SELECT
                   SUM(CASE WHEN kind = ? THEN 1 ELSE 0 END) AS successes,
                   SUM(CASE WHEN kind = ? THEN 1 ELSE 0 END) AS errors,
                   SUM(COALESCE(tokens_in,  0)) AS tokens_in,
                   SUM(COALESCE(tokens_out, 0)) AS tokens_out,
                   AVG(CASE WHEN kind = ? THEN latency_ms END) AS avg_latency_ms,
                   MAX(occurred_at) AS last_at,
                   SUM(COALESCE(cache_hit_tokens,  0)) AS cache_hit_tokens,
                   SUM(COALESCE(cache_miss_tokens, 0)) AS cache_miss_tokens
               FROM metric_events
               WHERE kind IN (?, ?) AND occurred_at >= ?""",
            (ok_kind, err_kind, ok_kind, ok_kind, err_kind, cutoff),
        )
        row = await cursor.fetchone()
        successes = (row[0] or 0) if row else 0
        errors = (row[1] or 0) if row else 0
        calls = successes + errors

        # Per-purpose / per-model breakdowns
        by_purpose: dict = {}
        by_model: dict = {}
        cursor = await db.execute(
            """SELECT purpose, COUNT(*) FROM metric_events
               WHERE kind IN (?, ?) AND occurred_at >= ? AND purpose IS NOT NULL
               GROUP BY purpose""",
            (ok_kind, err_kind, cutoff),
        )
        for purpose, n in await cursor.fetchall():
            if purpose:
                by_purpose[purpose] = n
        cursor = await db.execute(
            """SELECT model, COUNT(*) FROM metric_events
               WHERE kind IN (?, ?) AND occurred_at >= ? AND model IS NOT NULL AND model != ''
               GROUP BY model""",
            (ok_kind, err_kind, cutoff),
        )
        for model, n in await cursor.fetchall():
            if model:
                by_model[model] = n

        # Last error message in the window (for the "what's broken" header)
        cursor = await db.execute(
            """SELECT error_msg, occurred_at FROM metric_events
               WHERE kind = ? AND occurred_at >= ? AND error_msg IS NOT NULL
               ORDER BY occurred_at DESC LIMIT 1""",
            (err_kind, cutoff),
        )
        err_row = await cursor.fetchone()

        cache_hit = (row[6] or 0) if row else 0
        cache_miss = (row[7] or 0) if row else 0
        cache_total = cache_hit + cache_miss
        return {
            "calls": calls,
            "successes": successes,
            "errors": errors,
            "success_rate": (
                f"{(successes / calls * 100):.1f}%" if calls else "—"
            ),
            "tokens_in": row[2] if row else 0,
            "tokens_out": row[3] if row else 0,
            "avg_latency_ms": row[4] if row else None,
            "last_call_at": row[5] if row else None,
            "last_error_at": err_row[1] if err_row else None,
            "last_error_msg": err_row[0] if err_row else None,
            "by_purpose": by_purpose,
            "by_model": by_model,
            "cache_hit_tokens": cache_hit,
            "cache_miss_tokens": cache_miss,
            "cache_hit_ratio": (cache_hit / cache_total) if cache_total else 0.0,
        }

    async def _aggregate_reactor(self, db, cutoff: float) -> dict:
        cursor = await db.execute(
            """SELECT kind, skip_reason, COUNT(*) FROM metric_events
               WHERE kind LIKE 'reactor_%' AND occurred_at >= ?
               GROUP BY kind, skip_reason""",
            (cutoff,),
        )
        rows = await cursor.fetchall()
        out: dict = {
            "evaluations": 0,
            "reactions_sent": 0,
            "responses_triggered": 0,
            "errors": 0,
        }
        # Pre-seed the known reasons so the dashboard shows explicit zeros.
        for reason in REACTOR_SKIP_REASONS:
            out[f"skipped_{reason}"] = 0
        for kind, reason, n in rows:
            if kind == KIND_REACTOR_EVAL:
                out["evaluations"] += n
            elif kind == KIND_REACTOR_REACT:
                out["reactions_sent"] += n
            elif kind == KIND_REACTOR_RESPONSE:
                out["responses_triggered"] += n
            elif kind == KIND_REACTOR_ERROR:
                out["errors"] += n
            elif kind == KIND_REACTOR_SKIP:
                # Accumulate whatever reason the row carries, seeded or not.
                # The old `if key in out` dropped unrecognised reasons on the
                # floor, so a reason added to the reactor without being
                # registered here vanished from the dashboard with no error.
                key = f"skipped_{reason}" if reason else "skipped_unknown"
                out[key] = out.get(key, 0) + n

        cursor = await db.execute(
            """SELECT emoji, COUNT(*) FROM metric_events
               WHERE kind = ? AND occurred_at >= ? AND emoji IS NOT NULL
               GROUP BY emoji ORDER BY COUNT(*) DESC LIMIT 8""",
            (KIND_REACTOR_REACT, cutoff),
        )
        out["top_emojis"] = [
            [emoji, n] for emoji, n in await cursor.fetchall()
        ]

        # Self-reported worthiness distribution across every emoji_react
        # pick — both the ones that landed and the ones a post-LLM brake
        # dropped. This is the readout that turns reactor_min_score from a
        # guess into a percentile: pick the threshold off the histogram
        # rather than cold, since self-scores cluster high.
        cursor = await db.execute(
            """SELECT score, COUNT(*) FROM metric_events
               WHERE kind IN (?, ?) AND occurred_at >= ? AND score IS NOT NULL
               GROUP BY score ORDER BY score""",
            (KIND_REACTOR_REACT, KIND_REACTOR_SKIP, cutoff),
        )
        score_rows = await cursor.fetchall()
        out["by_score"] = [[int(s), n] for s, n in score_rows]
        scored_total = sum(n for _, n in score_rows)
        out["avg_score"] = (
            sum(int(s) * n for s, n in score_rows) / scored_total
            if scored_total else None
        )

        cursor = await db.execute(
            """SELECT MAX(occurred_at) FROM metric_events
               WHERE kind = ? AND occurred_at >= ?""",
            (KIND_REACTOR_REACT, cutoff),
        )
        row = await cursor.fetchone()
        out["last_reaction_at"] = (row[0] if row else None)
        return out

    @staticmethod
    async def _count(db, cutoff: float, kind: str) -> int:
        cursor = await db.execute(
            "SELECT COUNT(*) FROM metric_events WHERE kind = ? AND occurred_at >= ?",
            (kind, cutoff),
        )
        row = await cursor.fetchone()
        return (row[0] or 0) if row else 0

    # ── Async prune ────────────────────────────────────────────────────

    async def prune(self, retention_seconds: float) -> int:
        """Delete events older than the retention floor. Run periodically
        to keep the table from growing without bound. Returns the deleted
        row count for logging."""
        await self._ensure_initialized()
        cutoff = time.time() - max(0.0, float(retention_seconds))
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM metric_events WHERE occurred_at < ?", (cutoff,),
            )
            await db.commit()
            return cursor.rowcount or 0


# ── Singleton + window selectors for the dashboard ───────────────────────

_log_instance: Optional[MetricsLog] = None
_log_instance_lock = threading.Lock()


def get_metrics_log(db_path: str = "data/watchlist.db") -> MetricsLog:
    global _log_instance
    if _log_instance is None:
        with _log_instance_lock:
            if _log_instance is None:
                _log_instance = MetricsLog(db_path)
    return _log_instance


# Window definitions surfaced in the dashboard. Keep these in sync with
# the radio-group labels in dashboard.html.
WINDOWS: dict[str, int] = {
    "24h": 24 * 3600,
    "7d":  7 * 24 * 3600,
    "30d": 30 * 24 * 3600,
}
DEFAULT_WINDOW = "24h"


def parse_window(name: Optional[str]) -> tuple[str, int]:
    """Resolve a window name from URL/query input to (canonical_name, seconds).
    Unknown values fall back to the default."""
    key = (name or DEFAULT_WINDOW).strip().lower()
    if key not in WINDOWS:
        key = DEFAULT_WINDOW
    return key, WINDOWS[key]
