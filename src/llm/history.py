"""
Per-context rolling conversation history backed by SQLite.

Scoping rules for `context_key`:
  * DM            → sha256 of sender phone (same effect as "per user")
  * Group         → the raw group_id (shared thread for the group)

`sender_tail` is retained so replayed turns can be attributed back to their
speaker — useful in group threads where multiple users contribute.

Pruning is triggered on every write:
  * per-context row cap: latest `turns_per_user * 2` rows kept
  * global age cap:      rows older than `llm_retention_days` deleted
"""

import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 7


class ConversationHistory:
    """
    Rows: (id, context_key, sender_tail, role, content, created_at).

    `turns_per_user` means role-pairs, so 6 → up to 12 rows per context.
    """

    def __init__(
        self,
        db_path: str = "data/watchlist.db",
        turns_per_user: int = 6,
        settings_store=None,
        name_registry=None,
    ):
        self.db_path = Path(db_path)
        self.turns_per_user = turns_per_user
        self.settings_store = settings_store
        # Optional: maps user_hash → display name for group attribution.
        # When unset, attribution falls back to `[...tail]`.
        self.name_registry = name_registry
        self._initialized = False

    def _retention_seconds(self) -> float:
        days = DEFAULT_RETENTION_DAYS
        if self.settings_store is not None:
            try:
                raw = self.settings_store.get("llm_retention_days")
                if raw is not None:
                    days = max(1, int(raw))
            except (TypeError, ValueError):
                pass
        return days * 86400.0

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            # Fresh installs get the new schema directly.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_turns (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT,
                    context_key TEXT,
                    sender_tail TEXT,
                    role TEXT NOT NULL,
                    content TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            # Upgrade path for existing installs: add columns if missing.
            cursor = await db.execute("PRAGMA table_info(conversation_turns)")
            cols = {row[1] for row in await cursor.fetchall()}
            if "context_key" not in cols:
                await db.execute("ALTER TABLE conversation_turns ADD COLUMN context_key TEXT")
            if "sender_tail" not in cols:
                await db.execute("ALTER TABLE conversation_turns ADD COLUMN sender_tail TEXT")
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_ctx_time "
                "ON conversation_turns(context_key, created_at)"
            )
            # Per-context rolling summary. `summary_through_id` is the highest
            # conversation_turns.id that's already been folded in — newer
            # turns above that are still verbatim in conversation_turns and
            # haven't been compressed yet.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    context_key TEXT PRIMARY KEY,
                    summary TEXT NOT NULL,
                    summary_through_id INTEGER NOT NULL DEFAULT 0,
                    turns_summarized INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()
        self._initialized = True

    async def get_summary(self, context_key: str) -> Optional[dict]:
        """Return the rolling summary for a context, or None if absent.

        Shape: {summary: str, summary_through_id: int, turns_summarized: int,
                updated_at: float}.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT summary, summary_through_id, turns_summarized, updated_at
                   FROM conversation_summaries WHERE context_key = ?""",
                (context_key,),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        return {
            "summary": row[0],
            "summary_through_id": row[1],
            "turns_summarized": row[2],
            "updated_at": row[3],
        }

    async def upsert_summary(
        self,
        context_key: str,
        summary: str,
        summary_through_id: int,
        turns_summarized: int,
    ) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO conversation_summaries
                   (context_key, summary, summary_through_id, turns_summarized, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(context_key) DO UPDATE SET
                     summary = excluded.summary,
                     summary_through_id = excluded.summary_through_id,
                     turns_summarized = excluded.turns_summarized,
                     updated_at = excluded.updated_at""",
                (context_key, summary, summary_through_id, turns_summarized, time.time()),
            )
            await db.commit()

    async def turns_to_summarize(
        self, context_key: str, summary_through_id: int, keep_recent: int
    ) -> list[dict]:
        """Return turns that should be folded into the summary.

        Excludes the most recent `keep_recent` rows (kept verbatim) and rows
        already covered by the existing summary. Used by the summarizer to
        decide what fresh material to feed the LLM.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT MAX(id) FROM conversation_turns WHERE context_key = ?",
                (context_key,),
            )
            max_row = await cursor.fetchone()
            max_id = (max_row[0] or 0) if max_row else 0
            if max_id == 0:
                return []
            # Boundary: anything with id > max_id - keep_recent stays verbatim.
            recent_floor = max_id - keep_recent
            if recent_floor <= summary_through_id:
                return []
            cursor = await db.execute(
                """SELECT id, role, content, sender_tail, user_hash
                   FROM conversation_turns
                   WHERE context_key = ?
                     AND id > ?
                     AND id <= ?
                   ORDER BY id ASC""",
                (context_key, summary_through_id, recent_floor),
            )
            rows = await cursor.fetchall()
        return [
            {"id": r[0], "role": r[1], "content": r[2],
             "sender_tail": r[3], "user_hash": r[4]}
            for r in rows
        ]

    async def load(
        self,
        context_key: str,
        turns_per_user: Optional[int] = None,
        attribute_senders: bool = False,
    ) -> list[dict]:
        """Return the last 2*N rows for this context in chronological order.

        When `attribute_senders` is True, user messages are prefixed with
        `[...tail]` so the LLM can tell speakers apart in a group thread.
        """
        await self._ensure_initialized()
        n = (turns_per_user or self.turns_per_user) * 2
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT role, content, sender_tail, user_hash
                   FROM conversation_turns
                   WHERE context_key = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (context_key, n),
            )
            rows = await cursor.fetchall()

        turns: list[dict] = []
        for role, content, sender_tail, user_hash in reversed(rows):
            text = content
            if attribute_senders and role == "user":
                label = self._attribution_label(user_hash, sender_tail)
                if label:
                    text = f"[{label}] {content}"
            turns.append({"role": role, "content": text})
        return turns

    def _attribution_label(
        self, user_hash: Optional[str], sender_tail: Optional[str]
    ) -> Optional[str]:
        """Return the bracket label for a user message in group playback.

        Prefers a registered name from the registry; falls back to the
        last-4-digits tail. Returns None when nothing's available so the
        caller can skip attribution entirely.
        """
        if self.name_registry is not None:
            try:
                return self.name_registry.display_name_sync(
                    user_hash=user_hash, tail=sender_tail
                )
            except Exception:
                pass
        if sender_tail:
            return f"...{sender_tail}"
        return None

    async def append(
        self,
        context_key: str,
        role: str,
        content: str,
        user_hash: Optional[str] = None,
        sender_tail: Optional[str] = None,
    ) -> None:
        """Insert one turn and prune by row-cap (per context) and age (global)."""
        await self._ensure_initialized()

        now = time.time()
        age_cutoff = now - self._retention_seconds()
        max_rows = self.turns_per_user * 2

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO conversation_turns
                   (user_hash, context_key, sender_tail, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_hash, context_key, sender_tail, role, content, now),
            )
            # Age-based purge (whole table)
            await db.execute(
                "DELETE FROM conversation_turns WHERE created_at < ?",
                (age_cutoff,),
            )
            # Per-context row cap
            await db.execute(
                """DELETE FROM conversation_turns
                   WHERE context_key = ?
                     AND id NOT IN (
                         SELECT id FROM conversation_turns
                         WHERE context_key = ?
                         ORDER BY created_at DESC, id DESC
                         LIMIT ?
                     )""",
                (context_key, context_key, max_rows),
            )
            await db.commit()

    async def clear(self, context_key: str) -> int:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM conversation_turns WHERE context_key = ?", (context_key,)
            )
            # The summary captures pruned content; clearing the conversation
            # has to wipe it too or "forget" leaks via a stale recap.
            await db.execute(
                "DELETE FROM conversation_summaries WHERE context_key = ?",
                (context_key,),
            )
            await db.commit()
            return cursor.rowcount
