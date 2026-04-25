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
    ):
        self.db_path = Path(db_path)
        self.turns_per_user = turns_per_user
        self.settings_store = settings_store
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
            await db.commit()
        self._initialized = True

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
                """SELECT role, content, sender_tail FROM conversation_turns
                   WHERE context_key = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (context_key, n),
            )
            rows = await cursor.fetchall()

        turns: list[dict] = []
        for role, content, sender_tail in reversed(rows):
            text = content
            if attribute_senders and role == "user" and sender_tail:
                text = f"[...{sender_tail}] {content}"
            turns.append({"role": role, "content": text})
        return turns

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
            await db.commit()
            return cursor.rowcount
