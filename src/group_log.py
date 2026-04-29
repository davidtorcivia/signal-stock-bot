"""
Rolling log of inbound group chat messages.

Used as LLM context when !ask is invoked inside a group. Only inbound
user messages are stored — no bot replies. Sender is masked to the last
4 digits before being handed to the LLM.
"""

import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .database import db_session

logger = logging.getLogger(__name__)

# Hard cap per group — protects the DB from runaway chat volume.
# The admin-configurable setting (group_context_messages) controls how many
# of these are actually fed to the LLM.
MAX_ROWS_PER_GROUP = 500

# Default retention when no setting is provided. Used as a safety floor.
DEFAULT_RETENTION_DAYS = 7

# Sentinel `sender` value for the bot's own replies. Stored in the same
# `sender` column as user phones so group_context renders chronologically;
# label resolution layers (NameRegistry.label_for, the per-command
# `_sender_label` helpers) translate this back to the configured bot name.
# Chosen to be unambiguously not a phone number.
BOT_SENDER = "__bot__"


class GroupMessageLog:
    """SQLite-backed rolling log of group chat messages.

    Two layers of pruning run on every append:
      1. Row cap per group (hard ceiling — `MAX_ROWS_PER_GROUP`).
      2. Age cap across the whole table (soft — `llm_retention_days` from
         the settings store, defaulting to 7).
    """

    def __init__(
        self,
        db_path: str = "data/watchlist.db",
        settings_store=None,
        enricher=None,
    ):
        self.db_path = Path(db_path)
        self.settings_store = settings_store
        # Optional async callable: text -> expanded text. Used to inline tweet
        # content (and any other URL types added later) so LLM context contains
        # the substance of shared links, not just the URLs.
        self.enricher = enricher
        self._initialized = False

    def _retention_seconds(self) -> float:
        if self.settings_store is None:
            return DEFAULT_RETENTION_DAYS * 86400.0
        return self.settings_store.get_int(
            "llm_retention_days", DEFAULT_RETENTION_DAYS, min_value=1,
        ) * 86400.0

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            from .database import apply_db_pragmas
            await apply_db_pragmas(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS group_messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    group_id TEXT NOT NULL,
                    sender TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_group_msgs_time "
                "ON group_messages(group_id, created_at)"
            )
            await db.commit()
        self._initialized = True

    async def append(self, group_id: str, sender: str, text: str) -> None:
        """Record a message and prune by both row-cap and age."""
        if not group_id or not text:
            return

        # Inline-expand any embedded link content (tweets, etc.) so LLM context
        # later sees the substance of shared links rather than opaque URLs.
        if self.enricher is not None:
            try:
                text = await self.enricher.expand(text)
            except Exception as e:
                logger.debug(f"Enrichment failed (storing raw): {e}")

        await self._ensure_initialized()
        now = time.time()
        age_cutoff = now - self._retention_seconds()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO group_messages (group_id, sender, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (group_id, sender, text, now),
            )
            # Age-based purge (applies to the whole table)
            await db.execute(
                "DELETE FROM group_messages WHERE created_at < ?",
                (age_cutoff,),
            )
            # Row-cap purge for this group
            await db.execute(
                """DELETE FROM group_messages
                   WHERE group_id = ?
                     AND id NOT IN (
                         SELECT id FROM group_messages
                         WHERE group_id = ?
                         ORDER BY created_at DESC, id DESC
                         LIMIT ?
                     )""",
                (group_id, group_id, MAX_ROWS_PER_GROUP),
            )
            await db.commit()

    async def find_recent_sender_by_tail(
        self, group_id: str, tail: str, *, limit: int = 200
    ) -> Optional[str]:
        """Find the most recent phone number in `group_id` whose last
        4 chars match `tail`. Returns None when no match found.

        Used by `predict_for` to attribute a prediction to an
        unregistered user the LLM only knows by their `...4810` tail
        from the group_context block. We scan inbound messages (the
        bot's own posts have BOT_SENDER as their sender, so they're
        naturally excluded).
        """
        tail = (tail or "").strip()
        if not tail or not group_id:
            return None
        async with db_session(self) as db:
            cursor = await db.execute(
                """SELECT sender FROM group_messages
                   WHERE group_id = ?
                     AND sender != ?
                     AND substr(sender, -4) = ?
                   ORDER BY created_at DESC
                   LIMIT 1""",
                (group_id, BOT_SENDER, tail),
            )
            row = await cursor.fetchone()
        return row[0] if row else None

    async def append_bot(self, group_id: str, text: str) -> None:
        """Record one of the bot's own replies under the BOT_SENDER sentinel.

        Same row-cap and age purges as `append`, but skips the enricher —
        bot answers come out of the writer LLM already enriched, and any
        URLs they contain were the model's choice; re-expanding could
        bloat or distort the message that actually went to the chat.
        """
        if not group_id or not text:
            return
        await self._ensure_initialized()
        now = time.time()
        age_cutoff = now - self._retention_seconds()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "INSERT INTO group_messages (group_id, sender, text, created_at) "
                "VALUES (?, ?, ?, ?)",
                (group_id, BOT_SENDER, text, now),
            )
            await db.execute(
                "DELETE FROM group_messages WHERE created_at < ?",
                (age_cutoff,),
            )
            await db.execute(
                """DELETE FROM group_messages
                   WHERE group_id = ?
                     AND id NOT IN (
                         SELECT id FROM group_messages
                         WHERE group_id = ?
                         ORDER BY created_at DESC, id DESC
                         LIMIT ?
                     )""",
                (group_id, group_id, MAX_ROWS_PER_GROUP),
            )
            await db.commit()

    async def recent(self, group_id: str, limit: int, exclude_last: int = 0) -> list[dict]:
        """
        Return up to `limit` recent messages for a group in chronological order.

        `exclude_last` skips the N newest rows — used by !ask so the current
        command itself isn't echoed into its own context.
        """
        if limit <= 0 or not group_id:
            return []
        async with db_session(self) as db:
            cursor = await db.execute(
                """SELECT sender, text, created_at FROM group_messages
                   WHERE group_id = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ? OFFSET ?""",
                (group_id, limit, exclude_last),
            )
            rows = await cursor.fetchall()
        return [
            {"sender": r[0], "text": r[1], "created_at": r[2]}
            for r in reversed(rows)
        ]
