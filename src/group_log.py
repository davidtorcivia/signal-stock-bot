"""
Rolling log of inbound group chat messages.

Used as LLM context when !ask is invoked inside a group. Only inbound
user messages are stored — no bot replies. Sender is masked to the last
4 digits before being handed to the LLM.
"""

import logging
import time
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# Hard cap per group — protects the DB from runaway chat volume.
# The admin-configurable setting (group_context_messages) controls how many
# of these are actually fed to the LLM.
MAX_ROWS_PER_GROUP = 500

# Default retention when no setting is provided. Used as a safety floor.
DEFAULT_RETENTION_DAYS = 7


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

    async def recent(self, group_id: str, limit: int, exclude_last: int = 0) -> list[dict]:
        """
        Return up to `limit` recent messages for a group in chronological order.

        `exclude_last` skips the N newest rows — used by !ask so the current
        command itself isn't echoed into its own context.
        """
        if limit <= 0 or not group_id:
            return []
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
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
