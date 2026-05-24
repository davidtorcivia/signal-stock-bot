"""
SQLite-backed ContextPolicy store.

Two "default" rows seed on init:
  * default:group — fallback for any group not explicitly registered
  * default:dm    — fallback for any DM not explicitly registered

Resolution: explicit row wins, otherwise the matching default. Groups the bot
has never seen are auto-registered as empty stub rows on first sight (so the
admin can edit them from the UI).
"""

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import db_session

from .policy import ContextPolicy, MODE_ALLOW_ALL

logger = logging.getLogger(__name__)

DEFAULT_GROUP_KEY = "default:group"
DEFAULT_DM_KEY = "default:dm"


class ContextRegistry:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                from ..database import apply_db_pragmas
                await apply_db_pragmas(db)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS contexts (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        kind TEXT NOT NULL,
                        key TEXT NOT NULL UNIQUE,
                        label TEXT NOT NULL DEFAULT '',
                        command_mode TEXT NOT NULL DEFAULT 'allow_all',
                        commands TEXT NOT NULL DEFAULT '[]',
                        mcp_mode TEXT NOT NULL DEFAULT 'allow_all',
                        mcp_servers TEXT NOT NULL DEFAULT '[]',
                        system_prompt TEXT,
                        llm_intent INTEGER NOT NULL DEFAULT 0,
                        reactor_enabled INTEGER NOT NULL DEFAULT 1,
                        reactor_prompt TEXT,
                        natural_response INTEGER NOT NULL DEFAULT 0,
                        deep_think_enabled INTEGER NOT NULL DEFAULT 1,
                        memory_writes_enabled INTEGER NOT NULL DEFAULT 1,
                        reactor_memory_writes INTEGER NOT NULL DEFAULT 0,
                        first_seen REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                # Upgrade path for pre-existing installs — add columns if missing.
                cursor = await db.execute("PRAGMA table_info(contexts)")
                existing_cols = {r[1] for r in await cursor.fetchall()}
                if "llm_intent" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN llm_intent INTEGER NOT NULL DEFAULT 0"
                    )
                if "reactor_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN reactor_enabled INTEGER NOT NULL DEFAULT 1"
                    )
                if "reactor_prompt" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN reactor_prompt TEXT"
                    )
                if "natural_response" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN natural_response INTEGER NOT NULL DEFAULT 0"
                    )
                if "deep_think_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN deep_think_enabled INTEGER NOT NULL DEFAULT 1"
                    )
                if "memory_writes_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN memory_writes_enabled INTEGER NOT NULL DEFAULT 1"
                    )
                if "reactor_memory_writes" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN reactor_memory_writes INTEGER NOT NULL DEFAULT 0"
                    )
                if "default_bot_id" not in existing_cols:
                    # Multi-bot scoping: which bot answers in this
                    # context when no one is mentioned. NULL = fall back
                    # to BotRegistry.default_for_kind_sync(). Backfilled
                    # to the seeded sigil bot post-init.
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN default_bot_id INTEGER"
                    )
                if "transcript_logging_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN "
                        "transcript_logging_enabled INTEGER NOT NULL DEFAULT 0"
                    )
                if "history_turns_override" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN "
                        "history_turns_override INTEGER"
                    )
                if "purge_floor_at" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE contexts ADD COLUMN "
                        "purge_floor_at REAL"
                    )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_contexts_key ON contexts(key)"
                )
                now = time.time()
                for default_key, label in (
                    (DEFAULT_GROUP_KEY, "Default (groups)"),
                    (DEFAULT_DM_KEY, "Default (DMs)"),
                ):
                    await db.execute(
                        """INSERT OR IGNORE INTO contexts
                           (kind, key, label, command_mode, commands, mcp_mode,
                            mcp_servers, system_prompt, first_seen, updated_at)
                           VALUES ('default', ?, ?, 'allow_all', '[]', 'allow_all',
                                   '[]', NULL, ?, ?)""",
                        (default_key, label, now, now),
                    )
                await db.commit()
            self._initialized = True

    SELECT_COLS = (
        "id, kind, key, label, command_mode, commands, "
        "mcp_mode, mcp_servers, system_prompt, llm_intent"
    )

    _SELECT_FIELDS = (
        "id, kind, key, label, command_mode, commands, "
        "mcp_mode, mcp_servers, system_prompt, llm_intent, "
        "reactor_enabled, reactor_prompt, natural_response, "
        "deep_think_enabled, memory_writes_enabled, reactor_memory_writes, "
        "default_bot_id, transcript_logging_enabled, history_turns_override, "
        "purge_floor_at"
    )

    @staticmethod
    def _row_to_policy(row) -> ContextPolicy:
        return ContextPolicy(
            id=row[0],
            kind=row[1],
            key=row[2],
            label=row[3] or "",
            command_mode=row[4] or MODE_ALLOW_ALL,
            commands=json.loads(row[5] or "[]"),
            mcp_mode=row[6] or MODE_ALLOW_ALL,
            mcp_servers=json.loads(row[7] or "[]"),
            system_prompt=row[8],
            llm_intent=bool(row[9]) if len(row) > 9 else False,
            reactor_enabled=bool(row[10]) if len(row) > 10 and row[10] is not None else True,
            reactor_prompt=row[11] if len(row) > 11 else None,
            natural_response=bool(row[12]) if len(row) > 12 and row[12] is not None else False,
            deep_think_enabled=bool(row[13]) if len(row) > 13 and row[13] is not None else True,
            memory_writes_enabled=bool(row[14]) if len(row) > 14 and row[14] is not None else True,
            reactor_memory_writes=bool(row[15]) if len(row) > 15 and row[15] is not None else False,
            default_bot_id=row[16] if len(row) > 16 else None,
            transcript_logging_enabled=(
                bool(row[17]) if len(row) > 17 and row[17] is not None else False
            ),
            history_turns_override=(
                row[18] if len(row) > 18 and row[18] is not None else None
            ),
            purge_floor_at=(
                row[19] if len(row) > 19 and row[19] is not None else None
            ),
        )

    async def list(self) -> list[ContextPolicy]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_FIELDS}
                    FROM contexts
                    ORDER BY
                      CASE kind WHEN 'default' THEN 0 WHEN 'group' THEN 1 ELSE 2 END,
                      label, key"""
            )
            rows = await cursor.fetchall()
        return [self._row_to_policy(r) for r in rows]

    async def get(self, context_id: int) -> Optional[ContextPolicy]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM contexts WHERE id = ?",
                (context_id,),
            )
            row = await cursor.fetchone()
        return self._row_to_policy(row) if row else None

    async def get_by_key(self, key: str) -> Optional[ContextPolicy]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM contexts WHERE key = ?",
                (key,),
            )
            row = await cursor.fetchone()
        return self._row_to_policy(row) if row else None

    async def resolve(self, group_id: Optional[str], sender: str) -> ContextPolicy:
        """
        Return the policy that applies to this incoming message.

        For groups, a stub row is auto-created on first sight so the admin
        can edit it from the UI. For DMs, explicit rows are required —
        otherwise the DM default is returned (never auto-created).
        """
        await self._ensure_initialized()

        if group_id:
            hit = await self.get_by_key(group_id)
            if hit:
                return hit
            await self._auto_register_group(group_id)
            hit = await self.get_by_key(group_id)
            if hit:
                return hit
            fallback = await self.get_by_key(DEFAULT_GROUP_KEY)
        else:
            hit = await self.get_by_key(sender)
            if hit:
                return hit
            fallback = await self.get_by_key(DEFAULT_DM_KEY)

        if fallback:
            return fallback
        # Last-resort fully permissive — only hit if defaults were deleted.
        from .policy import PERMISSIVE
        return PERMISSIVE

    async def _auto_register_group(self, group_id: str) -> None:
        """Insert a blank stub for a group we've never seen."""
        now = time.time()
        try:
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """INSERT OR IGNORE INTO contexts
                       (kind, key, label, command_mode, commands, mcp_mode,
                        mcp_servers, system_prompt, first_seen, updated_at)
                       VALUES ('group', ?, '', 'allow_all', '[]', 'allow_all',
                               '[]', NULL, ?, ?)""",
                    (group_id, now, now),
                )
                await db.commit()
            logger.info(f"Auto-registered context for new group {group_id[:10]}...")
        except Exception as e:
            logger.error(f"Failed to auto-register group context: {e}")

    async def upsert(self, policy: ContextPolicy) -> int:
        await self._ensure_initialized()
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            if policy.id is None:
                cursor = await db.execute(
                    """INSERT INTO contexts
                       (kind, key, label, command_mode, commands, mcp_mode,
                        mcp_servers, system_prompt, llm_intent,
                        reactor_enabled, reactor_prompt, natural_response,
                        deep_think_enabled, memory_writes_enabled,
                        reactor_memory_writes, default_bot_id,
                        transcript_logging_enabled, history_turns_override,
                        first_seen, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        policy.kind,
                        policy.key,
                        policy.label,
                        policy.command_mode,
                        json.dumps(policy.commands),
                        policy.mcp_mode,
                        json.dumps(policy.mcp_servers),
                        policy.system_prompt or None,
                        1 if policy.llm_intent else 0,
                        1 if policy.reactor_enabled else 0,
                        policy.reactor_prompt or None,
                        1 if policy.natural_response else 0,
                        1 if policy.deep_think_enabled else 0,
                        1 if policy.memory_writes_enabled else 0,
                        1 if policy.reactor_memory_writes else 0,
                        policy.default_bot_id,
                        1 if policy.transcript_logging_enabled else 0,
                        policy.history_turns_override,
                        now,
                        now,
                    ),
                )
                await db.commit()
                return cursor.lastrowid or 0
            else:
                await db.execute(
                    """UPDATE contexts SET
                           label = ?, command_mode = ?, commands = ?,
                           mcp_mode = ?, mcp_servers = ?, system_prompt = ?,
                           llm_intent = ?, reactor_enabled = ?,
                           reactor_prompt = ?, natural_response = ?,
                           deep_think_enabled = ?,
                           memory_writes_enabled = ?,
                           reactor_memory_writes = ?,
                           default_bot_id = ?,
                           transcript_logging_enabled = ?,
                           history_turns_override = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (
                        policy.label,
                        policy.command_mode,
                        json.dumps(policy.commands),
                        policy.mcp_mode,
                        json.dumps(policy.mcp_servers),
                        policy.system_prompt or None,
                        1 if policy.llm_intent else 0,
                        1 if policy.reactor_enabled else 0,
                        policy.reactor_prompt or None,
                        1 if policy.natural_response else 0,
                        1 if policy.deep_think_enabled else 0,
                        1 if policy.memory_writes_enabled else 0,
                        1 if policy.reactor_memory_writes else 0,
                        policy.default_bot_id,
                        1 if policy.transcript_logging_enabled else 0,
                        policy.history_turns_override,
                        now,
                        policy.id,
                    ),
                )
                await db.commit()
                return policy.id

    async def set_purge_floor(self, context_id: int, ts: float) -> bool:
        """Stamp the purge floor on a context row.

        Read paths (history, group_log, summarizer) filter out anything
        with `created_at < purge_floor_at`. The floor is intentionally
        not exposed in the admin upsert path — only this method mutates
        it, so an admin-form bug can't accidentally clear or backdate it.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "UPDATE contexts SET purge_floor_at = ?, updated_at = ? "
                "WHERE id = ?",
                (ts, time.time(), context_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete(self, context_id: int) -> bool:
        """Delete a row. Default rows are protected — deleting them re-seeds."""
        await self._ensure_initialized()
        existing = await self.get(context_id)
        if not existing:
            return False
        if existing.kind == "default":
            return False
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM contexts WHERE id = ?", (context_id,))
            await db.commit()
            return cursor.rowcount > 0
