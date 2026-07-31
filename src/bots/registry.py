"""
SQLite-backed BotRegistry.

Owns two tables:
  * `bots` — one row per bot identity (slug, display_name, aliases,
    persona, signal_phone/url overrides, default-for-kind flags,
    deep_think mode + handoff prompt).
  * `bot_llm_settings` — composite-keyed settings store
    `(bot_id, role, key) -> value` with `role` in
    {'writer','reactor','deep_think'}. Reads in PR2 fall back from
    bot-scoped value -> global admin_settings -> code default.

On first init the registry seeds a `sigil` row using the current
`bot_name` admin setting (or the env-loaded Config default) so existing
single-bot installs keep working with no manual setup. Backfill of
`bot_id=<sigil.id>` on consumer tables is handled by
`backfill_consumer_tables()` — called once from `main.py` after every
store has run its own `_ensure_initialized`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import sqlite3
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import db_session
from .models import (
    AUDIO_PART_STYLE_INPUT_AUDIO,
    AUDIO_PART_STYLES,
    Bot,
    DEEP_THINK_MODE_REPLACE,
    DEEP_THINK_MODES,
)

logger = logging.getLogger(__name__)


SIGIL_SLUG = "sigil"

# Tables that get a bot-scoping column at PR1 time. Most use `bot_id`
# to mean "which bot owns/wrote this row." `contexts` is the exception:
# it stores `default_bot_id`, meaning "which bot answers in this
# context by default" — a different concept (assignment, not
# ownership). Listed in one place so PR1 doesn't sprinkle the knowledge
# across modules; extend as later PRs add bot-scoped tables.
CONSUMER_TABLES = {
    "contexts": "default_bot_id",
    "conversation_turns": "bot_id",
    "context_memories": "bot_id",
    "predictions": "bot_id",
    "portfolios": "bot_id",
    "metric_events": "bot_id",
}


class BotRegistry:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()
        # Hot-path cache. Bots change rarely (admin edit) and the
        # signal handler / dispatcher hits the registry on every
        # inbound message for alias matching and context-default
        # resolution. Sync read of the cache replaces a sqlite round
        # trip in those paths.
        self._cache: dict[int, Bot] = {}
        self._slug_index: dict[str, int] = {}
        self._cache_loaded = False

    # ------------------------------------------------------------------
    # Schema + seeding
    # ------------------------------------------------------------------

    async def _ensure_initialized(self, default_display_name: str = "Sigil") -> None:
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
                    CREATE TABLE IF NOT EXISTS bots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        aliases TEXT NOT NULL DEFAULT '[]',
                        persona TEXT,
                        signal_phone TEXT,
                        signal_api_url TEXT,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        default_for_dm INTEGER NOT NULL DEFAULT 0,
                        default_for_group INTEGER NOT NULL DEFAULT 0,
                        deep_think_mode TEXT NOT NULL DEFAULT 'replace',
                        deep_think_handoff_prompt TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bot_llm_settings (
                        bot_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (bot_id, role, key),
                        FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bots_enabled ON bots(enabled)"
                )
                # Upgrade path: add columns missing on pre-existing installs.
                cursor = await db.execute("PRAGMA table_info(bots)")
                existing_cols = {r[1] for r in await cursor.fetchall()}
                if "vision_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "vision_enabled INTEGER NOT NULL DEFAULT 0"
                    )
                if "deep_think_enabled" not in existing_cols:
                    # Default 1 (= on) so existing installs keep their
                    # prior behavior; admins can flip individual bots off
                    # from /admin/bots/<id> without affecting globals.
                    await db.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "deep_think_enabled INTEGER NOT NULL DEFAULT 1"
                    )
                if "audio_enabled" not in existing_cols:
                    await db.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "audio_enabled INTEGER NOT NULL DEFAULT 0"
                    )
                if "audio_part_style" not in existing_cols:
                    # Text rather than a CHECK constraint: SQLite can't
                    # add one via ALTER, and _row_to_bot normalizes an
                    # unrecognized value back to the default anyway.
                    await db.execute(
                        "ALTER TABLE bots ADD COLUMN audio_part_style TEXT"
                    )
                if "routing_blurb" not in existing_cols:
                    # Nullable: reactor multi-bot routing falls back to
                    # the persona's first sentence when blurb is unset.
                    await db.execute(
                        "ALTER TABLE bots ADD COLUMN routing_blurb TEXT"
                    )
                # Seed Sigil if the table is empty. Aliases default to a
                # single-element list with the display name; admin can
                # add more from /admin/bots in PR3.
                cursor = await db.execute("SELECT COUNT(*) FROM bots")
                row = await cursor.fetchone()
                if row and row[0] == 0:
                    name = (default_display_name or "Sigil").strip() or "Sigil"
                    now = time.time()
                    await db.execute(
                        """
                        INSERT INTO bots (
                            slug, display_name, aliases, persona,
                            signal_phone, signal_api_url, enabled,
                            default_for_dm, default_for_group,
                            deep_think_mode, deep_think_handoff_prompt,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, NULL, NULL, NULL, 1, 1, 1,
                                  'replace', NULL, ?, ?)
                        """,
                        (SIGIL_SLUG, name, json.dumps([name]), now, now),
                    )
                    logger.info(
                        f"BotRegistry seeded default bot {SIGIL_SLUG!r} "
                        f"(display={name!r})"
                    )
                await db.commit()
            self._initialized = True
            await self._load_cache()

    async def _load_cache(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM bots ORDER BY id"
            )
            rows = await cursor.fetchall()
        # Build into local dicts then atomic rebind. Without this,
        # concurrent get_sync / default_for_kind_sync calls (signal
        # handler thread) would briefly observe an empty cache while
        # the loop is running, returning None and silently dropping
        # bot resolution for any inbound message that lands during an
        # admin edit.
        new_cache: dict[int, Bot] = {}
        new_slug_index: dict[str, int] = {}
        for r in rows:
            bot = self._row_to_bot(r)
            if bot.id is not None:
                new_cache[bot.id] = bot
                new_slug_index[bot.slug] = bot.id
        self._cache = new_cache
        self._slug_index = new_slug_index
        self._cache_loaded = True

    def warm_sync(self, default_display_name: str = "Sigil") -> None:
        """Sync table creation + seed + cache load.

        Mirrors `NameRegistry._warm_cache_sync` — main.py uses it during
        the synchronous wiring phase (before the event loop starts) so
        LLMClient / DeepThinkClient can be constructed with the seeded
        bot's id. Idempotent: re-running is safe and effectively
        free once the table is populated. The async `_ensure_initialized`
        path still works alongside this for callers that don't run
        through main.py (tests, ad-hoc scripts).
        """
        if self._cache_loaded:
            return
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute("PRAGMA synchronous=NORMAL")
                conn.execute("PRAGMA foreign_keys=ON")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bots (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        slug TEXT NOT NULL UNIQUE,
                        display_name TEXT NOT NULL,
                        aliases TEXT NOT NULL DEFAULT '[]',
                        persona TEXT,
                        signal_phone TEXT,
                        signal_api_url TEXT,
                        enabled INTEGER NOT NULL DEFAULT 1,
                        default_for_dm INTEGER NOT NULL DEFAULT 0,
                        default_for_group INTEGER NOT NULL DEFAULT 0,
                        deep_think_mode TEXT NOT NULL DEFAULT 'replace',
                        deep_think_handoff_prompt TEXT,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS bot_llm_settings (
                        bot_id INTEGER NOT NULL,
                        role TEXT NOT NULL,
                        key TEXT NOT NULL,
                        value TEXT NOT NULL,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (bot_id, role, key),
                        FOREIGN KEY (bot_id) REFERENCES bots(id) ON DELETE CASCADE
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_bots_enabled ON bots(enabled)"
                )
                # Upgrade path mirrors _ensure_initialized so warm_sync
                # leaves the schema in the same shape regardless of which
                # init path ran first.
                existing_cols = {
                    r[1] for r in conn.execute("PRAGMA table_info(bots)")
                }
                if "vision_enabled" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "vision_enabled INTEGER NOT NULL DEFAULT 0"
                    )
                if "deep_think_enabled" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "deep_think_enabled INTEGER NOT NULL DEFAULT 1"
                    )
                if "audio_enabled" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE bots ADD COLUMN "
                        "audio_enabled INTEGER NOT NULL DEFAULT 0"
                    )
                if "audio_part_style" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE bots ADD COLUMN audio_part_style TEXT"
                    )
                if "routing_blurb" not in existing_cols:
                    conn.execute(
                        "ALTER TABLE bots ADD COLUMN routing_blurb TEXT"
                    )
                cur = conn.execute("SELECT COUNT(*) FROM bots")
                row = cur.fetchone()
                if row and row[0] == 0:
                    name = (default_display_name or "Sigil").strip() or "Sigil"
                    now = time.time()
                    conn.execute(
                        """INSERT INTO bots (
                            slug, display_name, aliases, persona,
                            signal_phone, signal_api_url, enabled,
                            default_for_dm, default_for_group,
                            deep_think_mode, deep_think_handoff_prompt,
                            created_at, updated_at
                        ) VALUES (?, ?, ?, NULL, NULL, NULL, 1, 1, 1,
                                  'replace', NULL, ?, ?)""",
                        (SIGIL_SLUG, name, json.dumps([name]), now, now),
                    )
                    logger.info(
                        f"BotRegistry.warm_sync seeded {SIGIL_SLUG!r} "
                        f"(display={name!r})"
                    )
                conn.commit()
                cur = conn.execute(
                    f"SELECT {self._SELECT_FIELDS} FROM bots ORDER BY id"
                )
                rows = cur.fetchall()
            # Atomic rebind to keep concurrent sync readers consistent.
            new_cache: dict[int, Bot] = {}
            new_slug_index: dict[str, int] = {}
            for r in rows:
                bot = self._row_to_bot(r)
                if bot.id is not None:
                    new_cache[bot.id] = bot
                    new_slug_index[bot.slug] = bot.id
            self._cache = new_cache
            self._slug_index = new_slug_index
            self._cache_loaded = True
            # Mark the async initializer too: schema is already in place
            # so `_ensure_initialized` becomes a no-op on its happy path.
            self._initialized = True
        except Exception as e:
            logger.warning(f"BotRegistry.warm_sync failed: {e}")

    # ------------------------------------------------------------------
    # Row mapping
    # ------------------------------------------------------------------

    _SELECT_FIELDS = (
        "id, slug, display_name, aliases, persona, "
        "signal_phone, signal_api_url, enabled, "
        "default_for_dm, default_for_group, "
        "deep_think_mode, deep_think_handoff_prompt, "
        "created_at, updated_at, vision_enabled, deep_think_enabled, "
        "routing_blurb, audio_enabled, audio_part_style"
    )

    @staticmethod
    def _row_to_bot(row) -> Bot:
        try:
            aliases = json.loads(row[3] or "[]")
            if not isinstance(aliases, list):
                aliases = []
        except Exception:
            aliases = []
        return Bot(
            id=row[0],
            slug=row[1],
            display_name=row[2] or "",
            aliases=aliases,
            persona=row[4],
            signal_phone=row[5],
            signal_api_url=row[6],
            enabled=bool(row[7]),
            default_for_dm=bool(row[8]),
            default_for_group=bool(row[9]),
            deep_think_mode=row[10] or DEEP_THINK_MODE_REPLACE,
            deep_think_handoff_prompt=row[11],
            created_at=row[12] or 0.0,
            updated_at=row[13] or 0.0,
            vision_enabled=bool(row[14]) if len(row) > 14 and row[14] is not None else False,
            deep_think_enabled=(
                bool(row[15]) if len(row) > 15 and row[15] is not None else True
            ),
            routing_blurb=(row[16] if len(row) > 16 else None),
            audio_enabled=(
                bool(row[17]) if len(row) > 17 and row[17] is not None else False
            ),
            # Normalize here so a hand-edited row (or a NULL from the
            # ALTER) can never put an unknown part shape in front of the
            # payload builder.
            audio_part_style=(
                row[18]
                if len(row) > 18 and row[18] in AUDIO_PART_STYLES
                else AUDIO_PART_STYLE_INPUT_AUDIO
            ),
        )

    # ------------------------------------------------------------------
    # Reads (sync from cache + async fallback)
    # ------------------------------------------------------------------

    def get_sync(self, bot_id: int) -> Optional[Bot]:
        """Cache-only sync read. Returns None if cache is cold or the bot
        isn't in it. Used in hot paths that can't await."""
        if not self._cache_loaded:
            return None
        return self._cache.get(bot_id)

    def get_by_slug_sync(self, slug: str) -> Optional[Bot]:
        if not self._cache_loaded:
            return None
        bot_id = self._slug_index.get(slug)
        return self._cache.get(bot_id) if bot_id is not None else None

    def list_sync(self) -> list[Bot]:
        if not self._cache_loaded:
            return []
        return list(self._cache.values())

    def default_for_kind_sync(self, kind: str) -> Optional[Bot]:
        """Return the enabled bot flagged as default for the given kind
        ('group' or 'dm'). Falls back to any enabled bot, then any bot."""
        if not self._cache_loaded:
            return None
        bots = list(self._cache.values())
        flag = "default_for_dm" if kind == "dm" else "default_for_group"
        for b in bots:
            if b.enabled and getattr(b, flag):
                return b
        for b in bots:
            if b.enabled:
                return b
        return bots[0] if bots else None

    def match_alias(self, text: str, *, candidates: Optional[list[Bot]] = None) -> Optional[Bot]:
        """Return the first bot whose alias is a whole-word match in `text`.

        Matching mirrors `signal/handler.py:_resolve_addressed_bot`:
        case-insensitive, whole-word boundary. Used in PR4 to decide
        which bot a "hey Sigil" / "hey Artaud" inbound is addressing.
        Restrict the search to `candidates` (e.g. bots assigned to the
        current context) to avoid Bot A in chat A reacting to a mention
        of Bot B from a different room."""
        if not text:
            return None
        pool = candidates if candidates is not None else self.list_sync()
        for bot in pool:
            for alias in bot.alias_set():
                if re.search(rf"\b{re.escape(alias)}\b", text, re.IGNORECASE):
                    return bot
        return None

    async def list(self) -> list[Bot]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM bots ORDER BY id"
            )
            rows = await cursor.fetchall()
        return [self._row_to_bot(r) for r in rows]

    async def get(self, bot_id: int) -> Optional[Bot]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM bots WHERE id = ?",
                (bot_id,),
            )
            row = await cursor.fetchone()
        return self._row_to_bot(row) if row else None

    async def get_by_slug(self, slug: str) -> Optional[Bot]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._SELECT_FIELDS} FROM bots WHERE slug = ?",
                (slug,),
            )
            row = await cursor.fetchone()
        return self._row_to_bot(row) if row else None

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    async def upsert(self, bot: Bot) -> int:
        """Insert or update. Enforces single-default-per-kind by clearing
        the flag on other rows when this row sets it."""
        await self._ensure_initialized()
        if bot.deep_think_mode not in DEEP_THINK_MODES:
            raise ValueError(
                f"deep_think_mode must be one of {DEEP_THINK_MODES}, "
                f"got {bot.deep_think_mode!r}"
            )
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
            if bot.id is None:
                cursor = await db.execute(
                    """INSERT INTO bots (
                        slug, display_name, aliases, persona,
                        signal_phone, signal_api_url, enabled,
                        default_for_dm, default_for_group,
                        deep_think_mode, deep_think_handoff_prompt,
                        vision_enabled, deep_think_enabled,
                        routing_blurb, audio_enabled, audio_part_style,
                        created_at, updated_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        bot.slug,
                        bot.display_name,
                        json.dumps(bot.aliases or []),
                        bot.persona or None,
                        bot.signal_phone or None,
                        bot.signal_api_url or None,
                        1 if bot.enabled else 0,
                        1 if bot.default_for_dm else 0,
                        1 if bot.default_for_group else 0,
                        bot.deep_think_mode,
                        bot.deep_think_handoff_prompt or None,
                        1 if bot.vision_enabled else 0,
                        1 if bot.deep_think_enabled else 0,
                        bot.routing_blurb or None,
                        1 if bot.audio_enabled else 0,
                        bot.audio_part_style or AUDIO_PART_STYLE_INPUT_AUDIO,
                        now,
                        now,
                    ),
                )
                new_id = cursor.lastrowid or 0
                await self._enforce_single_default(db, new_id, bot)
                await db.commit()
                await self._load_cache()
                return new_id
            else:
                await db.execute(
                    """UPDATE bots SET
                           slug = ?, display_name = ?, aliases = ?,
                           persona = ?, signal_phone = ?, signal_api_url = ?,
                           enabled = ?, default_for_dm = ?, default_for_group = ?,
                           deep_think_mode = ?, deep_think_handoff_prompt = ?,
                           vision_enabled = ?, deep_think_enabled = ?,
                           routing_blurb = ?, audio_enabled = ?,
                           audio_part_style = ?,
                           updated_at = ?
                       WHERE id = ?""",
                    (
                        bot.slug,
                        bot.display_name,
                        json.dumps(bot.aliases or []),
                        bot.persona or None,
                        bot.signal_phone or None,
                        bot.signal_api_url or None,
                        1 if bot.enabled else 0,
                        1 if bot.default_for_dm else 0,
                        1 if bot.default_for_group else 0,
                        bot.deep_think_mode,
                        bot.deep_think_handoff_prompt or None,
                        1 if bot.vision_enabled else 0,
                        1 if bot.deep_think_enabled else 0,
                        bot.routing_blurb or None,
                        1 if bot.audio_enabled else 0,
                        bot.audio_part_style or AUDIO_PART_STYLE_INPUT_AUDIO,
                        now,
                        bot.id,
                    ),
                )
                await self._enforce_single_default(db, bot.id, bot)
                await db.commit()
                await self._load_cache()
                return bot.id

    @staticmethod
    async def _enforce_single_default(
        db: aiosqlite.Connection, bot_id: int, bot: Bot
    ) -> None:
        """Enforce two invariants on the default flags:

        1. At most one bot is default-for-kind: clearing the flag on
           every other row when this bot sets it. Without this,
           default_for_kind_sync would return a non-deterministic
           "first one we find" answer when multiple rows are flagged.

        2. At least one ENABLED bot is default-for-kind. The check
           reads back current state after step 1 to confirm — if this
           bot is the would-be default and is enabled, we're fine; if
           not, some other enabled bot must hold the flag, otherwise
           inbound messages in that kind have no bot to answer them.

        The second check is a guardrail, not validation against bad
        intent: BotRegistry never auto-clears flags on unrelated rows
        beyond rule 1. Callers (admin form) get a ValueError they can
        surface to the user."""
        if bot.default_for_dm:
            await db.execute(
                "UPDATE bots SET default_for_dm = 0 WHERE id != ?",
                (bot_id,),
            )
        if bot.default_for_group:
            await db.execute(
                "UPDATE bots SET default_for_group = 0 WHERE id != ?",
                (bot_id,),
            )

        # Invariant: at least one enabled bot per kind is default.
        # Skip the check when inserting the very first row (there's
        # no "other" bot yet to fall back on, and seed always sets
        # both defaults). The check looks at the state AFTER step 1
        # so the bot we just upserted is included.
        for col in ("default_for_dm", "default_for_group"):
            cursor = await db.execute(
                f"SELECT COUNT(*) FROM bots WHERE {col} = 1 AND enabled = 1"
            )
            row = await cursor.fetchone()
            count = row[0] if row else 0
            if count == 0:
                kind = "DMs" if col == "default_for_dm" else "groups"
                raise ValueError(
                    f"Save would leave no default bot for {kind}. "
                    f"At least one enabled bot must hold "
                    f"default_for_{kind[:-1] if kind.endswith('s') else kind}. "
                    f"Toggle the flag on another bot first, or keep this "
                    f"one's flag set."
                )

    async def delete(self, bot_id: int) -> bool:
        """Refuse to delete a bot still referenced by contexts OR by
        scoped portfolios (`@bot:N` in context_key). Caller should
        reassign/reset first. The seeded sigil row is also protected
        (slug=='sigil') because it's the migration anchor.

        Cascading scoped portfolios is unsafe: silent rewrite would
        merge two bots' positions into one, and silent abandonment
        leaves the cron + orders + settlement workers polling rows
        that resolve to a deleted bot. We require explicit admin
        intervention (reset the portfolio first) so cash routing
        stays honest.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
            cursor = await db.execute(
                "SELECT slug FROM bots WHERE id = ?", (bot_id,)
            )
            row = await cursor.fetchone()
            if not row:
                return False
            if row[0] == SIGIL_SLUG:
                return False
            cursor = await db.execute(
                "SELECT COUNT(*) FROM contexts WHERE default_bot_id = ?",
                (bot_id,),
            )
            ref = await cursor.fetchone()
            if ref and ref[0] > 0:
                return False
            # Scoped portfolios sanity check. The portfolio store lives
            # in the same SQLite database, so we can query its table
            # directly. If portfolios doesn't exist yet (fresh install,
            # consumer not initialized), there's nothing to protect.
            try:
                cursor = await db.execute(
                    "SELECT COUNT(*) FROM portfolios "
                    "WHERE context_key LIKE '%@bot:' || ? || '%'",
                    (str(bot_id),),
                )
                scoped_ref = await cursor.fetchone()
                if scoped_ref and scoped_ref[0] > 0:
                    logger.warning(
                        f"BotRegistry.delete({bot_id}): refusing — "
                        f"{scoped_ref[0]} scoped portfolio(s) still "
                        f"reference @bot:{bot_id}. Reset those "
                        f"portfolios first."
                    )
                    return False
            except Exception:
                # portfolios table doesn't exist yet — no cascade
                # concern.
                pass
            cursor = await db.execute(
                "DELETE FROM bots WHERE id = ?", (bot_id,)
            )
            # Defense in depth: even though we checked for context refs
            # above, NULL out any stale pin in case there's a row that
            # wasn't enforced (older schema without FK). Cheap and
            # idempotent.
            await db.execute(
                "UPDATE contexts SET default_bot_id = NULL "
                "WHERE default_bot_id = ?",
                (bot_id,),
            )
            await db.commit()
        await self._load_cache()
        return cursor.rowcount > 0

    # ------------------------------------------------------------------
    # Backfill — runs once after consumer stores have created their tables
    # ------------------------------------------------------------------

    async def backfill_consumer_tables(self) -> None:
        """Set bot_id on every row that's still NULL to the seeded
        sigil bot. Idempotent — re-running does nothing once columns are
        populated. Intentionally lenient about missing tables/columns:
        a fresh install may not have run the consumer's
        `_ensure_initialized` yet, in which case there's nothing to
        backfill and we no-op."""
        await self._ensure_initialized()
        sigil = await self.get_by_slug(SIGIL_SLUG)
        if not sigil or sigil.id is None:
            logger.warning("BotRegistry backfill: no sigil row, skipping")
            return
        async with aiosqlite.connect(self.db_path) as db:
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
            for table, col in CONSUMER_TABLES.items():
                try:
                    cursor = await db.execute(f"PRAGMA table_info({table})")
                    cols = {r[1] for r in await cursor.fetchall()}
                except Exception:
                    continue
                if not cols:
                    continue  # table doesn't exist yet
                if col not in cols:
                    continue  # consumer hasn't migrated yet
                cursor = await db.execute(
                    f"UPDATE {table} SET {col} = ? WHERE {col} IS NULL",
                    (sigil.id,),
                )
                if cursor.rowcount > 0:
                    logger.info(
                        f"BotRegistry backfill: {table}.{col} <- "
                        f"{sigil.id} ({cursor.rowcount} rows)"
                    )
            await db.commit()
