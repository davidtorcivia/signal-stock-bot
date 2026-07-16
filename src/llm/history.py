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

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import db_session

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 7


def format_history_timestamp(created_at: float) -> str:
    """Render a turn's creation time as an absolute UTC wall-clock stamp.

    Format: ``2026-06-06 18:30 UTC``. Absolute on purpose — and that is the
    whole point. A turn's timestamp must be identical on every request, or
    the backend's prefix cache (DeepSeek does hours-to-days prefix-match
    caching) misses on the entire payload: system text, tool schemas, and
    every prior turn become un-cacheable. The old relative form ("2m ago")
    changed every minute, so the history block — which sits in the cached
    prefix — was rewritten on each call and the cache never hit.

    Recency is still recoverable by the model: `_inject_current_time`
    stamps the current UTC+ET time onto the tail of the last user message,
    so the model subtracts that anchor from these absolute stamps itself.
    Minute precision (no seconds) keeps the string stable for a full minute
    even if two reads of the *same* turn ever disagreed on sub-minute time.
    """
    dt = datetime.fromtimestamp(created_at, tz=timezone.utc)
    return dt.strftime("%Y-%m-%d %H:%M UTC")


def _join_bracket(name: Optional[str], stamp: Optional[str]) -> Optional[str]:
    """Compose the inside of a `[...]` label from the parts that exist."""
    if name and stamp:
        return f"{name}, {stamp}"
    return name or stamp


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
        # Per-context locks so a user+assistant turn pair is written
        # without another ask's pair interleaving (which would garble
        # the replayed alternation). Callers hold lock_for() around
        # the paired append() calls.
        self._ctx_locks: dict[str, asyncio.Lock] = {}

    def lock_for(self, context_key: str) -> asyncio.Lock:
        """Return the (lazily created) lock serializing writes for one
        context. Bounded: idle locks are evicted once the table grows
        past 512 contexts."""
        lock = self._ctx_locks.get(context_key)
        if lock is None:
            if len(self._ctx_locks) > 512:
                for k in [
                    k for k, v in self._ctx_locks.items()
                    if not v.locked()
                ][:256]:
                    del self._ctx_locks[k]
            lock = self._ctx_locks[context_key] = asyncio.Lock()
        return lock

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
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
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
            if "bot_id" not in cols:
                # Multi-bot scoping: which bot authored the turn (assistant
                # rows) or which bot's history this user turn belongs to.
                # NULL on existing rows is backfilled to the seeded sigil
                # bot. Filtering on bot_id when shared-history mode is
                # enabled lets two bots co-exist in one context without
                # mixing their assistant turns. Indexed alongside
                # context_key so per-(bot,context) reads stay cheap.
                await db.execute(
                    "ALTER TABLE conversation_turns ADD COLUMN bot_id INTEGER"
                )
            if "image_refs" not in cols:
                # Vision history: JSON-encoded list of
                # {mime, data_b64, filename} dicts attached to the user
                # turn when the inbound message carried images. NULL on
                # text-only turns. Replay window (last N user turns)
                # is gated in ask_command — older rows keep the column
                # populated until they age out via the standard
                # row-cap / retention prune, but the model only sees
                # the last N.
                await db.execute(
                    "ALTER TABLE conversation_turns ADD COLUMN image_refs TEXT"
                )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_ctx_time "
                "ON conversation_turns(context_key, created_at)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_conv_bot_ctx_time "
                "ON conversation_turns(bot_id, context_key, created_at)"
            )
            # Per-(context, bot) rolling summary. `summary_through_id`
            # is the highest conversation_turns.id that's already been
            # folded in — newer turns above that are still verbatim in
            # conversation_turns and haven't been compressed yet.
            # Composite PK on (context_key, bot_id) so each bot in a
            # multi-bot group has its OWN compressed memory — otherwise
            # bot A's summary blob leaks into bot B's prompt, defeating
            # the per-bot history filter on conversation_turns.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    context_key TEXT NOT NULL,
                    bot_id INTEGER NOT NULL DEFAULT 0,
                    summary TEXT NOT NULL,
                    summary_through_id INTEGER NOT NULL DEFAULT 0,
                    turns_summarized INTEGER NOT NULL DEFAULT 0,
                    updated_at REAL NOT NULL,
                    PRIMARY KEY (context_key, bot_id)
                )
                """
            )
            # Migration: legacy summaries lived as one row per
            # context_key with PRIMARY KEY (context_key) and no bot_id.
            # ALTER TABLE can't change a primary key, so adding the
            # column isn't enough: upsert_summary's
            # ON CONFLICT(context_key, bot_id) needs a matching
            # constraint, and the single-column PK also forbids two
            # bots' rows coexisting in one context. Rebuild the table
            # when the composite PK is missing, promoting legacy rows
            # to bot_id=0 (the "shared / legacy" sentinel every reader
            # falls back to when the per-bot row is absent).
            cursor = await db.execute(
                "PRAGMA table_info(conversation_summaries)"
            )
            summ_info = await cursor.fetchall()
            summ_cols = {r[1] for r in summ_info}
            pk_cols = {r[1] for r in summ_info if r[5]}
            if pk_cols != {"context_key", "bot_id"}:
                await db.execute(
                    "ALTER TABLE conversation_summaries "
                    "RENAME TO conversation_summaries_legacy"
                )
                await db.execute(
                    """
                    CREATE TABLE conversation_summaries (
                        context_key TEXT NOT NULL,
                        bot_id INTEGER NOT NULL DEFAULT 0,
                        summary TEXT NOT NULL,
                        summary_through_id INTEGER NOT NULL DEFAULT 0,
                        turns_summarized INTEGER NOT NULL DEFAULT 0,
                        updated_at REAL NOT NULL,
                        PRIMARY KEY (context_key, bot_id)
                    )
                    """
                )
                # Copy whichever of the new columns the legacy table
                # actually has; missing ones take their defaults.
                copy_cols = [
                    c for c in (
                        "context_key", "bot_id", "summary",
                        "summary_through_id", "turns_summarized",
                        "updated_at",
                    ) if c in summ_cols
                ]
                select_exprs = [
                    "COALESCE(bot_id, 0)" if c == "bot_id" else c
                    for c in copy_cols
                ]
                await db.execute(
                    f"INSERT OR IGNORE INTO conversation_summaries "
                    f"({', '.join(copy_cols)}) "
                    f"SELECT {', '.join(select_exprs)} "
                    f"FROM conversation_summaries_legacy"
                )
                await db.execute("DROP TABLE conversation_summaries_legacy")
                logger.info(
                    "Migrated conversation_summaries to composite "
                    "(context_key, bot_id) primary key"
                )
            await db.commit()
        self._initialized = True

    async def get_summary(
        self,
        context_key: str,
        floor_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> Optional[dict]:
        """Return the rolling summary for a (context, bot) pair, or None
        if absent. Each bot in a multi-bot group has its OWN summary so
        bot A's compressed memory never leaks into bot B's prompt.

        Reader fallback: if a per-bot row is missing, fall through to
        the legacy `bot_id=0` row (pre-migration single-bot summary).
        This keeps single-bot installs unaffected.

        Shape: {summary: str, summary_through_id: int, turns_summarized: int,
                updated_at: float}.

        `floor_at` is the context's purge floor. A summary written before
        the floor is invisible — it was built from pre-purge turns and
        would re-inject the very state the purge was meant to erase. The
        purge action also DELETEs the summary row directly, but this
        filter is the defense-in-depth that makes the floor hold forever.
        """
        bid = bot_id if bot_id is not None else 0
        async with db_session(self) as db:
            cursor = await db.execute(
                """SELECT summary, summary_through_id, turns_summarized, updated_at
                   FROM conversation_summaries
                   WHERE context_key = ? AND bot_id = ?""",
                (context_key, bid),
            )
            row = await cursor.fetchone()
            # Fallback: per-bot row missing → try the legacy bot_id=0
            # row (pre-multi-bot single-bot summary). Each bot reads
            # the same legacy blob exactly once before generating its
            # own.
            if row is None and bid != 0:
                cursor = await db.execute(
                    """SELECT summary, summary_through_id, turns_summarized, updated_at
                       FROM conversation_summaries
                       WHERE context_key = ? AND bot_id = 0""",
                    (context_key,),
                )
                row = await cursor.fetchone()
        if not row:
            return None
        if floor_at is not None and (row[3] or 0) < floor_at:
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
        bot_id: Optional[int] = None,
    ) -> None:
        bid = bot_id if bot_id is not None else 0
        async with db_session(self) as db:
            await db.execute(
                """INSERT INTO conversation_summaries
                   (context_key, bot_id, summary, summary_through_id,
                    turns_summarized, updated_at)
                   VALUES (?, ?, ?, ?, ?, ?)
                   ON CONFLICT(context_key, bot_id) DO UPDATE SET
                     summary = excluded.summary,
                     summary_through_id = excluded.summary_through_id,
                     turns_summarized = excluded.turns_summarized,
                     updated_at = excluded.updated_at""",
                (
                    context_key, bid, summary, summary_through_id,
                    turns_summarized, time.time(),
                ),
            )
            await db.commit()

    async def turns_to_summarize(
        self,
        context_key: str,
        summary_through_id: int,
        keep_recent: int,
        floor_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> list[dict]:
        """Return turns that should be folded into the summary.

        Excludes the most recent `keep_recent` rows (kept verbatim) and rows
        already covered by the existing summary. Used by the summarizer to
        decide what fresh material to feed the LLM.

        `floor_at` (the context's purge floor) further excludes any turn
        whose `created_at` precedes the floor — even though the purge
        action also DELETEs pre-floor rows, the filter is the
        defense-in-depth that prevents stale/replayed pre-floor rows from
        leaking back into the summary.

        `bot_id` (when set) restricts to user turns + this bot's
        assistant turns + legacy-NULL rows — same scoping as `load()`,
        so each bot's summary captures only its own conversation arc.
        """
        async with db_session(self) as db:
            # MAX(id) must be scoped to the SAME filter the summarizer
            # will consume — otherwise bot A's recent_floor is computed
            # off bot B's chatty turns, and bot A's summary either
            # bloats with unrelated content (if floor falls below A's
            # turns) or skips A's own recent turns (if floor sits above
            # them). Scope MAX by `(role='user' OR bot_id=? OR bot_id
            # IS NULL)` for per-bot summaries.
            if bot_id is not None:
                cursor = await db.execute(
                    "SELECT MAX(id) FROM conversation_turns "
                    "WHERE context_key = ? "
                    "  AND (role = 'user' OR bot_id = ? OR bot_id IS NULL)",
                    (context_key, bot_id),
                )
            else:
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
            params: list = [context_key, summary_through_id, recent_floor]
            floor_clause = ""
            if floor_at is not None:
                floor_clause = " AND created_at >= ?"
                params.append(floor_at)
            bot_clause = ""
            if bot_id is not None:
                bot_clause = (
                    " AND (role = 'user' OR bot_id = ? OR bot_id IS NULL)"
                )
                params.append(bot_id)
            cursor = await db.execute(
                f"""SELECT id, role, content, sender_tail, user_hash
                   FROM conversation_turns
                   WHERE context_key = ?
                     AND id > ?
                     AND id <= ?{floor_clause}{bot_clause}
                   ORDER BY id ASC""",
                tuple(params),
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
        now: Optional[float] = None,
        floor_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> list[dict]:
        """Return the last 2*N rows for this context in chronological order.

        When `attribute_senders` is True, user messages are prefixed with
        `[...tail]` so the LLM can tell speakers apart in a group thread.

        When `now` is provided, user messages also carry an absolute UTC
        timestamp in the same bracket (`[David, 2026-06-06 18:30 UTC]`, or
        `[2026-06-06 18:30 UTC]` in DMs) so the model can tell which turns
        are fresh vs. days old. `now` is only a presence gate here — the
        rendered stamp comes from each row's own `created_at`, not from
        `now`, so a given turn renders identically across requests and the
        prompt prefix stays cacheable. See `format_history_timestamp`.

        `floor_at` is the context's purge floor — rows with
        `created_at < floor_at` are filtered out so a purge can't be
        undone by stale or replayed pre-floor data.

        Multi-bot scoping (`bot_id` set): the returned alternation
        includes ALL user turns (they're shared across bots in the
        chat) but only the calling bot's own assistant turns. Legacy
        rows with bot_id IS NULL are treated as the calling bot's own
        — they were written before the column existed and must replay
        as the bot's own past for backward compat. Other bots' prior
        replies surface in the `<group_context>` block instead, where
        the renderer labels them with each writer's display name.
        """
        await self._ensure_initialized()
        # ``0`` is an explicit one-shot/no-history setting. Using ``or``
        # here silently replaced it with the global default and leaked old
        # turns into contexts whose policy promised stateless behavior.
        effective_turns = (
            self.turns_per_user
            if turns_per_user is None
            else turns_per_user
        )
        n = max(0, int(effective_turns)) * 2
        async with aiosqlite.connect(self.db_path) as db:
            params: list = [context_key]
            floor_clause = ""
            if floor_at is not None:
                floor_clause = " AND created_at >= ?"
                params.append(floor_at)
            bot_clause = ""
            if bot_id is not None:
                # All user turns + this bot's assistant turns + legacy
                # (bot_id IS NULL) assistant turns. Two-bot groups thus
                # see their own conversation arc cleanly without other
                # bots' replies polluting the assistant-role slot.
                bot_clause = (
                    " AND (role = 'user' OR bot_id = ? OR bot_id IS NULL)"
                )
                params.append(bot_id)
            params.append(n)
            cursor = await db.execute(
                f"""SELECT role, content, sender_tail, user_hash, created_at,
                          image_refs
                   FROM conversation_turns
                   WHERE context_key = ?{floor_clause}{bot_clause}
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                tuple(params),
            )
            rows = await cursor.fetchall()

        turns: list[dict] = []
        for role, content, sender_tail, user_hash, created_at, image_refs_json in reversed(rows):
            text = content
            if role == "user":
                name = (
                    self._attribution_label(user_hash, sender_tail)
                    if attribute_senders else None
                )
                stamp = (
                    format_history_timestamp(created_at)
                    if now is not None and created_at is not None
                    else None
                )
                bracket = _join_bracket(name, stamp)
                if bracket:
                    text = f"[{bracket}] {content}"
            elif role == "assistant" and attribute_senders:
                # Group playback: tag the bot's prior reply with whom it
                # answered so the model can pair questions with answers
                # across speakers. `user_hash` / `sender_tail` were stored
                # at append time as the addressee of this reply.
                name = self._attribution_label(user_hash, sender_tail)
                stamp = (
                    format_history_timestamp(created_at)
                    if now is not None and created_at is not None
                    else None
                )
                if name:
                    bracket = _join_bracket(f"to {name}", stamp)
                    if bracket:
                        text = f"[{bracket}] {content}"
            turn: dict = {"role": role, "content": text}
            # Image attachments stored at append time. ask_command
            # inflates these into multimodal parts for the last N user
            # turns; older turns keep the column populated but the LLM
            # only sees the text. Empty / malformed JSON yields no
            # images (the row falls back to text-only safely).
            if image_refs_json:
                try:
                    refs = json.loads(image_refs_json)
                    if isinstance(refs, list) and refs:
                        turn["image_refs"] = refs
                except (TypeError, ValueError):
                    pass
            turns.append(turn)
        return turns

    async def latest_turn_timestamp(
        self, context_key: str, floor_at: Optional[float] = None
    ) -> Optional[float]:
        """Return the unix timestamp of the most recent turn, or None if empty.

        Used by the prompt builder to decide whether the prior conversation
        is stale enough to warrant an explicit advisory to the model.

        `floor_at` is the context's purge floor — turns older than the
        floor are invisible, so the staleness check operates on what the
        LLM will actually see.
        """
        async with db_session(self) as db:
            if floor_at is not None:
                cursor = await db.execute(
                    "SELECT MAX(created_at) FROM conversation_turns "
                    "WHERE context_key = ? AND created_at >= ?",
                    (context_key, floor_at),
                )
            else:
                cursor = await db.execute(
                    "SELECT MAX(created_at) FROM conversation_turns "
                    "WHERE context_key = ?",
                    (context_key,),
                )
            row = await cursor.fetchone()
        return row[0] if row and row[0] is not None else None

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
        turns_per_user: Optional[int] = None,
        image_refs: Optional[list[dict]] = None,
        bot_id: Optional[int] = None,
    ) -> None:
        """Insert one turn and prune by row-cap (per context) and age (global).

        `turns_per_user` overrides the constructor default so a per-context
        override can keep more (or fewer) rows than the global setting —
        otherwise append-time pruning would clip whatever load() asks for.

        `image_refs` is a list of `{mime, data_b64, filename}` dicts
        captured by the signal handler from inbound attachments. Serialized
        to JSON in the `image_refs` column; ask_command re-inflates the
        last N user turns' refs into OpenAI multimodal parts so follow-up
        questions can refer back to the picture.
        """
        await self._ensure_initialized()

        now = time.time()
        age_cutoff = now - self._retention_seconds()
        effective_turns = (
            turns_per_user if turns_per_user is not None else self.turns_per_user
        )
        max_rows = max(0, effective_turns) * 2
        image_refs_json: Optional[str] = None
        if image_refs:
            try:
                image_refs_json = json.dumps(image_refs)
            except (TypeError, ValueError) as e:
                # Malformed payload — log and persist without images
                # rather than dropping the turn entirely.
                logger.warning(f"history.append: image_refs serialize failed: {e}")

        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT INTO conversation_turns
                   (user_hash, context_key, sender_tail, role, content,
                    created_at, image_refs, bot_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (user_hash, context_key, sender_tail, role, content,
                 now, image_refs_json, bot_id),
            )
            # Age-based purge (whole table)
            await db.execute(
                "DELETE FROM conversation_turns WHERE created_at < ?",
                (age_cutoff,),
            )
            # Row-cap purge: per (context_key, role-or-bot) so a chatty
            # bot in a multi-bot group can't evict another bot's history
            # by burning the per-context budget. We compute two pools:
            #   - assistant rows scoped to THIS bot_id (or NULL bot row)
            #   - user rows (which are shared but tagged with the asker's
            #     bot turn — they're scoped here by bot_id too so each
            #     bot's "asker history" stays whole)
            # Falls back to the legacy per-context prune for any rows
            # whose bot_id is NULL (pre-migration) so the existing
            # row-cap semantics still apply to legacy data.
            if bot_id is not None:
                await db.execute(
                    """DELETE FROM conversation_turns
                       WHERE context_key = ?
                         AND bot_id = ?
                         AND id NOT IN (
                             SELECT id FROM conversation_turns
                             WHERE context_key = ? AND bot_id = ?
                             ORDER BY created_at DESC, id DESC
                             LIMIT ?
                         )""",
                    (context_key, bot_id, context_key, bot_id, max_rows),
                )
            else:
                # Legacy single-bot prune path.
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
        async with db_session(self) as db:
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
