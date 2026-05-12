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

import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import db_session

logger = logging.getLogger(__name__)

DEFAULT_RETENTION_DAYS = 7


def format_relative_age(seconds: float) -> str:
    """Render a non-negative age as a short relative-time string.

    Granularity buckets: "just now" (<60s), "Nm ago" (<60m), "Nh ago" (<24h),
    "Nd ago" (<7d), "Nw ago" otherwise. Coarse on purpose — the model only
    needs to distinguish "fresh" from "day-old" from "ancient", not
    minute-precision recall.
    """
    if seconds < 0:
        seconds = 0
    if seconds < 60:
        return "just now"
    if seconds < 3600:
        return f"{int(seconds // 60)}m ago"
    if seconds < 86400:
        return f"{int(seconds // 3600)}h ago"
    if seconds < 7 * 86400:
        return f"{int(seconds // 86400)}d ago"
    return f"{int(seconds // (7 * 86400))}w ago"


def _join_bracket(name: Optional[str], ago: Optional[str]) -> Optional[str]:
    """Compose the inside of a `[...]` label from the parts that exist."""
    if name and ago:
        return f"{name}, {ago}"
    return name or ago


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
        async with db_session(self) as db:
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
        async with db_session(self) as db:
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
        async with db_session(self) as db:
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
        now: Optional[float] = None,
    ) -> list[dict]:
        """Return the last 2*N rows for this context in chronological order.

        When `attribute_senders` is True, user messages are prefixed with
        `[...tail]` so the LLM can tell speakers apart in a group thread.

        When `now` is provided, user messages also carry a relative-time
        suffix in the same bracket (`[David, 2h ago]`, `[3d ago]` in DMs)
        so the model can tell which turns are fresh vs. days old.
        """
        await self._ensure_initialized()
        n = (turns_per_user or self.turns_per_user) * 2
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT role, content, sender_tail, user_hash, created_at,
                          image_refs
                   FROM conversation_turns
                   WHERE context_key = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT ?""",
                (context_key, n),
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
                ago = (
                    format_relative_age(now - created_at)
                    if now is not None and created_at is not None
                    else None
                )
                bracket = _join_bracket(name, ago)
                if bracket:
                    text = f"[{bracket}] {content}"
            elif role == "assistant" and attribute_senders:
                # Group playback: tag the bot's prior reply with whom it
                # answered so the model can pair questions with answers
                # across speakers. `user_hash` / `sender_tail` were stored
                # at append time as the addressee of this reply.
                name = self._attribution_label(user_hash, sender_tail)
                ago = (
                    format_relative_age(now - created_at)
                    if now is not None and created_at is not None
                    else None
                )
                if name:
                    bracket = _join_bracket(f"to {name}", ago)
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

    async def latest_turn_timestamp(self, context_key: str) -> Optional[float]:
        """Return the unix timestamp of the most recent turn, or None if empty.

        Used by the prompt builder to decide whether the prior conversation
        is stale enough to warrant an explicit advisory to the model.
        """
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT MAX(created_at) FROM conversation_turns WHERE context_key = ?",
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
                    created_at, image_refs)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (user_hash, context_key, sender_tail, role, content,
                 now, image_refs_json),
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
