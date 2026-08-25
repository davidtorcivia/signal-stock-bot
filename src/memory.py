"""
Per-context memory store.

Memories are facts the bot has learned about people (or the room itself, or
free-text entities) within a specific chat context. Same person can have
different memories in different contexts — the woo-chat David and the
trading-chat David are independent rows.

Schema:
  context_memories(id, context_id, subject_key, subject_label, kind,
                   content, confidence, source, source_user_hash,
                   source_message_at, distinct_speakers, corroborations,
                   created_at, updated_at, bot_id)

Keys:
  * context_id    — FK to contexts.id (per-context isolation)
  * subject_key   — who/what this is about. Three shapes:
      - 64-char sha256 hex   → a Signal user (matches NameRegistry user_hash)
      - "__context__"        → the room/conversation itself
      - "freetext:<slug>"    → an unrelated entity (the cat, their boss…)
  * subject_label — denormalized human label ("David", "the room", "the cat")
  * kind          — 'identity' | 'preference' | 'fact' | 'event'

Audit columns:
  * source             — 'reactor' | 'explicit' | 'admin'
  * source_user_hash   — sha256 of the speaker who triggered the write.
                         For explicit and reactor writes, this is the user
                         whose message produced the memory; for admin writes,
                         empty. Lets admins trace a memory back to a message.
  * source_message_at  — unix ts of the triggering message

The `confidence`, `corroborations` and `distinct_speakers` columns are
vestigial — kept so old rows and the admin views still read, but every
new row is written with confidence 1.0 and nothing promotes or decays.
"""

import logging
import re
import time
from pathlib import Path
from typing import Iterable, Optional

import aiosqlite

from .database import db_session

logger = logging.getLogger(__name__)


SUBJECT_CONTEXT = "__context__"          # the room itself
SUBJECT_SELF = "__self__"                # the bot itself (its persona,
                                         # model, capabilities, voice)
SUBJECT_FREETEXT_PREFIX = "freetext:"

KIND_IDENTITY = "identity"
KIND_PREFERENCE = "preference"
KIND_FACT = "fact"
KIND_EVENT = "event"
KINDS = (KIND_IDENTITY, KIND_PREFERENCE, KIND_FACT, KIND_EVENT)

SOURCE_REACTOR = "reactor"
SOURCE_EXPLICIT = "explicit"
SOURCE_ADMIN = "admin"

# Vestigial column, written as a constant on every insert.
DEFAULT_CONFIDENCE = 1.0

_USER_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")
# "[...4160]", "...4160", "4160" — the anonymous speaker label.
_PHONE_TAIL_RE = re.compile(r"^\[?\.{0,3}\d{4}\]?$")


def is_user_hash(subject_key: str) -> bool:
    return bool(_USER_HASH_RE.match(subject_key or ""))


def freetext_subject_key(label: str) -> str:
    """Normalize a free-text label into a stable subject key.

    "The Cat" → "freetext:the-cat". Returns "" on empty input so callers
    can guard. Idempotent: passing an already-prefixed key returns it
    unchanged.
    """
    s = (label or "").strip()
    if not s:
        return ""
    if s.startswith(SUBJECT_FREETEXT_PREFIX):
        return s
    if s in (SUBJECT_CONTEXT, SUBJECT_SELF):
        return s
    if is_user_hash(s):
        return s
    slug = _SLUG_RE.sub("-", s.lower()).strip("-")
    return f"{SUBJECT_FREETEXT_PREFIX}{slug}" if slug else ""


class MemoryStore:
    """Async SQLite-backed memory rows, keyed per context."""

    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:  # noqa: D401
        # Wraps schema setup; tuning pragmas applied here too. WAL is sticky
        # on the file (so it persists for every other store sharing this DB)
        # but we re-apply on every init because all stores call this lazily
        # and the first one to land sets it.
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            from .database import apply_db_pragmas
            await apply_db_pragmas(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS context_memories (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_id INTEGER NOT NULL,
                    subject_key TEXT NOT NULL,
                    subject_label TEXT NOT NULL DEFAULT '',
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    confidence REAL NOT NULL DEFAULT 1.0,
                    source TEXT NOT NULL DEFAULT 'explicit',
                    source_user_hash TEXT NOT NULL DEFAULT '',
                    source_message_at REAL,
                    distinct_speakers TEXT NOT NULL DEFAULT '[]',
                    corroborations INTEGER NOT NULL DEFAULT 1,
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            # Upgrade path for installs that pre-date the audit columns.
            cursor = await db.execute("PRAGMA table_info(context_memories)")
            cols = {r[1] for r in await cursor.fetchall()}
            if "source_user_hash" not in cols:
                await db.execute(
                    "ALTER TABLE context_memories ADD COLUMN "
                    "source_user_hash TEXT NOT NULL DEFAULT ''"
                )
            if "source_message_at" not in cols:
                await db.execute(
                    "ALTER TABLE context_memories ADD COLUMN "
                    "source_message_at REAL"
                )
            if "distinct_speakers" not in cols:
                await db.execute(
                    "ALTER TABLE context_memories ADD COLUMN "
                    "distinct_speakers TEXT NOT NULL DEFAULT '[]'"
                )
            if "bot_id" not in cols:
                # Multi-bot scoping: which bot learned this memory. NULL
                # on existing rows is backfilled to the seeded sigil bot.
                # We default to per-bot memories (different personas may
                # extract different facts about the same context) but
                # the read path can ignore bot_id when the install
                # prefers shared memory.
                await db.execute(
                    "ALTER TABLE context_memories ADD COLUMN bot_id INTEGER"
                )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ctx_subj "
                "ON context_memories(context_id, subject_key)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ctx_kind "
                "ON context_memories(context_id, kind)"
            )
            # Composite index covering the per-bot dedup query in add()
            # — `WHERE context_id=? AND subject_key=? AND kind=? AND
            # (bot_id=? OR bot_id IS NULL)`. With multi-bot writes,
            # the per-(context, subject) slice grows linearly in
            # active bots; this index keeps the dedup seek fast.
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ctx_subj_kind_bot "
                "ON context_memories(context_id, subject_key, kind, bot_id)"
            )
            await db.commit()
        self._initialized = True

    async def add(
        self,
        *,
        context_id: int,
        subject_key: str,
        subject_label: str,
        kind: str,
        content: str,
        source: str = SOURCE_EXPLICIT,
        source_user_hash: str = "",
        source_message_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> Optional[int]:
        """Insert a memory. Returns the row id, or None on bad input.

        Dedup: if a row with the same (context_id, subject_key, kind)
        already holds near-duplicate content, return that row's id
        instead of inserting a parallel one. Read+write runs under
        BEGIN IMMEDIATE so two concurrent writers can't both miss the
        duplicate and insert.
        """
        if not subject_key or kind not in KINDS or not (content or "").strip():
            return None

        await self._ensure_initialized()
        now = time.time()
        content = content.strip()
        speaker = source_user_hash or ""

        async with db_session(self) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                # Dedup scope: each bot maintains its own mental model of
                # the chat, but legacy NULL-bot rows (pre-multi-bot
                # installs and admin-added rows without a bot tag) are
                # treated as shared — a per-bot writer that finds a
                # near-duplicate NULL-bot row reuses it rather than
                # creating a duplicate the reader would see twice (the
                # read path is `(bot_id = ? OR bot_id IS NULL)`).
                if bot_id is None:
                    bot_clause = " AND bot_id IS NULL"
                    bot_params: tuple = ()
                else:
                    bot_clause = " AND (bot_id = ? OR bot_id IS NULL)"
                    bot_params = (bot_id,)
                cursor = await db.execute(
                    """SELECT id, content
                       FROM context_memories
                       WHERE context_id = ? AND subject_key = ? AND kind = ?"""
                    + bot_clause,
                    (context_id, subject_key, kind) + bot_params,
                )
                for row in await cursor.fetchall():
                    if _is_near_duplicate(row[1], content):
                        await db.commit()
                        return row[0]

                cursor = await db.execute(
                    """INSERT INTO context_memories
                       (context_id, subject_key, subject_label, kind, content,
                        confidence, source, source_user_hash,
                        source_message_at, created_at, updated_at, bot_id)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        context_id, subject_key, subject_label, kind, content,
                        DEFAULT_CONFIDENCE, source, speaker, source_message_at,
                        now, now, bot_id,
                    ),
                )
                await db.commit()
                return cursor.lastrowid
            except Exception:
                await db.rollback()
                raise

    _SELECT_COLS = (
        "id, subject_key, subject_label, kind, content, confidence, "
        "source, created_at, updated_at, "
        "source_user_hash, source_message_at"
    )

    async def list_for_subject(
        self,
        *,
        context_id: int,
        subject_key: str,
        kinds: Optional[tuple] = None,
        bot_id: Optional[int] = None,
    ) -> list[dict]:
        """Per-bot scoping (`bot_id` set): returns rows owned by this
        bot plus legacy NULL-bot rows (shared from before the column
        existed). When `bot_id` is None, returns ALL rows regardless
        of bot_id — used by admin views and legacy single-bot reads.
        """
        await self._ensure_initialized()
        sql = (
            f"SELECT {self._SELECT_COLS} "
            "FROM context_memories "
            "WHERE context_id = ? AND subject_key = ?"
        )
        params: list = [context_id, subject_key]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        if bot_id is not None:
            sql += " AND (bot_id = ? OR bot_id IS NULL)"
            params.append(bot_id)
        sql += " ORDER BY confidence DESC, updated_at DESC"
        async with db_session(self) as db:
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def list_for_context(
        self, context_id: int, limit: int = 500,
        bot_id: Optional[int] = None,
    ) -> list[dict]:
        """List every memory in a context. `bot_id=None` returns ALL
        rows (admin/audit view); `bot_id=N` scopes to bot N's view
        (own rows + legacy NULL-bot rows)."""
        bot_clause = ""
        params: tuple = (context_id,)
        if bot_id is not None:
            bot_clause = " AND (bot_id = ? OR bot_id IS NULL)"
            params = params + (bot_id,)
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}
                    FROM context_memories
                    WHERE context_id = ?""" + bot_clause + """
                    ORDER BY subject_label, kind, updated_at DESC
                    LIMIT ?""",
                params + (limit,),
            )
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def search(
        self,
        *,
        context_id: int,
        query: str,
        limit: int = 12,
        bot_id: Optional[int] = None,
    ) -> list[dict]:
        """Substring match across subject_label and content. Best-effort
        fuzzy retrieval for the `recall` tool — real ranking is the LLM's job.

        `bot_id` (when set): restrict results to memories owned by this
        bot, plus legacy NULL-bot rows. Other bots' memories are hidden
        so each bot recalls its own mental model.
        """
        await self._ensure_initialized()
        q = (query or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        bot_clause = ""
        params: tuple = (context_id, like, like)
        if bot_id is not None:
            bot_clause = " AND (bot_id = ? OR bot_id IS NULL)"
            params = params + (bot_id,)
        params = params + (limit,)
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}
                    FROM context_memories
                    WHERE context_id = ?
                      AND (subject_label LIKE ? OR content LIKE ?)"""
                + bot_clause +
                """
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT ?""",
                params,
            )
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def get(self, memory_id: int) -> Optional[dict]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}, context_id
                    FROM context_memories WHERE id = ?""",
                (memory_id,),
            )
            row = await cursor.fetchone()
        if not row:
            return None
        d = _row_to_dict(row)
        d["context_id"] = row[11]
        return d

    async def update(
        self,
        memory_id: int,
        *,
        content: Optional[str] = None,
        subject_key: Optional[str] = None,
        subject_label: Optional[str] = None,
    ) -> bool:
        await self._ensure_initialized()
        sets: list[str] = []
        params: list = []
        if content is not None:
            sets.append("content = ?")
            params.append(content.strip())
        if subject_key is not None:
            sk = (subject_key or "").strip()
            if not sk:
                return False
            # Validate so admins can't write malformed keys (e.g. arbitrary
            # 32-char strings that don't fit any of our three shapes).
            if not (
                sk == SUBJECT_CONTEXT
                or is_user_hash(sk)
                or sk.startswith(SUBJECT_FREETEXT_PREFIX)
            ):
                sk = freetext_subject_key(sk) or sk
            sets.append("subject_key = ?")
            params.append(sk)
        if subject_label is not None:
            sets.append("subject_label = ?")
            params.append(subject_label)
        if not sets:
            return False
        sets.append("updated_at = ?")
        params.append(time.time())
        params.append(memory_id)
        async with db_session(self) as db:
            cursor = await db.execute(
                f"UPDATE context_memories SET {', '.join(sets)} WHERE id = ?",
                params,
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete(self, memory_id: int) -> bool:
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM context_memories WHERE id = ?", (memory_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_for_context(self, context_id: int) -> int:
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM context_memories WHERE context_id = ?",
                (context_id,),
            )
            await db.commit()
            return cursor.rowcount


def _row_to_dict(row) -> dict:
    return {
        "id": row[0],
        "subject_key": row[1],
        "subject_label": row[2],
        "kind": row[3],
        "content": row[4],
        "confidence": row[5],
        "source": row[6],
        "created_at": row[7],
        "updated_at": row[8],
        "source_user_hash": row[9] if len(row) > 9 else "",
        "source_message_at": row[10] if len(row) > 10 else None,
    }


_PUNCT_RE = re.compile(r"[^a-z0-9]+")
# Tiny English stopword list. Matters for short factual strings where one
# extra "and"/"a"/"the" otherwise drags Jaccard below the threshold.
_DEDUP_STOPWORDS = frozenset({
    "a", "an", "and", "the", "of", "to", "in", "on", "is", "are",
    "was", "were", "be", "with", "for", "or", "but", "as",
})


def _tokenize_for_match(s: str) -> set[str]:
    """Lowercase, strip punctuation, drop tiny stopwords."""
    tokens = _PUNCT_RE.split((s or "").lower())
    return {t for t in tokens if t and t not in _DEDUP_STOPWORDS}


def _is_near_duplicate(a: str, b: str, threshold: float = 0.85) -> bool:
    """Cheap near-duplicate check for corroboration.

    Strips punctuation + stopwords, tokenizes, and computes Jaccard overlap.
    Above threshold counts as the same memory, so `add()` reuses the
    existing row instead of inserting a parallel one. Avoids pulling in a
    real similarity library for what's a back-pocket dedup.
    """
    ta = _tokenize_for_match(a)
    tb = _tokenize_for_match(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= threshold


def canonical_key(name_registry, key: str) -> str:
    """One person, one subject key.

    The two Signal accounts report the same contact as a phone on one and
    a UUID on the other, so a regular holds two user hashes under one
    registered name. Everything that writes or reads a memory routes the
    hash through here first (registry-cache only, no I/O). getattr'd so
    a None / stub registry is a no-op.
    """
    fn = getattr(name_registry, "canonical_hash", None)
    return fn(key) if fn and key else key


class SubjectResolver:
    """Resolve a free-form subject hint from the LLM into a canonical key.

    The bot model writes `remember(subject="David", ...)` — we have to map
    that to either a NameRegistry user_hash (so the woo-chat David and
    the trading-chat David collapse to the same person across all chats
    where they're named David), the literal `__context__` for the room,
    or a free-text slug.

    Name resolution is best-effort: case-insensitive exact match against
    the NameRegistry cache. Ambiguity (two registered users sharing a
    first name) resolves to the speaker when they're one of the matches,
    otherwise to the most recently updated match — never to free-text,
    which would fragment a name that IS registered.
    """

    def __init__(self, name_registry):
        self.name_registry = name_registry

    def resolve(
        self,
        subject_hint: str,
        *,
        sender_phone: Optional[str] = None,
        bot_names: Optional[Iterable[str]] = None,
    ) -> tuple[str, str]:
        """Return (subject_key, subject_label).

        Special hints:
          - "" / "self" / "me" / "the user" / "speaker" → the current sender
            (if `sender_phone` is set), otherwise free-text "self"
          - a phone-tail placeholder ("...4160", "[...4160]") → the sender
          - the running bot's own name/slug/alias (`bot_names`) → SUBJECT_SELF
          - "this chat" / "the room" / "context" / "here" → SUBJECT_CONTEXT
          - exact-match (case-insensitive) registered name → that user_hash
          - 64-char hex → assumed to be a user_hash directly
          - anything else → freetext slug
        """
        s = (subject_hint or "").strip()
        low = s.lower()

        # Bot-self hints. Listed BEFORE the human-self hints below because
        # "myself" is ambiguous and we want the LLM (which is "the bot")
        # to pin it on the bot side. The unambiguous human-side keyword
        # is "speaker" / "the user" — the LLM is told to use those when
        # it means the human in the chat, not "self".
        self_aliases = {
            "yourself", "myself", "the bot", "you", "the assistant",
            "the ai", "bot", "you (the bot)", "self (bot)",
        }
        self_aliases.update(
            n.strip().lower() for n in (bot_names or []) if n and n.strip()
        )
        if low in self_aliases:
            return SUBJECT_SELF, "the bot"

        if low in ("", "self", "me", "the user", "speaker", "the speaker"):
            if sender_phone and self.name_registry is not None:
                from .database import hash_phone
                h = canonical_key(self.name_registry, hash_phone(sender_phone))
                label = self.name_registry.label_for(sender_phone)
                return h, label
            return f"{SUBJECT_FREETEXT_PREFIX}self", "self"

        if low in ("this chat", "the room", "the context", "context",
                   "here", "this group", "this conversation"):
            return SUBJECT_CONTEXT, "this chat"

        # Phone-tail placeholder ("[...4160]", "...4160", "4160") — that's
        # the anonymous label the prompt shows for an unregistered speaker,
        # not a name. Minting freetext:4160 would scatter one person's
        # memories across every tail the model happens to echo.
        if _PHONE_TAIL_RE.match(low) and sender_phone:
            return self.resolve("speaker", sender_phone=sender_phone)

        if is_user_hash(s):
            label = ""
            if self.name_registry is not None:
                label = self.name_registry._cache.get(s, "") or ""
            return s, label or s[:8]

        if self.name_registry is not None:
            cache = getattr(self.name_registry, "_cache", {}) or {}
            matches = [
                (h, name) for h, name in cache.items()
                if name.lower() == low
            ]
            if len(matches) == 1:
                return canonical_key(self.name_registry, matches[0][0]), matches[0][1]
            if len(matches) > 1:
                # Never fall through to freetext for a name that IS
                # registered — that fragments the person's memories.
                # Prefer the speaker; else the most recently updated
                # registration (the cache is loaded ORDER BY updated_at).
                logger.warning(
                    "Ambiguous memory subject %r: %d registered users share "
                    "that name", s, len(matches),
                )
                # All matches share a name, so canonical_key folds any of
                # them onto the same (oldest) hash — the speaker preference
                # below only decides which label wins.
                if sender_phone:
                    from .database import hash_phone
                    h = hash_phone(sender_phone)
                    for match in matches:
                        if match[0] == h:
                            return canonical_key(self.name_registry, h), match[1]
                return (
                    canonical_key(self.name_registry, matches[-1][0]),
                    matches[-1][1],
                )

        return freetext_subject_key(s), s


# ---------------------------------------------------------------------------
# Preamble assembly — auto-injected into the writer LLM's system prompt
# ---------------------------------------------------------------------------

# Preamble cap. Memory rendering is cheap individually but multiplicative
# across many active subjects, so cap total lines hard. Each subject gets
# its most recent rows first, then we fall through to other subjects.
MAX_PREAMBLE_LINES = 30

# Always-visible subjects, in render order. Bot-self FIRST so the model
# reads its own identity before anything else; room context next; then
# the current sender / named subjects are appended downstream.
ALWAYS_INCLUDE_KEYS = (SUBJECT_SELF, SUBJECT_CONTEXT)


def _kind_label(kind: str) -> str:
    return {
        KIND_IDENTITY: "identity",
        KIND_PREFERENCE: "preference",
        KIND_FACT: "fact",
        KIND_EVENT: "event",
    }.get(kind, kind)


def _detect_named_subjects(
    message_text: str, name_registry
) -> list[tuple[str, str]]:
    """Return (user_hash, label) for any registered name appearing in the text.

    Case-insensitive whole-word match against NameRegistry's cache. Used
    to expand the preamble's active-subject set when a user mentions
    someone by name. Skips short (<3-char) names — too many false positives.
    """
    if not message_text or name_registry is None:
        return []
    cache = getattr(name_registry, "_cache", {}) or {}
    if not cache:
        return []
    text = message_text.lower()
    found: list[tuple[str, str]] = []
    for h, name in cache.items():
        if not name or len(name) < 3:
            continue
        n = name.lower()
        if re.search(rf"\b{re.escape(n)}\b", text):
            found.append((h, name))
    return found


async def build_preamble(
    *,
    memory_store: "MemoryStore",
    context_id: int,
    sender_phone: Optional[str],
    sender_user_hash: Optional[str],
    sender_label: Optional[str],
    current_message_text: str = "",
    name_registry=None,
    recent_subject_keys: Optional[list[str]] = None,
    bot_id: Optional[int] = None,
) -> str:
    """Build the memory preamble for the writer LLM's system suffix.

    Active subjects are: the room itself, the current sender, anyone
    `name_registry` recognized in the current message, and (optionally)
    subject keys passed in `recent_subject_keys` for prior speakers in
    the thread. Returns an empty string when there's nothing to inject.
    """
    if context_id is None:
        return ""

    subjects: list[tuple[str, str]] = []
    seen: set[str] = set()

    def _add(key: str, label: str) -> None:
        if not key or key in seen:
            return
        seen.add(key)
        subjects.append((key, label))

    for k in ALWAYS_INCLUDE_KEYS:
        _add(k, _label_for_key(k, name_registry))

    if sender_user_hash:
        _add(canonical_key(name_registry, sender_user_hash),
             sender_label or "the speaker")
    elif sender_phone:
        _add(f"{SUBJECT_FREETEXT_PREFIX}sender",
             sender_label or "the speaker")

    for h, name in _detect_named_subjects(current_message_text, name_registry):
        _add(canonical_key(name_registry, h), name)

    for key in recent_subject_keys or []:
        if not key:
            continue
        key = canonical_key(name_registry, key)
        label = ""
        if name_registry is not None and is_user_hash(key):
            label = name_registry._cache.get(key, "") or ""
        _add(key, label or key[:8])

    blocks: list[str] = []
    line_budget = MAX_PREAMBLE_LINES
    for key, label in subjects:
        if line_budget <= 0:
            break
        rows = await memory_store.list_for_subject(
            context_id=context_id,
            subject_key=key,
            bot_id=bot_id,
        )
        if not rows:
            continue
        rendered: list[str] = []
        for r in rows:
            if line_budget <= 0:
                break
            rendered.append(
                f"  - [{_kind_label(r['kind'])}] {r['content']}"
            )
            line_budget -= 1
        if not rendered:
            continue
        # Refresh user-hash labels from the registry so renames don't leave
        # stale denormalized labels showing in the preamble.
        live_label = _live_label_for_key(key, name_registry) or label
        header = live_label or _label_for_key(key, name_registry)
        blocks.append(f"About {header}:\n" + "\n".join(rendered))

    if not blocks:
        return ""

    intro = (
        "Stored memories scoped to this chat; the first block — \"About you "
        "(the bot)\" — is your own identity, plus views, calls, and "
        "commitments you made earlier; stay consistent with them. "
        "If a memory contradicts what the user just said about themselves, "
        "trust the user and call `remember` (when available) to update it."
    )
    return intro + "\n\n" + "\n\n".join(blocks)


def _label_for_key(key: str, name_registry) -> str:
    if key == SUBJECT_CONTEXT:
        return "this chat"
    if key == SUBJECT_SELF:
        return "you (the bot)"
    if key.startswith(SUBJECT_FREETEXT_PREFIX):
        return key[len(SUBJECT_FREETEXT_PREFIX):].replace("-", " ")
    if is_user_hash(key) and name_registry is not None:
        cached = name_registry._cache.get(key)
        if cached:
            return cached
        return f"...{key[:6]}"
    return key


def _live_label_for_key(key: str, name_registry) -> str:
    """Return the registry-current display name for user-hash keys, else "".

    Used by callers that already have a denormalized label but want to
    prefer the registry's live value when available (renames stay
    visible without rewriting every memory row).
    """
    if not key or name_registry is None:
        return ""
    if not is_user_hash(key):
        return ""
    return name_registry._cache.get(key, "") or ""


# ---------------------------------------------------------------------------
# Tool schemas — exposed to the writer LLM (ask) and the reactor
# ---------------------------------------------------------------------------

REMEMBER_TOOL = {
    "type": "function",
    "function": {
        "name": "remember",
        "description": (
            "Save a memory about a person, the room, or a thing — scoped "
            "to THIS chat. Use when the user (or another speaker) reveals "
            "something durable about themselves or someone else: their "
            "role, preferences, a meaningful fact, or an event worth "
            "recalling later — and for your OWN durable state: a view "
            "you formed, a call or prediction you made, something you "
            "did or committed to. Memories you save are auto-injected into "
            "future replies in this chat when the subject is active. "
            "Skip banter, jokes, and anything the speaker clearly doesn't "
            "want stored. One memory per call — make multiple calls if you "
            "need to record several distinct facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": (
                        "Who/what this memory is about. Options:\n"
                        "  - 'speaker' → the human currently talking. Use "
                        "this for anything they state about THEMSELVES.\n"
                        "  - 'yourself' → YOU, the bot — your identity and "
                        "persona, but also your own opinions, calls you "
                        "made, positions you took, things you did or "
                        "promised in this chat.\n"
                        "  - a registered name (e.g. 'David') → that other "
                        "user, when the speaker is talking about someone "
                        "else\n"
                        "  - 'this chat' → the room itself\n"
                        "  - free-text label → non-Signal entities ('the "
                        "cat', 'their boss')\n"
                        "Names map to that user across all contexts; "
                        "free-text and bot-self memories are local to this "
                        "chat."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": list(KINDS),
                    "description": (
                        "identity = stable trait (sun sign, role, where "
                        "they live, model name, persona). preference = "
                        "what they like/dislike/want. fact = neutral "
                        "durable info that isn't identity or preference. "
                        "event = something that happened on a specific "
                        "occasion."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The memory itself, in one or two sentences. "
                        "Phrase as a stand-alone fact ('Sun in Sag, Moon "
                        "in Virgo, Sag rising'), not as a transcript "
                        "('David said his sun is...'). Future-you will "
                        "see only this string + the kind label."
                    ),
                },
            },
            "required": ["subject", "kind", "content"],
        },
    },
}


RECALL_TOOL = {
    "type": "function",
    "function": {
        "name": "recall",
        "description": (
            "Look up stored memories in THIS chat. Memories about active "
            "speakers are auto-injected into your context already — only "
            "use this tool when you need memories about someone NOT in "
            "the current message, or a free-form search across everyone "
            "('what do we know about astrology in this chat?'). Returns "
            "matching memories with their id and kind."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": (
                        "Optional: limit to a specific subject (a name, "
                        "'this chat', or a free-text label). Omit to "
                        "search across all subjects."
                    ),
                },
                "query": {
                    "type": "string",
                    "description": (
                        "Optional: substring to match against memory "
                        "content. Omit to return everything for the "
                        "subject."
                    ),
                },
            },
        },
    },
}


FORGET_TOOL = {
    "type": "function",
    "function": {
        "name": "forget",
        "description": (
            "Delete a memory by its id (which `recall` returns). Use when "
            "the user asks you to forget something, or when you discover "
            "a stored memory is wrong and you can't fix it with a "
            "corroborating `remember` call."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "integer",
                    "description": "The id of the memory to delete.",
                },
            },
            "required": ["memory_id"],
        },
    },
}


def render_recall_results(
    rows: list[dict], name_registry=None
) -> str:
    """Format memory rows for the LLM as the result of a `recall` tool call.

    When `name_registry` is provided, user-hash subjects are re-rendered
    with the live registry name so a recent rename shows up immediately.
    """
    if not rows:
        return "(no matching memories)"
    lines: list[str] = []
    for r in rows:
        live = _live_label_for_key(r["subject_key"], name_registry)
        label = live or r["subject_label"] or r["subject_key"][:8]
        lines.append(
            f"#{r['id']} [{_kind_label(r['kind'])}] about "
            f"{label}: {r['content']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Tool exposure + dispatch — shared by the writer (ask_command) and the
# deep_think / tool_bot loops so all three see the same memory tools.
# ---------------------------------------------------------------------------


def memory_tool_schemas(store, policy) -> list[dict]:
    """Which memory tools this chat gets.

    `recall` needs a wired store, the per-context memory_enabled master
    switch, and a real (non-default) context row — default rows are
    excluded so writes/reads don't bleed across unregistered DMs sharing
    the default:dm policy. `remember`/`forget` additionally need the
    per-context memory_writes_enabled flag.
    """
    if (
        store is None
        or policy is None
        or getattr(policy, "id", None) is None
        or getattr(policy, "kind", None) == "default"
        or not getattr(policy, "memory_enabled", True)
    ):
        return []
    schemas = [RECALL_TOOL]
    if getattr(policy, "memory_writes_enabled", True):
        schemas.extend((REMEMBER_TOOL, FORGET_TOOL))
    return schemas


def bot_names_of(bot) -> list[str]:
    """Name/slug/aliases of the running bot, for self-subject matching."""
    if bot is None:
        return []
    names = [
        getattr(bot, "display_name", None) or "",
        getattr(bot, "slug", None) or "",
    ]
    names.extend(getattr(bot, "aliases", None) or [])
    return [n for n in names if n]


async def dispatch_memory_tool(
    *,
    store,
    resolver,
    name: str,
    args: dict,
    caller_ctx,
    bot_id: Optional[int] = None,
) -> str:
    """Run one remember/recall/forget call. Returns the tool result text."""
    policy = getattr(caller_ctx, "policy", None) if caller_ctx else None
    if caller_ctx is None:
        return "(memory unavailable: no caller context)"
    if (
        store is None
        or policy is None
        or policy.id is None
        or not getattr(policy, "memory_enabled", True)
    ):
        return "(memory unavailable in this chat)"
    if policy.kind == "default":
        return "(memory unavailable: this chat has no explicit context row)"

    from .database import hash_phone

    sender_phone = getattr(caller_ctx, "sender", None)
    sender_user_hash = hash_phone(sender_phone) if sender_phone else ""
    name_registry = getattr(resolver, "name_registry", None)
    bot_names = bot_names_of(getattr(caller_ctx, "bot", None))

    if name == "recall":
        subject_hint = (args.get("subject") or "").strip()
        query = (args.get("query") or "").strip()
        # Multi-bot scoping: each bot recalls only its own memories
        # (plus legacy NULL-bot rows). Other bots' impressions of the
        # same chat/people stay private to them.
        if subject_hint and resolver is not None:
            key, _ = resolver.resolve(
                subject_hint, sender_phone=sender_phone, bot_names=bot_names,
            )
            if not key:
                return "(could not resolve subject)"
            rows = await store.list_for_subject(
                context_id=policy.id, subject_key=key, bot_id=bot_id,
            )
            if query:
                ql = query.lower()
                rows = [r for r in rows if ql in r["content"].lower()]
        elif query:
            rows = await store.search(
                context_id=policy.id, query=query, limit=12, bot_id=bot_id,
            )
        else:
            rows = await store.list_for_context(
                policy.id, limit=20, bot_id=bot_id,
            )
        return render_recall_results(rows, name_registry=name_registry)

    if name == "remember":
        if not getattr(policy, "memory_writes_enabled", True):
            return "(memory writes disabled for this chat)"
        subject_hint = (args.get("subject") or "").strip()
        kind = (args.get("kind") or "").strip().lower()
        content = (args.get("content") or "").strip()
        if not subject_hint or kind not in KINDS or not content:
            return (
                "ERROR: remember requires non-empty subject, content, "
                f"and kind in {sorted(KINDS)}."
            )
        if resolver is None:
            return "(subject resolver not configured)"
        key, label = resolver.resolve(
            subject_hint, sender_phone=sender_phone, bot_names=bot_names,
        )
        if not key:
            return "(could not resolve subject)"
        mem_id = await store.add(
            context_id=policy.id,
            subject_key=key,
            subject_label=label,
            kind=kind,
            content=content,
            source=SOURCE_EXPLICIT,
            source_user_hash=sender_user_hash,
            source_message_at=time.time(),
            bot_id=bot_id,
        )
        if mem_id is None:
            return "(memory not saved — invalid input)"
        return f"saved memory #{mem_id} about {label or subject_hint}"

    if name == "forget":
        if not getattr(policy, "memory_writes_enabled", True):
            return "(memory writes disabled for this chat)"
        try:
            memory_id = int(args.get("memory_id") or 0)
        except (TypeError, ValueError):
            return "ERROR: forget requires an integer memory_id"
        if memory_id <= 0:
            return "ERROR: memory_id must be a positive integer"
        existing = await store.get(memory_id)
        if not existing or existing.get("context_id") != policy.id:
            return f"(no memory #{memory_id} in this chat)"
        ok = await store.delete(memory_id)
        return f"forgot memory #{memory_id}" if ok else "(forget failed)"

    return f"(unknown memory tool: {name})"
