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
                   created_at, updated_at)

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
  * distinct_speakers  — JSON-encoded set of source_user_hashes that have
                         corroborated this memory. Promotion to confidence
                         1.0 requires at least 2 distinct speakers — a
                         single user can't self-promote a hallucination
                         by repeating the same prompt.

Confidence:
  * 0.6  default for reactor passive-learning writes
  * 1.0  for self / context-level explicit writes
  * 0.7  for explicit writes about third parties (anti-poisoning soft tag —
         the bot may be repeating something someone made up about another
         user; corroboration from a second speaker promotes to 1.0)
On corroboration (same kind/subject re-asserted with similar content), the
existing row's `corroborations` increments and confidence rises toward 1.0.
Reactor rows promote to confidence 1.0 only after corroborations from
PROMOTION_CORROBORATION_THRESHOLD distinct speakers.
"""

import json
import logging
import re
import time
from pathlib import Path
from typing import Optional

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

# Reactor is restricted to slow-changing kinds. `event` is reserved for
# explicit `remember` calls so the reactor doesn't litter with one-shot
# observations like "David said hi today".
REACTOR_ALLOWED_KINDS = (KIND_IDENTITY, KIND_PREFERENCE, KIND_FACT)

SOURCE_REACTOR = "reactor"
SOURCE_EXPLICIT = "explicit"
SOURCE_ADMIN = "admin"

REACTOR_DEFAULT_CONFIDENCE = 0.6
EXPLICIT_SELF_CONFIDENCE = 1.0
# Third-party explicit confidence is intentionally below the
# preamble's "(low confidence)" cutoff (0.8) so memories the bot
# stored about user B from user A's prompt land tagged as soft —
# corroboration from a second speaker raises it to 1.0.
EXPLICIT_THIRD_PARTY_CONFIDENCE = 0.7
EXPLICIT_DEFAULT_CONFIDENCE = EXPLICIT_SELF_CONFIDENCE  # legacy alias

# Promote a reactor-sourced memory to confidence 1.0 once corroborated by
# this many DISTINCT speakers. Single-speaker repetition can't promote —
# stops a single user from self-priming a hallucination across messages.
PROMOTION_CORROBORATION_THRESHOLD = 3

_USER_HASH_RE = re.compile(r"^[0-9a-f]{64}$")
_SLUG_RE = re.compile(r"[^a-z0-9]+")


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
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ctx_subj "
                "ON context_memories(context_id, subject_key)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_mem_ctx_kind "
                "ON context_memories(context_id, kind)"
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
        confidence: float = EXPLICIT_SELF_CONFIDENCE,
        source: str = SOURCE_EXPLICIT,
        source_user_hash: str = "",
        source_message_at: Optional[float] = None,
    ) -> Optional[int]:
        """Insert or corroborate a memory. Returns the row id, or None on bad input.

        Corroboration logic: if a row with the same (context_id, subject_key,
        kind) exists and the new content is a near-duplicate, bump
        corroborations + updated_at and lift confidence rather than inserting
        a duplicate. The whole read+write happens under BEGIN IMMEDIATE so
        two concurrent writers can't race past each other into parallel rows.

        Reactor rows promote to confidence 1.0 only when corroborations come
        from at least PROMOTION_CORROBORATION_THRESHOLD distinct speakers
        (tracked via `distinct_speakers`). Single-speaker repetition can't
        self-promote a hallucination — a different user has to say the same
        thing for confidence to climb.
        """
        if not subject_key or kind not in KINDS or not (content or "").strip():
            return None
        if not 0.0 <= confidence <= 1.0:
            confidence = max(0.0, min(1.0, confidence))

        await self._ensure_initialized()
        now = time.time()
        content = content.strip()
        speaker = source_user_hash or ""

        async with aiosqlite.connect(self.db_path) as db:
            # BEGIN IMMEDIATE acquires the reserved lock at start of read,
            # so a concurrent writer waits instead of read-then-double-insert.
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    """SELECT id, content, confidence, corroborations, source,
                              distinct_speakers
                       FROM context_memories
                       WHERE context_id = ? AND subject_key = ? AND kind = ?""",
                    (context_id, subject_key, kind),
                )
                existing = await cursor.fetchall()

                dup: Optional[tuple] = None  # (id, conf, corr, distinct_json)
                for row in existing:
                    if _is_near_duplicate(row[1], content):
                        dup = (row[0], row[2], row[3], row[5])
                        break

                if dup is not None:
                    dup_id, existing_conf, existing_corr, distinct_json = dup
                    speakers = _decode_speakers(distinct_json)
                    if speaker and speaker not in speakers:
                        speakers.append(speaker)
                    new_corr = existing_corr + 1
                    new_conf = min(1.0, max(existing_conf, confidence))
                    # Promotion gate: at least N DISTINCT speakers must have
                    # corroborated. Reactor and explicit third-party writes
                    # both ride this rule — explicit writes about a third
                    # party that no one else corroborates stay tagged soft.
                    if len(speakers) >= PROMOTION_CORROBORATION_THRESHOLD:
                        new_conf = 1.0
                    elif (
                        source == SOURCE_EXPLICIT
                        and existing_conf < EXPLICIT_SELF_CONFIDENCE
                        and len(speakers) >= 2
                    ):
                        new_conf = max(new_conf, EXPLICIT_SELF_CONFIDENCE)
                    await db.execute(
                        """UPDATE context_memories
                           SET confidence = ?, corroborations = ?,
                               updated_at = ?, subject_label = ?,
                               distinct_speakers = ?
                           WHERE id = ?""",
                        (
                            new_conf, new_corr, now, subject_label,
                            json.dumps(speakers), dup_id,
                        ),
                    )
                    await db.commit()
                    return dup_id

                cursor = await db.execute(
                    """INSERT INTO context_memories
                       (context_id, subject_key, subject_label, kind, content,
                        confidence, source, source_user_hash,
                        source_message_at, distinct_speakers, corroborations,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?)""",
                    (
                        context_id, subject_key, subject_label, kind, content,
                        confidence, source, speaker, source_message_at,
                        json.dumps([speaker] if speaker else []),
                        now, now,
                    ),
                )
                await db.commit()
                return cursor.lastrowid
            except Exception:
                await db.rollback()
                raise

    _SELECT_COLS = (
        "id, subject_key, subject_label, kind, content, confidence, "
        "source, corroborations, created_at, updated_at, "
        "source_user_hash, source_message_at, distinct_speakers"
    )

    async def list_for_subject(
        self,
        *,
        context_id: int,
        subject_key: str,
        kinds: Optional[tuple] = None,
        min_confidence: float = 0.0,
    ) -> list[dict]:
        await self._ensure_initialized()
        sql = (
            f"SELECT {self._SELECT_COLS} "
            "FROM context_memories "
            "WHERE context_id = ? AND subject_key = ? AND confidence >= ?"
        )
        params: list = [context_id, subject_key, min_confidence]
        if kinds:
            sql += f" AND kind IN ({','.join('?' * len(kinds))})"
            params.extend(kinds)
        sql += " ORDER BY confidence DESC, updated_at DESC"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(sql, params)
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def list_for_context(
        self, context_id: int, limit: int = 500
    ) -> list[dict]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}
                    FROM context_memories
                    WHERE context_id = ?
                    ORDER BY subject_label, kind, updated_at DESC
                    LIMIT ?""",
                (context_id, limit),
            )
            rows = await cursor.fetchall()
        return [_row_to_dict(r) for r in rows]

    async def find_similar_for_subject(
        self,
        *,
        context_id: int,
        subject_key: str,
        content: str,
        threshold: float = 0.5,
    ) -> Optional[dict]:
        """Return the first stored memory for `subject_key` whose content
        is a near-duplicate of `content`, regardless of kind. None if
        nothing matches.

        Used by the reactor's pre-write skip so the same fact written
        under a different `kind` (or in a slightly reworded way) doesn't
        land as a parallel row. Distinct from the strict same-kind
        corroboration in `add()` (Jaccard ≥ 0.85): this one uses the
        looser bidirectional containment check (`_is_similar_for_dedup`)
        because we explicitly WANT to catch the cross-kind / rephrasing
        cases the strict path misses.

        Bounded by `LIMIT 50` ordered by recency so a subject with a
        runaway memory list doesn't make every reactor write expensive.
        Recent memories matter more anyway — older near-duplicates can
        coexist if the subject's facts have meaningfully changed.
        """
        if not content or not subject_key:
            return None
        async with db_session(self) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}
                    FROM context_memories
                    WHERE context_id = ? AND subject_key = ?
                    ORDER BY updated_at DESC
                    LIMIT 50""",
                (context_id, subject_key),
            )
            rows = await cursor.fetchall()
        for r in rows:
            existing_content = r[4]
            if _is_similar_for_dedup(content, existing_content, threshold):
                return _row_to_dict(r)
        return None

    async def search(
        self,
        *,
        context_id: int,
        query: str,
        limit: int = 12,
    ) -> list[dict]:
        """Substring match across subject_label and content. Best-effort
        fuzzy retrieval for the `recall` tool — real ranking is the LLM's job.
        """
        await self._ensure_initialized()
        q = (query or "").strip()
        if not q:
            return []
        like = f"%{q}%"
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"""SELECT {self._SELECT_COLS}
                    FROM context_memories
                    WHERE context_id = ?
                      AND (subject_label LIKE ? OR content LIKE ?)
                    ORDER BY confidence DESC, updated_at DESC
                    LIMIT ?""",
                (context_id, like, like, limit),
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
        d["context_id"] = row[13]
        return d

    async def update(
        self,
        memory_id: int,
        *,
        content: Optional[str] = None,
        confidence: Optional[float] = None,
        subject_key: Optional[str] = None,
        subject_label: Optional[str] = None,
    ) -> bool:
        await self._ensure_initialized()
        sets: list[str] = []
        params: list = []
        if content is not None:
            sets.append("content = ?")
            params.append(content.strip())
        if confidence is not None:
            sets.append("confidence = ?")
            params.append(max(0.0, min(1.0, confidence)))
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
        async with aiosqlite.connect(self.db_path) as db:
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
        "corroborations": row[7],
        "created_at": row[8],
        "updated_at": row[9],
        "source_user_hash": row[10] if len(row) > 10 else "",
        "source_message_at": row[11] if len(row) > 11 else None,
        "distinct_speakers": _decode_speakers(
            row[12] if len(row) > 12 else None
        ),
    }


def _decode_speakers(raw) -> list[str]:
    """Decode the JSON-encoded `distinct_speakers` set. Lenient on shape."""
    if not raw:
        return []
    if isinstance(raw, list):
        return [s for s in raw if isinstance(s, str)]
    try:
        decoded = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(decoded, list):
        return []
    return [s for s in decoded if isinstance(s, str)]


def compute_explicit_confidence(
    *,
    subject_key: str,
    sender_user_hash: Optional[str],
) -> float:
    """Return the starting confidence for an explicit `remember` write.

    Self-memories and context-level memories land at full confidence —
    the user is presumably authoritative about themselves and about the
    room. Memories about *other* people (or free-text entities) start
    soft because the writer LLM was just told what to store via prompt
    and may have been manipulated; corroboration from a second speaker
    promotes them to full.
    """
    if not subject_key:
        return EXPLICIT_THIRD_PARTY_CONFIDENCE
    # Self / context / bot-self all land at full confidence: the speaker
    # is presumably authoritative about themselves and the room, and a
    # bot-self memory ("you are Sigil, a tarot reader" / "you are running
    # Claude Opus 4.7") is an admin-or-user declaration *about the bot*
    # that the bot should adopt, not a claim about a third party.
    if subject_key in (SUBJECT_CONTEXT, SUBJECT_SELF):
        return EXPLICIT_SELF_CONFIDENCE
    if sender_user_hash and subject_key == sender_user_hash:
        return EXPLICIT_SELF_CONFIDENCE
    return EXPLICIT_THIRD_PARTY_CONFIDENCE


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
    Above threshold counts as the same memory and bumps the existing row's
    corroboration counter rather than inserting a parallel one. Avoids
    pulling in a real similarity library for what's a back-pocket dedup.
    """
    ta = _tokenize_for_match(a)
    tb = _tokenize_for_match(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    overlap = len(ta & tb) / len(ta | tb)
    return overlap >= threshold


def _is_similar_for_dedup(a: str, b: str, threshold: float = 0.5) -> bool:
    """Looser similarity check used for the reactor's cross-kind pre-write
    skip. Bidirectional containment: if most of either side's tokens
    appear in the other, they're likely rephrasings of the same fact.

    Distinct from `_is_near_duplicate` (Jaccard ≥ 0.85) which is the
    strict same-kind corroboration rule in `MemoryStore.add()`. The
    reactor needs a wider net because it also sneaks duplicates past
    the strict check by writing the same content under a different
    `kind` (fact vs. preference vs. identity) — those don't share a
    WHERE clause in the corroboration query, so they slip through as
    parallel rows.

    Trade-off: this catches more rephrasings ("loves astrology" vs.
    "is into astrology" → forward overlap 0.5) at the cost of some
    false positives ("lives in Brooklyn" vs. "favorite borough is
    Brooklyn"). For passive reactor learning, false-positive skips are
    cheap (we just don't store a new variant) and the threshold is
    tunable per-call.
    """
    ta = _tokenize_for_match(a)
    tb = _tokenize_for_match(b)
    if not ta or not tb:
        return False
    if ta == tb:
        return True
    intersection = len(ta & tb)
    if intersection == 0:
        return False
    forward = intersection / len(ta)
    backward = intersection / len(tb)
    return max(forward, backward) >= threshold


class SubjectResolver:
    """Resolve a free-form subject hint from the LLM into a canonical key.

    The bot model writes `remember(subject="David", ...)` — we have to map
    that to either a NameRegistry user_hash (so the woo-chat David and
    the trading-chat David collapse to the same person across all chats
    where they're named David), the literal `__context__` for the room,
    or a free-text slug.

    Name resolution is best-effort: case-insensitive exact match against
    the NameRegistry cache. Ambiguity (two registered users sharing a
    first name) falls back to free-text — admins can fix the mapping by
    editing the resulting memory's subject in the admin UI.
    """

    def __init__(self, name_registry):
        self.name_registry = name_registry

    def resolve(
        self,
        subject_hint: str,
        *,
        sender_phone: Optional[str] = None,
    ) -> tuple[str, str]:
        """Return (subject_key, subject_label).

        Special hints:
          - "" / "self" / "me" / "the user" / "speaker" → the current sender
            (if `sender_phone` is set), otherwise free-text "self"
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
        if low in (
            "yourself", "myself", "the bot", "you", "the assistant",
            "the ai", "bot", "you (the bot)", "self (bot)",
        ):
            return SUBJECT_SELF, "the bot"

        if low in ("", "self", "me", "the user", "speaker", "the speaker"):
            if sender_phone and self.name_registry is not None:
                from .database import hash_phone
                h = hash_phone(sender_phone)
                label = self.name_registry.label_for(sender_phone)
                return h, label
            return f"{SUBJECT_FREETEXT_PREFIX}self", "self"

        if low in ("this chat", "the room", "the context", "context",
                   "here", "this group", "this conversation"):
            return SUBJECT_CONTEXT, "this chat"

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
                h, name = matches[0]
                return h, name

        return freetext_subject_key(s), s


# ---------------------------------------------------------------------------
# Preamble assembly — auto-injected into the writer LLM's system prompt
# ---------------------------------------------------------------------------

# Preamble cap. Memory rendering is cheap individually but multiplicative
# across many active subjects, so cap total lines hard. Each subject gets
# its highest-confidence rows first, then we fall through to other subjects.
MAX_PREAMBLE_LINES = 30
MIN_PREAMBLE_CONFIDENCE = 0.5

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
        _add(sender_user_hash, sender_label or "the speaker")
    elif sender_phone:
        _add(f"{SUBJECT_FREETEXT_PREFIX}sender",
             sender_label or "the speaker")

    for h, name in _detect_named_subjects(current_message_text, name_registry):
        _add(h, name)

    for key in recent_subject_keys or []:
        if not key:
            continue
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
            min_confidence=MIN_PREAMBLE_CONFIDENCE,
        )
        if not rows:
            continue
        rendered: list[str] = []
        for r in rows:
            if line_budget <= 0:
                break
            kind_part = _kind_label(r["kind"])
            conf = r["confidence"]
            tags: list[str] = []
            if conf < 0.8:
                tags.append(f"confidence {conf:.1f}")
            # Surface provenance so the model can weigh reactor (passive,
            # may have misread the chat) differently from explicit
            # (someone in the chat told the bot to remember this).
            src = r.get("source") or ""
            if src == SOURCE_REACTOR:
                tags.append("observed passively")
            elif src == SOURCE_EXPLICIT:
                tags.append("told to bot")
            tail = f" ({'; '.join(tags)})" if tags else ""
            rendered.append(f"  - [{kind_part}] {r['content']}{tail}")
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
        "Stored memories scoped to this chat. The first block — \"About you "
        "(the bot)\" — is your own identity, persona, and capabilities as "
        "set in this chat; treat it as authoritative. The rest are about "
        "the room itself, the current speaker, and anyone they mentioned. "
        "Use them when relevant; do not parrot them at users unprompted. "
        "If a stored memory contradicts what the user just said about "
        "themselves, trust the user and call `remember` (when available) "
        "to update it."
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
            "recalling later. Memories you save are auto-injected into "
            "future replies in this chat when the subject is active. "
            "Don't save banter, jokes the user is making, anything the "
            "user clearly doesn't want stored, or the same fact you "
            "already saved (corroborations dedupe automatically). One "
            "memory per call — make multiple calls if you need to record "
            "several distinct facts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": (
                        "Who/what this memory is about. Options:\n"
                        "  - a registered name (e.g. 'David') → that user\n"
                        "  - 'speaker' or 'the user' → the human currently "
                        "talking\n"
                        "  - 'yourself' or 'the bot' → YOU, the bot — your "
                        "own identity, persona, model, voice, capabilities. "
                        "These are what the chat has told you about who "
                        "YOU are in this room.\n"
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
            "matching memories with their kind and confidence."
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


# Reactor uses a separate tool — narrower description, restricted kinds, and
# never returns text to the model (the reactor doesn't iterate). Listed
# alongside emoji_react / should_respond when reactor_memory_writes is on.
NOTE_MEMORY_TOOL = {
    "type": "function",
    "function": {
        "name": "note_memory",
        "description": (
            "PASSIVE LEARNING: save a durable memory about someone in "
            "this chat from what they (or someone else) just said. Only "
            "use for slow-changing facts: identity (role, where they "
            "live, astrological/personality identifiers they've named), "
            "preference (what they like or dislike), or general fact. "
            "Skip events, banter, jokes, sarcasm, and anything the "
            "speaker clearly doesn't want recorded. You can call this "
            "alongside emoji_react in the same message — they're not "
            "mutually exclusive. If nothing memory-worthy is here, "
            "don't call this tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "subject": {
                    "type": "string",
                    "description": (
                        "Who the memory is about. Registered names "
                        "('David'), 'this chat' for the room, 'speaker' "
                        "for the human currently talking, or 'yourself' "
                        "for the bot itself (its persona / model / voice "
                        "as set in this chat). Use the registered name "
                        "when the speaker is talking about themselves — "
                        "the system maps it to the right person."
                    ),
                },
                "kind": {
                    "type": "string",
                    "enum": list(REACTOR_ALLOWED_KINDS),
                    "description": (
                        "identity (stable trait), preference (likes/"
                        "dislikes), or fact (neutral durable info)."
                    ),
                },
                "content": {
                    "type": "string",
                    "description": (
                        "The memory in one short sentence, phrased as a "
                        "stand-alone fact."
                    ),
                },
            },
            "required": ["subject", "kind", "content"],
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
        conf = r["confidence"]
        tags: list[str] = []
        if conf < 0.8:
            tags.append(f"confidence {conf:.1f}")
        src = r.get("source") or ""
        if src and src != "explicit":
            tags.append(f"source: {src}")
        tag = f" ({'; '.join(tags)})" if tags else ""
        live = _live_label_for_key(r["subject_key"], name_registry)
        label = live or r["subject_label"] or r["subject_key"][:8]
        lines.append(
            f"#{r['id']} [{_kind_label(r['kind'])}] about "
            f"{label}{tag}: {r['content']}"
        )
    return "\n".join(lines)
