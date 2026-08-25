"""
Per-user nickname registry.

Maps the sha256 hash of a phone number → a human-readable name. The bot
otherwise refers to users by the last four digits of their phone (e.g.
`...4137`), which is privacy-respectful but reads as noise to the LLM and
makes group-thread playback hard to follow.

Names are admin-curated: the registry only exposes set/get/list — there's
no auto-discovery from Signal profile data.

Hash-keyed (not phone-keyed) so the table never holds PII directly. The
hash function matches the one in `database.py` (sha256 of the phone).
"""

import logging
import sqlite3
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .database import db_session

from .database import hash_phone
from .group_log import BOT_SENDER

logger = logging.getLogger(__name__)


class NameRegistry:
    """Async sqlite-backed map of sender hash → display name."""

    def __init__(self, db_path: str = "data/watchlist.db", bot_name: str = "Bot"):
        self.db_path = Path(db_path)
        # Display name for the bot itself. Used to translate the BOT_SENDER
        # sentinel into a real label in group_context lines, so the model
        # sees its own past messages as "[Sigil, 5m ago] ..." rather than
        # an opaque sentinel.
        self.bot_name = bot_name or "Bot"
        self._initialized = False
        # Hot-path cache. Names change rarely, so a simple in-memory dict
        # avoids hitting sqlite for every history-replay attribution.
        self._cache: dict[str, str] = {}
        self._cache_loaded = False
        # Eager sync warm-up. Without this, display_name_sync returns the
        # `...tail` fallback until something async wakes the registry —
        # but the consumers (ask, reactor, history) are all sync and
        # never await display_name(), so the cache would stay cold for
        # the whole bot lifetime under normal traffic. Sync sqlite is
        # used here because __init__ can't be async; the table CREATE is
        # idempotent and tiny so this is safe to do at boot.
        self._warm_cache_sync()

    def _warm_cache_sync(self) -> None:
        """Load names into the cache via blocking sqlite3 — runs once at init."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            with sqlite3.connect(str(self.db_path)) as conn:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS user_names (
                        user_hash TEXT PRIMARY KEY,
                        name TEXT NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                rows = conn.execute(
                    # ORDER BY updated_at: dict preserves insertion order,
                    # so canonical_hash() can fold a person's several
                    # hashes onto their first registration (and
                    # SubjectResolver can pick the latest one for a label).
                    "SELECT user_hash, name FROM user_names "
                    "ORDER BY updated_at"
                ).fetchall()
            self._cache = {r[0]: r[1] for r in rows}
            self._cache_loaded = True
            self._initialized = True  # async path can skip the table creation
            logger.info(f"NameRegistry warm-up loaded {len(self._cache)} names")
        except Exception as e:
            # Don't fail bot startup if warm-up trips — async path will
            # retry on first read/write.
            logger.warning(f"NameRegistry warm-up failed: {e}")

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            from .database import apply_db_pragmas
            await apply_db_pragmas(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS user_names (
                    user_hash TEXT PRIMARY KEY,
                    name TEXT NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()
        self._initialized = True
        await self._load_cache()

    async def _load_cache(self) -> None:
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT user_hash, name FROM user_names ORDER BY updated_at"
            )
            rows = await cursor.fetchall()
        self._cache = {r[0]: r[1] for r in rows}
        self._cache_loaded = True

    def canonical_hash(self, user_hash: str) -> str:
        """Collapse one person's several hashes onto their oldest one.

        Two Signal accounts report the same contact differently (phone
        vs UUID), so a regular ends up with two `user_names` rows sharing
        one name. Reads and writes both route through here so memories
        land on a single key. Cache-only; the cache is loaded ORDER BY
        updated_at, so the first name match is the oldest registration.
        """
        name = (self._cache.get(user_hash) or "").lower()
        if not name:
            return user_hash
        for h, n in self._cache.items():
            if (n or "").lower() == name:
                return h
        return user_hash

    @staticmethod
    def _resolve_hash(*, user_hash: Optional[str], phone: Optional[str]) -> Optional[str]:
        if user_hash:
            return user_hash
        if phone:
            return hash_phone(phone)
        return None

    def display_name_sync(
        self,
        *,
        user_hash: Optional[str] = None,
        phone: Optional[str] = None,
        tail: Optional[str] = None,
    ) -> str:
        """Synchronous lookup against the in-memory cache.

        Returns the registered name if known, otherwise falls back to the
        `...tail` form. Safe to call from non-async paths; just returns the
        fallback if the cache hasn't been warmed yet.
        """
        h = self._resolve_hash(user_hash=user_hash, phone=phone)
        if h and self._cache_loaded:
            name = self._cache.get(h)
            if name:
                return name
        if tail:
            return f"...{tail}"
        if phone:
            return f"...{phone[-4:]}"
        return "...?"

    def label_for(self, phone: Optional[str]) -> str:
        """Phone-only convenience wrapper: returns name or `...tail`.

        Consolidates the same `(tail, registry.display_name_sync, fallback)`
        pattern that ask, reactor, and poll-voter were each reimplementing.
        Swallows lookup errors — there's no caller-meaningful difference
        between "registry unavailable" and "no name registered".

        The BOT_SENDER sentinel resolves to the configured bot name so the
        bot's own past replies (logged into group_log) read as themselves
        in group_context, not as `...___b`.
        """
        if phone == BOT_SENDER:
            return self.bot_name
        tail = (phone or "")[-4:] or "????"
        try:
            return self.display_name_sync(phone=phone, tail=tail)
        except Exception:
            return f"...{tail}"

    async def display_name(
        self,
        *,
        user_hash: Optional[str] = None,
        phone: Optional[str] = None,
        tail: Optional[str] = None,
    ) -> str:
        """Async lookup. Warms the cache on first call."""
        await self._ensure_initialized()
        return self.display_name_sync(user_hash=user_hash, phone=phone, tail=tail)

    async def set_name(
        self,
        name: str,
        *,
        user_hash: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> bool:
        h = self._resolve_hash(user_hash=user_hash, phone=phone)
        if not h:
            return False
        name = (name or "").strip()
        if not name:
            return await self.delete(user_hash=h)
        async with db_session(self) as db:
            await db.execute(
                """INSERT INTO user_names(user_hash, name, updated_at)
                   VALUES(?, ?, ?)
                   ON CONFLICT(user_hash) DO UPDATE SET
                     name = excluded.name,
                     updated_at = excluded.updated_at""",
                (h, name, time.time()),
            )
            await db.commit()
        self._cache[h] = name
        return True

    async def delete(
        self,
        *,
        user_hash: Optional[str] = None,
        phone: Optional[str] = None,
    ) -> bool:
        h = self._resolve_hash(user_hash=user_hash, phone=phone)
        if not h:
            return False
        async with db_session(self) as db:
            await db.execute("DELETE FROM user_names WHERE user_hash = ?", (h,))
            await db.commit()
        self._cache.pop(h, None)
        return True

    async def list_all(self) -> list[dict]:
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT user_hash, name, updated_at FROM user_names ORDER BY name"
            )
            rows = await cursor.fetchall()
        return [
            {"user_hash": r[0], "name": r[1], "updated_at": r[2]} for r in rows
        ]

    async def list_seen(self, limit: int = 50, samples_per_user: int = 4) -> list[dict]:
        """Recent senders the bot has heard from, regardless of whether they
        have a registered name. Sourced from `conversation_turns` (where we
        already store user_hash and sender_tail). Each row also includes a
        few recent message previews and the labels of the contexts they've
        appeared in — the bot otherwise has no way to tell `...4137` apart
        from `...4137` in another group.
        """
        async with db_session(self) as db:
            cursor = await db.execute(
                """
                SELECT user_hash, sender_tail, MAX(created_at) AS last_seen,
                       COUNT(*) AS turn_count
                FROM conversation_turns
                WHERE user_hash IS NOT NULL AND user_hash != ''
                GROUP BY user_hash
                ORDER BY last_seen DESC
                LIMIT ?
                """,
                (limit,),
            )
            agg_rows = await cursor.fetchall()

            # Build context_key (e.g. "group:abc=") -> label map. Strip the
            # "group:" / "dm:" prefix because that's how `contexts.key` is
            # stored.
            cursor = await db.execute("SELECT kind, key, label FROM contexts")
            label_rows = await cursor.fetchall()
            label_map: dict[str, str] = {}
            for kind, key, label in label_rows:
                if kind in ("group", "dm") and key:
                    label_map[f"{kind}:{key}"] = label or "(unnamed)"

            # Pull recent user messages for the senders we just listed. One
            # bounded query, then bucket per user_hash in Python — avoids
            # a per-user SELECT loop.
            hashes = [r[0] for r in agg_rows]
            samples_by_hash: dict[str, list[dict]] = {h: [] for h in hashes}
            contexts_by_hash: dict[str, set[str]] = {h: set() for h in hashes}
            if hashes:
                # `?,?,?,…` placeholder block sized to the input. Capped at
                # `limit` users above so this can't go unbounded.
                placeholders = ",".join(["?"] * len(hashes))
                # Pull more raw rows than we'll display; we'll trim per-user
                # in Python so each user actually gets `samples_per_user`
                # rather than the global most-recent skewing toward a few.
                cursor = await db.execute(
                    f"""
                    SELECT user_hash, content, created_at, context_key
                    FROM conversation_turns
                    WHERE role = 'user'
                      AND user_hash IN ({placeholders})
                    ORDER BY created_at DESC
                    LIMIT ?
                    """,
                    (*hashes, len(hashes) * samples_per_user * 4),
                )
                msg_rows = await cursor.fetchall()
                for h, content, created_at, ctx_key in msg_rows:
                    label = label_map.get(ctx_key, ctx_key) or "(unknown)"
                    bucket = samples_by_hash.setdefault(h, [])
                    if len(bucket) < samples_per_user:
                        bucket.append({
                            "text": content,
                            "created_at": created_at,
                            "context_label": label,
                        })
                    contexts_by_hash.setdefault(h, set()).add(label)

        out: list[dict] = []
        for r in agg_rows:
            user_hash = r[0]
            out.append({
                "user_hash": user_hash,
                "sender_tail": r[1] or "",
                "last_seen": r[2],
                "turn_count": r[3],
                "name": self._cache.get(user_hash),
                "samples": samples_by_hash.get(user_hash, []),
                "contexts": sorted(contexts_by_hash.get(user_hash, set())),
            })
        return out
