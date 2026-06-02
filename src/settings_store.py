"""
SQLite-backed key/value store for admin-editable settings.

Two layers of configuration:
  * Config.from_env — static bootstrap values (read once at startup)
  * SettingsStore — live-editable overrides written by the admin UI

Settings that require a restart to take effect (provider API keys,
new provider registrations) still live in .env. The admin UI surfaces
them for convenience but marks them as "restart required".
"""

import json
import logging
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

# Keys touched live by the running bot. Changes apply on the next request.
LIVE_KEYS = {
    "bot_name",
    "user_rate_limit",
    "max_message_length",
    "admin_numbers",
    "webhook_secret",
    # LLM (phase 2) — all live
    "llm_enabled",
    "llm_base_url",
    "llm_api_key",
    "llm_model",
    "llm_temperature",
    "llm_max_tokens",
    "llm_system_prompt",
    "llm_timeout_seconds",
    "llm_history_turns",
    "llm_extra_body",
    "llm_provider_order",
    "llm_provider_only",
    "llm_provider_sort",
    "llm_retention_days",
    "llm_augment_commands",
    "llm_augment_prompt",
    "llm_max_tool_rounds",
    "llm_response_style",
    "ask_command_name",
    "group_context_messages",
    # Emoji reactor (secondary cheap-LLM path)
    "reactor_enabled",
    "reactor_model",
    "reactor_max_tokens",
    "reactor_temperature",
    "reactor_extra_body",
    "reactor_system_prompt",
    "reactor_min_length",
    "reactor_sender_cooldown",
    "reactor_group_cooldown",
    "reactor_context_messages",
    # Natural response (piggybacks on the reactor LLM call)
    "natural_response_enabled",
    "natural_response_cooldown",
    "natural_response_extra_prompt",
    # Deep think (separate "smarter" model exposed as a tool to the writer)
    "deep_think_enabled",
    "deep_think_base_url",
    "deep_think_api_key",
    "deep_think_model",
    "deep_think_temperature",
    "deep_think_max_tokens",
    "deep_think_timeout_seconds",
    "deep_think_extra_body",
    "deep_think_system_prompt",
    "deep_think_context_max_chars",
    "deep_think_max_tool_rounds",
    # Cap infrastructure — disabled by default; counters track usage either way
    "deep_think_caps_enabled",
    "deep_think_user_daily_cap",
    "deep_think_group_daily_cap",
    # WSB daily digest (the wsb_digest oracle kind) — all live. The static
    # output dir is env-only (config.wsb_static_dir / WSB_STATIC_DIR).
    "wsb_redlib_base_url",
    "wsb_subreddit",
    "wsb_public_base_url",
    "wsb_indexable",
    "wsb_user_agent",
}

# Legacy keys kept readable so the migration in
# `src/contexts/oracles.py:prepopulate_for_existing_contexts` can carry
# over the old global setting on first boot of the per-context oracles.
# Removed from LIVE_KEYS so the admin settings page stops surfacing
# them — admins manage oracles per-context now via /admin/contexts/<id>.
_LEGACY_DEPRECATED_KEYS = {
    "daily_oracle_enabled",
    "daily_oracle_group_id",
}

# Keys the admin UI exposes but that require a process restart.
RESTART_KEYS = {
    "command_prefix",
    "ALPHAVANTAGE_API_KEY",
    "POLYGON_API_KEY",
    "FINNHUB_API_KEY",
    "TWELVEDATA_API_KEY",
    "FRED_API_KEY",
    "MASSIVE_PRO",
}

# All keys that the admin UI is allowed to write. Anything else is rejected.
ALLOWED_KEYS = LIVE_KEYS | RESTART_KEYS


class SettingsStore:
    """Persistent key/value store with an in-memory read cache.

    Single-process use; an RLock guards cache + write consistency.
    Values are JSON-encoded so lists/bools/ints round-trip cleanly.
    """

    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._lock = threading.RLock()
        self._cache: dict[str, Any] = {}
        # Per-bot, per-role overrides indexed as (bot_id, role) -> {key: value}.
        # Lives alongside the global admin_settings cache; populated on
        # first access from `bot_llm_settings`. Hot reads hit it
        # synchronously — matches admin_settings_'s pattern so callers can
        # use either through the same RLock.
        self._bot_cache: dict[tuple[int, str], dict[str, Any]] = {}
        self._loaded = False
        self._bot_loaded = False

    def _ensure_table(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS admin_settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL
            )
            """
        )
        # bot_llm_settings is created and migrated by BotRegistry;
        # creating it here too keeps SettingsStore usable in tests that
        # don't initialize the registry.
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS bot_llm_settings (
                bot_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                key TEXT NOT NULL,
                value TEXT NOT NULL,
                updated_at REAL NOT NULL,
                PRIMARY KEY (bot_id, role, key)
            )
            """
        )
        conn.commit()

    def _load_all(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table(conn)
            cursor = conn.execute("SELECT key, value FROM admin_settings")
            self._cache = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        self._loaded = True

    def _load_bot_cache(self) -> None:
        """Populate the per-bot cache. Cheap: usually empty in single-bot
        installs, and even with 5 bots × 3 roles × 20 keys the row
        count is < 300."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table(conn)
            cursor = conn.execute(
                "SELECT bot_id, role, key, value FROM bot_llm_settings"
            )
            self._bot_cache = {}
            for bot_id, role, key, value in cursor.fetchall():
                slot = self._bot_cache.setdefault((bot_id, role), {})
                try:
                    slot[key] = json.loads(value)
                except (TypeError, ValueError):
                    slot[key] = value
        self._bot_loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if not self._loaded:
                self._load_all()
            return self._cache.get(key, default)

    # Typed accessors — every live-config read used to repeat
    #   try: int(store.get(key) or default) except (TypeError, ValueError): return default
    # in each consumer. Centralizing here means a missing/garbage value
    # falls back uniformly and `min_value` clamping is one keyword instead
    # of one extra `max(...)` line at every call site.

    def get_int(
        self, key: str, default: int, *, min_value: Optional[int] = None
    ) -> int:
        raw = self.get(key)
        try:
            v = int(raw) if raw not in (None, "") else default
        except (TypeError, ValueError):
            v = default
        if min_value is not None and v < min_value:
            v = min_value
        return v

    def get_float(self, key: str, default: float) -> float:
        raw = self.get(key)
        try:
            return float(raw) if raw not in (None, "") else default
        except (TypeError, ValueError):
            return default

    def get_bool(self, key: str, default: bool = False) -> bool:
        raw = self.get(key)
        if raw is None:
            return default
        return bool(raw)

    def get_str(self, key: str, default: str = "") -> str:
        raw = self.get(key)
        return str(raw) if raw is not None else default

    def get_stripped(self, key: str, default: str = "") -> str:
        return self.get_str(key, default).strip()

    def set(self, key: str, value: Any) -> None:
        if key not in ALLOWED_KEYS:
            raise ValueError(f"Setting '{key}' is not admin-editable")
        with self._lock:
            serialized = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_table(conn)
                conn.execute(
                    """INSERT INTO admin_settings (key, value, updated_at)
                       VALUES (?, ?, ?)
                       ON CONFLICT(key) DO UPDATE SET
                           value = excluded.value,
                           updated_at = excluded.updated_at""",
                    (key, serialized, time.time()),
                )
                conn.commit()
            self._cache[key] = value
            logger.info(f"Setting updated: {key}")

    def delete(self, key: str) -> None:
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_table(conn)
                conn.execute("DELETE FROM admin_settings WHERE key = ?", (key,))
                conn.commit()
            self._cache.pop(key, None)

    def all(self) -> dict[str, Any]:
        with self._lock:
            if not self._loaded:
                self._load_all()
            return dict(self._cache)

    # ── Per-bot, per-role overrides ────────────────────────────────────
    # `bot_llm_settings(bot_id, role, key) -> value` with role in
    # {'writer','reactor','deep_think'}. `resolve_setting` in
    # src/bots/settings.py composes these with global fallback chains
    # (e.g. deep_think -> deep_think_global -> llm_global -> default).
    # We don't validate `key` against an allow-list here: per-role
    # vocabularies are role-specific (writer has llm_*, deep_think has
    # deep_think_*) and the resolver is what binds bot-scoped keys to
    # their global counterparts. Admin UI is the natural gate.

    def get_bot(
        self,
        bot_id: int,
        role: str,
        key: str,
        default: Any = None,
    ) -> Any:
        with self._lock:
            if not self._bot_loaded:
                self._load_bot_cache()
            slot = self._bot_cache.get((bot_id, role))
            if not slot or key not in slot:
                return default
            return slot[key]

    def set_bot(self, bot_id: int, role: str, key: str, value: Any) -> None:
        with self._lock:
            serialized = json.dumps(value)
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_table(conn)
                conn.execute(
                    """INSERT INTO bot_llm_settings
                       (bot_id, role, key, value, updated_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(bot_id, role, key) DO UPDATE SET
                           value = excluded.value,
                           updated_at = excluded.updated_at""",
                    (bot_id, role, key, serialized, time.time()),
                )
                conn.commit()
            slot = self._bot_cache.setdefault((bot_id, role), {})
            slot[key] = value
            logger.info(f"Bot setting updated: bot={bot_id} role={role} key={key}")

    def delete_bot(
        self,
        bot_id: int,
        role: Optional[str] = None,
        key: Optional[str] = None,
    ) -> int:
        """Delete bot-scoped settings. Pass key to remove one, omit it to
        clear an entire (bot, role) slot, omit role to clear every role
        for the bot. Returns number of rows removed."""
        with self._lock:
            with sqlite3.connect(self.db_path) as conn:
                self._ensure_table(conn)
                clauses = ["bot_id = ?"]
                params: list = [bot_id]
                if role is not None:
                    clauses.append("role = ?")
                    params.append(role)
                if key is not None:
                    clauses.append("key = ?")
                    params.append(key)
                cursor = conn.execute(
                    f"DELETE FROM bot_llm_settings WHERE {' AND '.join(clauses)}",
                    params,
                )
                conn.commit()
                removed = cursor.rowcount
            # Update cache: drop matching entries.
            for cache_key in list(self._bot_cache.keys()):
                cb_id, cb_role = cache_key
                if cb_id != bot_id:
                    continue
                if role is not None and cb_role != role:
                    continue
                if key is None:
                    self._bot_cache.pop(cache_key, None)
                else:
                    self._bot_cache.get(cache_key, {}).pop(key, None)
            return removed

    def list_bot(
        self,
        bot_id: int,
        role: Optional[str] = None,
    ) -> dict[str, Any]:
        """Return the bot's settings as either a flat dict (when role is
        given) or a {role: {key: value}} dict (when role is None). Used
        by the admin UI to populate edit forms."""
        with self._lock:
            if not self._bot_loaded:
                self._load_bot_cache()
            if role is not None:
                return dict(self._bot_cache.get((bot_id, role), {}))
            out: dict[str, Any] = {}
            for (b_id, b_role), slot in self._bot_cache.items():
                if b_id == bot_id:
                    out[b_role] = dict(slot)
            return out
