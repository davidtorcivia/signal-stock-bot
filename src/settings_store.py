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
from typing import Any

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
    "llm_retention_days",
    "llm_augment_commands",
    "llm_augment_prompt",
    "llm_max_tool_rounds",
    "ask_command_name",
    "group_context_messages",
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
        self._loaded = False

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
        conn.commit()

    def _load_all(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            self._ensure_table(conn)
            cursor = conn.execute("SELECT key, value FROM admin_settings")
            self._cache = {row[0]: json.loads(row[1]) for row in cursor.fetchall()}
        self._loaded = True

    def get(self, key: str, default: Any = None) -> Any:
        with self._lock:
            if not self._loaded:
                self._load_all()
            return self._cache.get(key, default)

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
