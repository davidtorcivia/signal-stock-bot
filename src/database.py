"""
Database module for persistent storage.

Currently provides:
- WatchlistDB: Per-user watchlist storage with hashed phone numbers
"""

import hashlib
import logging
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


def hash_phone(phone: str) -> str:
    """Hash phone number for privacy-safe storage."""
    return hashlib.sha256(phone.encode()).hexdigest()


async def apply_db_pragmas(db) -> None:
    """Apply tuning pragmas. WAL is sticky on the file (set once, persists);
    synchronous=NORMAL and foreign_keys are per-connection. Cheap to call
    repeatedly in `_ensure_initialized`. Concrete wins: writers no longer
    block readers (WAL), write fsync drops from per-commit to per-checkpoint
    (NORMAL), and FK constraints declared on tables (currently
    `bot_llm_settings.bot_id REFERENCES bots(id) ON DELETE CASCADE`) are
    actually enforced — without `foreign_keys = ON` SQLite parses the
    constraint then silently ignores it, leaving rows orphaned on bot
    deletion if the app-level cleanup is ever skipped. Safe to enable
    everywhere because it's the only FK in the schema."""
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA synchronous=NORMAL")
    await db.execute("PRAGMA foreign_keys=ON")


class _DBContext:
    """Async context manager that ensures the store is initialized then
    yields an aiosqlite connection. Replaces the
        async with db_session(self) as db:
            ...
    pattern at every store call site. Bound to a store via
    `db_session(store)`; commits are still the caller's responsibility
    (so read-only queries don't pay the commit cost)."""

    def __init__(self, store) -> None:
        self._store = store
        self._cm = None
        self._db = None

    async def __aenter__(self):
        await self._store._ensure_initialized()
        self._cm = aiosqlite.connect(self._store.db_path)
        self._db = await self._cm.__aenter__()
        # Apply pragmas on every checkout. WAL is file-sticky so this
        # is a no-op there, but `synchronous` and `foreign_keys` are
        # per-connection — without this every db_session connection
        # would open with foreign_keys=OFF and silently skip FK
        # cascades.
        await apply_db_pragmas(self._db)
        return self._db

    async def __aexit__(self, exc_type, exc, tb):
        cm = self._cm
        self._db = None
        self._cm = None
        if cm is not None:
            return await cm.__aexit__(exc_type, exc, tb)
        return False


def db_session(store) -> "_DBContext":
    """Open a connection on `store.db_path` after ensuring the schema
    exists. The store must expose `db_path` and `_ensure_initialized`.

        async with db_session(self) as db:
            await db.execute("...")
            await db.commit()
    """
    return _DBContext(store)


class WatchlistDB:
    """
    SQLite-backed watchlist storage.
    
    Users are identified by SHA-256 hash of their phone number.
    Symbols are stored uppercase and deduplicated per user.
    """
    
    MAX_SYMBOLS_PER_USER = 50
    
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return
        
        # Ensure parent directory exists
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await apply_db_pragmas(db)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS watchlists (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    UNIQUE(user_hash, symbol)
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_user_hash
                ON watchlists(user_hash)
            """)
            await db.commit()
        
        self._initialized = True
        logger.info(f"Watchlist database initialized at {self.db_path}")
    
    async def add_symbols(self, user_hash: str, symbols: list[str]) -> tuple[int, list[str]]:
        """
        Add symbols to user's watchlist.
        
        Returns:
            Tuple of (count added, list of symbols that were skipped due to limit)
        """
        await self._ensure_initialized()
        
        current_count = await self.count(user_hash)
        remaining_capacity = self.MAX_SYMBOLS_PER_USER - current_count
        
        if remaining_capacity <= 0:
            return 0, symbols
        
        # Only add up to remaining capacity
        symbols_to_add = [s.upper() for s in symbols[:remaining_capacity]]
        skipped = [s.upper() for s in symbols[remaining_capacity:]]
        
        added = 0
        async with aiosqlite.connect(self.db_path) as db:
            for symbol in symbols_to_add:
                try:
                    await db.execute(
                        "INSERT INTO watchlists (user_hash, symbol) VALUES (?, ?)",
                        (user_hash, symbol)
                    )
                    added += 1
                except aiosqlite.IntegrityError:
                    # Symbol already exists for this user
                    pass
            await db.commit()
        
        return added, skipped
    
    async def remove_symbol(self, user_hash: str, symbol: str) -> bool:
        """Remove a symbol from user's watchlist. Returns True if removed."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM watchlists WHERE user_hash = ? AND symbol = ?",
                (user_hash, symbol.upper())
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def get_watchlist(self, user_hash: str) -> list[str]:
        """Get all symbols in user's watchlist, ordered by when they were added."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT symbol FROM watchlists WHERE user_hash = ? ORDER BY added_at",
                (user_hash,)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def clear(self, user_hash: str) -> int:
        """Clear all symbols from user's watchlist. Returns count removed."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM watchlists WHERE user_hash = ?",
                (user_hash,)
            )
            await db.commit()
            return cursor.rowcount
    
    async def count(self, user_hash: str) -> int:
        """Get count of symbols in user's watchlist."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM watchlists WHERE user_hash = ?",
                (user_hash,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0


class AlertsDB:
    """
    SQLite-backed price alerts storage.
    
    Alerts are context-aware - they notify in the same context (DM or group)
    where they were set.
    """
    
    MAX_ALERTS_PER_USER = 20
    
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
    
    async def _ensure_initialized(self) -> None:
        """Create alerts table if it doesn't exist."""
        if self._initialized:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await apply_db_pragmas(db)
            await db.execute("""
                CREATE TABLE IF NOT EXISTS alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT NOT NULL,
                    user_phone TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    condition TEXT NOT NULL,
                    target_value REAL NOT NULL,
                    group_id TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    triggered_at TIMESTAMP,
                    active INTEGER DEFAULT 1
                )
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_user 
                ON alerts(user_hash, active)
            """)
            await db.execute("""
                CREATE INDEX IF NOT EXISTS idx_alerts_active 
                ON alerts(active, symbol)
            """)
            await db.commit()
        
        self._initialized = True
        logger.info("Alerts database initialized")
    
    async def add_alert(
        self, 
        user_hash: str,
        user_phone: str,
        symbol: str, 
        condition: str, 
        target_value: float,
        group_id: Optional[str] = None
    ) -> Optional[int]:
        """
        Add a price alert.
        
        Args:
            user_hash: Hashed phone number for identification
            user_phone: Actual phone for notifications
            symbol: Stock symbol
            condition: 'above', 'below', or 'change_pct'
            target_value: Target price or percentage
            group_id: Group to notify in (None = DM)
        
        Returns:
            Alert ID if created, None if limit reached
        """
        await self._ensure_initialized()
        
        # Check limit
        count = await self.count_active(user_hash)
        if count >= self.MAX_ALERTS_PER_USER:
            return None
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO alerts 
                   (user_hash, user_phone, symbol, condition, target_value, group_id)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (user_hash, user_phone, symbol.upper(), condition, target_value, group_id)
            )
            await db.commit()
            return cursor.lastrowid
    
    async def get_active_alerts(self, user_hash: str) -> list[dict]:
        """Get all active alerts for a user."""
        async with db_session(self) as db:
            cursor = await db.execute(
                """SELECT id, symbol, condition, target_value, group_id, created_at
                   FROM alerts 
                   WHERE user_hash = ? AND active = 1
                   ORDER BY created_at""",
                (user_hash,)
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "symbol": row[1],
                    "condition": row[2],
                    "target_value": row[3],
                    "group_id": row[4],
                    "created_at": row[5],
                }
                for row in rows
            ]
    
    async def get_all_active_alerts(self) -> list[dict]:
        """Get all active alerts across all users (for background worker)."""
        async with db_session(self) as db:
            cursor = await db.execute(
                """SELECT id, user_hash, user_phone, symbol, condition, target_value, group_id
                   FROM alerts 
                   WHERE active = 1"""
            )
            rows = await cursor.fetchall()
            return [
                {
                    "id": row[0],
                    "user_hash": row[1],
                    "user_phone": row[2],
                    "symbol": row[3],
                    "condition": row[4],
                    "target_value": row[5],
                    "group_id": row[6],
                }
                for row in rows
            ]
    
    async def remove_alert(self, alert_id: int, user_hash: Optional[str] = None) -> bool:
        """
        Remove an alert.
        
        Args:
            alert_id: Alert ID
            user_hash: If provided, only remove if owned by this user (security)
        """
        async with db_session(self) as db:
            if user_hash:
                cursor = await db.execute(
                    "DELETE FROM alerts WHERE id = ? AND user_hash = ?",
                    (alert_id, user_hash)
                )
            else:
                # Admin removal - no user check
                cursor = await db.execute(
                    "DELETE FROM alerts WHERE id = ?",
                    (alert_id,)
                )
            await db.commit()
            return cursor.rowcount > 0
    
    async def trigger_alert(self, alert_id: int) -> bool:
        """Mark an alert as triggered (deactivates it)."""
        async with db_session(self) as db:
            cursor = await db.execute(
                """UPDATE alerts 
                   SET active = 0, triggered_at = CURRENT_TIMESTAMP
                   WHERE id = ?""",
                (alert_id,)
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def count_active(self, user_hash: str) -> int:
        """Count active alerts for a user."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM alerts WHERE user_hash = ? AND active = 1",
                (user_hash,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
    
    async def clear_user_alerts(self, user_hash: str) -> int:
        """Clear all alerts for a user."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM alerts WHERE user_hash = ?",
                (user_hash,)
            )
            await db.commit()
            return cursor.rowcount

