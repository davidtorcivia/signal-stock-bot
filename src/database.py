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
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM watchlists WHERE user_hash = ? AND symbol = ?",
                (user_hash, symbol.upper())
            )
            await db.commit()
            return cursor.rowcount > 0
    
    async def get_watchlist(self, user_hash: str) -> list[str]:
        """Get all symbols in user's watchlist, ordered by when they were added."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT symbol FROM watchlists WHERE user_hash = ? ORDER BY added_at",
                (user_hash,)
            )
            rows = await cursor.fetchall()
            return [row[0] for row in rows]
    
    async def clear(self, user_hash: str) -> int:
        """Clear all symbols from user's watchlist. Returns count removed."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "DELETE FROM watchlists WHERE user_hash = ?",
                (user_hash,)
            )
            await db.commit()
            return cursor.rowcount
    
    async def count(self, user_hash: str) -> int:
        """Get count of symbols in user's watchlist."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COUNT(*) FROM watchlists WHERE user_hash = ?",
                (user_hash,)
            )
            row = await cursor.fetchone()
            return row[0] if row else 0
