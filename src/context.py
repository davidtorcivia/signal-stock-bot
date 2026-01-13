"""
Context management for NLP interactions.
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


@dataclass
class Context:
    """Conversation context state."""
    user_hash: str
    last_symbol: Optional[str] = None
    last_intent: Optional[str] = None
    updated_at: float = 0.0

    @property
    def is_stale(self) -> bool:
        """Check if context is older than 5 minutes."""
        return (time.time() - self.updated_at) > 300


class ContextManager:
    """
    Manages conversation context per user.
    Backed by SQLite for persistence across restarts.
    """
    
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        """Create tables if they don't exist."""
        if self._initialized:
            return
        
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS conversation_context (
                    user_hash TEXT PRIMARY KEY,
                    last_symbol TEXT,
                    last_intent TEXT,
                    updated_at REAL
                )
            """)
            await db.commit()
        
        self._initialized = True
        logger.debug("Context manager initialized")

    async def get_context(self, user_hash: str) -> Context:
        """Get active context for user."""
        await self._ensure_initialized()
        
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT last_symbol, last_intent, updated_at FROM conversation_context WHERE user_hash = ?",
                (user_hash,)
            )
            row = await cursor.fetchone()
            
            if row:
                ctx = Context(
                    user_hash=user_hash,
                    last_symbol=row[0],
                    last_intent=row[1],
                    updated_at=row[2]
                )
                if ctx.is_stale:
                    # Clear stale context, but return clear object
                    return Context(user_hash=user_hash)
                return ctx
            
            return Context(user_hash=user_hash)

    async def update_context(
        self, 
        user_hash: str, 
        symbol: Optional[str] = None, 
        intent: Optional[str] = None
    ) -> None:
        """Update context components."""
        await self._ensure_initialized()
        
        # Get existing to merge if needed, or just overwrite?
        # Usually we want to overwrite symbol if new one provided.
        # If symbol is None, keep existing? No, typically context updates mean "this is the new focus".
        # But if I say "Show RSI", I inherit symbol. If I say "Chart Apple", I update symbol.
        
        # Strategy:
        # 1. Fetch existing
        # 2. Merge values (if new is provided, use it. if None, keep old?)
        # User requested: "update_context(user_id, symbol=None, intent=None): Updates state."
        # If I only provide intent, I probably want to keep the symbol.
        
        current = await self.get_context(user_hash)
        
        new_symbol = symbol if symbol else current.last_symbol
        new_intent = intent if intent else current.last_intent
        
        # If strictly None explicitly passed implies "keep"?
        # Or do we want to support clearing?
        # For now, presume partial updates merge.
        
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """INSERT OR REPLACE INTO conversation_context 
                   (user_hash, last_symbol, last_intent, updated_at)
                   VALUES (?, ?, ?, ?)""",
                (user_hash, new_symbol, new_intent, time.time())
            )
            await db.commit()
