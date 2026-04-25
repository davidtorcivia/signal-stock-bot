"""SQLite-backed CRUD for MCP server configurations."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from .models import MCPServerConfig

logger = logging.getLogger(__name__)


class MCPRegistry:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS mcp_servers (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    transport TEXT NOT NULL,
                    enabled INTEGER NOT NULL DEFAULT 0,
                    command TEXT,
                    args TEXT NOT NULL DEFAULT '[]',
                    env TEXT NOT NULL DEFAULT '{}',
                    url TEXT,
                    headers TEXT NOT NULL DEFAULT '{}',
                    created_at REAL NOT NULL,
                    updated_at REAL NOT NULL
                )
                """
            )
            await db.commit()
        self._initialized = True

    @staticmethod
    def _row_to_config(row) -> MCPServerConfig:
        return MCPServerConfig(
            id=row[0],
            name=row[1],
            transport=row[2],
            enabled=bool(row[3]),
            command=row[4],
            args=json.loads(row[5] or "[]"),
            env=json.loads(row[6] or "{}"),
            url=row[7],
            headers=json.loads(row[8] or "{}"),
        )

    async def list(self) -> list[MCPServerConfig]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, name, transport, enabled, command, args, env, url, headers "
                "FROM mcp_servers ORDER BY name"
            )
            rows = await cursor.fetchall()
        return [self._row_to_config(r) for r in rows]

    async def get(self, server_id: int) -> Optional[MCPServerConfig]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT id, name, transport, enabled, command, args, env, url, headers "
                "FROM mcp_servers WHERE id = ?",
                (server_id,),
            )
            row = await cursor.fetchone()
        return self._row_to_config(row) if row else None

    async def upsert(self, cfg: MCPServerConfig) -> int:
        """Create or update. Returns the row id."""
        errors = cfg.validate()
        if errors:
            raise ValueError("; ".join(errors))
        await self._ensure_initialized()
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            if cfg.id is None:
                cursor = await db.execute(
                    """INSERT INTO mcp_servers
                       (name, transport, enabled, command, args, env, url, headers, created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        cfg.name,
                        cfg.transport,
                        1 if cfg.enabled else 0,
                        cfg.command,
                        json.dumps(cfg.args),
                        json.dumps(cfg.env),
                        cfg.url,
                        json.dumps(cfg.headers),
                        now,
                        now,
                    ),
                )
                await db.commit()
                return cursor.lastrowid or 0
            else:
                await db.execute(
                    """UPDATE mcp_servers SET
                           name = ?, transport = ?, enabled = ?, command = ?,
                           args = ?, env = ?, url = ?, headers = ?, updated_at = ?
                       WHERE id = ?""",
                    (
                        cfg.name,
                        cfg.transport,
                        1 if cfg.enabled else 0,
                        cfg.command,
                        json.dumps(cfg.args),
                        json.dumps(cfg.env),
                        cfg.url,
                        json.dumps(cfg.headers),
                        now,
                        cfg.id,
                    ),
                )
                await db.commit()
                return cfg.id

    async def delete(self, server_id: int) -> bool:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
            await db.commit()
            return cursor.rowcount > 0
