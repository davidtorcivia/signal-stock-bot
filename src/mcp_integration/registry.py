"""SQLite-backed CRUD for MCP server configurations."""

import json
import logging
import time
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import db_session

from .models import MCPServerConfig

logger = logging.getLogger(__name__)


# MCP servers we ship as defaults. `ensure_defaults` inserts these on
# first boot (idempotent — uniqueness is enforced by the registry's
# UNIQUE constraint on `name`). Admin can disable via the UI; deletion
# would let auto-register recreate on the next boot, so the admin UI
# uses the enabled flag rather than delete for shipped defaults.
#
# Pyodide is pre-installed in the Dockerfile so first-call doesn't
# trigger an npm fetch (would corrupt the stdio MCP handshake the same
# way it does for brave-search). Cache dir lives on the persistent
# volume so wheel downloads (yfinance, statsmodels, etc.) survive
# restarts.
_DEFAULT_SERVERS: tuple[MCPServerConfig, ...] = (
    MCPServerConfig(
        id=None,
        name="pyodide",
        transport="stdio",
        enabled=True,
        command="mcp-pyodide",
        args=[],
        env={"PYODIDE_CACHE_DIR": "/app/data/pyodide-cache"},
    ),
)


class MCPRegistry:
    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            from ..database import apply_db_pragmas
            await apply_db_pragmas(db)
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
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT id, name, transport, enabled, command, args, env, url, headers "
                "FROM mcp_servers ORDER BY name"
            )
            rows = await cursor.fetchall()
        return [self._row_to_config(r) for r in rows]

    async def get(self, server_id: int) -> Optional[MCPServerConfig]:
        async with db_session(self) as db:
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

    async def ensure_defaults(self) -> None:
        """Idempotently register the default MCP servers we ship.

        Called once at boot before `start_all_enabled` so the writer LLM
        gets the python sandbox (and any future shipped defaults)
        without admin setup. Existing rows with the same name are left
        alone — admin edits to command/args/env survive across deploys.
        """
        existing = await self.list()
        by_name = {c.name: c for c in existing}
        for default in _DEFAULT_SERVERS:
            if default.name in by_name:
                continue
            try:
                await self.upsert(default)
                logger.info(
                    f"Auto-registered default MCP server: {default.name}"
                )
            except Exception as e:
                logger.error(
                    f"Failed to auto-register {default.name}: {e}"
                )

    async def delete(self, server_id: int) -> bool:
        async with db_session(self) as db:
            cursor = await db.execute("DELETE FROM mcp_servers WHERE id = ?", (server_id,))
            await db.commit()
            return cursor.rowcount > 0
