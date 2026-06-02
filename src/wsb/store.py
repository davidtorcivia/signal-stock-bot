"""
wsb_digests table — one canonical row per day.

Both render targets read from here: the static daily page and the Signal chat
teaser. The structured tally + posts are stored as a JSON payload so the page
can render exact tables (the model only writes prose), and so we can compute
longitudinal ticker trends across days.

Mirrors the OracleStore idiom: lazy `_ensure_initialized`, `db_session(self)`,
schema via `CREATE TABLE IF NOT EXISTS` (no migration framework in this repo).
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import apply_db_pragmas, db_session

logger = logging.getLogger(__name__)


@dataclass
class WSBDigestRecord:
    date: str                       # YYYY-MM-DD (UTC), unique
    subreddit: str
    headline: str
    teaser: str
    body_md: str                    # the deep-think analysis (markdown)
    page_url: str
    tickers: list[dict] = field(default_factory=list)
    posts: list[dict] = field(default_factory=list)
    # Per-ticker price-spark metadata for the page grid: [{symbol, price,
    # change_percent, lean}]. The PNG lives at charts/<date>/<symbol>.png; only
    # symbols whose chart actually rendered appear here.
    charts: list[dict] = field(default_factory=list)
    posts_scanned: int = 0
    comments_scanned: int = 0
    generated_at: float = 0.0       # epoch seconds
    posted_ok: bool = False
    id: Optional[int] = None

    def payload(self) -> dict:
        return {
            "tickers": self.tickers,
            "posts": self.posts,
            "charts": self.charts,
            "posts_scanned": self.posts_scanned,
            "comments_scanned": self.comments_scanned,
        }


class WSBDigestStore:
    """SQLite-backed daily digest archive."""

    def __init__(self, db_path: str = "data/watchlist.db"):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await apply_db_pragmas(db)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS wsb_digests (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        date TEXT NOT NULL UNIQUE,
                        subreddit TEXT NOT NULL DEFAULT 'wallstreetbets',
                        generated_at REAL NOT NULL,
                        headline TEXT NOT NULL DEFAULT '',
                        teaser TEXT NOT NULL DEFAULT '',
                        body_md TEXT NOT NULL DEFAULT '',
                        page_url TEXT NOT NULL DEFAULT '',
                        posted_ok INTEGER NOT NULL DEFAULT 0,
                        payload_json TEXT NOT NULL DEFAULT '{}',
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_wsb_date ON wsb_digests(date DESC)"
                )
                await db.commit()
            self._initialized = True

    _COLS = (
        "id, date, subreddit, generated_at, headline, teaser, body_md, "
        "page_url, posted_ok, payload_json"
    )

    @staticmethod
    def _row_to_record(row) -> WSBDigestRecord:
        try:
            payload = json.loads(row[9]) if row[9] else {}
        except (ValueError, TypeError):
            payload = {}
        return WSBDigestRecord(
            id=row[0],
            date=row[1],
            subreddit=row[2],
            generated_at=row[3],
            headline=row[4],
            teaser=row[5],
            body_md=row[6],
            page_url=row[7],
            posted_ok=bool(row[8]),
            tickers=payload.get("tickers", []),
            posts=payload.get("posts", []),
            charts=payload.get("charts", []),
            posts_scanned=payload.get("posts_scanned", 0),
            comments_scanned=payload.get("comments_scanned", 0),
        )

    async def upsert(self, rec: WSBDigestRecord) -> int:
        """Insert or replace the row for rec.date. Returns the row id."""
        await self._ensure_initialized()
        now = time.time()
        gen = rec.generated_at or now
        payload_json = json.dumps(rec.payload(), ensure_ascii=False)
        async with aiosqlite.connect(self.db_path) as db:
            await apply_db_pragmas(db)
            cursor = await db.execute(
                """
                INSERT INTO wsb_digests
                    (date, subreddit, generated_at, headline, teaser, body_md,
                     page_url, posted_ok, payload_json, created_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(date) DO UPDATE SET
                    subreddit=excluded.subreddit,
                    generated_at=excluded.generated_at,
                    headline=excluded.headline,
                    teaser=excluded.teaser,
                    body_md=excluded.body_md,
                    page_url=excluded.page_url,
                    posted_ok=excluded.posted_ok,
                    payload_json=excluded.payload_json,
                    updated_at=excluded.updated_at
                """,
                (
                    rec.date, rec.subreddit, gen, rec.headline, rec.teaser,
                    rec.body_md, rec.page_url, 1 if rec.posted_ok else 0,
                    payload_json, now, now,
                ),
            )
            await db.commit()
            rowid = cursor.lastrowid or 0
        if not rowid:
            existing = await self.get_by_date(rec.date)
            rowid = (existing.id or 0) if existing else 0
        rec.id = rowid
        return rowid

    async def mark_posted(self, date: str, posted_ok: bool) -> None:
        async with db_session(self) as db:
            await db.execute(
                "UPDATE wsb_digests SET posted_ok = ?, updated_at = ? WHERE date = ?",
                (1 if posted_ok else 0, time.time(), date),
            )
            await db.commit()

    async def get_by_date(self, date: str) -> Optional[WSBDigestRecord]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._COLS} FROM wsb_digests WHERE date = ?", (date,)
            )
            row = await cursor.fetchone()
        return self._row_to_record(row) if row else None

    async def recent(self, limit: int = 60) -> list[WSBDigestRecord]:
        """Newest-first, for the index page and trend computation."""
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._COLS} FROM wsb_digests ORDER BY date DESC LIMIT ?",
                (limit,),
            )
            rows = await cursor.fetchall()
        return [self._row_to_record(r) for r in rows]
