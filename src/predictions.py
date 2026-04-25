"""
Prediction registry — log claims, follow up at the deadline, keep score.

Each prediction is owned by one user (hashed phone) and lives in one
context (group or DM, by `context_key`). The bot posts the resolution
verdict back to that context when the deadline fires.

Resolution paths:
  1. **Structured (ticker+threshold+direction)** — the prediction parser
     extracted a stock-price claim. The auto-resolver queries the live
     quote, compares, posts the verdict. Most precise path.
  2. **LLM verdict** — for free-form claims, the resolver asks the LLM
     to judge using its knowledge. May come back "unclear" if the model
     can't tell.
  3. **Manual** — anyone in the chat can `!resolve <id> right|wrong|unclear`
     before or after the deadline.

Status transitions:
  pending → resolved   (verdict in {right, wrong, unclear})
  pending → expired    (auto-resolver gave up; preserved for the leaderboard)

The leaderboard counts only `resolved` rows where verdict ∈ {right, wrong}.
`unclear` and `expired` are excluded — they shouldn't punish or reward
anyone.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiosqlite

logger = logging.getLogger(__name__)


VERDICT_RIGHT = "right"
VERDICT_WRONG = "wrong"
VERDICT_UNCLEAR = "unclear"
VALID_VERDICTS = {VERDICT_RIGHT, VERDICT_WRONG, VERDICT_UNCLEAR}

STATUS_PENDING = "pending"
STATUS_RESOLVED = "resolved"
STATUS_EXPIRED = "expired"

# Single source of truth for verdict glyphs — used in resolver auto-posts
# and in the manual !resolve confirmation. Defining it here so command
# code can import it without depending on the resolver module.
VERDICT_EMOJI = {
    VERDICT_RIGHT: "✅",
    VERDICT_WRONG: "❌",
    VERDICT_UNCLEAR: "🤷",
}


@dataclass
class Prediction:
    id: int
    user_hash: str
    user_label: str
    group_id: Optional[str]
    context_key: str
    claim: str
    deadline_utc: float
    created_at: float
    ticker: Optional[str]
    threshold: Optional[float]
    direction: Optional[str]   # "above" | "below"
    status: str
    verdict: Optional[str]
    resolution_note: Optional[str]
    resolved_at: Optional[float]
    resolver_user_hash: Optional[str]

    @property
    def is_structured(self) -> bool:
        return bool(self.ticker and self.threshold is not None and self.direction)


class PredictionStore:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialized = False

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute("""
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_hash TEXT NOT NULL,
                    user_label TEXT NOT NULL,
                    group_id TEXT,
                    context_key TEXT NOT NULL,
                    claim TEXT NOT NULL,
                    deadline_utc REAL NOT NULL,
                    created_at REAL NOT NULL,
                    ticker TEXT,
                    threshold REAL,
                    direction TEXT,
                    -- status values: STATUS_PENDING / STATUS_RESOLVED / STATUS_EXPIRED
                    -- (kept as literal in SQL alongside the module constants)
                    status TEXT NOT NULL DEFAULT 'pending',
                    verdict TEXT,
                    resolution_note TEXT,
                    resolved_at REAL,
                    resolver_user_hash TEXT
                )
            """)
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_status_deadline "
                "ON predictions(status, deadline_utc)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_user "
                "ON predictions(user_hash, status)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_context "
                "ON predictions(context_key, status)"
            )
            # Covers the leaderboard's GROUP BY user_hash + verdict CASEs
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_leaderboard "
                "ON predictions(context_key, user_hash, verdict)"
            )
            await db.commit()
        self._initialized = True

    @staticmethod
    def _row_to_pred(row) -> Prediction:
        return Prediction(
            id=row[0],
            user_hash=row[1],
            user_label=row[2],
            group_id=row[3],
            context_key=row[4],
            claim=row[5],
            deadline_utc=row[6],
            created_at=row[7],
            ticker=row[8],
            threshold=row[9],
            direction=row[10],
            status=row[11],
            verdict=row[12],
            resolution_note=row[13],
            resolved_at=row[14],
            resolver_user_hash=row[15],
        )

    _COLS = (
        "id, user_hash, user_label, group_id, context_key, claim, "
        "deadline_utc, created_at, ticker, threshold, direction, "
        "status, verdict, resolution_note, resolved_at, resolver_user_hash"
    )

    async def create(
        self,
        *,
        user_hash: str,
        user_label: str,
        group_id: Optional[str],
        context_key: str,
        claim: str,
        deadline_utc: float,
        ticker: Optional[str] = None,
        threshold: Optional[float] = None,
        direction: Optional[str] = None,
    ) -> int:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT INTO predictions
                   (user_hash, user_label, group_id, context_key, claim,
                    deadline_utc, created_at, ticker, threshold, direction,
                    status)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'pending')""",
                (user_hash, user_label, group_id, context_key, claim,
                 deadline_utc, time.time(), ticker, threshold, direction),
            )
            await db.commit()
            return cursor.lastrowid or 0

    async def get(self, pred_id: int) -> Optional[Prediction]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {self._COLS} FROM predictions WHERE id = ?",
                (pred_id,),
            ) as cur:
                row = await cur.fetchone()
        return self._row_to_pred(row) if row else None

    async def list_for_user(
        self,
        user_hash: str,
        *,
        context_key: Optional[str] = None,
        only_pending: bool = True,
        limit: int = 50,
    ) -> list[Prediction]:
        await self._ensure_initialized()
        clauses = ["user_hash = ?"]
        params: list = [user_hash]
        if context_key is not None:
            clauses.append("context_key = ?")
            params.append(context_key)
        if only_pending:
            clauses.append("status = 'pending'")
        where = " AND ".join(clauses)
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {self._COLS} FROM predictions WHERE {where} "
                f"ORDER BY deadline_utc ASC LIMIT ?",
                (*params, limit),
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_pred(r) for r in rows]

    async def list_due(self, *, now_utc: Optional[float] = None) -> list[Prediction]:
        """Return pending predictions whose deadline has passed."""
        await self._ensure_initialized()
        cutoff = now_utc if now_utc is not None else time.time()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                f"SELECT {self._COLS} FROM predictions "
                f"WHERE status = 'pending' AND deadline_utc <= ? "
                f"ORDER BY deadline_utc ASC",
                (cutoff,),
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_pred(r) for r in rows]

    async def resolve(
        self,
        pred_id: int,
        *,
        verdict: str,
        note: Optional[str] = None,
        resolver_user_hash: Optional[str] = None,
    ) -> bool:
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid verdict: {verdict!r}")
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """UPDATE predictions
                   SET status = 'resolved',
                       verdict = ?,
                       resolution_note = ?,
                       resolved_at = ?,
                       resolver_user_hash = ?
                   WHERE id = ? AND status = 'pending'""",
                (verdict, note, time.time(), resolver_user_hash, pred_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def expire(self, pred_id: int, *, note: Optional[str] = None) -> bool:
        """Mark as expired — auto-resolver tried and gave up. Doesn't count."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """UPDATE predictions
                   SET status = 'expired',
                       resolution_note = ?,
                       resolved_at = ?
                   WHERE id = ? AND status = 'pending'""",
                (note, time.time(), pred_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def leaderboard(
        self, context_key: str, *, limit: int = 20
    ) -> list[dict]:
        """Per-user accuracy in this chat. Excludes 'unclear' and 'expired'.

        The `label` field is a *fallback* — the most recent stored label
        for that user. Callers should prefer resolving from the live
        NameRegistry so renamed users show their current name.
        """
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT user_hash,
                          (SELECT user_label FROM predictions p2
                             WHERE p2.user_hash = p.user_hash
                             ORDER BY created_at DESC LIMIT 1) AS label,
                          SUM(CASE WHEN verdict='right' THEN 1 ELSE 0 END) AS rights,
                          SUM(CASE WHEN verdict='wrong' THEN 1 ELSE 0 END) AS wrongs,
                          SUM(CASE WHEN status='pending' THEN 1 ELSE 0 END) AS pending
                   FROM predictions p
                   WHERE context_key = ?
                   GROUP BY user_hash
                   ORDER BY rights * 1.0 / (rights + wrongs + 0.001) DESC,
                            rights DESC""",
                (context_key,),
            ) as cur:
                rows = await cur.fetchall()
        out = []
        for r in rows:
            user_hash, label, rights, wrongs, pending = r
            judged = (rights or 0) + (wrongs or 0)
            out.append({
                "user_hash": user_hash,
                "label": label,
                "right": rights or 0,
                "wrong": wrongs or 0,
                "pending": pending or 0,
                "accuracy": (rights / judged) if judged else None,
            })
        return out[:limit]

    async def user_record(self, user_hash: str, context_key: str) -> dict:
        """Single-user record for inline display ("David's record: 7/12, 58%")."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                """SELECT
                       SUM(CASE WHEN verdict='right' THEN 1 ELSE 0 END) AS rights,
                       SUM(CASE WHEN verdict='wrong' THEN 1 ELSE 0 END) AS wrongs
                   FROM predictions
                   WHERE user_hash = ? AND context_key = ?""",
                (user_hash, context_key),
            ) as cur:
                row = await cur.fetchone()
        rights = (row[0] if row else 0) or 0
        wrongs = (row[1] if row else 0) or 0
        judged = rights + wrongs
        return {
            "right": rights,
            "wrong": wrongs,
            "accuracy": (rights / judged) if judged else None,
        }
