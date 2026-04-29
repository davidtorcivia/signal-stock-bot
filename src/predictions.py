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

from .database import db_session

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
            from .database import apply_db_pragmas
            await apply_db_pragmas(db)
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
            # Partial indexes on pending rows. Resolved/expired predictions
            # never get scanned by the resolver cron or the upcoming-list
            # queries, so we keep them out of the index entirely. Index
            # stays small forever even as the table accumulates years of
            # historical resolved predictions — the cost of `list_due`
            # doesn't degrade with table size.
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_pending_deadline "
                "ON predictions(deadline_utc) WHERE status = 'pending'"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_pred_pending_user "
                "ON predictions(user_hash, deadline_utc) WHERE status = 'pending'"
            )
            # Drop the old non-partial index — the partial ones above are
            # strictly better for every query that filters on status='pending'.
            await db.execute("DROP INDEX IF EXISTS idx_pred_status_deadline")
            # Multi-user consensus voting on resolutions. Without this,
            # any single chat member could mark any prediction however
            # they wanted — fine for "trust everyone" groups, terrible
            # for any chat where someone has incentive to manipulate
            # the leaderboard. Two distinct non-predictor voters
            # agreeing on the same verdict applies the resolution;
            # admins resolve solo. The predictor is excluded from
            # voting on themselves entirely.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS resolution_votes (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    prediction_id INTEGER NOT NULL,
                    voter_user_hash TEXT NOT NULL,
                    verdict TEXT NOT NULL,
                    note TEXT,
                    created_at REAL NOT NULL,
                    UNIQUE(prediction_id, voter_user_hash)
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_resolution_votes_pred "
                "ON resolution_votes(prediction_id)"
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
        async with db_session(self) as db:
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
        async with db_session(self) as db:
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
        async with db_session(self) as db:
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

    async def force_set_verdict(
        self,
        pred_id: int,
        *,
        verdict: str,
        note: Optional[str] = None,
        resolver_user_hash: Optional[str] = None,
    ) -> bool:
        """Admin override path: set the verdict regardless of current status.

        Distinct from `resolve()` which only fires on still-pending rows
        (the auto-resolver's safety against double-posting). Admin
        editing on the dashboard needs to fix already-resolved rows that
        the auto-resolver got wrong, so we update unconditionally and
        leave status='resolved' (or upgrade expired → resolved).
        """
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid verdict: {verdict!r}")
        async with db_session(self) as db:
            cursor = await db.execute(
                """UPDATE predictions
                   SET status = 'resolved',
                       verdict = ?,
                       resolution_note = ?,
                       resolved_at = ?,
                       resolver_user_hash = ?
                   WHERE id = ?""",
                (verdict, note, time.time(), resolver_user_hash, pred_id),
            )
            await db.commit()
            return cursor.rowcount > 0

    async def revert_to_pending(self, pred_id: int) -> bool:
        """Admin path: send a resolved/expired prediction back to pending so
        the auto-resolver can re-judge it (or the admin can override it
        again). Clears the verdict but keeps the original deadline so the
        resolver picks it up immediately.
        """
        async with db_session(self) as db:
            cursor = await db.execute(
                """UPDATE predictions
                   SET status = 'pending',
                       verdict = NULL,
                       resolution_note = NULL,
                       resolved_at = NULL,
                       resolver_user_hash = NULL
                   WHERE id = ?""",
                (pred_id,),
            )
            await db.commit()
            return cursor.rowcount > 0

    # Number of distinct non-predictor non-admin voters who must agree
    # on the same verdict before a resolution applies. Admins resolve
    # solo; auto-resolver (cron) bypasses voting entirely.
    RESOLUTION_VOTE_THRESHOLD = 2

    async def vote_to_resolve(
        self,
        pred_id: int,
        *,
        voter_user_hash: str,
        verdict: str,
        note: Optional[str] = None,
        is_admin: bool = False,
    ) -> dict:
        """Cast a resolution vote; return a status dict.

        Outcomes (returned in `status` field):
          - "resolved"   → verdict applied (admin path OR threshold met)
          - "voted"      → vote recorded; not enough agreeing voters yet
          - "self_vote"  → the predictor cannot vote on themselves
          - "not_found"  → no prediction with this id
          - "not_pending"→ already resolved/expired

        Additional fields when applicable:
          - `agreeing_count`: distinct voters with this verdict (resolved/voted)
          - `dissenting_summary`: string like "1 right, 2 wrong" when split
          - `predictor_label`: convenience for response rendering
        """
        if verdict not in VALID_VERDICTS:
            raise ValueError(f"invalid verdict: {verdict!r}")
        async with db_session(self) as db:
            # Lock the row group up-front so concurrent voters don't
            # both think they're casting the threshold-meeting vote.
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT user_hash, user_label, status FROM predictions WHERE id = ?",
                    (pred_id,),
                )
                row = await cursor.fetchone()
                if not row:
                    await db.rollback()
                    return {"status": "not_found"}
                predictor_hash, predictor_label, status = row
                if status != STATUS_PENDING:
                    await db.rollback()
                    return {
                        "status": "not_pending",
                        "predictor_label": predictor_label,
                    }
                if voter_user_hash == predictor_hash and not is_admin:
                    await db.rollback()
                    return {
                        "status": "self_vote",
                        "predictor_label": predictor_label,
                    }

                # Admin path: bypass voting, resolve immediately.
                if is_admin:
                    await db.execute(
                        """UPDATE predictions
                           SET status = 'resolved',
                               verdict = ?,
                               resolution_note = ?,
                               resolved_at = ?,
                               resolver_user_hash = ?
                           WHERE id = ? AND status = 'pending'""",
                        (verdict, note, time.time(), voter_user_hash, pred_id),
                    )
                    await db.commit()
                    return {
                        "status": "resolved",
                        "predictor_label": predictor_label,
                        "agreeing_count": 1,
                        "by_admin": True,
                    }

                # Non-admin: upsert this voter's vote (replaces any
                # prior verdict from the same voter — they're allowed
                # to change their mind).
                await db.execute(
                    """INSERT INTO resolution_votes
                       (prediction_id, voter_user_hash, verdict, note, created_at)
                       VALUES (?, ?, ?, ?, ?)
                       ON CONFLICT(prediction_id, voter_user_hash) DO UPDATE SET
                           verdict = excluded.verdict,
                           note = excluded.note,
                           created_at = excluded.created_at""",
                    (pred_id, voter_user_hash, verdict, note, time.time()),
                )

                # Count distinct voters per verdict for THIS prediction.
                cursor = await db.execute(
                    """SELECT verdict, COUNT(*) FROM resolution_votes
                       WHERE prediction_id = ?
                       GROUP BY verdict""",
                    (pred_id,),
                )
                tally = dict(await cursor.fetchall())
                agreeing = tally.get(verdict, 0)

                if agreeing >= self.RESOLUTION_VOTE_THRESHOLD:
                    await db.execute(
                        """UPDATE predictions
                           SET status = 'resolved',
                               verdict = ?,
                               resolution_note = ?,
                               resolved_at = ?,
                               resolver_user_hash = ?
                           WHERE id = ? AND status = 'pending'""",
                        (
                            verdict,
                            note or f"resolved by {agreeing}-user consensus",
                            time.time(),
                            voter_user_hash,
                            pred_id,
                        ),
                    )
                    await db.commit()
                    return {
                        "status": "resolved",
                        "predictor_label": predictor_label,
                        "agreeing_count": agreeing,
                        "by_admin": False,
                    }

                await db.commit()
                # Build a tally string for the response when there's
                # disagreement ("1 right, 2 wrong").
                summary_parts = sorted(
                    f"{count} {v}" for v, count in tally.items() if count > 0
                )
                return {
                    "status": "voted",
                    "predictor_label": predictor_label,
                    "agreeing_count": agreeing,
                    "needed": self.RESOLUTION_VOTE_THRESHOLD,
                    "dissenting_summary": ", ".join(summary_parts),
                }
            except Exception:
                await db.rollback()
                raise

    # Window during which a predictor (or the bot acting on their
    # behalf) can still revise the claim. Past this point the row is
    # locked to preserve leaderboard integrity — fixing wrong claims
    # later requires admin intervention via the dashboard.
    UPDATE_GRACE_SECONDS = 15 * 60

    async def update_pending(
        self,
        pred_id: int,
        *,
        claim: str,
        deadline_utc: float,
        ticker: Optional[str] = None,
        threshold: Optional[float] = None,
        direction: Optional[str] = None,
        expected_user_hash: Optional[str] = None,
    ) -> str:
        """Revise a pending prediction's claim/deadline within the grace
        window. Returns a status string:
          - "ok"           → the row was updated
          - "not_found"    → no row with that id
          - "not_pending"  → already resolved/expired (status check)
          - "stale"        → past the UPDATE_GRACE_SECONDS window
          - "not_owner"    → expected_user_hash doesn't match the predictor

        When `expected_user_hash` is provided, the row's user_hash must
        match. This guards the LLM-tool path: prompt-injection in a group
        message can otherwise coax Sigil into calling
        `predict_update(other_user_pred_id)` and rewriting someone else's
        claim within their grace window.
        """
        await self._ensure_initialized()
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT status, created_at, user_hash FROM predictions WHERE id = ?",
                (pred_id,),
            )
            row = await cursor.fetchone()
            if not row:
                return "not_found"
            status, created_at, row_user_hash = row[0], row[1], row[2]
            if status != STATUS_PENDING:
                return "not_pending"
            if (now - (created_at or 0)) > self.UPDATE_GRACE_SECONDS:
                return "stale"
            if expected_user_hash is not None and row_user_hash != expected_user_hash:
                return "not_owner"
            await db.execute(
                """UPDATE predictions
                   SET claim = ?, deadline_utc = ?,
                       ticker = ?, threshold = ?, direction = ?
                   WHERE id = ?""",
                (claim, deadline_utc, ticker, threshold, direction, pred_id),
            )
            await db.commit()
        return "ok"

    async def expire(self, pred_id: int, *, note: Optional[str] = None) -> bool:
        """Mark as expired — auto-resolver tried and gave up. Doesn't count."""
        async with db_session(self) as db:
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
        async with db_session(self) as db:
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

    async def total_counts(self) -> dict:
        """Aggregate counts across every context — for the admin dashboard.

        Returns: {total, pending, resolved, expired, right, wrong, unclear,
        accuracy} where accuracy is right / (right + wrong) over all
        resolved predictions, or None when nothing is judged yet.
        """
        async with db_session(self) as db:
            async with db.execute(
                """SELECT
                       COUNT(*) AS total,
                       SUM(CASE WHEN status='pending'  THEN 1 ELSE 0 END) AS pending,
                       SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) AS resolved,
                       SUM(CASE WHEN status='expired'  THEN 1 ELSE 0 END) AS expired,
                       SUM(CASE WHEN verdict='right'   THEN 1 ELSE 0 END) AS rights,
                       SUM(CASE WHEN verdict='wrong'   THEN 1 ELSE 0 END) AS wrongs,
                       SUM(CASE WHEN verdict='unclear' THEN 1 ELSE 0 END) AS unclears
                   FROM predictions"""
            ) as cur:
                row = await cur.fetchone()
        total, pending, resolved, expired, rights, wrongs, unclears = (
            row if row else (0, 0, 0, 0, 0, 0, 0)
        )
        rights = rights or 0
        wrongs = wrongs or 0
        judged = rights + wrongs
        return {
            "total": total or 0,
            "pending": pending or 0,
            "resolved": resolved or 0,
            "expired": expired or 0,
            "right": rights,
            "wrong": wrongs,
            "unclear": unclears or 0,
            "accuracy": (rights / judged) if judged else None,
        }

    async def recent_all(self, *, limit: int = 50) -> list[Prediction]:
        """Most recent predictions across every context, newest-first."""
        async with db_session(self) as db:
            async with db.execute(
                f"SELECT {self._COLS} FROM predictions "
                f"ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_pred(r) for r in rows]

    async def upcoming_all(self, *, limit: int = 50) -> list[Prediction]:
        """Pending predictions across every context, soonest-deadline first."""
        async with db_session(self) as db:
            async with db.execute(
                f"SELECT {self._COLS} FROM predictions "
                f"WHERE status = 'pending' "
                f"ORDER BY deadline_utc ASC LIMIT ?",
                (limit,),
            ) as cur:
                rows = await cur.fetchall()
        return [self._row_to_pred(r) for r in rows]

    async def contexts_with_predictions(self) -> list[dict]:
        """Distinct context_keys that have at least one prediction, with
        per-context status counts. Lets the dashboard render one
        leaderboard panel per active chat without separate round-trips.
        """
        async with db_session(self) as db:
            async with db.execute(
                """SELECT context_key,
                          COUNT(*) AS total,
                          SUM(CASE WHEN status='pending'  THEN 1 ELSE 0 END) AS pending,
                          SUM(CASE WHEN status='resolved' THEN 1 ELSE 0 END) AS resolved
                   FROM predictions
                   GROUP BY context_key
                   ORDER BY total DESC"""
            ) as cur:
                rows = await cur.fetchall()
        return [
            {"context_key": r[0], "total": r[1] or 0,
             "pending": r[2] or 0, "resolved": r[3] or 0}
            for r in rows
        ]

    async def user_record(self, user_hash: str, context_key: str) -> dict:
        """Single-user record for inline display ("David's record: 7/12, 58%")."""
        async with db_session(self) as db:
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
