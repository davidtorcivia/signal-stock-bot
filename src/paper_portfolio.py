"""
Paper-trading portfolio for Sigil.

Per-context ledger: each chat that enables the `portfolio` command gets
its own portfolio. The bot is the only trader; humans interact by tipping
cash into the pot (`!tip`) and watching the bot's positions and PnL.

Schema (sqlite, lives in the same db as predictions):
  portfolios   — one row per context_key. cash, starting_balance, label.
  positions    — long-only, fractional shares, average-cost lots.
  trades       — every fill, with reason + source ('cron'/'reactive'/'admin').
  tips         — every cash injection from a chat member, capped per ET day.

Design choices:
  - Cash is denormalized on `portfolios.cash` so status reads don't have
    to re-aggregate the entire trade history. All writes (buy/sell/tip)
    update cash atomically inside a transaction.
  - Positions use simple average-cost basis. Realized PnL on sells uses
    that average against the sell price; any remaining shares keep the
    same avg_cost.
  - Tip cap is $20/user/day in ET — same TZ boundary as the same-day
    prediction rule, for consistency.
  - V1 is long-only. `qty` is always > 0; sells reduce qty (deleting the
    row when it hits zero), they can't go short.

The execution layer (quote fetching, market-hours rejection) lives in
`paper_portfolio_executor.py`. This module is the persistence layer
only — it doesn't know what a Quote is.
"""

from __future__ import annotations

import datetime as dt
import logging
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import aiosqlite

from .database import db_session

logger = logging.getLogger(__name__)


# Day boundary for the per-user tip cap. Same TZ as the prediction
# same-day rule so the rules feel consistent to users.
_TIP_CAP_TZ = ZoneInfo("America/New_York")


SOURCE_CRON = "cron"
SOURCE_REACTIVE = "reactive"
SOURCE_ADMIN = "admin"
# Order-watcher driven trades. Distinct from `reactive` so admins /
# audits can tell at a glance which fills came from auto-fired
# conditional orders vs which came from a real chat conversation.
# Used both for the trigger fill itself and for any follow-up trades
# the bot makes inside the order-fired !ask reaction.
SOURCE_ORDER = "order"
VALID_SOURCES = {SOURCE_CRON, SOURCE_REACTIVE, SOURCE_ADMIN, SOURCE_ORDER}


def _is_valid_amount(value: float) -> bool:
    """Reject NaN and infinities. Dollar/share inputs that aren't real
    finite numbers must never reach the SQL layer — NaN compares False
    against every threshold (so `<=0` checks pass), and Inf would burn
    the entire portfolio in one trade."""
    return math.isfinite(value)

SIDE_BUY = "buy"
SIDE_SELL = "sell"

# Conditional-order kinds. `stop` triggers AGAINST the prevailing
# direction (buy on rise, sell on fall — protective + breakout). `limit`
# triggers WITH the direction (buy on dip, sell on rise — entry +
# take-profit). The four (side, kind) combinations and their
# trigger-relative-to-current-price predicates are encoded in
# `Order.should_fire`.
KIND_STOP = "stop"
KIND_LIMIT = "limit"
VALID_ORDER_KINDS = {KIND_STOP, KIND_LIMIT}

# Order lifecycle:
#   pending   →  in_flight  →  {filled, failed}
#   pending   →  cancelled                          (LLM/admin cancel)
#   pending   →  expired                            (past expires_at)
#
# `in_flight` is a short-lived claim state used by the watcher to
# guarantee at-most-once execution: claim_for_fill atomically moves
# pending → in_flight, the executor runs, then finalize_fill moves
# in_flight → filled (or → failed). If the process crashes between
# claim and finalize, the row stays `in_flight` and is NEVER re-fired
# by the next watcher tick — better to leak one stuck row (the trade
# is in the `trades` ledger regardless) than to double-fire and
# double-debit cash.
ORDER_PENDING = "pending"
ORDER_IN_FLIGHT = "in_flight"
ORDER_FILLED = "filled"
ORDER_CANCELLED = "cancelled"
ORDER_EXPIRED = "expired"
ORDER_FAILED = "failed"
VALID_ORDER_STATUSES = {
    ORDER_PENDING, ORDER_IN_FLIGHT, ORDER_FILLED, ORDER_CANCELLED,
    ORDER_EXPIRED, ORDER_FAILED,
}


@dataclass
class Portfolio:
    context_key: str
    cash: float
    starting_balance: float
    label: Optional[str]
    created_at: float


@dataclass
class Position:
    context_key: str
    ticker: str
    qty: float
    avg_cost: float


@dataclass
class Trade:
    id: int
    context_key: str
    ts: float
    ticker: str
    side: str
    qty: float
    price: float
    proceeds: float       # always positive; cash delta = +proceeds (sell) / -proceeds (buy)
    realized_pnl: Optional[float]
    reason: Optional[str]
    source: str


@dataclass
class Tip:
    id: int
    context_key: str
    ts: float
    tipper_user_hash: str
    tipper_label: Optional[str]
    amount: float
    note: Optional[str]


@dataclass
class Order:
    """Conditional order awaiting a price trigger.

    Exactly one of (`qty`, `dollars`, `close_position=True`) is set:
      - `qty`: exact share count (buy or sell)
      - `dollars`: dollar amount (BUY ONLY — converted to qty at fill
        using the trigger-time quote)
      - `close_position`: SELL ONLY — fill against the entire current
        position at trigger time (handles shares accumulated/trimmed
        between order placement and fill)
    """
    id: int
    context_key: str
    ticker: str
    side: str         # SIDE_BUY / SIDE_SELL
    kind: str         # KIND_STOP / KIND_LIMIT
    trigger_price: float
    qty: Optional[float]
    dollars: Optional[float]
    close_position: bool
    reason: Optional[str]
    status: str       # one of VALID_ORDER_STATUSES
    created_at: float
    expires_at: Optional[float]
    filled_at: Optional[float]
    fill_price: Optional[float]
    fill_qty: Optional[float]
    fill_note: Optional[str]   # error reason on status='failed'

    def should_fire(self, current_price: float) -> bool:
        """True iff the trigger has been crossed by `current_price`.

        Direction depends on the (side, kind) pairing:
          - buy + stop  → fires when price >= trigger (breakout entry)
          - buy + limit → fires when price <= trigger (pullback entry)
          - sell + stop → fires when price <= trigger (stop-loss)
          - sell + limit → fires when price >= trigger (take-profit)
        """
        if self.side == SIDE_BUY and self.kind == KIND_STOP:
            return current_price >= self.trigger_price
        if self.side == SIDE_BUY and self.kind == KIND_LIMIT:
            return current_price <= self.trigger_price
        if self.side == SIDE_SELL and self.kind == KIND_STOP:
            return current_price <= self.trigger_price
        if self.side == SIDE_SELL and self.kind == KIND_LIMIT:
            return current_price >= self.trigger_price
        # Defensive default — any unknown pair won't fire. Validation
        # at insert time should make this branch unreachable.
        return False

    def trigger_direction(self) -> str:
        """'above' if the order fires when price rises to/past trigger,
        'below' if it fires when price falls to/past trigger. Used in
        human-readable summaries."""
        if (self.side == SIDE_BUY and self.kind == KIND_STOP) or (
            self.side == SIDE_SELL and self.kind == KIND_LIMIT
        ):
            return "above"
        return "below"


def _et_day_bounds(now_ts: Optional[float] = None) -> tuple[float, float]:
    """[start_ts, end_ts) covering the current ET calendar day in unix
    seconds. Used to scope the per-user tip cap to one ET day."""
    now = dt.datetime.fromtimestamp(
        now_ts if now_ts is not None else time.time(), tz=_TIP_CAP_TZ,
    )
    start = now.replace(hour=0, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(days=1)
    return start.timestamp(), end.timestamp()


class PortfolioStore:
    """SQLite-backed paper portfolio per context_key.

    All write methods take a `context_key` and operate inside the same
    transaction. Auto-creates a portfolio on first write with the
    configured seed balance.
    """

    INITIAL_SEED = 1000.0
    DAILY_TIP_CAP = 20.0  # per user, per ET calendar day

    def __init__(
        self,
        db_path: str,
        *,
        initial_seed: float = INITIAL_SEED,
        daily_tip_cap: float = DAILY_TIP_CAP,
    ):
        self.db_path = Path(db_path)
        self.initial_seed = float(initial_seed)
        self.daily_tip_cap = float(daily_tip_cap)
        self._initialized = False
        # Init lock — without it, two coroutines that race into the
        # first call both open separate sqlite connections, both run
        # CREATE TABLE statements, and SQLite's single-writer model
        # raises "database is locked". The boolean flag alone is
        # insufficient: it's not set until the entire init block
        # completes, leaving a race window long enough that a
        # parallel `await store.foo()` slips through. The lock
        # narrows that window to zero.
        import asyncio as _asyncio
        self._init_lock = _asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            # Re-check inside the lock — another coroutine may have
            # finished init while we were waiting.
            if self._initialized:
                return
            await self._do_init()
            self._initialized = True

    async def _do_init(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        async with aiosqlite.connect(self.db_path) as db:
            from .database import apply_db_pragmas
            await apply_db_pragmas(db)
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolios (
                    context_key TEXT PRIMARY KEY,
                    cash REAL NOT NULL,
                    starting_balance REAL NOT NULL,
                    label TEXT,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS positions (
                    context_key TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    qty REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    PRIMARY KEY (context_key, ticker)
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_key TEXT NOT NULL,
                    ts REAL NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    qty REAL NOT NULL,
                    price REAL NOT NULL,
                    proceeds REAL NOT NULL,
                    realized_pnl REAL,
                    reason TEXT,
                    source TEXT NOT NULL
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_trades_context_ts "
                "ON trades(context_key, ts DESC)"
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS tips (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_key TEXT NOT NULL,
                    ts REAL NOT NULL,
                    tipper_user_hash TEXT NOT NULL,
                    tipper_label TEXT,
                    amount REAL NOT NULL,
                    note TEXT
                )
                """
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_tips_context_user_ts "
                "ON tips(context_key, tipper_user_hash, ts DESC)"
            )
            # Cron firing log — one row per (context_key, slot). The
            # trading worker checks this on every minute-tick to avoid
            # double-firing across bot restarts. Slots are stable
            # strings like 'open' / 'midday' / 'close'.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_cron_runs (
                    context_key TEXT NOT NULL,
                    slot TEXT NOT NULL,
                    last_fired_ts REAL NOT NULL,
                    PRIMARY KEY (context_key, slot)
                )
                """
            )
            # Conditional orders. Exactly one of (qty, dollars,
            # close_position) is set per row. Watcher polls
            # status='pending' rows every ~5 min during market hours.
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS portfolio_orders (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    context_key TEXT NOT NULL,
                    ticker TEXT NOT NULL,
                    side TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    trigger_price REAL NOT NULL,
                    qty REAL,
                    dollars REAL,
                    close_position INTEGER NOT NULL DEFAULT 0,
                    reason TEXT,
                    status TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    expires_at REAL,
                    filled_at REAL,
                    fill_price REAL,
                    fill_qty REAL,
                    fill_note TEXT
                )
                """
            )
            # Hot path for the watcher: "give me all pending orders, in
            # arrival order". Status filter is selective (most rows go
            # filled/cancelled within their lifetime), so a partial
            # index is overkill — a plain (status, ticker) covering
            # index handles both the watcher scan and the per-context
            # status display.
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_status_ticker "
                "ON portfolio_orders(status, ticker)"
            )
            await db.execute(
                "CREATE INDEX IF NOT EXISTS idx_orders_context_status "
                "ON portfolio_orders(context_key, status, created_at DESC)"
            )
            await db.commit()

    # ------------------------------------------------------------------
    # Portfolio lifecycle
    # ------------------------------------------------------------------

    async def ensure_portfolio(
        self,
        context_key: str,
        *,
        label: Optional[str] = None,
    ) -> Portfolio:
        """Create the portfolio for this context if it doesn't exist, or
        return the existing one. Idempotent — safe to call on every
        portfolio operation.

        Race-free: uses INSERT OR IGNORE so two concurrent first-time
        ensure_portfolio calls don't deadlock or crash with a UNIQUE
        violation. The follow-up SELECT picks up whichever row won.
        """
        await self._ensure_initialized()
        async with db_session(self) as db:
            await db.execute(
                """INSERT OR IGNORE INTO portfolios
                   (context_key, cash, starting_balance, label, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                (
                    context_key, self.initial_seed, self.initial_seed,
                    label, time.time(),
                ),
            )
            await db.commit()
            cursor = await db.execute(
                "SELECT context_key, cash, starting_balance, label, created_at "
                "FROM portfolios WHERE context_key = ?",
                (context_key,),
            )
            row = await cursor.fetchone()
        # Row is guaranteed to exist after the upsert above.
        assert row is not None
        return Portfolio(
            context_key=row[0], cash=row[1], starting_balance=row[2],
            label=row[3], created_at=row[4],
        )

    async def reset(self, context_key: str) -> bool:
        """Wipe the portfolio: drop all positions, trades, tips, cron-runs;
        re-seed cash to the initial seed. Returns True if a portfolio
        actually existed (and was reset).

        Admin-only — exposed via the dashboard. Not callable from chat
        because resetting under your own user would be a leaderboard cheat.
        """
        await self._ensure_initialized()
        async with db_session(self) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT 1 FROM portfolios WHERE context_key = ?",
                    (context_key,),
                )
                exists = await cursor.fetchone()
                if not exists:
                    await db.rollback()
                    return False
                for table in (
                    "positions", "trades", "tips", "portfolio_cron_runs",
                ):
                    await db.execute(
                        f"DELETE FROM {table} WHERE context_key = ?",
                        (context_key,),
                    )
                await db.execute(
                    "UPDATE portfolios SET cash = ?, starting_balance = ? "
                    "WHERE context_key = ?",
                    (self.initial_seed, self.initial_seed, context_key),
                )
                await db.commit()
                return True
            except Exception:
                await db.rollback()
                raise

    async def get_portfolio(self, context_key: str) -> Optional[Portfolio]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT context_key, cash, starting_balance, label, created_at "
                "FROM portfolios WHERE context_key = ?",
                (context_key,),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return Portfolio(
            context_key=row[0], cash=row[1], starting_balance=row[2],
            label=row[3], created_at=row[4],
        )

    async def positions(self, context_key: str) -> list[Position]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT context_key, ticker, qty, avg_cost "
                "FROM positions WHERE context_key = ? ORDER BY ticker",
                (context_key,),
            )
            rows = await cursor.fetchall()
        return [
            Position(context_key=r[0], ticker=r[1], qty=r[2], avg_cost=r[3])
            for r in rows
        ]

    async def get_position(
        self, context_key: str, ticker: str,
    ) -> Optional[Position]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT context_key, ticker, qty, avg_cost "
                "FROM positions WHERE context_key = ? AND ticker = ?",
                (context_key, ticker.upper()),
            )
            row = await cursor.fetchone()
        if row is None:
            return None
        return Position(context_key=row[0], ticker=row[1], qty=row[2], avg_cost=row[3])

    # ------------------------------------------------------------------
    # Trading
    # ------------------------------------------------------------------

    async def buy(
        self,
        context_key: str,
        ticker: str,
        qty: float,
        price: float,
        *,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
    ) -> dict:
        """Execute a paper-money buy. Atomically: deducts cash, inserts
        trade row, upserts position with rolling average cost.

        Returns:
          {"ok": True, "trade_id": int, "cash_after": float,
           "qty_after": float, "avg_cost_after": float, "proceeds": float}
        or
          {"ok": False, "error": "<reason>"}
        """
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        if not _is_valid_amount(qty) or not _is_valid_amount(price):
            return {"ok": False, "error": "Quantity and price must be finite numbers."}
        if qty <= 0:
            return {"ok": False, "error": "Quantity must be positive."}
        if price <= 0:
            return {"ok": False, "error": "Price must be positive."}
        ticker = ticker.upper().strip()
        if not ticker:
            return {"ok": False, "error": "Ticker required."}

        proceeds = qty * price
        await self.ensure_portfolio(context_key)

        async with db_session(self) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT cash FROM portfolios WHERE context_key = ?",
                    (context_key,),
                )
                row = await cursor.fetchone()
                if row is None:
                    await db.rollback()
                    return {"ok": False, "error": "Portfolio not found."}
                cash = row[0]
                if proceeds > cash + 1e-9:
                    await db.rollback()
                    return {
                        "ok": False,
                        "error": (
                            f"Insufficient cash: need ${proceeds:,.2f}, "
                            f"have ${cash:,.2f}."
                        ),
                    }

                # Update position with rolling average cost
                cursor = await db.execute(
                    "SELECT qty, avg_cost FROM positions "
                    "WHERE context_key = ? AND ticker = ?",
                    (context_key, ticker),
                )
                pos_row = await cursor.fetchone()
                if pos_row is None:
                    new_qty = qty
                    new_avg = price
                    await db.execute(
                        "INSERT INTO positions (context_key, ticker, qty, avg_cost) "
                        "VALUES (?, ?, ?, ?)",
                        (context_key, ticker, new_qty, new_avg),
                    )
                else:
                    old_qty, old_avg = pos_row[0], pos_row[1]
                    new_qty = old_qty + qty
                    new_avg = ((old_qty * old_avg) + (qty * price)) / new_qty
                    await db.execute(
                        "UPDATE positions SET qty = ?, avg_cost = ? "
                        "WHERE context_key = ? AND ticker = ?",
                        (new_qty, new_avg, context_key, ticker),
                    )

                # Deduct cash
                cash_after = cash - proceeds
                await db.execute(
                    "UPDATE portfolios SET cash = ? WHERE context_key = ?",
                    (cash_after, context_key),
                )

                # Record trade
                cursor = await db.execute(
                    """INSERT INTO trades
                       (context_key, ts, ticker, side, qty, price, proceeds,
                        realized_pnl, reason, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, NULL, ?, ?)""",
                    (
                        context_key, time.time(), ticker, SIDE_BUY,
                        qty, price, proceeds, reason, source,
                    ),
                )
                trade_id = cursor.lastrowid or 0
                await db.commit()
                return {
                    "ok": True,
                    "trade_id": trade_id,
                    "cash_after": cash_after,
                    "qty_after": new_qty,
                    "avg_cost_after": new_avg,
                    "proceeds": proceeds,
                }
            except Exception:
                await db.rollback()
                raise

    async def sell(
        self,
        context_key: str,
        ticker: str,
        qty: float,
        price: float,
        *,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
    ) -> dict:
        """Execute a paper-money sell. Atomically: credits cash, inserts
        trade row with realized_pnl, decrements position (deleting the
        row if qty hits zero).

        Returns:
          {"ok": True, "trade_id": int, "cash_after": float,
           "qty_after": float, "realized_pnl": float, "proceeds": float}
        or
          {"ok": False, "error": "<reason>"}

        V1 is long-only — selling more than you hold is rejected, not
        opened as a short.
        """
        if source not in VALID_SOURCES:
            raise ValueError(f"invalid source: {source!r}")
        if not _is_valid_amount(qty) or not _is_valid_amount(price):
            return {"ok": False, "error": "Quantity and price must be finite numbers."}
        if qty <= 0:
            return {"ok": False, "error": "Quantity must be positive."}
        if price <= 0:
            return {"ok": False, "error": "Price must be positive."}
        ticker = ticker.upper().strip()
        if not ticker:
            return {"ok": False, "error": "Ticker required."}

        await self.ensure_portfolio(context_key)
        proceeds = qty * price

        async with db_session(self) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                cursor = await db.execute(
                    "SELECT qty, avg_cost FROM positions "
                    "WHERE context_key = ? AND ticker = ?",
                    (context_key, ticker),
                )
                pos_row = await cursor.fetchone()
                if pos_row is None:
                    await db.rollback()
                    return {"ok": False, "error": f"No position in {ticker}."}
                old_qty, old_avg = pos_row[0], pos_row[1]
                # Treat tiny float drift as full-close — caller asking to
                # sell "all" rounds against the stored qty.
                if qty > old_qty + 1e-6:
                    await db.rollback()
                    return {
                        "ok": False,
                        "error": (
                            f"Only hold {old_qty:g} {ticker}; can't sell "
                            f"{qty:g}."
                        ),
                    }
                # Snap exact-close requests to the stored qty so the row
                # cleanly deletes instead of leaving a 1e-12 stub.
                if abs(qty - old_qty) < 1e-6:
                    qty = old_qty
                    proceeds = qty * price
                realized = (price - old_avg) * qty
                new_qty = old_qty - qty

                if new_qty <= 1e-9:
                    await db.execute(
                        "DELETE FROM positions WHERE context_key = ? AND ticker = ?",
                        (context_key, ticker),
                    )
                else:
                    # avg_cost stays the same on a partial sell — we're
                    # consuming shares with the existing average basis.
                    await db.execute(
                        "UPDATE positions SET qty = ? "
                        "WHERE context_key = ? AND ticker = ?",
                        (new_qty, context_key, ticker),
                    )

                cursor = await db.execute(
                    "SELECT cash FROM portfolios WHERE context_key = ?",
                    (context_key,),
                )
                row = await cursor.fetchone()
                cash = row[0] if row else 0.0
                cash_after = cash + proceeds
                await db.execute(
                    "UPDATE portfolios SET cash = ? WHERE context_key = ?",
                    (cash_after, context_key),
                )

                cursor = await db.execute(
                    """INSERT INTO trades
                       (context_key, ts, ticker, side, qty, price, proceeds,
                        realized_pnl, reason, source)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        context_key, time.time(), ticker, SIDE_SELL,
                        qty, price, proceeds, realized, reason, source,
                    ),
                )
                trade_id = cursor.lastrowid or 0
                await db.commit()
                return {
                    "ok": True,
                    "trade_id": trade_id,
                    "cash_after": cash_after,
                    "qty_after": new_qty if new_qty > 1e-9 else 0.0,
                    "realized_pnl": realized,
                    "proceeds": proceeds,
                }
            except Exception:
                await db.rollback()
                raise

    async def list_trades(
        self, context_key: str, *, limit: int = 20,
    ) -> list[Trade]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT id, context_key, ts, ticker, side, qty, price,
                          proceeds, realized_pnl, reason, source
                   FROM trades WHERE context_key = ?
                   ORDER BY ts DESC LIMIT ?""",
                (context_key, limit),
            )
            rows = await cursor.fetchall()
        return [
            Trade(
                id=r[0], context_key=r[1], ts=r[2], ticker=r[3],
                side=r[4], qty=r[5], price=r[6], proceeds=r[7],
                realized_pnl=r[8], reason=r[9], source=r[10],
            ) for r in rows
        ]

    async def realized_pnl_total(self, context_key: str) -> float:
        """Sum of realized PnL across all sell trades."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(realized_pnl), 0.0) FROM trades "
                "WHERE context_key = ? AND side = 'sell'",
                (context_key,),
            )
            row = await cursor.fetchone()
        return float(row[0]) if row else 0.0

    # ------------------------------------------------------------------
    # Tips
    # ------------------------------------------------------------------

    async def tip_total_today(
        self,
        context_key: str,
        tipper_user_hash: str,
        *,
        now_ts: Optional[float] = None,
    ) -> float:
        """Sum of this user's tips into this portfolio during the current
        ET calendar day. Used by `tip()` to enforce the cap and by the
        command/tool layer to render the remaining allowance."""
        await self._ensure_initialized()
        start, end = _et_day_bounds(now_ts)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT COALESCE(SUM(amount), 0.0) FROM tips
                   WHERE context_key = ? AND tipper_user_hash = ?
                     AND ts >= ? AND ts < ?""",
                (context_key, tipper_user_hash, start, end),
            )
            row = await cursor.fetchone()
        return float(row[0]) if row else 0.0

    async def tip(
        self,
        context_key: str,
        *,
        tipper_user_hash: str,
        tipper_label: Optional[str],
        amount: float,
        note: Optional[str] = None,
    ) -> dict:
        """Add cash to the portfolio from a chat member. Enforces the
        per-user, per-ET-day cap.

        Returns one of:
          {"ok": True, "tip_id": int, "cash_after": float,
           "today_total": float, "remaining_today": float}
          {"ok": False, "reason": "non_positive"}
          {"ok": False, "reason": "cap_exceeded",
           "today_total": float, "cap": float, "remaining_today": float}
        """
        if not _is_valid_amount(amount) or amount <= 0:
            return {"ok": False, "reason": "non_positive"}
        await self.ensure_portfolio(context_key)

        async with db_session(self) as db:
            await db.execute("BEGIN IMMEDIATE")
            try:
                start, end = _et_day_bounds()
                cursor = await db.execute(
                    """SELECT COALESCE(SUM(amount), 0.0) FROM tips
                       WHERE context_key = ? AND tipper_user_hash = ?
                         AND ts >= ? AND ts < ?""",
                    (context_key, tipper_user_hash, start, end),
                )
                row = await cursor.fetchone()
                today_total = float(row[0]) if row else 0.0
                remaining = self.daily_tip_cap - today_total
                if amount > remaining + 1e-9:
                    await db.rollback()
                    return {
                        "ok": False,
                        "reason": "cap_exceeded",
                        "today_total": today_total,
                        "cap": self.daily_tip_cap,
                        "remaining_today": max(0.0, remaining),
                    }

                cursor = await db.execute(
                    """INSERT INTO tips
                       (context_key, ts, tipper_user_hash, tipper_label, amount, note)
                       VALUES (?, ?, ?, ?, ?, ?)""",
                    (context_key, time.time(), tipper_user_hash,
                     tipper_label, amount, note),
                )
                tip_id = cursor.lastrowid or 0

                cursor = await db.execute(
                    "SELECT cash FROM portfolios WHERE context_key = ?",
                    (context_key,),
                )
                row = await cursor.fetchone()
                cash = row[0] if row else 0.0
                cash_after = cash + amount
                await db.execute(
                    "UPDATE portfolios SET cash = ? WHERE context_key = ?",
                    (cash_after, context_key),
                )
                await db.commit()
                return {
                    "ok": True,
                    "tip_id": tip_id,
                    "cash_after": cash_after,
                    "today_total": today_total + amount,
                    "remaining_today": remaining - amount,
                }
            except Exception:
                await db.rollback()
                raise

    async def tip_total(self, context_key: str) -> float:
        """All-time tip total for this portfolio. Used by PnL math
        (total_funded = starting_balance + tips)."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT COALESCE(SUM(amount), 0.0) FROM tips WHERE context_key = ?",
                (context_key,),
            )
            row = await cursor.fetchone()
        return float(row[0]) if row else 0.0

    # ------------------------------------------------------------------
    # Cron tracking (used by TradingCronWorker)
    # ------------------------------------------------------------------

    async def list_portfolio_keys(self) -> list[str]:
        """Every existing portfolio's context_key — used by the cron
        worker to know which contexts to consider firing."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                "SELECT context_key FROM portfolios ORDER BY context_key"
            )
            rows = await cursor.fetchall()
        return [r[0] for r in rows]

    async def mark_cron_fired(
        self, context_key: str, slot: str,
        *, now_ts: Optional[float] = None,
    ) -> None:
        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            await db.execute(
                """INSERT INTO portfolio_cron_runs (context_key, slot, last_fired_ts)
                   VALUES (?, ?, ?)
                   ON CONFLICT(context_key, slot) DO UPDATE SET
                     last_fired_ts = excluded.last_fired_ts""",
                (context_key, slot, ts),
            )
            await db.commit()

    async def cron_fired_today(
        self, context_key: str, slot: str,
        *, now_ts: Optional[float] = None,
    ) -> bool:
        """True if `(context_key, slot)` already fired during the
        current ET calendar day."""
        await self._ensure_initialized()
        start, end = _et_day_bounds(now_ts)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT last_fired_ts FROM portfolio_cron_runs
                   WHERE context_key = ? AND slot = ?""",
                (context_key, slot),
            )
            row = await cursor.fetchone()
        if row is None:
            return False
        last = float(row[0])
        return start <= last < end

    async def list_tips(
        self, context_key: str, *, limit: int = 20,
    ) -> list[Tip]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """SELECT id, context_key, ts, tipper_user_hash,
                          tipper_label, amount, note
                   FROM tips WHERE context_key = ?
                   ORDER BY ts DESC LIMIT ?""",
                (context_key, limit),
            )
            rows = await cursor.fetchall()
        return [
            Tip(
                id=r[0], context_key=r[1], ts=r[2], tipper_user_hash=r[3],
                tipper_label=r[4], amount=r[5], note=r[6],
            ) for r in rows
        ]

    # ------------------------------------------------------------------
    # Conditional orders
    # ------------------------------------------------------------------

    _ORDER_COLUMNS = (
        "id, context_key, ticker, side, kind, trigger_price, qty, "
        "dollars, close_position, reason, status, created_at, "
        "expires_at, filled_at, fill_price, fill_qty, fill_note"
    )

    @staticmethod
    def _order_from_row(r) -> Order:
        return Order(
            id=r[0],
            context_key=r[1],
            ticker=r[2],
            side=r[3],
            kind=r[4],
            trigger_price=float(r[5]),
            qty=(float(r[6]) if r[6] is not None else None),
            dollars=(float(r[7]) if r[7] is not None else None),
            close_position=bool(r[8]),
            reason=r[9],
            status=r[10],
            created_at=float(r[11]),
            expires_at=(float(r[12]) if r[12] is not None else None),
            filled_at=(float(r[13]) if r[13] is not None else None),
            fill_price=(float(r[14]) if r[14] is not None else None),
            fill_qty=(float(r[15]) if r[15] is not None else None),
            fill_note=r[16],
        )

    async def create_order(
        self,
        context_key: str,
        *,
        ticker: str,
        side: str,
        kind: str,
        trigger_price: float,
        qty: Optional[float] = None,
        dollars: Optional[float] = None,
        close_position: bool = False,
        reason: Optional[str] = None,
        expires_at: Optional[float] = None,
        now_ts: Optional[float] = None,
    ) -> int:
        """Insert a pending order. Validates inputs at the persistence
        boundary as defence-in-depth — the executor does its own checks
        but the store is the last line so SQL never sees garbage."""
        if side not in (SIDE_BUY, SIDE_SELL):
            raise ValueError(f"invalid side {side!r}")
        if kind not in VALID_ORDER_KINDS:
            raise ValueError(f"invalid kind {kind!r}")
        if not _is_valid_amount(trigger_price) or trigger_price <= 0:
            raise ValueError("trigger_price must be a positive finite number")
        # Exactly one of qty / dollars / close_position must be set.
        # `close_position` is sell-only; `dollars` is buy-only (we don't
        # know fill price at registration so we can't safely convert
        # dollars to a sell qty).
        n_set = sum(
            1 for x in (
                qty is not None, dollars is not None, bool(close_position),
            ) if x
        )
        if n_set != 1:
            raise ValueError(
                "exactly one of qty / dollars / close_position must be set"
            )
        if dollars is not None:
            if side != SIDE_BUY:
                raise ValueError("dollars is only valid on buy orders")
            if not _is_valid_amount(dollars) or dollars <= 0:
                raise ValueError("dollars must be a positive finite number")
        if qty is not None:
            if not _is_valid_amount(qty) or qty <= 0:
                raise ValueError("qty must be a positive finite number")
        if close_position and side != SIDE_SELL:
            raise ValueError("close_position is only valid on sell orders")

        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            cursor = await db.execute(
                """INSERT INTO portfolio_orders
                   (context_key, ticker, side, kind, trigger_price,
                    qty, dollars, close_position, reason, status,
                    created_at, expires_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    context_key, ticker, side, kind, float(trigger_price),
                    qty, dollars, 1 if close_position else 0,
                    reason, ORDER_PENDING, ts, expires_at,
                ),
            )
            await db.commit()
            return cursor.lastrowid or 0

    async def get_order(self, order_id: int) -> Optional[Order]:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT {self._ORDER_COLUMNS} FROM portfolio_orders "
                f"WHERE id = ?",
                (order_id,),
            )
            row = await cursor.fetchone()
        return self._order_from_row(row) if row else None

    async def list_pending_orders(self) -> list[Order]:
        """Every pending order across all portfolios — the watcher's
        primary read. Sorted by ticker first so a single grouped pass
        can fetch one quote per ticker."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT {self._ORDER_COLUMNS} FROM portfolio_orders "
                f"WHERE status = ? "
                f"ORDER BY ticker, created_at",
                (ORDER_PENDING,),
            )
            rows = await cursor.fetchall()
        return [self._order_from_row(r) for r in rows]

    async def list_orders_by_ids(
        self, ids: list[int],
    ) -> list[Order]:
        """Bulk fetch by id list. Used by the watcher's notify path so
        we don't make N+1 round-trips after a tick. Empty list →
        empty result, never queries SQL."""
        if not ids:
            return []
        await self._ensure_initialized()
        # Cap the IN clause width — sqlite tolerates ~999 placeholders
        # but the watcher pre-tick set is bounded by
        # _MAX_PENDING_ORDERS_PER_CONTEXT × #contexts, well under that.
        # Still, slice defensively against pathological inputs.
        ids_capped = list(ids)[:500]
        placeholders = ",".join("?" for _ in ids_capped)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT {self._ORDER_COLUMNS} FROM portfolio_orders "
                f"WHERE id IN ({placeholders})",
                ids_capped,
            )
            rows = await cursor.fetchall()
        return [self._order_from_row(r) for r in rows]

    async def list_orders_for_context(
        self,
        context_key: str,
        *,
        statuses: Optional[set[str]] = None,
        limit: int = 50,
    ) -> list[Order]:
        """All orders for a context, filtered by status set (default:
        pending only). Used by status display + admin UI."""
        await self._ensure_initialized()
        if statuses is None:
            statuses = {ORDER_PENDING}
        invalid = statuses - VALID_ORDER_STATUSES
        if invalid:
            raise ValueError(f"invalid statuses: {invalid}")
        # Build the IN clause dynamically — sqlite has no native array.
        placeholders = ",".join("?" for _ in statuses)
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT {self._ORDER_COLUMNS} FROM portfolio_orders "
                f"WHERE context_key = ? AND status IN ({placeholders}) "
                f"ORDER BY created_at DESC LIMIT ?",
                (context_key, *statuses, limit),
            )
            rows = await cursor.fetchall()
        return [self._order_from_row(r) for r in rows]

    async def cancel_order(
        self,
        order_id: int,
        *,
        context_key: Optional[str] = None,
        now_ts: Optional[float] = None,
    ) -> str:
        """Move pending → cancelled. `context_key` (when supplied)
        scopes the cancellation: an LLM acting in chat A can never
        cancel chat B's orders even if it hallucinates a stale id.
        Returns 'ok', 'not_found', 'wrong_context', or 'not_pending'.
        """
        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            cursor = await db.execute(
                "SELECT context_key, status FROM portfolio_orders "
                "WHERE id = ?",
                (order_id,),
            )
            row = await cursor.fetchone()
            if row is None:
                return "not_found"
            if context_key is not None and row[0] != context_key:
                return "wrong_context"
            if row[1] != ORDER_PENDING:
                return "not_pending"
            await db.execute(
                "UPDATE portfolio_orders SET status = ?, filled_at = ? "
                "WHERE id = ? AND status = ?",
                (ORDER_CANCELLED, ts, order_id, ORDER_PENDING),
            )
            await db.commit()
        return "ok"

    async def claim_for_fill(self, order_id: int) -> bool:
        """Atomically move pending → in_flight to claim an order for
        firing. Returns True if the claim succeeded (caller should
        proceed to execute), False if the row wasn't pending (already
        cancelled, expired, in flight elsewhere, or filled).

        This is the at-most-once guarantee: only one tick can claim a
        given order. If the process crashes after claim but before
        finalize, the row stays `in_flight` permanently — visible in
        admin/logs but invisible to list_pending_orders, so it can
        never be re-fired and double-debit cash. The trade row in
        `trades` (committed atomically inside execute_buy/sell) is
        the source of truth for what actually happened."""
        await self._ensure_initialized()
        async with db_session(self) as db:
            cursor = await db.execute(
                "UPDATE portfolio_orders SET status = ? "
                "WHERE id = ? AND status = ?",
                (ORDER_IN_FLIGHT, order_id, ORDER_PENDING),
            )
            await db.commit()
        return (cursor.rowcount or 0) > 0

    async def list_in_flight_orders(self) -> list[Order]:
        """Orders stuck in claim state, used by startup reconciliation
        and admin tooling. In normal operation this list is empty —
        every claim should be followed by a finalize within ~1s."""
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                f"SELECT {self._ORDER_COLUMNS} FROM portfolio_orders "
                f"WHERE status = ? ORDER BY created_at",
                (ORDER_IN_FLIGHT,),
            )
            rows = await cursor.fetchall()
        return [self._order_from_row(r) for r in rows]

    async def mark_order_filled(
        self,
        order_id: int,
        *,
        fill_price: float,
        fill_qty: float,
        now_ts: Optional[float] = None,
    ) -> bool:
        """Stamp a successful fill. UPDATE matches BOTH pending and
        in_flight: legacy callers (predictions resolver, admin UI)
        haven't gone through claim_for_fill, while the watcher path
        does. Either way we transition to `filled` exactly once."""
        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            cursor = await db.execute(
                "UPDATE portfolio_orders "
                "SET status = ?, filled_at = ?, fill_price = ?, fill_qty = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (
                    ORDER_FILLED, ts, float(fill_price), float(fill_qty),
                    order_id, ORDER_PENDING, ORDER_IN_FLIGHT,
                ),
            )
            await db.commit()
        return (cursor.rowcount or 0) > 0

    async def mark_order_failed(
        self,
        order_id: int,
        *,
        note: str,
        now_ts: Optional[float] = None,
    ) -> bool:
        """Trigger fired but execute_buy/sell rejected — record the
        reason and stop polling this order. We do not auto-retry; the
        bot can place a new order if it still wants the action.

        Like mark_order_filled, accepts both pending and in_flight as
        the source state so the watcher's claim-then-finalize path and
        any direct-from-pending path both end here cleanly."""
        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            cursor = await db.execute(
                "UPDATE portfolio_orders "
                "SET status = ?, filled_at = ?, fill_note = ? "
                "WHERE id = ? AND status IN (?, ?)",
                (
                    ORDER_FAILED, ts, note[:400],
                    order_id, ORDER_PENDING, ORDER_IN_FLIGHT,
                ),
            )
            await db.commit()
        return (cursor.rowcount or 0) > 0

    async def expire_stale_orders(
        self,
        *,
        now_ts: Optional[float] = None,
    ) -> int:
        """Move past-expiry pending orders to status='expired'. Returns
        the number of rows moved. Called by the watcher on each tick."""
        await self._ensure_initialized()
        ts = now_ts if now_ts is not None else time.time()
        async with db_session(self) as db:
            cursor = await db.execute(
                "UPDATE portfolio_orders "
                "SET status = ?, filled_at = ? "
                "WHERE status = ? AND expires_at IS NOT NULL "
                "AND expires_at <= ?",
                (ORDER_EXPIRED, ts, ORDER_PENDING, ts),
            )
            await db.commit()
        return cursor.rowcount or 0
