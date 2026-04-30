"""Tests for the paper-trading portfolio store + executor.

Covers the core invariants:
  - Auto-seed on first interaction; idempotent ensure_portfolio.
  - Buy/sell math (cash deltas, avg_cost rolling, realized PnL).
  - Long-only enforcement: can't sell more than held.
  - Per-user tip cap enforced inside one ET day; rolls over.
  - Market-hours gate rejects trades outside RTH.
  - Status snapshot marks positions to market and survives quote misses.
"""

from __future__ import annotations

import datetime as dt
import time
from dataclasses import dataclass
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from src.paper_portfolio import PortfolioStore
from src.paper_portfolio_executor import (
    PaperPortfolioExecutor,
    is_market_open,
    market_closed_reason,
)


CTX = "group:test-money-marge"
ET = ZoneInfo("America/New_York")


@pytest.fixture
def store(tmp_path):
    return PortfolioStore(db_path=str(tmp_path / "portfolio.db"))


# ---------- store: seed + ensure ---------------------------------------------

@pytest.mark.asyncio
async def test_ensure_portfolio_auto_seeds(store):
    p = await store.ensure_portfolio(CTX, label="Money Marge Simps")
    assert p.cash == pytest.approx(1000.0)
    assert p.starting_balance == pytest.approx(1000.0)
    assert p.label == "Money Marge Simps"


@pytest.mark.asyncio
async def test_ensure_portfolio_is_idempotent(store):
    p1 = await store.ensure_portfolio(CTX)
    # Manually mutate cash to make sure the second call doesn't re-seed.
    await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=5.0)
    p2 = await store.ensure_portfolio(CTX)
    assert p2.cash == pytest.approx(p1.cash + 5.0)


# ---------- store: buy/sell math ---------------------------------------------

@pytest.mark.asyncio
async def test_buy_deducts_cash_and_creates_position(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy(CTX, "AAPL", qty=2, price=100.0)
    assert res["ok"]
    assert res["cash_after"] == pytest.approx(800.0)
    assert res["qty_after"] == pytest.approx(2)
    assert res["avg_cost_after"] == pytest.approx(100.0)
    pos = await store.get_position(CTX, "AAPL")
    assert pos is not None
    assert pos.qty == pytest.approx(2)


@pytest.mark.asyncio
async def test_buy_rolls_avg_cost(store):
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=2, price=100.0)
    res = await store.buy(CTX, "AAPL", qty=2, price=120.0)
    # Average of (2 @ 100) + (2 @ 120) = 110
    assert res["qty_after"] == pytest.approx(4)
    assert res["avg_cost_after"] == pytest.approx(110.0)
    assert res["cash_after"] == pytest.approx(1000 - 200 - 240)


@pytest.mark.asyncio
async def test_buy_rejects_when_insufficient_cash(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy(CTX, "AAPL", qty=20, price=100.0)
    assert not res["ok"]
    assert "insufficient" in res["error"].lower()
    # No position created, no cash moved.
    pos = await store.get_position(CTX, "AAPL")
    assert pos is None
    p = await store.get_portfolio(CTX)
    assert p is not None
    assert p.cash == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_sell_realizes_pnl_and_credits_cash(store):
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=2, price=100.0)
    res = await store.sell(CTX, "AAPL", qty=1, price=150.0)
    assert res["ok"]
    # Realized PnL = (150 - 100) * 1 = 50
    assert res["realized_pnl"] == pytest.approx(50.0)
    assert res["qty_after"] == pytest.approx(1.0)
    # Cash: 1000 - 200 + 150 = 950
    assert res["cash_after"] == pytest.approx(950.0)
    # Avg cost stays 100 on the partial sell
    pos = await store.get_position(CTX, "AAPL")
    assert pos is not None
    assert pos.avg_cost == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_sell_full_close_removes_position(store):
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=2, price=100.0)
    res = await store.sell(CTX, "AAPL", qty=2, price=110.0)
    assert res["ok"]
    assert res["qty_after"] == 0.0
    pos = await store.get_position(CTX, "AAPL")
    assert pos is None


@pytest.mark.asyncio
async def test_sell_rejects_more_than_held(store):
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=2, price=100.0)
    res = await store.sell(CTX, "AAPL", qty=5, price=110.0)
    assert not res["ok"]
    # Position untouched.
    pos = await store.get_position(CTX, "AAPL")
    assert pos is not None
    assert pos.qty == pytest.approx(2)


@pytest.mark.asyncio
async def test_sell_rejects_when_no_position(store):
    await store.ensure_portfolio(CTX)
    res = await store.sell(CTX, "TSLA", qty=1, price=100.0)
    assert not res["ok"]
    assert "no position" in res["error"].lower()


# ---------- store: tipping ---------------------------------------------------

@pytest.mark.asyncio
async def test_tip_credits_cash_and_records(store):
    await store.ensure_portfolio(CTX)
    res = await store.tip(
        CTX, tipper_user_hash="u1", tipper_label="Alice",
        amount=10.0, note="good call",
    )
    assert res["ok"]
    assert res["cash_after"] == pytest.approx(1010.0)
    assert res["today_total"] == pytest.approx(10.0)
    assert res["remaining_today"] == pytest.approx(10.0)


@pytest.mark.asyncio
async def test_tip_enforces_per_user_daily_cap(store):
    await store.ensure_portfolio(CTX)
    res1 = await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=15.0)
    assert res1["ok"]
    res2 = await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=10.0)
    assert not res2["ok"]
    assert res2["reason"] == "cap_exceeded"
    assert res2["today_total"] == pytest.approx(15.0)
    assert res2["remaining_today"] == pytest.approx(5.0)
    # Exactly the remaining amount still works.
    res3 = await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=5.0)
    assert res3["ok"]


@pytest.mark.asyncio
async def test_tip_cap_is_per_user_not_per_chat(store):
    await store.ensure_portfolio(CTX)
    res1 = await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=20.0)
    assert res1["ok"]
    # Second user has their own $20 budget.
    res2 = await store.tip(CTX, tipper_user_hash="u2", tipper_label="B", amount=20.0)
    assert res2["ok"]


@pytest.mark.asyncio
async def test_tip_total_today_uses_et_day_boundary(store):
    """Tips placed before/after the ET midnight line shouldn't count
    against the next day's allowance — and vice versa."""
    await store.ensure_portfolio(CTX)
    # Insert a tip-row directly via the underlying connection at a
    # timestamp that's 25h in the past — outside today's ET window.
    import aiosqlite
    yesterday_ts = time.time() - 25 * 3600
    async with aiosqlite.connect(store.db_path) as db:
        await db.execute(
            """INSERT INTO tips
               (context_key, ts, tipper_user_hash, tipper_label, amount, note)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (CTX, yesterday_ts, "u1", "A", 20.0, "yesterday"),
        )
        await db.commit()
    today_total = await store.tip_total_today(CTX, "u1")
    # Yesterday's $20 must NOT be in today's total.
    assert today_total == pytest.approx(0.0)
    # And today's allowance is still $20.
    res = await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=20.0)
    assert res["ok"]


# ---------- executor: market hours -------------------------------------------

def test_is_market_open_weekday_open():
    open_dt = datetime(2026, 4, 29, 10, 0, tzinfo=ET)  # Wed 10:00 ET
    assert is_market_open(open_dt) is True
    assert market_closed_reason(open_dt) is None


def test_is_market_open_weekend():
    sat = datetime(2026, 5, 2, 10, 0, tzinfo=ET)
    assert is_market_open(sat) is False
    reason = market_closed_reason(sat)
    assert reason is not None
    assert "weekend" in reason.lower()


def test_is_market_open_premarket():
    pre = datetime(2026, 4, 29, 8, 0, tzinfo=ET)
    assert is_market_open(pre) is False
    reason = market_closed_reason(pre)
    assert reason is not None
    assert "not open yet" in reason


def test_is_market_open_afterhours():
    after = datetime(2026, 4, 29, 17, 0, tzinfo=ET)
    assert is_market_open(after) is False
    reason = market_closed_reason(after)
    assert reason is not None
    assert "after" in reason.lower()


def test_is_market_open_at_close_boundary():
    """16:00 ET sharp is closed (boundary is exclusive on the close)."""
    close_sharp = datetime(2026, 4, 29, 16, 0, tzinfo=ET)
    assert is_market_open(close_sharp) is False


def test_is_market_open_at_open_boundary():
    """9:30 ET sharp is open."""
    open_sharp = datetime(2026, 4, 29, 9, 30, tzinfo=ET)
    assert is_market_open(open_sharp) is True


# ---------- executor: integration with mock provider -------------------------

@dataclass
class _FakeQuote:
    symbol: str
    price: float
    change: float = 0.0
    change_percent: float = 0.0
    volume: int = 0


class _FakeProviders:
    """ProviderManager stand-in: returns scripted quotes per ticker."""

    def __init__(self, prices: dict[str, float]):
        self.prices = prices

    async def get_quote(self, symbol: str):
        if symbol not in self.prices:
            raise ValueError(f"no price for {symbol}")
        return _FakeQuote(symbol=symbol, price=self.prices[symbol])


@pytest.fixture
def open_market(monkeypatch):
    """Pin the market-hours check to 'open' so executor tests don't
    depend on when CI runs."""
    fake_now = datetime(2026, 4, 29, 11, 0, tzinfo=ET)  # Wed 11:00 ET
    monkeypatch.setattr(
        "src.paper_portfolio_executor.dt.datetime",
        _FrozenDatetime(fake_now),
    )


class _FrozenDatetime(dt.datetime):
    """datetime subclass with a fixed `now()` — patched into the module
    under test to make market-hours assertions deterministic."""

    _frozen: dt.datetime

    def __new__(cls, frozen=None, *args, **kwargs):
        return super().__new__(cls, 1, 1, 1)

    def __init__(self, frozen):
        self.__class__._frozen = frozen

    @classmethod
    def now(cls, tz=None):
        if tz is None:
            return cls._frozen.replace(tzinfo=None)
        return cls._frozen.astimezone(tz)

    @classmethod
    def fromtimestamp(cls, ts, tz=None):
        if tz is None:
            return dt.datetime.fromtimestamp(ts)
        return dt.datetime.fromtimestamp(ts, tz=tz)


@pytest.mark.asyncio
async def test_executor_buy_with_dollars(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    res = await executor.execute_buy(
        CTX, ticker="AAPL", dollars=400.0, reason="testing",
    )
    assert res["ok"]
    assert res["price"] == 200.0
    # 400 / 200 = 2 shares
    assert res["qty_after"] == pytest.approx(2.0)
    assert res["cash_after"] == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_executor_rejects_outside_market_hours(store, monkeypatch):
    # Pin to Saturday (weekend → closed).
    sat = datetime(2026, 5, 2, 11, 0, tzinfo=ET)
    monkeypatch.setattr(
        "src.paper_portfolio_executor.dt.datetime",
        _FrozenDatetime(sat),
    )
    # Pre-seed so we can prove the buy didn't move cash; without this
    # the portfolio wouldn't even exist after a rejected buy and the
    # test would be checking the wrong invariant.
    await store.ensure_portfolio(CTX)
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    res = await executor.execute_buy(
        CTX, ticker="AAPL", dollars=400.0, reason="weekend",
    )
    assert not res["ok"]
    assert "market closed" in res["error"].lower()
    p = await store.get_portfolio(CTX)
    assert p is not None
    assert p.cash == pytest.approx(1000.0)
    pos = await store.get_position(CTX, "AAPL")
    assert pos is None


@pytest.mark.asyncio
async def test_executor_buy_force_market_open(store, monkeypatch):
    """Admin path: force_market_open=True bypasses the gate so the
    dashboard can fix bad fills out-of-hours."""
    sat = datetime(2026, 5, 2, 11, 0, tzinfo=ET)
    monkeypatch.setattr(
        "src.paper_portfolio_executor.dt.datetime",
        _FrozenDatetime(sat),
    )
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    res = await executor.execute_buy(
        CTX, ticker="AAPL", dollars=400.0, reason="admin override",
        source="admin", force_market_open=True,
    )
    assert res["ok"]


@pytest.mark.asyncio
async def test_executor_sell_all_closes_position(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=2.0, reason="open")
    res = await executor.execute_sell(
        CTX, ticker="AAPL", qty="all", reason="close",
    )
    assert res["ok"]
    assert res["qty_after"] == 0.0
    pos = await store.get_position(CTX, "AAPL")
    assert pos is None


@pytest.mark.asyncio
async def test_executor_status_marks_to_market(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    # Buy fills at the provider's current price (200).
    await executor.execute_buy(CTX, ticker="AAPL", qty=2.0, reason="open")
    # Bump the quote so status reads a higher mark and the position
    # shows unrealized gains.
    providers.prices["AAPL"] = 250.0
    snap = await executor.status(CTX)
    assert snap["market_value"] == pytest.approx(500.0)
    assert snap["unrealized_pnl"] == pytest.approx(100.0)
    # Equity = cash + market_value = (1000-400) + 500 = 1100
    assert snap["equity"] == pytest.approx(1100.0)
    # Total funded is just the seed (no tips)
    assert snap["total_pnl"] == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_executor_status_survives_quote_miss(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=2.0, reason="open")
    # Drop AAPL from the provider so status's mark-to-market lookup fails.
    providers.prices.clear()
    snap = await executor.status(CTX)
    # Falls back to cost basis, no crash.
    pos_view = snap["positions"][0]
    assert pos_view["mark_price"] is None
    assert pos_view["market_value"] == pytest.approx(400.0)  # 2 * 200 cost


# ---------- cron worker: slot windows + idempotency --------------------------

def test_cron_slot_in_window_at_open_time():
    from src.paper_portfolio_cron import TradingCronWorker
    worker = TradingCronWorker(
        store=None, ask_command=None, signal_handler=None,
        context_registry=None, bot_phone="",
    )
    at_945 = datetime(2026, 4, 29, 9, 45, tzinfo=ET)
    assert worker._slot_in_window(at_945) == "open"


def test_cron_slot_late_grace():
    """Bot restarts at 9:50 ET — open slot must still fire (grace = 10min)."""
    from src.paper_portfolio_cron import TradingCronWorker
    worker = TradingCronWorker(
        store=None, ask_command=None, signal_handler=None,
        context_registry=None, bot_phone="",
    )
    at_950 = datetime(2026, 4, 29, 9, 50, tzinfo=ET)
    assert worker._slot_in_window(at_950) == "open"


def test_cron_slot_outside_grace():
    from src.paper_portfolio_cron import TradingCronWorker
    worker = TradingCronWorker(
        store=None, ask_command=None, signal_handler=None,
        context_registry=None, bot_phone="",
    )
    # 9:56 ET is past the open slot's 10-minute grace and well before
    # midday. Nothing should fire.
    at_956 = datetime(2026, 4, 29, 9, 56, tzinfo=ET)
    assert worker._slot_in_window(at_956) is None


def test_cron_slot_midday_and_close():
    from src.paper_portfolio_cron import TradingCronWorker
    worker = TradingCronWorker(
        store=None, ask_command=None, signal_handler=None,
        context_registry=None, bot_phone="",
    )
    midday = datetime(2026, 4, 29, 12, 30, tzinfo=ET)
    close = datetime(2026, 4, 29, 15, 30, tzinfo=ET)
    assert worker._slot_in_window(midday) == "midday"
    assert worker._slot_in_window(close) == "close"


@pytest.mark.asyncio
async def test_cron_fired_today_uses_et_day_window(store):
    """Stamping a fire at noon ET means cron_fired_today returns True
    for the rest of the ET day, then False after midnight ET."""
    await store.ensure_portfolio(CTX)
    # Stamp a fire timestamp inside today's ET window.
    now_et = datetime.now(ET)
    today_noon = now_et.replace(hour=12, minute=0, second=0, microsecond=0)
    await store.mark_cron_fired(CTX, "open", now_ts=today_noon.timestamp())
    assert await store.cron_fired_today(CTX, "open") is True
    # Different slot — not fired.
    assert await store.cron_fired_today(CTX, "midday") is False


@pytest.mark.asyncio
async def test_cron_mark_fired_idempotent(store):
    await store.ensure_portfolio(CTX)
    await store.mark_cron_fired(CTX, "open")
    await store.mark_cron_fired(CTX, "open")  # second call should not error
    assert await store.cron_fired_today(CTX, "open") is True


# ---------- security: NaN / Inf rejection -----------------------------------

@pytest.mark.asyncio
async def test_store_buy_rejects_nan_qty(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy(CTX, "AAPL", qty=float("nan"), price=100.0)
    assert not res["ok"]
    p = await store.get_portfolio(CTX)
    assert p is not None and p.cash == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_store_buy_rejects_inf_price(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy(CTX, "AAPL", qty=1.0, price=float("inf"))
    assert not res["ok"]


@pytest.mark.asyncio
async def test_store_tip_rejects_nan(store):
    """NaN compares False against `<= 0`, so without the isfinite gate
    a NaN tip would pass validation and corrupt cash."""
    await store.ensure_portfolio(CTX)
    res = await store.tip(
        CTX, tipper_user_hash="u1", tipper_label="A",
        amount=float("nan"),
    )
    assert not res["ok"]
    p = await store.get_portfolio(CTX)
    assert p is not None and p.cash == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_store_tip_rejects_inf(store):
    await store.ensure_portfolio(CTX)
    res = await store.tip(
        CTX, tipper_user_hash="u1", tipper_label="A",
        amount=float("inf"),
    )
    assert not res["ok"]
    p = await store.get_portfolio(CTX)
    assert p is not None and p.cash == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_executor_buy_rejects_nan_dollars(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    res = await executor.execute_buy(
        CTX, ticker="AAPL", dollars=float("nan"), reason="testing",
    )
    assert not res["ok"]


# ---------- ensure_portfolio race-safety ------------------------------------

@pytest.mark.asyncio
async def test_ensure_portfolio_concurrent_first_create(store):
    """Two concurrent first-time creates must both return the same
    portfolio and never raise. Pre-fix, the second INSERT would crash
    on a UNIQUE PK violation."""
    import asyncio as _asyncio
    a, b = await _asyncio.gather(
        store.ensure_portfolio("group:concurrent-test"),
        store.ensure_portfolio("group:concurrent-test"),
    )
    assert a.cash == b.cash == pytest.approx(1000.0)
    # Only one row in storage.
    keys = await store.list_portfolio_keys()
    assert keys.count("group:concurrent-test") == 1


# ---------- source tagging --------------------------------------------------

@pytest.mark.asyncio
async def test_buy_records_source_tag(store):
    """The trade row must preserve the source tag so the admin panel
    can tell cron-driven from chat-reactive trades."""
    from src.paper_portfolio import SOURCE_CRON
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=1.0, price=100.0, source=SOURCE_CRON)
    trades = await store.list_trades(CTX, limit=1)
    assert len(trades) == 1
    assert trades[0].source == SOURCE_CRON


@pytest.mark.asyncio
async def test_buy_rejects_unknown_source(store):
    """Strict source validation prevents unrecognized tags from leaking
    in via the executor / tool-call paths."""
    await store.ensure_portfolio(CTX)
    with pytest.raises(ValueError):
        await store.buy(CTX, "AAPL", qty=1.0, price=100.0, source="bogus")


# ---------- reset -----------------------------------------------------------

@pytest.mark.asyncio
async def test_reset_wipes_portfolio_state(store):
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=2.0, price=100.0)
    await store.tip(CTX, tipper_user_hash="u1", tipper_label="A", amount=10.0)
    await store.mark_cron_fired(CTX, "open")
    ok = await store.reset(CTX)
    assert ok is True
    p = await store.get_portfolio(CTX)
    assert p is not None
    assert p.cash == pytest.approx(1000.0)
    assert await store.get_position(CTX, "AAPL") is None
    assert await store.list_trades(CTX, limit=10) == []
    assert await store.list_tips(CTX, limit=10) == []
    assert await store.cron_fired_today(CTX, "open") is False


@pytest.mark.asyncio
async def test_reset_returns_false_for_unknown_context(store):
    ok = await store.reset("group:never-existed")
    assert ok is False


# ---------- executor status: parallel quote fetches -------------------------

# ---------- conditional orders ----------------------------------------------

@pytest.mark.asyncio
async def test_order_should_fire_directions(store):
    """Trigger semantics for the four (side, kind) combinations."""
    from src.paper_portfolio import (
        Order, ORDER_PENDING, KIND_LIMIT, KIND_STOP, SIDE_BUY, SIDE_SELL,
    )

    def _o(side, kind, trig):
        return Order(
            id=0, context_key="x", ticker="AAPL", side=side, kind=kind,
            trigger_price=trig, qty=None, dollars=None, close_position=False,
            reason=None, status=ORDER_PENDING, created_at=0,
            expires_at=None, filled_at=None, fill_price=None, fill_qty=None,
            fill_note=None,
        )

    # buy stop: fires when price >= trigger (breakout)
    assert _o(SIDE_BUY, KIND_STOP, 200).should_fire(201)
    assert not _o(SIDE_BUY, KIND_STOP, 200).should_fire(199)
    # buy limit: fires when price <= trigger (pullback)
    assert _o(SIDE_BUY, KIND_LIMIT, 200).should_fire(199)
    assert not _o(SIDE_BUY, KIND_LIMIT, 200).should_fire(201)
    # sell stop: fires when price <= trigger (stop-loss)
    assert _o(SIDE_SELL, KIND_STOP, 180).should_fire(179)
    assert not _o(SIDE_SELL, KIND_STOP, 180).should_fire(181)
    # sell limit: fires when price >= trigger (take-profit)
    assert _o(SIDE_SELL, KIND_LIMIT, 220).should_fire(221)
    assert not _o(SIDE_SELL, KIND_LIMIT, 220).should_fire(219)


@pytest.mark.asyncio
async def test_create_and_list_pending_order(store):
    await store.ensure_portfolio(CTX)
    # Must hold AAPL before placing a sell order at the executor layer,
    # but the store-level create_order has no such gate — useful for
    # testing direct CRUD.
    oid = await store.create_order(
        CTX, ticker="AAPL", side="sell", kind="stop",
        trigger_price=180.0, close_position=True,
        reason="protect downside",
    )
    assert oid > 0
    pending = await store.list_pending_orders()
    assert len(pending) == 1
    assert pending[0].ticker == "AAPL"
    assert pending[0].close_position is True
    assert pending[0].kind == "stop"


@pytest.mark.asyncio
async def test_create_order_rejects_invalid_combos(store):
    await store.ensure_portfolio(CTX)
    # Two of (qty, dollars, close_position) → reject.
    with pytest.raises(ValueError):
        await store.create_order(
            CTX, ticker="AAPL", side="buy", kind="stop",
            trigger_price=200.0, qty=1.0, dollars=100.0, reason="r",
        )
    # dollars on a sell → reject.
    with pytest.raises(ValueError):
        await store.create_order(
            CTX, ticker="AAPL", side="sell", kind="limit",
            trigger_price=200.0, dollars=100.0, reason="r",
        )
    # close_position on a buy → reject.
    with pytest.raises(ValueError):
        await store.create_order(
            CTX, ticker="AAPL", side="buy", kind="stop",
            trigger_price=200.0, close_position=True, reason="r",
        )
    # Negative trigger → reject.
    with pytest.raises(ValueError):
        await store.create_order(
            CTX, ticker="AAPL", side="buy", kind="stop",
            trigger_price=-1.0, qty=1.0, reason="r",
        )


@pytest.mark.asyncio
async def test_cancel_order_scopes_by_context(store):
    """cancel_order with context_key set must reject mismatched chats."""
    await store.ensure_portfolio(CTX)
    oid = await store.create_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="r",
    )
    # Wrong context fails.
    status = await store.cancel_order(oid, context_key="group:other-chat")
    assert status == "wrong_context"
    # Right context succeeds.
    status = await store.cancel_order(oid, context_key=CTX)
    assert status == "ok"
    # Re-cancel is not_pending now.
    status = await store.cancel_order(oid, context_key=CTX)
    assert status == "not_pending"


@pytest.mark.asyncio
async def test_expire_stale_orders_moves_only_past_expiry(store):
    await store.ensure_portfolio(CTX)
    now = time.time()
    await store.create_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="future",
        expires_at=now + 3600, now_ts=now,
    )
    await store.create_order(
        CTX, ticker="TSLA", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="past",
        expires_at=now - 60, now_ts=now,
    )
    n = await store.expire_stale_orders(now_ts=now)
    assert n == 1
    pending = await store.list_pending_orders()
    assert len(pending) == 1
    assert pending[0].ticker == "AAPL"


@pytest.mark.asyncio
async def test_executor_place_order_rejects_sell_with_no_position(
    store, open_market,
):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    res = await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="stop",
        trigger_price=180.0, close_position=True, reason="protect",
    )
    assert res["ok"] is False
    assert "no aapl position" in res["error"].lower()


@pytest.mark.asyncio
async def test_executor_place_order_warns_on_already_crossed_trigger(
    store, open_market,
):
    """A buy-stop with trigger at $190 when current is $200 should
    register but flag a warning so the LLM understands the order is
    about to fire on the next watcher tick."""
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    res = await executor.place_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=190.0, qty=1.0, reason="entry",
    )
    assert res["ok"] is True
    assert res["warning"] is not None
    assert "already crossed" in res["warning"]


@pytest.mark.asyncio
async def test_orders_watcher_fires_triggered_buy(store, open_market):
    """End-to-end: place a buy-stop, drop a price that crosses, run
    try_fill_pending → order is filled, position exists."""
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    res = await executor.place_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=210.0, qty=2.0, reason="breakout entry",
    )
    assert res["ok"]
    order_id = res["order_id"]

    # Trigger not yet crossed: nothing fires.
    stats = await executor.try_fill_pending()
    assert stats["filled"] == 0
    pending = await store.list_pending_orders()
    assert len(pending) == 1

    # Price moves up past trigger.
    providers.prices["AAPL"] = 215.0
    stats = await executor.try_fill_pending()
    assert stats["filled"] == 1
    pending = await store.list_pending_orders()
    assert pending == []
    pos = await store.get_position(CTX, "AAPL")
    assert pos is not None
    assert pos.qty == pytest.approx(2.0)
    order = await store.get_order(order_id)
    assert order is not None
    assert order.status == "filled"
    assert order.fill_price == pytest.approx(215.0)


@pytest.mark.asyncio
async def test_orders_watcher_fires_stop_loss(store, open_market):
    """Sell-stop: bot is long AAPL, places a stop at $180. Price drops
    to $178 → order fills, position closes."""
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=5.0, reason="seed")
    res = await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="stop",
        trigger_price=180.0, close_position=True, reason="protect",
    )
    assert res["ok"]

    # Above trigger: no fire.
    stats = await executor.try_fill_pending()
    assert stats["filled"] == 0

    # Drop below trigger.
    providers.prices["AAPL"] = 178.0
    stats = await executor.try_fill_pending()
    assert stats["filled"] == 1
    assert await store.get_position(CTX, "AAPL") is None


@pytest.mark.asyncio
async def test_orders_watcher_groups_quotes_per_ticker(store, open_market):
    """Two pending orders on the same ticker should result in ONE
    quote fetch, not two — the watcher batches by ticker."""
    providers = _FakeProviders({"AAPL": 200.0})
    fetch_count = {"n": 0}

    original_get_quote = providers.get_quote

    async def counting_get_quote(symbol):
        fetch_count["n"] += 1
        return await original_get_quote(symbol)

    providers.get_quote = counting_get_quote  # type: ignore[method-assign]

    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=5.0, reason="seed")
    fetch_count["n"] = 0  # zero out the buy quote
    await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="limit",
        trigger_price=250.0, qty=1.0, reason="trim 1",
    )
    await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="limit",
        trigger_price=260.0, qty=1.0, reason="trim 2",
    )
    fetch_count["n"] = 0  # zero out the place_order sanity-check quotes

    await executor.try_fill_pending()
    # One ticker → one fetch in the batched parallel pass.
    assert fetch_count["n"] == 1


@pytest.mark.asyncio
async def test_claim_for_fill_atomic_at_most_once(store):
    """Two concurrent claims on the same order: only one wins."""
    await store.ensure_portfolio(CTX)
    oid = await store.create_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="r",
    )
    a = await store.claim_for_fill(oid)
    b = await store.claim_for_fill(oid)
    assert a is True
    assert b is False
    # And it's not in the pending list anymore — won't be re-fired.
    assert await store.list_pending_orders() == []
    # But mark_order_filled accepts in_flight too, so the normal
    # finalize path still works.
    ok = await store.mark_order_filled(oid, fill_price=200.0, fill_qty=1.0)
    assert ok is True
    order = await store.get_order(oid)
    assert order is not None
    assert order.status == "filled"


@pytest.mark.asyncio
async def test_in_flight_orders_invisible_to_pending_list(store):
    """A stuck in_flight order must not show up in list_pending_orders
    so the next watcher tick can never re-fire it."""
    await store.ensure_portfolio(CTX)
    oid = await store.create_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="r",
    )
    await store.claim_for_fill(oid)
    pending = await store.list_pending_orders()
    assert oid not in {o.id for o in pending}
    in_flight = await store.list_in_flight_orders()
    assert oid in {o.id for o in in_flight}


@pytest.mark.asyncio
async def test_executor_rejects_oversize_pending_order_count(
    store, open_market,
):
    """LLM-flood guard: more than _MAX_PENDING_ORDERS_PER_CONTEXT
    pending orders in one context returns an error."""
    from src.paper_portfolio_executor import _MAX_PENDING_ORDERS_PER_CONTEXT
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    # Fill exactly the cap with valid orders.
    for i in range(_MAX_PENDING_ORDERS_PER_CONTEXT):
        res = await executor.place_order(
            CTX, ticker="AAPL", side="buy", kind="limit",
            trigger_price=100.0 + i * 0.01,
            dollars=1.0, reason=f"r{i}",
        )
        assert res["ok"], f"#{i} unexpected reject: {res}"
    # Cap+1 is rejected.
    res = await executor.place_order(
        CTX, ticker="AAPL", side="buy", kind="limit",
        trigger_price=200.0, dollars=1.0, reason="overflow",
    )
    assert res["ok"] is False
    assert "Too many pending orders" in res["error"]


@pytest.mark.asyncio
async def test_executor_rejects_excessive_trigger_price(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    res = await executor.place_order(
        CTX, ticker="AAPL", side="buy", kind="limit",
        trigger_price=1e10, dollars=1.0, reason="absurd",
    )
    assert res["ok"] is False
    assert "ceiling" in res["error"].lower()


@pytest.mark.asyncio
async def test_executor_rejects_oversize_ticker(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    res = await executor.place_order(
        CTX, ticker="X" * 64, side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason="long",
    )
    assert res["ok"] is False
    assert "too long" in res["error"].lower()


@pytest.mark.asyncio
async def test_executor_truncates_long_reason(store, open_market):
    """Over-long reason is silently truncated (better UX than
    rejecting the order outright when the model gets verbose)."""
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await store.ensure_portfolio(CTX)
    long_reason = "x" * 5000
    res = await executor.place_order(
        CTX, ticker="AAPL", side="buy", kind="stop",
        trigger_price=200.0, qty=1.0, reason=long_reason,
    )
    assert res["ok"] is True
    order = await store.get_order(res["order_id"])
    assert order is not None
    assert order.reason is not None
    assert len(order.reason) <= 500


@pytest.mark.asyncio
async def test_close_position_records_actual_qty(store, open_market):
    """Sell-stop with close_position=True records the position size
    that was actually held at fire time, not 0."""
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    # Buy 4 shares ($800) — fits in the $1000 starting cash with room
    # to spare. The exact qty isn't important; we just need a non-
    # trivial number to verify it round-trips through close_position.
    buy = await executor.execute_buy(CTX, ticker="AAPL", qty=4.0, reason="seed")
    assert buy["ok"], buy
    res = await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="stop",
        trigger_price=180.0, close_position=True, reason="protect",
    )
    assert res["ok"], res
    order_id = res["order_id"]
    # Drop below trigger.
    providers.prices["AAPL"] = 175.0
    await executor.try_fill_pending()
    order = await store.get_order(order_id)
    assert order is not None
    assert order.status == "filled"
    assert order.fill_qty == pytest.approx(4.0)


@pytest.mark.asyncio
async def test_status_includes_pending_orders(store, open_market):
    providers = _FakeProviders({"AAPL": 200.0})
    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=1.0, reason="seed")
    await executor.place_order(
        CTX, ticker="AAPL", side="sell", kind="stop",
        trigger_price=180.0, close_position=True, reason="protect",
    )
    snap = await executor.status(CTX)
    assert "pending_orders" in snap
    assert len(snap["pending_orders"]) == 1
    o = snap["pending_orders"][0]
    assert o["ticker"] == "AAPL"
    assert o["trigger_direction"] == "below"
    assert o["close_position"] is True


@pytest.mark.asyncio
async def test_status_parallelizes_quote_fetches(store, open_market):
    """The status snapshot should fetch all marks concurrently. We can't
    measure timing reliably in unit tests, but we can confirm correctness
    when fetches are interleaved by gather."""
    providers = _FakeProviders({"AAPL": 200.0, "TSLA": 300.0, "NVDA": 100.0})
    executor = PaperPortfolioExecutor(store, providers)
    await executor.execute_buy(CTX, ticker="AAPL", qty=1.0, reason="a")
    await executor.execute_buy(CTX, ticker="TSLA", qty=1.0, reason="b")
    await executor.execute_buy(CTX, ticker="NVDA", qty=1.0, reason="c")
    snap = await executor.status(CTX)
    tickers = {p["ticker"] for p in snap["positions"]}
    assert tickers == {"AAPL", "TSLA", "NVDA"}
    # Each position carries its mark — proves the gather'd fetches
    # all populated correctly with no off-by-one in the zip.
    by_ticker = {p["ticker"]: p for p in snap["positions"]}
    assert by_ticker["AAPL"]["mark_price"] == pytest.approx(200.0)
    assert by_ticker["TSLA"]["mark_price"] == pytest.approx(300.0)
    assert by_ticker["NVDA"]["mark_price"] == pytest.approx(100.0)
