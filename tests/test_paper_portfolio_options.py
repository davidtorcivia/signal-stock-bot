"""Tests for the options portfolio store + executor.

Mirrors test_paper_portfolio.py but for the options surface:
  - buy_option / sell_option math (multiplier × premium × qty).
  - Cost basis rolling on add-to-open.
  - Long-only enforcement (can't sell more than held; no shorts).
  - Settlement: ITM cash-settles at intrinsic, OTM expires worthless.
  - status() includes options positions and folds them into equity.
"""

from __future__ import annotations

import datetime as dt
import time

import pytest

from src.options_symbols import build_occ
from src.paper_portfolio import PortfolioStore, SOURCE_SETTLEMENT
from src.paper_portfolio_executor import PaperPortfolioExecutor


CTX = "group:test-options"


class _FakeOptionQuote:
    """Stand-in for the OptionQuote dataclass returned by the provider."""
    def __init__(self, price, *, symbol="AAPL260620C00175000",
                 underlying="AAPL", option_type="call", strike=175.0,
                 expiration=None):
        self.symbol = symbol
        self.underlying = underlying
        self.type = option_type
        self.strike = strike
        self.price = price
        self.expiration = expiration or dt.datetime(2026, 6, 20)
        self.change = 0.0
        self.change_percent = 0.0
        self.volume = 100
        self.open_interest = 500
        self.implied_volatility = 0.35
        self.greeks = {"delta": 0.55}
        self.timestamp = dt.datetime.now()
        self.provider = "fake"


class _FakeQuote:
    """Stand-in for a Quote on the underlying (for settlement spot)."""
    def __init__(self, price):
        self.price = price


class FakeProvider:
    """Deterministic provider — caller pins quotes/chains for each test."""
    def __init__(self, *, option_premium=2.50, spot=180.0):
        self.option_premium = option_premium
        self.spot = spot
        # Per-contract override; falls back to option_premium.
        self.contract_premiums: dict[str, float] = {}

    async def get_quote(self, ticker):
        return _FakeQuote(self.spot)

    async def get_option_quote(self, symbol):
        premium = self.contract_premiums.get(symbol, self.option_premium)
        return _FakeOptionQuote(premium, symbol=symbol)

    async def get_options_chain(self, underlying, expiration=None, limit=100):
        return [_FakeOptionQuote(self.option_premium, symbol=f"{underlying}260620C00175000")]


@pytest.fixture
def store(tmp_path):
    return PortfolioStore(db_path=str(tmp_path / "options.db"))


@pytest.fixture
def executor(store):
    return PaperPortfolioExecutor(store, FakeProvider())


# ---------- store math -----------------------------------------------------

@pytest.mark.asyncio
async def test_buy_option_deducts_premium_times_multiplier(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy_option(
        CTX,
        contract_symbol="AAPL260620C00175000",
        underlying="AAPL",
        option_type="call",
        strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2,
        premium=3.00,
    )
    assert res["ok"], res
    # Cost = 2 contracts × 100 × $3.00 = $600
    assert res["proceeds"] == pytest.approx(600.0)
    assert res["cash_after"] == pytest.approx(400.0)
    assert res["qty_after"] == pytest.approx(2)
    assert res["avg_premium_after"] == pytest.approx(3.00)


@pytest.mark.asyncio
async def test_buy_option_rolls_avg_premium(store):
    """Rolling avg_premium math, both purchases fit in starting cash."""
    await store.ensure_portfolio(CTX)
    occ = "AAPL260620C00175000"
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2, premium=1.50,
    )
    res = await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2, premium=2.50,
    )
    assert res["ok"], res
    assert res["qty_after"] == pytest.approx(4)
    # Weighted average: ((2*1.50)+(2*2.50))/4 = 2.00
    assert res["avg_premium_after"] == pytest.approx(2.00)
    # Cash: 1000 - (2*100*1.50) - (2*100*2.50) = 1000 - 300 - 500 = 200
    assert res["cash_after"] == pytest.approx(200.0)


@pytest.mark.asyncio
async def test_buy_option_rejects_insufficient_cash(store):
    await store.ensure_portfolio(CTX)
    res = await store.buy_option(
        CTX, contract_symbol="AAPL260620C00175000", underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=20, premium=10.00,  # 20 * 100 * 10 = $20k vs $1000 cash
    )
    assert not res["ok"]
    assert "insufficient" in res["error"].lower()


@pytest.mark.asyncio
async def test_sell_option_realizes_pnl(store):
    await store.ensure_portfolio(CTX)
    occ = "AAPL260620C00175000"
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2, premium=2.00,
    )
    # Sell 1 contract at $5/sh → proceeds 1*100*5 = 500;
    # realized = (5 - 2) * 1 * 100 = 300
    res = await store.sell_option(
        CTX, contract_symbol=occ, qty=1, premium=5.00,
    )
    assert res["ok"], res
    assert res["proceeds"] == pytest.approx(500.0)
    assert res["realized_pnl"] == pytest.approx(300.0)
    assert res["qty_after"] == pytest.approx(1)
    # Cash: 1000 - 400 (buy) + 500 (sell) = 1100
    assert res["cash_after"] == pytest.approx(1100.0)


@pytest.mark.asyncio
async def test_sell_option_full_close_removes_position(store):
    await store.ensure_portfolio(CTX)
    occ = "AAPL260620C00175000"
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2, premium=2.00,
    )
    await store.sell_option(
        CTX, contract_symbol=occ, qty=2, premium=3.50,
    )
    pos = await store.get_option_position(CTX, occ)
    assert pos is None


@pytest.mark.asyncio
async def test_sell_option_rejects_overshoot(store):
    await store.ensure_portfolio(CTX)
    occ = "AAPL260620C00175000"
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=1, premium=2.00,
    )
    res = await store.sell_option(
        CTX, contract_symbol=occ, qty=5, premium=2.50,
    )
    assert not res["ok"]
    assert "can't close" in res["error"].lower()


@pytest.mark.asyncio
async def test_sell_option_no_position_errors(store):
    await store.ensure_portfolio(CTX)
    res = await store.sell_option(
        CTX, contract_symbol="AAPL260620C00175000",
        qty=1, premium=1.00,
    )
    assert not res["ok"]
    assert "no options position" in res["error"].lower()


@pytest.mark.asyncio
async def test_realized_pnl_total_includes_options(store):
    await store.ensure_portfolio(CTX)
    occ = "AAPL260620C00175000"
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2026, 6, 20).timestamp(),
        qty=2, premium=2.00,
    )
    await store.sell_option(
        CTX, contract_symbol=occ, qty=2, premium=3.00,
    )
    total = await store.realized_pnl_total(CTX)
    # 2 contracts × 100 × $1 premium gain = $200
    assert total == pytest.approx(200.0)


# ---------- settlement -----------------------------------------------------

@pytest.mark.asyncio
async def test_settle_itm_call_cash_settles_at_intrinsic(executor):
    store = executor.store
    occ = build_occ("AAPL", dt.date(2025, 1, 1), "call", 175.0)
    await store.ensure_portfolio(CTX)
    # Position expired in the past so the sweep will pick it up.
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2025, 1, 1).timestamp(),
        qty=2, premium=2.00,
    )
    # FakeProvider quotes AAPL at $180 → intrinsic = max(180-175, 0) = $5.
    # Settlement proceeds: 2 * 100 * 5 = $1000.
    # Realized PnL: (5 - 2) * 2 * 100 = $600.
    stats = await executor.settle_expired_options()
    assert stats["settled"] == 1
    assert stats["errors"] == 0
    pos = await store.get_option_position(CTX, occ)
    assert pos is None
    portfolio = await store.get_portfolio(CTX)
    # Starting 1000 - 400 (buy) + 1000 (settle) = 1600
    assert portfolio.cash == pytest.approx(1600.0)
    # Verify the trade ledger marked it as a settlement.
    trades = await store.list_option_trades(CTX, limit=10)
    settle_trades = [t for t in trades if t.source == SOURCE_SETTLEMENT]
    assert len(settle_trades) == 1
    assert settle_trades[0].side == "settle"
    assert settle_trades[0].realized_pnl == pytest.approx(600.0)


@pytest.mark.asyncio
async def test_settle_otm_call_expires_worthless(executor):
    store = executor.store
    # Underlying spot ($180) is BELOW strike ($200) → call is OTM.
    executor.providers.spot = 180.0
    occ = build_occ("AAPL", dt.date(2025, 1, 1), "call", 200.0)
    await store.ensure_portfolio(CTX)
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=200.0,
        expiration=dt.datetime(2025, 1, 1).timestamp(),
        qty=1, premium=1.50,
    )
    stats = await executor.settle_expired_options()
    assert stats["settled"] == 1
    portfolio = await store.get_portfolio(CTX)
    # Lost full premium: 1000 - 150 = 850, no proceeds back.
    assert portfolio.cash == pytest.approx(850.0)
    trades = await store.list_option_trades(CTX, limit=10)
    settle = [t for t in trades if t.source == SOURCE_SETTLEMENT][0]
    assert settle.premium == pytest.approx(0.0)
    assert settle.realized_pnl == pytest.approx(-150.0)  # lost the full premium


@pytest.mark.asyncio
async def test_settle_otm_put_expires_worthless(executor):
    store = executor.store
    # Underlying spot ($180) is ABOVE strike ($150) → put is OTM.
    executor.providers.spot = 180.0
    occ = build_occ("AAPL", dt.date(2025, 1, 1), "put", 150.0)
    await store.ensure_portfolio(CTX)
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="put", strike=150.0,
        expiration=dt.datetime(2025, 1, 1).timestamp(),
        qty=1, premium=0.75,
    )
    stats = await executor.settle_expired_options()
    assert stats["settled"] == 1
    portfolio = await store.get_portfolio(CTX)
    # Lost full premium: 1000 - 75 = 925.
    assert portfolio.cash == pytest.approx(925.0)


@pytest.mark.asyncio
async def test_settle_idempotent_no_double_settle(executor):
    """A second sweep after settlement is a no-op."""
    store = executor.store
    occ = build_occ("AAPL", dt.date(2025, 1, 1), "call", 175.0)
    await store.ensure_portfolio(CTX)
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=dt.datetime(2025, 1, 1).timestamp(),
        qty=1, premium=2.00,
    )
    stats1 = await executor.settle_expired_options()
    stats2 = await executor.settle_expired_options()
    assert stats1["settled"] == 1
    assert stats2["settled"] == 0
    # When there's nothing due, the executor short-circuits before
    # computing `due`/`errors`; only `checked` and `settled` are set.
    assert stats2.get("due", 0) == 0
    pos = await executor.store.get_option_position(CTX, occ)
    assert pos is None


@pytest.mark.asyncio
async def test_settle_skips_unexpired_positions(executor):
    """Positions with expiration in the future are left alone."""
    store = executor.store
    far_future = time.time() + 365 * 86400
    occ = build_occ("AAPL", dt.date(2099, 12, 31), "call", 175.0)
    await store.ensure_portfolio(CTX)
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=far_future,
        qty=1, premium=2.00,
    )
    stats = await executor.settle_expired_options()
    assert stats["settled"] == 0
    pos = await store.get_option_position(CTX, occ)
    assert pos is not None


# ---------- status snapshot ------------------------------------------------

@pytest.mark.asyncio
async def test_status_includes_options_positions(executor):
    store = executor.store
    # Use a non-expired contract (status() shouldn't hit settle).
    far_future = time.time() + 30 * 86400
    occ = build_occ("AAPL", dt.date(2099, 12, 31), "call", 175.0)
    await store.ensure_portfolio(CTX)
    await store.buy_option(
        CTX, contract_symbol=occ, underlying="AAPL",
        option_type="call", strike=175.0,
        expiration=far_future,
        qty=1, premium=2.00,
    )
    # FakeProvider returns $2.50 premium → mark > cost → unrealized > 0.
    executor.providers.option_premium = 2.50
    snap = await executor.status(CTX)
    assert "options_positions" in snap
    rows = snap["options_positions"]
    assert len(rows) == 1
    row = rows[0]
    assert row["contract"] == occ
    assert row["qty"] == pytest.approx(1)
    assert row["mark_premium"] == pytest.approx(2.50)
    # Unrealized PnL = (2.50 - 2.00) * 1 * 100 = $50
    assert row["unrealized_pnl"] == pytest.approx(50.0)
    # Options market value rolled into total equity.
    assert snap["options_market_value"] == pytest.approx(250.0)
