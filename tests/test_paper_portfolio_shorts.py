"""
Tests for writing options: collateral, covered calls, credit spreads,
buy-to-close, and short settlement.

The invariant under all of it: collateral leaves cash when a short opens
and comes back when it closes, so no other trade can spend money pledged
against an obligation, and equity still counts it.
"""

import tempfile
import time
from pathlib import Path

import pytest

from src.paper_portfolio import (
    LONG,
    SHORT,
    PortfolioStore,
    _short_collateral,
    _structure_label,
)

CTX = "dm:test"
DAY = 86400.0
# The default seed is $1,000, which cannot cash-secure a single
# $100-strike put. Fund these books well enough that a rejection means
# the collateral rule fired, not that the portfolio was too small.
SEED = 100_000.0


@pytest.fixture
async def store():
    with tempfile.TemporaryDirectory() as d:
        s = PortfolioStore(db_path=str(Path(d) / "p.db"), initial_seed=SEED)
        await s.ensure_portfolio(CTX)
        yield s


def _exp(days: int = 30) -> float:
    return time.time() + days * DAY


async def _assert_short_book_consistent(store):
    """Every short call's pledge agrees with what is actually backing it.

    The whole design rests on `collateral == 0` meaning "stock is doing
    the work" and nothing else. That is maintained by four separate
    facts in paper_portfolio.py, so rather than testing each one, assert
    the property they exist to produce — this catches a regression in any
    of them, including ones nobody thought to write a case for.
    """
    positions = await store.options_positions(CTX)
    shares = {p.ticker: p.qty for p in await store.positions(CTX)}
    for op in positions:
        if not op.is_short or op.option_type != "call":
            continue
        claimed_by_others = sum(
            other.qty * other.multiplier
            for other in positions
            if other.is_short
            and other.option_type == "call"
            and other.underlying == op.underlying
            and other.collateral <= 0
            and other.contract_symbol != op.contract_symbol
        )
        free = shares.get(op.underlying, 0.0) - claimed_by_others
        notional = op.qty * op.multiplier
        if op.collateral <= 0:
            assert free + 1e-9 >= notional, (
                f"{op.contract_symbol} pledges nothing but only {free:g} "
                f"shares back its {notional:g}"
            )
        else:
            assert free < notional, (
                f"{op.contract_symbol} pledges ${op.collateral:,.2f} while "
                f"{free:g} shares would cover its {notional:g} — the row is "
                f"stock-backed but outside the collateral<=0 bucket, so "
                f"sell() will release those shares"
            )


async def _write_put(store, *, strike=100.0, qty=1, premium=2.0, expiration=None):
    return await store.write_option(
        CTX,
        contract_symbol=f"ACME991231P{int(strike * 1000):08d}",
        underlying="ACME", option_type="put", strike=strike,
        expiration=expiration if expiration is not None else _exp(),
        qty=qty, premium=premium,
    )


# ── collateral rules in isolation ──────────────────────────────────────────

def test_cash_secured_put_pledges_the_strike():
    collateral, err = _short_collateral(
        "put", 100.0, 2, 100, shares_held=0, protective_strike=None,
    )
    assert err is None
    assert collateral == pytest.approx(20_000.0)


def test_put_spread_pledges_only_the_width():
    collateral, err = _short_collateral(
        "put", 100.0, 2, 100, shares_held=0, protective_strike=95.0,
    )
    assert err is None
    assert collateral == pytest.approx(1_000.0)   # 5 wide x 100 x 2


def test_covered_call_pledges_nothing():
    collateral, err = _short_collateral(
        "call", 100.0, 2, 100, shares_held=200, protective_strike=None,
    )
    assert err is None
    assert collateral == 0.0


def test_partially_covered_call_is_not_covered():
    """199 shares does not cover 2 contracts."""
    _collateral, err = _short_collateral(
        "call", 100.0, 2, 100, shares_held=199, protective_strike=None,
    )
    assert err is not None and "naked" in err.lower()


def test_call_spread_pledges_the_width():
    collateral, err = _short_collateral(
        "call", 100.0, 1, 100, shares_held=0, protective_strike=110.0,
    )
    assert err is None
    assert collateral == pytest.approx(1_000.0)


def test_naked_call_is_refused():
    collateral, err = _short_collateral(
        "call", 100.0, 1, 100, shares_held=0, protective_strike=None,
    )
    assert collateral == 0.0
    assert err is not None and "unbounded" in err


def test_a_lower_long_call_does_not_cap_a_short_call():
    """A long call BELOW the short strike caps nothing above it."""
    _c, err = _short_collateral(
        "call", 100.0, 1, 100, shares_held=0, protective_strike=90.0,
    )
    assert err is not None


def test_a_higher_long_put_does_not_cap_a_short_put():
    collateral, err = _short_collateral(
        "put", 100.0, 1, 100, shares_held=0, protective_strike=110.0,
    )
    assert err is None
    assert collateral == pytest.approx(10_000.0)   # falls back to cash-secured


@pytest.mark.parametrize("kind,shares,protective,expected", [
    ("call", 100, None, "covered call"),
    ("call", 0, 110.0, "call credit spread"),
    ("put", 0, None, "cash-secured put"),
    ("put", 0, 95.0, "put credit spread"),
])
def test_structure_labels(kind, shares, protective, expected):
    assert _structure_label(kind, shares, protective, 100, 1) == expected


# ── writing ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_write_credits_premium_and_pledges_collateral(store):
    before = (await store.ensure_portfolio(CTX)).cash
    result = await _write_put(store, strike=100.0, qty=1, premium=2.0)
    assert result["ok"] is True
    assert result["collateral"] == pytest.approx(10_000.0)
    assert result["structure"] == "cash-secured put"
    # +$200 premium, -$10,000 pledged
    assert result["cash_after"] == pytest.approx(before + 200.0 - 10_000.0)

    pos = await store.get_option_position(CTX, "ACME991231P00100000")
    assert pos.side == SHORT
    assert pos.is_short is True
    assert pos.signed_qty() == -1
    assert pos.collateral == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_write_refused_when_collateral_exceeds_cash(store):
    result = await _write_put(store, strike=100_000.0, qty=1, premium=1.0)
    assert result["ok"] is False
    assert "collateral" in result["error"].lower()


@pytest.mark.asyncio
async def test_premium_counts_toward_its_own_collateral(store):
    """A write is affordable when cash plus the credit covers the pledge,
    which is how a real cash-secured write settles."""
    portfolio = await store.ensure_portfolio(CTX)
    strike = (portfolio.cash + 500.0) / 100.0
    result = await _write_put(store, strike=strike, qty=1, premium=6.0)
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_naked_call_write_is_refused_end_to_end(store):
    result = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(),
        qty=1, premium=3.0,
    )
    assert result["ok"] is False
    assert "naked" in result["error"].lower()


@pytest.mark.asyncio
async def test_covered_call_write_succeeds_and_pledges_nothing(store):
    await store.buy(CTX, "ACME", 100, 50.0)
    cash = (await store.ensure_portfolio(CTX)).cash
    result = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(),
        qty=1, premium=3.0,
    )
    assert result["ok"] is True
    assert result["collateral"] == 0.0
    assert result["structure"] == "covered call"
    assert result["cash_after"] == pytest.approx(cash + 300.0)


@pytest.mark.asyncio
async def test_credit_spread_pledges_only_the_width(store):
    exp = _exp()
    long_leg = await store.buy_option(
        CTX, contract_symbol="ACME991231P00095000", underlying="ACME",
        option_type="put", strike=95.0, expiration=exp, qty=1, premium=1.0,
    )
    assert long_leg["ok"] is True
    result = await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=exp)
    assert result["ok"] is True
    assert result["collateral"] == pytest.approx(500.0)
    assert result["structure"] == "put credit spread"


@pytest.mark.asyncio
async def test_one_long_leg_cannot_cap_two_shorts(store):
    """The protective leg must cover every short in the expiry, or the
    second write falls back to full cash-secured collateral."""
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231P00095000", underlying="ACME",
        option_type="put", strike=95.0, expiration=exp, qty=1, premium=1.0,
    )
    first = await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=exp)
    assert first["collateral"] == pytest.approx(500.0)

    second = await store.write_option(
        CTX, contract_symbol="ACME991231P00101000", underlying="ACME",
        option_type="put", strike=101.0, expiration=exp, qty=1, premium=2.0,
    )
    # The single long put is already spoken for, so this one is naked-
    # but-secured, not a spread.
    assert second["ok"] is True
    assert second["collateral"] == pytest.approx(10_100.0)


@pytest.mark.asyncio
async def test_writing_a_contract_already_held_long_is_refused(store):
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231P00100000", underlying="ACME",
        option_type="put", strike=100.0, expiration=exp, qty=1, premium=2.0,
    )
    result = await _write_put(store, strike=100.0, expiration=exp)
    assert result["ok"] is False
    assert "already long" in result["error"].lower()


@pytest.mark.asyncio
async def test_buying_a_contract_held_short_is_refused(store):
    await _write_put(store)
    result = await store.buy_option(
        CTX, contract_symbol="ACME991231P00100000", underlying="ACME",
        option_type="put", strike=100.0, expiration=_exp(), qty=1, premium=2.0,
    )
    assert result["ok"] is False
    assert "portfolio_close_option" in result["error"]


@pytest.mark.asyncio
async def test_sell_option_refuses_a_short_position(store):
    await _write_put(store)
    result = await store.sell_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=1.0,
    )
    assert result["ok"] is False
    assert "short" in result["error"].lower()


# ── covered-call share lock ────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_shares_backing_a_covered_call_cannot_be_sold(store):
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(),
        qty=1, premium=3.0,
    )
    result = await store.sell(CTX, "ACME", 100, 55.0)
    assert result["ok"] is False
    assert "covering short calls" in result["error"]


@pytest.mark.asyncio
async def test_uncovered_excess_shares_stay_sellable(store):
    await store.buy(CTX, "ACME", 150, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(),
        qty=1, premium=3.0,
    )
    assert (await store.sell(CTX, "ACME", 50, 55.0))["ok"] is True
    assert (await store.sell(CTX, "ACME", 1, 55.0))["ok"] is False


@pytest.mark.asyncio
async def test_a_spread_short_call_does_not_lock_shares(store):
    """Only calls covered BY SHARES reserve them. A call capped by a
    long leg pledges collateral instead, so stock stays free."""
    exp = _exp()
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=2, premium=1.0,
    )
    written = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=exp, qty=2, premium=2.0,
    )
    assert written["collateral"] == pytest.approx(1_000.0)
    await _assert_short_book_consistent(store)
    assert (await store.sell(CTX, "ACME", 100, 55.0))["ok"] is True


# ── closing ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_buy_to_close_returns_collateral_and_books_pnl(store):
    opened = await _write_put(store, strike=100.0, qty=1, premium=2.0)
    cash_after_write = opened["cash_after"]

    closed = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=0.5,
    )
    assert closed["ok"] is True
    # Collected $200, paid back $50.
    assert closed["realized_pnl"] == pytest.approx(150.0)
    assert closed["collateral_released"] == pytest.approx(10_000.0)
    assert closed["cash_after"] == pytest.approx(cash_after_write - 50.0 + 10_000.0)
    assert await store.get_option_position(CTX, "ACME991231P00100000") is None


@pytest.mark.asyncio
async def test_partial_close_releases_pro_rata_collateral(store):
    await _write_put(store, strike=100.0, qty=2, premium=2.0)
    closed = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=1.0,
    )
    assert closed["collateral_released"] == pytest.approx(10_000.0)
    pos = await store.get_option_position(CTX, "ACME991231P00100000")
    assert pos.qty == pytest.approx(1)
    assert pos.collateral == pytest.approx(10_000.0)


@pytest.mark.asyncio
async def test_closing_more_than_held_is_refused(store):
    await _write_put(store, qty=1)
    result = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=5, premium=1.0,
    )
    assert result["ok"] is False
    assert "only short" in result["error"].lower()


@pytest.mark.asyncio
async def test_close_refuses_a_long_position(store):
    await store.buy_option(
        CTX, contract_symbol="ACME991231P00100000", underlying="ACME",
        option_type="put", strike=100.0, expiration=_exp(), qty=1, premium=2.0,
    )
    result = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=2.0,
    )
    assert result["ok"] is False
    assert "long position" in result["error"].lower()


@pytest.mark.asyncio
async def test_loss_on_a_short_is_the_mirror_of_a_long(store):
    await _write_put(store, strike=100.0, qty=1, premium=2.0)
    closed = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=5.0,
    )
    # Collected $200, bought back at $500.
    assert closed["realized_pnl"] == pytest.approx(-300.0)


# ── settlement ─────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_short_expiring_worthless_keeps_the_whole_credit(store):
    before = (await store.ensure_portfolio(CTX)).cash
    await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=time.time() - DAY)
    settled = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=0.0,
        source="settle", settlement_intrinsic=0.0,
    )
    assert settled["ok"] is True
    assert settled["realized_pnl"] == pytest.approx(200.0)
    assert settled["cash_after"] == pytest.approx(before + 200.0)


@pytest.mark.asyncio
async def test_short_settling_itm_pays_intrinsic_out_of_collateral(store):
    before = (await store.ensure_portfolio(CTX)).cash
    await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=time.time() - DAY)
    settled = await store.close_short_option(
        CTX, contract_symbol="ACME991231P00100000", qty=1, premium=8.0,
        source="settle", settlement_intrinsic=8.0,
    )
    # Collected $200, owed $800 at expiry.
    assert settled["realized_pnl"] == pytest.approx(-600.0)
    assert settled["cash_after"] == pytest.approx(before - 600.0)


@pytest.mark.asyncio
async def test_settlement_may_overdraw_but_a_voluntary_close_may_not(store):
    """Expiry is not optional, so it is allowed to run the book down;
    a discretionary buy-back is checked against available cash."""
    portfolio = await store.ensure_portfolio(CTX)
    strike = portfolio.cash / 100.0
    await _write_put(store, strike=strike, qty=1, premium=1.0)
    huge = strike * 10

    voluntary = await store.close_short_option(
        CTX, contract_symbol=f"ACME991231P{int(strike * 1000):08d}",
        qty=1, premium=huge,
    )
    assert voluntary["ok"] is False
    assert "insufficient cash" in voluntary["error"].lower()

    forced = await store.close_short_option(
        CTX, contract_symbol=f"ACME991231P{int(strike * 1000):08d}",
        qty=1, premium=huge, source="settle", settlement_intrinsic=huge,
    )
    assert forced["ok"] is True


# ── the pledged-cash invariant ─────────────────────────────────────────────

@pytest.mark.asyncio
async def test_pledged_collateral_cannot_be_spent_elsewhere(store):
    """The whole reason collateral is held out of cash rather than
    tracked separately: every existing cash check inherits it."""
    portfolio = await store.ensure_portfolio(CTX)
    strike = (portfolio.cash * 0.9) / 100.0
    written = await _write_put(store, strike=strike, qty=1, premium=1.0)
    assert written["ok"] is True

    spendable = written["cash_after"]
    over = await store.buy(CTX, "OTHER", 1, spendable + 1_000.0)
    assert over["ok"] is False
    assert (await store.buy(CTX, "OTHER", 1, spendable * 0.5))["ok"] is True


@pytest.mark.asyncio
async def test_option_position_defaults_to_long_for_legacy_rows(store):
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(), qty=1, premium=1.0,
    )
    pos = await store.get_option_position(CTX, "ACME991231C00100000")
    assert pos.side == LONG
    assert pos.is_short is False
    assert pos.collateral == 0.0
    assert pos.signed_qty() == 1


# ── protective-leg lock ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_protective_leg_cannot_be_sold_out_from_under_a_spread(store):
    """Selling the long leg would leave the naked short call that
    write_option refuses to open."""
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=1, premium=1.0,
    )
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=exp, qty=1, premium=3.0,
    )
    result = await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=1.0,
    )
    assert result["ok"] is False
    assert "capping short" in result["error"]


@pytest.mark.asyncio
async def test_the_leg_frees_up_once_the_short_is_closed(store):
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=1, premium=1.0,
    )
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=exp, qty=1, premium=3.0,
    )
    await store.close_short_option(
        CTX, contract_symbol="ACME991231C00100000", qty=1, premium=2.0,
    )
    result = await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=1.0,
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_excess_legs_beyond_the_obligation_stay_sellable(store):
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=3, premium=1.0,
    )
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=exp, qty=1, premium=3.0,
    )
    assert (await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=2, premium=1.0,
    ))["ok"] is True
    assert (await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=1.0,
    ))["ok"] is False


@pytest.mark.asyncio
async def test_a_cash_secured_put_does_not_lock_an_unrelated_long(store):
    """The short pledged its full strike, so a long put below it is not
    load-bearing and must stay sellable."""
    exp = _exp()
    await store.write_option(
        CTX, contract_symbol="ACME991231P00050000", underlying="ACME",
        option_type="put", strike=50.0, expiration=exp, qty=1, premium=2.0,
    )
    await store.buy_option(
        CTX, contract_symbol="ACME991231P00040000", underlying="ACME",
        option_type="put", strike=40.0, expiration=exp, qty=1, premium=0.5,
    )
    result = await store.sell_option(
        CTX, contract_symbol="ACME991231P00040000", qty=1, premium=0.5,
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_shares_covering_a_call_do_not_lock_option_legs(store):
    """A share-covered call pledges nothing, so a higher long call in the
    same expiry is incidental, not protective."""
    exp = _exp()
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=exp, qty=1, premium=3.0,
    )
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00120000", underlying="ACME",
        option_type="call", strike=120.0, expiration=exp, qty=1, premium=0.5,
    )
    result = await store.sell_option(
        CTX, contract_symbol="ACME991231C00120000", qty=1, premium=0.5,
    )
    assert result["ok"] is True


@pytest.mark.asyncio
async def test_a_leg_in_another_expiry_is_not_protection(store):
    """A calendar leg does not cap the loss, so it neither collateralizes
    the short nor gets locked."""
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=_exp(60), qty=1, premium=2.0,
    )
    written = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(30), qty=1, premium=3.0,
    )
    assert written["ok"] is False
    assert "naked" in written["error"].lower()
    assert (await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=2.0,
    ))["ok"] is True


# ── share reservation across writes ────────────────────────────────────────

@pytest.mark.asyncio
async def test_the_same_shares_cannot_cover_two_calls(store):
    """Writing one contract at a time against the same 100 shares must
    not report 'covered' twice — that is a naked call assembled in
    instalments."""
    await store.buy(CTX, "ACME", 100, 50.0)
    first = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(), qty=1, premium=3.0,
    )
    assert first["structure"] == "covered call"
    second = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(), qty=1, premium=3.0,
    )
    assert second["ok"] is False
    assert "naked" in second["error"].lower()
    await _assert_short_book_consistent(store)


@pytest.mark.asyncio
async def test_covered_shares_are_spent_across_strikes_and_expiries(store):
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(30), qty=1, premium=3.0,
    )
    other_strike = await store.write_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=_exp(30), qty=1, premium=2.0,
    )
    other_expiry = await store.write_option(
        CTX, contract_symbol="ACME991231C00100000X", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(60), qty=1, premium=4.0,
    )
    assert other_strike["ok"] is False
    assert other_expiry["ok"] is False
    await _assert_short_book_consistent(store)


@pytest.mark.asyncio
async def test_two_hundred_shares_cover_two_separate_calls(store):
    await store.buy(CTX, "ACME", 200, 50.0)
    for strike in (100.0, 110.0):
        result = await store.write_option(
            CTX, contract_symbol=f"ACME991231C{int(strike * 1000):08d}",
            underlying="ACME", option_type="call", strike=strike,
            expiration=_exp(), qty=1, premium=2.0,
        )
        assert result["ok"] is True, result.get("error")
        assert result["structure"] == "covered call"
    await _assert_short_book_consistent(store)


@pytest.mark.asyncio
async def test_closing_a_covered_call_returns_the_shares_to_service(store):
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=_exp(), qty=1, premium=3.0,
    )
    await store.close_short_option(
        CTX, contract_symbol="ACME991231C00100000", qty=1, premium=1.0,
    )
    again = await store.write_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=_exp(), qty=1, premium=2.0,
    )
    assert again["ok"] is True
    assert again["structure"] == "covered call"


# ── settlement must not be blocked by the protective-leg lock ──────────────

@pytest.mark.asyncio
async def test_a_spread_long_leg_can_still_settle(store):
    """Both legs expire the same day. Refusing the long would strand it
    until the next sweep and mark it against a different spot."""
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=1, premium=1.0,
    )
    await store.write_option(
        CTX, contract_symbol="ACME991231C00100000", underlying="ACME",
        option_type="call", strike=100.0, expiration=exp, qty=1, premium=3.0,
    )
    blocked = await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=1.0,
    )
    assert blocked["ok"] is False

    settled = await store.sell_option(
        CTX, contract_symbol="ACME991231C00110000", qty=1, premium=0.0,
        source="settle", settlement_intrinsic=0.0,
    )
    assert settled["ok"] is True


# ── order routing on shorts ────────────────────────────────────────────────

class _StubProviders:
    """Enough of ProviderManager for the executor's order paths."""

    providers: list = []

    async def get_quote(self, symbol):
        raise RuntimeError("no quotes in this test")


@pytest.mark.asyncio
async def test_a_sell_order_on_a_short_is_refused_with_the_right_verb(store):
    from src.paper_portfolio_executor import PaperPortfolioExecutor

    await _write_put(store, strike=100.0, qty=1, premium=2.0)
    ex = PaperPortfolioExecutor(store, _StubProviders())
    result = await ex.place_order(
        CTX, ticker="ACME", side="sell", kind="stop",
        trigger_price=1.0, qty=1, reason="stop",
        contract="ACME991231P00100000",
    )
    assert result["ok"] is False
    assert "buy+stop" in result["error"]


@pytest.mark.asyncio
async def test_settlement_leaves_an_unquotable_short_open(store):
    """Force-settling a short at $0 would book the entire credit as
    profit on exactly the rows most likely to have expired ITM."""
    from src.paper_portfolio_executor import PaperPortfolioExecutor

    await _write_put(
        store, strike=100.0, qty=1, premium=2.0,
        expiration=time.time() - 10 * DAY,
    )
    ex = PaperPortfolioExecutor(store, _StubProviders())
    stats = await ex.settle_expired_options()
    assert stats["settled"] == 0
    assert stats["errors"] == 1
    assert await store.get_option_position(CTX, "ACME991231P00100000") is not None


# ── one row, one collateral basis ──────────────────────────────────────────

@pytest.mark.asyncio
async def test_adding_to_a_covered_call_reprices_the_whole_row(store):
    """A row must not mix bases. Once the second contract outgrows the
    share cover, the whole row is priced as the spread it became — a
    per-tranche pledge would leave it short by a full width while
    `collateral > 0` told sell() the shares were free."""
    exp = _exp()
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=2, premium=1.0,
    )
    first = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=exp, qty=1, premium=3.0,
    )
    assert first["structure"] == "covered call"
    assert first["collateral"] == 0.0

    second = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=exp, qty=1, premium=3.0,
    )
    assert second["ok"] is True
    assert second["structure"] == "call credit spread"

    pos = await store.get_option_position(CTX, "ACME991231C00105000")
    # Max loss on 2 contracts of a 5-wide spread is $1,000, all pledged.
    assert pos.qty == pytest.approx(2)
    assert pos.collateral == pytest.approx(1_000.0)
    await _assert_short_book_consistent(store)


@pytest.mark.asyncio
async def test_a_fully_spread_backed_row_releases_the_shares(store):
    """Once nothing on the row depends on stock, the stock is free."""
    exp = _exp()
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=2, premium=1.0,
    )
    for _ in range(2):
        await store.write_option(
            CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
            option_type="call", strike=105.0, expiration=exp, qty=1, premium=3.0,
        )
    assert (await store.sell(CTX, "ACME", 100, 55.0))["ok"] is True


@pytest.mark.asyncio
async def test_adding_to_a_covered_call_with_no_leg_is_still_refused(store):
    """No protective leg means the second contract has nothing to fall
    back on, so the whole row is naked and the write is rejected."""
    await store.buy(CTX, "ACME", 100, 50.0)
    await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=_exp(), qty=1, premium=3.0,
    )
    second = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=_exp(), qty=1, premium=3.0,
    )
    assert second["ok"] is False
    assert "naked" in second["error"].lower()


@pytest.mark.asyncio
async def test_adding_to_a_short_put_pledges_the_second_strike_too(store):
    await _write_put(store, strike=100.0, qty=1, premium=2.0)
    second = await _write_put(store, strike=100.0, qty=1, premium=2.0)
    assert second["ok"] is True
    assert second["collateral"] == pytest.approx(10_000.0)
    pos = await store.get_option_position(CTX, "ACME991231P00100000")
    assert pos.collateral == pytest.approx(20_000.0)


@pytest.mark.asyncio
async def test_a_row_that_gets_cheaper_refunds_the_difference(store):
    """A cash-secured put that gains a protective leg becomes a spread,
    and the row is re-priced down to the width. Refusing to refund would
    strand the difference AND, worse, keep the row out of whichever
    collateral bucket now describes it."""
    exp = _exp()
    await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=exp)
    pos = await store.get_option_position(CTX, "ACME991231P00100000")
    assert pos.collateral == pytest.approx(10_000.0)

    await store.buy_option(
        CTX, contract_symbol="ACME991231P00095000", underlying="ACME",
        option_type="put", strike=95.0, expiration=exp, qty=2, premium=1.0,
    )
    second = await _write_put(store, strike=100.0, qty=1, premium=2.0, expiration=exp)
    assert second["ok"] is True
    # 2 contracts of a 5-wide spread: $1,000, down from $10,000.
    pos = await store.get_option_position(CTX, "ACME991231P00100000")
    assert pos.collateral == pytest.approx(1_000.0)
    assert second["collateral"] == pytest.approx(-9_000.0)


@pytest.mark.asyncio
async def test_a_row_backed_by_stock_never_keeps_a_pledge(store):
    """The invariant both consumers depend on: collateral == 0 exactly
    when stock is doing the work. A row that re-prices to covered must
    drop its pledge, or `sell()` stops seeing it and releases the very
    shares backing it."""
    exp = _exp()
    await store.buy_option(
        CTX, contract_symbol="ACME991231C00110000", underlying="ACME",
        option_type="call", strike=110.0, expiration=exp, qty=1, premium=1.0,
    )
    spread = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=exp, qty=1, premium=3.0,
    )
    assert spread["collateral"] == pytest.approx(500.0)

    await store.buy(CTX, "ACME", 200, 50.0)
    second = await store.write_option(
        CTX, contract_symbol="ACME991231C00105000", underlying="ACME",
        option_type="call", strike=105.0, expiration=exp, qty=1, premium=3.0,
    )
    assert second["ok"] is True
    assert second["structure"] == "covered call"

    pos = await store.get_option_position(CTX, "ACME991231C00105000")
    assert pos.qty == pytest.approx(2)
    assert pos.collateral == pytest.approx(0.0)

    # The shares are what backs it now, so they must not be sellable.
    sold = await store.sell(CTX, "ACME", 200, 55.0)
    assert sold["ok"] is False
    assert "covering short calls" in sold["error"]
    await _assert_short_book_consistent(store)
