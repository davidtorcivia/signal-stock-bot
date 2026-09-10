"""
Tests for the credit dashboard and the options-derived vol read-outs.

The arithmetic is the whole product here, so these are known-answer
tests over hand-built inputs rather than round-trips through a provider.
Network paths (yfinance, FRED) are stubbed.
"""

import datetime as dt
import math
from types import SimpleNamespace

import pytest

from src.commands.base import CommandContext
from src.commands.macro_commands import (
    MONTH_ROWS,
    WEEK_ROWS,
    CreditCommand,
    PricedInCommand,
    _bp,
    _bp_delta,
    _diff_change,
    _load,
    _Row,
)
from src.commands.options_analytics import (
    MAX_IV,
    FlowCommand,
    RealizedVolCommand,
    SkewCommand,
    _annualized,
    _atm_iv,
    _close_to_close,
    _intraday_rv,
    _nearest,
    _resolve_expiry_arg,
    _resolve_symbol_arg,
    _usable,
)


def _q(strike, kind, iv, price=1.0, volume=0, oi=0):
    return SimpleNamespace(
        strike=float(strike), type=kind, implied_volatility=iv,
        price=float(price), volume=volume, open_interest=oi,
        symbol=f"X{int(strike)}{kind[0].upper()}",
    )


def _ctx(name, args):
    return CommandContext(
        sender="t", group_id=None, raw_message=f"!{name}",
        command=name, args=args,
    )


# ── chain hygiene ──────────────────────────────────────────────────────────

@pytest.mark.parametrize("iv,price,ok", [
    (0.25, 1.0, True),
    (None, 1.0, False),
    (float("nan"), 1.0, False),
    (0.001, 1.0, False),          # below MIN_IV — stale quote artifact
    (MAX_IV + 0.1, 1.0, False),   # 300%+ vol on a listed name is junk
    (0.25, 0.0, False),           # never quoted
])
def test_usable_filters_junk(iv, price, ok):
    assert _usable(_q(100, "call", iv, price)) is ok


def test_nearest_picks_by_moneyness_not_absolute_strike():
    spot = 100.0
    chain = [_q(90, "put", 0.30), _q(95, "put", 0.26), _q(99, "put", 0.22)]
    assert _nearest(chain, 0.95, spot, "put").strike == 95.0


def test_nearest_skips_unusable_and_far_wings():
    spot = 100.0
    chain = [
        _q(95, "put", 8.0),      # exactly at 0.95 but junk IV
        _q(90, "put", 0.30),     # usable
        _q(50, "put", 0.28),     # usable but outside the moneyness window
    ]
    assert _nearest(chain, 0.95, spot, "put").strike == 90.0


def test_nearest_returns_none_when_nothing_usable():
    assert _nearest([_q(95, "put", None)], 0.95, 100.0, "put") is None


def test_atm_iv_averages_call_and_put():
    chain = [_q(100, "call", 0.20), _q(100, "put", 0.30)]
    assert _atm_iv(chain, 100.0) == pytest.approx(0.25)


def test_atm_iv_uses_one_side_when_the_other_is_junk():
    chain = [_q(100, "call", 0.20), _q(100, "put", 9.0)]
    assert _atm_iv(chain, 100.0) == pytest.approx(0.20)


def test_atm_iv_none_on_empty_chain():
    assert _atm_iv([], 100.0) is None


# ── realized vol arithmetic ────────────────────────────────────────────────

def test_annualized_matches_hand_computation():
    returns = [0.01, -0.01, 0.01, -0.01]
    # sample stdev of that set is 0.011547...; annualize by sqrt(252)
    expected = math.sqrt(
        sum((r - 0.0) ** 2 for r in returns) / 3
    ) * math.sqrt(252)
    assert _annualized(returns, 252) == pytest.approx(expected)


def test_annualized_needs_two_returns():
    assert _annualized([0.01], 252) is None
    assert _annualized([], 252) is None


def test_close_to_close_uses_window_plus_one_close():
    closes = [100.0 * (1.01 ** i) for i in range(50)]
    # A perfectly constant growth rate has zero return variance.
    assert _close_to_close(closes, 20) == pytest.approx(0.0, abs=1e-9)


def test_close_to_close_ignores_nonpositive_prices():
    """A zero close is bad data: it drops the two returns that touch it
    rather than producing a log-of-zero blow-up."""
    clean = [100.0, 101.0, 102.0, 103.0, 104.0]
    dirty = [100.0, 101.0, 0.0, 102.0, 103.0, 104.0]
    assert _close_to_close(dirty, 20) is not None
    assert not math.isnan(_close_to_close(dirty, 20))
    assert _close_to_close(clean, 20) is not None


def test_intraday_rv_does_not_span_the_overnight_gap():
    """A gap between sessions must not be annualized as a 5-minute move."""
    day1 = dt.date(2026, 9, 8)
    day2 = dt.date(2026, 9, 9)
    flat_then_gap = (
        [(day1, 100.0), (day1, 100.0), (day1, 100.0)]
        + [(day2, 130.0), (day2, 130.0), (day2, 130.0)]
    )
    # Every within-session return is zero; only a cross-session return
    # would register, and there must not be one.
    assert _intraday_rv(flat_then_gap, sessions=2) == pytest.approx(0.0)


def test_intraday_rv_limits_to_requested_sessions():
    days = [dt.date(2026, 9, d) for d in (7, 8, 9)]
    bars = []
    for i, day in enumerate(days):
        base = 100.0
        step = 0.0 if i < 2 else 1.0   # only the last session moves
        bars += [(day, base), (day, base + step), (day, base + 2 * step)]
    assert _intraday_rv(bars, sessions=2) > 0        # includes the mover
    assert _intraday_rv(bars[:6], sessions=2) == pytest.approx(0.0)


def test_intraday_rv_none_without_bars():
    assert _intraday_rv([], sessions=5) is None


# ── argument parsing ───────────────────────────────────────────────────────

def test_symbol_and_expiry_args_are_told_apart():
    ctx = _ctx("skew", ["nvda", "2026-10-16"])
    assert _resolve_symbol_arg(ctx) == "NVDA"
    assert _resolve_expiry_arg(ctx) == "2026-10-16"


def test_symbol_arg_ignores_flags_and_missing_symbol():
    assert _resolve_symbol_arg(_ctx("skew", ["-help"])) is None
    assert _resolve_symbol_arg(_ctx("skew", [])) is None


def test_expiry_arg_rejects_non_iso():
    assert _resolve_expiry_arg(_ctx("skew", ["SPY", "oct16"])) is None


# ── command error paths (no network) ───────────────────────────────────────

class _Providers:
    """Stands in for ProviderManager. `providers` stays empty so the
    FRED lookup in macro_commands finds nothing."""

    def __init__(self, expirations=None, chain=None, spot=None):
        self.providers = []
        self._expirations = expirations or []
        self._chain = chain or []
        self._spot = spot

    async def get_option_expirations(self, underlying):
        if not self._expirations:
            raise RuntimeError("no options")
        return self._expirations

    async def get_options_chain(self, underlying, expiration=None, limit=100):
        return self._chain

    async def get_quote(self, symbol):
        if self._spot is None:
            raise RuntimeError("no quote")
        return SimpleNamespace(price=self._spot)


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [SkewCommand, FlowCommand, RealizedVolCommand])
async def test_commands_require_a_symbol(cls):
    result = await cls(_Providers()).execute(_ctx(cls.name, []))
    assert result.success is False


@pytest.mark.asyncio
async def test_skew_reports_missing_chain_rather_than_raising():
    cmd = SkewCommand(_Providers(spot=100.0))
    result = await cmd.execute(_ctx("skew", ["SPY"]))
    assert result.success is False
    assert "no listed options" in result.text.lower()


@pytest.mark.asyncio
async def test_skew_rejects_an_unlisted_expiry():
    cmd = SkewCommand(_Providers(expirations=["2026-10-16"], spot=100.0))
    result = await cmd.execute(_ctx("skew", ["SPY", "2027-01-15"]))
    assert result.success is False
    assert "2026-10-16" in result.text


@pytest.mark.asyncio
async def test_skew_computes_from_a_stub_chain():
    chain = [
        _q(95, "put", 0.30), _q(100, "put", 0.24), _q(100, "call", 0.22),
        _q(105, "call", 0.18), _q(90, "put", 0.34), _q(110, "call", 0.16),
    ]
    far = (dt.date.today() + dt.timedelta(days=40)).isoformat()
    cmd = SkewCommand(_Providers(expirations=[far], chain=chain, spot=100.0))
    result = await cmd.execute(_ctx("skew", ["SPY"]))
    assert result.success is True
    assert "ATM IV" in result.text
    # 95-strike put 30% vs 105-strike call 18% => +12.0 vol points
    assert "+12.0 vol pts" in result.text


@pytest.mark.asyncio
async def test_flow_counts_volume_by_side():
    chain = [
        _q(100, "call", 0.2, volume=300, oi=1000),
        _q(105, "call", 0.2, volume=100, oi=500),
        _q(95, "put", 0.3, volume=800, oi=2000),
    ]
    far = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    cmd = FlowCommand(_Providers(expirations=[far], chain=chain, spot=100.0))
    result = await cmd.execute(_ctx("flow", ["SPY"]))
    assert result.success is True
    assert "puts     800" in result.text
    assert "2.00" in result.text          # 800 puts / 400 calls
    assert "defensive" in result.text


@pytest.mark.asyncio
async def test_flow_survives_a_session_with_no_volume():
    chain = [_q(100, "call", 0.2, volume=0, oi=10)]
    far = (dt.date.today() + dt.timedelta(days=30)).isoformat()
    cmd = FlowCommand(_Providers(expirations=[far], chain=chain, spot=100.0))
    result = await cmd.execute(_ctx("flow", ["SPY"]))
    assert result.success is True
    assert "No volume yet" in result.text


# ── credit dashboard ───────────────────────────────────────────────────────

def _series(*values):
    """Newest-first (date, value) rows, one per trading day."""
    return [(f"2026-09-{30 - i:02d}", v) for i, v in enumerate(values)]


def test_row_change_reads_the_right_lookback():
    row = _Row("HY", "X", _series(*[3.0] * WEEK_ROWS, 2.5))
    assert row.change(WEEK_ROWS) == pytest.approx(0.5)


def test_row_change_none_when_history_is_short():
    assert _Row("HY", "X", _series(3.0)).change(MONTH_ROWS) is None


def test_diff_change_tracks_the_spread_not_the_legs():
    """Both legs up 50bp is a zero change in the differential."""
    hy = _Row("HY", "X", _series(*([3.5] + [3.0] * WEEK_ROWS)))
    ig = _Row("IG", "Y", _series(*([1.5] + [1.0] * WEEK_ROWS)))
    assert _diff_change(hy, ig, WEEK_ROWS) == pytest.approx(0.0)


def test_bp_formatting_converts_percentage_points():
    assert _bp(2.71).strip() == "271"
    assert _bp(None).strip() == "n/a"
    assert _bp_delta(0.05).strip() == "+5"
    assert _bp_delta(-0.11).strip() == "-11"
    assert _bp_delta(None).strip() == "."


@pytest.mark.asyncio
async def test_load_returns_empty_without_a_fred_provider():
    assert await _load(_Providers(), {"HY": "BAMLH0A0HYM2"}) == {}


@pytest.mark.asyncio
async def test_load_drops_only_the_series_that_failed():
    class _Fred:
        name = "fred"

        async def get_series(self, series_id, limit=30):
            if series_id == "DEAD":
                raise RuntimeError("discontinued")
            return _series(2.0, 1.9)

    providers = _Providers()
    providers.providers = [_Fred()]
    rows = await _load(providers, {"HY": "GOOD", "SWAP": "DEAD"})
    assert set(rows) == {"HY"}


@pytest.mark.asyncio
@pytest.mark.parametrize("cls", [CreditCommand, PricedInCommand])
async def test_macro_commands_report_missing_fred(cls):
    result = await cls(_Providers()).execute(_ctx(cls.name, []))
    assert result.success is False


@pytest.mark.asyncio
async def test_credit_renders_spreads_and_the_hy_ig_differential():
    class _Fred:
        name = "fred"

        async def get_series(self, series_id, limit=30):
            table = {
                "BAMLH0A0HYM2": _series(*([2.71] + [2.66] * MONTH_ROWS)),
                "BAMLC0A0CM": _series(*([0.81] + [0.81] * MONTH_ROWS)),
            }
            if series_id not in table:
                raise RuntimeError("not in this fixture")
            return table[series_id]

    providers = _Providers()
    providers.providers = [_Fred()]
    result = await CreditCommand(providers).execute(_ctx("credit", []))
    assert result.success is True
    assert "HY     271" in result.text
    assert "IG      81" in result.text
    # 271 - 81 = 190bp, and HY widened 5bp against a flat IG.
    assert "HY-IG  190    +5" in result.text
