"""
Positioning and volatility read-outs derived from the options chain.

Provides: !skew, !flow, !rvol

All three run off one chain snapshot plus a spot quote, so they need no
data source beyond what the options provider already serves. Skew is
measured in moneyness rather than delta on purpose: delta would need a
Black-Scholes inversion with a rate and dividend assumption per name,
and the 95/105 pair answers the same question (what does downside cost
relative to upside) without inventing those inputs.

Chain hygiene matters more than the arithmetic. A free chain carries
rows that never traded, one-sided books, and implied vols in the
hundreds of percent on far wings where the pricing model divides by
almost nothing. `_usable` drops those before anything is averaged; skip
it and a single 900%-IV strike drags the whole surface.
"""

import asyncio
import logging
import math
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager
from ..executor import run_blocking

logger = logging.getLogger(__name__)

# Whole chain for one expiry. The provider slices by strike, so a small
# limit returns a wing rather than a cross-section.
FULL_CHAIN = 10_000

# Junk filter. An IV outside this band on a listed equity option means
# the model divided by a stale or crossed quote, not that the market
# expects a 400% move.
MIN_IV = 0.02
MAX_IV = 3.0

# Moneyness window kept for surface work. Beyond this the quotes are
# pennies wide and the implied vols stop meaning anything.
MIN_MONEY = 0.70
MAX_MONEY = 1.30

# Skip the front expiry for skew: 0DTE and next-day vols are dominated
# by pin risk and print as either near-zero or absurd.
MIN_SKEW_DAYS = 14

TRADING_DAYS = 252
# 6.5h session / 5min bars.
BARS_PER_DAY = 78


def _usable(quote) -> bool:
    """True when this contract's implied vol is worth averaging."""
    iv = quote.implied_volatility
    return (
        iv is not None
        and math.isfinite(iv)
        and MIN_IV <= iv <= MAX_IV
        and quote.price > 0
    )


def _nearest(quotes: list, target_money: float, spot: float, kind: str):
    """Contract of `kind` whose strike sits closest to `target_money`."""
    pool = [
        q for q in quotes
        if q.type == kind and _usable(q)
        and MIN_MONEY <= q.strike / spot <= MAX_MONEY
    ]
    if not pool:
        return None
    return min(pool, key=lambda q: abs(q.strike / spot - target_money))


def _atm_iv(quotes: list, spot: float) -> Optional[float]:
    """Average of the nearest usable call and put to spot."""
    call = _nearest(quotes, 1.0, spot, "call")
    put = _nearest(quotes, 1.0, spot, "put")
    ivs = [q.implied_volatility for q in (call, put) if q is not None]
    return sum(ivs) / len(ivs) if ivs else None


def _pct(value: Optional[float]) -> str:
    return " n/a" if value is None else f"{value * 100:5.1f}%"


async def _spot(providers: ProviderManager, symbol: str) -> Optional[float]:
    try:
        return float((await providers.get_quote(symbol)).price)
    except Exception as e:
        logger.warning(f"options analytics: spot lookup failed for {symbol}: {e}")
        return None


async def _expirations(providers: ProviderManager, symbol: str) -> list[str]:
    """Listed expiries, ISO, ascending. Empty when the name has no chain."""
    try:
        return list(await providers.get_option_expirations(symbol))
    except Exception as e:
        logger.warning(f"options analytics: expiry list failed for {symbol}: {e}")
        return []


def _days_to(expiry: str) -> int:
    import datetime as dt
    try:
        return (dt.date.fromisoformat(expiry) - dt.date.today()).days
    except ValueError:
        return 0


def _resolve_symbol_arg(ctx: CommandContext) -> Optional[str]:
    for arg in ctx.args:
        token = arg.strip().upper()
        if token and not token.startswith("-") and "-" not in token:
            return token
    return None


def _resolve_expiry_arg(ctx: CommandContext) -> Optional[str]:
    import re
    for arg in ctx.args:
        if re.match(r"^\d{4}-\d{2}-\d{2}$", arg.strip()):
            return arg.strip()
    return None


class SkewCommand(BaseCommand):
    """Implied vol by moneyness: what downside costs vs upside."""

    name = "skew"
    aliases = ["vol", "smile"]
    description = (
        "Implied volatility skew and term structure for a symbol. Shows ATM "
        "IV, put IV vs call IV at matched moneyness (95/105 and 90/110), the "
        "skew in vol points, and ATM IV across the next expiries. Positive "
        "skew means downside protection is bid, which is the normal state; a "
        "flattening or inverted skew means calls are being chased. Use this "
        "rather than reading premiums off a chain, which conflates vol with "
        "moneyness and time."
    )
    usage = "!skew SPY  or  !skew NVDA 2026-10-16"
    help_explanation = """Implied vol surface for one name.

**ATM IV** — the market's expected annualized move.

**Skew (95/105)** — IV of the 5%-out put minus IV of the 5%-out call.
Positive is normal: crash protection costs more than upside. Watch the
change in it, not the level.

**Term structure** — ATM IV by expiry. Upward sloping is calm; inverted
(front above back) means an event is priced into the near date.

Skipped: expiries under two weeks, where pin risk dominates the vols."""

    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()
        symbol = _resolve_symbol_arg(ctx)
        if not symbol:
            return CommandResult.error("Specify a symbol: !skew SPY")

        spot = await _spot(self.providers, symbol)
        if not spot:
            return CommandResult.error(f"No spot price for {symbol}.")

        expiries = await _expirations(self.providers, symbol)
        if not expiries:
            return CommandResult.error(f"No listed options for {symbol}.")

        wanted = _resolve_expiry_arg(ctx)
        if wanted and wanted not in expiries:
            return CommandResult.error(
                f"{symbol} has no {wanted} expiry. Listed: "
                f"{', '.join(expiries[:8])}"
            )
        if not wanted:
            wanted = next(
                (e for e in expiries if _days_to(e) >= MIN_SKEW_DAYS),
                expiries[-1],
            )

        try:
            chain = await self.providers.get_options_chain(
                symbol, wanted, FULL_CHAIN,
            )
        except Exception as e:
            return CommandResult.error(f"Chain fetch failed for {symbol}: {e}")

        usable = [q for q in chain if _usable(q)]
        if not usable:
            return CommandResult.error(
                f"{symbol} {wanted}: no contracts with a usable quote."
            )

        atm = _atm_iv(chain, spot)
        lines = [
            f"◈ SKEW {symbol}  spot {spot:,.2f}",
            f"  expiry {wanted} ({_days_to(wanted)}d)",
            "",
            f"  ATM IV   {_pct(atm)}",
        ]

        for lo, hi in ((0.95, 1.05), (0.90, 1.10)):
            put = _nearest(chain, lo, spot, "put")
            call = _nearest(chain, hi, spot, "call")
            if put is None or call is None:
                continue
            skew = put.implied_volatility - call.implied_volatility
            lines.append(
                f"  {int(lo * 100)}P {_pct(put.implied_volatility)}  "
                f"{int(hi * 100)}C {_pct(call.implied_volatility)}  "
                f"skew {skew * 100:+5.1f} vol pts"
            )

        # Term structure. The requested expiry is already in hand, so
        # only the others are fetched, and they go out together: these
        # are blocking chain downloads and running four in series is
        # four times the latency for no reason.
        others = [
            e for e in expiries
            if e != wanted and _days_to(e) >= MIN_SKEW_DAYS
        ][:3]
        legs = await asyncio.gather(
            *(
                self.providers.get_options_chain(symbol, e, FULL_CHAIN)
                for e in others
            ),
            return_exceptions=True,
        )
        term = [(wanted, atm)] if atm is not None else []
        for expiry, leg in zip(others, legs):
            if isinstance(leg, BaseException):
                continue
            leg_atm = _atm_iv(leg, spot)
            if leg_atm is not None:
                term.append((expiry, leg_atm))
        term.sort(key=lambda pair: _days_to(pair[0]))

        shown = len(term)
        if term:
            lines += ["", "  Term structure (ATM IV)"]
            for expiry, value in term:
                lines.append(
                    f"    {expiry} ({_days_to(expiry):>3d}d)  {_pct(value)}"
                )

        if shown >= 2:
            lines.append("")
            lines.append(
                "  Front above back = event priced into the near date."
            )
        return CommandResult.ok("\n".join(lines))


class FlowCommand(BaseCommand):
    """Where today's option volume actually went."""

    name = "flow"
    aliases = ["optionflow", "positioning"]
    description = (
        "Options positioning for a symbol: today's call vs put volume, the "
        "put/call ratio, volume relative to open interest, and the strikes "
        "carrying the most volume and the most open interest. Volume over "
        "open interest is new positioning; open interest alone is what is "
        "already on the books. Use this to see what is being bought for a "
        "given expiry instead of inferring it from premium levels. Quotes "
        "are delayed roughly 15 minutes."
    )
    usage = "!flow SPY  or  !flow TSLA 2026-10-16"
    help_explanation = """Options flow for one name.

**Put/call volume** — today's contracts traded. Above ~1.0 is defensive.

**Vol/OI** — today's volume over existing open interest. High means new
positions opening rather than existing ones changing hands.

**Top strikes** — by volume (what traded today) and by open interest
(what is already sitting there). Big OI strikes act as magnets into
expiry.

Yahoo quotes are delayed ~15 minutes, and volume resets each session, so
early in the day the ratios are noisy."""

    TOP_N = 5

    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()
        symbol = _resolve_symbol_arg(ctx)
        if not symbol:
            return CommandResult.error("Specify a symbol: !flow SPY")

        expiries = await _expirations(self.providers, symbol)
        if not expiries:
            return CommandResult.error(f"No listed options for {symbol}.")
        wanted = _resolve_expiry_arg(ctx) or expiries[0]
        if wanted not in expiries:
            return CommandResult.error(
                f"{symbol} has no {wanted} expiry. Listed: "
                f"{', '.join(expiries[:8])}"
            )

        try:
            chain = await self.providers.get_options_chain(
                symbol, wanted, FULL_CHAIN,
            )
        except Exception as e:
            return CommandResult.error(f"Chain fetch failed for {symbol}: {e}")
        if not chain:
            return CommandResult.error(f"{symbol} {wanted}: empty chain.")

        spot = await _spot(self.providers, symbol)
        calls = [q for q in chain if q.type == "call"]
        puts = [q for q in chain if q.type == "put"]
        call_vol = sum(q.volume for q in calls)
        put_vol = sum(q.volume for q in puts)
        call_oi = sum(q.open_interest for q in calls)
        put_oi = sum(q.open_interest for q in puts)
        total_vol, total_oi = call_vol + put_vol, call_oi + put_oi

        header = f"◈ FLOW {symbol}  {wanted} ({_days_to(wanted)}d)"
        if spot:
            header += f"  spot {spot:,.2f}"
        lines = [header, ""]

        if total_vol == 0:
            lines.append("  No volume yet this session.")
        else:
            pcr = (put_vol / call_vol) if call_vol else float("inf")
            tone = "defensive" if pcr > 1 else "call-heavy"
            lines += [
                f"  Volume   calls {call_vol:>7,}   puts {put_vol:>7,}",
                f"  Put/call {pcr:>6.2f}  ({tone})",
            ]
        if total_oi:
            oi_pcr = (put_oi / call_oi) if call_oi else float("inf")
            lines.append(
                f"  Open int calls {call_oi:>7,}   puts {put_oi:>7,}  "
                f"P/C {oi_pcr:.2f}"
            )
            if total_vol:
                ratio = total_vol / total_oi
                if _days_to(wanted) <= 1:
                    note = "0DTE — intraday churn, not positioning"
                elif ratio > 0.3:
                    note = "new positioning"
                else:
                    note = "mostly existing"
                lines.append(f"  Vol/OI   {ratio:>6.2f}  ({note})")

        by_volume = sorted(chain, key=lambda q: q.volume, reverse=True)
        top_vol = [q for q in by_volume if q.volume > 0][: self.TOP_N]
        if top_vol:
            lines += ["", "  Most traded today"]
            lines += [self._strike_line(q, spot) for q in top_vol]

        by_oi = sorted(chain, key=lambda q: q.open_interest, reverse=True)
        top_oi = [q for q in by_oi if q.open_interest > 0][: self.TOP_N]
        if top_oi:
            lines += ["", "  Largest open interest"]
            lines += [self._strike_line(q, spot, use_oi=True) for q in top_oi]

        lines += ["", "  Yahoo chain, delayed ~15min."]
        return CommandResult.ok("\n".join(lines))

    @staticmethod
    def _strike_line(quote, spot: Optional[float], use_oi: bool = False) -> str:
        kind = "C" if quote.type == "call" else "P"
        count = quote.open_interest if use_oi else quote.volume
        label = "oi" if use_oi else "vol"
        money = ""
        if spot:
            money = f" ({quote.strike / spot - 1:+.1%})"
        return (
            f"    {quote.strike:>8,.1f}{kind}  {label} {count:>7,}  "
            f"${quote.price:>6.2f}{money}"
        )


class RealizedVolCommand(BaseCommand):
    """Realized vol against implied — is vol cheap or dear."""

    name = "rvol"
    aliases = ["realized", "vrp"]
    description = (
        "Realized volatility for a symbol against implied. Shows intraday "
        "realized vol from 5-minute bars (last session and 5-session), "
        "close-to-close realized over 20 and 60 days, and the variance risk "
        "premium — ATM implied minus 20-day realized. A positive premium "
        "means options are pricing more movement than the stock has "
        "delivered, which is when selling vol gets paid. All figures "
        "annualized in percent."
    )
    usage = "!rvol SPY"
    help_explanation = """Realized vs implied volatility.

**Intraday RV** — from 5-minute bars, annualized. Catches movement that
close-to-close misses when a name round-trips within the day.

**RV 20d / 60d** — close-to-close, the standard measure.

**VRP** — ATM implied minus 20-day realized, in vol points. Positive is
the normal state and is what option sellers collect. Negative means
realized is outrunning implied: bad time to be short vol.

Compare the two RV numbers: intraday well above close-to-close means the
name is choppy but mean-reverting."""

    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()
        symbol = _resolve_symbol_arg(ctx)
        if not symbol:
            return CommandResult.error("Specify a symbol: !rvol SPY")

        daily, intraday = await run_blocking(
            _fetch_bars, symbol, timeout=45.0,
        )
        if not daily:
            return CommandResult.error(f"No price history for {symbol}.")

        rv20 = _close_to_close(daily, 20)
        rv60 = _close_to_close(daily, 60)
        rv_day = _intraday_rv(intraday, sessions=1)
        rv_week = _intraday_rv(intraday, sessions=5)

        lines = [f"◈ RVOL {symbol}", ""]
        if rv_day is not None:
            lines.append(f"  Intraday (last session) {_pct(rv_day)}")
        if rv_week is not None:
            lines.append(f"  Intraday (5 sessions)   {_pct(rv_week)}")
        lines.append(f"  Close-to-close 20d      {_pct(rv20)}")
        lines.append(f"  Close-to-close 60d      {_pct(rv60)}")

        implied = await self._atm_implied(symbol)
        if implied is not None and rv20 is not None:
            vrp = implied - rv20
            verdict = (
                "vol is dear — selling gets paid"
                if vrp > 0 else
                "realized is outrunning implied — bad time to be short vol"
            )
            lines += [
                "",
                f"  ATM implied             {_pct(implied)}",
                f"  VRP (IV - RV20)         {vrp * 100:+5.1f} vol pts",
                f"  {verdict}",
            ]
        elif implied is None:
            lines += ["", "  No usable implied vol — chain empty or unquoted."]

        if rv_day is not None and rv20 is not None and rv20 > 0:
            ratio = rv_day / rv20
            if ratio > 1.5:
                lines.append("  Intraday well above 20d: choppy, mean-reverting.")
            elif ratio < 0.6:
                lines.append("  Intraday well below 20d: drifting, not thrashing.")
        return CommandResult.ok("\n".join(lines))

    async def _atm_implied(self, symbol: str) -> Optional[float]:
        expiries = await _expirations(self.providers, symbol)
        if not expiries:
            return None
        target = next(
            (e for e in expiries if _days_to(e) >= MIN_SKEW_DAYS), expiries[-1]
        )
        spot = await _spot(self.providers, symbol)
        if not spot:
            return None
        try:
            chain = await self.providers.get_options_chain(
                symbol, target, FULL_CHAIN,
            )
        except Exception:
            return None
        return _atm_iv(chain, spot)


def _fetch_bars(symbol: str):
    """(daily closes, 5-minute closes). Either may be empty."""
    import yfinance as yf

    daily: list[float] = []
    intraday: list[tuple] = []
    try:
        hist = yf.Ticker(symbol).history(period="6mo", interval="1d")
        daily = [float(c) for c in hist["Close"].tolist() if math.isfinite(float(c))]
    except Exception as e:
        logger.warning(f"rvol: daily history failed for {symbol}: {e}")
    try:
        bars = yf.Ticker(symbol).history(period="5d", interval="5m")
        intraday = [
            (idx.date(), float(close))
            for idx, close in zip(bars.index, bars["Close"].tolist())
            if math.isfinite(float(close))
        ]
    except Exception as e:
        logger.warning(f"rvol: intraday history failed for {symbol}: {e}")
    return daily, intraday


def _annualized(returns: list[float], periods_per_year: int) -> Optional[float]:
    """Sample stdev of log returns, annualized. None under 2 returns."""
    if len(returns) < 2:
        return None
    mean = sum(returns) / len(returns)
    var = sum((r - mean) ** 2 for r in returns) / (len(returns) - 1)
    return math.sqrt(var) * math.sqrt(periods_per_year)


def _close_to_close(closes: list[float], window: int) -> Optional[float]:
    tail = closes[-(window + 1):]
    returns = [
        math.log(b / a)
        for a, b in zip(tail, tail[1:])
        if a > 0 and b > 0
    ]
    return _annualized(returns, TRADING_DAYS)


def _intraday_rv(bars: list[tuple], sessions: int) -> Optional[float]:
    """Realized vol from 5-minute bars over the last `sessions` days.

    Returns are taken within a session only. Spanning the overnight gap
    would fold a jump into a 5-minute bucket and annualize it as if the
    market moved that much every five minutes.
    """
    if not bars:
        return None
    days = sorted({day for day, _ in bars})[-sessions:]
    returns: list[float] = []
    for day in days:
        closes = [c for d, c in bars if d == day]
        returns += [
            math.log(b / a)
            for a, b in zip(closes, closes[1:])
            if a > 0 and b > 0
        ]
    return _annualized(returns, TRADING_DAYS * BARS_PER_DAY)
