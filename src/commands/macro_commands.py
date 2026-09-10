"""
Credit and rates dashboards off FRED.

Provides: !credit, !pricedin

Both read raw FRED series ids through FredProvider.get_series. Every
series here was checked live; the ones that look like they belong but
aren't here are absent on purpose:

  * Swap spreads. FRED's swap-rate series (DSWP2/DSWP10) stopped
    updating in October 2016. There is no free replacement.
  * Cross-currency basis. Never published by FRED; the sources that
    carry it are all paid.
  * TED spread. DISCONTINUED in January 2022 when LIBOR was retired.

SOFR minus IORB and the bill-to-policy gap cover the same funding-stress
ground with series that are still alive, so that is what the funding
block shows.
"""

import asyncio
import logging
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager

logger = logging.getLogger(__name__)

# Trading days back for the change columns. FRED daily series skip
# weekends and holidays, so these are row offsets, not calendar days.
WEEK_ROWS = 5
MONTH_ROWS = 21

# Enough rows to reach MONTH_ROWS back even across a holiday-heavy month.
_FETCH_ROWS = 40


class _Row:
    """One series plus its lookbacks, in whatever unit FRED publishes."""

    def __init__(self, label: str, series_id: str, values: list[tuple[str, float]]):
        self.label = label
        self.series_id = series_id
        self.values = values

    @property
    def latest(self) -> Optional[float]:
        return self.values[0][1] if self.values else None

    @property
    def asof(self) -> str:
        return self.values[0][0] if self.values else "?"

    def _back(self, rows: int) -> Optional[float]:
        return self.values[rows][1] if len(self.values) > rows else None

    def change(self, rows: int) -> Optional[float]:
        prior, now = self._back(rows), self.latest
        if prior is None or now is None:
            return None
        return now - prior


async def _load(providers: ProviderManager, spec: dict[str, str]) -> dict[str, _Row]:
    """Fetch every series in `spec` concurrently. Failures are dropped.

    One dead series id must not blank the whole dashboard — FRED retires
    series without warning, which is exactly how the swap-spread rows
    that used to live here died.
    """
    fetch = next(
        (
            getattr(p, "get_series", None)
            for p in providers.providers
            if p.name == "fred" and hasattr(p, "get_series")
        ),
        None,
    )
    if fetch is None:
        return {}

    async def one(key: str, series_id: str):
        try:
            return key, _Row(key, series_id, await fetch(series_id, _FETCH_ROWS))
        except Exception as e:
            logger.warning(f"credit: series {series_id} failed: {e}")
            return key, None

    results = await asyncio.gather(*(one(k, v) for k, v in spec.items()))
    return {k: row for k, row in results if row is not None and row.values}


def _bp(value: Optional[float]) -> str:
    """Percentage points to basis points. FRED publishes OAS as 2.71."""
    if value is None:
        return "  n/a"
    return f"{value * 100:>4.0f}"


def _bp_delta(value: Optional[float]) -> str:
    if value is None:
        return "   ."
    return f"{value * 100:+4.0f}"


def _pct(value: Optional[float], places: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{places}f}"


class CreditCommand(BaseCommand):
    """Credit spreads, funding stress, and financial conditions."""

    name = "credit"
    aliases = ["spreads", "oas"]
    description = (
        "Live credit spreads (ICE BofA OAS) with 1-week and 1-month changes, "
        "plus funding stress and financial conditions. Use this instead of the "
        "VIX to judge whether risk is actually being repriced: HY OAS widening "
        "while equity vol is calm is the tell. All values in basis points "
        "unless marked. Data is FRED, so it is T-1 for OAS and rates."
    )
    usage = "!credit"
    help_explanation = """Credit market dashboard.

**Spreads (option-adjusted, in bp):**
- IG / BBB / HY / CCC / EM — compensation over Treasuries
- HY-IG — the compression gauge. Widening = risk being repriced

**Funding:**
- SOFR-IORB — repo pressure. Persistently positive = collateral scarcity
- 3M bill - EFFR — bill supply/demand vs policy
- RRP — cash parked at the Fed, in $bn

**Conditions:**
- NFCI / ANFCI — Chicago Fed indexes. Positive = tighter than average

**Not available:** swap spreads and cross-currency basis. FRED's swap
series died in 2016 and the basis was never free. Don't ask for them."""

    SPREADS = {
        "IG": "BAMLC0A0CM",
        "BBB": "BAMLC0A4CBBB",
        "HY": "BAMLH0A0HYM2",
        "CCC": "BAMLH0A3HYC",
        "EM": "BAMLEMCBPIOAS",
    }
    RATES = {
        "SOFR": "SOFR",
        "EFFR": "EFFR",
        "IORB": "IORB",
        "3M": "DGS3MO",
        "2Y": "DGS2",
        "10Y": "DGS10",
        "RRP": "RRPONTSYD",
    }
    CONDITIONS = {
        "NFCI": "NFCI",
        "ANFCI": "ANFCI",
    }

    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()

        spec = {**self.SPREADS, **self.RATES, **self.CONDITIONS}
        rows = await _load(self.providers, spec)
        if not rows:
            return CommandResult.error(
                "Credit data unavailable — FRED provider not configured or down."
            )

        lines = ["◈ CREDIT", ""]
        asof = next(
            (rows[k].asof for k in self.SPREADS if k in rows), None
        )
        lines.append(f"OAS in bp{f'  (as of {asof})' if asof else ''}")
        lines.append("            now    1w    1m")
        for key in self.SPREADS:
            row = rows.get(key)
            if row is None:
                continue
            lines.append(
                f"  {key:<5s} {_bp(row.latest)}  "
                f"{_bp_delta(row.change(WEEK_ROWS))}  "
                f"{_bp_delta(row.change(MONTH_ROWS))}"
            )

        hy, ig = rows.get("HY"), rows.get("IG")
        if hy and ig and hy.latest is not None and ig.latest is not None:
            diff_now = hy.latest - ig.latest
            diff_1w = _diff_change(hy, ig, WEEK_ROWS)
            diff_1m = _diff_change(hy, ig, MONTH_ROWS)
            lines.append(
                f"  HY-IG {_bp(diff_now)}  {_bp_delta(diff_1w)}  {_bp_delta(diff_1m)}"
            )

        funding = self._funding_lines(rows)
        if funding:
            lines += ["", "Funding"] + funding

        cond = []
        for key in self.CONDITIONS:
            row = rows.get(key)
            if row and row.latest is not None:
                stance = "tighter" if row.latest > 0 else "looser"
                cond.append(
                    f"  {key:<5s} {row.latest:+.2f} ({stance} than average)"
                )
        if cond:
            lines += ["", "Conditions"] + cond

        return CommandResult.ok("\n".join(lines))

    def _funding_lines(self, rows: dict[str, _Row]) -> list[str]:
        out: list[str] = []
        sofr, iorb, effr = rows.get("SOFR"), rows.get("IORB"), rows.get("EFFR")
        if sofr and iorb and sofr.latest is not None and iorb.latest is not None:
            gap = (sofr.latest - iorb.latest) * 100
            note = "collateral tight" if gap > 5 else "normal"
            out.append(f"  SOFR-IORB  {gap:+.0f}bp ({note})")
        if sofr and effr and sofr.latest is not None and effr.latest is not None:
            out.append(f"  SOFR-EFFR  {(sofr.latest - effr.latest) * 100:+.0f}bp")
        bill, effr_row = rows.get("3M"), rows.get("EFFR")
        if bill and effr_row and bill.latest is not None and effr_row.latest is not None:
            out.append(f"  3M bill-EFFR {(bill.latest - effr_row.latest) * 100:+.0f}bp")
        two, ten = rows.get("2Y"), rows.get("10Y")
        if two and ten and two.latest is not None and ten.latest is not None:
            out.append(
                f"  2s10s      {(ten.latest - two.latest) * 100:+.0f}bp "
                f"(2Y {_pct(two.latest)}%, 10Y {_pct(ten.latest)}%)"
            )
        rrp = rows.get("RRP")
        if rrp and rrp.latest is not None:
            # RRPONTSYD is already in $bn; sub-billion days round to
            # zero at 0dp, and "RRP $0bn" reads as a data failure.
            out.append(f"  RRP        ${rrp.latest:,.1f}bn")
        return out


def _diff_change(a: _Row, b: _Row, rows: int) -> Optional[float]:
    """Change in (a - b) over `rows` observations."""
    a_then = a.values[rows][1] if len(a.values) > rows else None
    b_then = b.values[rows][1] if len(b.values) > rows else None
    if a_then is None or b_then is None or a.latest is None or b.latest is None:
        return None
    return (a.latest - b.latest) - (a_then - b_then)


class PricedInCommand(BaseCommand):
    """What the market already expects, for judging a data surprise."""

    name = "pricedin"
    aliases = ["priced", "expectations"]
    description = (
        "Market-implied inflation and rate expectations with 1-week and "
        "1-month changes. Use this to judge whether an economic print was "
        "actually a surprise: compare the release against what breakevens "
        "and the front end had already moved to. This is NOT a "
        "consensus-surprise index — no free source publishes economist "
        "consensus, so there is no actual-vs-forecast number available. "
        "For event-level odds before a print, use !kalshi instead."
    )
    usage = "!pricedin"
    help_explanation = """What the market has already priced.

- 5Y / 10Y breakevens — TIPS-implied inflation compensation
- 5y5y forward — the long-run inflation anchor. Moves here matter more
  than moves in spot breakevens
- 2Y vs EFFR — how much cutting or hiking the front end expects
- Real 10Y — the actual cost of money

**Reading a print:** a hot CPI that breakevens already moved to is not a
surprise. A mild one that breakevens had not is.

**No consensus data.** Finnhub's economic calendar is a paid endpoint
and FRED publishes actuals only. Use !kalshi for pre-print market odds."""

    SERIES = {
        "5Y BE": "T5YIE",
        "10Y BE": "T10YIE",
        "5y5y": "T5YIFR",
        "10Y real": "DFII10",
        "2Y": "DGS2",
        "EFFR": "EFFR",
    }

    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()

        rows = await _load(self.providers, self.SERIES)
        if not rows:
            return CommandResult.error(
                "Expectations data unavailable — FRED provider not configured."
            )

        asof = next((r.asof for r in rows.values()), "?")
        lines = [f"◈ PRICED IN  (as of {asof})", "", "             now    1w    1m"]
        for key in self.SERIES:
            row = rows.get(key)
            if row is None or row.latest is None:
                continue
            lines.append(
                f"  {key:<8s} {row.latest:5.2f}%  "
                f"{_bp_delta(row.change(WEEK_ROWS))}  "
                f"{_bp_delta(row.change(MONTH_ROWS))}"
            )

        two, effr = rows.get("2Y"), rows.get("EFFR")
        if two and effr and two.latest is not None and effr.latest is not None:
            gap = (two.latest - effr.latest) * 100
            direction = "hikes" if gap > 0 else "cuts"
            lines += [
                "",
                f"  2Y-EFFR {gap:+.0f}bp — front end prices {direction} "
                f"({abs(gap) / 25:.1f}x 25bp over 2y)",
            ]
        lines += ["", "Changes in bp. No consensus feed: judge surprise against", "these levels, or !kalshi for pre-print odds."]
        return CommandResult.ok("\n".join(lines))
