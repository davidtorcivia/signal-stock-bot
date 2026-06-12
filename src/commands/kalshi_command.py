"""
Kalshi prediction-market lookup.

Read-only market data from Kalshi's public Trade API v2 — market-data
endpoints (markets/events/series) take no authentication, so there is no
key to configure. Trading endpoints are deliberately out of scope: the
bots quote odds, they don't hold positions.

The API has no free-text search endpoint, so search runs against two
locally cached layers:

  * The series catalog (GET /series) — every recurring market template
    ("Bitcoin price today", "Fed decision") in one response. This is the
    only place high-frequency price ladders are findable: their events
    are multivariate and GET /events silently excludes them, and the
    flat /markets sweep is 40k+ rows of auto-generated sports parlays.
    Matched series are resolved to their nearest-closing open events
    live, so quotes are always fresh.
  * The open-events index (GET /events?status=open) — one-off events
    (politics, novelty) that don't belong to a searchable series title.

Like every dispatcher command, this is auto-exposed to the LLMs as the
`bot__kalshi` tool, so the personas can pull live odds into answers.
"""

import asyncio
import logging
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..providers.base import SharedSession

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://external-api.kalshi.com/trade-api/v2"

_PAGE_SIZE = 200
_INDEX_PAGES = 8          # open events run ~800 today; headroom, not a cap
_INDEX_TTL_SECONDS = 300  # odds drift fast; the event *set* doesn't
_SERIES_TTL_SECONDS = 6 * 3600  # templates change rarely; payload is ~MBs

# How many query-matched series get resolved to live events, and how many
# nearest-closing events each contributes. Bounds the per-query fan-out.
_SERIES_FANOUT = 4
_EVENTS_PER_SERIES = 3

# Looks like a Kalshi market/event ticker (KXBTCD-25JUN13-T105000,
# INXD-26JUN12): all-caps with at least one digit, no spaces. Plain
# words like "FED" stay in search-land — a ticker miss falls back to
# search anyway, this just picks which path to try first.
_TICKER_RE = re.compile(r"^[A-Z][A-Z0-9._-]*\d[A-Z0-9._-]*$")

_FOOTER = "% = market-implied odds of YES (= the YES price in ¢)"


def _price_cents(market: dict) -> Optional[int]:
    """Best-effort YES price in cents (== implied probability %).

    Prefers last trade, falls back to bid/ask midpoint. The API ships
    both `*_dollars` fixed-point strings and legacy integer-cent
    fields depending on endpoint vintage; accept either.
    """
    def _cents(dollar_key: str, cent_key: str) -> Optional[int]:
        v = market.get(dollar_key)
        if v not in (None, ""):
            try:
                return round(float(v) * 100)
            except (TypeError, ValueError):
                pass
        v = market.get(cent_key)
        if v not in (None, ""):
            try:
                return int(v)
            except (TypeError, ValueError):
                pass
        return None

    last = _cents("last_price_dollars", "last_price")
    if last:  # 0 means "never traded" — fall through to the book
        return last
    bid = _cents("yes_bid_dollars", "yes_bid")
    ask = _cents("yes_ask_dollars", "yes_ask")
    if bid is not None and ask is not None and (bid or ask):
        return round((bid + ask) / 2)
    return last if last is not None else (bid or ask)


def _volume_24h(market: dict) -> int:
    for key in ("volume_24h_fp", "volume_24h", "volume_fp", "volume"):
        v = market.get(key)
        if v not in (None, ""):
            try:
                return int(float(v))
            except (TypeError, ValueError):
                continue
    return 0


def _event_volume(ev: dict) -> int:
    return sum(_volume_24h(m) for m in ev.get("markets") or [])


def _fmt_volume(n: int) -> str:
    if n >= 1_000_000:
        return f"{n / 1_000_000:.1f}M"
    if n >= 1_000:
        return f"{n / 1_000:.1f}k"
    return str(n)


def _close_dt(market_or_event: dict) -> Optional[datetime]:
    raw = market_or_event.get("close_time") or ""
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None


def _fmt_close(market: dict) -> str:
    dt = _close_dt(market)
    if dt is None:
        return ""
    days = (dt - datetime.now(timezone.utc)).days
    stamp = dt.strftime("%b %d") if days < 330 else dt.strftime("%b %d %Y")
    return f"closes {stamp}"


def _market_line(market: dict, label: Optional[str] = None) -> str:
    """One market as `  • label: 63% (vol 12k, closes Jun 30)`.

    The YES price in cents IS the implied probability in percent, and
    percent is the number people (and the tool-calling LLM) actually
    want — so that's what leads. The footer explains the equivalence.
    """
    name = label or market.get("yes_sub_title") or market.get("ticker", "?")
    cents = _price_cents(market)
    price = f"{cents}%" if cents is not None else "no quotes"
    extras = []
    vol = _volume_24h(market)
    if vol:
        extras.append(f"vol {_fmt_volume(vol)}")
    close = _fmt_close(market)
    if close:
        extras.append(close)
    suffix = f" ({', '.join(extras)})" if extras else ""
    return f"  • {name}: {price}{suffix}"


def _tokens(query: str) -> list[str]:
    return [t for t in query.lower().split() if t]


def _hits(tokens: list[str], hay: str) -> int:
    return sum(1 for t in tokens if t in hay)


def _event_haystack(ev: dict) -> str:
    return " ".join(
        str(part).lower()
        for part in (
            ev.get("title"), ev.get("sub_title"), ev.get("event_ticker"),
            *(m.get("yes_sub_title") for m in ev.get("markets") or []),
        )
        if part
    )


def _series_haystack(series: dict) -> str:
    return " ".join(
        str(part).lower()
        for part in (
            series.get("title"), series.get("ticker"),
            series.get("category"), *(series.get("tags") or []),
        )
        if part
    )


def _match_series(catalog: list[dict], query: str, limit: int) -> list[dict]:
    """Rank series templates against a free-text query.

    Only full-token matches qualify: the catalog is ~11k entries, so a
    one-token-of-three partial hit ("cut" in "Haircut futures") is far
    more likely noise than signal. The events index handles fuzzier
    coverage; this layer is for "the user named a thing Kalshi tracks".
    """
    tokens = _tokens(query)
    if not tokens:
        return []
    out = [s for s in catalog if _hits(tokens, _series_haystack(s)) == len(tokens)]
    # Shorter titles are tighter matches ("Bitcoin price today" over
    # "Will a bitcoin ETF holder buy a soccer team?").
    out.sort(key=lambda s: len(str(s.get("title") or "")))
    return out[:limit]


def _match_events(events: list[dict], query: str, limit: int) -> list[dict]:
    """Rank cached events against a free-text query.

    All-token matches outrank partial matches; 24h volume breaks ties so
    the liquid market for a topic beats its dead duplicates. Tokens match
    as substrings (so "shutdown" hits "Government Shutdown in 2026?").
    """
    tokens = _tokens(query)
    if not tokens:
        return []
    scored: list[tuple[int, int, dict]] = []
    for ev in events:
        hits = _hits(tokens, _event_haystack(ev))
        if hits == 0:
            continue
        scored.append((hits, _event_volume(ev), ev))
    scored.sort(key=lambda s: (s[0], s[1]), reverse=True)
    full = [s for s in scored if s[0] == len(tokens)]
    return [s[2] for s in (full or scored)[:limit]]


def _format_event(ev: dict, max_markets: int = 3) -> list[str]:
    """Compact event rendering for search-result lists.

    Truncates to the most-traded outcomes, but the truncation note names
    the event ticker so a human (or the LLM on its next tool call) can
    pull the full board with `!kalshi <EVENT_TICKER>`.
    """
    lines = [f"◆ {ev.get('title') or ev.get('event_ticker', '?')}"]
    markets = sorted(
        ev.get("markets") or [], key=_volume_24h, reverse=True,
    )
    # Single-market events repeat the event title in the market row;
    # show the bare ticker there instead of saying the same thing twice.
    if len(markets) == 1:
        lines.append(_market_line(markets[0], label=markets[0].get("ticker", "?")))
    else:
        for m in markets[:max_markets]:
            lines.append(_market_line(m))
        if len(markets) > max_markets:
            ticker = ev.get("event_ticker", "?")
            lines.append(
                f"  … +{len(markets) - max_markets} more — "
                f"!kalshi {ticker} for all outcomes"
            )
    return lines


def _format_event_full(ev: dict, max_markets: int = 40) -> list[str]:
    """Full event rendering for explicit ticker lookups.

    Shows every outcome that has a live opinion. Price ladders carry
    dozens of strikes pinned at ≤1% or ≥99% — those are noise for
    "what are the odds", so they're folded into one summary line each
    rather than burning the message budget. Markets stay in API order
    (strike order for ladders), not volume order: a ladder read out of
    order is unreadable.
    """
    lines = [f"◆ {ev.get('title') or ev.get('event_ticker', '?')}"]
    markets = ev.get("markets") or []
    if len(markets) == 1:
        lines.append(_market_line(markets[0], label=markets[0].get("ticker", "?")))
        return lines

    interesting, floor, ceiling = [], 0, 0
    for m in markets:
        cents = _price_cents(m)
        if cents is not None and cents <= 1:
            floor += 1
        elif cents is not None and cents >= 99:
            ceiling += 1
        else:
            interesting.append(m)

    for m in interesting[:max_markets]:
        lines.append(_market_line(m))
    if len(interesting) > max_markets:
        lines.append(f"  … +{len(interesting) - max_markets} more outcomes")
    if ceiling:
        lines.append(f"  • ({ceiling} outcome(s) at ≥99% — near-certain YES)")
    if floor:
        lines.append(f"  • ({floor} outcome(s) at ≤1% — near-certain NO)")
    return lines


class KalshiCommand(BaseCommand):
    """Search Kalshi prediction markets or look up a ticker."""
    name = "kalshi"
    aliases = ["odds"]
    description = (
        "Kalshi prediction-market odds. Search live markets by topic "
        "(e.g. \"fed rate cut\", \"bitcoin price\", \"government shutdown\") "
        "or look up a specific market/event ticker for the full outcome "
        "board. Numbers shown are market-implied probabilities in percent."
    )
    usage = "!kalshi fed decision  |  !kalshi bitcoin price  |  !kalshi KXBTCD-25JUN13  |  !kalshi shutdown -n 8"
    help_explanation = """Live odds from Kalshi, the regulated prediction market.

**Two ways to ask:**
• Topic search: !kalshi fed decision — finds open markets matching the words.
• Ticker lookup: !kalshi KXFEDDECISION-26JUN — full outcome board for an event or market ticker.

**Reading the numbers:**
• 63% means the market prices a 63% chance the event happens.
• That's also the YES price: pay 63¢, collect $1 if it resolves yes.
• vol is 24h contracts traded — low volume means stale/noisy odds.

**Flags:**
• -n N: show up to N matching events (default 5, max 10)."""

    def __init__(self, base_url: Optional[str] = None):
        self.base_url = (
            base_url or os.getenv("KALSHI_BASE_URL") or DEFAULT_BASE_URL
        ).rstrip("/")
        self._events: list[dict] = []
        self._events_at: float = 0.0
        self._series: list[dict] = []
        self._series_at: float = 0.0
        # One lock per cache: serializes refreshes so a burst of queries
        # doesn't stampede the sweep; losers reuse the winner's result.
        self._events_lock = asyncio.Lock()
        self._series_lock = asyncio.Lock()

    # ---------------------------------------------------------------- HTTP

    async def _get(self, path: str, params: Optional[dict] = None) -> Optional[dict]:
        """GET a Kalshi endpoint; None on 404 (caller decides fallback).

        Unauthenticated callers share a modest per-IP rate budget, and a
        cold search fans out a dozen requests — one polite retry on 429
        usually rides out the throttle instead of dropping results.
        """
        session = SharedSession.get()
        url = f"{self.base_url}{path}"
        for attempt in (0, 1):
            async with session.get(url, params=params or {}) as resp:
                if resp.status == 404:
                    return None
                if resp.status == 429 and attempt == 0:
                    try:
                        delay = float(resp.headers.get("Retry-After", "1"))
                    except ValueError:
                        delay = 1.0
                    await asyncio.sleep(min(delay, 3.0))
                    continue
                resp.raise_for_status()
                return await resp.json()
        return None  # unreachable; keeps the type-checker happy

    async def _event_index(self) -> list[dict]:
        """Open one-off events with nested markets, TTL-cached."""
        async with self._events_lock:
            if self._events and time.monotonic() - self._events_at < _INDEX_TTL_SECONDS:
                return self._events
            events: list[dict] = []
            cursor = None
            for _ in range(_INDEX_PAGES):
                params = {
                    "status": "open",
                    "limit": str(_PAGE_SIZE),
                    "with_nested_markets": "true",
                }
                if cursor:
                    params["cursor"] = cursor
                data = await self._get("/events", params) or {}
                events.extend(data.get("events") or [])
                cursor = data.get("cursor")
                if not cursor:
                    break
            if events:
                self._events = events
                self._events_at = time.monotonic()
            return self._events

    async def _series_catalog(self) -> list[dict]:
        """All series templates, trimmed to searchable fields, TTL-cached.

        The full payload carries contract legalese per series (~MBs for
        ~11k rows); keep only what matching and display need.
        """
        async with self._series_lock:
            if self._series and time.monotonic() - self._series_at < _SERIES_TTL_SECONDS:
                return self._series
            data = await self._get("/series", {"limit": "500"}) or {}
            trimmed = [
                {
                    "ticker": s.get("ticker"),
                    "title": s.get("title"),
                    "category": s.get("category"),
                    "tags": s.get("tags") or [],
                }
                for s in data.get("series") or []
                if s.get("ticker")
            ]
            if trimmed:
                self._series = trimmed
                self._series_at = time.monotonic()
            return self._series

    async def _events_for_series(self, series_ticker: str) -> list[dict]:
        """Nearest-closing open events of one series, quotes included."""
        try:
            data = await self._get("/events", {
                "series_ticker": series_ticker,
                "status": "open",
                "limit": "50",
                "with_nested_markets": "true",
            }) or {}
        except Exception as e:
            # One bad series shouldn't sink a multi-series search.
            logger.warning(f"Kalshi series fetch failed for {series_ticker}: {e}")
            return []
        events = data.get("events") or []
        far_future = datetime.max.replace(tzinfo=timezone.utc)
        events.sort(key=lambda e: min(
            (_close_dt(m) or far_future for m in e.get("markets") or []),
            default=far_future,
        ))
        return events[:_EVENTS_PER_SERIES]

    # ------------------------------------------------------------- search

    async def _search(self, query: str, count: int) -> tuple[list[dict], bool]:
        """Series-catalog matches (resolved live) + event-index matches,
        deduped, series hits first — a named series ("bitcoin price") is
        a stronger signal than a substring hit in a one-off title.

        Returns (events, exact): exact is False when nothing matched all
        the query tokens and the results are best-effort partials — the
        caller labels those, because silently answering "government
        shutdown" with an Amazon-antitrust market misleads both humans
        and the tool-calling LLM."""
        catalog, index = await asyncio.gather(
            self._series_catalog(), self._event_index(),
        )
        series = _match_series(catalog, query, _SERIES_FANOUT)
        # Sequential on purpose: four concurrent hits on top of a cold
        # index sweep is exactly the shape Kalshi's per-IP throttle
        # punishes, and these are ~150ms each.
        per_series = [
            await self._events_for_series(s["ticker"]) for s in series
        ]

        merged: list[dict] = []
        seen: set[str] = set()
        for events in per_series:
            for ev in events:
                key = ev.get("event_ticker") or ""
                if key not in seen:
                    seen.add(key)
                    merged.append(ev)
        tokens = _tokens(query)
        index_full_match = False
        for ev in _match_events(index, query, count):
            if _hits(tokens, _event_haystack(ev)) == len(tokens):
                index_full_match = True
            key = ev.get("event_ticker") or ""
            if key not in seen:
                seen.add(key)
                merged.append(ev)
        exact = any(bool(evs) for evs in per_series) or index_full_match
        return merged[:count], exact

    # ------------------------------------------------------------- lookups

    async def _lookup_ticker(self, ticker: str) -> Optional[str]:
        """Resolve a market ticker, falling back to event ticker."""
        data = await self._get("/markets", {"tickers": ticker})
        markets = (data or {}).get("markets") or []
        if markets:
            m = markets[0]
            lines = [f"◈ Kalshi {m.get('ticker', ticker)}", ""]
            if m.get("title"):
                lines.append(m["title"])
            lines.append(_market_line(m, label=m.get("yes_sub_title") or "Market"))
            rules = (m.get("rules_primary") or "").strip()
            if rules:
                lines.append(f"Resolves: {rules[:300]}")
            lines.append("")
            lines.append(_FOOTER)
            return "\n".join(lines)

        data = await self._get(
            f"/events/{ticker}", {"with_nested_markets": "true"},
        )
        event = (data or {}).get("event")
        if event:
            return "\n".join(
                [f"◈ Kalshi {ticker}", ""]
                + _format_event_full(event)
                + ["", _FOOTER]
            )
        return None

    # ------------------------------------------------------------- execute

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if self.has_help_flag(ctx):
            return self.get_help_result()

        count = 5
        terms: list[str] = []
        args = iter(ctx.args)
        for arg in args:
            if arg.lower() in ("-n", "--count"):
                nxt = next(args, "")
                if nxt.isdigit():
                    count = max(1, min(10, int(nxt)))
                continue
            if arg.isdigit() and terms:
                count = max(1, min(10, int(arg)))
                continue
            terms.append(arg)

        if not terms:
            return CommandResult.error(
                f"What market? Try: {self.usage}"
            )
        query = " ".join(terms)

        try:
            if len(terms) == 1 and _TICKER_RE.match(terms[0].upper()):
                detail = await self._lookup_ticker(terms[0].upper())
                if detail:
                    return CommandResult.ok(detail)
                # Unknown ticker → maybe it was a search word like "CPI25".

            matches, exact = await self._search(query, count)
        except asyncio.TimeoutError:
            return CommandResult.error("Kalshi API timed out")
        except Exception as e:
            logger.warning(f"Kalshi lookup failed for {query!r}: {e}")
            return CommandResult.error(f"Kalshi lookup failed: {e}")

        if not matches:
            return CommandResult.error(
                f"No open Kalshi markets matching '{query}'"
            )

        lines = [f"◈ Kalshi: {query}", ""]
        if not exact:
            lines[0] += " — no exact match, closest open markets:"
        for ev in matches:
            lines.extend(_format_event(ev))
            lines.append("")
        lines.append(_FOOTER)
        return CommandResult.ok("\n".join(lines).strip())
