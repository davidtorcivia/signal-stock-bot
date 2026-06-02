"""
Ticker tally + WSB digest assembly.

The tricky part is separating real tickers from WSB slang (DD, YOLO, FD, CEO,
ATH ...). Strategy, no per-token network calls:

  1. Cashtags ($TSLA) are unambiguous -> always counted, and they SEED a
     "confirmed" set for this run.
  2. Bare uppercase tokens (TSLA) are counted only if they're confirmed —
     i.e. they appeared as a cashtag somewhere in the corpus OR are in the
     curated known-ticker allowlist — AND are not slang/stopwords, length >= 2.

This keeps the false-positive rate low while still catching popular tickers that
are typed without a `$`. Mentions are counted once per post/comment (so one
ranty post can't inflate a count), weighted by the source's score, with a rough
bull/bear lean from co-occurring sentiment words.

`compile_wsb_digest()` orchestrates the scraper fetches; everything else is pure
and unit-tested without a network.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import re
from dataclasses import dataclass, field
from typing import Optional

from ..utils.stopwords import STOPWORDS
from ..utils.symbols import SYMBOL_ALIASES, is_valid_symbol_format
from .redlib import RedditPost, RedlibSource

logger = logging.getLogger(__name__)

# WSB-specific slang / acronyms that look like tickers but aren't. Kept separate
# from STOPWORDS (which is general English). Uppercased for membership tests.
WSB_SLANG = {
    "DD", "YOLO", "FD", "FDS", "FOMO", "HODL", "BTFD", "GUH", "TENDIES",
    "CEO", "CFO", "CTO", "COO", "IPO", "ETF", "ETFS", "SEC", "FED", "FOMC",
    "CPI", "PPI", "GDP", "ATH", "ATL", "EOD", "EOW", "EOY", "AH", "PM",
    "IMO", "IMHO", "TLDR", "USA", "US", "UK", "EU", "UN", "USD",
    "AI", "ML", "EV", "PE", "EPS", "ER", "IV", "ITM", "OTM", "OTC",
    "NFA", "WSB", "OP", "ELI", "RIP", "LOL", "LMAO", "WTF", "OMG", "NGL",
    "TBH", "IDK", "SMH", "FYI", "AKA", "ASAP", "DCA", "ROI", "YTD", "YOY",
    "QOQ", "MACD", "RSI", "EMA", "SMA", "ATM", "IRA", "ROTH", "HSA",
    "CALL", "CALLS", "PUT", "PUTS", "BUY", "SELL", "HOLD", "LONG", "SHORT",
    "BULL", "BEAR", "MOON", "PT", "SL", "TP", "GG", "JPOW", "QE", "QT",
    "AF", "AND", "OR", "FOR", "THE", "NOW", "NEW", "ALL", "ANY", "BIG",
    "RED", "WAY", "WSP", "GAIN", "LOSS", "DAY", "WEEK", "YEAR", "ELI5",
}

# Curated known tickers so popular names typed without a `$` still count, on top
# of the cashtag-seeded set. SYMBOL_ALIASES values give us ~100 common names.
_WSB_COMMON = {
    "SPY", "QQQ", "IWM", "VOO", "VTI", "DIA", "VIX", "UVXY", "SQQQ", "TQQQ",
    "SOXL", "SOXS", "TLT", "GLD", "SLV", "USO", "GME", "AMC", "BBBY", "BB",
    "NVDA", "TSLA", "AAPL", "MSFT", "AMZN", "META", "GOOG", "GOOGL", "AMD",
    "INTC", "PLTR", "HOOD", "COIN", "MSTR", "SMCI", "ARM", "AVGO", "MU",
    "NFLX", "DIS", "BABA", "NIO", "F", "T", "BAC", "JPM", "PYPL", "SOFI",
    "RIVN", "LCID", "RIOT", "MARA", "DKNG", "ABNB", "UBER", "SNAP", "PFE",
    "BA", "GE", "WMT", "COST", "XOM", "CVX", "OXY", "DJT", "SPCE", "LUNR",
}
_KNOWN_TICKERS = {
    s.upper() for s in SYMBOL_ALIASES.values() if s and s.replace("-", "").isalpha()
} | _WSB_COMMON

_CASHTAG_RE = re.compile(r"\$([A-Za-z]{1,5}(?:[-.][A-Za-z]{1,5})?)\b")
_BARE_RE = re.compile(r"\b[A-Z]{2,5}\b")

_BULL_RE = re.compile(
    r"\b(?:calls?|long|buy(?:ing)?|bull(?:ish)?|moon\w*|rocket|squeez\w*|breakout|"
    r"pump\w*|ripp?ing|print\w*|green|tendies|all[\s-]?in|loaded|yolo\w*|lfg|"
    r"send(?:ing)?|gains?|up\b)\b|🚀|💎|📈|🌙|🦍",
    re.IGNORECASE,
)
_BEAR_RE = re.compile(
    r"\b(?:puts?|short(?:ing)?|bear(?:ish)?|drill\w*|crash|dump\w*|tank\w*|"
    r"sell(?:ing)?|red|baghold\w*|dead|rug\w*|fuk\w*|rekt|overbought|fade|guh|"
    r"roll(?:ing)?)\b|📉|💀|🩸",
    re.IGNORECASE,
)


def _sentiment(text: str) -> int:
    """+1 net-bullish, -1 net-bearish, 0 neutral, for one chunk of text."""
    return (1 if len(_BULL_RE.findall(text)) > len(_BEAR_RE.findall(text))
            else -1 if len(_BEAR_RE.findall(text)) > len(_BULL_RE.findall(text))
            else 0)


@dataclass
class TickerStat:
    symbol: str
    mentions: int = 0
    cashtags: int = 0
    weight: int = 0          # sum of source scores (capped per source)
    bull: int = 0
    bear: int = 0
    samples: list[str] = field(default_factory=list)

    @property
    def lean(self) -> str:
        # Loosened: a clear one-sided signal reads directional even when light;
        # only call it "mixed" when both sides are present and close.
        if self.bull and self.bear == 0:
            return "bullish"
        if self.bear and self.bull == 0:
            return "bearish"
        if self.bull >= 2 and self.bull > self.bear:
            return "bullish"
        if self.bear >= 2 and self.bear > self.bull:
            return "bearish"
        return "mixed"


@dataclass
class WSBDigest:
    date: str                      # YYYY-MM-DD (UTC)
    subreddit: str
    generated_at: str              # ISO8601 UTC
    tickers: list[TickerStat] = field(default_factory=list)
    top_posts: list[RedditPost] = field(default_factory=list)
    megathreads: list[RedditPost] = field(default_factory=list)
    # Every post we pulled comments from (megathreads + the day's biggest
    # discussion posts), each with .comments populated. The tally scans all of
    # these; the rendered context shows the top replies from a few.
    comment_threads: list[RedditPost] = field(default_factory=list)
    posts_scanned: int = 0
    comments_scanned: int = 0
    source_label: str = ""

    @property
    def is_empty(self) -> bool:
        return not self.top_posts and not self.comment_threads and not self.tickers


_SCORE_CAP = 10_000   # one mega-post can't dominate the weight ranking


def build_tally(
    sources: list[tuple[str, int, Optional[str]]],
    *,
    top_n: int = 12,
) -> list[TickerStat]:
    """sources: list of (text, score, sample_title_or_None). Counts each ticker
    once per source. Returns ranked TickerStats (mentions desc, weight desc)."""
    # Pass 1: seed confirmed set from cashtags across the whole corpus.
    confirmed = set(_KNOWN_TICKERS)
    for text, _score, _ in sources:
        for m in _CASHTAG_RE.finditer(text or ""):
            confirmed.add(m.group(1).upper().replace(".", "-"))

    stats: dict[str, TickerStat] = {}
    for text, score, sample in sources:
        if not text:
            continue
        present: set[str] = set()
        present_cashtag: set[str] = set()
        for m in _CASHTAG_RE.finditer(text):
            sym = m.group(1).upper().replace(".", "-")
            if sym in WSB_SLANG:
                continue
            present.add(sym)
            present_cashtag.add(sym)
        for m in _BARE_RE.finditer(text):
            sym = m.group(0).upper()
            if (sym in confirmed and sym not in WSB_SLANG
                    and sym.lower() not in STOPWORDS
                    and is_valid_symbol_format(sym)):
                present.add(sym)
        if not present:
            continue
        lean = _sentiment(text)
        w = min(max(score, 0), _SCORE_CAP)
        for sym in present:
            st = stats.get(sym)
            if st is None:
                st = stats[sym] = TickerStat(symbol=sym)
            st.mentions += 1
            st.weight += w
            if sym in present_cashtag:
                st.cashtags += 1
            if lean > 0:
                st.bull += 1
            elif lean < 0:
                st.bear += 1
            if sample and len(st.samples) < 3 and sample not in st.samples:
                st.samples.append(sample)

    ranked = sorted(stats.values(), key=lambda s: (s.mentions, s.weight), reverse=True)
    return ranked[:top_n]


async def compile_wsb_digest(
    source: RedlibSource,
    *,
    top_n: int = 14,
    max_posts: int = 25,
    max_megathreads: int = 2,
    max_discussion_posts: int = 10,
    min_post_comments: int = 40,
    max_parent_threads: int = 40,
    now: Optional[dt.datetime] = None,
) -> WSBDigest:
    """Crawl the subreddit and assemble a structured digest. Scrapes the
    discussion megathread(s) deep AND the day's biggest non-megathread posts,
    in parallel. The ticker tally counts EVERY comment at any depth (fixes the
    undercount from only sampling one thread's top-N), while the rendered
    context keeps the top parent threads with their replies. Never raises;
    returns a possibly-empty digest if the instance is unreachable."""
    now = now or dt.datetime.now(dt.timezone.utc)
    digest = WSBDigest(
        date=now.date().isoformat(),
        subreddit=source.subreddit,
        generated_at=now.isoformat(),
        source_label=source.base_url,
    )

    top = (await source.top_posts("day"))[:max_posts]
    hot = await source.hot_posts()
    megathreads = (await source.find_megathreads(hot))[:max_megathreads]
    mega_ids = {m.id for m in megathreads}
    discussion = sorted(
        [p for p in top if p.id not in mega_ids and p.num_comments >= min_post_comments],
        key=lambda p: p.num_comments, reverse=True,
    )[:max_discussion_posts]
    targets = list(megathreads) + discussion

    flats: list[list] = []
    results: list = []
    if targets:
        results = await asyncio.gather(
            *[source.fetch_threads(p, max_parents=max_parent_threads) for p in targets],
            return_exceptions=True,
        )
        for p, res in zip(targets, results):
            if isinstance(res, tuple):
                flat, threads = res
                p.comments = threads          # threaded top parents for context
                flats.append(flat)
            else:
                logger.warning("WSB: comment fetch failed for %s: %s", p.id, res)
                p.comments = []
                flats.append([])

    digest.top_posts = top
    digest.megathreads = megathreads
    digest.comment_threads = targets
    digest.posts_scanned = len(top)

    sources: list[tuple[str, int, Optional[str]]] = []
    for p in top:
        sources.append((f"{p.flair} {p.title}\n{p.body}".strip(), p.score, p.title))
    comment_count = 0
    for flat in flats:
        for c in flat:
            sources.append((c.body, c.score, None))
            comment_count += 1
    digest.comments_scanned = comment_count

    failed = sum(1 for res in results if not isinstance(res, tuple))
    logger.info(
        "WSB crawl: %d posts, %d comments from %d/%d threads (%d fetch failure(s))",
        len(top), comment_count, len(targets) - failed, len(targets), failed,
    )

    digest.tickers = build_tally(sources, top_n=top_n)
    return digest


def _trim(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    return text if len(text) <= n else text[: n - 1].rstrip() + "…"


def render_digest_text(digest: WSBDigest, *, max_chars: int = 7000) -> str:
    """Render the digest to the plain-text block handed to deep-think as
    context. Ordered most-signal-first and hard-capped to a budget."""
    lines: list[str] = []
    lines.append(
        f"r/{digest.subreddit} daily digest for {digest.date} "
        f"({digest.posts_scanned} top posts, {digest.comments_scanned} comments scanned)"
    )

    if digest.tickers:
        lines.append("\nMOST-MENTIONED TICKERS (mentions / lean / cashtags):")
        for t in digest.tickers:
            lines.append(
                f"  {t.symbol}: {t.mentions} mentions, {t.lean}"
                f" (bull {t.bull}/bear {t.bear}), {t.cashtags} cashtags,"
                f" upvote-weight {t.weight}"
            )

    if digest.top_posts:
        lines.append("\nTOP POSTS OF THE DAY:")
        for p in digest.top_posts[:15]:
            flair = f"[{p.flair}] " if p.flair else ""
            line = f"  • {flair}{_trim(p.title, 160)} ({p.score} upvotes, {p.num_comments} comments)"
            if p.is_self and p.body:
                line += f"\n      {_trim(p.body, 240)}"
            lines.append(line)

    threads_shown = 0
    for thread in digest.comment_threads:
        parents = thread.comments
        if not parents:
            continue
        if threads_shown >= 3:
            break
        flair = f"[{thread.flair}] " if thread.flair else ""
        lines.append(
            f"\nTHREAD: {flair}{_trim(thread.title, 100)} "
            f"({thread.num_comments} comments), top parent comments + replies:"
        )
        for c in parents[:6]:
            lines.append(f"  ↑{c.score} {_trim(c.body, 190)}")
            for r in c.replies[:2]:
                lines.append(f"     ↳ ↑{r.score} {_trim(r.body, 150)}")
        threads_shown += 1

    text = "\n".join(lines)
    if len(text) > max_chars:
        text = text[: max_chars - 20].rstrip() + "\n…[truncated]"
    return text
