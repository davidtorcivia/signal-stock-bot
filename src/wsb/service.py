"""
WSB digest orchestrator.

Ties the pieces together for one daily run:
  crawl Redlib -> compile digest -> deep-think analysis (with live price/news
  cross-check) -> persist -> render static page + index + og:image -> return a
  teaser + permalink for the worker to post to Signal.

Config (Redlib URL, subreddit, public base URL, indexability, user-agent) is
read LIVE from the settings store on every run so admin edits apply without a
restart; the static output dir comes from Config (an in-container path).

`run()` never raises — it returns None on any failure (empty crawl, deep-think
unavailable, render error) so the oracle worker treats it as "no post" and
leaves the day's slot unstamped for a retry within the grace window. The model
only writes prose; all numbers on the page come from the deterministic tally.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import json
import logging
import re
from dataclasses import dataclass
from typing import Optional

from ..executor import RENDER_TIMEOUT, run_blocking
from .digest import WSBDigest, compile_wsb_digest, render_digest_text
from .redlib import DEFAULT_BASE_URL, DEFAULT_SUBREDDIT, DEFAULT_USER_AGENT, RedlibSource
from .site import WSBSiteGenerator
from .store import WSBDigestRecord, WSBDigestStore

logger = logging.getLogger(__name__)

# Marks every deep-think non-answer (unavailable / rate-limited / no content).
_DT_FAILURE_PREFIX = "(deep_think "

_INDEX_LIMIT = 90
_CONTEXT_MAX_CHARS = 7000

# Price-spark charts on the page: how many of the top tickers to chart, and the
# historical window. 1-month daily closes give a clean sparkline.
_CHART_TICKERS = 6
_CHART_PERIOD = "1mo"
_CHART_INTERVAL = "1d"
_CHART_MIN_BARS = 4


@dataclass
class WSBPostPayload:
    date: str
    headline: str
    teaser: str
    page_url: str
    body_md: str
    tldr: str = ""          # brief humorous day-summary for the chat message
    chart_count: int = 0
    # Rendered OG card PNG bytes (same image the page uses for og:image). The
    # worker attaches it to the chat post — signal-cli-rest-api does not
    # auto-unfurl links, so without attaching it no image ever reaches the chat.
    og_png: Optional[bytes] = None


def _build_question(bot_name: str, subreddit: str, digest: WSBDigest) -> str:
    top = ", ".join(t.symbol for t in digest.tickers[:8]) or "(none extracted)"
    return (
        f"You are {bot_name}, writing the daily r/{subreddit} read for a finance "
        f"chat and a public page. The supporting context below is a compiled "
        f"digest of today's top posts, the discussion threads broken into top "
        f"parent comments with their replies, and a PROGRAMMATIC tally of the "
        f"most-mentioned tickers (exact mention counts and bull/bear sentiment "
        f"counts already computed from every comment). Treat those counts as "
        f"ground truth for what the crowd is fixated on.\n\n"
        f"The crowd's most-mentioned names today: {top}.\n\n"
        f"USE YOUR TOOLS aggressively. For the top 4-6 tickers, pull current "
        f"price and percent move, recent news, and where available technicals "
        f"(RSI, support/resistance) and notable options activity (unusual flow, "
        f"the strikes/expiries WSB is actually piling into). Weigh the crowd "
        f"against the tape: where they are chasing, where sentiment and price "
        f"diverge, what the options positioning implies.\n\n"
        f"Be dense and SELECTIVE, not exhaustive. Lead with the dominant story, "
        f"then cover only the two or three things that actually matter today "
        f"(including the buried lede the crowd is sleeping on), surface ONE "
        f"genuine deep gem worth the dig (an underfollowed name, a real DD post, a "
        f"contrarian setup, or smart-money positioning under the meme volume, and "
        f"quote the specific comment or post), and give a short options read where "
        f"the data supports it. Skip the ticker-by-ticker roll call. Depth means "
        f"insight per sentence, not word count. Keep the body TIGHT: roughly 450 "
        f"to 650 words total, a few short sections. Cut throat-clearing and "
        f"filler so every sentence earns its place.\n\n"
        f"Be genuinely, openly FUNNY, this is the entire point of the piece and "
        f"the bar is high. The voice is a dry, sardonic, WSB-fluent market "
        f"commentator who finds the daily parade of degenerates, dilution filings, "
        f"and 90-RSI melt-ups genuinely hilarious and cannot resist saying so. "
        f"Land a real laugh or a sharp barb in EVERY section, not only the "
        f"headline. Get the comedy from absurd-but-precise analogies, deadpan "
        f"understatement of insane numbers, mock-heroic or mock-tragic framing of "
        f"the crowd's worst decisions, vivid specific imagery, and the occasional "
        f"well-aimed cultural or finance reference that rewards a literate reader. "
        f"Treat the bagholders and YOLO degenerates as tragicomic characters whose "
        f"pain is the joke. Litmus test for every sentence: if it could run "
        f"verbatim in a Reuters market wrap, it has failed, so rewrite it with a "
        f"joke or a knife in it. The headline and the teaser are the marquee and "
        f"must be the two funniest, most quotable lines in the whole piece. Punch "
        f"at the absurdity and the crowd's behavior, be savage but earn it, never "
        f"reach for a cheap or lazy gag. In the BODY every joke has to live inside "
        f"a long, flowing sentence rather than a clipped one-liner (the headline "
        f"and teaser are the only place a short punchy line is allowed).\n\n"
        f"STYLE RULES (hard constraints, violating any one ruins the piece):\n"
        f"  1. NEVER use an em dash or en dash (the — or – characters). Use "
        f"commas, parentheses, or separate sentences. If you reach for a dash for "
        f"emphasis, rewrite the sentence instead.\n"
        f"  2. EVERY sentence must be a long, flowing sentence that carries real "
        f"reasoning. Short declarative sentences and fragments are BANNED "
        f"outright, with NO exceptions: not as punchlines, not as openers, not "
        f"for rhythm or emphasis. Do not write clipped lines like 'SPCE was the "
        f"story.', 'RSI sits at 92.1.', 'There is no gamma ramp here.', or 'The "
        f"pun is there.'; each must be folded into a longer sentence that does "
        f"actual work. Never stack two short sentences back to back ('RSI hit "
        f"90.5. Short interest sits at 22.6%.'; write 'RSI hit 90.5 while short "
        f"interest sits at 22.6%.' instead) and never spell a word out for "
        f"emphasis ('Ninety. Point. Five.'). If a sentence runs under roughly a "
        f"dozen words and stands on its own, join it to the surrounding "
        f"reasoning. The wit comes entirely from word choice and framing INSIDE "
        f"these long sentences.\n"
        f"  3. NEVER make a point by negating-then-correcting. This whole family "
        f"is forbidden in EVERY form, including across two sentences: 'That's not "
        f"X, that's Y', 'This is not X. It is Y.', 'X is not A, it is B', 'this "
        f"isn't A, it's B', 'not A but B', 'X? Hardly.' Do not state what "
        f"something is NOT in order to land what it is. State the point directly "
        f"and positively. (BAD: 'This is not a thesis. It is a flow event.' GOOD: "
        f"'This is a pure flow event with no underlying thesis.' BAD: 'The crowd "
        f"is not buying space exposure. It is buying a squeeze.' GOOD: 'The crowd "
        f"is buying a squeeze setup dressed up as space exposure.')\n"
        f"  4. Dry, skeptical, specific, substantive. Earn the cynicism with "
        f"numbers, do not posture.\n\n"
        f"Return STRICT JSON (no prose outside it, no code fence) with exactly "
        f"these keys:\n"
        f'  "headline": a funny, sharp title, <= 75 chars, no ticker spam, no '
        f"dashes. Make it land.\n"
        f'  "tldr": a brief 2 to 3 sentence TLDR of the WHOLE day for a chat '
        f"message, the funny gist of what the crowd piled into and what actually "
        f"moved (the top one or two names and your take), in the same voice as the "
        f"piece. This is posted to the chat right above the link, so it has to "
        f"land on its own. Plain text, no markdown, no dashes.\n"
        f'  "teaser": ONE punchy sentence used as the link-preview blurb (the hook '
        f"that makes someone click), distinct from the tldr. Plain text, no "
        f"markdown, no dashes.\n"
        f'  "body_markdown": the read in a few tight sections (450-650 words total) '
        f"under '## ' subheadings, with '- ' bullets and **bold** where useful. "
        f"Reference the tally numbers in prose; do NOT reprint the ticker table "
        f"(the page renders it)."
    )


# Em dash / en dash safety net. The prompt forbids them, but the model slips
# occasionally; replace with commas so the published voice is dash-free.
_DASH_RE = re.compile(r"\s*[—–―]+\s*")


def _strip_dashes(text: str) -> str:
    if not text:
        return text
    out = _DASH_RE.sub(", ", text)
    out = re.sub(r",\s*,", ",", out)   # collapse doubled commas
    out = re.sub(r"\s+,", ",", out)    # no space before a comma
    return out


def _parse_analysis(text: str, *, date: str) -> tuple[str, str, str, str]:
    """Extract (headline, teaser, body_md, tldr) from the model output. Tolerant:
    handles bare JSON, fenced JSON, or prose; never raises. Dashes stripped.
    tldr is the chat blurb (falls back to the teaser when the model omits it)."""
    obj = _extract_json(text)
    if obj:
        headline = str(obj.get("headline") or "").strip()
        teaser = str(obj.get("teaser") or "").strip()
        tldr = str(obj.get("tldr") or "").strip()
        body = str(obj.get("body_markdown") or obj.get("body") or "").strip()
        if body or teaser or tldr:
            if not headline:
                headline = f"WSB Daily, {date}"
            if not teaser:
                teaser = tldr or _first_sentences(body, 2)
            if not tldr:
                tldr = teaser
            return (_strip_dashes(headline[:120]), _strip_dashes(teaser),
                    _strip_dashes(body or teaser), _strip_dashes(tldr))
    # Fallback: treat the whole thing as the body.
    clean = text.strip()
    blurb = _strip_dashes(_first_sentences(clean, 2))
    return (f"WSB Daily, {date}", blurb, _strip_dashes(clean), blurb)


def _extract_json(text: str) -> Optional[dict]:
    if not text:
        return None
    # Strip a ```json ... ``` fence if present.
    fence = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, re.DOTALL)
    candidate = fence.group(1) if fence else None
    if candidate is None:
        start = text.find("{")
        end = text.rfind("}")
        candidate = text[start : end + 1] if 0 <= start < end else None
    if not candidate:
        return None
    try:
        obj = json.loads(candidate)
        return obj if isinstance(obj, dict) else None
    except (ValueError, TypeError):
        return None


def _first_sentences(text: str, n: int) -> str:
    text = " ".join((text or "").split())
    # strip markdown markers for a clean teaser
    text = re.sub(r"[#*_`>]+", "", text)
    parts = re.split(r"(?<=[.!?])\s+", text)
    return " ".join(parts[:n]).strip()[:400]


class WSBDigestService:
    """Runs one WSB digest cycle. Construct once; call run() per fire."""

    def __init__(
        self,
        *,
        settings_store,
        store: WSBDigestStore,
        static_dir: str,
        provider_manager=None,
    ):
        self.settings = settings_store
        self.store = store
        self.static_dir = static_dir
        # Optional: used to fetch historical closes for the per-ticker price
        # sparks on the page. None (or any failure) just skips the charts.
        self.provider_manager = provider_manager

    def _cfg(self) -> dict:
        s = self.settings
        return {
            "base_url": s.get_stripped("wsb_redlib_base_url", DEFAULT_BASE_URL),
            "subreddit": s.get_stripped("wsb_subreddit", DEFAULT_SUBREDDIT),
            "public_base_url": s.get_stripped("wsb_public_base_url", "https://sigil.disinfo.zone"),
            "indexable": s.get_bool("wsb_indexable", True),
            "user_agent": s.get_stripped("wsb_user_agent", DEFAULT_USER_AGENT),
        }

    async def _fetch_price_data(self, tickers: list) -> list[dict]:
        """Pull recent daily closes for the top tickers, concurrently. Returns
        [{symbol, lean, closes, price, change_percent}] for the names that
        resolved; missing/failed symbols are simply dropped. Never raises."""
        pm = self.provider_manager
        if pm is None:
            logger.info("WSB charts: no provider_manager wired; skipping price charts")
            return []
        if not tickers:
            return []
        top = tickers[:_CHART_TICKERS]

        async def one(t: dict) -> Optional[dict]:
            sym = str(t.get("symbol", "")).upper()
            if not sym:
                return None
            try:
                bars = await pm.get_historical(sym, period=_CHART_PERIOD,
                                               interval=_CHART_INTERVAL)
            except Exception as e:
                logger.debug("WSB chart: historical fetch failed for %s: %s", sym, e)
                return None
            closes = [b.close for b in (bars or []) if b.close is not None]
            if len(closes) < _CHART_MIN_BARS:
                return None
            pct = ((closes[-1] / closes[0]) - 1.0) * 100.0 if closes[0] else 0.0
            return {
                "symbol": sym, "lean": str(t.get("lean", "mixed")),
                "closes": closes, "price": closes[-1], "change_percent": pct,
            }

        results = await asyncio.gather(*(one(t) for t in top))
        ok = [r for r in results if r]
        if ok:
            logger.info("WSB charts: fetched price history for %d/%d top tickers (%s)",
                        len(ok), len(top), ", ".join(r["symbol"] for r in ok))
        else:
            logger.warning(
                "WSB charts: no price history for any of %s — is a "
                "historical-capable price provider configured?",
                [str(t.get("symbol")) for t in top],
            )
        return ok

    async def run(
        self,
        *,
        deep_think_client,
        caller_ctx=None,
        bot_name: str = "Sigil",
        now: Optional[dt.datetime] = None,
    ) -> Optional[WSBPostPayload]:
        try:
            return await self._run(deep_think_client, caller_ctx, bot_name, now)
        except Exception as e:  # never crash the worker
            logger.exception("WSB digest run failed: %s", e)
            return None

    async def _run(self, deep_think_client, caller_ctx, bot_name, now) -> Optional[WSBPostPayload]:
        cfg = self._cfg()
        now = now or dt.datetime.now(dt.timezone.utc)

        source = RedlibSource(
            base_url=cfg["base_url"],
            subreddit=cfg["subreddit"],
            user_agent=cfg["user_agent"],
        )
        try:
            digest = await compile_wsb_digest(source, now=now)
        finally:
            await source.close()

        if digest.is_empty:
            logger.warning("WSB digest: empty crawl from %s — skipping", cfg["base_url"])
            return None

        # Deep-think analysis with the live cross-check.
        if deep_think_client is None:
            logger.warning("WSB digest: no deep-think client — skipping")
            return None
        question = _build_question(bot_name, digest.subreddit, digest)
        context = render_digest_text(digest, max_chars=_CONTEXT_MAX_CHARS)
        # group_id=None so the scheduled call never trips a per-group deep-think cap.
        analysis = await deep_think_client.think(
            question=question,
            context=context,
            user_hash="wsb-digest",
            group_id=None,
            caller_ctx=caller_ctx,
        )
        if not analysis or analysis.strip().startswith(_DT_FAILURE_PREFIX):
            logger.warning("WSB digest: deep-think unavailable (%r) — skipping",
                           (analysis or "")[:80])
            return None

        headline, teaser, body_md, tldr = _parse_analysis(analysis, date=digest.date)

        rec = self._record_from(digest, headline, teaser, body_md, now)
        site = WSBSiteGenerator(
            static_dir=self.static_dir,
            public_base_url=cfg["public_base_url"],
            indexable=cfg["indexable"],
            bot_name=bot_name,
            subreddit=digest.subreddit,
        )
        rec.page_url = site.page_url_for(rec.date)

        # Fetch price history for the top tickers (event-loop side; the render
        # itself is blocking and runs in the executor below). rec.tickers is the
        # dict form (digest.tickers are TickerStat objects).
        price_data = await self._fetch_price_data(rec.tickers)

        # Render + publish the static read OFF the event loop — matplotlib
        # (og:image + price sparks) and the file writes are blocking.
        # Best-effort: a render failure must not lose the persisted record or
        # block the chat post.
        def _render_and_write_day():
            from ..charts.og_card import render_og_card
            from ..charts.wsb_chart import render_wsb_spark
            og_png = render_og_card(rec.date, headline, rec.tickers, teaser=teaser,
                                    bot_name=bot_name, subreddit=digest.subreddit)
            charts: dict[str, bytes] = {}
            chart_meta: list[dict] = []
            for d in price_data:
                png = render_wsb_spark(
                    d["symbol"], d["closes"], price=d["price"],
                    change_percent=d["change_percent"], lean=d["lean"],
                )
                if png:
                    charts[d["symbol"]] = png
                    chart_meta.append({
                        "symbol": d["symbol"],
                        "price": round(d["price"], 2),
                        "change_percent": round(d["change_percent"], 2),
                        "lean": d["lean"],
                    })
            rec.charts = chart_meta
            logger.info("WSB charts: rendered %d price spark image(s)", len(charts))
            site.write_day(rec, og_png=og_png or None, charts=charts)
            return og_png or None

        og_png: Optional[bytes] = None
        try:
            og_png = await run_blocking(_render_and_write_day, timeout=RENDER_TIMEOUT)
        except Exception as e:
            logger.warning("WSB digest: page render failed (continuing): %s", e)

        await self.store.upsert(rec)
        try:
            recent = await self.store.recent(_INDEX_LIMIT)
            await run_blocking(site.write_index, recent)
        except Exception as e:
            logger.warning("WSB digest: index render failed: %s", e)

        logger.info("WSB digest ready for %s: %s (%d charts)",
                    rec.date, rec.page_url, len(rec.charts))
        return WSBPostPayload(
            date=rec.date, headline=headline, teaser=teaser,
            page_url=rec.page_url, body_md=body_md, tldr=tldr,
            chart_count=len(rec.charts), og_png=og_png,
        )

    @staticmethod
    def _record_from(digest: WSBDigest, headline, teaser, body_md, now) -> WSBDigestRecord:
        tickers = [{
            "symbol": t.symbol, "mentions": t.mentions, "lean": t.lean,
            "bull": t.bull, "bear": t.bear, "cashtags": t.cashtags, "weight": t.weight,
        } for t in digest.tickers]
        posts = [{
            "title": p.title, "score": p.score, "comments": p.num_comments,
            "flair": p.flair, "permalink": p.permalink, "is_self": p.is_self,
        } for p in digest.top_posts[:12]]
        return WSBDigestRecord(
            date=digest.date, subreddit=digest.subreddit,
            headline=headline, teaser=teaser, body_md=body_md, page_url="",
            tickers=tickers, posts=posts,
            posts_scanned=digest.posts_scanned, comments_scanned=digest.comments_scanned,
            generated_at=now.timestamp(),
        )

    async def mark_posted(self, date: str, ok: bool) -> None:
        try:
            await self.store.mark_posted(date, ok)
        except Exception as e:
            logger.debug("WSB digest: mark_posted failed: %s", e)
