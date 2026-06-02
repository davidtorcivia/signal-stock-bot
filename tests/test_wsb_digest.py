"""
Tests for the WSB digest data layer (scraper parsing + ticker tally + render).

All tests are network-free: parser tests feed saved Redlib HTML fixtures, tally
tests feed synthetic source lists, and the compile test uses a fake source. No
@pytest.mark.integration here — these run in the default suite.
"""

import asyncio
import datetime as dt
import pathlib

import pytest
from bs4 import BeautifulSoup

from src.wsb import redlib as redlib_mod
from src.wsb.digest import (
    WSBDigest,
    build_tally,
    compile_wsb_digest,
    render_digest_text,
    _sentiment,
)
from src.wsb.redlib import RedditComment, RedditPost, RedlibSource, _parse_score

FIXTURES = pathlib.Path(__file__).parent / "fixtures" / "wsb"


def _fixture(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


@pytest.fixture
def source() -> RedlibSource:
    return RedlibSource(base_url="https://redlib.test", subreddit="wallstreetbets")


# ---- scraper parsing --------------------------------------------------------

def test_parse_listing_extracts_posts(source):
    posts = source._parse_listing(_fixture("listing_top_day.html"))
    assert len(posts) == 4
    first = posts[0]
    # Exact score must come from the `title` attr, not the abbreviated "8.7k".
    assert first.score == 8707
    assert first.id == "1ttthhw"
    assert first.flair == "News"
    assert first.num_comments == 562
    assert "Iran" in first.title
    # Permalink is absolute, into the configured instance.
    assert first.permalink.startswith("https://redlib.test/r/wallstreetbets/comments/")


def test_parse_listing_self_vs_link(source):
    posts = {p.id: p for p in source._parse_listing(_fixture("listing_top_day.html"))}
    # The Michael Burry post is a link post (body is a URL).
    assert posts["1ttul6p"].is_self is False
    assert posts["1ttul6p"].body.startswith("http")


def test_parse_comments_extracts_scores_and_op(source):
    op, comments = source._parse_comments(_fixture("comments_thread.html"))
    assert op is not None
    assert "Iran" in op.title
    assert not op.title.startswith("News")  # flair stripped from the title text
    assert op.flair == "News"               # ...but captured in the flair field
    assert len(comments) >= 3
    top = comments[0]
    assert top.score == 4116            # exact, from the title attr
    assert top.author == "whammy5555"
    assert top.body


def test_parse_empty_html_is_safe(source):
    assert source._parse_listing("") == []
    assert source._parse_comments("") == (None, [])


def test_score_parsing_title_attr_and_abbrev():
    def el(html):
        return BeautifulSoup(html, "lxml").find("div")
    assert _parse_score(el('<div title="3440">3.4k</div>')) == 3440
    assert _parse_score(el('<div>3.4k Upvotes</div>')) == 3400
    assert _parse_score(el('<div>1.2M</div>')) == 1_200_000
    assert _parse_score(el('<div>•</div>')) == 0
    assert _parse_score(None) == 0


# ---- ticker tally -----------------------------------------------------------

def test_tally_cashtag_seeds_and_slang_filter():
    sources = [
        ("$GME to the moon, loaded calls", 500, "GME play"),
        ("NVDA earnings look strong, buying calls", 300, "NVDA DD"),
        ("My DD on YOLO FD plays, CEO said ATH", 200, "slang post"),
        ("ZZZZ is my secret pick", 100, "unknown"),
        ("still holding GME and TSLA", 50, None),  # bare, TSLA seeded below
        ("$TSLA puts printing, bearish drill", 80, "TSLA puts"),
    ]
    tally = {t.symbol: t for t in build_tally(sources, top_n=20)}
    # GME: cashtag once + bare once = 2 mentions.
    assert tally["GME"].mentions == 2
    assert tally["GME"].cashtags == 1
    # NVDA bare token counts because it's a known ticker.
    assert "NVDA" in tally
    # TSLA bare ("holding GME and TSLA") counts because $TSLA appears in corpus.
    assert tally["TSLA"].mentions == 2
    # Slang and unknown bare tokens are never tickers.
    for junk in ("DD", "YOLO", "FD", "CEO", "ATH", "ZZZZ"):
        assert junk not in tally


def test_tally_counts_once_per_source():
    sources = [("GME GME GME $GME to the moon", 100, None)]
    tally = {t.symbol: t for t in build_tally(sources)}
    assert tally["GME"].mentions == 1     # one source -> one mention
    assert tally["GME"].cashtags == 1


def test_tally_sentiment_lean():
    bull = [("$ABNB calls moon rocket breakout, bullish", 300, None)] * 3
    tally = {t.symbol: t for t in build_tally(bull)}
    assert tally["ABNB"].bull >= 2
    assert tally["ABNB"].lean == "bullish"


def test_lean_loosened_directional_on_light_signal():
    # WSB-flavored bullish phrasing now registers (all in / loaded / yolo / lfg).
    bull = {t.symbol: t for t in build_tally([("$XYZ all in, loaded, lfg", 100, None)])}
    assert bull["XYZ"].bull == 1 and bull["XYZ"].bear == 0
    assert bull["XYZ"].lean == "bullish"          # one-sided signal -> directional
    bear = {t.symbol: t for t in build_tally([("$XYZ overbought, fade it", 100, None)])}
    assert bear["XYZ"].lean == "bearish"
    # genuinely two-sided stays mixed
    mixed = {t.symbol: t for t in build_tally(
        [("$XYZ calls", 50, None), ("$XYZ puts", 50, None)])}
    assert mixed["XYZ"].lean == "mixed"


def test_sentiment_helper():
    assert _sentiment("loaded calls, bullish moon") > 0
    assert _sentiment("puts printing, bearish drill crash") < 0
    assert _sentiment("sideways chop nothing happening") == 0


# ---- render -----------------------------------------------------------------

def test_render_digest_text_sections_and_cap():
    post = RedditPost(
        id="x", title="GME squeeze incoming", score=900, num_comments=120,
        author="ape", flair="DD", body="long thesis here", permalink="u",
    )
    meg = RedditPost(
        id="m", title="Daily Discussion Thread", score=5, num_comments=2,
        author="AutoModerator", flair="Daily Discussion", body="", permalink="u",
        comments=[RedditComment(score=300, author="a", body="$GME calls")],
    )
    digest = WSBDigest(
        date="2026-06-01", subreddit="wallstreetbets",
        generated_at="2026-06-01T00:00:00+00:00",
        tickers=build_tally([("$GME calls moon", 900, "GME squeeze incoming")]),
        top_posts=[post], megathreads=[meg], comment_threads=[meg],
        posts_scanned=1, comments_scanned=1,
    )
    text = render_digest_text(digest, max_chars=7000)
    assert "MOST-MENTIONED TICKERS" in text
    assert "GME" in text
    assert "TOP POSTS OF THE DAY" in text
    assert "THREAD" in text
    assert "Daily Discussion Thread" in text  # the thread title rendered

    capped = render_digest_text(digest, max_chars=120)
    assert len(capped) <= 120


# ---- compile (fake source, no network) --------------------------------------

class _FakeSource:
    subreddit = "wallstreetbets"
    base_url = "https://redlib.test"

    def __init__(self, top, hot, megs, comments):
        self._top, self._hot, self._megs, self._comments = top, hot, megs, comments

    async def top_posts(self, period="day"):
        return list(self._top)

    async def hot_posts(self):
        return list(self._hot)

    async def find_megathreads(self, candidates):
        return list(self._megs)

    async def top_comments(self, post, *, limit=40):
        return list(self._comments)

    async def fetch_threads(self, post, *, max_parents=40):
        cs = list(self._comments)
        return cs, cs   # (all_flat, top_threads)


@pytest.mark.asyncio
async def test_compile_assembles_digest():
    top = [RedditPost(id="1", title="$NVDA to the moon calls", score=900,
                      num_comments=10, author="a", flair="YOLO", body="",
                      permalink="u")]
    meg = RedditPost(id="m", title="What Are Your Moves Tomorrow", score=5,
                     num_comments=3, author="AutoModerator",
                     flair="Daily Discussion", body="", permalink="u")
    comments = [RedditComment(score=200, author="b", body="$GME and NVDA calls")]
    src = _FakeSource(top, [meg], [meg], comments)

    digest = await compile_wsb_digest(
        src, now=dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    )
    assert digest.date == "2026-06-01"
    assert digest.posts_scanned == 1
    assert digest.comments_scanned == 1
    assert digest.megathreads[0].comments == comments
    symbols = {t.symbol for t in digest.tickers}
    assert "NVDA" in symbols and "GME" in symbols
    assert not digest.is_empty


@pytest.mark.asyncio
async def test_compile_empty_source_is_safe():
    src = _FakeSource([], [], [], [])
    digest = await compile_wsb_digest(
        src, now=dt.datetime(2026, 6, 1, tzinfo=dt.timezone.utc)
    )
    assert digest.is_empty
    # render must not blow up on an empty digest
    assert "daily digest" in render_digest_text(digest)


# ---- _fetch retry/robustness (the run-to-run variance fix) ------------------

class _FakeContent:
    def __init__(self, body):
        self._body = body

    async def iter_chunked(self, n):
        yield self._body


class _FakeResp:
    def __init__(self, status=200, body=b"<html>ok</html>", raise_exc=None):
        self.status = status
        self.headers = {"Content-Type": "text/html; charset=utf-8"}
        self.content = _FakeContent(body)
        self._raise = raise_exc

    async def __aenter__(self):
        if self._raise is not None:
            raise self._raise
        return self

    async def __aexit__(self, *a):
        return False


class _FakeSession:
    """Returns a scripted status (or raises a scripted exception) per .get()."""
    def __init__(self, script):
        self._script = list(script)
        self.calls = 0
        self.closed = False

    def get(self, url, allow_redirects=True):
        self.calls += 1
        item = self._script.pop(0)
        return _FakeResp(raise_exc=item) if isinstance(item, Exception) \
            else _FakeResp(status=item)


@pytest.mark.asyncio
async def test_fetch_retries_transient_5xx_then_succeeds(source, monkeypatch):
    monkeypatch.setattr(redlib_mod, "RETRY_BACKOFF", 0)
    source._session = _FakeSession([503, 200])
    html = await source._fetch("/r/x/top")
    assert html is not None and "ok" in html
    assert source._session.calls == 2  # retried once


@pytest.mark.asyncio
async def test_fetch_retries_on_timeout(source, monkeypatch):
    monkeypatch.setattr(redlib_mod, "RETRY_BACKOFF", 0)
    source._session = _FakeSession([asyncio.TimeoutError(), 200])
    assert await source._fetch("/r/x/top") is not None
    assert source._session.calls == 2


@pytest.mark.asyncio
async def test_fetch_gives_up_after_max_attempts(source, monkeypatch):
    monkeypatch.setattr(redlib_mod, "RETRY_BACKOFF", 0)
    source._session = _FakeSession([503] * redlib_mod.FETCH_ATTEMPTS)
    assert await source._fetch("/r/x/top") is None
    assert source._session.calls == redlib_mod.FETCH_ATTEMPTS


@pytest.mark.asyncio
async def test_fetch_does_not_retry_permanent_404(source, monkeypatch):
    monkeypatch.setattr(redlib_mod, "RETRY_BACKOFF", 0)
    source._session = _FakeSession([404, 200])
    assert await source._fetch("/r/x/top") is None
    assert source._session.calls == 1  # 404 is permanent, no retry
