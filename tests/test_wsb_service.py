"""
Tests for the WSB digest orchestrator (service.py).

Network- and LLM-free: compile_wsb_digest is monkeypatched to a canned digest,
the deep-think client is a fake returning a JSON string, the store is a tmp
SQLite file, and the site writes to a tmp dir.
"""

import datetime as dt

import pytest

from src.wsb import service as service_mod
from src.wsb.digest import WSBDigest, TickerStat
from src.wsb.redlib import RedditPost
from src.wsb.service import (
    WSBDigestService,
    _extract_json,
    _parse_analysis,
    _strip_dashes,
)
from src.wsb.store import WSBDigestStore

NOW = dt.datetime(2026, 6, 2, tzinfo=dt.timezone.utc)


class _FakeSettings:
    def __init__(self, **vals):
        self._v = vals

    def get_stripped(self, key, default=""):
        return self._v.get(key, default)

    def get_bool(self, key, default=False):
        return self._v.get(key, default)


class _FakeDeepThink:
    def __init__(self, reply):
        self.reply = reply
        self.calls = []

    async def think(self, *, question, context, user_hash, group_id, caller_ctx=None):
        self.calls.append({"context": context, "group_id": group_id, "user_hash": user_hash})
        return self.reply


def _bars(prices):
    from src.providers.base import HistoricalBar
    base = dt.datetime(2026, 5, 1, tzinfo=dt.timezone.utc)
    return [HistoricalBar(timestamp=base + dt.timedelta(days=i),
                          open=p, high=p, low=p, close=p, volume=100)
            for i, p in enumerate(prices)]


class _FakeProviderManager:
    """get_historical returns canned bars; an unknown symbol raises (dropped)."""
    def __init__(self, bars_by_symbol):
        self._bars = bars_by_symbol
        self.calls = []

    async def get_historical(self, symbol, period="1mo", interval="1d"):
        self.calls.append((symbol, period, interval))
        bars = self._bars.get(symbol.upper())
        if bars is None:
            raise RuntimeError("no data for " + symbol)
        return bars


def _canned_digest():
    d = WSBDigest(date="2026-06-02", subreddit="wallstreetbets",
                  generated_at=NOW.isoformat())
    d.tickers = [TickerStat(symbol="SPCE", mentions=22, bull=4, bear=1, weight=4000),
                 TickerStat(symbol="HPE", mentions=8, bull=2, bear=0, cashtags=1, weight=800)]
    d.top_posts = [RedditPost(id="1", title="Trading of SPCE halted", score=3440,
                              num_comments=844, author="x", flair="News", body="",
                              permalink="https://r.test/1")]
    d.posts_scanned = 1
    d.comments_scanned = 50
    return d


@pytest.fixture
def service(tmp_path, monkeypatch):
    async def fake_compile(source, **kw):
        return _canned_digest()
    monkeypatch.setattr(service_mod, "compile_wsb_digest", fake_compile)
    store = WSBDigestStore(str(tmp_path / "wsb.db"))
    svc = WSBDigestService(
        settings_store=_FakeSettings(
            wsb_public_base_url="https://sigil.disinfo.zone",
            wsb_indexable=True,
        ),
        store=store,
        static_dir=str(tmp_path / "site"),
    )
    return svc


# ---- JSON parsing -----------------------------------------------------------

def test_extract_json_bare_and_fenced():
    assert _extract_json('{"a": 1}')["a"] == 1
    assert _extract_json('prefix ```json\n{"a": 2}\n``` suffix')["a"] == 2
    assert _extract_json("blah {\"a\": 3} trailing")["a"] == 3
    assert _extract_json("no json here") is None


def test_parse_analysis_json():
    text = '{"headline": "SPCE eats the tape", "teaser": "One name owns it.", "body_markdown": "## Vibe\\n\\nLong."}'
    h, t, b, tldr = _parse_analysis(text, date="2026-06-02")
    assert h == "SPCE eats the tape"
    assert t == "One name owns it."
    assert "Vibe" in b
    assert tldr == "One name owns it."  # falls back to teaser when tldr absent


def test_parse_analysis_extracts_tldr():
    text = ('{"headline": "h", "tldr": "The crowd YOLOd SPCE into a dilution wall '
            'while the smart money quietly bought GOOG.", "teaser": "Click me", '
            '"body_markdown": "## b\\n\\nx"}')
    h, t, b, tldr = _parse_analysis(text, date="2026-06-02")
    assert t == "Click me"
    assert "dilution wall" in tldr and tldr != t


def test_strip_dashes_removes_em_en_dashes():
    assert _strip_dashes("RSI is 90.5 — deep overbought") == "RSI is 90.5, deep overbought"
    for ch in ("—", "–", "―"):
        assert ch not in _strip_dashes(f"a {ch} b{ch}c")
    h, t, b, tldr = _parse_analysis(
        '{"headline":"A — B","teaser":"x — y","body_markdown":"p — q"}',
        date="2026-06-02")
    assert "—" not in h and "—" not in t and "—" not in b


def test_parse_analysis_prose_fallback():
    text = "WSB is fixated on SPCE today. The AI names idle in the background."
    h, t, b, tldr = _parse_analysis(text, date="2026-06-02")
    assert h == "WSB Daily, 2026-06-02"
    assert "SPCE" in t
    assert b == text
    assert tldr == t  # prose fallback reuses the blurb for the chat tldr


# ---- service.run ------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_happy_path(service, tmp_path):
    dt_client = _FakeDeepThink(
        '{"headline": "SPCE eats the tape", "tldr": "SPCE ate the whole thread on '
        'its way to a dilution cliff while nobody clapped for GOOG.", "teaser": '
        '"One halted name owns the thread; Sigil is unimpressed.", '
        '"body_markdown": "## The vibe\\n\\nWSB is **long** SPCE."}'
    )
    payload = await service.run(deep_think_client=dt_client, bot_name="Sigil", now=NOW)
    assert payload is not None
    assert payload.headline == "SPCE eats the tape"
    assert "dilution cliff" in payload.tldr and payload.tldr != payload.teaser
    assert payload.page_url == "https://sigil.disinfo.zone/wsb/2026-06-02.html"
    # deep-think was called with group_id=None (cap-safe) and got the digest text
    assert dt_client.calls[0]["group_id"] is None
    assert "SPCE" in dt_client.calls[0]["context"]
    # page + og + index written
    site = tmp_path / "site"
    assert (site / "wsb" / "2026-06-02.html").exists()
    assert (site / "og" / "2026-06-02.png").exists()
    assert (site / "index.html").exists()
    # persisted
    rec = await service.store.get_by_date("2026-06-02")
    assert rec is not None and rec.headline == "SPCE eats the tape"
    assert rec.tickers[0]["symbol"] == "SPCE"
    # no provider_manager on this fixture -> no price charts, no charts dir
    assert rec.charts == []
    assert not (site / "charts").exists()


@pytest.mark.asyncio
async def test_run_renders_price_charts_when_provider_present(tmp_path, monkeypatch):
    async def fake_compile(source, **kw):
        return _canned_digest()
    monkeypatch.setattr(service_mod, "compile_wsb_digest", fake_compile)
    # SPCE resolves; HPE has no data and must be dropped (no chart, not recorded).
    pm = _FakeProviderManager({"SPCE": _bars([10, 12, 11, 14, 13, 18])})
    svc = WSBDigestService(
        settings_store=_FakeSettings(
            wsb_public_base_url="https://sigil.disinfo.zone", wsb_indexable=True),
        store=WSBDigestStore(str(tmp_path / "wsb.db")),
        static_dir=str(tmp_path / "site"),
        provider_manager=pm,
    )
    dt_client = _FakeDeepThink(
        '{"headline":"h","teaser":"t","body_markdown":"## b\\n\\nx"}')
    payload = await svc.run(deep_think_client=dt_client, bot_name="Sigil", now=NOW)
    assert payload is not None
    site = tmp_path / "site"
    assert (site / "charts" / "2026-06-02" / "SPCE.png").exists()
    assert not (site / "charts" / "2026-06-02" / "HPE.png").exists()
    rec = await svc.store.get_by_date("2026-06-02")
    assert [c["symbol"] for c in rec.charts] == ["SPCE"]
    # 10 -> 18 over the window is +80%
    assert rec.charts[0]["change_percent"] == 80.0
    assert "SPCE" in [c[0] for c in pm.calls]


@pytest.mark.asyncio
async def test_run_deep_think_unavailable_returns_none(service, tmp_path):
    dt_client = _FakeDeepThink("(deep_think unavailable: not configured)")
    payload = await service.run(deep_think_client=dt_client, now=NOW)
    assert payload is None
    # nothing persisted, no page written (slot not burned)
    assert await service.store.get_by_date("2026-06-02") is None
    assert not (tmp_path / "site" / "wsb" / "2026-06-02.html").exists()


@pytest.mark.asyncio
async def test_run_no_client_returns_none(service):
    assert await service.run(deep_think_client=None, now=NOW) is None


@pytest.mark.asyncio
async def test_run_empty_crawl_returns_none(tmp_path, monkeypatch):
    async def empty_compile(source, **kw):
        return WSBDigest(date="2026-06-02", subreddit="wallstreetbets",
                         generated_at=NOW.isoformat())
    monkeypatch.setattr(service_mod, "compile_wsb_digest", empty_compile)
    svc = WSBDigestService(
        settings_store=_FakeSettings(),
        store=WSBDigestStore(str(tmp_path / "wsb.db")),
        static_dir=str(tmp_path / "site"),
    )
    dt_client = _FakeDeepThink('{"headline":"x","teaser":"y","body_markdown":"z"}')
    assert await svc.run(deep_think_client=dt_client, now=NOW) is None
    # deep-think never called on an empty crawl
    assert dt_client.calls == []
