"""
Tests for the WSB store, static site generator, and og:image card.
Network-free; the store uses a tmp_path SQLite file and the site writes to tmp.
"""

import struct

import pytest

from src.charts.og_card import render_og_card
from src.charts.wsb_chart import render_wsb_spark
from src.wsb.site import WSBSiteGenerator, md_to_html
from src.wsb.store import WSBDigestRecord, WSBDigestStore


def _record(date="2026-06-02", **kw):
    base = dict(
        date=date, subreddit="wallstreetbets",
        headline="SPCE eats the tape", teaser="One halted small-cap owns the thread.",
        body_md="## Vibe\n\nWSB is **long** SPCE.\n\n- point one\n- point two",
        page_url="", posts_scanned=25, comments_scanned=50,
        tickers=[{"symbol": "SPCE", "mentions": 22, "lean": "bullish",
                  "bull": 4, "bear": 1, "cashtags": 0, "weight": 4000},
                 {"symbol": "HPE", "mentions": 8, "lean": "mixed",
                  "bull": 2, "bear": 0, "cashtags": 1, "weight": 800}],
        posts=[{"title": "Trading of SPCE halted", "score": 3440, "comments": 844,
                "flair": "News", "permalink": "https://r.test/x", "is_self": True}],
    )
    base.update(kw)
    return WSBDigestRecord(**base)


# ---- markdown ---------------------------------------------------------------

def test_md_to_html_basics():
    out = md_to_html("## Heading\n\nA **bold** and _italic_ line.\n\n- one\n- two")
    assert "<h3>Heading</h3>" in out
    assert "<strong>bold</strong>" in out
    assert "<em>italic</em>" in out
    assert "<ul><li>one</li><li>two</li></ul>" in out


def test_md_to_html_escapes_html():
    out = md_to_html("watch out <script>alert(1)</script> & co")
    assert "<script>" not in out
    assert "&lt;script&gt;" in out
    assert "&amp;" in out


def test_md_to_html_links():
    out = md_to_html("see [the source](https://example.com/x)")
    assert '<a href="https://example.com/x"' in out
    assert ">the source</a>" in out


# ---- site rendering ---------------------------------------------------------

@pytest.fixture
def gen(tmp_path):
    return WSBSiteGenerator(
        static_dir=str(tmp_path / "site"),
        public_base_url="https://sigil.disinfo.zone/",
        indexable=True,
    )


def test_render_day_has_sections_and_meta(gen):
    doc = gen.render_day(_record())
    assert 'property="og:image" content="https://sigil.disinfo.zone/og/2026-06-02.png"' in doc
    assert 'name="twitter:card" content="summary_large_image"' in doc
    assert "Most-mentioned tickers" in doc
    assert 'class="lean bullish"' in doc
    assert "Top posts of the day" in doc
    assert "3,440" in doc  # score formatting
    assert "<strong>long</strong>" in doc  # body markdown rendered + marked safe


def test_chart_url_for(gen):
    assert gen.chart_url_for("2026-06-02", "NVDA") == \
        "https://sigil.disinfo.zone/charts/2026-06-02/NVDA.png"


def test_render_day_price_charts_grid(gen):
    rec = _record(charts=[
        {"symbol": "NVDA", "price": 115.0, "change_percent": 14.3, "lean": "bullish"},
        {"symbol": "GME", "price": 21.05, "change_percent": -11.7, "lean": "mixed"},
    ])
    doc = gen.render_day(rec)
    assert "Price action" in doc
    assert 'src="https://sigil.disinfo.zone/charts/2026-06-02/NVDA.png"' in doc
    assert 'href="https://finviz.com/quote.ashx?t=GME"' in doc
    assert "+14.3% over the last month" in doc  # signed alt text
    assert "-11.7% over the last month" in doc


def test_render_day_no_charts_section_without_charts(gen):
    assert "Price action" not in gen.render_day(_record())  # charts default []


def test_write_day_writes_chart_pngs(gen, tmp_path):
    gen.write_day(
        _record(charts=[{"symbol": "NVDA", "price": 1, "change_percent": 1, "lean": "bullish"}]),
        og_png=b"\x89PNG og",
        charts={"NVDA": b"\x89PNG chart", "EMPTY": b""},
    )
    site = tmp_path / "site"
    assert (site / "charts" / "2026-06-02" / "NVDA.png").read_bytes() == b"\x89PNG chart"
    assert not (site / "charts" / "2026-06-02" / "EMPTY.png").exists()  # empty skipped
    assert not list(site.glob("**/*.tmp"))


def test_render_day_noindex_toggle(tmp_path):
    g = WSBSiteGenerator(static_dir=str(tmp_path), public_base_url="https://x.test",
                         indexable=False)
    assert 'name="robots" content="noindex, nofollow"' in g.render_day(_record())


def test_write_day_and_index_are_atomic(gen, tmp_path):
    url = gen.write_day(_record(), og_png=b"\x89PNG fake")
    gen.write_index([_record()])
    site = tmp_path / "site"
    assert url == "https://sigil.disinfo.zone/wsb/2026-06-02.html"
    assert (site / "wsb" / "2026-06-02.html").exists()
    assert (site / "og" / "2026-06-02.png").read_bytes() == b"\x89PNG fake"
    assert (site / "index.html").exists()
    # no leftover temp files
    assert not list(site.glob("**/*.tmp"))


def test_index_lists_records(gen):
    doc = gen.render_index([_record(date="2026-06-02"), _record(date="2026-06-01")])
    assert "2026-06-02" in doc and "2026-06-01" in doc
    assert "SPCE (22)" in doc


# ---- og card ----------------------------------------------------------------

def test_og_card_is_1200x630_png():
    png = render_og_card("2026-06-02", "SPCE eats the tape",
                         [{"symbol": "SPCE", "mentions": 22, "lean": "bullish"}])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (1200, 630)


def test_og_card_handles_empty_tickers():
    png = render_og_card("2026-06-02", "Quiet day", [])
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


def test_og_card_long_headline_and_teaser_stays_1200x630():
    # A long headline used to run off the right edge; auto-fit + the teaser
    # must still produce an exact 1200x630 frame with no crash.
    png = render_og_card(
        "2026-06-02",
        "NVDA Ate the Thread While the Smart Money Quietly Lapped the Entire Field",
        [{"symbol": "NVDA", "mentions": 312, "lean": "bullish"}],
        teaser=("The crowd piled into NVDA at five to one bullish and bought every "
                "weekly call in sight, but the real action was somewhere far less "
                "crowded and far more interesting today."),
    )
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    assert (w, h) == (1200, 630)


# ---- price spark ------------------------------------------------------------

def test_wsb_spark_is_a_760_wide_png():
    png = render_wsb_spark("NVDA", [100, 102, 101, 108, 115], price=115.0,
                           change_percent=15.0, lean="bullish")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"
    w, h = struct.unpack(">II", png[16:24])
    # sparks are CSS-scaled on the page, so a sub-pixel rounding wobble in the
    # height is harmless; just pin the width and the rough aspect.
    assert w == 760
    assert abs(h - 240) <= 1


def test_wsb_spark_needs_at_least_two_points():
    assert render_wsb_spark("NVDA", []) == b""
    assert render_wsb_spark("NVDA", [100.0]) == b""


def test_wsb_spark_flat_series_does_not_blow_up():
    # min == max would divide by zero on the y-padding without a guard.
    png = render_wsb_spark("FLAT", [50.0, 50.0, 50.0], lean="mixed")
    assert png[:8] == b"\x89PNG\r\n\x1a\n"


# ---- store ------------------------------------------------------------------

@pytest.fixture
def store(tmp_path):
    return WSBDigestStore(str(tmp_path / "wsb.db"))


@pytest.mark.asyncio
async def test_store_upsert_get_recent(store):
    rid = await store.upsert(_record(date="2026-06-02"))
    assert rid > 0
    got = await store.get_by_date("2026-06-02")
    assert got is not None
    assert got.headline == "SPCE eats the tape"
    assert got.tickers[0]["symbol"] == "SPCE"
    assert got.comments_scanned == 50

    # upsert same date updates in place (no duplicate row)
    await store.upsert(_record(date="2026-06-02", headline="changed"))
    again = await store.get_by_date("2026-06-02")
    assert again.headline == "changed"

    await store.upsert(_record(date="2026-06-01"))
    recent = await store.recent(limit=10)
    assert [r.date for r in recent] == ["2026-06-02", "2026-06-01"]  # newest first


@pytest.mark.asyncio
async def test_store_roundtrips_charts(store):
    await store.upsert(_record(charts=[
        {"symbol": "NVDA", "price": 115.0, "change_percent": 14.3, "lean": "bullish"}]))
    got = await store.get_by_date("2026-06-02")
    assert got.charts and got.charts[0]["symbol"] == "NVDA"
    assert got.charts[0]["change_percent"] == 14.3


@pytest.mark.asyncio
async def test_store_mark_posted(store):
    await store.upsert(_record(date="2026-06-02", posted_ok=False))
    await store.mark_posted("2026-06-02", True)
    got = await store.get_by_date("2026-06-02")
    assert got.posted_ok is True
