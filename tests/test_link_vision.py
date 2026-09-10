"""
Tests for preview images pulled out of pasted links.

Covers:
  * tweet media URLs are read off the fxtwitter payload (and downgraded
    from ?name=orig)
  * og:image is parsed off a link preview
  * download_image rejects non-images and oversized bodies
  * CompositeEnricher.images fans out and tolerates a broken enricher
"""

import base64

import aiohttp
import pytest

from src.enrichment.links import (
    IMAGE_MAX_BYTES,
    MAX_PREVIEW_IMAGES,
    CompositeEnricher,
    RichLinkExpander,
    _PublicOnlyConnector,
    _is_public_ip,
    _direct_image_urls,
    download_image,
)
from src.enrichment.twitter import TwitterExpander


PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmM"
    "IQAAAABJRU5ErkJggg=="
)


class _Resp:
    def __init__(self, status=200, headers=None, body=b""):
        self.status = status
        self.headers = headers or {}
        self._body = body

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    @property
    def content(self):
        body = self._body

        class _Content:
            @staticmethod
            async def iter_chunked(_n):
                for i in range(0, len(body), 4):
                    yield body[i:i + 4]

        return _Content()


class _Session:
    """Minimal aiohttp.ClientSession stand-in: one canned response."""

    def __init__(self, resp):
        self._resp = resp
        self.calls = []

    def get(self, url, **kw):
        self.calls.append(url)
        return self._resp


# ── download_image ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_download_image_returns_part():
    session = _Session(_Resp(headers={"Content-Type": "image/png"}, body=PNG))
    part = await download_image(session, "https://ex.test/a.png")
    assert part["mime"] == "image/png"
    assert base64.b64decode(part["data_b64"]) == PNG
    assert part["filename"] == "a.png"


@pytest.mark.asyncio
async def test_download_image_rejects_non_image():
    session = _Session(_Resp(headers={"Content-Type": "text/html"}, body=b"<html>"))
    assert await download_image(session, "https://ex.test/a") is None


@pytest.mark.asyncio
async def test_download_image_rejects_declared_oversize():
    session = _Session(_Resp(
        headers={"Content-Type": "image/png",
                 "Content-Length": str(IMAGE_MAX_BYTES + 1)},
        body=PNG,
    ))
    assert await download_image(session, "https://ex.test/a.png") is None


@pytest.mark.asyncio
async def test_download_image_rejects_streamed_oversize():
    """A lying (or absent) Content-Length doesn't get to blow the cap."""
    session = _Session(_Resp(
        headers={"Content-Type": "image/png"},
        body=b"\x00" * (IMAGE_MAX_BYTES + 8),
    ))
    assert await download_image(session, "https://ex.test/a.png") is None


@pytest.mark.asyncio
async def test_download_image_swallows_errors():
    class _Boom:
        def get(self, url, **kw):
            raise RuntimeError("nope")

    assert await download_image(_Boom(), "https://ex.test/a.png") is None


# ── tweet media ────────────────────────────────────────────────────────────

def test_media_urls_downgrades_orig():
    urls = TwitterExpander._media_urls({
        "media": {"photos": [
            {"url": "https://pbs.twimg.com/media/abc.jpg?name=orig"},
        ]},
    })
    assert urls == ["https://pbs.twimg.com/media/abc.jpg?name=medium"]


def test_media_urls_includes_video_poster_and_caps():
    urls = TwitterExpander._media_urls({
        "media": {
            "photos": [{"url": f"https://p/{i}.jpg"} for i in range(5)],
            "videos": [{"thumbnail_url": "https://p/v.jpg"}],
        },
    })
    assert len(urls) == 2  # MAX_IMAGES_PER_TWEET


def test_media_urls_empty_when_no_media():
    assert TwitterExpander._media_urls({"text": "hi"}) == []


@pytest.mark.asyncio
async def test_tweet_images_end_to_end(monkeypatch):
    exp = TwitterExpander()

    async def fake_meta(tweet_id):
        return f"[@x] tweet {tweet_id}", ["https://pbs.twimg.com/media/a.png"]

    async def fake_session():
        return _Session(_Resp(headers={"Content-Type": "image/png"}, body=PNG))

    monkeypatch.setattr(exp, "_fetch_meta", fake_meta)
    monkeypatch.setattr(exp, "_get_session", fake_session)

    parts = await exp.images("look https://x.com/a/status/12345")
    assert len(parts) == 1
    assert parts[0]["mime"] == "image/png"


@pytest.mark.asyncio
async def test_tweet_images_none_without_url():
    assert await TwitterExpander().images("no links here") == []


# ── og:image ───────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_og_image_parsed(monkeypatch):
    html = (
        b'<html><head>'
        b'<meta property="og:title" content="Headline">'
        b'<meta property="og:image" content="https://cdn.test/p.jpg?a=1&amp;b=2">'
        b'</head></html>'
    )
    exp = RichLinkExpander()

    async def fake_session():
        return _Session(_Resp(headers={"Content-Type": "text/html"}, body=html))

    monkeypatch.setattr(exp, "_get_session", fake_session)
    snippet, image = await exp._fetch_meta("https://news.test/story")
    assert "Headline" in snippet
    assert image == "https://cdn.test/p.jpg?a=1&b=2"


@pytest.mark.asyncio
async def test_relative_og_image_dropped(monkeypatch):
    html = b'<html><head><meta property="og:image" content="/p.jpg"></head></html>'
    exp = RichLinkExpander()

    async def fake_session():
        return _Session(_Resp(headers={"Content-Type": "text/html"}, body=html))

    monkeypatch.setattr(exp, "_get_session", fake_session)
    _snippet, image = await exp._fetch_meta("https://news.test/story")
    assert image is None


# ── composite ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_composite_images_survives_a_broken_enricher():
    class _Good:
        async def images(self, text):
            return [{"mime": "image/png", "data_b64": "x"}]

    class _Bad:
        async def images(self, text):
            raise RuntimeError("boom")

    class _TextOnly:
        async def expand(self, text):
            return text

    out = await CompositeEnricher(_Bad(), _Good(), _TextOnly()).images("hi")
    assert len(out) == 1


# ── direct image links ─────────────────────────────────────────────────────

def test_direct_image_urls_found_despite_skip_suffixes():
    urls = _direct_image_urls(
        "see https://i.imgur.com/a.PNG and "
        "https://pbs.twimg.com/media/b.jpg?name=medium and "
        "https://news.test/story"
    )
    assert urls == [
        "https://i.imgur.com/a.PNG",
        "https://pbs.twimg.com/media/b.jpg?name=medium",
    ]


def test_direct_image_urls_capped_and_deduped():
    text = " ".join(["https://ex.test/a.png"] * 3 + [
        f"https://ex.test/{i}.jpg" for i in range(6)
    ])
    assert len(_direct_image_urls(text)) == MAX_PREVIEW_IMAGES


@pytest.mark.asyncio
async def test_images_downloads_direct_link_without_page_fetch(monkeypatch):
    exp = RichLinkExpander()
    session = _Session(_Resp(headers={"Content-Type": "image/png"}, body=PNG))

    async def fake_session():
        return session

    async def boom(_url):
        raise AssertionError("should not fetch a page for a direct image URL")

    monkeypatch.setattr(exp, "_get_session", fake_session)
    monkeypatch.setattr(exp, "_fetch_meta", boom)

    parts = await exp.images("https://i.imgur.com/a.png")
    assert len(parts) == 1
    assert session.calls == ["https://i.imgur.com/a.png"]


@pytest.mark.parametrize("host", [
    "127.0.0.1", "10.0.0.5", "192.168.1.1", "169.254.169.254",
    "::1", "::ffff:127.0.0.1",
])
@pytest.mark.asyncio
async def test_connector_refuses_private_literals(host):
    """aiohttp hands back IP literals without asking the resolver, so the
    guard has to sit in the connector. Calls the private hook directly:
    if upstream renames it, this fails instead of going quiet."""
    conn = _PublicOnlyConnector()
    try:
        with pytest.raises(OSError):
            await conn._resolve_host(host, 80)
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_connector_allows_public_literal():
    conn = _PublicOnlyConnector()
    try:
        infos = await conn._resolve_host("93.184.216.34", 80)
        assert infos and infos[0]["host"] == "93.184.216.34"
    finally:
        await conn.close()


@pytest.mark.asyncio
async def test_connector_refuses_when_any_record_is_private(monkeypatch):
    """DNS rebinding: one public A record doesn't excuse a private one."""
    async def mixed(self, host, port, traces=None):
        return [
            {"hostname": host, "host": "93.184.216.34", "port": port},
            {"hostname": host, "host": "127.0.0.1", "port": port},
        ]

    monkeypatch.setattr(aiohttp.TCPConnector, "_resolve_host", mixed)
    conn = _PublicOnlyConnector()
    try:
        with pytest.raises(OSError):
            await conn._resolve_host("evil.test", 80)
    finally:
        await conn.close()


def test_ipv4_mapped_ipv6_is_not_public():
    assert _is_public_ip("::ffff:10.0.0.1") is False
    assert _is_public_ip("93.184.216.34") is True


@pytest.mark.asyncio
async def test_both_expander_sessions_use_the_guarded_connector():
    """download_image follows URLs from third-party content, so every
    session it can be handed must go through the public-only connector."""
    for expander in (RichLinkExpander(), TwitterExpander()):
        session = await expander._get_session()
        try:
            assert isinstance(
                session.connector, _PublicOnlyConnector
            ), type(expander).__name__
        finally:
            await expander.close()
