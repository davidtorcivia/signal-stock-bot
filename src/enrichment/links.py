"""
Rich link expander — fetches non-Twitter URLs and pulls Open Graph metadata
(or `<title>` as a fallback) so the LLM sees what the link is about rather
than an opaque URL.

Companion to `TwitterExpander`: this one explicitly skips Twitter/X URLs
(those have a dedicated expander with API-quality content).

Behaviour:
  * Detects http(s) URLs in arbitrary text
  * Skips known-irrelevant hosts (Twitter/X — handled elsewhere) and known
    binary suffixes (images, video, archives) so we don't fetch megabyte
    blobs to extract nothing
  * Fetches with a tight timeout, caps the body read, and parses og:title,
    og:description, og:site_name from <meta> tags. Falls back to <title>.
  * In-memory TTL cache (positive + negative) so repeated mentions don't
    re-fetch
  * Hard-capped per-message so a wall of links can't stall the dispatcher
  * Failure is silent — the original URL stays in the text untouched
"""

import asyncio
import base64
import ipaddress
import logging
import re
import time
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


URL_RE = re.compile(r"https?://[^\s<>\"']+", re.IGNORECASE)

# These have their own dedicated expander, are private content, or are
# guaranteed-irrelevant resource URLs. Skipped before fetching.
SKIP_HOSTS = {
    "twitter.com", "x.com", "fxtwitter.com", "fixupx.com", "vxtwitter.com",
    "t.co",
}


def _is_public_ip(host: str) -> bool:
    """True only for globally-routable addresses. False on parse failure."""
    try:
        ip = ipaddress.ip_address(host)
    except ValueError:
        return False
    # ::ffff:127.0.0.1 is loopback wearing a v6 costume.
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if isinstance(ip, ipaddress.IPv4Address):
        return not (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        )
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


class _PublicOnlyConnector(aiohttp.TCPConnector):
    """TCPConnector that refuses to open a socket to a non-public address.

    The check lives here, not in a custom resolver, because
    `TCPConnector._resolve_host` returns IP literals without consulting
    the resolver at all — so a resolver-based guard never sees
    `http://127.0.0.1:8093/...` or `http://169.254.169.254/...`. Every
    connection funnels through this method, which also covers the hops
    aiohttp opens when following a redirect and the case where one of a
    hostname's several A/AAAA records points inside (DNS rebinding).

    `_resolve_host` is aiohttp-private. `test_connector_refuses_*` calls
    it directly so a rename upstream fails the suite instead of silently
    disabling the guard.
    """

    async def _resolve_host(self, host: str, port: int, traces=None):
        infos = await super()._resolve_host(host, port, traces=traces)
        if not all(_is_public_ip(info["host"]) for info in infos):
            raise OSError(f"refusing non-public host: {host}")
        return infos
SKIP_SUFFIXES = (
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".svg",
    ".mp4", ".mov", ".avi", ".webm", ".mp3", ".wav", ".ogg",
    ".pdf", ".zip", ".tar", ".gz", ".7z", ".dmg", ".exe",
)

CACHE_TTL_SECONDS = 600          # 10 min positive
NEGATIVE_TTL_SECONDS = 300       # 5 min for 4xx/5xx/timeouts
FETCH_TIMEOUT = 5.0              # short — never block the dispatcher
MAX_URLS_PER_MESSAGE = 4
MAX_BODY_BYTES = 256 * 1024      # 256 KB is enough for <head>; anything bigger is rejected
MAX_TITLE_LEN = 200
MAX_DESC_LEN = 400

# A polite-ish UA — many sites will return their bot landing page or
# 403 if we send Python's default. Treat this as best-effort.
USER_AGENT = (
    "Mozilla/5.0 (compatible; SignalStockBot/1.0; "
    "+https://github.com/davidtorcivia/signal-stock-bot)"
)

_OG_TITLE_RE = re.compile(
    r"<meta[^>]*?(?:property|name)=[\"']og:title[\"'][^>]*?content=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_OG_DESC_RE = re.compile(
    r"<meta[^>]*?(?:property|name)=[\"']og:description[\"'][^>]*?content=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_OG_IMAGE_RE = re.compile(
    r"<meta[^>]*?(?:property|name)=[\"']og:image(?::url)?[\"'][^>]*?content=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_OG_SITE_RE = re.compile(
    r"<meta[^>]*?(?:property|name)=[\"']og:site_name[\"'][^>]*?content=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_DESC_RE = re.compile(
    r"<meta[^>]*?name=[\"']description[\"'][^>]*?content=[\"']([^\"']+)[\"']",
    re.IGNORECASE | re.DOTALL,
)
_TITLE_RE = re.compile(r"<title[^>]*>([^<]+)</title>", re.IGNORECASE | re.DOTALL)


def _should_skip(url: str) -> bool:
    """Cheap pre-fetch filter for URLs we shouldn't try to expand."""
    lower = url.lower()
    # Suffix test on the path only: `.../b.jpg?name=medium` is still a
    # JPEG, and letting it through costs a page fetch that Content-Type
    # throws away.
    if lower.split("?", 1)[0].split("#", 1)[0].endswith(SKIP_SUFFIXES):
        return True
    # Crude host extraction — avoids importing urllib for hot path.
    host_start = lower.find("://") + 3
    host_end = lower.find("/", host_start)
    host = lower[host_start:host_end if host_end > 0 else None]
    host = host.split(":")[0]  # strip port
    if host in SKIP_HOSTS:
        return True
    # Subdomain match (mobile.twitter.com, www.x.com, etc.)
    for skip in SKIP_HOSTS:
        if host.endswith("." + skip):
            return True
    return False


def _decode_entities(s: str) -> str:
    """Decode the handful of HTML entities that show up in og: tags."""
    return (
        s.replace("&amp;", "&")
        .replace("&lt;", "<")
        .replace("&gt;", ">")
        .replace("&quot;", '"')
        .replace("&#39;", "'")
        .replace("&apos;", "'")
    )


def _truncate(s: str, n: int) -> str:
    s = s.strip()
    return s if len(s) <= n else s[: n - 1].rstrip() + "…"


# Preview-image download caps. Deliberately tighter than the Signal
# attachment caps in signal/handler.py: a link preview is a bonus, not
# the point of the message, so it never gets to dominate the payload.
IMAGE_MAX_BYTES = 3 * 1024 * 1024
# Downloads per message, across every enricher. Matches the vision budget
# in ask_command so we don't pay for bytes that get sliced off anyway.
MAX_PREVIEW_IMAGES = 4
IMAGE_MIMES = frozenset({"image/jpeg", "image/png", "image/gif", "image/webp"})


async def download_image(session, url: str) -> Optional[dict]:
    """Fetch `url` as a `{mime, data_b64, filename}` part, or None.

    Same dict shape `signal.handler._read_inbound_image_attachments`
    produces, so link previews and real attachments travel the identical
    path into `ask_command._image_parts`.

    We download rather than handing the model a remote URL because
    providers that fetch the URL themselves fail the WHOLE completion
    when the host blocks them (OpenRouter returns a 400 for e.g.
    Wikimedia's UA filter). Downloading here means a dead preview is one
    dropped image, not a dead reply.
    """
    try:
        async with session.get(url, allow_redirects=True) as resp:
            if resp.status >= 400:
                return None
            mime = (resp.headers.get("Content-Type") or "").split(";")[0].strip().lower()
            if mime not in IMAGE_MIMES:
                return None
            declared = resp.headers.get("Content-Length")
            if declared and declared.isdigit() and int(declared) > IMAGE_MAX_BYTES:
                return None
            chunks: list[bytes] = []
            read = 0
            async for chunk in resp.content.iter_chunked(64 * 1024):
                read += len(chunk)
                if read > IMAGE_MAX_BYTES:
                    return None
                chunks.append(chunk)
            payload = b"".join(chunks)
    except (asyncio.TimeoutError, aiohttp.ClientError) as e:
        logger.debug(f"Preview image fetch failed for {url}: {e}")
        return None
    except Exception as e:
        logger.debug(f"Preview image unexpected error for {url}: {e}")
        return None
    if not payload:
        return None
    return {
        "mime": mime,
        "data_b64": base64.b64encode(payload).decode("ascii"),
        "filename": url.rsplit("/", 1)[-1][:80] or "preview",
    }


_IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg", ".gif", ".webp")


def _direct_image_urls(text: str) -> list[str]:
    """URLs in `text` that ARE an image rather than a page linking one.

    `_should_skip` filters these out of the text expander (there are no
    og: tags on a JPEG), which would otherwise make the simplest case of
    all — someone pasting an image link — the one case vision misses.
    """
    seen: set[str] = set()
    out: list[str] = []
    for u in URL_RE.findall(text or ""):
        u = u.rstrip(".,;:!?)\"'")
        path = u.split("?", 1)[0].split("#", 1)[0].lower()
        if not path.endswith(_IMAGE_SUFFIXES) or u in seen:
            continue
        seen.add(u)
        out.append(u)
        if len(out) >= MAX_PREVIEW_IMAGES:
            break
    return out


class RichLinkExpander:
    def __init__(self):
        # url -> (expires_at, (formatted_snippet, og_image_url)) — either
        # element may be None; the pair is cached together so pulling the
        # preview image costs no extra page fetch.
        self._cache: dict[str, tuple[float, tuple[Optional[str], Optional[str]]]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT, connect=2.0)
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                    connector=_PublicOnlyConnector(),
                )
            return self._session

    def _cache_get(
        self, url: str,
    ) -> Optional[tuple[Optional[str], Optional[str]]]:
        entry = self._cache.get(url)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            self._cache.pop(url, None)
            return None
        return value

    def _cache_set(
        self, url: str, value: tuple[Optional[str], Optional[str]],
    ) -> None:
        ttl = CACHE_TTL_SECONDS if any(value) else NEGATIVE_TTL_SECONDS
        self._cache[url] = (time.time() + ttl, value)

    @staticmethod
    def _format(url: str, site: Optional[str], title: Optional[str], desc: Optional[str]) -> Optional[str]:
        if not (title or desc):
            return None
        title = _truncate(_decode_entities(title), MAX_TITLE_LEN) if title else ""
        desc = _truncate(_decode_entities(desc), MAX_DESC_LEN) if desc else ""
        site = _decode_entities(site).strip() if site else ""
        head = f"[{site}] {title}" if site and title else (title or site)
        if head and desc:
            return f"{head} — {desc} ({url})"
        if head:
            return f"{head} ({url})"
        return f"{desc} ({url})"

    async def _fetch(self, url: str) -> Optional[str]:
        return (await self._fetch_meta(url))[0]

    async def _fetch_meta(
        self, url: str,
    ) -> tuple[Optional[str], Optional[str]]:
        """`(snippet, og_image_url)` for `url`. One fetch serves both."""
        cached = self._cache_get(url)
        if cached is not None:
            return cached

        session = await self._get_session()
        try:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status >= 400:
                    self._cache_set(url, (None, None))
                    return None, None
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "html" not in ctype and "xml" not in ctype:
                    self._cache_set(url, (None, None))
                    return None, None
                # Read at most MAX_BODY_BYTES — og: tags live in <head>, so
                # we're not throwing away anything useful by capping.
                chunks: list[bytes] = []
                read = 0
                async for chunk in resp.content.iter_chunked(16 * 1024):
                    chunks.append(chunk)
                    read += len(chunk)
                    if read >= MAX_BODY_BYTES:
                        break
                body = b"".join(chunks).decode("utf-8", errors="replace")
        except (asyncio.TimeoutError, aiohttp.ClientError) as e:
            logger.debug(f"Rich link fetch failed for {url}: {e}")
            self._cache_set(url, (None, None))
            return None, None
        except Exception as e:
            logger.debug(f"Rich link fetch unexpected error for {url}: {e}")
            self._cache_set(url, (None, None))
            return None, None

        title_match = _OG_TITLE_RE.search(body) or _TITLE_RE.search(body)
        desc_match = _OG_DESC_RE.search(body) or _DESC_RE.search(body)
        site_match = _OG_SITE_RE.search(body)
        image_match = _OG_IMAGE_RE.search(body)

        formatted = self._format(
            url,
            site_match.group(1) if site_match else None,
            title_match.group(1) if title_match else None,
            desc_match.group(1) if desc_match else None,
        )
        image_url = (
            _decode_entities(image_match.group(1)).strip()
            if image_match else None
        )
        if image_url and not image_url.lower().startswith(("http://", "https://")):
            image_url = None  # relative/data: og:image — not worth resolving
        self._cache_set(url, (formatted, image_url))
        return formatted, image_url

    @staticmethod
    def _urls(text: str) -> list[str]:
        """Fetchable, deduped, capped URLs in `text` — in order."""
        seen: set[str] = set()
        unique: list[str] = []
        for u in URL_RE.findall(text or ""):
            # Strip trailing punctuation that often clings to URLs in chat.
            u = u.rstrip(".,;:!?)\"'")
            if _should_skip(u) or u in seen:
                continue
            seen.add(u)
            unique.append(u)
            if len(unique) >= MAX_URLS_PER_MESSAGE:
                break
        return unique

    async def images(self, text: str) -> list[dict]:
        """Downloaded preview images for the links in `text`.

        Two sources, in that order of preference: a URL that is itself an
        image, and the og:image of a URL that is a page.
        """
        image_urls = _direct_image_urls(text)
        page_urls = self._urls(text)
        if page_urls and len(image_urls) < MAX_PREVIEW_IMAGES:
            metas = await asyncio.gather(
                *(self._fetch_meta(u) for u in page_urls)
            )
            image_urls += [img for _snippet, img in metas if img]
        image_urls = image_urls[:MAX_PREVIEW_IMAGES]
        if not image_urls:
            return []
        session = await self._get_session()
        parts = await asyncio.gather(
            *(download_image(session, u) for u in image_urls)
        )
        return [p for p in parts if p]

    async def expand(self, text: str) -> str:
        """Append link snippets after the original text, prefixed with `→ `."""
        if not text:
            return text
        unique = self._urls(text)
        if not unique:
            return text

        results = await asyncio.gather(
            *(self._fetch(u) for u in unique),
            return_exceptions=False,
        )
        # Idempotency: drop snippets already inlined in `text`. Lets us
        # re-run expand() on already-enriched messages without doubling
        # the snippets — required for the "always enrich at LLM boundary"
        # invariant.
        snippets = [r for r in results if r and r not in text]
        if not snippets:
            return text

        return text + "\n" + "\n".join(f"  → {s}" for s in snippets)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()


class CompositeEnricher:
    """Run a chain of enrichers in order; each may append its own snippets."""

    def __init__(self, *enrichers):
        self.enrichers = [e for e in enrichers if e is not None]

    async def expand(self, text: str) -> str:
        for e in self.enrichers:
            try:
                text = await e.expand(text)
            except Exception as exc:
                logger.debug(f"Enricher {type(e).__name__} failed: {exc}")
        return text

    async def images(self, text: str) -> list[dict]:
        """Preview images for every link in `text`, across all enrichers."""
        out: list[dict] = []
        for e in self.enrichers:
            fetch = getattr(e, "images", None)
            if fetch is None:
                continue
            try:
                out.extend(await fetch(text))
            except Exception as exc:
                logger.debug(f"Enricher {type(e).__name__} images failed: {exc}")
        return out

    async def close(self) -> None:
        for e in self.enrichers:
            close = getattr(e, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
