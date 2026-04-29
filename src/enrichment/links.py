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
import ipaddress
import logging
import re
import socket
import time
from typing import Optional

import aiohttp
from aiohttp.abc import AbstractResolver
from aiohttp.resolver import DefaultResolver

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
    if isinstance(ip, ipaddress.IPv4Address):
        return not (
            ip.is_private or ip.is_loopback or ip.is_link_local
            or ip.is_multicast or ip.is_reserved or ip.is_unspecified
        )
    return not (
        ip.is_private or ip.is_loopback or ip.is_link_local
        or ip.is_multicast or ip.is_reserved or ip.is_unspecified
    )


class _PublicOnlyResolver(AbstractResolver):
    """aiohttp resolver that refuses non-public IPs.

    Plugged into the ClientSession's TCPConnector so the SSRF check runs
    at every DNS lookup — including the lookups aiohttp does for redirect
    targets, which a hostname-only allow/deny list can't catch. Reject
    paths: literal IPs in the URL (e.g. `http://169.254.169.254`), DNS
    names that resolve to RFC1918/loopback/link-local, and DNS names where
    any A/AAAA record points internal (DNS rebinding mitigation).
    """

    def __init__(self) -> None:
        self._inner = DefaultResolver()

    async def resolve(self, host: str, port: int = 0, family: socket.AddressFamily = socket.AF_INET):
        try:
            ipaddress.ip_address(host)
            is_literal = True
        except ValueError:
            is_literal = False
        if is_literal and not _is_public_ip(host):
            raise OSError(f"refusing private IP literal: {host}")
        infos = await self._inner.resolve(host, port=port, family=family)
        safe = [info for info in infos if _is_public_ip(info["host"])]
        if not safe:
            raise OSError(f"refusing host with only private resolutions: {host}")
        return safe

    async def close(self) -> None:
        await self._inner.close()
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
    if lower.endswith(SKIP_SUFFIXES):
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


class RichLinkExpander:
    def __init__(self):
        # url -> (expires_at, formatted_snippet or None)
        self._cache: dict[str, tuple[float, Optional[str]]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT, connect=2.0)
                connector = aiohttp.TCPConnector(resolver=_PublicOnlyResolver())
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={"User-Agent": USER_AGENT, "Accept-Language": "en"},
                    connector=connector,
                )
            return self._session

    def _cache_get(self, url: str) -> Optional[tuple[bool, Optional[str]]]:
        entry = self._cache.get(url)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            self._cache.pop(url, None)
            return None
        return True, value

    def _cache_set(self, url: str, value: Optional[str]) -> None:
        ttl = CACHE_TTL_SECONDS if value else NEGATIVE_TTL_SECONDS
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
        cached = self._cache_get(url)
        if cached is not None:
            return cached[1]

        session = await self._get_session()
        try:
            async with session.get(url, allow_redirects=True) as resp:
                if resp.status >= 400:
                    self._cache_set(url, None)
                    return None
                ctype = (resp.headers.get("Content-Type") or "").lower()
                if "html" not in ctype and "xml" not in ctype:
                    self._cache_set(url, None)
                    return None
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
            self._cache_set(url, None)
            return None
        except Exception as e:
            logger.debug(f"Rich link fetch unexpected error for {url}: {e}")
            self._cache_set(url, None)
            return None

        title_match = _OG_TITLE_RE.search(body) or _TITLE_RE.search(body)
        desc_match = _OG_DESC_RE.search(body) or _DESC_RE.search(body)
        site_match = _OG_SITE_RE.search(body)

        formatted = self._format(
            url,
            site_match.group(1) if site_match else None,
            title_match.group(1) if title_match else None,
            desc_match.group(1) if desc_match else None,
        )
        self._cache_set(url, formatted)
        return formatted

    async def expand(self, text: str) -> str:
        """Append link snippets after the original text, prefixed with `→ `."""
        if not text:
            return text
        urls = URL_RE.findall(text)
        if not urls:
            return text

        seen: set[str] = set()
        unique: list[str] = []
        for u in urls:
            # Strip trailing punctuation that often clings to URLs in chat.
            u = u.rstrip(".,;:!?)\"'")
            if _should_skip(u):
                continue
            if u in seen:
                continue
            seen.add(u)
            unique.append(u)
            if len(unique) >= MAX_URLS_PER_MESSAGE:
                break

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

    async def close(self) -> None:
        for e in self.enrichers:
            close = getattr(e, "close", None)
            if callable(close):
                try:
                    await close()
                except Exception:
                    pass
