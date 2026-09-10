"""
Tweet expander — turns Twitter / X status URLs into inline text snippets so
the LLM can use them as conversation context.

Implementation:
  * Extracts status IDs from twitter.com / x.com / fxtwitter.com URLs
  * Fetches text + author from api.fxtwitter.com (free, no auth)
  * 10-min in-memory TTL cache (positive + negative — broken tweets aren't
    re-fetched on every message)
  * Hard-capped per-message to MAX_URLS_PER_MESSAGE so a wall of links
    can't stall the dispatcher
  * Failure is silent — original URL stays in the text untouched
"""

import asyncio
import logging
import re
import time
from typing import Optional

import aiohttp

from .links import MAX_PREVIEW_IMAGES, _PublicOnlyConnector, download_image

logger = logging.getLogger(__name__)

# Matches https://(www.)?(twitter|x|fxtwitter|fixupx).com/<handle>/status/<id>
STATUS_ID_RE = re.compile(
    r"https?://(?:www\.)?(?:twitter|x|fxtwitter|fixupx|vxtwitter)\.com/"
    r"[^/\s]+/status/(\d+)",
    re.IGNORECASE,
)

CACHE_TTL_SECONDS = 600          # 10 min positive cache
NEGATIVE_TTL_SECONDS = 300       # 5 min for 404/error
FETCH_TIMEOUT = 5.0              # short — don't block dispatcher
MAX_URLS_PER_MESSAGE = 5
MAX_TEXT_LEN = 500               # truncate long tweets in context
MAX_IMAGES_PER_TWEET = 2         # a 4-photo tweet shouldn't eat the payload


class TwitterExpander:
    BASE = "https://api.fxtwitter.com/status"

    def __init__(self):
        # tweet_id -> (expires_at, (formatted_text, [image_url, ...])) —
        # media travels with the snippet so the vision path costs no
        # extra API round-trip.
        self._cache: dict[str, tuple[float, tuple[Optional[str], list[str]]]] = {}
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(total=FETCH_TIMEOUT, connect=3)
                # Skip brotli — aiohttp doesn't ship a decoder by default,
                # and fxtwitter happily returns gzip if asked.
                headers = {
                    "Accept": "application/json",
                    "Accept-Encoding": "gzip, deflate",
                    "User-Agent": "signal-stock-bot/1.0",
                }
                # Same public-only connector the rich-link expander uses:
                # media URLs arrive from a third-party API response, and
                # download_image follows them.
                self._session = aiohttp.ClientSession(
                    connector=_PublicOnlyConnector(),
                    timeout=timeout, headers=headers,
                )
            return self._session

    def _cache_get(
        self, tweet_id: str,
    ) -> Optional[tuple[Optional[str], list[str]]]:
        entry = self._cache.get(tweet_id)
        if not entry:
            return None
        expires_at, value = entry
        if time.time() > expires_at:
            self._cache.pop(tweet_id, None)
            return None
        return value

    def _cache_set(
        self, tweet_id: str, value: tuple[Optional[str], list[str]],
    ) -> None:
        ttl = CACHE_TTL_SECONDS if any(value) else NEGATIVE_TTL_SECONDS
        self._cache[tweet_id] = (time.time() + ttl, value)

    @staticmethod
    def _format(author: str, text: str) -> str:
        text = " ".join(text.split())  # collapse whitespace
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN - 3] + "..."
        return f"[@{author}] {text}"

    @staticmethod
    def _media_urls(tweet: dict) -> list[str]:
        """Photo URLs (and video posters) attached to a tweet.

        fxtwitter hands back `?name=orig` variants — full-resolution
        camera files. Downgraded to `name=medium` (long edge 1200px):
        plenty for a model that resamples anyway, and a fraction of the
        bytes.
        """
        media = tweet.get("media") or {}
        urls: list[str] = []
        for photo in media.get("photos") or []:
            url = (photo or {}).get("url")
            if url:
                urls.append(url.replace("?name=orig", "?name=medium"))
        for video in media.get("videos") or []:
            thumb = (video or {}).get("thumbnail_url")
            if thumb:
                urls.append(thumb)
        return urls[:MAX_IMAGES_PER_TWEET]

    async def _fetch(self, tweet_id: str) -> Optional[str]:
        return (await self._fetch_meta(tweet_id))[0]

    async def _fetch_meta(
        self, tweet_id: str,
    ) -> tuple[Optional[str], list[str]]:
        cached = self._cache_get(tweet_id)
        if cached is not None:
            return cached

        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE}/{tweet_id}") as resp:
                if resp.status != 200:
                    self._cache_set(tweet_id, (None, []))
                    return None, []
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Tweet fetch failed for {tweet_id}: {e}")
            self._cache_set(tweet_id, (None, []))
            return None, []
        except Exception as e:
            logger.warning(f"Tweet fetch unexpected error for {tweet_id}: {e}")
            self._cache_set(tweet_id, (None, []))
            return None, []

        tweet = data.get("tweet") or {}
        text = tweet.get("text") or ""
        author = (tweet.get("author") or {}).get("screen_name") or "unknown"
        media_urls = self._media_urls(tweet)
        if not text and not media_urls:
            self._cache_set(tweet_id, (None, []))
            return None, []

        # An image-only tweet still needs a snippet, or expand() drops it
        # and the model never learns the link had anything behind it.
        formatted = self._format(author, text or "(image)")
        self._cache_set(tweet_id, (formatted, media_urls))
        return formatted, media_urls

    @staticmethod
    def _tweet_ids(text: str) -> list[str]:
        """Deduped, capped status IDs in `text` — in order."""
        seen: set[str] = set()
        unique: list[str] = []
        for tid in STATUS_ID_RE.findall(text or ""):
            if tid in seen:
                continue
            seen.add(tid)
            unique.append(tid)
            if len(unique) >= MAX_URLS_PER_MESSAGE:
                break
        return unique

    async def images(self, text: str) -> list[dict]:
        """Downloaded photos from the tweets linked in `text`."""
        ids = self._tweet_ids(text)
        if not ids:
            return []
        metas = await asyncio.gather(*(self._fetch_meta(t) for t in ids))
        urls = [u for _snippet, media in metas for u in media][:MAX_PREVIEW_IMAGES]
        if not urls:
            return []
        session = await self._get_session()
        parts = await asyncio.gather(
            *(download_image(session, u) for u in urls)
        )
        return [p for p in parts if p]

    async def expand(self, text: str) -> str:
        """Return `text` with each tweet URL annotated by its content.

        Snippets are appended after the original message, prefixed with `→ `,
        so the LLM gets both the user's framing and the actual tweet text.
        Original message is returned unchanged when nothing can be fetched.
        """
        if not text:
            return text
        unique = self._tweet_ids(text)
        if not unique:
            return text

        results = await asyncio.gather(
            *(self._fetch(tid) for tid in unique),
            return_exceptions=False,
        )
        # Idempotency: drop snippets whose content is already inlined in
        # `text` (because expand() was called on it previously). Without
        # this, re-running expand() on stored, already-enriched messages
        # — which happens at every LLM feed boundary — would double the
        # snippets each pass.
        snippets = [r for r in results if r and r not in text]
        if not snippets:
            return text

        return text + "\n" + "\n".join(f"  → {s}" for s in snippets)

    async def close(self) -> None:
        if self._session and not self._session.closed:
            await self._session.close()
