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


class TwitterExpander:
    BASE = "https://api.fxtwitter.com/status"

    def __init__(self):
        # tweet_id -> (expires_at, formatted_text or None)
        self._cache: dict[str, tuple[float, Optional[str]]] = {}
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
                self._session = aiohttp.ClientSession(
                    timeout=timeout, headers=headers,
                )
            return self._session

    def _cache_get(self, tweet_id: str) -> tuple[bool, Optional[str]]:
        """Return (hit, value). value=None means negative-cached."""
        entry = self._cache.get(tweet_id)
        if not entry:
            return False, None
        expires_at, value = entry
        if time.time() > expires_at:
            self._cache.pop(tweet_id, None)
            return False, None
        return True, value

    def _cache_set(self, tweet_id: str, value: Optional[str]) -> None:
        ttl = CACHE_TTL_SECONDS if value else NEGATIVE_TTL_SECONDS
        self._cache[tweet_id] = (time.time() + ttl, value)

    @staticmethod
    def _format(author: str, text: str) -> str:
        text = " ".join(text.split())  # collapse whitespace
        if len(text) > MAX_TEXT_LEN:
            text = text[:MAX_TEXT_LEN - 3] + "..."
        return f"[@{author}] {text}"

    async def _fetch(self, tweet_id: str) -> Optional[str]:
        hit, cached = self._cache_get(tweet_id)
        if hit:
            return cached

        try:
            session = await self._get_session()
            async with session.get(f"{self.BASE}/{tweet_id}") as resp:
                if resp.status != 200:
                    self._cache_set(tweet_id, None)
                    return None
                data = await resp.json(content_type=None)
        except (aiohttp.ClientError, asyncio.TimeoutError) as e:
            logger.debug(f"Tweet fetch failed for {tweet_id}: {e}")
            self._cache_set(tweet_id, None)
            return None
        except Exception as e:
            logger.warning(f"Tweet fetch unexpected error for {tweet_id}: {e}")
            self._cache_set(tweet_id, None)
            return None

        tweet = data.get("tweet") or {}
        text = tweet.get("text") or ""
        author = (tweet.get("author") or {}).get("screen_name") or "unknown"
        if not text:
            self._cache_set(tweet_id, None)
            return None

        formatted = self._format(author, text)
        self._cache_set(tweet_id, formatted)
        return formatted

    async def expand(self, text: str) -> str:
        """Return `text` with each tweet URL annotated by its content.

        Snippets are appended after the original message, prefixed with `→ `,
        so the LLM gets both the user's framing and the actual tweet text.
        Original message is returned unchanged when nothing can be fetched.
        """
        if not text:
            return text
        ids = STATUS_ID_RE.findall(text)
        if not ids:
            return text

        # Dedupe while preserving order; cap to avoid abuse.
        seen: set[str] = set()
        unique: list[str] = []
        for tid in ids:
            if tid not in seen:
                seen.add(tid)
                unique.append(tid)
            if len(unique) >= MAX_URLS_PER_MESSAGE:
                break

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
