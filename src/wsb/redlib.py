"""
Redlib HTML scraper for r/wallstreetbets.

Reads a self-hosted Redlib instance (a privacy-preserving Reddit frontend) and
parses its server-rendered HTML into structured posts and comments. We scrape
our own instance, so there are no rate-limit or ToS concerns; the only network
quirk is Cloudflare in front, which 403s non-browser user-agents — hence the
Mozilla-style default UA.

Reddit's official API is unavailable to us (the Responsible Builder Policy
auto-denies personal API apps), so this is the data layer for the WSB digest.

Parsing notes (Redlib markup, verified against gaegis.disinfo.zone):
  - Listing pages (/r/<sub>/top?t=day): 25 `div.post` blocks, `id` = post id.
      .post_title    h2 containing the title <a href=".../comments/..."> and,
                     often, a `.post_flair` <a> (so strip flair for a clean title)
      .post_score    div; EXACT score in the `title` attr (visible text is "3.4k")
      .post_author    a -> "u/name"
      .post_flair      a -> flair label (may be absent)
      .post_comments   a; href = permalink, `title` = "N comments"
      .post_body       div; selftext, or the external URL for link posts
      span.created     `title` = full UTC timestamp
  - Comment pages (/r/<sub>/comments/<id>/): the OP's `.post_title`/`.post_body`
    plus `div.comment` blocks, each with `.comment_score` (EXACT in `title`),
    `.comment_author`, `.comment_body`.

The class is non-throwing on the network path (fetch failures return None / empty
lists and are logged at debug), mirroring src/enrichment/links.py so it can never
take down the caller. Parsing is split into pure static methods so tests feed
saved HTML without touching the network.
"""

from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import quote

import aiohttp
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://gaegis.disinfo.zone"
DEFAULT_SUBREDDIT = "wallstreetbets"
# Cloudflare in front of the instance rejects non-browser UAs, so present as a
# real browser. Admin-overridable via the wsb_user_agent setting.
DEFAULT_USER_AGENT = (
    "Mozilla/5.0 (X11; Linux x86_64; rv:128.0) Gecko/20100101 Firefox/128.0"
)

FETCH_TIMEOUT = 12.0          # seconds, total
CONNECT_TIMEOUT = 4.0
MAX_BODY_BYTES = 3 * 1024 * 1024   # Redlib comment pages run ~180 KB; cap generously

# Transient failures (timeouts, Cloudflare rate-limits on rapid re-runs, 5xx)
# used to silently drop a whole page, which swings the ticker tally run-to-run
# when the page is the dominant megathread. Retry a few times with backoff so a
# single hiccup no longer erases a thread's comments.
FETCH_ATTEMPTS = 3
RETRY_BACKOFF = 0.8           # seconds, multiplied by the attempt number
_RETRY_STATUSES = frozenset({429, 500, 502, 503, 504})

# Megathreads we want the picks from. WSB posts these via a bot account
# ("wsbapp"), NOT AutoModerator, and flairs them "Daily Discussion" — so match
# on the distinctive title OR the discussion flair, not the author.
_MEGATHREAD_PATTERNS = (
    re.compile(r"moves?\s+tomorrow", re.IGNORECASE),
    re.compile(r"daily\s+discussion", re.IGNORECASE),
    re.compile(r"weekend\s+discussion", re.IGNORECASE),
)
_MEGATHREAD_FLAIRS = {"daily discussion", "weekend discussion", "moves tomorrow"}

_ABBREV_RE = re.compile(r"^([\d,.]+)\s*([kmb])?", re.IGNORECASE)


@dataclass
class RedditComment:
    score: int
    author: str
    body: str
    depth: int = 0
    replies: list["RedditComment"] = field(default_factory=list)


def flatten_comments(comments: list[RedditComment]) -> list[RedditComment]:
    """Depth-first flatten of a threaded comment list (for tallying)."""
    out: list[RedditComment] = []
    for c in comments:
        out.append(c)
        out.extend(flatten_comments(c.replies))
    return out


@dataclass
class RedditPost:
    id: str
    title: str
    score: int
    num_comments: int
    author: str
    flair: str
    body: str               # selftext (self posts) or external URL (link posts)
    permalink: str          # absolute URL into the Redlib instance
    created_utc: Optional[str] = None   # raw "Jun 01 2026, 14:16:28 UTC"
    comments: list[RedditComment] = field(default_factory=list)

    @property
    def is_self(self) -> bool:
        return not (self.body or "").lower().startswith(("http://", "https://"))


def _parse_score(el) -> int:
    """Exact score from the `title` attr; fall back to the abbreviated text
    ("3.4k" -> 3400). Returns 0 when unparseable (e.g. score hidden)."""
    if el is None:
        return 0
    title = (el.get("title") or "").strip().replace(",", "")
    if title.lstrip("-").isdigit():
        return int(title)
    text = el.get_text(" ", strip=True)
    m = _ABBREV_RE.match(text)
    if not m:
        return 0
    try:
        val = float(m.group(1).replace(",", ""))
    except ValueError:
        return 0
    mult = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}.get(
        (m.group(2) or "").lower(), 1
    )
    return int(val * mult)


def _post_title(post) -> str:
    """Clean title, dropping the inline flair anchor that Redlib renders inside
    the h2.post_title."""
    h2 = post.select_one(".post_title")
    if h2 is None:
        return ""
    link = h2.select_one('a[href*="/comments/"]')
    if link is not None:
        return link.get_text(" ", strip=True)
    for flair in h2.select(".post_flair"):
        flair.extract()
    return h2.get_text(" ", strip=True)


def _int_from_text(text: str) -> int:
    m = re.search(r"[\d,]+", text or "")
    return int(m.group(0).replace(",", "")) if m else 0


class RedlibSource:
    """Async scraper over a Redlib instance. Non-throwing on the network path."""

    def __init__(
        self,
        base_url: str = DEFAULT_BASE_URL,
        subreddit: str = DEFAULT_SUBREDDIT,
        user_agent: str = DEFAULT_USER_AGENT,
        *,
        session: Optional[aiohttp.ClientSession] = None,
    ):
        self.base_url = (base_url or DEFAULT_BASE_URL).strip().rstrip("/")
        self.subreddit = (subreddit or DEFAULT_SUBREDDIT).strip().lstrip("r/").strip("/")
        self.user_agent = user_agent or DEFAULT_USER_AGENT
        self._session = session
        self._owns_session = session is None
        self._session_lock = asyncio.Lock()

    # ---- network ------------------------------------------------------------

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                timeout = aiohttp.ClientTimeout(
                    total=FETCH_TIMEOUT, connect=CONNECT_TIMEOUT
                )
                self._session = aiohttp.ClientSession(
                    timeout=timeout,
                    headers={
                        "User-Agent": self.user_agent,
                        "Accept": "text/html,application/xhtml+xml",
                        "Accept-Language": "en-US,en;q=0.9",
                        # Avoid Brotli: aiohttp can't decode `br` unless the
                        # brotli package is present (it isn't in the slim prod
                        # image), and the CF frontend will happily send gzip.
                        "Accept-Encoding": "gzip, deflate",
                    },
                )
                self._owns_session = True
            return self._session

    async def _fetch(self, path: str) -> Optional[str]:
        """GET an absolute-or-relative path off the instance; return HTML text or
        None. Never raises. Retries transient failures (timeouts, rate-limits,
        5xx) with backoff so a single hiccup doesn't drop a whole page — that's
        what made the ticker tally swing run-to-run."""
        url = path if path.startswith("http") else f"{self.base_url}{path}"
        session = await self._get_session()
        for attempt in range(FETCH_ATTEMPTS):
            last_attempt = attempt == FETCH_ATTEMPTS - 1
            try:
                async with session.get(url, allow_redirects=True) as resp:
                    if resp.status >= 400:
                        if resp.status in _RETRY_STATUSES and not last_attempt:
                            await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                            continue
                        logger.debug("Redlib %s -> HTTP %s", url, resp.status)
                        return None
                    ctype = (resp.headers.get("Content-Type") or "").lower()
                    if "html" not in ctype:
                        logger.debug("Redlib %s -> non-HTML %s", url, ctype)
                        return None
                    chunks: list[bytes] = []
                    read = 0
                    async for chunk in resp.content.iter_chunked(32 * 1024):
                        chunks.append(chunk)
                        read += len(chunk)
                        if read >= MAX_BODY_BYTES:
                            break
                    return b"".join(chunks).decode("utf-8", errors="replace")
            except (asyncio.TimeoutError, aiohttp.ClientError) as e:
                if not last_attempt:
                    await asyncio.sleep(RETRY_BACKOFF * (attempt + 1))
                    continue
                logger.debug("Redlib fetch failed for %s after %d tries: %s",
                             url, FETCH_ATTEMPTS, e)
                return None
            except Exception as e:  # pragma: no cover - defensive
                logger.debug("Redlib fetch unexpected error for %s: %s", url, e)
                return None
        return None

    async def close(self) -> None:
        if self._owns_session and self._session is not None and not self._session.closed:
            await self._session.close()

    # ---- parsing (pure, no network) ----------------------------------------

    def _abs(self, href: str) -> str:
        if not href:
            return ""
        if href.startswith("http"):
            return href
        return f"{self.base_url}{href}"

    def _parse_listing(self, html: str) -> list[RedditPost]:
        posts: list[RedditPost] = []
        if not html:
            return posts
        soup = BeautifulSoup(html, "lxml")
        for el in soup.select("div.post"):
            pid = (el.get("id") or "").strip()
            if not pid:
                continue
            comments_a = el.select_one(".post_comments")
            author_a = el.select_one(".post_author")
            flair_a = el.select_one(".post_flair")
            body_el = el.select_one(".post_body")
            created = el.select_one("span.created")
            permalink = self._abs(comments_a.get("href")) if comments_a else (
                f"{self.base_url}/r/{self.subreddit}/comments/{pid}/"
            )
            num_comments = (
                _int_from_text(comments_a.get("title") or comments_a.get_text())
                if comments_a else 0
            )
            posts.append(RedditPost(
                id=pid,
                title=_post_title(el),
                score=_parse_score(el.select_one(".post_score")),
                num_comments=num_comments,
                author=(author_a.get_text(strip=True) if author_a else "").lstrip("u/"),
                flair=(flair_a.get_text(" ", strip=True) if flair_a else ""),
                body=(body_el.get_text(" ", strip=True) if body_el else ""),
                permalink=permalink,
                created_utc=(created.get("title") if created else None),
            ))
        return posts

    def _parse_comments(self, html: str) -> tuple[Optional[RedditPost], list[RedditComment]]:
        """Return (op_post_or_None, comments) for a comment page."""
        if not html:
            return None, []
        soup = BeautifulSoup(html, "lxml")
        comments: list[RedditComment] = []
        for el in soup.select("div.comment"):
            body_el = el.select_one(".comment_body")
            author_a = el.select_one(".comment_author")
            body = body_el.get_text(" ", strip=True) if body_el else ""
            if not body:
                continue
            comments.append(RedditComment(
                score=_parse_score(el.select_one(".comment_score")),
                author=(author_a.get_text(strip=True) if author_a else "").lstrip("u/"),
                body=body,
            ))
        # The OP block on a comment page: reuse listing-style selectors.
        op = None
        title_el = soup.select_one(".post_title")
        if title_el is not None:
            body_el = soup.select_one(".post_body")
            author_a = soup.select_one(".post_author")
            flair_a = soup.select_one(".post_flair")
            op = RedditPost(
                id="",
                title=_post_title(soup),
                score=_parse_score(soup.select_one(".post_score")),
                num_comments=len(comments),
                author=(author_a.get_text(strip=True) if author_a else "").lstrip("u/"),
                flair=(flair_a.get_text(" ", strip=True) if flair_a else ""),
                body=(body_el.get_text(" ", strip=True) if body_el else ""),
                permalink="",
            )
        return op, comments

    def _build_comment(self, el, depth: int) -> RedditComment:
        """Build one comment node + recurse into its DIRECT reply children.
        The parent's own score/body/author are the first matching descendants
        (Redlib renders a comment's content before its nested replies)."""
        body_el = el.select_one(".comment_body")
        author_a = el.select_one(".comment_author")
        c = RedditComment(
            score=_parse_score(el.select_one(".comment_score")),
            author=(author_a.get_text(strip=True) if author_a else "").lstrip("u/"),
            body=(body_el.get_text(" ", strip=True) if body_el else ""),
            depth=depth,
        )
        for child in el.find_all("div", class_="comment"):
            # direct reply == nearest comment-ancestor is this element
            if child.find_parent("div", class_="comment") is el:
                c.replies.append(self._build_comment(child, depth + 1))
        return c

    def _parse_comment_tree(self, html: str) -> tuple[Optional[RedditPost], list[RedditComment]]:
        """Return (op, top_level_threads) with replies nested to full depth."""
        if not html:
            return None, []
        soup = BeautifulSoup(html, "lxml")
        tops = [
            el for el in soup.select("div.comment")
            if el.find_parent("div", class_="comment") is None
        ]
        threads = [self._build_comment(el, 0) for el in tops if el.select_one(".comment_body")]
        op = None
        title_el = soup.select_one(".post_title")
        if title_el is not None:
            body_el = soup.select_one(".post_body")
            author_a = soup.select_one(".post_author")
            flair_a = soup.select_one(".post_flair")
            op = RedditPost(
                id="", title=_post_title(soup),
                score=_parse_score(soup.select_one(".post_score")),
                num_comments=len(flatten_comments(threads)),
                author=(author_a.get_text(strip=True) if author_a else "").lstrip("u/"),
                flair=(flair_a.get_text(" ", strip=True) if flair_a else ""),
                body=(body_el.get_text(" ", strip=True) if body_el else ""),
                permalink="",
            )
        return op, threads

    # ---- high-level fetches -------------------------------------------------

    async def top_posts(self, period: str = "day") -> list[RedditPost]:
        html = await self._fetch(
            f"/r/{self.subreddit}/top?t={quote(period)}"
        )
        return self._parse_listing(html or "")

    async def hot_posts(self) -> list[RedditPost]:
        html = await self._fetch(f"/r/{self.subreddit}/hot")
        return self._parse_listing(html or "")

    async def top_comments(
        self, post: RedditPost, *, limit: int = 25
    ) -> list[RedditComment]:
        """Fetch a post's comments sorted by score (highest first)."""
        path = (
            post.permalink
            or f"{self.base_url}/r/{self.subreddit}/comments/{post.id}/"
        )
        html = await self._fetch(f"{path}?sort=top")
        _op, comments = self._parse_comments(html or "")
        comments.sort(key=lambda c: c.score, reverse=True)
        return comments[:limit]

    async def fetch_threads(
        self, post: RedditPost, *, max_parents: int = 40
    ) -> tuple[list[RedditComment], list[RedditComment]]:
        """Fetch a post's comment tree once. Returns (all_flat, top_threads):
          all_flat    — every comment at any depth, for tallying mentions;
          top_threads — the top `max_parents` top-level comments by score, each
                        with replies nested, for the LLM's structured context.
        """
        path = (
            post.permalink
            or f"{self.base_url}/r/{self.subreddit}/comments/{post.id}/"
        )
        html = await self._fetch(f"{path}?sort=top")
        _op, threads = self._parse_comment_tree(html or "")
        threads.sort(key=lambda c: c.score, reverse=True)
        flat = flatten_comments(threads)
        return flat, threads[:max_parents]

    async def find_megathreads(self, candidates: list[RedditPost]) -> list[RedditPost]:
        """Pick the discussion megathreads (Moves Tomorrow / Daily Discussion)
        out of a listing (hot/root is best — they're pinned near the top)."""
        out: list[RedditPost] = []
        for p in candidates:
            title_match = any(rx.search(p.title) for rx in _MEGATHREAD_PATTERNS)
            flair_match = p.flair.strip().lower() in _MEGATHREAD_FLAIRS
            if title_match or flair_match:
                out.append(p)
        return out
