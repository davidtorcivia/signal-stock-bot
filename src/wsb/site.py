"""
Static site generator for the WSB daily read.

Writes plain HTML files into a directory that a host Caddy serves (via a
Cloudflare tunnel) at the public base URL — no Flask route, no app-side serving.
Each day gets `wsb/<date>.html`; `index.html` is a reverse-chron list. OG/Twitter
meta tags make the Signal link unfurl into a card. Per-day og:image PNGs (from
charts.og_card) live under `og/<date>.png`.

The deterministic tally + posts are rendered into tables here (the model only
writes the prose body) so the numbers on the page are always exact. Writes are
atomic (temp file + os.replace) so Caddy never serves a half-written page.

Jinja2 is used directly (it ships with Flask, already a dependency).
"""

from __future__ import annotations

import html
import logging
import os
import re
import tempfile
from pathlib import Path
from typing import Optional

from jinja2 import Environment, FileSystemLoader, select_autoescape

from .store import WSBDigestRecord

logger = logging.getLogger(__name__)

_TEMPLATES_DIR = Path(__file__).parent / "templates"


# ---- minimal, dependency-free markdown -> HTML for the analysis body --------

_LINK_RE = re.compile(r"\[([^\]]+)\]\((https?://[^\s)]+)\)")
_BOLD_RE = re.compile(r"\*\*([^*]+)\*\*|__([^_]+)__")
_ITALIC_RE = re.compile(r"(?<![\*_])[\*_]([^\*_]+)[\*_](?![\*_])")
_CODE_RE = re.compile(r"`([^`]+)`")


def _inline(text: str) -> str:
    """Escape, then apply inline markdown (links, bold, italic, code)."""
    out = html.escape(text, quote=False)
    out = _LINK_RE.sub(
        lambda m: f'<a href="{html.escape(m.group(2))}" rel="nofollow noopener" '
                  f'target="_blank">{m.group(1)}</a>',
        out,
    )
    out = _BOLD_RE.sub(lambda m: f"<strong>{m.group(1) or m.group(2)}</strong>", out)
    out = _ITALIC_RE.sub(lambda m: f"<em>{m.group(1)}</em>", out)
    out = _CODE_RE.sub(lambda m: f"<code>{m.group(1)}</code>", out)
    return out


def md_to_html(text: str) -> str:
    """Tiny markdown renderer: headings, bullet lists, paragraphs + inline.
    Sufficient for the model-generated analysis; we control the prompt."""
    if not text:
        return ""
    blocks: list[str] = []
    lines = text.replace("\r\n", "\n").split("\n")
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].rstrip()
        if not line.strip():
            i += 1
            continue
        m = re.match(r"^(#{1,4})\s+(.*)$", line)
        if m:
            level = min(len(m.group(1)) + 1, 5)  # ## -> h3, keep h1/h2 for layout
            blocks.append(f"<h{level}>{_inline(m.group(2).strip())}</h{level}>")
            i += 1
            continue
        if re.match(r"^\s*[-*]\s+", line):
            items = []
            while i < n and re.match(r"^\s*[-*]\s+", lines[i]):
                item_text = re.sub(r"^\s*[-*]\s+", "", lines[i]).strip()
                items.append(f"<li>{_inline(item_text)}</li>")
                i += 1
            blocks.append("<ul>" + "".join(items) + "</ul>")
            continue
        # paragraph: gather consecutive non-blank, non-special lines
        para = [line]
        i += 1
        while i < n and lines[i].strip() and not re.match(r"^(#{1,4}\s|\s*[-*]\s)", lines[i]):
            para.append(lines[i].rstrip())
            i += 1
        blocks.append("<p>" + _inline(" ".join(para)) + "</p>")
    return "\n".join(blocks)


def _atomic_write(path: Path, data, *, binary: bool = False) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        if binary:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        else:
            with os.fdopen(fd, "w", encoding="utf-8") as f:
                f.write(data)
        # mkstemp creates 0600; a public static site needs world-readable files
        # so a non-root Caddy can serve them.
        os.chmod(tmp, 0o644)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


class WSBSiteGenerator:
    """Renders + writes the static daily read and index."""

    def __init__(
        self,
        static_dir: str,
        public_base_url: str,
        *,
        indexable: bool = True,
        bot_name: str = "Sigil",
        subreddit: str = "wallstreetbets",
    ):
        self.static_dir = Path(static_dir)
        self.public_base_url = (public_base_url or "").rstrip("/")
        self.indexable = indexable
        self.bot_name = bot_name
        self.subreddit = subreddit
        self.env = Environment(
            loader=FileSystemLoader(str(_TEMPLATES_DIR)),
            autoescape=select_autoescape(["html", "xml"]),
            trim_blocks=True,
            lstrip_blocks=True,
        )
        self.env.filters["md"] = md_to_html

    # ---- URLs ---------------------------------------------------------------

    def page_url_for(self, date: str) -> str:
        return f"{self.public_base_url}/wsb/{date}.html"

    def og_url_for(self, date: str) -> str:
        return f"{self.public_base_url}/og/{date}.png"

    def chart_url_for(self, date: str, symbol: str) -> str:
        return f"{self.public_base_url}/charts/{date}/{symbol}.png"

    # ---- rendering ----------------------------------------------------------

    def render_day(self, rec: WSBDigestRecord) -> str:
        tmpl = self.env.get_template("day.html")
        max_mentions = max((int(t.get("mentions", 0)) for t in rec.tickers), default=1) or 1
        return tmpl.render(
            rec=rec,
            bot_name=self.bot_name,
            subreddit=rec.subreddit or self.subreddit,
            indexable=self.indexable,
            page_url=self.page_url_for(rec.date),
            og_image=self.og_url_for(rec.date),
            chart_url_for=self.chart_url_for,
            max_mentions=max_mentions,
        )

    def render_index(self, records: list[WSBDigestRecord]) -> str:
        tmpl = self.env.get_template("index.html")
        return tmpl.render(
            records=records,
            bot_name=self.bot_name,
            subreddit=self.subreddit,
            indexable=self.indexable,
            public_base_url=self.public_base_url,
            page_url_for=self.page_url_for,
        )

    # ---- writing ------------------------------------------------------------

    def write_day(
        self,
        rec: WSBDigestRecord,
        og_png: Optional[bytes] = None,
        charts: Optional[dict[str, bytes]] = None,
    ) -> str:
        """Write the day page, its og:image, and any per-ticker price-spark
        PNGs (symbol -> bytes). Returns the public page URL."""
        if og_png:
            _atomic_write(self.static_dir / "og" / f"{rec.date}.png", og_png, binary=True)
        for symbol, png in (charts or {}).items():
            if png:
                _atomic_write(
                    self.static_dir / "charts" / rec.date / f"{symbol}.png",
                    png, binary=True,
                )
        html_doc = self.render_day(rec)
        _atomic_write(self.static_dir / "wsb" / f"{rec.date}.html", html_doc)
        logger.info("WSB page written: %s", self.static_dir / "wsb" / f"{rec.date}.html")
        return self.page_url_for(rec.date)

    def write_index(self, records: list[WSBDigestRecord]) -> str:
        _atomic_write(self.static_dir / "index.html", self.render_index(records))
        return f"{self.public_base_url}/"
