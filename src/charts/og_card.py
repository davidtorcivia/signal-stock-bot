"""
OpenGraph card for the WSB daily read (1200x630 PNG).

Used as the og:image so the Signal link unfurls into a branded card. Mirrors
src/charts/portfolio.py: Agg backend, dark palette, a 0..1 figure-relative
canvas with a pixel->fraction helper, the text.parse_math=False guard (so a
literal "$" survives), DejaVu Sans glyphs, and the figure-leak try/finally.

CRITICAL: save with bbox_inches=None, pad_inches=0 so the frame is EXACTLY
1200x630 (bbox_inches='tight' would crop to content and break the card size).

Returns raw PNG bytes (the static-site generator writes them to a file); does
not go through the base64/Signal attachment path.
"""

from __future__ import annotations

import io
import logging
import textwrap
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)

_WIDTH_PX = 1200
_HEIGHT_PX = 630
_DPI = 110

_BG = "#0B0B0F"
_PANEL = "#15161B"
_INK = "#E8E8EC"
_DIM = "#8A8A95"
_FAINT = "#5A5B66"
_ACCENT = "#8B7FD6"
_GREEN = "#3FB950"
_RED = "#F85149"
_AMBER = "#D29922"
_FAMILY = "DejaVu Sans"

_LEAN_COLOR = {"bullish": _GREEN, "bearish": _RED, "mixed": _AMBER}


def render_og_card(
    date_label: str,
    headline: str,
    tickers: list[dict],
    *,
    teaser: str = "",
    bot_name: str = "Sigil",
    subreddit: str = "wallstreetbets",
) -> bytes:
    """Render the card to PNG bytes. Never raises out — returns b'' on failure
    so the page can still publish without an og:image."""
    try:
        return _render(date_label, headline, tickers, teaser, bot_name, subreddit)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("og card render failed: %s", e)
        return b""


def _measure_width_px(fig, text, fontsize, *, bold) -> float:
    """Rendered width of `text` in display pixels, via a throwaway artist."""
    renderer = fig.canvas.get_renderer()
    t = fig.text(0, 0, text, fontsize=fontsize, family=_FAMILY,
                 fontweight=("bold" if bold else "normal"))
    bb = t.get_window_extent(renderer=renderer)
    t.remove()
    return bb.width


def _fit_headline(fig, text, max_width_px, *, max_lines=3):
    """Pick the largest font size + wrap width that fits the headline within
    `max_width_px` in at most `max_lines` lines. Real headlines vary a lot in
    length, so a fixed wrap width either clips long ones or wastes space on
    short ones; measuring the rendered width is what keeps it on-canvas."""
    for fontsize in (46, 42, 38, 34, 30):
        for width in range(42, 13, -1):
            lines = textwrap.wrap(text, width=width)
            if not lines or len(lines) > max_lines:
                continue
            if all(_measure_width_px(fig, ln, fontsize, bold=True) <= max_width_px
                   for ln in lines):
                return fontsize, lines
    return 30, textwrap.wrap(text, width=24)[:max_lines]


def _wrap_to_width(fig, text, fontsize, max_width_px, *, max_lines):
    """Wrap `text` so every line fits `max_width_px`, capped at `max_lines`
    (ellipsised if it would run longer)."""
    for width in range(58, 19, -1):
        lines = textwrap.wrap(text, width=width)
        if not lines:
            return []
        if len(lines) <= max_lines and all(
                _measure_width_px(fig, ln, fontsize, bold=False) <= max_width_px
                for ln in lines):
            return lines
    full = textwrap.wrap(text, width=46)
    lines = full[:max_lines]
    if lines and len(full) > max_lines:
        lines[-1] = lines[-1].rstrip(".,;: ") + "…"
    return lines


def _render(date_label, headline, tickers, teaser, bot_name, subreddit) -> bytes:
    fig_w = _WIDTH_PX / _DPI
    fig_h = _HEIGHT_PX / _DPI

    rc_ctx = plt.rc_context({"text.parse_math": False})
    rc_ctx.__enter__()
    fig: Optional[Figure] = None
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, facecolor=_BG)
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_facecolor(_BG)

        def py(px_from_top: float) -> float:
            return 1.0 - (px_from_top / _HEIGHT_PX)

        lx = 70 / _WIDTH_PX  # left margin in fig fraction

        # Accent rule down the left edge.
        ax.add_patch(plt.Rectangle((0, 0), 10 / _WIDTH_PX, 1, color=_ACCENT, zorder=1))

        # Brand row.
        ax.text(lx, py(78), "◈", color=_ACCENT, fontsize=30,
                family=_FAMILY, va="center", zorder=3)
        ax.text(lx + 46 / _WIDTH_PX, py(76), bot_name.upper(), color=_INK,
                fontsize=26, fontweight="bold", family=_FAMILY, va="center", zorder=3)
        ax.text(lx + 46 / _WIDTH_PX + (len(bot_name) * 19 + 24) / _WIDTH_PX, py(78),
                "·  WSB DAILY", color=_DIM, fontsize=18, family=_FAMILY,
                va="center", zorder=3)
        ax.text(1 - lx, py(78), date_label, color=_DIM, fontsize=18,
                family=_FAMILY, va="center", ha="right", zorder=3)

        # Headline + teaser, auto-fitted and vertically centred in the band
        # between the brand row and the chips so the card never has the long
        # dead gap (and the headline never runs off the right edge).
        max_w = _WIDTH_PX - 70 - 60  # left margin + a right gutter, in px
        head = (headline or "What WallStreetBets did today").strip()
        h_size, head_lines = _fit_headline(fig, head, max_w, max_lines=3)
        head_lh = h_size * 1.45  # px per headline line

        teaser_lines = _wrap_to_width(fig, teaser.strip(), 19, max_w, max_lines=2) \
            if teaser else []
        teaser_lh = 30.0
        gap = 26.0 if teaser_lines else 0.0

        band_top, band_bot = 120.0, 440.0  # px from top
        block_h = len(head_lines) * head_lh + gap + len(teaser_lines) * teaser_lh
        y = max(band_top, band_top + (band_bot - band_top - block_h) / 2)
        for ln in head_lines:
            ax.text(lx, py(y + head_lh / 2), ln, color=_INK, fontsize=h_size,
                    fontweight="bold", family=_FAMILY, va="center", zorder=3)
            y += head_lh
        if teaser_lines:
            y += gap
            for ln in teaser_lines:
                ax.text(lx, py(y + teaser_lh / 2), ln, color=_DIM, fontsize=19,
                        family=_FAMILY, va="center", zorder=3)
                y += teaser_lh

        # Ticker chips row.
        chips = tickers[:5]
        if chips:
            ax.text(lx, py(452), "MOST MENTIONED", color=_ACCENT, fontsize=15,
                    fontweight="bold", family=_FAMILY, va="center", zorder=3)
            cx = lx
            for t in chips:
                sym = str(t.get("symbol", ""))
                mentions = int(t.get("mentions", 0))
                color = _LEAN_COLOR.get(str(t.get("lean", "mixed")), _AMBER)
                label = f"{sym}"
                # chip width scales with label length
                w = (len(label) * 20 + 78) / _WIDTH_PX
                ax.add_patch(plt.Rectangle(
                    (cx, py(530)), w, 56 / _HEIGHT_PX, color=_PANEL, zorder=2))
                ax.add_patch(plt.Rectangle(
                    (cx, py(530)), 5 / _WIDTH_PX, 56 / _HEIGHT_PX, color=color, zorder=3))
                ax.text(cx + 18 / _WIDTH_PX, py(530 - 20), label, color=_INK,
                        fontsize=22, fontweight="bold", family=_FAMILY,
                        va="center", zorder=4)
                ax.text(cx + 18 / _WIDTH_PX, py(530 - 42), f"{mentions} mentions",
                        color=_DIM, fontsize=12, family=_FAMILY, va="center", zorder=4)
                cx += w + 16 / _WIDTH_PX

        ax.text(lx, py(600), f"r/{subreddit}  ·  not financial advice",
                color=_FAINT, fontsize=14, family=_FAMILY, va="center", zorder=3)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, facecolor=_BG,
                    edgecolor="none", bbox_inches=None, pad_inches=0)
        return buf.getvalue()
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")
        rc_ctx.__exit__(None, None, None)
