"""
Compact price-spark card for the WSB daily read — one per top ticker.

Rendered to PNG bytes and written to `charts/<date>/<SYMBOL>.png` by the static
site generator, then shown in a responsive grid on the daily page. Each card
carries the symbol, the last close, the move over the window, and a filled
sparkline coloured by the price direction (so the reader can eyeball the crowd's
lean against the actual tape).

Mirrors src/charts/og_card.py: Agg backend, the page's dark palette, a 0..1
figure-relative canvas with a pixel->fraction helper, the text.parse_math=False
guard (so a literal "$" survives), DejaVu Sans glyphs, and the figure-leak
try/finally. Save with bbox_inches=None, pad_inches=0 so the frame keeps the
requested ~760x240 size and aspect (bbox_inches='tight' would crop to content);
the page scales it to the grid cell with CSS, so a sub-pixel rounding wobble in
the height is irrelevant.

Returns raw PNG bytes; never raises out (returns b'' on failure so the page can
still publish without that card).
"""

from __future__ import annotations

import io
import logging
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

logger = logging.getLogger(__name__)

_WIDTH_PX = 760
_HEIGHT_PX = 240
_DPI = 110

# Shared with og_card / the page template's CSS variables.
_PANEL = "#15161B"
_INK = "#E8E8EC"
_FAINT = "#5A5B66"
_GREEN = "#3FB950"
_RED = "#F85149"
_AMBER = "#D29922"
_FAMILY = "DejaVu Sans"

_LEAN_COLOR = {"bullish": _GREEN, "bearish": _RED, "mixed": _AMBER}


def render_wsb_spark(
    symbol: str,
    closes: Sequence[float],
    *,
    price: Optional[float] = None,
    change_percent: Optional[float] = None,
    lean: str = "mixed",
    period_label: str = "1-month",
) -> bytes:
    """Render one ticker's price-spark card to PNG bytes. b'' on failure."""
    try:
        series = [float(c) for c in closes if c is not None]
        if len(series) < 2:
            return b""
        return _render(symbol, series, price, change_percent, lean, period_label)
    except Exception as e:  # pragma: no cover - defensive
        logger.warning("wsb spark render failed for %s: %s", symbol, e)
        return b""


def _render(symbol, series, price, change_percent, lean, period_label) -> bytes:
    fig_w = _WIDTH_PX / _DPI
    fig_h = _HEIGHT_PX / _DPI

    start, last = series[0], series[-1]
    if price is None:
        price = last
    if change_percent is None:
        change_percent = ((last / start) - 1.0) * 100.0 if start else 0.0

    # Colour the line by the actual price direction over the window; the crowd's
    # WSB lean rides along as a small tag so the two can be compared at a glance.
    up = last >= start
    line_color = _GREEN if up else _RED
    lean_color = _LEAN_COLOR.get(str(lean), _AMBER)

    rc_ctx = plt.rc_context({"text.parse_math": False})
    rc_ctx.__enter__()
    fig: Optional[Figure] = None
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, facecolor=_PANEL)

        # --- background / text layer (full-canvas, 0..1) --------------------
        bg = fig.add_axes((0, 0, 1, 1))
        bg.set_xlim(0, 1)
        bg.set_ylim(0, 1)
        bg.set_xticks([])
        bg.set_yticks([])
        for spine in bg.spines.values():
            spine.set_visible(False)
        bg.set_facecolor(_PANEL)

        def py(px_from_top: float) -> float:
            return 1.0 - (px_from_top / _HEIGHT_PX)

        lx = 30 / _WIDTH_PX
        rx = 1 - 30 / _WIDTH_PX

        # Symbol + crowd lean (top-left).
        bg.text(lx, py(40), symbol.upper(), color=_INK, fontsize=27,
                fontweight="bold", family=_FAMILY, va="center", zorder=3)
        bg.text(lx, py(74), f"WSB {lean}", color=lean_color, fontsize=13,
                fontweight="bold", family=_FAMILY, va="center", zorder=3)

        # Price + window move (top-right).
        bg.text(rx, py(38), f"${price:,.2f}", color=_INK, fontsize=23,
                fontweight="bold", family=_FAMILY, va="center", ha="right", zorder=3)
        arrow = "▲" if up else "▼"
        bg.text(rx, py(72), f"{arrow} {abs(change_percent):.1f}%  {period_label}",
                color=line_color, fontsize=16, family=_FAMILY, va="center",
                ha="right", zorder=3)

        # --- sparkline layer (inset axes in the lower band) -----------------
        spark = fig.add_axes((lx, 24 / _HEIGHT_PX, rx - lx,
                              (_HEIGHT_PX - 110 - 24) / _HEIGHT_PX))
        spark.set_facecolor(_PANEL)
        for spine in spark.spines.values():
            spine.set_visible(False)
        spark.set_xticks([])
        spark.set_yticks([])
        spark.margins(0)

        n = len(series)
        xs = list(range(n))
        lo, hi = min(series), max(series)
        pad = (hi - lo) * 0.18 or (abs(hi) * 0.02 + 0.5)
        spark.set_xlim(-0.4, n - 0.6)
        spark.set_ylim(lo - pad, hi + pad)

        # Faint baseline at the window's opening price so the move reads visually.
        spark.axhline(start, color=_FAINT, lw=0.9, ls=(0, (2, 3)), zorder=1)
        spark.fill_between(xs, series, lo - pad, color=line_color, alpha=0.13, zorder=2)
        spark.plot(xs, series, color=line_color, lw=2.4, solid_capstyle="round",
                   zorder=3)
        spark.plot([xs[-1]], [last], "o", ms=6, color=line_color,
                   markeredgecolor=_PANEL, markeredgewidth=1.4, zorder=4)

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=_DPI, facecolor=_PANEL,
                    edgecolor="none", bbox_inches=None, pad_inches=0)
        return buf.getvalue()
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")
        rc_ctx.__exit__(None, None, None)
