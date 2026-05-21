"""
Portfolio dashboard image renderer.

Signal renders monospace text/markdown tables poorly on narrow phone
screens; this produces a clean PNG card so chat members see a real
dashboard instead of a wrapped pseudo-table. The text form
(``render_status`` in commands/portfolio_command.py) is still used for
LLM tool I/O and for accessibility/fallback when the image can't be
generated.

Layout:

    +------------------------------------------------------+
    |  ◈ Sigil's portfolio                  ● market open  |
    |  EQUITY $1,234.56                                    |
    |  ▲ +$234.56  (+23.46%)                               |
    |                                                      |
    |  Cash $312  |  Holdings $922  |  Funded $1000        |
    |  Realized $45.20                                     |
    |                                                      |
    |  TICKER   QTY      AVG       MARK     P/L      %     |
    |  ────────────────────────────────────────────────    |
    |  AAPL     5        $190.00   $215.30  ▲ $126   +13   |
    |  TSLA     2        $250.00   $245.00  ▼ -$10   -2.0  |
    +------------------------------------------------------+
"""

from __future__ import annotations

import base64
import io
import logging
from typing import Optional

import matplotlib

matplotlib.use("Agg")  # headless

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

logger = logging.getLogger(__name__)


# Visual constants — tuned to match the dark "nightclouds"-derived palette
# used by the stock chart generator so the two image kinds feel like
# siblings.
_BG = "#0B0B0F"
_PANEL = "#15161B"
_INK = "#FFFFFF"
_DIM = "#888A93"
_GRID = "#22232A"
_GREEN = "#00C853"
_RED = "#FF1744"
_AMBER = "#FFB300"

_WIDTH_PX = 880
_DPI = 110
_ROW_HEIGHT_PX = 36       # per position row
_OPTION_ROW_PX = 32       # per options-position row (compact one-liner)
_OPTION_HEADER_PX = 56    # space reserved for the "OPTIONS" sub-heading
_ORDER_ROW_PX = 30        # per pending-order row (slightly tighter)
_ORDER_HEADER_PX = 56     # space reserved for the "OPEN ORDERS" heading
_BASE_HEIGHT_PX = 380     # header + summary block + table header + padding


def _fmt_dollars(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _fmt_pct(pct: float) -> str:
    return f"{pct * 100:+.2f}%"


def _fmt_qty(qty: float) -> str:
    if abs(qty - round(qty)) < 1e-6:
        return f"{int(round(qty))}"
    return f"{qty:.4f}".rstrip("0").rstrip(".")


def render_portfolio_image(
    snap: dict,
    *,
    bot_name: str = "Sigil",
) -> str:
    """Render a portfolio snapshot as a base64-encoded PNG.

    `snap` is the dict returned by PaperPortfolioExecutor.status — see
    that method for the full field list. Caller is responsible for
    catching exceptions and falling back to text rendering; we let any
    matplotlib failure propagate so the caller can log + degrade.
    """
    positions = snap.get("positions") or []
    options_positions = snap.get("options_positions") or []
    pending_orders = snap.get("pending_orders") or []
    n_positions = len(positions)
    n_options = len(options_positions)
    n_orders = len(pending_orders)

    height_px = _BASE_HEIGHT_PX + max(1, n_positions) * _ROW_HEIGHT_PX
    if n_positions == 0 and n_options == 0:
        # Reserve a single "no open positions" row.
        height_px = _BASE_HEIGHT_PX + _ROW_HEIGHT_PX
    elif n_positions == 0:
        # Skip the "no positions" filler when options exist — the
        # OPTIONS section below carries the visual weight.
        height_px = _BASE_HEIGHT_PX
    # Options block — appended when the bot holds at least one
    # contract. Rendered as a compact one-liner per contract (the
    # OCC-derived friendly name needs the full row width).
    if n_options > 0:
        height_px += _OPTION_HEADER_PX + n_options * _OPTION_ROW_PX
    # Orders block — only added when there are pending orders. Empty
    # case keeps the card compact (the chat sees no clutter when
    # nothing's queued).
    if n_orders > 0:
        height_px += _ORDER_HEADER_PX + n_orders * _ORDER_ROW_PX

    fig_w = _WIDTH_PX / _DPI
    fig_h = height_px / _DPI

    # `text.parse_math` (mpl 3.5+) disables `$…$` -> mathtext for every
    # text artist constructed while the override is active. Without it,
    # formatters that emit literal dollars ("seed $1,000.00 + tips
    # $50.00") get parsed as math mode and the dollar signs vanish.
    # We push rcParams via rc_context across the entire build so every
    # ax.text(...) inside picks up the override at construction time.
    rc_ctx = plt.rc_context({"text.parse_math": False})
    rc_ctx.__enter__()
    fig: Optional[Figure] = None
    try:
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=_DPI, facecolor=_BG)

        # One full-figure axes used purely as a drawing canvas in
        # 0..1 figure-relative coordinates. Hides ticks + spines so we
        # have a flat black canvas for text.
        ax = fig.add_axes([0, 0, 1, 1])
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_facecolor(_BG)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Card panel (slightly inset, rounded corners) — a subtle frame
        # so the eye sees a "card" rather than a poster.
        panel = FancyBboxPatch(
            (0.012, 0.012),
            0.976,
            0.976,
            boxstyle="round,pad=0.002,rounding_size=0.012",
            linewidth=1.0,
            edgecolor=_GRID,
            facecolor=_PANEL,
            transform=ax.transAxes,
            zorder=0,
        )
        ax.add_patch(panel)

        # Convert pixel offsets to figure-relative y so layout is
        # predictable across position counts.
        def py(px_from_top: float) -> float:
            return 1.0 - (px_from_top / height_px)

        # ---- Header row -------------------------------------------------
        title = f"◈ {bot_name}'s portfolio"
        if snap.get("label"):
            title += f"  ·  {snap['label']}"
        ax.text(
            0.04, py(38),
            title,
            color=_INK,
            fontsize=15,
            fontweight="bold",
            family="DejaVu Sans",
            va="center",
            zorder=2,
        )

        market_open = bool(snap.get("market_open"))
        market_label = "● market open" if market_open else "○ market closed"
        market_color = _GREEN if market_open else _DIM
        ax.text(
            0.96, py(38),
            market_label,
            color=market_color,
            fontsize=10,
            family="DejaVu Sans",
            va="center",
            ha="right",
            zorder=2,
        )

        # ---- Equity + PnL block ----------------------------------------
        equity = float(snap.get("equity", 0.0))
        pnl = float(snap.get("total_pnl", 0.0))
        pnl_pct = float(snap.get("total_pnl_pct", 0.0))
        pnl_color = _GREEN if pnl >= 0 else _RED
        arrow = "▲" if pnl >= 0 else "▼"

        ax.text(
            0.04, py(86),
            "EQUITY",
            color=_DIM,
            fontsize=10,
            family="DejaVu Sans",
            va="center",
            zorder=2,
        )
        ax.text(
            0.04, py(120),
            _fmt_dollars(equity),
            color=_INK,
            fontsize=30,
            fontweight="bold",
            family="DejaVu Sans",
            va="center",
            zorder=2,
        )
        ax.text(
            0.04, py(160),
            f"{arrow} {_fmt_dollars(pnl)}   ({_fmt_pct(pnl_pct)})",
            color=pnl_color,
            fontsize=14,
            fontweight="bold",
            family="DejaVu Sans",
            va="center",
            zorder=2,
        )

        # ---- Stat strip: Cash | Holdings | Funded ----------------------
        cash = float(snap.get("cash", 0.0))
        market_value = float(snap.get("market_value", 0.0))
        funded = float(snap.get("total_funded", 0.0))
        starting = float(snap.get("starting_balance", 0.0))
        tip_total = float(snap.get("tip_total", 0.0))

        strip_y = py(210)
        for col_x, label, value, sub in (
            (0.04, "CASH", _fmt_dollars(cash), None),
            (0.36, "HOLDINGS", _fmt_dollars(market_value), None),
            (
                0.68,
                "FUNDED",
                _fmt_dollars(funded),
                f"seed {_fmt_dollars(starting)} + tips {_fmt_dollars(tip_total)}",
            ),
        ):
            ax.text(
                col_x, strip_y + 0.024,
                label,
                color=_DIM,
                fontsize=9,
                family="DejaVu Sans",
                va="center",
                zorder=2,
            )
            ax.text(
                col_x, strip_y - 0.008,
                value,
                color=_INK,
                fontsize=15,
                fontweight="bold",
                family="DejaVu Sans",
                va="center",
                zorder=2,
            )
            if sub:
                ax.text(
                    col_x, strip_y - 0.044,
                    sub,
                    color=_DIM,
                    fontsize=8,
                    family="DejaVu Sans",
                    va="center",
                    zorder=2,
                )

        # ---- Realized PnL strip (only if non-zero) ---------------------
        realized = float(snap.get("realized_pnl", 0.0))
        if abs(realized) > 0.005:
            r_color = _GREEN if realized >= 0 else _RED
            ax.text(
                0.04, py(268),
                f"Realized P/L: {_fmt_dollars(realized)}",
                color=r_color,
                fontsize=10,
                family="DejaVu Sans",
                va="center",
                zorder=2,
            )

        # ---- Positions table -------------------------------------------
        # Column x-anchors. Tickers align left; numeric columns align
        # right against their anchor for a clean "ledger" look.
        col_x = {
            "ticker": 0.04,
            "qty": 0.30,
            "avg": 0.46,
            "mark": 0.62,
            "pl": 0.80,
            "pct": 0.96,
        }
        header_y_px = 308
        ax.text(col_x["ticker"], py(header_y_px), "TICKER",
                color=_DIM, fontsize=9, family="DejaVu Sans",
                va="center", ha="left", zorder=2)
        for key, label in (
            ("qty", "QTY"),
            ("avg", "AVG"),
            ("mark", "MARK"),
            ("pl", "P/L"),
            ("pct", "%"),
        ):
            ax.text(col_x[key], py(header_y_px), label,
                    color=_DIM, fontsize=9, family="DejaVu Sans",
                    va="center", ha="right", zorder=2)

        # Divider under the header row.
        ax.plot(
            [0.04, 0.96], [py(header_y_px + 16), py(header_y_px + 16)],
            color=_GRID, linewidth=0.8, zorder=1,
        )

        if not positions:
            ax.text(
                0.5, py(header_y_px + 56),
                "(no open positions)",
                color=_DIM,
                fontsize=11,
                family="DejaVu Sans",
                va="center",
                ha="center",
                zorder=2,
            )
        else:
            row_y_px = header_y_px + 38
            for i, p in enumerate(positions):
                # Faint zebra stripe on alternating rows.
                if i % 2 == 1:
                    stripe = FancyBboxPatch(
                        (0.025, py(row_y_px + 16) - 0.001),
                        0.95,
                        (_ROW_HEIGHT_PX / height_px),
                        boxstyle="round,pad=0,rounding_size=0.004",
                        linewidth=0,
                        facecolor="#1B1C22",
                        transform=ax.transAxes,
                        zorder=0,
                    )
                    ax.add_patch(stripe)

                ticker = str(p.get("ticker") or "")
                qty = float(p.get("qty") or 0.0)
                avg_cost = float(p.get("avg_cost") or 0.0)
                mark = p.get("mark_price")
                upnl = float(p.get("unrealized_pnl") or 0.0)
                upct = float(p.get("unrealized_pct") or 0.0)

                row_color = _GREEN if upnl >= 0 else _RED
                row_arrow = "▲" if upnl >= 0 else "▼"

                ax.text(col_x["ticker"], py(row_y_px), ticker,
                        color=_INK, fontsize=12, fontweight="bold",
                        family="DejaVu Sans", va="center", ha="left",
                        zorder=2)
                ax.text(col_x["qty"], py(row_y_px), _fmt_qty(qty),
                        color=_INK, fontsize=11, family="DejaVu Sans",
                        va="center", ha="right", zorder=2)
                ax.text(col_x["avg"], py(row_y_px), _fmt_dollars(avg_cost),
                        color=_INK, fontsize=11, family="DejaVu Sans",
                        va="center", ha="right", zorder=2)
                if mark is None:
                    ax.text(col_x["mark"], py(row_y_px), "—",
                            color=_AMBER, fontsize=11, family="DejaVu Sans",
                            va="center", ha="right", zorder=2)
                    ax.text(col_x["pl"], py(row_y_px), "no quote",
                            color=_AMBER, fontsize=10,
                            family="DejaVu Sans",
                            va="center", ha="right", zorder=2)
                    ax.text(col_x["pct"], py(row_y_px), "—",
                            color=_AMBER, fontsize=11,
                            family="DejaVu Sans",
                            va="center", ha="right", zorder=2)
                else:
                    ax.text(col_x["mark"], py(row_y_px),
                            _fmt_dollars(float(mark)),
                            color=_INK, fontsize=11, family="DejaVu Sans",
                            va="center", ha="right", zorder=2)
                    ax.text(col_x["pl"], py(row_y_px),
                            f"{row_arrow} {_fmt_dollars(upnl)}",
                            color=row_color, fontsize=11,
                            fontweight="bold",
                            family="DejaVu Sans",
                            va="center", ha="right", zorder=2)
                    ax.text(col_x["pct"], py(row_y_px), _fmt_pct(upct),
                            color=row_color, fontsize=11,
                            family="DejaVu Sans",
                            va="center", ha="right", zorder=2)

                row_y_px += _ROW_HEIGHT_PX

        # ---- Options positions ---------------------------------------
        # Compact one-line-per-contract section. The OCC-derived
        # friendly name (e.g. "AAPL $175C 2026-06-20") needs the full
        # row width, so we don't fight the equity columns — render as
        # a single justified row with the value/P&L right-anchored.
        options_block_y_px = (
            (header_y_px + 38 + max(1, n_positions) * _ROW_HEIGHT_PX)
            if n_positions
            else (header_y_px + 38 + _ROW_HEIGHT_PX if (not options_positions) else header_y_px + 38)
        )
        if options_positions:
            opt_y_px = options_block_y_px + 14
            ax.text(
                0.04, py(opt_y_px),
                f"OPTIONS ({n_options})",
                color=_DIM, fontsize=10, fontweight="bold",
                family="DejaVu Sans", va="center", ha="left", zorder=2,
            )
            ax.plot(
                [0.04, 0.96],
                [py(opt_y_px + 14), py(opt_y_px + 14)],
                color=_GRID, linewidth=0.8, zorder=1,
            )
            opt_row_y_px = opt_y_px + 32
            for op in options_positions:
                friendly = str(op.get("friendly") or op.get("contract") or "")
                qty = float(op.get("qty") or 0.0)
                avg_prem = float(op.get("avg_premium") or 0.0)
                mark_prem = op.get("mark_premium")
                upnl = float(op.get("unrealized_pnl") or 0.0)
                upct = float(op.get("unrealized_pct") or 0.0)
                row_color = _GREEN if upnl >= 0 else _RED
                row_arrow = "▲" if upnl >= 0 else "▼"
                # Left: contract name + qty.
                ax.text(
                    0.04, py(opt_row_y_px), friendly,
                    color=_INK, fontsize=11, fontweight="bold",
                    family="DejaVu Sans", va="center", ha="left",
                    zorder=2,
                )
                ax.text(
                    0.55, py(opt_row_y_px),
                    f"×{_fmt_qty(qty)}  "
                    f"${avg_prem:.2f} → "
                    f"{'—' if mark_prem is None else f'${float(mark_prem):.2f}'}",
                    color=(_AMBER if mark_prem is None else _INK),
                    fontsize=10, family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                # Right: P/L + percent.
                if mark_prem is None:
                    ax.text(
                        0.96, py(opt_row_y_px), "no quote",
                        color=_AMBER, fontsize=10, family="DejaVu Sans",
                        va="center", ha="right", zorder=2,
                    )
                else:
                    ax.text(
                        0.96, py(opt_row_y_px),
                        f"{row_arrow} {_fmt_dollars(upnl)} ({_fmt_pct(upct)})",
                        color=row_color, fontsize=11, fontweight="bold",
                        family="DejaVu Sans", va="center", ha="right",
                        zorder=2,
                    )
                opt_row_y_px += _OPTION_ROW_PX

        # ---- Pending orders -------------------------------------------
        # Only rendered when there are orders. Without this section the
        # card stays compact for the common "no orders" case.
        if pending_orders:
            # Section gap + heading. Anchor depends on whether options
            # block consumed space above us.
            orders_y_px = (
                (header_y_px + 38 + max(1, n_positions) * _ROW_HEIGHT_PX)
                if n_positions
                else (header_y_px + 38 + _ROW_HEIGHT_PX)
            )
            if options_positions:
                orders_y_px += _OPTION_HEADER_PX + n_options * _OPTION_ROW_PX
            orders_y_px += 14  # breathing room above the heading
            ax.text(
                0.04, py(orders_y_px),
                f"OPEN ORDERS ({len(pending_orders)})",
                color=_DIM,
                fontsize=10,
                fontweight="bold",
                family="DejaVu Sans",
                va="center",
                ha="left",
                zorder=2,
            )
            # Divider under the heading.
            ax.plot(
                [0.04, 0.96],
                [py(orders_y_px + 14), py(orders_y_px + 14)],
                color=_GRID, linewidth=0.8, zorder=1,
            )

            order_row_y_px = orders_y_px + 32
            for o in pending_orders:
                # Per-row arrow encodes trigger direction so the eye can
                # scan "is this protecting a long or chasing a breakout"
                # without parsing the side+kind text.
                direction = o.get("trigger_direction") or "below"
                row_arrow = "▲" if direction == "above" else "▼"
                row_color = _GREEN if direction == "above" else _RED
                # Subtle: sells going DOWN (stops) get red, buys going
                # UP (breakouts) get green. Limit-buys-on-dip get red,
                # limit-sells-up get green. The arrow alone carries
                # direction; the colour just reinforces.

                side = (o.get("side") or "?").lower()
                kind = (o.get("kind") or "?").lower()
                ticker = o.get("ticker") or "?"
                trigger = float(o.get("trigger_price") or 0.0)

                if o.get("close_position"):
                    size_part = "close all"
                elif o.get("dollars") is not None:
                    size_part = (
                        f"{_fmt_dollars(float(o['dollars']))} → qty TBD"
                    )
                elif o.get("qty") is not None:
                    size_part = f"{_fmt_qty(float(o['qty']))} sh"
                else:
                    size_part = "?"

                # Row layout (left to right):
                #   #ID  ▲ kind side ticker @ trigger  ·  size  · reason
                ax.text(
                    0.04, py(order_row_y_px),
                    f"#{o.get('id')}",
                    color=_DIM, fontsize=10, fontweight="bold",
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                ax.text(
                    0.10, py(order_row_y_px),
                    row_arrow,
                    color=row_color, fontsize=12,
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                ax.text(
                    0.13, py(order_row_y_px),
                    f"{kind} {side}",
                    color=_INK, fontsize=11,
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                ax.text(
                    0.27, py(order_row_y_px),
                    ticker,
                    color=_INK, fontsize=11, fontweight="bold",
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                ax.text(
                    0.36, py(order_row_y_px),
                    f"@ {_fmt_dollars(trigger)}",
                    color=_INK, fontsize=11,
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                ax.text(
                    0.52, py(order_row_y_px),
                    size_part,
                    color=_DIM, fontsize=10,
                    family="DejaVu Sans",
                    va="center", ha="left", zorder=2,
                )
                # Reason — truncate hard so a chatty reason doesn't
                # bleed off the right edge of the card.
                reason = (o.get("reason") or "").strip()
                if reason:
                    if len(reason) > 38:
                        reason = reason[:36] + "…"
                    ax.text(
                        0.96, py(order_row_y_px),
                        reason,
                        color=_DIM, fontsize=9, fontstyle="italic",
                        family="DejaVu Sans",
                        va="center", ha="right", zorder=2,
                    )
                order_row_y_px += _ORDER_ROW_PX

        # Watermark — same convention as the chart generator so the two
        # image kinds share branding placement.
        ax.text(
            0.985, 0.018,
            bot_name,
            color="#3A3B42",
            fontsize=8,
            family="DejaVu Sans",
            va="bottom",
            ha="right",
            zorder=2,
        )

        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=_DPI,
            facecolor=_BG,
            edgecolor="none",
            bbox_inches=None,
            pad_inches=0,
        )
    finally:
        if fig is not None:
            plt.close(fig)
        else:
            plt.close("all")
        rc_ctx.__exit__(None, None, None)

    buf.seek(0)
    encoded = base64.b64encode(buf.read()).decode("utf-8")
    logger.debug(
        f"Rendered portfolio image: {len(positions)} positions, "
        f"{len(options_positions)} options, {len(encoded)} b64 bytes"
    )
    return encoded
