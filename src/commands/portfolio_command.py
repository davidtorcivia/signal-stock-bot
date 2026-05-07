"""
Paper-portfolio commands.

  !portfolio          — current holdings + PnL summary
  !tip <amount> [note]— add cash to the bot's pot (per-user daily cap)
  !trades [N]         — recent trades (default 10)
  !pnl                — alias for !portfolio with a numeric focus

The portfolio is per-context (one per chat). Auto-seeded with $1000 on
first interaction. The bot is the only trader; humans contribute via
`!tip`.
"""

from __future__ import annotations

import datetime as dt
import logging
import math

from ..charts.portfolio import render_portfolio_image
from ..database import hash_phone
from ..paper_portfolio import PortfolioStore
from ..paper_portfolio_executor import PaperPortfolioExecutor
from .base import BaseCommand, CommandContext, CommandResult

logger = logging.getLogger(__name__)


def _fmt_dollars(amount: float) -> str:
    sign = "-" if amount < 0 else ""
    return f"{sign}${abs(amount):,.2f}"


def _fmt_pct(pct: float) -> str:
    return f"{pct * 100:+.2f}%"


def _fmt_qty(qty: float) -> str:
    """Compact share count: integers as ints, fractions to 4 places."""
    if abs(qty - round(qty)) < 1e-6:
        return f"{int(round(qty))}"
    return f"{qty:.4f}".rstrip("0").rstrip(".")


def _fmt_ts(ts: float) -> str:
    return dt.datetime.fromtimestamp(ts).strftime("%Y-%m-%d %H:%M")


def render_status_caption(snap: dict, *, bot_name: str = "Sigil") -> str:
    """One-line caption shown in chat alongside the portfolio image.

    The image carries the full layout — equity, cash split, positions
    table — so the visible caption is intentionally tiny. Mirrors the
    headline numbers as plain text so users with image attachments
    disabled still get the punchline.
    """
    label = f" ({snap['label']})" if snap.get("label") else ""
    equity = snap.get("equity", 0.0)
    pnl = snap.get("total_pnl", 0.0)
    pnl_pct = snap.get("total_pnl_pct", 0.0)
    arrow = "▲" if pnl >= 0 else "▼"
    n_positions = len(snap.get("positions") or [])
    n_orders = len(snap.get("pending_orders") or [])
    pos_part = (
        f" · {n_positions} position{'s' if n_positions != 1 else ''}"
        if n_positions else ""
    )
    order_part = (
        f" · {n_orders} pending order{'s' if n_orders != 1 else ''}"
        if n_orders else ""
    )
    return (
        f"◈ {bot_name}'s portfolio{label}: {_fmt_dollars(equity)} "
        f"{arrow} {_fmt_dollars(pnl)} ({_fmt_pct(pnl_pct)}){pos_part}{order_part}"
    )


def _fmt_order_summary(o: dict) -> str:
    """One-line per-order text used in render_status' Pending Orders
    block. Format: ``#42  ▲ stop sell AAPL @ $200.00 — close all``
    The arrow encodes trigger direction (▲ above / ▼ below) so the
    LLM can read intent at a glance without re-deriving from side+kind."""
    arrow = "▲" if o.get("trigger_direction") == "above" else "▼"
    side = o.get("side") or "?"
    kind = o.get("kind") or "?"
    ticker = o.get("ticker") or "?"
    trigger = o.get("trigger_price") or 0.0
    if o.get("close_position"):
        size_part = "close all"
    elif o.get("dollars") is not None:
        size_part = f"{_fmt_dollars(float(o['dollars']))} (qty TBD at fill)"
    elif o.get("qty") is not None:
        size_part = f"{_fmt_qty(float(o['qty']))} sh"
    else:
        size_part = "?"
    reason = (o.get("reason") or "").strip()
    reason_part = f" — {reason}" if reason else ""
    return (
        f"#{o.get('id')}  {arrow} {kind} {side} {ticker} "
        f"@ {_fmt_dollars(trigger)} · {size_part}{reason_part}"
    )


def render_status(snap: dict, *, bot_name: str = "Sigil") -> str:
    """Plain-text portfolio summary. The chat path now sends an image
    instead, but this function is still used as the LLM tool result for
    `portfolio_status` (so the model can reason about the data without
    needing vision) and as a fallback if image rendering fails."""
    lines = [f"◈ {bot_name}'s portfolio"]
    if snap.get("label"):
        lines[0] += f" ({snap['label']})"

    funded = snap["total_funded"]
    equity = snap["equity"]
    pnl = snap["total_pnl"]
    pnl_pct = snap["total_pnl_pct"]
    arrow = "▲" if pnl >= 0 else "▼"

    lines.append(
        f"  Equity: {_fmt_dollars(equity)}  {arrow} "
        f"{_fmt_dollars(pnl)} ({_fmt_pct(pnl_pct)})"
    )
    lines.append(
        f"  Cash: {_fmt_dollars(snap['cash'])} | "
        f"Holdings: {_fmt_dollars(snap['market_value'])} | "
        f"Funded: {_fmt_dollars(funded)} "
        f"(seed {_fmt_dollars(snap['starting_balance'])} + "
        f"tips {_fmt_dollars(snap['tip_total'])})"
    )
    realized = snap["realized_pnl"]
    if abs(realized) > 0.005:
        lines.append(f"  Realized: {_fmt_dollars(realized)}")

    if not snap["market_open"]:
        lines.append("  (market closed — quotes may be stale)")

    positions = snap["positions"]
    if positions:
        lines.append("")
        lines.append("Positions:")
        for p in positions:
            mark = (
                _fmt_dollars(p["mark_price"])
                if p["mark_price"] is not None
                else "no quote"
            )
            arrow = "▲" if p["unrealized_pnl"] >= 0 else "▼"
            lines.append(
                f"  {p['ticker']:<6} {_fmt_qty(p['qty']):>10} @ "
                f"{_fmt_dollars(p['avg_cost'])} → {mark}  "
                f"{arrow} {_fmt_dollars(p['unrealized_pnl'])} "
                f"({_fmt_pct(p['unrealized_pct'])})"
            )
    else:
        lines.append("  (no open positions)")

    # Pending orders — surfaced here so the LLM sees them when calling
    # portfolio_status (it gets the order ids it needs to cancel or
    # reference). Empty list collapses to nothing rather than printing
    # an empty section header — keeps the common "no orders" case
    # quiet.
    pending_orders = snap.get("pending_orders") or []
    if pending_orders:
        lines.append("")
        lines.append(f"Pending orders ({len(pending_orders)}):")
        for o in pending_orders:
            lines.append(f"  {_fmt_order_summary(o)}")

    return "\n".join(lines)


class PortfolioCommand(BaseCommand):
    name = "portfolio"
    aliases = ["pf"]
    description = "Show the bot's paper-trading portfolio."
    usage = "!portfolio"
    help_explanation = (
        "Shows the bot's current paper-trading positions, cash, and "
        "PnL for this chat. The portfolio auto-seeds with $1000; "
        "members can add to it with `!tip <amount>` (capped at $20 per "
        "user per ET day). The bot decides trades on a cron schedule and "
        "in response to chat."
    )

    def __init__(
        self,
        executor: PaperPortfolioExecutor,
        *,
        bot_name: str = "Sigil",
    ):
        self.executor = executor
        self.bot_name = bot_name

    async def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            snap = await self.executor.status(ctx.context_key())
        except Exception as e:
            logger.exception(f"portfolio: status failed: {e}")
            return CommandResult.error("Couldn't load the portfolio.")

        caption = render_status_caption(snap, bot_name=self.bot_name)
        # Beautiful PNG dashboard is the primary surface in chat. If
        # rendering fails (matplotlib hiccup, weird snap shape) fall
        # back to the legacy plain-text layout so the user still gets
        # their portfolio rather than a generic error.
        try:
            image_b64 = render_portfolio_image(snap, bot_name=self.bot_name)
        except Exception as e:
            logger.warning(f"portfolio: image render failed, falling back to text: {e}")
            return CommandResult.ok(render_status(snap, bot_name=self.bot_name))

        return CommandResult(
            text=caption,
            success=True,
            attachments=[image_b64],
        )


class PnlCommand(BaseCommand):
    name = "pnl"
    aliases = ["performance"]
    description = "Show the bot's portfolio PnL."
    usage = "!pnl"
    help_explanation = (
        "Same data as !portfolio, focused on the gain/loss numbers."
    )

    def __init__(
        self,
        executor: PaperPortfolioExecutor,
        *,
        bot_name: str = "Sigil",
    ):
        self.executor = executor
        self.bot_name = bot_name

    async def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            snap = await self.executor.status(ctx.context_key())
        except Exception as e:
            logger.exception(f"pnl: status failed: {e}")
            return CommandResult.error("Couldn't load the portfolio.")

        funded = snap["total_funded"]
        equity = snap["equity"]
        pnl = snap["total_pnl"]
        pnl_pct = snap["total_pnl_pct"]
        arrow = "▲" if pnl >= 0 else "▼"
        lines = [
            f"◈ {self.bot_name}'s PnL",
            f"  {arrow} {_fmt_dollars(pnl)} ({_fmt_pct(pnl_pct)})",
            f"  Equity: {_fmt_dollars(equity)}  Funded: {_fmt_dollars(funded)}",
            f"  Realized: {_fmt_dollars(snap['realized_pnl'])}  "
            f"Unrealized: {_fmt_dollars(snap['unrealized_pnl'])}",
        ]
        return CommandResult.ok("\n".join(lines))


class TipCommand(BaseCommand):
    name = "tip"
    aliases = ["fund"]
    description = "Tip the bot cash for its paper portfolio."
    usage = "!tip <amount> [reason]"
    help_explanation = (
        "Adds cash to the bot's paper-trading portfolio for this chat. "
        "Cap: $20 per user per ET day. Example: `!tip 5 nice call on TSLA`."
    )

    def __init__(
        self,
        store: PortfolioStore,
        *,
        name_registry=None,
    ):
        self.store = store
        self.name_registry = name_registry

    def _label(self, phone: str) -> str:
        if self.name_registry is not None:
            return self.name_registry.label_for(phone)
        return f"...{(phone or '')[-4:]}"

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                f"Usage: {self.usage}\nDaily cap: "
                f"${self.store.daily_tip_cap:.0f} per user (ET day)."
            )
        raw_amount = ctx.args[0].lstrip("$").rstrip(".")
        try:
            amount = float(raw_amount)
        except ValueError:
            return CommandResult.error(
                f"`{ctx.args[0]}` isn't a dollar amount. "
                f"Try `!tip 5` or `!tip 10.50`."
            )
        if not math.isfinite(amount):
            return CommandResult.error(
                "Amount must be a real finite number."
            )
        if amount <= 0:
            return CommandResult.error("Amount must be positive.")

        note = " ".join(ctx.args[1:]).strip() or None
        tipper_hash = hash_phone(ctx.sender)
        tipper_label = self._label(ctx.sender)

        try:
            result = await self.store.tip(
                ctx.context_key(),
                tipper_user_hash=tipper_hash,
                tipper_label=tipper_label,
                amount=amount,
                note=note,
            )
        except Exception as e:
            logger.exception(f"tip: store.tip failed: {e}")
            return CommandResult.error("Couldn't record the tip.")

        if not result.get("ok"):
            reason = result.get("reason")
            if reason == "cap_exceeded":
                return CommandResult.error(
                    f"Daily tip cap is ${result['cap']:.0f} per person — "
                    f"you've already tipped ${result['today_total']:.2f} "
                    f"today (${result['remaining_today']:.2f} remaining). "
                    f"Resets at midnight ET."
                )
            return CommandResult.error("Tip rejected.")

        remaining = result["remaining_today"]
        remaining_part = (
            f"  ${remaining:.2f} of your daily cap left."
            if remaining > 0 else
            "  You've hit today's cap."
        )
        note_part = f" ({note})" if note else ""
        return CommandResult.ok(
            f"✓ {tipper_label} tipped {_fmt_dollars(amount)}{note_part}.\n"
            f"  Pot: {_fmt_dollars(result['cash_after'])}.\n"
            f"{remaining_part}"
        )


class TradesCommand(BaseCommand):
    name = "trades"
    aliases = ["history"]
    description = "Recent trades by the bot's paper portfolio."
    usage = "!trades [count]"
    help_explanation = (
        "Lists the most recent paper trades (default 10, max 30). Each "
        "row shows side, qty, ticker, fill price, and the bot's reason."
    )

    def __init__(self, store: PortfolioStore):
        self.store = store

    async def execute(self, ctx: CommandContext) -> CommandResult:
        limit = 10
        if ctx.args:
            try:
                limit = max(1, min(30, int(ctx.args[0])))
            except ValueError:
                return CommandResult.error(
                    f"`{ctx.args[0]}` isn't a number. Try `!trades 20`."
                )

        try:
            trades = await self.store.list_trades(ctx.context_key(), limit=limit)
        except Exception as e:
            logger.exception(f"trades: list_trades failed: {e}")
            return CommandResult.error("Couldn't load the trade history.")

        if not trades:
            return CommandResult.ok("(no trades yet)")

        lines = [f"◈ Recent trades ({len(trades)})"]
        for t in trades:
            side_arrow = "↑" if t.side == "buy" else "↓"
            pnl_part = ""
            if t.realized_pnl is not None and abs(t.realized_pnl) > 0.005:
                pnl_arrow = "▲" if t.realized_pnl >= 0 else "▼"
                pnl_part = f" {pnl_arrow} {_fmt_dollars(t.realized_pnl)}"
            reason_part = f" — {t.reason}" if t.reason else ""
            lines.append(
                f"  {_fmt_ts(t.ts)}  {side_arrow} {_fmt_qty(t.qty):>8} "
                f"{t.ticker:<6} @ {_fmt_dollars(t.price)}{pnl_part}{reason_part}"
            )
        return CommandResult.ok("\n".join(lines))


# --- LLM-facing tool schemas + handlers ---------------------------------

PORTFOLIO_BUY_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_buy",
        "description": (
            "Buy a stock with YOUR paper-trading account in "
            "this chat. Fills at the live quote during regular US "
            "market hours (9:30-16:00 ET). Pass EITHER `dollars` (and "
            "the executor computes fractional shares) OR `qty` (exact "
            "share count). Always include a `reason` — your one-sentence "
            "thesis. Outside market hours: trade is rejected; wait for "
            "the next session.\n\n"
            "Use this when you've decided to take a position based on "
            "research, news, or chat discussion. Don't trade on every "
            "ticker mention — only when you have a real view. The "
            "portfolio auto-seeds with $1000; members can tip more in "
            "via !tip."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker symbol (e.g. \"AAPL\", \"TSLA\").",
                },
                "dollars": {
                    "type": "number",
                    "description": (
                        "Dollar amount to allocate. Mutually exclusive "
                        "with `qty`."
                    ),
                },
                "qty": {
                    "type": "number",
                    "description": (
                        "Exact share quantity (fractional ok). Mutually "
                        "exclusive with `dollars`."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One-sentence thesis for the trade. Logged with "
                        "the fill so the chat can see WHY you bought."
                    ),
                },
            },
            "required": ["ticker", "reason"],
        },
    },
}


PORTFOLIO_SELL_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_sell",
        "description": (
            "Sell from YOUR paper portfolio in this chat. "
            "Fills at live quote during regular hours. Pass `qty` for "
            "a partial close, or omit / pass `\"all\"` to close the "
            "position entirely. Always include a `reason`.\n\n"
            "V1 is long-only — you can't sell what you don't hold, "
            "and selling more than the position size errors instead "
            "of opening a short."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Ticker to sell.",
                },
                "qty": {
                    # Use anyOf rather than oneOf — broader compatibility
                    # across LLM tool-call validators (some treat oneOf
                    # at a leaf property as ambiguous and discard the
                    # schema). Functionally equivalent here since the two
                    # branches are mutually exclusive.
                    "anyOf": [
                        {"type": "number"},
                        {"type": "string", "enum": ["all"]},
                    ],
                    "description": (
                        "Share count (fractional ok), or \"all\" to "
                        "close the position. Defaults to \"all\" if "
                        "omitted."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": "One-sentence thesis for selling.",
                },
            },
            "required": ["ticker", "reason"],
        },
    },
}


PORTFOLIO_STATUS_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_status",
        "description": (
            "Read YOUR current paper-portfolio state in this "
            "chat. Returns cash, every open position with mark-to-market, "
            "realized + unrealized PnL, total funded (seed + tips). "
            "Call this BEFORE deciding to trade so you know what you "
            "already hold and how much cash is free.\n\n"
            "IMPORTANT: When you call this in response to a user asking "
            "to see the portfolio, a polished portfolio dashboard image "
            "is automatically attached to your reply. Do NOT reproduce "
            "the holdings as a markdown table or monospace list in your "
            "text — the user already sees them in the image, and "
            "Signal renders tables badly. Reference specific positions "
            "or numbers in plain prose if you have something to say "
            "about them; otherwise a brief comment ('here's where we "
            "stand') is enough. The full text data is included in the "
            "tool result so you can reason about it without vision."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
        },
    },
}


PORTFOLIO_PLACE_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_place_order",
        "description": (
            "Register a CONDITIONAL order on YOUR paper "
            "portfolio that fires automatically when the live price "
            "crosses a trigger. The order watcher polls every ~5 min "
            "during US regular hours and fills crossings via the same "
            "execution path as portfolio_buy/sell. The originating chat "
            "gets a notification when the fill happens.\n\n"
            "Use this for stop-losses, breakout entries, take-profit "
            "limits, and pullback buys — anywhere you'd want to act "
            "but don't want to (or can't) wait for the next chat turn. "
            "Pick `kind` and `side` to encode the trigger direction:\n"
            "  - sell + stop  → stop-loss (fires when price ≤ trigger)\n"
            "  - sell + limit → take-profit (fires when price ≥ trigger)\n"
            "  - buy + stop   → breakout entry (fires when price ≥ trigger)\n"
            "  - buy + limit  → pullback entry (fires when price ≤ trigger)\n\n"
            "Sizing: pass exactly ONE of `qty`, `dollars`, or "
            "`close_position`. `dollars` is buy-only (we don't know the "
            "fill price ahead of time so we can't compute shares for a "
            "sell). `close_position=true` is sell-only and means \"sell "
            "whatever the position holds at trigger time\" — ideal for "
            "stop-losses on positions whose size may change before fill. "
            "`reason` is required: one-sentence thesis logged with the "
            "fill. Default expiry is 30 days; pass `expires_in_days` to "
            "override (max 90, min 1)."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "ticker": {
                    "type": "string",
                    "description": "Stock ticker (e.g. \"AAPL\").",
                },
                "side": {
                    "type": "string",
                    "enum": ["buy", "sell"],
                    "description": "Direction of the trade once triggered.",
                },
                "kind": {
                    "type": "string",
                    "enum": ["stop", "limit"],
                    "description": (
                        "Trigger semantics — see top-level description. "
                        "stop = act AGAINST the move (sell on fall, buy "
                        "on rise); limit = act WITH the move (sell on "
                        "rise, buy on fall)."
                    ),
                },
                "trigger_price": {
                    "type": "number",
                    "description": "Price at which the order fires (USD).",
                },
                "qty": {
                    "type": "number",
                    "description": (
                        "Exact share count (fractional ok). Mutually "
                        "exclusive with dollars and close_position."
                    ),
                },
                "dollars": {
                    "type": "number",
                    "description": (
                        "Dollar amount to spend (BUY ONLY). Converted "
                        "to fractional shares at fill price. Mutually "
                        "exclusive with qty and close_position."
                    ),
                },
                "close_position": {
                    "type": "boolean",
                    "description": (
                        "SELL ONLY — sell the entire position at "
                        "trigger time. Mutually exclusive with qty and "
                        "dollars."
                    ),
                },
                "reason": {
                    "type": "string",
                    "description": (
                        "One-sentence thesis. Logged with the fill so "
                        "the chat sees WHY this order existed."
                    ),
                },
                "expires_in_days": {
                    "type": "number",
                    "description": (
                        "Order is auto-cancelled after this many days "
                        "if not triggered. Default 30, min 1, max 90."
                    ),
                },
            },
            "required": ["ticker", "side", "kind", "trigger_price", "reason"],
        },
    },
}


PORTFOLIO_JOURNAL_APPEND_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_journal_append",
        "description": (
            "Add a timestamped entry to YOUR trading journal "
            "for this chat. The journal is your private notebook — a "
            "free-form markdown file where you track the evolution of "
            "your thinking, acknowledge mistakes, document what worked, "
            "and plan ahead. Distinct from the structured `remember` "
            "tool: that's for atomic facts about people / topics; THIS "
            "is for narrative reflection on your own trading.\n\n"
            "Write in your own voice — paragraph-style, conversational, "
            "honest. The journal is for YOU; only you (the bot in "
            "future sessions) and admins ever read it. Chat members do "
            "not see it.\n\n"
            "Strong moments to journal:\n"
            "  - After placing a trade: what's the thesis, what would "
            "    invalidate it, what's the exit plan.\n"
            "  - When an order fires (the order watcher will ping you): "
            "    note whether the thesis worked, what you'd do "
            "    differently next time.\n"
            "  - At the end of a trading session (cron close window): "
            "    summarize the day, flag patterns you noticed.\n"
            "  - When you change your mind about a position or a "
            "    broader market read.\n"
            "Don't journal every tick. Quality over volume. If you "
            "have nothing new to say, skip."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "entry": {
                    "type": "string",
                    "description": (
                        "Free-form markdown text. Multi-paragraph "
                        "encouraged when you have something real to "
                        "say. ~1-3 paragraphs ideal. Hard cap "
                        "~4000 chars (over-long entries are silently "
                        "trimmed)."
                    ),
                },
            },
            "required": ["entry"],
        },
    },
}


PORTFOLIO_JOURNAL_READ_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_journal_read",
        "description": (
            "Read the most recent entries from YOUR trading journal in "
            "this chat. Use this BEFORE checking in on the portfolio "
            "or making a fresh trade — the journal is your record of "
            "what you've been thinking, what's worked, what hasn't. "
            "Recall it the way a real trader would re-read their own "
            "notebook before placing the next order.\n\n"
            "Returns the last N entries by timestamp. Each entry has "
            "a `ts` (UTC timestamp) and a `body` (the markdown text "
            "you wrote). If the journal is empty, returns an empty "
            "list — that's the signal to start writing one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "limit": {
                    "type": "integer",
                    "description": (
                        "How many recent entries to return. Default "
                        "10, max 50. Use a small number for routine "
                        "check-ins; larger when you want to look back "
                        "across a longer arc."
                    ),
                },
            },
        },
    },
}


PORTFOLIO_CANCEL_ORDER_TOOL = {
    "type": "function",
    "function": {
        "name": "portfolio_cancel_order",
        "description": (
            "Cancel one of YOUR pending conditional orders by "
            "id. Order ids are visible in `portfolio_status` under the "
            "Pending orders block. Cancel succeeds only on orders that "
            "are still pending — already-filled / already-cancelled / "
            "expired orders return an error. The cancel is scoped to "
            "this chat, so a stale id from another chat fails."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "order_id": {
                    "type": "integer",
                    "description": "Numeric id of the order to cancel.",
                },
            },
            "required": ["order_id"],
        },
    },
}


# Optional: tipping leaderboard so Sigil can thank top tippers, etc.
# Not exposed as a tool in V1 — keep the surface tight.
