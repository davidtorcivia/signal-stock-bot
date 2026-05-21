"""
Trading cron worker — gives Sigil scheduled decision points to trade.

Three weekday firings in ET:
  09:45  open    (post-open settle, look at overnight news)
  12:30  midday  (mid-session check-in)
  15:30  close   (final adjustments before bell)

Each firing builds a synthetic ask_command call with a slot-specific
prompt. The writer LLM has the full tool kit (research + portfolio
tools) and decides whether to trade. Output is posted to the chat
only when the writer actually moved (buy/sell/place_order/
cancel_order); silent sit-outs don't post — checked via
`ctx.portfolio_mutation_count` after the ask returns.

Idempotency: `portfolio_cron_runs` row per (context_key, slot) is
stamped after each successful fire; the worker checks the ET-day
window before firing to avoid double-posting across restarts.

Errors during a single context's firing don't kill the worker —
logged and skipped. Same shape as predictions_resolver.run_forever.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Optional
from zoneinfo import ZoneInfo

from .commands.base import CommandContext
from .paper_portfolio import parse_portfolio_key
from .paper_portfolio import PortfolioStore, SOURCE_CRON

logger = logging.getLogger(__name__)


_ET = ZoneInfo("America/New_York")


# (slot_name, hour, minute) — keep these stable; they're the primary key
# in `portfolio_cron_runs`.
_TRADING_SLOTS = (
    ("open", 9, 45),
    ("midday", 12, 30),
    ("close", 15, 30),
)


_OPEN_PROMPT = (
    "It's 9:45 ET — first trading window of the day for your paper "
    "portfolio.\n\n"
    "Open routine:\n"
    "  1. `portfolio_journal_read` — read your last few entries so "
    "     you know what you were thinking yesterday and which "
    "     theses are still live.\n"
    "  2. `portfolio_status` — see current positions, cash, and any "
    "     pending stop/limit orders.\n"
    "  3. Look at overnight news / pre-market moves on names you "
    "     hold and tickers the chat has been talking about.\n"
    "  4. Decide. Use `portfolio_buy` / `portfolio_sell` for "
    "     immediate moves. After a new entry, strongly consider "
    "     `portfolio_place_order` for a protective stop-loss "
    "     (sell+stop) below your invalidation level — that's how "
    "     stops actually work as automatic protection. Set "
    "     take-profit limits (sell+limit) when you have a clear "
    "     target. Options are also available: "
    "     `portfolio_options_chain` to discover contracts, "
    "     `portfolio_buy_option` for long calls/puts, "
    "     `portfolio_place_option_order` for premium-based stops/"
    "     limits on contracts you hold. ITM options auto-cash-"
    "     settle at expiration — no need to close before, but you "
    "     can lock in remaining time value with "
    "     `portfolio_sell_option`.\n"
    "  5. `portfolio_journal_append` — write a short entry: "
    "     today's plan, your active theses, what would invalidate "
    "     them. Future-you will thank present-you.\n\n"
    "You're not obligated to trade every cron — quality over volume. "
    "Always include a one-sentence `reason` on every trade and order; "
    "the chat reads those. If you do trade, end your message with what "
    "you did and why. If you decide to sit out (no buys, sells, or "
    "orders), don't post anything to the chat — your reply is "
    "suppressed automatically when no moves are made, so don't waste "
    "tokens writing a justification no one will read."
)

_MIDDAY_PROMPT = (
    "It's 12:30 ET — midday check-in for your paper portfolio.\n\n"
    "Routine:\n"
    "  1. `portfolio_status` — including any pending orders. Did "
    "     anything fire? Anything close to a trigger?\n"
    "  2. Are positions still aligned with the thesis you opened "
    "     them on? News that flips the setup?\n"
    "  3. Adjust if you have a real edge: tighten stops on winners, "
    "     cancel orders that no longer make sense via "
    "     `portfolio_cancel_order`, place new ones if the setup "
    "     calls for it. Sit if there's nothing.\n"
    "  4. Journal only if something material changed in your "
    "     thinking — not every routine check. Quality over volume.\n\n"
    "If you trade, post 1-3 brief sentences leading with what you "
    "did and the why. If you sit out (no buys, sells, or orders), "
    "stay silent — the chat post is suppressed automatically when "
    "no moves are made, so don't write a justification."
)

_CLOSE_PROMPT = (
    "It's 15:30 ET — 30 minutes to the close, your last trading window "
    "today.\n\n"
    "Routine:\n"
    "  1. `portfolio_status` — final state for the day.\n"
    "  2. Decide on close adjustments: trim winners, cut losers, "
    "     reset stops for the overnight gap, or hold. Tomorrow's "
    "     opens get judged against tonight's close, so this is the "
    "     right time to position. If you hold options, check their "
    "     expiration dates — anything expiring this week needs an "
    "     active decision (close at premium, roll, or let "
    "     auto-settle if the math is right).\n"
    "  3. `portfolio_journal_append` — end-of-day reflection. What "
    "     worked, what didn't, what patterns you noticed in the "
    "     tape. The journal is a real trading tool; this is the "
    "     window where it earns its keep.\n\n"
    "If you make adjustments, post a brief 1-3 sentence end-of-day "
    "take for the chat — your journal entry is separate (private, "
    "paragraph-style). If you sit out (no buys, sells, or orders), "
    "stay silent — the chat post is suppressed automatically when "
    "no moves are made, so don't write a justification. The journal "
    "entry can still go in regardless."
)


_PROMPTS = {
    "open": _OPEN_PROMPT,
    "midday": _MIDDAY_PROMPT,
    "close": _CLOSE_PROMPT,
}


_HEADERS = {
    "open": "🔔 9:45 ET — open trade window",
    "midday": "🔔 12:30 ET — midday window",
    "close": "🔔 15:30 ET — close window",
}


def _is_weekday(now_et: dt.datetime) -> bool:
    return now_et.weekday() < 5  # Mon..Fri


class TradingCronWorker:
    """Long-lived async task that fires trading slots per portfolio."""

    POLL_INTERVAL_SECONDS = 60
    # Tolerate restarts up to this many seconds late: if the bot was
    # down at 9:45 and comes back at 9:48, still fire the open slot.
    LATE_FIRE_GRACE_SECONDS = 600

    def __init__(
        self,
        *,
        store: PortfolioStore,
        ask_command,
        signal_pool=None,
        signal_handler=None,
        context_registry,
        bot_phone: str = "",
        bot_registry=None,
    ):
        self.store = store
        self.ask = ask_command
        self.signal_pool = signal_pool
        self.signal = signal_handler or (
            signal_pool.default() if signal_pool is not None else None
        )
        self.contexts = context_registry
        self.bot_phone = bot_phone or (
            signal_pool.default_phone if signal_pool is not None else ""
        )
        self.bot_registry = bot_registry

    def _resolve_bot_for(self, policy):
        """Resolve the bot whose voice this cron post should speak in.
        Per-context pin first, then registry default-for-group. Trading
        cron only runs in groups."""
        if self.bot_registry is None:
            return None
        pinned = getattr(policy, "default_bot_id", None)
        if pinned is not None:
            bot = self.bot_registry.get_sync(pinned)
            if bot and bot.enabled:
                return bot
        return self.bot_registry.default_for_kind_sync("group")

    async def run_forever(self) -> None:
        logger.info(
            f"Trading cron worker started (slots: "
            f"{[s[0] for s in _TRADING_SLOTS]})"
        )
        while True:
            try:
                await self.tick()
            except asyncio.CancelledError:
                logger.info("Trading cron worker cancelled")
                raise
            except Exception as e:
                logger.exception(f"Trading cron tick failed: {e}")
            await asyncio.sleep(self.POLL_INTERVAL_SECONDS)

    async def tick(self) -> None:
        now_et = dt.datetime.now(_ET)
        if not _is_weekday(now_et):
            return

        # Find slots whose firing window is currently open. The window
        # starts at the slot's clock time and stays open for
        # LATE_FIRE_GRACE_SECONDS so a restarted bot still catches up.
        active_slot = self._slot_in_window(now_et)
        if active_slot is None:
            return

        portfolio_keys = await self.store.list_portfolio_keys()
        if not portfolio_keys:
            return

        for ctx_key in portfolio_keys:
            try:
                if await self.store.cron_fired_today(ctx_key, active_slot):
                    continue
                fired = await self._fire_for_context(ctx_key, active_slot)
                if fired:
                    await self.store.mark_cron_fired(ctx_key, active_slot)
            except Exception as e:
                logger.exception(
                    f"Trading cron: {ctx_key} {active_slot} failed: {e}"
                )

    def _slot_in_window(self, now_et: dt.datetime) -> Optional[str]:
        for name, h, m in _TRADING_SLOTS:
            slot_dt = now_et.replace(hour=h, minute=m, second=0, microsecond=0)
            delta = (now_et - slot_dt).total_seconds()
            if 0 <= delta <= self.LATE_FIRE_GRACE_SECONDS:
                return name
        return None

    async def _fire_for_context(self, ctx_key: str, slot: str) -> bool:
        """Run one firing for one context+slot. Returns True if a post
        actually went out (so the caller can stamp `mark_cron_fired`)."""
        # Unwrap per-bot scope so group_id routing + policy lookup
        # operate on the raw chat key. The embedded bot_id (when
        # present) is used below to route the synthetic !ask through
        # the correct bot — each bot has its own portfolio + cron arc
        # in a multi-bot group.
        chat_key, scope_bot_id = parse_portfolio_key(ctx_key)
        if not chat_key.startswith("group:"):
            logger.info(
                f"Trading cron: {ctx_key} is not a group context; skipping"
            )
            return False
        group_id = chat_key[len("group:"):]

        policy = None
        if self.contexts is not None:
            try:
                policy = await self.contexts.get_by_key(group_id)
            except Exception as e:
                logger.warning(
                    f"Trading cron: policy lookup failed for {ctx_key}: {e}"
                )
                return False
        # Defensive: skip when we can't confirm the chat is opted in.
        # A registered group with `portfolio` allowed reaches this point
        # with a real ContextPolicy. Anything else (no registry wired,
        # unregistered group, or disallowed) bails — the cron should
        # never trade in a chat we can't positively identify as having
        # opted in via policy.
        if policy is None or not policy.allows_command("portfolio"):
            logger.info(
                f"Trading cron: {ctx_key} not opted in (policy={policy!r}); "
                f"skipping"
            )
            return False

        prompt = _PROMPTS.get(slot)
        if prompt is None:
            logger.warning(f"Trading cron: unknown slot {slot!r}")
            return False

        if self.ask is None or self.signal is None:
            logger.warning(
                "Trading cron: ask_command or signal_handler not wired"
            )
            return False

        # Scoped-key bot_id wins so each bot in a multi-bot group runs
        # its own cron arc with its own portfolio.
        cron_bot = None
        if scope_bot_id is not None and self.bot_registry is not None:
            try:
                cron_bot = self.bot_registry.get_sync(scope_bot_id)
            except Exception:
                cron_bot = None
        if cron_bot is None:
            cron_bot = self._resolve_bot_for(policy)
        cron_phone = (
            (cron_bot.signal_phone if cron_bot else None)
            or self.bot_phone
            or ""
        )
        ctx = CommandContext(
            sender=cron_phone,
            group_id=group_id,
            raw_message=f"!ask {prompt}",
            command="ask",
            args=[prompt],
            policy=policy,
            bot=cron_bot,
            # Tag the trade source so portfolio tool calls inside this
            # ask invocation get logged as cron-driven, not reactive.
            automation_source=SOURCE_CRON,
        )
        try:
            result = await self.ask.execute(ctx)
        except Exception as e:
            logger.exception(
                f"Trading cron: ask failed for {ctx_key} {slot}: {e}"
            )
            return False
        if not result or not result.success:
            logger.warning(
                f"Trading cron: ask returned unsuccessful for {ctx_key} {slot}: "
                f"{getattr(result, 'text', '(none)')!r}"
            )
            return False

        # No buys/sells/orders this cron — stay silent. The writer was
        # told its reply is suppressed in this case, so we skip the chat
        # post entirely. Still return True so the slot is marked fired
        # and we don't re-invoke the LLM on the next tick.
        if ctx.portfolio_mutation_count == 0:
            logger.info(
                f"Trading cron fired silently: {ctx_key} {slot} "
                f"(no portfolio moves)"
            )
            return True

        body = result.text or ""
        header = _HEADERS.get(slot, "🔔 Trading window")
        if not body.startswith(header):
            body = f"{header}\n\n{body}"

        sender_handler = (
            self.signal_pool.for_bot(cron_bot)
            if self.signal_pool is not None
            else self.signal
        )
        if sender_handler is None:
            logger.warning(
                f"Trading cron: no signal handler for {ctx_key} {slot}; skipping"
            )
            return False
        try:
            await sender_handler.send_message(
                recipient="",
                message=body,
                group_id=group_id,
                attachments=result.attachments,
                styled=getattr(result, "styled", False),
            )
        except Exception as e:
            logger.error(
                f"Trading cron: send_message failed for {ctx_key} {slot}: {e}"
            )
            return False

        logger.info(
            f"Trading cron fired: {ctx_key} {slot} "
            f"({len(body)} chars, {ctx.portfolio_mutation_count} moves)"
        )
        return True
