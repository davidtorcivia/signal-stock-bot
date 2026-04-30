"""
Trading cron worker — gives Sigil scheduled decision points to trade.

Three weekday firings in ET:
  09:45  open    (post-open settle, look at overnight news)
  12:30  midday  (mid-session check-in)
  15:30  close   (final adjustments before bell)

Each firing builds a synthetic ask_command call with a slot-specific
prompt. The writer LLM has the full tool kit (research + portfolio
tools) and decides whether to trade. Output is posted to the chat so
members see Sigil's reasoning even when he chooses to sit out.

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
    "     target.\n"
    "  5. `portfolio_journal_append` — write a short entry: "
    "     today's plan, your active theses, what would invalidate "
    "     them. Future-you will thank present-you.\n\n"
    "You're not obligated to trade every cron — quality over volume. "
    "Always include a one-sentence `reason` on every trade and order; "
    "the chat reads those. End your message with what you did and why."
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
    "Brief, conversational post — 1-3 sentences. Lead with what you "
    "did (or didn't), then the why."
)

_CLOSE_PROMPT = (
    "It's 15:30 ET — 30 minutes to the close, your last trading window "
    "today.\n\n"
    "Routine:\n"
    "  1. `portfolio_status` — final state for the day.\n"
    "  2. Decide on close adjustments: trim winners, cut losers, "
    "     reset stops for the overnight gap, or hold. Tomorrow's "
    "     opens get judged against tonight's close, so this is the "
    "     right time to position.\n"
    "  3. `portfolio_journal_append` — end-of-day reflection. What "
    "     worked, what didn't, what patterns you noticed in the "
    "     tape. The journal is a real trading tool; this is the "
    "     window where it earns its keep.\n\n"
    "Brief end-of-day take in 1-3 sentences for the chat after you "
    "act — your journal entry is separate (private, paragraph-style)."
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
        signal_handler,
        context_registry,
        bot_phone: str,
    ):
        self.store = store
        self.ask = ask_command
        self.signal = signal_handler
        self.contexts = context_registry
        self.bot_phone = bot_phone

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
        if not ctx_key.startswith("group:"):
            logger.info(
                f"Trading cron: {ctx_key} is not a group context; skipping"
            )
            return False
        group_id = ctx_key[len("group:"):]

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

        ctx = CommandContext(
            sender=self.bot_phone or "",
            group_id=group_id,
            raw_message=f"!ask {prompt}",
            command="ask",
            args=[prompt],
            policy=policy,
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

        body = result.text or ""
        header = _HEADERS.get(slot, "🔔 Trading window")
        if not body.startswith(header):
            body = f"{header}\n\n{body}"

        try:
            await self.signal.send_message(
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
            f"({len(body)} chars)"
        )
        return True
