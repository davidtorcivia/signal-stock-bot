"""
Daily oracle worker (multi-oracle, per-context).

Each context (group) can have any number of oracles. The worker
iterates all enabled oracles, computes the soonest firing instant,
sleeps until then, and fires every oracle whose firing window has
opened. Each oracle is daily-cadence; only timing/kind varies.

Oracle kinds + delivery:
  tarot/iching   → call the existing command directly, replace the
                   header with a "today's oracle from Sigil" framing,
                   send the rendered image as an attachment.
  market_open/close, freeform → route through ask_command.execute so
                   Sigil's full tool kit is available (price/news/
                   etc.) and the LLM writes the post.

Idempotency: `last_fired_at` is bumped after each post. On bot
restart, an oracle whose last_fired_at is within today's window for
its schedule is skipped — preventing the "bot crashed at 9:24:50,
restarts at 9:25:30, posts twice" scenario.

All errors are logged and swallowed. A failing oracle never affects
the rest of the worker or the bot.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
from typing import Optional

from .commands.base import CommandContext
from .contexts.oracles import (
    ContextOracle,
    OracleStore,
    fire_time_for,
    next_fire_time,
)

logger = logging.getLogger(__name__)


# Headers from the existing tarot/iching commands that we strip and
# replace with the oracle framing. Keep these in sync if those
# commands ever change their default openers.
_DEFAULT_TAROT_HEADERS = (
    "✦ Your card:\n\n",
    "✦ Your card of the day:\n\n",
    "✦ Your card of the day (drawn earlier — same card all day):\n\n",
    "✦ Past · Present · Future:\n\n",
    "✦ Celtic Cross:\n\n",
)
_DEFAULT_ICHING_HEADERS = (
    "☷ Your hexagram:\n\n",
    "☷ Daily hexagram:\n\n",
    "☷ Yarrow stalks:\n\n",
)


def _replace_header(body: str, oracle_label: str) -> str:
    header = "🌅 Today's oracle from Sigil"
    if oracle_label:
        header = f"{header} — {oracle_label}"
    header += ":"

    for h in (*_DEFAULT_TAROT_HEADERS, *_DEFAULT_ICHING_HEADERS):
        if body.startswith(h):
            return f"{header}\n\n{body[len(h):]}"
    return f"{header}\n\n{body}"


_MARKET_OPEN_PROMPT = (
    "It's just before US cash open. Write a brief pre-market check for "
    "the chat: where futures are pointing for the major indices "
    "(S&P 500 / Nasdaq / Dow), the most important scheduled event for "
    "today (earnings, Fed, CPI, etc., if any), and one specific thing "
    "to watch in the first hour. Keep it 2-4 sentences, conversational, "
    "no preamble. Use bot__price for futures (^GSPC, ^IXIC, ^DJI) and "
    "bot__news / web search for the day's events."
)

_MARKET_CLOSE_PROMPT = (
    "US markets just closed. Write a brief recap for the chat: how the "
    "major indices finished (S&P / Nasdaq / Dow with percent moves), "
    "one or two notable leaders or laggards, and any meaningful news "
    "that drove the tape. Keep it 2-4 sentences, conversational, no "
    "preamble. Use bot__price for indices and major movers; bot__news "
    "for context."
)


class DailyOracleWorker:
    """Long-lived async task that fires per-oracle on each oracle's schedule."""

    # Polling cadence when no enabled oracles exist — admin can flip
    # one on via the UI and the worker picks it up within 5 min.
    _IDLE_RECHECK_SECONDS = 300

    # Window after fire time during which we'll still post (handles
    # bot restarts that land a few seconds late). Anything later than
    # this is treated as a missed firing — wait for tomorrow.
    _LATE_FIRE_GRACE_SECONDS = 600  # 10 minutes

    def __init__(
        self,
        *,
        oracle_store: OracleStore,
        context_registry,
        tarot_command,
        iching_command,
        ask_command,
        signal_handler,
        bot_phone: str,
    ):
        self.store = oracle_store
        self.contexts = context_registry
        self.tarot = tarot_command
        self.iching = iching_command
        self.ask = ask_command
        self.signal = signal_handler
        self.bot_phone = bot_phone

    async def run_forever(self) -> None:
        logger.info("Daily oracle worker started")
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                logger.info("Daily oracle worker cancelled")
                raise
            except Exception as e:
                logger.exception(f"Oracle worker tick failed: {e}")
                await asyncio.sleep(60)

    async def _tick(self) -> None:
        oracles = await self.store.list_enabled()
        now = dt.datetime.now(dt.timezone.utc)
        fireable = [o for o in oracles if not self._already_fired_today(o, now)]
        if not fireable:
            await asyncio.sleep(self._IDLE_RECHECK_SECONDS)
            return

        scheduled = [(next_fire_time(o, now), o) for o in fireable]
        scheduled.sort(key=lambda x: x[0])
        next_at, _ = scheduled[0]

        wait = max(0.0, (next_at - now).total_seconds())
        logger.info(
            f"Oracle worker: {len(fireable)} enabled, next at "
            f"{next_at.isoformat()} (in {wait/3600:.2f}h)"
        )
        # Cap the sleep at the idle recheck so newly-enabled oracles get
        # picked up within ~5 min instead of waiting until the previously-
        # nearest fire time.
        await asyncio.sleep(min(wait, float(self._IDLE_RECHECK_SECONDS)))

        # Re-list so any admin edits during the sleep apply to this firing
        # pass (e.g. enabled flipped off in the last few minutes).
        fresh = await self.store.list_enabled()
        now = dt.datetime.now(dt.timezone.utc)
        for oracle in fresh:
            if self._already_fired_today(oracle, now):
                continue
            fire_at = next_fire_time(oracle, now - dt.timedelta(hours=1))
            # If the next fire (computed from "an hour ago") is in the
            # past or within the grace window, fire it now. Otherwise
            # leave for the next tick.
            delta = (now - fire_at).total_seconds()
            if 0 <= delta <= self._LATE_FIRE_GRACE_SECONDS:
                await self._fire_oracle(oracle)

    @staticmethod
    def _already_fired_today(
        oracle: ContextOracle, now: dt.datetime
    ) -> bool:
        """True if the oracle's last_fired_at is within today's window
        in its local schedule frame. Prevents double-fire across bot
        restarts that land in the same minute."""
        if not oracle.last_fired_at:
            return False
        try:
            fire = fire_time_for(oracle, now.astimezone(dt.timezone.utc).date())
        except Exception:
            return False
        last = dt.datetime.fromtimestamp(oracle.last_fired_at, tz=dt.timezone.utc)
        # If last_fired is within ±12h of today's fire window, we already
        # posted. Wide window because clock-mode oracles in distant tz
        # can land on a different UTC date than the bot.
        return abs((last - fire).total_seconds()) <= 12 * 3600

    async def _fire_oracle(self, oracle: ContextOracle) -> None:
        ctx_policy = await self._lookup_context_for_oracle(oracle)
        if ctx_policy is None:
            logger.warning(
                f"Oracle #{oracle.id}: context_id {oracle.context_id} "
                f"not found, skipping"
            )
            return
        if ctx_policy.kind != "group":
            logger.warning(
                f"Oracle #{oracle.id}: context #{ctx_policy.id} is not a "
                f"group ({ctx_policy.kind}), skipping"
            )
            return
        group_id = ctx_policy.key

        try:
            if oracle.kind == "tarot":
                await self._fire_tarot(oracle, ctx_policy, group_id)
            elif oracle.kind == "iching":
                await self._fire_iching(oracle, ctx_policy, group_id)
            elif oracle.kind in ("market_open", "market_close", "freeform"):
                await self._fire_llm_oracle(oracle, ctx_policy, group_id)
            else:
                logger.warning(
                    f"Oracle #{oracle.id}: unknown kind {oracle.kind!r}"
                )
                return
        except Exception as e:
            logger.exception(f"Oracle #{oracle.id} fire failed: {e}")
            return

        await self.store.mark_fired(oracle.id, dt.datetime.now(dt.timezone.utc).timestamp())

    async def _lookup_context_for_oracle(self, oracle: ContextOracle):
        if self.contexts is None:
            return None
        return await self.contexts.get(oracle.context_id)

    async def _fire_tarot(self, oracle, ctx_policy, group_id) -> None:
        ctx = self._synth_ctx(group_id, ctx_policy, "!tarot", "tarot")
        result = await self.tarot.execute(ctx)
        await self._post_command_result(result, oracle, group_id)

    async def _fire_iching(self, oracle, ctx_policy, group_id) -> None:
        ctx = self._synth_ctx(group_id, ctx_policy, "!iching", "iching")
        result = await self.iching.execute(ctx)
        await self._post_command_result(result, oracle, group_id)

    async def _fire_llm_oracle(self, oracle, ctx_policy, group_id) -> None:
        if self.ask is None:
            logger.warning(
                f"Oracle #{oracle.id}: ask_command not wired, skipping"
            )
            return
        prompt = self._llm_prompt_for(oracle)
        ctx = self._synth_ctx(
            group_id, ctx_policy, f"!ask {prompt}", "ask",
            args=[prompt],
        )
        result = await self.ask.execute(ctx)
        if not result or not result.success:
            logger.warning(
                f"Oracle #{oracle.id}: ask returned unsuccessful: "
                f"{getattr(result, 'text', '(none)')!r}"
            )
            return
        body = result.text or ""
        # Headline so the chat sees this is a scheduled oracle, not a
        # passing comment.
        header = self._oracle_header(oracle)
        if not body.startswith(header):
            body = f"{header}\n\n{body}"
        await self.signal.send_message(
            recipient="",
            message=body,
            group_id=group_id,
            attachments=result.attachments,
            styled=getattr(result, "styled", False),
        )
        logger.info(
            f"Oracle #{oracle.id} ({oracle.kind}) posted to ...{group_id[-8:]}"
        )

    @staticmethod
    def _llm_prompt_for(oracle: ContextOracle) -> str:
        if oracle.kind == "market_open":
            return _MARKET_OPEN_PROMPT
        if oracle.kind == "market_close":
            return _MARKET_CLOSE_PROMPT
        return (oracle.prompt or "").strip()

    @staticmethod
    def _oracle_header(oracle: ContextOracle) -> str:
        prefix_by_kind = {
            "market_open": "🔔 Pre-market check",
            "market_close": "🔔 Closing recap",
            "freeform": "🌅 Today's oracle",
        }
        head = prefix_by_kind.get(oracle.kind, "🌅 Today's oracle")
        if oracle.label:
            head = f"{head} — {oracle.label}"
        return f"{head}:"

    def _synth_ctx(
        self,
        group_id: str,
        policy,
        raw_message: str,
        command: str,
        args: Optional[list[str]] = None,
    ) -> CommandContext:
        return CommandContext(
            sender=self.bot_phone or "",
            group_id=group_id,
            raw_message=raw_message,
            command=command,
            args=args or [],
            policy=policy,
        )

    async def _post_command_result(
        self, result, oracle: ContextOracle, group_id: str
    ) -> None:
        if not result or not result.success:
            logger.warning(
                f"Oracle #{oracle.id}: command returned unsuccessful: "
                f"{getattr(result, 'text', '(none)')!r}"
            )
            return
        body = _replace_header(result.text or "", oracle.label)
        await self.signal.send_message(
            recipient="",
            message=body,
            group_id=group_id,
            attachments=result.attachments,
            styled=False,
        )
        logger.info(
            f"Oracle #{oracle.id} ({oracle.kind}) posted to ...{group_id[-8:]}"
        )
