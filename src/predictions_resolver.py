"""
Background resolver for due predictions.

Runs as a long-lived asyncio task: every `interval_seconds`, queries due
predictions, judges them, posts the verdict back to the originating chat.

Two judgment paths:
  1. **Structured** — `Prediction.is_structured` (ticker + threshold +
     direction) → fetch live quote, compare, post directly.
  2. **LLM** — for free-form claims, ask the LLM to call the tool with a
     verdict + short note. If the model returns "unclear", we mark the
     prediction `unclear` rather than `expired` so the leaderboard stays
     truthful (an honestly-undecidable claim isn't a wrong one).

Errors during a single prediction's resolution don't kill the worker —
the row stays `pending` and we'll retry on the next sweep.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .commands.base import CommandContext
from .contexts.policy import ContextPolicy, MODE_ALLOW_ALL, MODE_ALLOW_LIST
from .llm.mcp_broker import (
    MCP_BROKER_TOOLS,
    MCP_DISCOVER_NAME,
    MCP_INVOKE_NAME,
    broker_should_be_exposed,
    discover_mcp_tools,
    invoke_mcp_tool,
)
from .llm.tool_runtime import ToolCallLedger
from .predictions import (
    Prediction,
    PredictionStore,
    VERDICT_EMOJI,
    VERDICT_RIGHT,
    VERDICT_WRONG,
    VERDICT_UNCLEAR,
)

logger = logging.getLogger(__name__)


# Read-only research commands the resolver lets Sigil call when judging
# free-form claims. Excludes anything that writes state (alert/watch/
# predict/resolve), random draws (tarot/iching), and admin-y commands.
# MCP servers are unrestricted — they're already read-only by nature
# (search, fetch, edgar lookups).
_RESEARCH_COMMANDS = [
    "price", "chart", "ta", "news", "rating", "earnings",
    "dividend", "insider", "short", "corr", "economy",
]

# Cap on tool-call rounds the resolver lets Sigil chain. Too few and it
# can't do real research; too many and one stuck prediction starves the
# sweep. 6 is enough for "fetch news → fetch price → judge".
_MAX_RESOLVER_ROUNDS = 6


_LLM_VERDICT_PROMPT_NO_TOOLS = """\
You are judging whether a prediction came true. Reply with ONLY a JSON \
object (no preamble, no explanation outside the JSON):

  {
    "verdict": "right" | "wrong" | "unclear",
    "note": "<one short sentence of reasoning>"
  }

Rules:
  - "right": the predicted outcome happened.
  - "wrong": the opposite happened, or the deadline passed without it.
  - "unclear": you genuinely cannot tell from public knowledge. Use this \
when the claim is unfalsifiable, the outcome is partially fulfilled, or \
recent events that would settle it are outside your knowledge. Don't use \
it as a cop-out — only when there's real ambiguity.
  - The note must be one sentence, plain prose, no hedging language."""


_LLM_VERDICT_PROMPT_WITH_TOOLS = """\
You are judging whether a prediction came true. The deadline has passed; \
your job is to decide right / wrong / unclear and explain in one sentence.

You have research tools available — USE THEM before judging. The point of \
the tools is that you can actually verify the claim instead of guessing \
from training data. Typical pattern:
  - Stock-price claim → call bot__price for the current quote
  - Earnings / dividend / rating claim → call the matching bot__ tool
  - News / event / world-state claim → search the web (Brave) or fetch a \
specific URL
  - Macro / economic claim → bot__economy or web search

Don't skip the tools. A confident-sounding "wrong" without a lookup is \
worse than "unclear". After at most a few tool calls, return your verdict.
MCP tools use a compact broker: call mcp__discover for the capability, then
mcp__invoke with an exact returned name and matching arguments.

Final reply MUST be a single JSON object with no other text:

  {
    "verdict": "right" | "wrong" | "unclear",
    "note": "<one short sentence citing what you found>"
  }

Verdict rules:
  - "right": the predicted outcome happened on or before the deadline.
  - "wrong": the opposite happened, or the deadline passed without it.
  - "unclear": you genuinely cannot tell even after using tools. Use this \
when the claim is unfalsifiable, partially fulfilled, or sources disagree. \
Don't use it as a cop-out for skipping research.
  - The note must be one sentence, plain prose, no hedging — cite the \
key fact your tools surfaced (e.g. "AAPL closed at $172, threshold was \
$200")."""


class PredictionResolver:
    def __init__(
        self,
        *,
        store: PredictionStore,
        provider_manager,
        signal_pool=None,
        signal_handler=None,
        llm_client=None,
        bot_tools=None,
        mcp_manager=None,
        bot_phone: str = "",
        interval_seconds: int = 900,  # 15 min
        bot_registry=None,
    ):
        self.store = store
        self.providers = provider_manager
        # Prefer the pool — per-bot phone routing — but accept a bare
        # handler for tests and legacy single-bot wiring. When both are
        # absent, posting is a no-op.
        self.signal_pool = signal_pool
        self.signal = signal_handler or (
            signal_pool.default() if signal_pool is not None else None
        )
        self.llm = llm_client
        # When both are wired, the LLM resolver runs as a tool-using
        # research loop instead of a single zero-shot call. bot_tools
        # exposes !price/!news/etc. as bot__* tools; mcp_manager exposes
        # web search / fetch / EDGAR. Either or both can be None — the
        # resolver degrades gracefully (no tools = vanilla LLM judgment).
        self.bot_tools = bot_tools
        self.mcp_manager = mcp_manager
        # Sender identity for tool dispatch. Bot tools eventually route
        # through dispatcher._execute_command which expects a real phone
        # number for hash_phone() and audit logging. Falls back to the
        # pool's default phone (or the legacy bot_phone arg in tests).
        self.bot_phone = bot_phone or (
            signal_pool.default_phone if signal_pool is not None else ""
        )
        self.interval = interval_seconds
        # Per-prediction-context: a prediction made in an Artaud-pinned
        # group should have Artaud judge it. The resolver picks the
        # default-for-kind today; per-context pinning can be wired in
        # later by passing the context_registry.
        self.bot_registry = bot_registry

    async def run_forever(self) -> None:
        logger.info(
            f"Prediction resolver started (interval={self.interval}s)"
        )
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                logger.info("Prediction resolver cancelled")
                raise
            except Exception as e:
                logger.error(f"Prediction resolver sweep error: {e}")
            await asyncio.sleep(self.interval)

    async def sweep_once(self) -> int:
        """Process all due predictions. Returns count handled.

        Strategy:
          1. Dedupe structured predictions by ticker; fetch each ticker's
             price once in parallel (5 AAPL predictions resolving on the
             same sweep share one quote fetch).
          2. Resolve all due predictions concurrently. Per-prediction
             errors don't affect the rest of the sweep.
          3. Post verdicts back to their groups sequentially — Signal
             rate-limits bursts and 20 verdict messages in a row would
             feel spammy anyway.
        """
        due = await self.store.list_due()
        if not due:
            return 0

        # Pre-fetch quotes for all unique tickers in a single gather. Use a
        # list (not a set) so the gather order matches the zip order — set
        # iteration order isn't guaranteed identical across two passes.
        seen: set[str] = set()
        tickers: list[str] = []
        for p in due:
            if p.is_structured and p.ticker and p.ticker not in seen:
                seen.add(p.ticker)
                tickers.append(p.ticker)
        quotes: dict[str, float] = {}
        if tickers:
            results = await asyncio.gather(
                *(self._safe_quote(t) for t in tickers),
                return_exceptions=False,
            )
            for ticker, price in zip(tickers, results):
                if price is not None:
                    quotes[ticker] = price

        # Resolve all due predictions in parallel
        outcomes = await asyncio.gather(
            *(self._resolve_one(pred, quotes) for pred in due),
            return_exceptions=True,
        )

        handled = 0
        for pred, outcome in zip(due, outcomes):
            if isinstance(outcome, Exception):
                logger.exception(
                    f"Resolver: error on prediction {pred.id}: {outcome}"
                )
                continue
            if outcome is None:
                continue
            verdict, note = outcome
            await self._post_resolution(pred, verdict, note)
            handled += 1

        if handled:
            logger.info(f"Resolver: handled {handled}/{len(due)} due predictions")
        return handled

    async def _safe_quote(self, ticker: str) -> Optional[float]:
        try:
            q = await self.providers.get_quote(ticker)
            return q.price
        except Exception as e:
            logger.warning(f"Resolver: quote failed for {ticker}: {e}")
            return None

    async def _resolve_one(
        self, pred: Prediction, quotes: dict[str, float]
    ) -> Optional[tuple[str, str]]:
        """Resolve and persist one prediction. Returns (verdict, note) on
        success, None when the row should stay pending for the next sweep.
        """
        if pred.is_structured:
            return await self._resolve_structured(pred, quotes)
        return await self._resolve_with_llm(pred)

    async def _resolve_structured(
        self, pred: Prediction, quotes: dict[str, float]
    ) -> Optional[tuple[str, str]]:
        price = quotes.get(pred.ticker or "")
        if price is None:
            return None  # quote fetch failed — retry next sweep

        threshold = pred.threshold or 0.0
        if pred.direction == "above":
            hit = price > threshold
        elif pred.direction == "below":
            hit = price < threshold
        else:
            return None
        verdict = VERDICT_RIGHT if hit else VERDICT_WRONG
        note = (
            f"{pred.ticker} at ${price:.2f}; threshold ${threshold:.2f} "
            f"({pred.direction})"
        )
        await self.store.resolve(pred.id, verdict=verdict, note=note)
        return verdict, note

    async def _resolve_with_llm(
        self, pred: Prediction
    ) -> Optional[tuple[str, str]]:
        if self.llm is None:
            return None
        try:
            if not self.llm.status().get("ready"):
                return None
        except Exception:
            return None

        # Two paths: tool-enabled research loop (preferred) or vanilla
        # zero-shot judgment (fallback when bot_tools/mcp aren't wired).
        # The tool path massively improves accuracy on time-sensitive
        # claims past the model's training cutoff — without it Sigil is
        # guessing for anything that needs a fresh price or news lookup.
        tools_available = self.bot_tools is not None or self.mcp_manager is not None

        user_content = (
            f"Prediction: \"{pred.claim}\"\n"
            f"Made by [{pred.user_label}]\n"
            f"Deadline (now passed): "
            f"{_iso(pred.deadline_utc)}\n\n"
            f"Judge it now."
        )

        if not tools_available:
            return await self._resolve_with_llm_zero_shot(pred, user_content)

        return await self._resolve_with_llm_tools(pred, user_content)

    async def _resolve_with_llm_zero_shot(
        self, pred: Prediction, user_content: str
    ) -> Optional[tuple[str, str]]:
        try:
            msg = await self.llm.chat_messages(
                messages=[
                    {"role": "system", "content": _LLM_VERDICT_PROMPT_NO_TOOLS},
                    {"role": "user", "content": user_content},
                ],
                overrides={"max_tokens": 200, "temperature": 0.2},
                suppress_response_style=True,
                purpose="predict_resolve",
            )
        except Exception as e:
            logger.warning(f"Resolver: LLM judgement failed for #{pred.id}: {e}")
            return None

        verdict, note = _parse_verdict(msg.get("content") or "")
        if verdict is None:
            logger.info(
                f"Resolver: couldn't parse verdict for #{pred.id} from "
                f"{msg.get('content')!r}"
            )
            return None
        await self.store.resolve(pred.id, verdict=verdict, note=note)
        return verdict, note

    async def _resolve_with_llm_tools(
        self, pred: Prediction, user_content: str
    ) -> Optional[tuple[str, str]]:
        """Tool-loop verdict path. Sigil can call price/news/web tools
        before deciding. Returns the verdict on success or None to leave
        the row pending for the next sweep.
        """
        # Synthetic context for bot-tool dispatch. Caller_ctx.sender must
        # be a real phone for hash_phone()/audit; the bot's own phone is
        # the appropriate identity here. Permissive policy in allow_list
        # mode restricts Sigil to research-only commands so the resolver
        # can't accidentally write state (no !alert/!watch/!predict).
        synth_policy = ContextPolicy(
            id=None,
            kind="default",
            key="__resolver__",
            command_mode=MODE_ALLOW_LIST,
            commands=list(_RESEARCH_COMMANDS),
            mcp_mode=MODE_ALLOW_ALL,
        )
        # Stamp the resolving bot so per-bot writer overrides apply.
        # If the prediction lived in a context pinned to a non-default
        # bot, the resolver should judge in that bot's voice. Falls
        # back to the registry's default-for-group when bot_registry
        # is wired, or None for legacy single-bot installs.
        # We don't have a context_registry plumbed in here, so we use
        # the kind-based default rather than honoring per-context pins.
        # That's fine for PR1-4 single-bot, and PR5 can wire context
        # lookup if predictions need to be judged in a non-default
        # bot's voice.
        resolver_bot = None
        if self.bot_registry is not None:
            kind = "group" if pred.group_id else "dm"
            resolver_bot = self.bot_registry.default_for_kind_sync(kind)
        # The synthetic caller phone is the resolving bot's phone so
        # downstream tool routing (audit log, signal-cli identity)
        # matches the persona doing the judging.
        resolver_phone = (
            (resolver_bot.signal_phone if resolver_bot else None)
            or self.bot_phone
            or ""
        )
        caller_ctx = CommandContext(
            sender=resolver_phone,
            group_id=pred.group_id,
            raw_message="",
            command="(resolver)",
            args=[],
            policy=synth_policy,
            bot=resolver_bot,
        )

        tools = self._collect_resolver_tools(synth_policy)
        if not tools:
            # Filtered down to nothing — fall back to zero-shot.
            return await self._resolve_with_llm_zero_shot(pred, user_content)

        messages: list[dict] = [
            {"role": "system", "content": _LLM_VERDICT_PROMPT_WITH_TOOLS},
            {"role": "user", "content": user_content},
        ]
        ledger = ToolCallLedger()

        for round_idx in range(_MAX_RESOLVER_ROUNDS):
            try:
                assistant_msg = await self.llm.chat_messages(
                    messages,
                    tools=tools,
                    overrides={"max_tokens": 800, "temperature": 0.2},
                    suppress_response_style=True,
                    purpose="predict_resolve",
                )
            except Exception as e:
                logger.warning(
                    f"Resolver: LLM call failed for #{pred.id} "
                    f"round {round_idx}: {e}"
                )
                return None

            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # Final answer — parse JSON verdict from content.
                verdict, note = _parse_verdict(assistant_msg.get("content") or "")
                if verdict is None:
                    logger.info(
                        f"Resolver: couldn't parse verdict for #{pred.id} "
                        f"from {(assistant_msg.get('content') or '')[:200]!r}"
                    )
                    return None
                await self.store.resolve(pred.id, verdict=verdict, note=note)
                return verdict, note

            # Dispatch every tool call this round, then re-prompt.
            for call in tool_calls:
                await self._dispatch_tool_call(
                    call, messages, caller_ctx, ledger=ledger,
                )

        logger.warning(
            f"Resolver: hit tool-round cap ({_MAX_RESOLVER_ROUNDS}) "
            f"on #{pred.id} without verdict; leaving pending"
        )
        return None

    def _collect_resolver_tools(self, policy: ContextPolicy) -> list[dict]:
        out: list[dict] = []
        if self.bot_tools is not None:
            out.extend(self.bot_tools.list_tools(policy=policy))
        if broker_should_be_exposed(self.mcp_manager, policy):
            out.extend(MCP_BROKER_TOOLS)
        out.sort(key=lambda tool: (tool.get("function") or {}).get("name") or "")
        return out

    async def _dispatch_tool_call(
        self, call: dict, messages: list[dict], caller_ctx: CommandContext,
        *, ledger: Optional[ToolCallLedger] = None,
    ) -> None:
        """Run one tool call and append its result message. Errors come
        back as tool-result text so the model can recover, mirroring
        ask_command's contract."""
        call_id = call.get("id") or ""
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        if ledger is not None:
            reason, previous = ledger.lookup(
                call_id=call_id, name=name, arguments=args,
            )
            if reason is not None:
                content = (
                    previous
                    if reason == "duplicate_call_id_conflict"
                    else f"(duplicate tool call suppressed; prior result reused)\n{previous}"
                )
                messages.append({
                    "role": "tool", "tool_call_id": call_id,
                    "name": name, "content": content,
                })
                return

        is_bot_tool = (
            self.bot_tools is not None
            and name.startswith(self.bot_tools.NAMESPACE + "__")
        )
        try:
            if is_bot_tool:
                result = await self.bot_tools.call(name, args, caller_ctx)
                content = result.text if result else "(no result)"
            elif name == MCP_DISCOVER_NAME:
                content = discover_mcp_tools(
                    self.mcp_manager, caller_ctx.policy,
                    query=str(args.get("query") or ""),
                    server=str(args.get("server") or ""),
                    limit=args.get("limit", 5),
                )
            elif name == MCP_INVOKE_NAME:
                content = await invoke_mcp_tool(
                    self.mcp_manager, caller_ctx.policy,
                    name=str(args.get("name") or ""),
                    arguments=args.get("arguments"),
                )
            elif self.mcp_manager is not None:
                content = (
                    f"ERROR: direct MCP call {name!r} is unavailable. "
                    f"Use {MCP_DISCOVER_NAME}, then {MCP_INVOKE_NAME}."
                )
            else:
                content = f"ERROR: unknown tool {name}"
        except Exception as e:
            logger.warning(f"Resolver: tool {name} failed: {e}")
            content = f"ERROR: {e}"

        content = content if isinstance(content, str) else str(content)
        if ledger is not None:
            ledger.record(
                call_id=call_id, name=name, arguments=args, content=content,
            )

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })

    async def _post_resolution(
        self, pred: Prediction, verdict: str, note: str
    ) -> None:
        record = await self.store.user_record(pred.user_hash, pred.context_key)
        record_str = ""
        if record["accuracy"] is not None and (record["right"] + record["wrong"]):
            judged = record["right"] + record["wrong"]
            record_str = (
                f"\n  {pred.user_label}'s record: {record['right']}/{judged} "
                f"({int(record['accuracy'] * 100)}%)"
            )
        emoji = VERDICT_EMOJI.get(verdict, "•")
        text = (
            f"⏰ Prediction #{pred.id} by {pred.user_label} is in:\n"
            f"  \"{pred.claim}\"\n\n"
            f"  {emoji} {verdict.upper()}{f' — {note}' if note else ''}"
            f"{record_str}"
        )
        # Route the verdict post through the bot that judged it, so
        # multi-phone installs reply from the right number. Falls back
        # to the legacy single handler when no pool is wired.
        sender_handler = self.signal
        if self.signal_pool is not None and self.bot_registry is not None:
            kind = "group" if pred.group_id else "dm"
            resolver_bot = self.bot_registry.default_for_kind_sync(kind)
            sender_handler = self.signal_pool.for_bot(resolver_bot)
        try:
            if pred.group_id and sender_handler is not None:
                await sender_handler.send_message(
                    recipient="", message=text, group_id=pred.group_id,
                )
            else:
                # DM context — context_key is "dm:<hash>". We don't have
                # the phone here (only its hash), so DMs can't be
                # auto-posted. Log it and let !predictions show the
                # resolved status.
                logger.info(
                    f"Resolver: #{pred.id} resolved in DM context "
                    f"(no auto-post): {verdict} — {note}"
                )
        except Exception as e:
            logger.error(f"Resolver: post failed for #{pred.id}: {e}")


def _iso(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _parse_verdict(content: str) -> tuple[Optional[str], str]:
    """Pull the first JSON object from `content`, validate, return tuple.

    Returns (None, '') on any parse failure so the caller can keep the
    prediction pending instead of guessing.
    """
    if not content:
        return None, ""
    import re
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None, ""
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None, ""
    if not isinstance(parsed, dict):
        return None, ""
    verdict = parsed.get("verdict")
    note = (parsed.get("note") or "").strip()
    if verdict in (VERDICT_RIGHT, VERDICT_WRONG, VERDICT_UNCLEAR):
        return verdict, note
    return None, ""
