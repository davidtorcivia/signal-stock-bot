"""
DeepThinkClient — secondary LLM call routed to a "smarter" model on demand.

Exposed as a tool to the primary writer LLM (in src/commands/ask_command.py),
not as a user-facing command. The writer decides when a question outstrips
its own ability and calls deep_think(question, context); we route that
sub-question to the configured deep model and return its text.

Crucially, the deep model gets the SAME tool kit the writer has — every
allowed bot command and MCP tool, filtered by the same per-context policy.
This is the whole point of using a smarter model: it can do real research,
not just reason from training data. We run a multi-round tool loop here
(mirroring ask_command._run_tool_loop) and bubble any attachments produced
back up to the caller's attachments list.

Every config key falls back to its llm_* equivalent when blank, so admins
can override only what's different (typically: model + max_tokens + maybe
api_key/base_url) and inherit everything else.

Failure modes are SILENT to the writer — every error path returns a string
the writer can integrate or discard. Never raise out of think(); the writer
should not care whether deep think succeeded, just whether it has useful
text or a "(unavailable)" stub.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from collections import deque
from datetime import datetime, timezone
from typing import Optional
from urllib.parse import urlparse

import aiohttp

from ..admin.events import get_bus
from ..bots.settings import (
    resolve_bool,
    resolve_int,
    resolve_setting,
)
from ..cache import get_metrics
from .client import _ensure_reasoning_content, _extract_cache_tokens
from .mcp_broker import (
    MCP_BROKER_TOOLS,
    MCP_DISCOVER_NAME,
    MCP_INVOKE_NAME,
    broker_should_be_exposed,
    discover_mcp_tools,
    invoke_mcp_tool,
)
from .prompt_cache import PromptCachePlan
from .resilience import LLMHTTPFailure, resilient_chat_post
from .tool_runtime import (
    DurableToolLedger,
    ToolCallLedger,
    ToolExecution,
    ToolLoopRuntime,
    ensure_tool_result_envelope,
    tool_result_ok,
)

logger = logging.getLogger(__name__)


# Built-in system prompt when admin leaves deep_think_system_prompt blank.
# Different shape from the main LLM persona — no Signal-bot voice, no
# brevity. The writer integrates this output into a normal-tone reply.
DEFAULT_DEEP_THINK_PROMPT = """\
You are a careful reasoner answering one focused sub-question on behalf of
a chat assistant. Your output will be inlined into the assistant's reply.

You have access to the assistant's full tool kit — bot commands (price
lookups, charts, news, fundamentals, calendars) and any configured MCP
servers (SEC EDGAR filings, Yahoo finance MCP, etc.). USE THESE TOOLS
LIBERALLY when the answer depends on live data, specific numbers, or
factual information beyond your training. Don't guess at a price, P/E,
or recent event — fetch it. MCP capabilities use a compact broker: call
mcp__discover with what you need, then mcp__invoke with the exact returned
name and argument schema.

Take your time. Think step-by-step where it helps. Chain multiple tool
calls when one finding leads to the next question. Show your work when the
answer depends on multi-step reasoning, comparisons, or non-obvious facts.
Be honest about uncertainty — flag it explicitly rather than confabulating.

When the question is concrete (a fact, a number, a definition), lead with
the answer and then justify it briefly. When the question is open-ended
(a judgment, a recommendation), give your reasoning first and your bottom
line at the end.

Format with plain prose, occasional bullets only when truly list-shaped.
No headers, no preamble like "Sure!" or "Great question". Just the answer
and its supporting reasoning. The writer will weave your output into a
casual chat reply, so don't over-format."""


# How many recent deep_think calls to retain in memory for the admin
# dashboard. Wipes on restart — matches the conversational half-life of
# "what was the bot just asked to think about?" inspection.
RECENT_CALLS_LOG_SIZE = 50
QUESTION_PREVIEW_CHARS = 160


class DeepThinkClient:
    def __init__(
        self,
        settings_store,
        bot_tools=None,
        mcp_manager=None,
        bot_id: Optional[int] = None,
    ):
        self.store = settings_store
        # bot_id scopes config reads to bot_llm_settings(bot_id, 'deep_think', ...)
        # before falling back to admin_settings.deep_think_* / .llm_*.
        # None = legacy global-only read (back-compat for tests / single-bot).
        self.bot_id = bot_id
        self.role = "deep_think"
        # Tool plumbing — both late-bindable since bot_tools depends on the
        # dispatcher, which depends on this client via ask_command. Wired in
        # main.py after both exist. When unset (or unconfigured), think()
        # falls back to a single-shot text call without tools.
        self.bot_tools = bot_tools
        self.mcp_manager = mcp_manager
        self.durable_tool_ledger = DurableToolLedger()
        # Rolling log for the admin dashboard. (ts, question, latency_ms,
        # model, tokens_in, tokens_out, ok, error_msg, user_tail, group_id,
        # tool_calls — names of tools the deep model called this turn).
        self._recent: deque = deque(maxlen=RECENT_CALLS_LOG_SIZE)
        # Daily counters for cap visibility / future enforcement. Reset
        # lazily when the date crosses (UTC midnight).
        self._daily_date: str = self._utc_date()
        self._daily_calls: int = 0
        self._daily_per_user: dict[str, int] = {}
        self._daily_per_group: dict[str, int] = {}
        # Long-lived session reused across rounds and calls (mirrors
        # LLMClient). A fresh session per round meant a tool loop paid
        # one TCP+TLS handshake per round — 25 handshakes on the worst
        # observed loop.
        self._session: Optional[aiohttp.ClientSession] = None
        self._session_lock = asyncio.Lock()

    async def _get_session(self) -> aiohttp.ClientSession:
        async with self._session_lock:
            if self._session is None or self._session.closed:
                self._session = aiohttp.ClientSession()
            return self._session

    async def close(self) -> None:
        if self._session is not None and not self._session.closed:
            await self._session.close()
            self._session = None

    @staticmethod
    def _utc_date() -> str:
        return datetime.now(timezone.utc).strftime("%Y-%m-%d")

    def _maybe_roll_day(self) -> None:
        today = self._utc_date()
        if today != self._daily_date:
            self._daily_date = today
            self._daily_calls = 0
            self._daily_per_user = {}
            self._daily_per_group = {}

    def _config(self) -> dict:
        """Live config with bot -> deep_think_* -> llm_* fallback chain.

        Resolution order at each key:
          1. bot_llm_settings(bot_id, 'deep_think', key)  — when bot_id set
          2. admin_settings.deep_think_<key>              — global override
          3. admin_settings.llm_<key>                     — writer fallback
          4. code default
        """
        store = self.store
        bot_id, role = self.bot_id, self.role

        def _str(key: str, *globals_: str) -> str:
            """Non-blank-string-aware resolver. The base resolve_setting
            returns the first non-None hit; this wrapper additionally
            skips empty strings so a blank `deep_think_model` falls
            through to `llm_model` rather than masking it."""
            if bot_id is not None:
                v = store.get_bot(bot_id, role, key, default=None)
                if isinstance(v, str) and v.strip():
                    return v
                if v is not None and not isinstance(v, str):
                    return str(v)
            for gk in globals_:
                v = store.get(gk)
                if isinstance(v, str) and v.strip():
                    return v
            return ""

        def _num(key: str, default: float, *globals_: str) -> float:
            """Skip blank-string values so they don't shadow a real
            numeric configured at the writer level."""
            if bot_id is not None:
                v = store.get_bot(bot_id, role, key, default=None)
                if v is not None and not (isinstance(v, str) and not v.strip()):
                    try:
                        return float(v)
                    except (TypeError, ValueError):
                        pass
            for gk in globals_:
                v = store.get(gk)
                if v is None or (isinstance(v, str) and not v.strip()):
                    continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return default

        sys_prompt = resolve_setting(
            store, bot_id, role, "system_prompt",
            global_keys=["deep_think_system_prompt"],
        )

        return {
            "enabled": resolve_bool(
                store, bot_id, role, "enabled",
                global_keys=["deep_think_enabled"], default=False,
            ),
            "base_url": _str("base_url", "deep_think_base_url", "llm_base_url").strip().rstrip("/"),
            "api_key": _str("api_key", "deep_think_api_key", "llm_api_key").strip(),
            "model": _str("model", "deep_think_model", "llm_model").strip(),
            "temperature": _num("temperature", 0.7, "deep_think_temperature", "llm_temperature"),
            "max_tokens": int(_num("max_tokens", 8000, "deep_think_max_tokens", "llm_max_tokens")),
            "timeout": int(_num("timeout_seconds", 120, "deep_think_timeout_seconds", "llm_timeout_seconds")),
            "extra_body": _str("extra_body", "deep_think_extra_body", "llm_extra_body"),
            "system_prompt": sys_prompt or DEFAULT_DEEP_THINK_PROMPT,
            "context_max_chars": resolve_int(
                store, bot_id, role, "context_max_chars",
                global_keys=["deep_think_context_max_chars"], default=8000,
            ),
            "max_tool_rounds": resolve_int(
                store, bot_id, role, "max_tool_rounds",
                global_keys=["deep_think_max_tool_rounds"], default=15,
            ),
            "caps_enabled": resolve_bool(
                store, bot_id, role, "caps_enabled",
                global_keys=["deep_think_caps_enabled"], default=False,
            ),
            "user_daily_cap": resolve_int(
                store, bot_id, role, "user_daily_cap",
                global_keys=["deep_think_user_daily_cap"], default=0,
            ),
            "group_daily_cap": resolve_int(
                store, bot_id, role, "group_daily_cap",
                global_keys=["deep_think_group_daily_cap"], default=0,
            ),
        }

    def status(self) -> dict:
        """Summary for the admin UI — does not leak the API key."""
        cfg = self._config()
        return {
            "enabled": cfg["enabled"],
            "base_url": cfg["base_url"],
            "model": cfg["model"],
            "api_key_set": bool(cfg["api_key"]),
            "ready": cfg["enabled"] and all(
                [cfg["base_url"], cfg["api_key"], cfg["model"]]
            ),
            "caps_enabled": cfg["caps_enabled"],
            "user_daily_cap": cfg["user_daily_cap"],
            "group_daily_cap": cfg["group_daily_cap"],
        }

    def usage_today(self) -> dict:
        """Per-day counters. Used by the admin dashboard."""
        self._maybe_roll_day()
        return {
            "date": self._daily_date,
            "total_calls": self._daily_calls,
            "per_user": dict(self._daily_per_user),
            "per_group": dict(self._daily_per_group),
        }

    def recent_calls(self, limit: int = 20) -> list[dict]:
        """Newest-first list of recent deep_think calls for the dashboard."""
        items = list(self._recent)[-max(1, limit):]
        items.reverse()
        return [
            {
                "ts": ts,
                "question": question,
                "latency_ms": latency_ms,
                "model": model,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "ok": ok,
                "error_msg": error_msg,
                "user_tail": user_tail,
                "group_id": group_id,
            }
            for (
                ts, question, latency_ms, model, tokens_in, tokens_out,
                ok, error_msg, user_tail, group_id,
            ) in items
        ]

    def _check_cap(self, *, user_hash: str, group_id: Optional[str], cfg: dict) -> Optional[str]:
        """Return a reason string if a cap is hit, otherwise None.

        Counters always advance regardless of whether enforcement is on —
        the dashboard shows usage either way. Enforcement is gated on
        `caps_enabled` so admins can monitor before turning it on.
        """
        if not cfg["caps_enabled"]:
            return None
        if cfg["user_daily_cap"] > 0:
            if self._daily_per_user.get(user_hash, 0) >= cfg["user_daily_cap"]:
                return f"user daily cap ({cfg['user_daily_cap']}) reached"
        if group_id and cfg["group_daily_cap"] > 0:
            if self._daily_per_group.get(group_id, 0) >= cfg["group_daily_cap"]:
                return f"group daily cap ({cfg['group_daily_cap']}) reached"
        return None

    def _record_call_attempt(self, *, user_hash: str, group_id: Optional[str]) -> None:
        """Bump the counters before the network call so concurrent requests
        see consistent state for cap enforcement."""
        self._maybe_roll_day()
        self._daily_calls += 1
        self._daily_per_user[user_hash] = self._daily_per_user.get(user_hash, 0) + 1
        if group_id:
            self._daily_per_group[group_id] = self._daily_per_group.get(group_id, 0) + 1

    async def _chat_call(
        self,
        messages: list[dict],
        *,
        tools: Optional[list[dict]],
        cfg: dict,
        sender_tail: str,
        group_id: Optional[str],
        context_id: Optional[int] = None,
        bot_id: Optional[int] = None,
        cache_plan: Optional[PromptCachePlan] = None,
    ) -> dict:
        """One chat-completion call. Records per-round metrics. Raises
        RuntimeError on failure — the outer loop catches and converts to
        the silent (deep_think unavailable: ...) stub."""
        # Mirror reasoning -> reasoning_content on prior assistant turns,
        # exactly as LLMClient.chat_messages does. DeepSeek thinking
        # models reject (HTTP 400) a round-tripped assistant turn whose
        # reasoning_content is missing, so without this every tool loop
        # died on round 2+.
        messages = _ensure_reasoning_content(messages)
        payload: dict = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": float(cfg["temperature"]),
            "max_tokens": int(cfg["max_tokens"]),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        extra_raw = cfg["extra_body"]
        if isinstance(extra_raw, str) and extra_raw.strip():
            try:
                extra = json.loads(extra_raw)
                if isinstance(extra, dict):
                    payload.update(extra)
            except json.JSONDecodeError as e:
                logger.warning(f"DeepThink extra_body not valid JSON, ignoring: {e}")
        elif isinstance(extra_raw, dict) and extra_raw:
            payload.update(extra_raw)

        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{cfg['base_url']}/chat/completions"
        # Hard-ceiling defense against the slow-trickle case (matches
        # LLMClient): aiohttp's total= is unreliable on stalled sockets.
        hard_timeout = max(cfg["timeout"] + 30, 60)

        metrics = get_metrics()
        started = time.time()

        session = await self._get_session()
        declared = cache_plan or PromptCachePlan.from_blocks(
            stable=[("deep_think_system", messages[0].get("content") if messages else "")],
            volatile=[("deep_think_tail", messages[-1].get("content") if messages else "")],
        )
        cache_manifest = declared.snapshot(
            messages,
            tools,
            purpose="deep_think",
            context_id=context_id,
            bot_id=bot_id,
        )
        host = urlparse(cfg["base_url"]).netloc or cfg["base_url"]
        provider_metrics = metrics.get_provider_metrics(
            f"deep_think:{host}:{cfg['model']}"
        )

        def _on_retry(reason: str, attempt: int, delay: float) -> None:
            metrics.record_llm_retry(
                reason=reason, purpose="deep_think", model=cfg["model"],
            )
            logger.warning(
                "DeepThink retry reason=%s after_attempt=%s delay=%.2fs",
                reason, attempt, delay,
            )

        try:
            data = await resilient_chat_post(
                session=session,
                url=url,
                payload=payload,
                headers=headers,
                request_timeout=cfg["timeout"],
                hard_timeout=hard_timeout,
                provider_metrics=provider_metrics,
                on_retry=_on_retry,
            )
        except asyncio.TimeoutError:
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg=f"hard timeout {hard_timeout}s",
            )
            raise RuntimeError(f"timeout {hard_timeout}s")
        except LLMHTTPFailure as e:
            metrics.record_deep_think_error(model=cfg["model"], error_msg=str(e))
            raise RuntimeError(str(e)) from e
        except aiohttp.ClientError as e:
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg=f"network: {e}",
            )
            raise RuntimeError(f"network: {e}")
        except RuntimeError as e:
            if "circuit is open" in str(e):
                metrics.record_llm_circuit_rejection(
                    purpose="deep_think", model=cfg["model"],
                )
            metrics.record_deep_think_error(model=cfg["model"], error_msg=str(e))
            raise

        latency_ms = (time.time() - started) * 1000.0

        try:
            msg = data["choices"][0]["message"]
            if msg.get("content") is None:
                msg["content"] = ""
            usage = data.get("usage") or {}
            tokens_in = int(usage.get("prompt_tokens") or 0)
            tokens_out = int(usage.get("completion_tokens") or 0)
            cache_hit, cache_miss = _extract_cache_tokens(usage)
        except (KeyError, IndexError, AttributeError, TypeError):
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg="unexpected response shape",
            )
            logger.error(f"DeepThink unexpected response: {data}")
            raise RuntimeError("unexpected response shape")

        metrics.record_deep_think_success(
            model=cfg["model"], latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
            cache_hit_tokens=cache_hit, cache_miss_tokens=cache_miss,
        )
        metrics.record_prompt_cache_observation(
            cache_manifest,
            cache_hit_tokens=cache_hit,
            cache_miss_tokens=cache_miss,
            latency_ms=latency_ms,
        )
        logger.debug(
            f"DeepThink round: {tokens_in}+{tokens_out} tok, "
            f"{latency_ms:.0f}ms (...{sender_tail})"
        )
        # Stash usage on the message dict so the loop can aggregate without
        # re-parsing. Stripped before the message is sent back to the model
        # — `_strip_usage` is called in _run_tool_loop right after.
        msg["_dt_tokens_in"] = tokens_in
        msg["_dt_tokens_out"] = tokens_out
        return msg

    def _collect_tools(self, policy=None) -> Optional[list[dict]]:
        """Build the tool list the deep model gets — same allow-listing as
        the writer, with deep_think itself never exposed (no recursion)."""
        schemas: list[dict] = []
        if self.bot_tools is not None:
            schemas.extend(self.bot_tools.list_tools(policy=policy))
        if broker_should_be_exposed(self.mcp_manager, policy):
            schemas.extend(MCP_BROKER_TOOLS)
        # Match the writer's canonical ordering. Session restart order should
        # not rewrite an otherwise identical request prefix.
        schemas.sort(
            key=lambda schema: str(
                (schema.get("function") or {}).get("name") or ""
            )
        )
        return schemas or None

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        caller_ctx,
        attachments: list,
        cfg: dict,
        sender_tail: str,
        group_id: Optional[str],
        cache_plan: Optional[PromptCachePlan] = None,
    ) -> tuple[str, list[str], int, int]:
        """Drive the deep model through tool calls until it produces text.

        Returns (final_text, tool_call_history, total_tokens_in,
        total_tokens_out). Token totals aggregate across all rounds for
        the recent-calls dashboard view; per-round metrics are recorded
        inside `_chat_call` for the LLM/deep_think totals.

        Budget or no-progress stops get one final no-tools synthesis constrained
        to completed evidence, with an honest fallback if synthesis fails.
        """
        max_rounds = max(1, cfg["max_tool_rounds"])
        context_id = None
        bot_id = None
        if caller_ctx is not None:
            policy = getattr(caller_ctx, "policy", None)
            context_id = getattr(policy, "id", None)
            bot_id = getattr(caller_ctx, "bot_id", None)

        async def chat_round(
            working_messages: list[dict], active_tools: Optional[list[dict]],
        ) -> dict:
            return await self._chat_call(
                working_messages, tools=active_tools, cfg=cfg,
                sender_tail=sender_tail, group_id=group_id,
                context_id=context_id, bot_id=bot_id, cache_plan=cache_plan,
            )

        async def execute_call(
            call: dict, ledger: ToolCallLedger,
        ) -> ToolExecution:
            local_messages: list[dict] = []
            local_attachments: list = []
            await self._dispatch_tool_call(
                call, local_messages, caller_ctx, local_attachments,
                tools=tools, ledger=ledger,
            )
            name = (call.get("function") or {}).get("name") or ""
            message = local_messages[-1] if local_messages else {
                "role": "tool", "tool_call_id": call.get("id") or "",
                "name": name,
                "content": ensure_tool_result_envelope(
                    name, "ERROR: tool returned no result", ok=False,
                ),
            }
            return ToolExecution(
                message, local_attachments,
                tool_result_ok(message.get("content", "")),
            )

        source_key = None
        if (
            caller_ctx is not None
            and getattr(caller_ctx, "message_timestamp", None) is not None
        ):
            source_key = (
                f"{caller_ctx.context_key()}:"
                f"{getattr(caller_ctx, 'bot_id', None) or 0}:"
                f"{caller_ctx.message_timestamp}"
            )
        outcome = await ToolLoopRuntime(max_rounds=max_rounds).run(
            messages=messages, tools=tools, chat=chat_round,
            execute=execute_call, attachments=attachments,
            source_key=source_key,
            durable_ledger=getattr(self, "durable_tool_ledger", None),
            empty_response=(
                "(deep_think produced no final answer — the model stopped "
                "while still reasoning)"
            ),
            incomplete_fallback=(
                f"(deep_think exhausted its {max_rounds}-round tool budget "
                "without enough completed evidence for a safe answer)"
            ),
        )
        logger.info(
            "DeepThink runtime stopped reason=%s rounds=%d calls=%d parallel=%d",
            outcome.stop_reason, outcome.rounds, outcome.calls,
            outcome.parallel_batches,
        )
        return (
            outcome.content, outcome.tool_history,
            outcome.tokens_in, outcome.tokens_out,
        )

    async def _dispatch_tool_call(
        self,
        call: dict,
        messages: list[dict],
        caller_ctx,
        attachments: list,
        tools: Optional[list[dict]] = None,
        ledger: Optional[ToolCallLedger] = None,
    ) -> None:
        """Run one tool call and append its result to `messages`.

        Mirrors ask_command._execute_tool_call but skips the deep_think
        tool entirely (the deep model should never get to call itself).
        """
        call_id = call.get("id") or ""
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        if isinstance(raw_args, str):
            log_args = raw_args[:200]
        else:
            log_args = json.dumps(args)[:200]
        logger.info(f"DeepThink tool call: {name} args={log_args}")

        if tools:
            valid_names = {
                (tool.get("function") or {}).get("name")
                for tool in tools
                if isinstance(tool, dict)
            }
            if name not in valid_names:
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": ensure_tool_result_envelope(
                        name,
                        f"ERROR: no tool named {name!r} is available. "
                        f"Use {MCP_DISCOVER_NAME} for MCP capabilities.",
                        ok=False,
                    ),
                })
                return

        if ledger is not None:
            duplicate_reason, duplicate_content = ledger.lookup(
                call_id=call_id, name=name, arguments=args,
            )
            if duplicate_reason is not None:
                logger.warning(
                    "Suppressing duplicate DeepThink tool call name=%s reason=%s",
                    name, duplicate_reason,
                )
                if duplicate_reason == "duplicate_call_id_conflict":
                    content = ensure_tool_result_envelope(
                        name,
                        duplicate_content or "ERROR: duplicate tool call suppressed",
                        ok=False,
                        reused="duplicate tool call suppressed: call-id conflict",
                    )
                else:
                    content = ensure_tool_result_envelope(
                        name, duplicate_content or "(no result)", ok=True,
                        reused=f"duplicate tool call suppressed: {duplicate_reason}",
                    )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": content,
                })
                return

        is_bot_tool = (
            self.bot_tools is not None
            and name.startswith(self.bot_tools.NAMESPACE + "__")
        )

        try:
            if name == "deep_think":
                # Defense in depth — we don't expose this in tools, but a
                # model could still hallucinate the call.
                content = "(deep_think cannot call itself recursively)"
            elif is_bot_tool:
                if caller_ctx is None:
                    content = "ERROR: no caller context for bot tool"
                else:
                    result = await self.bot_tools.call(name, args, caller_ctx)
                    content = result.text if result else "(no result)"
                    if result is not None and not result.success:
                        content = f"ERROR: {content}"
                    if result and result.attachments:
                        attachments.extend(result.attachments)
            elif name == MCP_DISCOVER_NAME:
                policy = getattr(caller_ctx, "policy", None) if caller_ctx else None
                content = discover_mcp_tools(
                    self.mcp_manager,
                    policy,
                    query=str(args.get("query") or ""),
                    server=str(args.get("server") or ""),
                    limit=args.get("limit", 5),
                )
            elif name == MCP_INVOKE_NAME:
                policy = getattr(caller_ctx, "policy", None) if caller_ctx else None
                content = await invoke_mcp_tool(
                    self.mcp_manager,
                    policy,
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
            logger.warning(f"DeepThink tool {name} failed: {e}")
            content = f"ERROR: {e}"

        content = ensure_tool_result_envelope(name, content)

        if ledger is not None:
            ledger.record(
                call_id=call_id,
                name=name,
                arguments=args,
                content=str(content),
            )

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })

    async def think(
        self,
        question: str,
        context: str = "",
        *,
        user_hash: str = "",
        group_id: Optional[str] = None,
        caller_ctx=None,
        attachments: Optional[list] = None,
    ) -> str:
        """Run one deep_think call and return text the writer can use.

        When `caller_ctx` is provided (and the client has bot_tools /
        mcp_manager attached), the deep model gets the same tool kit the
        writer has — filtered by `caller_ctx.policy`. Any attachments
        produced (charts, etc.) are appended to `attachments` so the
        outer ask_command can pass them through to the user.

        Never raises. Every failure mode resolves to a "(deep_think
        unavailable: <reason>)" string so the writer can keep going.
        """
        question = (question or "").strip()
        if not question:
            return "(deep_think unavailable: empty question)"
        if attachments is None:
            attachments = []

        cfg = self._config()

        if not cfg["enabled"]:
            self._publish("skip", question=question, reason="disabled",
                          user_hash=user_hash, group_id=group_id)
            return "(deep_think unavailable: feature disabled)"
        if not all([cfg["base_url"], cfg["api_key"], cfg["model"]]):
            self._publish("skip", question=question, reason="not_configured",
                          user_hash=user_hash, group_id=group_id)
            return "(deep_think unavailable: not configured)"

        cap_reason = self._check_cap(user_hash=user_hash, group_id=group_id, cfg=cfg)
        if cap_reason:
            self._publish("skip", question=question, reason=cap_reason,
                          user_hash=user_hash, group_id=group_id)
            return f"(deep_think rate-limited: {cap_reason})"

        # Cap context to budget. Hard cut at the boundary so a runaway writer
        # stuffing the entire chat history can't push the deep model into a
        # 100k-token bill. Kept simple — char count, not tokens.
        if context and len(context) > cfg["context_max_chars"]:
            context = context[: cfg["context_max_chars"]] + "\n\n[…context truncated by budget…]"

        # Same tool kit the writer has (filtered by the same policy), with
        # deep_think itself excluded so it can't recurse on itself.
        policy = getattr(caller_ctx, "policy", None) if caller_ctx else None
        tools = self._collect_tools(policy=policy)

        user_content = (
            f"Sub-question to think carefully about:\n{question}\n\n"
            f"Supporting context (may be empty):\n{context.strip() or '(none provided)'}"
        )
        messages: list[dict] = [
            {"role": "system", "content": cfg["system_prompt"]},
            {"role": "user", "content": user_content},
        ]
        cache_plan = PromptCachePlan.from_blocks(
            stable=[("deep_think_system", cfg["system_prompt"])],
            volatile=[("deep_think_request", user_content)],
        )

        sender_tail = user_hash[:8] if user_hash else "????"

        # Bump counters before the call so concurrent triggers don't both
        # think they're under the cap.
        self._record_call_attempt(user_hash=user_hash, group_id=group_id)

        self._publish(
            "start", question=question, user_hash=user_hash,
            group_id=group_id, model=cfg["model"],
            tools=len(tools or []),
        )

        outer_started = time.time()
        try:
            content, tool_history, agg_in, agg_out = await self._run_tool_loop(
                messages=messages,
                tools=tools,
                caller_ctx=caller_ctx,
                attachments=attachments,
                cfg=cfg,
                sender_tail=sender_tail,
                group_id=group_id,
                cache_plan=cache_plan,
            )
        except Exception as e:
            latency_ms = (time.time() - outer_started) * 1000.0
            err = str(e)[:200]
            logger.error(f"DeepThink loop error: {e}")
            self._log_call(
                question=question, latency_ms=latency_ms, model=cfg["model"],
                tokens_in=0, tokens_out=0, ok=False, error_msg=err,
                user_tail=sender_tail, group_id=group_id,
            )
            self._publish(
                "error", question=question, user_hash=user_hash,
                group_id=group_id, error_msg=err,
            )
            return "(deep_think unavailable: loop error)"

        latency_ms = (time.time() - outer_started) * 1000.0

        # Per-round metrics already recorded inside _chat_call. The recent-
        # calls log + done event aggregate across the whole think() so the
        # dashboard shows tokens spent on each invocation as a unit.
        self._log_call(
            question=question, latency_ms=latency_ms, model=cfg["model"],
            tokens_in=agg_in, tokens_out=agg_out,
            ok=bool(content), error_msg="",
            user_tail=sender_tail, group_id=group_id,
        )
        self._publish(
            "done", question=question, user_hash=user_hash, group_id=group_id,
            model=cfg["model"], latency_ms=int(latency_ms),
            tokens_in=agg_in, tokens_out=agg_out, chars=len(content),
            tools_used=len(tool_history),
            tool_history=" -> ".join(tool_history[-8:]) if tool_history else "",
        )
        logger.info(
            f"DeepThink: {len(content)}c, "
            f"{len(tool_history)} tools, {agg_in}+{agg_out} tok, "
            f"{latency_ms:.0f}ms ({cfg['model']})"
        )

        return content or "(deep_think returned no content)"

    async def health_check(self, prompt: Optional[str] = None) -> dict:
        """Single round-trip connectivity test for the deep_think
        endpoint. Bypasses the full tool loop / cap counters / event
        bus — this is for the admin's "test connection" button, not a
        real research call. Same return shape as `LLMClient.health_check`
        so the admin route can render both uniformly. Never raises."""
        cfg = self._config()
        if not cfg["enabled"]:
            return {
                "ok": False,
                "error": "deep_think is disabled for this bot.",
                "model": cfg.get("model") or "",
                "latency_ms": 0,
            }
        missing = [
            k for k in ("base_url", "api_key", "model") if not cfg.get(k)
        ]
        if missing:
            return {
                "ok": False,
                "error": f"Not configured: missing {', '.join(missing)}.",
                "model": cfg.get("model") or "",
                "latency_ms": 0,
            }
        user_msg = (prompt or "").strip() or (
            "Reply with one short sentence to confirm you're reachable."
        )
        started = time.time()
        try:
            msg = await self._chat_call(
                messages=[
                    {"role": "system", "content": cfg["system_prompt"]},
                    {"role": "user", "content": user_msg},
                ],
                tools=None,
                cfg=cfg,
                sender_tail="hcheck",
                group_id=None,
            )
        except RuntimeError as e:
            return {
                "ok": False,
                "error": str(e),
                "model": cfg.get("model") or "",
                "latency_ms": int((time.time() - started) * 1000),
            }
        return {
            "ok": True,
            "reply": (msg.get("content") or "").strip(),
            "model": cfg["model"],
            "latency_ms": int((time.time() - started) * 1000),
        }

    def _log_call(
        self, *, question: str, latency_ms: float, model: str,
        tokens_in: int, tokens_out: int, ok: bool, error_msg: str,
        user_tail: str, group_id: Optional[str],
    ) -> None:
        snippet = (question or "").replace("\n", " ").strip()
        if len(snippet) > QUESTION_PREVIEW_CHARS:
            snippet = snippet[: QUESTION_PREVIEW_CHARS - 1] + "…"
        self._recent.append((
            time.time(), snippet, int(latency_ms), model,
            int(tokens_in), int(tokens_out), bool(ok), error_msg or "",
            user_tail, group_id or "",
        ))

    def _publish(self, stage: str, **fields) -> None:
        try:
            payload = dict(fields)
            payload["stage"] = stage
            # Truncate question for the wire
            q = payload.get("question") or ""
            if q:
                payload["question"] = q[:200]
            user_hash = payload.pop("user_hash", "") or ""
            payload["user_tail"] = user_hash[:8] if user_hash else ""
            get_bus().publish("deep_think", **payload)
        except Exception as e:
            logger.debug(f"DeepThink bus publish failed: {e}")
