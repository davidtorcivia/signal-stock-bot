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

import aiohttp

from ..admin.events import get_bus
from ..cache import get_metrics

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
or recent event — fetch it.

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
    ):
        self.store = settings_store
        # Tool plumbing — both late-bindable since bot_tools depends on the
        # dispatcher, which depends on this client via ask_command. Wired in
        # main.py after both exist. When unset (or unconfigured), think()
        # falls back to a single-shot text call without tools.
        self.bot_tools = bot_tools
        self.mcp_manager = mcp_manager
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
        """Live config with llm_* fallback for unset deep_think_* fields."""
        store = self.store

        def _str(key: str, fallback_key: str = "") -> str:
            """First non-blank string value, with optional llm_* fallback."""
            v = store.get(key)
            if isinstance(v, str) and v.strip():
                return v
            if fallback_key:
                fb = store.get(fallback_key)
                if isinstance(fb, str) and fb.strip():
                    return fb
            return ""

        def _num(key: str, fallback_key: str, default: float) -> float:
            """First non-zero numeric value, with llm_* fallback."""
            for k in (key, fallback_key):
                v = store.get(k)
                if v is None or (isinstance(v, str) and not v.strip()):
                    continue
                try:
                    return float(v)
                except (TypeError, ValueError):
                    continue
            return default

        return {
            "enabled": store.get_bool("deep_think_enabled", False),
            "base_url": _str("deep_think_base_url", "llm_base_url").strip().rstrip("/"),
            "api_key": _str("deep_think_api_key", "llm_api_key").strip(),
            "model": _str("deep_think_model", "llm_model").strip(),
            "temperature": _num("deep_think_temperature", "llm_temperature", 0.7),
            "max_tokens": int(_num("deep_think_max_tokens", "llm_max_tokens", 8000)),
            "timeout": int(_num("deep_think_timeout_seconds", "llm_timeout_seconds", 120)),
            "extra_body": _str("deep_think_extra_body", "llm_extra_body"),
            "system_prompt": store.get("deep_think_system_prompt") or DEFAULT_DEEP_THINK_PROMPT,
            "context_max_chars": store.get_int("deep_think_context_max_chars", 8000),
            "max_tool_rounds": store.get_int("deep_think_max_tool_rounds", 15),
            "caps_enabled": store.get_bool("deep_think_caps_enabled", False),
            "user_daily_cap": store.get_int("deep_think_user_daily_cap", 0),
            "group_daily_cap": store.get_int("deep_think_group_daily_cap", 0),
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
    ) -> dict:
        """One chat-completion call. Records per-round metrics. Raises
        RuntimeError on failure — the outer loop catches and converts to
        the silent (deep_think unavailable: ...) stub."""
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
        timeout = aiohttp.ClientTimeout(total=cfg["timeout"], connect=10)
        # Hard-ceiling defense against the slow-trickle case (matches
        # LLMClient): aiohttp's total= is unreliable on stalled sockets.
        hard_timeout = max(cfg["timeout"] + 30, 60)

        metrics = get_metrics()
        started = time.time()

        async def _do_call():
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status >= 400:
                        snippet = body[:200].replace("\n", " ")
                        raise RuntimeError(f"HTTP {resp.status}: {snippet}")
                    return await resp.json(content_type=None)

        try:
            data = await asyncio.wait_for(_do_call(), timeout=hard_timeout)
        except asyncio.TimeoutError:
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg=f"hard timeout {hard_timeout}s",
            )
            raise RuntimeError(f"timeout {hard_timeout}s")
        except aiohttp.ClientError as e:
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg=f"network: {e}",
            )
            raise RuntimeError(f"network: {e}")
        except RuntimeError as e:
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
        except (KeyError, IndexError, AttributeError, TypeError):
            metrics.record_deep_think_error(
                model=cfg["model"], error_msg="unexpected response shape",
            )
            logger.error(f"DeepThink unexpected response: {data}")
            raise RuntimeError("unexpected response shape")

        metrics.record_deep_think_success(
            model=cfg["model"], latency_ms=latency_ms,
            tokens_in=tokens_in, tokens_out=tokens_out,
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
        if self.mcp_manager is not None:
            mcp_tools = self.mcp_manager.all_tools()
            if policy is not None:
                mcp_tools = [t for t in mcp_tools if policy.allows_mcp(t.server_name)]
            schemas.extend(t.to_openai_tool() for t in mcp_tools)
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
    ) -> tuple[str, list[str], int, int]:
        """Drive the deep model through tool calls until it produces text.

        Returns (final_text, tool_call_history, total_tokens_in,
        total_tokens_out). Token totals aggregate across all rounds for
        the recent-calls dashboard view; per-round metrics are recorded
        inside `_chat_call` for the LLM/deep_think totals.

        Hitting the round cap returns whatever text the model has produced
        so far plus an honest warning — accuracy beats appearing helpful.
        """
        max_rounds = max(1, cfg["max_tool_rounds"])
        tool_call_history: list[str] = []
        total_in = 0
        total_out = 0

        for round_idx in range(max_rounds):
            assistant_msg = await self._chat_call(
                messages, tools=tools, cfg=cfg, sender_tail=sender_tail,
                group_id=group_id,
            )
            total_in += assistant_msg.pop("_dt_tokens_in", 0)
            total_out += assistant_msg.pop("_dt_tokens_out", 0)
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                # Deep models in thinking mode sometimes return reasoning
                # in a sibling field with empty content. Surface either.
                content = (assistant_msg.get("content") or "").strip()
                if not content:
                    content = (
                        assistant_msg.get("reasoning_content")
                        or assistant_msg.get("reasoning")
                        or ""
                    ).strip()
                logger.info(
                    f"DeepThink: final answer in round {round_idx + 1} "
                    f"({len(content)}c, {len(tool_call_history)} tools, "
                    f"{total_in}+{total_out} tok)"
                )
                return content, tool_call_history, total_in, total_out

            for call in tool_calls:
                fn_name = (call.get("function") or {}).get("name") or "?"
                tool_call_history.append(fn_name)
                await self._dispatch_tool_call(call, messages, caller_ctx, attachments)

        logger.warning(
            f"DeepThink: tool loop hit cap ({max_rounds} rounds). "
            f"Tools tried: {' -> '.join(tool_call_history)}"
        )
        return (
            f"(deep_think hit tool-loop cap of {max_rounds} rounds; "
            f"last tools: {' -> '.join(tool_call_history[-6:])})",
            tool_call_history,
            total_in,
            total_out,
        )

    async def _dispatch_tool_call(
        self,
        call: dict,
        messages: list[dict],
        caller_ctx,
        attachments: list,
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
                    if result and result.attachments:
                        attachments.extend(result.attachments)
            elif self.mcp_manager is not None:
                content = await self.mcp_manager.call_tool(name, args)
            else:
                content = f"ERROR: unknown tool {name}"
        except Exception as e:
            logger.warning(f"DeepThink tool {name} failed: {e}")
            content = f"ERROR: {e}"

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
