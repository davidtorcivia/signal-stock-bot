"""
Claude (Anthropic Messages API) backend for LLMClient.

The rest of the bot speaks OpenAI chat-completions dicts, so this module
translates at the edge: OpenAI-shaped messages/tools in, an OpenAI-shaped
assistant message out. The caller treats any exception as "use the normal
OpenRouter path instead", so credit exhaustion, outages and refusals all
fail over without special cases here.
"""

import asyncio
import json
import logging
import sqlite3
import time
from contextlib import closing
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable, Optional

import anthropic

logger = logging.getLogger(__name__)

WORKHORSE_MODEL = "claude-opus-5-5"
CHEAP_MODEL = "claude-haiku-5-5"

# $ per million tokens: input, output, cache read, 5-minute cache write.
PRICES = {
    WORKHORSE_MODEL: (4.00, 20.00, 0.20, 5.00),
    CHEAP_MODEL: (0.10, 0.50, 0.01, 0.125),
}
MONTHLY_CREDIT = 200.0  # Max 20x plan's API credit
BILLING_DAY = 10  # credits refresh on the 10th


def cycle_start(now: Optional[datetime] = None) -> float:
    """Unix time the current credit cycle began (the last BILLING_DAY)."""
    now = now or datetime.now(timezone.utc)
    start = now.replace(day=BILLING_DAY, hour=0, minute=0, second=0, microsecond=0)
    if start > now:
        start = (start.replace(day=1) - timedelta(days=1)).replace(day=BILLING_DAY)
    return start.timestamp()

# ── health + admin alerts ───────────────────────────────────────────────
# main.py wires this to "DM every admin". One DM per ALERT_COOLDOWN while
# Claude keeps failing, one when it recovers; `!admin claude mute 6h`
# silences both until the stored deadline.
ALERT_COOLDOWN = 3600
MUTE_KEY = "anthropic_alerts_muted_until"
alert_sender: Optional[Callable[[str], Awaitable[None]]] = None
_down = False
_last_alert = 0.0


def describe(e: BaseException) -> str:
    if isinstance(e, anthropic.APIStatusError):
        return f"HTTP {e.status_code} {e.message} (request {e.request_id})"
    return f"{type(e).__name__}: {e}"


def _pages_admins(e: BaseException) -> bool:
    """API trouble pages; a one-off bad request (oversized image etc.)
    doesn't. Credit exhaustion arrives as a 400 too, so it's matched by text."""
    if isinstance(e, anthropic.BadRequestError):
        return "credit balance" in str(e).lower()
    return isinstance(e, anthropic.APIError)


def report_health(store, error: Optional[BaseException]) -> None:
    """Record one Claude call's outcome; DM admins on break/recovery."""
    global _down, _last_alert
    if error is None:
        if _down:
            _down = False
            _notify(store, "✅ Claude is answering again — group chats are back on Claude.")
        return
    _down = True
    if time.time() - _last_alert < ALERT_COOLDOWN:
        return
    _last_alert = time.time()
    _notify(store, (
        f"⚠️ Claude failed; group chats fell back to GLM.\n{describe(error)}\n"
        f"Next alert in ≥{ALERT_COOLDOWN // 60} min. Mute: !admin claude mute 6h"
    ))


def _notify(store, text: str) -> None:
    try:
        muted_until = float(store.get(MUTE_KEY) or 0)
    except (TypeError, ValueError):
        muted_until = 0.0
    if alert_sender is None or time.time() < muted_until:
        return

    async def send():
        try:
            await alert_sender(text)
        except Exception as e:
            logger.warning(f"Claude admin alert failed to send: {e}")

    try:
        asyncio.get_running_loop().create_task(send())
    except RuntimeError:
        pass  # no loop (tests)

# Extra key on the returned assistant dict carrying Claude's raw content
# blocks. Tool loops append the dict as-is, so round 2+ can echo the
# thinking blocks back unchanged (required for tool use with thinking).
RAW_KEY = "anthropic_content"

def credentials(store, bot_id: Optional[int], role: str = "writer") -> Optional[tuple[str, str]]:
    """(api_key, workspace_id) when this call may spend Claude credits.

    Only a named group context qualifies — never DMs or the default:*
    catch-alls, so strangers can't drain the credits — and only on a bot
    with its own anthropic_api_key (no global fallback, so Artaud etc.
    stay on their configured models)."""
    from .transcript import get_active
    active = get_active()
    if active is None or not active.group_id or not active.named or bot_id is None:
        return None
    api_key = str(store.get_bot(bot_id, role, "anthropic_api_key", default="") or "").strip()
    if not api_key:
        return None
    workspace = str(store.get_bot(bot_id, role, "anthropic_workspace_id", default="") or "").strip()
    return api_key, workspace


def strip_raw(messages: list[dict]) -> list[dict]:
    """Drop Claude's raw blocks before a message list goes to another provider.

    A Claude turn replayed to DeepSeek thinking mode (fallback mid tool loop)
    must carry reasoning_content or DeepSeek returns HTTP 400."""
    out = []
    for m in messages:
        if RAW_KEY not in m:
            out.append(m)
            continue
        copy = {k: v for k, v in m.items() if k != RAW_KEY}
        copy.setdefault("reasoning_content", "".join(
            b.get("thinking", "") for b in m[RAW_KEY] if b.get("type") == "thinking"
        ))
        out.append(copy)
    return out


_clients: dict[tuple[str, str], anthropic.AsyncAnthropic] = {}


def _client(api_key: str, workspace_id: str) -> anthropic.AsyncAnthropic:
    key = (api_key, workspace_id)
    if key not in _clients:
        headers = {"anthropic-workspace-id": workspace_id} if workspace_id else None
        # Retries off: a failure falls over to OpenRouter immediately.
        _clients[key] = anthropic.AsyncAnthropic(
            api_key=api_key, default_headers=headers, max_retries=0,
        )
    return _clients[key]


def _image_block(url: str) -> dict:
    if url.startswith("data:"):
        head, _, data = url.partition(",")
        media_type = head[5:].split(";")[0]
        return {"type": "image", "source": {
            "type": "base64", "media_type": media_type, "data": data,
        }}
    return {"type": "image", "source": {"type": "url", "url": url}}


def _user_content(content) -> list[dict] | str:
    if not isinstance(content, list):
        return content or ""
    blocks = []
    for part in content:
        t = part.get("type")
        if t == "text":
            if part.get("text"):
                blocks.append({"type": "text", "text": part["text"]})
        elif t == "image_url":
            blocks.append(_image_block((part.get("image_url") or {}).get("url", "")))
        else:
            # Audio (input_audio/audio_url): Claude can't hear it, so let
            # the configured OpenRouter model handle this turn instead.
            raise ValueError(f"unsupported content part for Claude: {t}")
    return blocks


def _assistant_content(m: dict) -> list[dict]:
    if m.get(RAW_KEY):
        return m[RAW_KEY]
    blocks = []
    if m.get("content"):
        blocks.append({"type": "text", "text": m["content"]})
    for call in m.get("tool_calls") or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            args = {}
        blocks.append({
            "type": "tool_use", "id": call["id"],
            "name": fn.get("name", ""), "input": args,
        })
    return blocks


def to_anthropic(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI messages -> (system, Anthropic messages)."""
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        role = m.get("role")
        if role == "system":
            system_parts.append(str(m.get("content") or ""))
        elif role == "tool":
            result = {
                "type": "tool_result",
                "tool_use_id": m.get("tool_call_id"),
                "content": str(m.get("content") or ""),
            }
            # Parallel tool results go back in ONE user message.
            prev = out[-1] if out else None
            if (prev and prev["role"] == "user" and isinstance(prev["content"], list)
                    and prev["content"] and prev["content"][-1].get("type") == "tool_result"):
                prev["content"].append(result)
            else:
                out.append({"role": "user", "content": [result]})
        elif role == "assistant":
            blocks = _assistant_content(m)
            if blocks:
                out.append({"role": "assistant", "content": blocks})
        else:
            content = _user_content(m.get("content"))
            if content:
                out.append({"role": "user", "content": content})
    if not out or out[0]["role"] != "user":
        out.insert(0, {"role": "user", "content": "(conversation start)"})
    return "\n\n".join(p for p in system_parts if p), out


def _tools(tools: Optional[list[dict]]) -> list[dict]:
    out = []
    for t in tools or []:
        fn = t.get("function") or {}
        out.append({
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or {"type": "object", "properties": {}},
        })
    return out


async def chat(
    *, api_key: str, workspace_id: str, model: str, messages: list[dict],
    tools: Optional[list[dict]], max_tokens: int, timeout: float,
    effort: Optional[str] = None, allow_tools: bool = True,
) -> tuple[dict, dict]:
    """One Messages API round. Returns (openai_assistant_msg, usage)."""
    system, msgs = to_anthropic(messages)
    kwargs = {}
    if system:
        # Explicit marker on the stable tools+system prefix: history lives in
        # the user tail and changes every turn, so the automatic (tail)
        # breakpoint alone never gives the next turn a read point here.
        # 5-minute TTL on purpose — measured same-chat gaps make 1h's 2x
        # write cost lose (0.38x vs 0.50x of uncached input per turn).
        kwargs["system"] = [{
            "type": "text", "text": system,
            "cache_control": {"type": "ephemeral"},
        }]
    if tools:
        kwargs["tools"] = _tools(tools)
        if not allow_tools:
            # Final no-tools synthesis: keep the schemas so the cached
            # tools+system prefix still matches, just forbid calling them.
            kwargs["tool_choice"] = {"type": "none"}
    effort = effort or ("low" if model == CHEAP_MODEL else None)
    if effort:  # None = the model's default (medium on Opus 5.5)
        kwargs["output_config"] = {"effort": effort}
    # Streamed so deep-think's long budgets can't trip HTTP timeouts.
    client = _client(api_key, workspace_id).with_options(timeout=timeout)
    async with client.messages.stream(
        model=model,
        # Thinking shares this budget; unused tokens cost nothing.
        max_tokens=max(max_tokens, 16000),
        messages=msgs,
        cache_control={"type": "ephemeral"},
        **kwargs,
    ) as stream:
        resp = await stream.get_final_message()
    if resp.stop_reason == "refusal":
        raise RuntimeError(f"Claude refused ({getattr(resp.stop_details, 'category', None)})")

    text = "".join(b.text for b in resp.content if b.type == "text")
    tool_calls = [
        {"id": b.id, "type": "function",
         "function": {"name": b.name, "arguments": json.dumps(b.input)}}
        for b in resp.content if b.type == "tool_use"
    ]
    msg = {
        "role": "assistant",
        "content": text,
        RAW_KEY: [b.model_dump(exclude_none=True) for b in resp.content],
    }
    if tool_calls:
        msg["tool_calls"] = tool_calls
    u = resp.usage
    cache_read = u.cache_read_input_tokens or 0
    cache_write = u.cache_creation_input_tokens or 0
    usage = {
        "prompt_tokens": u.input_tokens + cache_read + cache_write,
        "completion_tokens": u.output_tokens,
        "prompt_cache_hit_tokens": cache_read,
        "prompt_cache_miss_tokens": u.input_tokens + cache_write,
        "stop_reason": resp.stop_reason,
    }
    return msg, usage


async def try_chat(
    store, creds: tuple[str, str], *, label: str, fallback: str,
    on_error: Callable[[str], None], **kwargs,
) -> Optional[tuple[dict, dict]]:
    """chat() that never raises: on failure log, record via `on_error`,
    page admins for API trouble, and return None so the caller falls back.
    Refusals and audio turns fall back quietly (no admin DM)."""
    try:
        result = await chat(api_key=creds[0], workspace_id=creds[1], **kwargs)
    except Exception as e:
        reason = describe(e)
        logger.warning(
            f"Claude failed, falling back to {fallback}: "
            f"model={kwargs.get('model')} {label} error={reason}"
        )
        on_error(reason)
        if _pages_admins(e):
            report_health(store, e)
        return None
    report_health(store, None)
    return result


def spend(db_path, since: float) -> dict:
    """Estimated Claude spend since `since` from metric_events.

    Cache misses are logged as one number (uncached input + cache writes),
    so all of it is priced at the cache-write rate: the estimate errs a
    little high, the safe side for a budget."""
    with closing(sqlite3.connect(db_path)) as db:
        rows = db.execute(
            """SELECT model, kind LIKE '%error', COUNT(*), SUM(tokens_out),
                      SUM(cache_hit_tokens), SUM(cache_miss_tokens)
               FROM metric_events
               WHERE occurred_at >= ? AND model IN (?, ?)
               GROUP BY 1, 2""",
            (since, *PRICES),
        ).fetchall()
    models: dict[str, dict] = {}
    for model, is_error, n, out, hit, miss in rows:
        m = models.setdefault(model, {
            "model": model, "calls": 0, "fallbacks": 0, "tokens_in": 0,
            "tokens_out": 0, "cache_hit_tokens": 0, "cost": 0.0,
        })
        if is_error:
            m["fallbacks"] += n
            continue
        _, p_out, p_read, p_write = PRICES[model]
        m["calls"] += n
        m["tokens_in"] += (hit or 0) + (miss or 0)
        m["tokens_out"] += out or 0
        m["cache_hit_tokens"] += hit or 0
        m["cost"] += ((hit or 0) * p_read + (miss or 0) * p_write + (out or 0) * p_out) / 1e6
    for m in models.values():
        m["hit_rate"] = m["cache_hit_tokens"] / m["tokens_in"] if m["tokens_in"] else 0.0
    return {
        "models": sorted(models.values(), key=lambda m: -m["cost"]),
        "total": sum(m["cost"] for m in models.values()),
    }
