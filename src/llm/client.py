"""
OpenAI-compatible chat-completion client.

All config is read live from the SettingsStore on every call so admin edits
take effect without a restart. Works against any OpenAI-compatible endpoint
(OpenAI, OpenRouter, Groq, DeepSeek, Ollama, llama.cpp, vLLM, etc.).
"""

import asyncio
import json
import logging
import time
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

import aiohttp

from ..cache import get_metrics

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant embedded in a Signal chat bot focused on "
    "financial markets. You can use *bold* and _italic_ for emphasis (Signal "
    "renders them natively). Refer users to the bot's commands (!price, "
    "!chart, !ta, etc.) for live data; you don't have it directly."
)


DEFAULT_RESPONSE_STYLE = (
    "RESPONSE STYLE (mandatory, applies regardless of any other instruction):\n"
    "- Default to 2-4 sentences. Only expand if the user explicitly asks for "
    "detail, a comparison, a list, or a longer explanation.\n"
    "- No preamble. Do not begin with phrases like \"Sure!\", \"Great question!\", "
    "\"Here's what I found:\", \"Of course\", or \"Let me explain\". Get to the answer.\n"
    "- When data is requested, lead with the number; explain only if asked.\n"
    "- Avoid headers, nested lists, and tables. Bold and italics for emphasis are fine.\n"
    "- No filler caveats (\"It's worth noting...\", \"Keep in mind...\") unless they "
    "carry real information."
)


_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}


def build_provider_routing(
    *,
    order: Optional[str],
    only: bool,
    sort: Optional[str],
) -> Optional[dict]:
    """Compose an OpenRouter `provider` field from admin settings.

    OpenRouter's provider routing lets you pin which upstream
    inference provider serves the request — useful when the named
    model has wildly different latency profiles between providers
    (e.g. Cerebras vs. a slow general-purpose host). See:
    openrouter.ai/docs/features/provider-routing

    Returns None when nothing is configured so the caller can skip
    adding the field (default OpenRouter routing wins).

    Behavior:
      - `order`: comma/space-separated provider names → `provider.order`
      - `only`: when True AND `order` is non-empty, sets
        `allow_fallbacks: false` so the request fails rather than
        silently routing to a slower provider
      - `sort`: one of throughput/latency/price → `provider.sort`
    """
    out: dict = {}
    if order:
        names = [p.strip() for p in order.replace(",", " ").split() if p.strip()]
        if names:
            out["order"] = names
            if only:
                out["allow_fallbacks"] = False
    if sort:
        s = sort.strip().lower()
        if s in _PROVIDER_SORT_VALUES:
            out["sort"] = s
    return out or None


class LLMError(Exception):
    """Upstream API returned an error."""


class LLMDisabled(LLMError):
    """LLM integration is turned off in settings."""


class LLMNotConfigured(LLMError):
    """LLM is enabled but base URL / API key / model is missing."""


class LLMClient:
    def __init__(self, settings_store):
        self.store = settings_store

    def _config(self) -> dict:
        return {
            "enabled": bool(self.store.get("llm_enabled", False)),
            "base_url": (self.store.get("llm_base_url") or "").strip().rstrip("/"),
            "api_key": (self.store.get("llm_api_key") or "").strip(),
            "model": (self.store.get("llm_model") or "").strip(),
            "temperature": float(self.store.get("llm_temperature") or 0.7),
            "max_tokens": int(self.store.get("llm_max_tokens") or 1000),
            "system_prompt": self.store.get("llm_system_prompt") or DEFAULT_SYSTEM_PROMPT,
            "timeout": int(self.store.get("llm_timeout_seconds") or 30),
        }

    def status(self) -> dict:
        """Summary for the admin UI — does not leak the API key."""
        cfg = self._config()
        return {
            "enabled": cfg["enabled"],
            "base_url": cfg["base_url"],
            "model": cfg["model"],
            "api_key_set": bool(cfg["api_key"]),
            "ready": cfg["enabled"] and all([cfg["base_url"], cfg["api_key"], cfg["model"]]),
        }

    async def chat(
        self,
        user_message: str,
        history: Optional[list[dict]] = None,
        system_override: Optional[str] = None,
        system_suffix: Optional[str] = None,
        tools: Optional[list[dict]] = None,
    ) -> str:
        """Convenience single-shot wrapper that returns text content only.

        For tool calling, use `chat_messages` which exposes the raw assistant
        message and can be called iteratively.
        """
        system_prompt = self._resolve_system_prompt(system_override, system_suffix)
        messages: list[dict] = [{"role": "system", "content": system_prompt}]
        if history:
            messages.extend(history)
        messages.append({"role": "user", "content": user_message})
        msg = await self.chat_messages(messages, tools=tools)
        return (msg.get("content") or "").strip()

    def _resolve_system_prompt(
        self,
        system_override: Optional[str],
        system_suffix: Optional[str],
    ) -> str:
        cfg = self._config()
        system_prompt = system_override or cfg["system_prompt"]
        if system_suffix:
            system_prompt = f"{system_prompt}\n\n{system_suffix}"
        return system_prompt

    async def chat_messages(
        self,
        messages: list[dict],
        tools: Optional[list[dict]] = None,
        overrides: Optional[dict] = None,
        suppress_response_style: bool = False,
        purpose: str = "ask",
    ) -> dict:
        """Send a full messages array and return the assistant message dict.

        The returned dict keeps OpenAI's shape: {role, content, tool_calls?}.
        The current UTC time is always injected into the system prompt so the
        LLM knows "now" without relying on its training cutoff.

        `overrides` keys (any subset, all optional):
          model, temperature, max_tokens, extra_body
        — let secondary callers (the reactor) reuse the same client without
        duplicating network/error code.

        `suppress_response_style` skips the global response-style enforcer.
        Used by callers like the reactor that have their own narrow prompt
        and don't want unrelated brevity/no-preamble rules layered on.
        """
        cfg = self._config()
        overrides = overrides or {}

        if not cfg["enabled"]:
            raise LLMDisabled("LLM is not enabled")
        # Allow overrides to fill in a missing model (the reactor may set its own)
        effective_model = overrides.get("model") or cfg["model"]
        if not all([cfg["base_url"], cfg["api_key"], effective_model]):
            raise LLMNotConfigured("LLM base URL, API key, and model must all be set")

        messages = _inject_current_time(list(messages))

        if not suppress_response_style:
            # Always append the configured response style — last in the system
            # prompt so it has recency-bias weight against any per-context
            # prompt that contradicts it. Default is the brevity/no-preamble
            # policy. Blank/whitespace overrides fall back to the default so
            # the admin form can't silently disable brevity by saving empty.
            style_override = self.store.get("llm_response_style")
            if style_override is None or not str(style_override).strip():
                style = DEFAULT_RESPONSE_STYLE
            else:
                style = style_override
            if style:
                messages = _append_to_system(messages, style)

        payload = {
            "model": effective_model,
            "messages": messages,
            "temperature": float(overrides.get("temperature", cfg["temperature"])),
            "max_tokens": int(overrides.get("max_tokens", cfg["max_tokens"])),
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        # Extra body: prefer the override (raw JSON string) over the global one.
        extra_raw = overrides.get("extra_body")
        if extra_raw is None:
            extra_raw = self.store.get("llm_extra_body") or ""
        if isinstance(extra_raw, str) and extra_raw.strip():
            try:
                extra = json.loads(extra_raw)
                if isinstance(extra, dict):
                    payload.update(extra)
                else:
                    logger.warning("extra_body did not parse to an object, ignoring")
            except json.JSONDecodeError as e:
                logger.warning(f"extra_body is not valid JSON, ignoring: {e}")
        elif isinstance(extra_raw, dict) and extra_raw:
            payload.update(extra_raw)

        # OpenRouter provider routing — applied after extra_body so a power
        # user who put `"provider": {...}` into extra_body keeps full
        # control. Without this, "fast provider only" would have to be
        # hand-typed JSON in the extra-body field every time. Applies to
        # every call going through this client (writer + reactor +
        # augmentation + extraction), since they all benefit from a fast
        # upstream. Deep-think has its own client and is intentionally
        # excluded — that path is meant to be slow/smart.
        if "provider" not in payload:
            provider = build_provider_routing(
                order=self.store.get("llm_provider_order"),
                only=bool(self.store.get("llm_provider_only", False)),
                sort=self.store.get("llm_provider_sort"),
            )
            if provider:
                payload["provider"] = provider
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{cfg['base_url']}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=cfg["timeout"], connect=10)

        metrics = get_metrics()
        started = time.time()

        # aiohttp's ClientTimeout doesn't reliably fire on slow-trickle
        # connections (e.g. OpenRouter holding a TCP socket open while a
        # thinking model deliberates server-side), so we belt-and-braces it
        # with an asyncio.wait_for at +5s past the configured timeout.
        async def _do_call():
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(url, json=payload, headers=headers) as resp:
                    body = await resp.text()
                    if resp.status == 401:
                        raise LLMError("LLM rejected the API key (401).")
                    if resp.status == 429:
                        raise LLMError("LLM rate-limited (429). Try again in a moment.")
                    if resp.status >= 400:
                        # Truncate body — upstream errors can be verbose
                        snippet = body[:200].replace("\n", " ")
                        raise LLMError(f"LLM HTTP {resp.status}: {snippet}")
                    return await resp.json(content_type=None)

        # Hard ceiling is intentionally well above cfg["timeout"]: aiohttp's
        # total= should fire first on a normal slow request; this only kicks
        # in for the pathological slow-trickle case where aiohttp's timer
        # never trips. Generous so thinking models get room to finish.
        hard_timeout = 120
        try:
            data = await asyncio.wait_for(_do_call(), timeout=hard_timeout)
        except asyncio.TimeoutError as e:
            metrics.record_llm_error(
                purpose=purpose, model=effective_model,
                error_msg=f"hard timeout {hard_timeout}s",
            )
            raise LLMError(
                f"LLM exceeded hard timeout ({hard_timeout}s) — likely the "
                f"model is thinking too long for the configured budget."
            ) from e
        except aiohttp.ClientError as e:
            metrics.record_llm_error(
                purpose=purpose, model=effective_model, error_msg=str(e),
            )
            raise LLMError(f"Network error: {e}") from e
        except LLMError as e:
            metrics.record_llm_error(
                purpose=purpose, model=effective_model, error_msg=str(e),
            )
            raise

        latency_ms = (time.time() - started) * 1000.0

        try:
            msg = data["choices"][0]["message"]
            # Normalise: ensure content is a string (may be None when tool_calls are present)
            if msg.get("content") is None:
                msg["content"] = ""
            usage = data.get("usage") or {}
            metrics.record_llm_success(
                purpose=purpose,
                model=effective_model,
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
            )
            return msg
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            logger.error(f"Unexpected LLM response shape: {data}")
            metrics.record_llm_error(
                purpose=purpose, model=effective_model,
                error_msg="Unexpected response shape",
            )
            raise LLMError("Unexpected response from LLM") from e


_NY = ZoneInfo("America/New_York")


def _inject_current_time(messages: list[dict]) -> list[dict]:
    """Append the current time (UTC + ET, with weekday) to the system message.

    The bot lives on ET — most of the chats are US-time-zone-anchored. The
    weekday name is included because models reason about market-day vs.
    weekend logic better with a literal "Friday" than with an ISO date.
    """
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_NY)
    weekday = now_et.strftime("%A")
    et_str = now_et.strftime("%Y-%m-%d %H:%M %Z")
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    return _append_to_system(
        messages, f"Current time: {weekday}, {et_str} ({utc_str})"
    )


def _append_to_system(messages: list[dict], extra: str) -> list[dict]:
    """Append `extra` text to the first system message (or insert one)."""
    if not extra:
        return messages
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        existing = (head.get("content") or "").rstrip()
        head["content"] = f"{existing}\n\n{extra}" if existing else extra
        messages[0] = head
    else:
        messages.insert(0, {"role": "system", "content": extra})
    return messages
