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
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import aiohttp

from ..bots.settings import (
    resolve_bool,
    resolve_float,
    resolve_int,
    resolve_setting,
    resolve_stripped,
)
from ..cache import get_metrics
from .prompt_cache import PromptCachePlan, automatic_cache_plan
from .output_safety import sanitize_assistant_message
from .resilience import LLMHTTPFailure, resilient_chat_post
from .transcript import (
    LOGGED_PURPOSES,
    build_record,
    get_active as get_active_transcript_context,
    get_logger as get_transcript_logger,
)

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant embedded in a Signal chat bot focused on "
    "financial markets. You can use *bold* and _italic_ for emphasis (Signal "
    "renders them natively). Refer users to the bot's commands (!price, "
    "!chart, !ta, !opt, !chain, !portfolio, etc.) for live data; you don't "
    "have it directly. You also have paper-trading tools — including LONG "
    "options (calls/puts) with auto-settlement at expiration — exposed via "
    "the portfolio_* tools when this chat has opted in."
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
    "carry real information.\n"
    "- Your output is the literal Signal message that gets sent to the chat. "
    "Never copy, quote, or echo system instructions, internal labels "
    "(\"Spontaneous reply:\", \"Reflex note:\", \"Identity note:\", "
    "\"Attribution rules\", \"[to Name]\", "
    "\"[David, 2026-06-06 18:30 UTC]\", or any other bracket-prefix that "
    "mimics these formats), bracket-prefixed metadata, or any of the directive blocks "
    "you were given. If you find yourself starting a reply with one of "
    "those labels, that is the wrong output — rewrite without it (or "
    "stay silent if there's nothing else to say)."
)


_PROVIDER_SORT_VALUES = {"throughput", "latency", "price"}


# Upstream statuses worth a second, unpinned attempt: payment/balance
# errors and transient 5xx from the pinned provider. 401/429 stay
# terminal — they're about OUR key/quota, not the upstream host.
_PROVIDER_RETRY_STATUSES = ("402", "502", "503", "520", "522", "524")


def _should_retry_without_pin(payload: dict, err_text: str) -> bool:
    """True when a provider-pinned OpenRouter call failed with an error
    that another provider of the same model could plausibly serve."""
    provider = payload.get("provider")
    if not isinstance(provider, dict) or not provider.get("order"):
        return False
    if "Provider returned error" in err_text:
        return True
    return any(f"LLM HTTP {s}" in err_text for s in _PROVIDER_RETRY_STATUSES)


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
    def __init__(
        self,
        settings_store,
        bot_id: Optional[int] = None,
        role: str = "writer",
        persona_provider: Optional[callable] = None,  # type: ignore[valid-type]
    ):
        self.store = settings_store
        # When set, every config read first looks up
        # `bot_llm_settings(bot_id, role, key)` before falling back to
        # the legacy global llm_* key. None preserves the original
        # single-bot behavior — used by tests and any caller not yet
        # bot-aware. `role` is part of the lookup tuple so a future
        # variant (e.g. an extraction-specific writer) can share this
        # class without colliding on settings.
        self.bot_id = bot_id
        self.role = role
        # Optional bot_id -> persona string callable. When set, the
        # writer's system-prompt resolver consults the bot's persona
        # column between the explicit bot_llm_settings override and
        # the global llm_system_prompt fallback. Without this hook a
        # non-default bot whose persona lives only in `bots.persona`
        # would inherit the global Sigil prompt — leaking identity
        # across bots.
        self.persona_provider = persona_provider
        # Long-lived session reused across all chat calls. Without this,
        # every call (writer, reactor, deep_think extractors, augmentation)
        # opens a fresh TCP+TLS connection — a measurable hit at scale and
        # a hard ceiling on connection-pool reuse to OpenRouter/etc.
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

    def _config(self) -> dict:
        bot_id, role = self.bot_id, self.role
        store = self.store
        # System-prompt resolution chain (first non-empty wins):
        #   1. bot_llm_settings(bot_id, role, 'system_prompt')  — admin's
        #      explicit per-bot writer override on /admin/bots/<id>/llm
        #   2. bots.persona                                      — the bot's
        #      identity text on /admin/bots/<id> (resolved via the persona
        #      provider, so this class doesn't need a BotRegistry import)
        #   3. admin_settings.llm_system_prompt                 — global
        #   4. DEFAULT_SYSTEM_PROMPT                            — built-in
        sys_prompt_raw: Optional[str] = None
        if bot_id is not None:
            override = store.get_bot(bot_id, role, "system_prompt", default=None)
            if override:
                sys_prompt_raw = override
        if sys_prompt_raw is None and bot_id is not None and self.persona_provider is not None:
            try:
                persona = self.persona_provider(bot_id)
            except Exception as e:
                logger.debug(f"persona_provider({bot_id}) raised: {e}")
                persona = None
            if persona:
                sys_prompt_raw = persona
        if sys_prompt_raw is None:
            sys_prompt_raw = store.get("llm_system_prompt")
        max_tokens = resolve_int(
            store, bot_id, role, "max_tokens",
            global_keys=["llm_max_tokens"], default=1000,
            min_value=1,
        )
        extended_max_tokens = resolve_int(
            store, bot_id, role, "extended_max_tokens",
            global_keys=["llm_extended_max_tokens"], default=1000,
            min_value=1,
        )
        return {
            "enabled": resolve_bool(
                store, bot_id, role, "enabled",
                global_keys=["llm_enabled"], default=False,
            ),
            "base_url": resolve_stripped(
                store, bot_id, role, "base_url",
                global_keys=["llm_base_url"],
            ).rstrip("/"),
            "api_key": resolve_stripped(
                store, bot_id, role, "api_key",
                global_keys=["llm_api_key"],
            ),
            "model": resolve_stripped(
                store, bot_id, role, "model",
                global_keys=["llm_model"],
            ),
            "temperature": resolve_float(
                store, bot_id, role, "temperature",
                global_keys=["llm_temperature"], default=0.7,
            ),
            "max_tokens": max_tokens,
            # Never make a request smaller because an inherited extended
            # ceiling lagged behind a bot-specific conversational budget.
            "extended_max_tokens": max(
                max_tokens, extended_max_tokens,
            ),
            "system_prompt": sys_prompt_raw or DEFAULT_SYSTEM_PROMPT,
            "timeout": resolve_int(
                store, bot_id, role, "timeout_seconds",
                global_keys=["llm_timeout_seconds"], default=30,
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
        cache_plan: Optional[PromptCachePlan] = None,
    ) -> dict:
        """Send a full messages array and return the assistant message dict.

        The returned dict keeps OpenAI's shape: {role, content, tool_calls?}.
        The current UTC + ET time is always prepended to the last user
        message so the LLM knows "now" without relying on its training
        cutoff. Kept out of the system block to preserve DeepSeek's
        prefix-match cache — see `_inject_current_time`.

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
        messages = _ensure_reasoning_content(messages)

        style = ""
        if not suppress_response_style:
            # Always append the configured response style — last in the system
            # prompt so it has recency-bias weight against any per-context
            # prompt that contradicts it. Default is the brevity/no-preamble
            # policy. Blank/whitespace overrides fall back to the default so
            # the admin form can't silently disable brevity by saving empty.
            style_override = resolve_setting(
                self.store, self.bot_id, self.role, "response_style",
                global_keys=["llm_response_style"],
            )
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
            extra_raw = resolve_setting(
                self.store, self.bot_id, self.role, "extra_body",
                global_keys=["llm_extra_body"], default="",
            ) or ""
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
                order=resolve_setting(
                    self.store, self.bot_id, self.role, "provider_order",
                    global_keys=["llm_provider_order"],
                ),
                only=resolve_bool(
                    self.store, self.bot_id, self.role, "provider_only",
                    global_keys=["llm_provider_only"], default=False,
                ),
                sort=resolve_setting(
                    self.store, self.bot_id, self.role, "provider_sort",
                    global_keys=["llm_provider_sort"],
                ),
            )
            if provider:
                payload["provider"] = provider
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{cfg['base_url']}/chat/completions"

        # One-line summary of the outbound payload's shape — specifically:
        # whether any user message uses the OpenAI multimodal parts array.
        # Lets us confirm from logs alone that vision payloads reach the
        # POST body in the correct form (vs. having been flattened to a
        # plain string somewhere upstream). Cheap — just counts parts.
        _log_payload_shape(payload, purpose=purpose, url=url)

        metrics = get_metrics()
        started = time.time()
        session = await self._get_session()

        # Snapshot the ACTUAL rendered request after time/style injection.
        # Copy the caller's declarations so repeated tool rounds cannot append
        # response_style to the same plan over and over.
        declared = cache_plan or automatic_cache_plan(messages)
        runtime_plan = PromptCachePlan(
            stable_blocks=list(declared.stable_blocks),
            volatile_blocks=list(declared.volatile_blocks),
        )
        if style and cache_plan is not None:
            runtime_plan.with_stable("response_style", style)
        active = get_active_transcript_context()
        cache_manifest = runtime_plan.snapshot(
            messages,
            tools,
            purpose=purpose,
            context_id=(active.context_id if active is not None else None),
            bot_id=(active.bot_id if active is not None else self.bot_id),
        )

        # Retries and pinned-provider fallback share this one deadline. This
        # retains the slow-trickle backstop without multiplying the configured
        # timeout by the number of attempts.
        hard_timeout = cfg["timeout"] + 15
        host = urlparse(cfg["base_url"]).netloc or cfg["base_url"]
        provider_metrics = metrics.get_provider_metrics(
            f"llm:{host}:{effective_model}"
        )

        def _on_retry(reason: str, attempt: int, delay: float) -> None:
            metrics.record_llm_retry(
                reason=reason, purpose=purpose, model=effective_model,
            )
            logger.warning(
                "LLM retry reason=%s after_attempt=%s delay=%.2fs",
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
                should_retry_unpinned=_should_retry_without_pin,
                on_retry=_on_retry,
            )
        except asyncio.TimeoutError as e:
            metrics.record_llm_error(
                purpose=purpose, model=effective_model,
                error_msg=f"hard timeout {hard_timeout}s",
            )
            raise LLMError(
                f"LLM exceeded hard timeout ({hard_timeout}s) — likely the "
                f"model is thinking too long for the configured budget."
            ) from e
        except LLMHTTPFailure as e:
            if e.status == 401:
                message = "LLM rejected the API key (401)."
            elif e.status == 429:
                message = "LLM rate-limited after bounded retries (429)."
            else:
                message = str(e)
            metrics.record_llm_error(
                purpose=purpose, model=effective_model, error_msg=message,
            )
            raise LLMError(message) from e
        except aiohttp.ClientError as e:
            metrics.record_llm_error(
                purpose=purpose, model=effective_model, error_msg=str(e),
            )
            raise LLMError(f"Network error: {e}") from e
        except RuntimeError as e:
            if "circuit is open" in str(e):
                metrics.record_llm_circuit_rejection(
                    purpose=purpose, model=effective_model,
                )
            metrics.record_llm_error(
                purpose=purpose, model=effective_model, error_msg=str(e),
            )
            raise LLMError(str(e)) from e
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
            cache_hit, cache_miss = _extract_cache_tokens(usage)
            metrics.record_llm_success(
                purpose=purpose,
                model=effective_model,
                latency_ms=latency_ms,
                tokens_in=usage.get("prompt_tokens", 0),
                tokens_out=usage.get("completion_tokens", 0),
                cache_hit_tokens=cache_hit,
                cache_miss_tokens=cache_miss,
            )
            cache_event = metrics.record_prompt_cache_observation(
                cache_manifest,
                cache_hit_tokens=cache_hit,
                cache_miss_tokens=cache_miss,
                latency_ms=latency_ms,
            )
            # Per-context transcript capture. Only writer rounds (ask /
            # augment) — reactor/summary/healthcheck are intentionally
            # excluded. Fire-and-forget background task so the write
            # never blocks the reply path.
            self._maybe_capture_transcript(
                purpose=purpose,
                model=effective_model,
                messages=messages,
                tools=tools,
                assistant_msg=msg,
                payload=payload,
                latency_ms=latency_ms,
                usage=usage,
                cache_manifest=cache_manifest,
                cache_event=cache_event,
            )
            return msg
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            logger.error(f"Unexpected LLM response shape: {data}")
            metrics.record_llm_error(
                purpose=purpose, model=effective_model,
                error_msg="Unexpected response shape",
            )
            raise LLMError("Unexpected response from LLM") from e

    def _maybe_capture_transcript(
        self,
        *,
        purpose: str,
        model: str,
        messages: list[dict],
        tools: Optional[list[dict]],
        assistant_msg: dict,
        payload: dict,
        latency_ms: float,
        usage: dict,
        cache_manifest: dict,
        cache_event: dict,
    ) -> None:
        """Append the round to this context's JSONL — if logging is on.

        Three gates: purpose must be a writer purpose, an active context
        must be set (dispatcher set it at message entry), and that
        context's `transcript_logging_enabled` must be True. All snapshotted
        at message-handling start, so toggling the flag mid-flight doesn't
        retroactively log in-progress calls.

        Schedules the disk write as a background task so the reply path
        doesn't pay for I/O serialization.
        """
        if purpose not in LOGGED_PURPOSES:
            return
        active = get_active_transcript_context()
        if active is None or not active.transcript_logging_enabled:
            return
        # Snapshot params that affect the model's output (provider routing
        # included so admins can tell which upstream served the round).
        params = {
            "temperature": payload.get("temperature"),
            "max_tokens": payload.get("max_tokens"),
        }
        if "provider" in payload:
            params["provider"] = payload["provider"]
        record = build_record(
            purpose=purpose,
            model=model,
            messages=messages,
            tools=tools,
            assistant_msg=sanitize_assistant_message(assistant_msg),
            params=params,
            latency_ms=latency_ms,
            bot_id=active.bot_id,
            context_id=active.context_id,
            context_label=active.context_label,
            sender_tail=active.sender_tail,
            group_id=active.group_id,
            usage=usage,
            cache_manifest=cache_manifest,
            cache_event=cache_event,
        )
        try:
            asyncio.create_task(
                get_transcript_logger().append(active.context_id, record)
            )
        except RuntimeError:
            # No running loop (test environment) — drop silently.
            pass

    async def health_check(self, prompt: Optional[str] = None) -> dict:
        """Single round-trip connectivity test against the bot's own
        config. Never raises — every failure mode (disabled,
        unconfigured, network, auth, slow upstream, malformed reply)
        resolves to `{ok: False, error: ...}` so the admin UI can
        render specifics without a 500 page. Used by the test-connection
        button on /admin/bots/<id>.

        Skips response-style + history injection so the test exercises
        the bare endpoint path. Records metrics under purpose=
        'healthcheck' so synthetic test calls don't pollute the
        writer's per-bot rollups."""
        cfg = self._config()
        if not cfg["enabled"]:
            return {
                "ok": False,
                "error": "LLM is disabled for this bot/role.",
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
            msg = await self.chat_messages(
                [{"role": "user", "content": user_msg}],
                suppress_response_style=True,
                purpose="healthcheck",
            )
        except (LLMDisabled, LLMNotConfigured, LLMError) as e:
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


_NY = ZoneInfo("America/New_York")


def _log_payload_shape(payload: dict, *, purpose: str, url: str) -> None:
    """Emit a one-line summary of the chat-completions payload shape.

    Records: total msg count, role breakdown, and for every user message
    whose `content` is the OpenAI multimodal list-of-parts form — the
    part types + image counts + total base64 length. Lets us confirm
    from logs alone that vision payloads survive the trip from inbound
    parsing to the HTTP body in the expected shape. Skipped silently on
    any error so a diagnostic line never breaks the request path.
    """
    try:
        messages = payload.get("messages") or []
        roles: dict[str, int] = {}
        multimodal_summaries: list[str] = []
        for i, m in enumerate(messages):
            role = m.get("role") or "?"
            roles[role] = roles.get(role, 0) + 1
            content = m.get("content")
            if isinstance(content, list):
                types: dict[str, int] = {}
                total_image_b64 = 0
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    t = part.get("type") or "?"
                    types[t] = types.get(t, 0) + 1
                    if t == "image_url":
                        url_field = (part.get("image_url") or {}).get("url") or ""
                        total_image_b64 += len(url_field)
                parts_str = ",".join(f"{t}x{n}" for t, n in types.items())
                multimodal_summaries.append(
                    f"msg[{i}]={role}({parts_str},b64={total_image_b64}b)"
                )
        msg_count = len(messages)
        roles_str = ",".join(f"{r}={n}" for r, n in sorted(roles.items()))
        endpoint = url.rsplit("/", 2)[-2] if "/" in url else url
        if multimodal_summaries:
            logger.info(
                f"LLM payload [{purpose}@{endpoint}]: {msg_count} msgs "
                f"({roles_str}) | multimodal: {' '.join(multimodal_summaries)}"
            )
        else:
            logger.debug(
                f"LLM payload [{purpose}@{endpoint}]: {msg_count} msgs ({roles_str})"
            )
    except Exception as e:
        logger.debug(f"_log_payload_shape failed: {e}")


def _extract_cache_tokens(usage: dict) -> tuple[int, int]:
    """Pull (cache_hit_tokens, cache_miss_tokens) from a usage block.

    DeepSeek returns `prompt_cache_hit_tokens` + `prompt_cache_miss_tokens`
    at the top level of `usage`. OpenAI nests cached tokens under
    `prompt_tokens_details.cached_tokens` with no miss field, so we infer
    miss from `prompt_tokens - cached`. Returns (0, 0) when neither shape
    is present (provider doesn't report cache, or hasn't cached yet).
    """
    if not isinstance(usage, dict) or not usage:
        return 0, 0
    if (
        usage.get("prompt_cache_hit_tokens") is not None
        or usage.get("prompt_cache_miss_tokens") is not None
    ):
        return (
            int(usage.get("prompt_cache_hit_tokens") or 0),
            int(usage.get("prompt_cache_miss_tokens") or 0),
        )
    details = usage.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") if isinstance(details, dict) else None
    if cached is not None:
        hit = int(cached or 0)
        miss = max(0, int(usage.get("prompt_tokens") or 0) - hit)
        return hit, miss
    return 0, 0


def _ensure_reasoning_content(messages: list[dict]) -> list[dict]:
    """Mirror OpenRouter's `reasoning` / `reasoning_details` onto each
    assistant turn's `reasoning_content`.

    DeepSeek's thinking-mode API requires the prior assistant turn's
    `reasoning_content` to be present on round-trip (rounds 2+ of any
    tool loop). OpenRouter exposes the same thing on response as
    `reasoning` (string) or `reasoning_details` (provider-structured
    array), but does NOT translate it back when `llm_provider_only=true`
    routes direct to DeepSeek. Without this mirror, every tool loop
    dies on round 2 with HTTP 400 "reasoning_content in the thinking
    mode must be passed back to the API".

    Shallow-copies any message we touch so the caller's list (often the
    in-memory tool-loop state) keeps its original shape.
    """
    out: list[dict] = []
    for m in messages:
        if (
            m.get("role") != "assistant"
            or m.get("reasoning_content")
            or not (m.get("reasoning") or m.get("reasoning_details"))
        ):
            out.append(m)
            continue
        text = m.get("reasoning") or ""
        if not text and isinstance(m.get("reasoning_details"), list):
            parts = []
            for d in m["reasoning_details"]:
                if isinstance(d, dict):
                    t = d.get("text") or d.get("content")
                    if t:
                        parts.append(str(t))
            text = "\n".join(parts)
        copy = dict(m)
        copy["reasoning_content"] = text or ""
        out.append(copy)
    return out


def _inject_current_time(messages: list[dict]) -> list[dict]:
    """Prepend the current time (UTC + ET, with weekday) to the last user
    message — explicitly NOT to the system block.

    The bot lives on ET — most of the chats are US-time-zone-anchored. The
    weekday name is included because models reason about market-day vs.
    weekend logic better with a literal "Friday" than with an ISO date.

    Why the last user message and not the system prompt: DeepSeek (the
    primary backend) does prefix-match caching with hours-to-days TTL.
    Putting a string that changes every minute inside the system block
    forces a cache miss on every call — system text, tool schemas, and
    prior turns all become un-cacheable. Stashing the timestamp at the
    tail of the last user message keeps the entire prefix stable so the
    cache hits on the bulk of the payload.

    Walks backward so a mid-tool-loop messages array (ending in tool
    responses) still attaches the time to the user's question rather than
    to an intermediate tool message.
    """
    now_utc = datetime.now(timezone.utc)
    now_et = now_utc.astimezone(_NY)
    weekday = now_et.strftime("%A")
    et_str = now_et.strftime("%Y-%m-%d %H:%M %Z")
    utc_str = now_utc.strftime("%Y-%m-%d %H:%M UTC")
    # Spell the ET time out in 12-hour form and state the live UTC offset.
    # Everything else in the prompt (history labels, group-context stamps)
    # is UTC, and a writer model asked "what time is it" will otherwise
    # anchor on those and apply its own idea of the ET offset — the local
    # writer believed ET was UTC-6 and answered two hours early. Giving
    # the colloquial time verbatim leaves it no arithmetic to get wrong.
    twelve_hr = now_et.strftime("%I:%M %p").lstrip("0")
    utc_off = now_et.utcoffset()
    offset_hours = int(utc_off.total_seconds() // 3600) if utc_off else 0
    time_line = (
        f"[Current time: {weekday}, {et_str} — {twelve_hr} Eastern "
        f"({utc_str}; ET is UTC{offset_hours:+d})]"
    )

    out = list(messages)
    for i in range(len(out) - 1, -1, -1):
        if out[i].get("role") != "user":
            continue
        head = dict(out[i])
        content = head.get("content")
        if isinstance(content, list):
            # Vision multimodal: prepend a text part so the time lands
            # before any image parts the model has to process.
            head["content"] = [
                {"type": "text", "text": time_line}, *content,
            ]
        else:
            existing = (content or "")
            head["content"] = (
                f"{time_line}\n\n{existing}" if existing else time_line
            )
        out[i] = head
        return out
    # No user message in the array — fall back to appending one so the
    # model still sees the time. Defensive; our call sites all include a
    # user turn.
    out.append({"role": "user", "content": time_line})
    return out


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
