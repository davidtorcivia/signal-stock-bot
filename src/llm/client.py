"""
OpenAI-compatible chat-completion client.

All config is read live from the SettingsStore on every call so admin edits
take effect without a restart. Works against any OpenAI-compatible endpoint
(OpenAI, OpenRouter, Groq, DeepSeek, Ollama, llama.cpp, vLLM, etc.).
"""

import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


DEFAULT_SYSTEM_PROMPT = (
    "You are a concise assistant embedded in a Signal chat bot focused on "
    "financial markets. Answer in plain text without markdown formatting, "
    "keep responses short (under 400 words unless asked for more), and be "
    "direct. You do not have access to live market data; refer the user to "
    "the bot's other commands (like !price, !chart, !ta) for current prices."
)


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
    ) -> dict:
        """Send a full messages array and return the assistant message dict.

        The returned dict keeps OpenAI's shape: {role, content, tool_calls?}.
        The current UTC time is always injected into the system prompt so the
        LLM knows "now" without relying on its training cutoff.
        """
        cfg = self._config()

        if not cfg["enabled"]:
            raise LLMDisabled("LLM is not enabled")
        if not all([cfg["base_url"], cfg["api_key"], cfg["model"]]):
            raise LLMNotConfigured("LLM base URL, API key, and model must all be set")

        messages = _inject_current_time(list(messages))

        payload = {
            "model": cfg["model"],
            "messages": messages,
            "temperature": cfg["temperature"],
            "max_tokens": cfg["max_tokens"],
            "stream": False,
        }
        if tools:
            payload["tools"] = tools

        extra_raw = self.store.get("llm_extra_body") or ""
        if extra_raw.strip():
            try:
                extra = json.loads(extra_raw)
                if isinstance(extra, dict):
                    payload.update(extra)
                else:
                    logger.warning("llm_extra_body did not parse to an object, ignoring")
            except json.JSONDecodeError as e:
                logger.warning(f"llm_extra_body is not valid JSON, ignoring: {e}")
        headers = {
            "Authorization": f"Bearer {cfg['api_key']}",
            "Content-Type": "application/json",
        }
        url = f"{cfg['base_url']}/chat/completions"
        timeout = aiohttp.ClientTimeout(total=cfg["timeout"], connect=10)

        try:
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
                    data = await resp.json(content_type=None)
        except aiohttp.ClientError as e:
            raise LLMError(f"Network error: {e}") from e

        try:
            msg = data["choices"][0]["message"]
            # Normalise: ensure content is a string (may be None when tool_calls are present)
            if msg.get("content") is None:
                msg["content"] = ""
            return msg
        except (KeyError, IndexError, AttributeError, TypeError) as e:
            logger.error(f"Unexpected LLM response shape: {data}")
            raise LLMError("Unexpected response from LLM") from e


def _inject_current_time(messages: list[dict]) -> list[dict]:
    """Append the current UTC time to the system message (or insert one)."""
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    stamp = f"Current time: {now}"
    if messages and messages[0].get("role") == "system":
        head = dict(messages[0])
        existing = (head.get("content") or "").rstrip()
        head["content"] = f"{existing}\n\n{stamp}" if existing else stamp
        messages[0] = head
    else:
        messages.insert(0, {"role": "system", "content": stamp})
    return messages
