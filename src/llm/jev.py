"""Bounded choices through OpenRouter's Decisions API."""

import asyncio
import json
import logging
import math
import os
import time
from urllib.parse import urlsplit

import aiohttp

from ..cache import get_metrics

logger = logging.getLogger(__name__)

JEV_URL = "https://openrouter.ai/api/alpha/decisions"
JEV_DEFAULTS = {
    "jev_enabled": True,
    "jev_model": "~typesafe/jev-latest",
    "jev_api_key": "",
    "jev_timeout_seconds": 5,
    "jev_min_confidence": 0.8,
}


class JevClient:
    def __init__(self, settings_store):
        self.store = settings_store
        self._session = None
        self._semaphore = asyncio.Semaphore(4)

    def _api_key(self):
        key = self.store.get_stripped("jev_api_key") or os.getenv("OPENROUTER_API_KEY", "").strip()
        if key:
            return key
        for prefix in ("llm", "deep_think"):
            url = self.store.get_stripped(f"{prefix}_base_url")
            if urlsplit(url).hostname == "openrouter.ai":
                key = self.store.get_stripped(f"{prefix}_api_key")
                if key:
                    return key
        return ""

    def enabled(self):
        return self.store.get_bool("jev_enabled", True) and bool(self._api_key())

    async def close(self):
        if self._session is not None:
            await self._session.close()
            self._session = None

    async def choose(self, *, state, questions, purpose):
        """Return confident, validated choices; missing answers use caller fallback."""
        if not questions or not self.enabled():
            return {}
        model = self.store.get_stripped("jev_model") or JEV_DEFAULTS["jev_model"]
        timeout = max(1, min(30, self.store.get_int("jev_timeout_seconds", 5)))
        threshold = self.store.get_float("jev_min_confidence", 0.8)
        if not math.isfinite(threshold) or not 0 <= threshold <= 1:
            threshold = 0.8
        started = time.monotonic()
        metrics = get_metrics()
        purpose = f"jev_{purpose}"
        provider = metrics.get_provider_metrics(f"jev:{model}")
        if not provider.is_healthy():
            metrics.record_llm_circuit_rejection(purpose=purpose, model=model)
            return {}
        try:
            payload = {"model": model, "state": state, "questions": questions}
            # Oversized inputs use the existing path instead of silently losing context.
            if len(json.dumps(payload, ensure_ascii=False).encode()) > 96_000:
                logger.info("JEV %s: oversized input; using fallback", purpose)
                return {}
            # The timeout includes waiting for capacity; bursts must not stall fallbacks.
            async with asyncio.timeout(timeout), self._semaphore:
                if self._session is None or self._session.closed:
                    self._session = aiohttp.ClientSession()
                async with self._session.post(
                    JEV_URL,
                    headers={"Authorization": f"Bearer {self._api_key()}"},
                    json=payload,
                ) as response:
                    response.raise_for_status()
                    data = await response.json()
            if not isinstance(data, dict):
                raise ValueError("Invalid decisions response")
            answers = data["answers"]
            if not isinstance(answers, dict):
                raise ValueError("Invalid decisions response")
            choices = {}
            for key, question in questions.items():
                answer = answers.get(key)
                if not isinstance(answer, dict) or answer.get("type") != "choice":
                    continue
                choice = answer.get("choice")
                confidence = answer.get("confidence")
                if (
                    isinstance(choice, str) and choice in question["criteria"]
                    and type(confidence) in (float, int)
                    and threshold <= confidence <= 1
                ):
                    choices[key] = choice
            usage = data.get("usage") or {}
            if not isinstance(usage, dict):
                usage = {}
            latency_ms = (time.monotonic() - started) * 1000
            metrics.record_llm_success(
                purpose=purpose, model=model,
                latency_ms=latency_ms,
                tokens_in=usage.get("input_tokens", 0),
                tokens_out=usage.get("output_tokens", 0),
            )
            provider.record_success(latency_ms)
            if provider.circuit_open:
                provider.close_circuit()
            logger.info("JEV %s: %d/%d confident choices", purpose, len(choices), len(questions))
            return choices
        except (aiohttp.ClientError, TimeoutError, ValueError, TypeError, KeyError) as exc:
            # Response bodies can contain chat content; log only the error class/status.
            error = f"{type(exc).__name__} status={getattr(exc, 'status', None)}"
            logger.warning("JEV %s failed: %s; using fallback", purpose, error)
            metrics.record_llm_error(purpose=purpose, model=model, error_msg=error)
            provider.record_error(error)
            if provider.consecutive_errors >= 3:
                provider.open_circuit(60)
            return {}
