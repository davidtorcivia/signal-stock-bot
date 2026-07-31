"""
Per-context conversation-transcript logger.

Writes one OpenAI-shape JSON record per `LLMClient.chat_messages` round to
`data/transcripts/ctx_<id>.jsonl`. Each line contains the full messages
array sent to the model, the assistant response (including tool calls),
the tool schemas offered, params (model/temperature/max_tokens/provider),
latency, bot_id, context_id, purpose, and a timestamp.

Files rotate at 100 MB, keep 3 backups (matches bot.log's pattern). Append
is serialized per-file by an asyncio.Lock so concurrent LLM calls in the
same context can't tear lines.

Plumbing: the dispatcher sets `set_active(...)` at message entry; the
LLMClient reads it via `get_active()` and calls `log_round(...)` when the
context has `transcript_logging_enabled=True`. Calls outside an active
context (oracle workers, healthcheck) see `get_active() == None` and
skip silently.
"""

from __future__ import annotations

import asyncio
import copy
import contextvars
import hashlib
import json
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


def _redacted_binary(value: str) -> str:
    """Return a stable, non-secret marker for an inline binary payload."""
    digest = hashlib.sha256(value.encode("utf-8", errors="replace")).hexdigest()[:16]
    return f"[inline binary omitted: sha256={digest}, chars={len(value)}]"


def _redacted_identifier(value: Optional[str], label: str) -> Optional[str]:
    """Keep identifiers correlatable inside an export without exposing them."""
    if not value:
        return None
    digest = hashlib.sha256(
        str(value).encode("utf-8", errors="replace")
    ).hexdigest()[:16]
    return f"[{label} sha256={digest}]"


def snapshot_payload(value: Any, *, key: Optional[str] = None) -> Any:
    """Freeze an API payload and omit inline attachment bytes.

    Transcript writes run in a background task while the tool loop continues
    mutating its message list.  Capturing by value here keeps each record tied
    to the exact provider request that produced its response.
    """
    if isinstance(value, dict):
        return {
            str(k): snapshot_payload(v, key=str(k))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [snapshot_payload(item) for item in value]
    if isinstance(value, tuple):
        return [snapshot_payload(item) for item in value]
    if isinstance(value, str):
        if key in {"data_b64", "base64", "bytes_b64"}:
            return _redacted_binary(value)
        # `input_audio` parts carry raw base64 under the generic key
        # "data" — no `data:` prefix to recognize it by. Length-gated
        # because "data" is common enough that a short legitimate value
        # (a tool argument, an id) must pass through untouched.
        if key == "data" and len(value) > 1024:
            return _redacted_binary(value)
        if value.startswith("data:") and ";base64," in value[:128]:
            header = value.split(",", 1)[0]
            return f"{header},{_redacted_binary(value)}"
        return value
    return copy.deepcopy(value)

# Rotate when the active file crosses 100 MB. Keep 3 backups so the admin
# can grab a fresh batch and still recover the prior one. Aligned with
# logs/bot.log's rotation pattern.
MAX_BYTES = 100 * 1024 * 1024
BACKUP_COUNT = 3

DEFAULT_TRANSCRIPT_DIR = Path("data/transcripts")

# Only these purposes are logged. The user's choice was "writer + tool
# results"; both translate to writer calls (each tool round is a fresh
# chat_messages call whose `messages` array contains the prior tool
# results). Reactor/summary/healthcheck are excluded.
LOGGED_PURPOSES = frozenset({"ask", "augment"})


@dataclass
class ActiveContext:
    """What the current async task is processing.

    Populated by the dispatcher at message entry; consumed by LLMClient.
    `policy_*` fields are snapshotted at set-time so a later policy edit
    doesn't retroactively change whether an in-flight call gets logged.
    """
    context_id: int
    context_label: str
    transcript_logging_enabled: bool
    bot_id: Optional[int]
    sender_tail: Optional[str] = None
    group_id: Optional[str] = None


_active: contextvars.ContextVar[Optional[ActiveContext]] = contextvars.ContextVar(
    "transcript_active_context", default=None
)


def set_active(ctx: Optional[ActiveContext]) -> contextvars.Token:
    """Set the active context for the current async task.

    Returns a Token the caller should pass to `reset_active` in a finally
    block. Each asyncio.Task gets its own contextvar copy, so concurrent
    inbound messages don't bleed across each other.
    """
    return _active.set(ctx)


def reset_active(token: contextvars.Token) -> None:
    _active.reset(token)


def get_active() -> Optional[ActiveContext]:
    return _active.get()


class TranscriptLogger:
    """Per-process singleton that owns the on-disk transcript files.

    One asyncio.Lock per context_id serializes appends so concurrent
    writer rounds in the same chat can't interleave bytes.
    """

    def __init__(self, base_dir: Path = DEFAULT_TRANSCRIPT_DIR):
        self.base_dir = Path(base_dir)
        self._locks: dict[int, asyncio.Lock] = {}
        self._locks_guard = asyncio.Lock()

    def path_for(self, context_id: int) -> Path:
        return self.base_dir / f"ctx_{context_id}.jsonl"

    def backup_path(self, context_id: int, index: int) -> Path:
        return self.base_dir / f"ctx_{context_id}.jsonl.{index}"

    async def _lock_for(self, context_id: int) -> asyncio.Lock:
        async with self._locks_guard:
            lock = self._locks.get(context_id)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[context_id] = lock
            return lock

    def _rotate(self, path: Path, context_id: int) -> None:
        """Roll path -> path.1, path.1 -> path.2, ..., dropping the oldest.

        Mirrors RotatingFileHandler's behavior. Called inside the per-
        context lock so no other writer is touching these files.
        """
        try:
            for i in range(BACKUP_COUNT - 1, 0, -1):
                src = self.backup_path(context_id, i)
                dst = self.backup_path(context_id, i + 1)
                if src.exists():
                    if dst.exists():
                        dst.unlink()
                    src.rename(dst)
            dst = self.backup_path(context_id, 1)
            if dst.exists():
                dst.unlink()
            if path.exists():
                path.rename(dst)
        except OSError as e:
            logger.warning(f"Transcript rotation failed for ctx {context_id}: {e}")

    async def append(self, context_id: int, record: dict) -> None:
        """Append one JSON record. Errors are logged and swallowed."""
        try:
            line = json.dumps(record, ensure_ascii=False, default=_json_default)
        except (TypeError, ValueError) as e:
            logger.warning(f"Transcript serialize failed for ctx {context_id}: {e}")
            return

        lock = await self._lock_for(context_id)
        async with lock:
            try:
                self.base_dir.mkdir(parents=True, exist_ok=True)
                path = self.path_for(context_id)
                # Rotate BEFORE writing if we'd cross the threshold —
                # otherwise a single oversized record could exceed the
                # cap. stat() may raise FileNotFoundError on first call.
                try:
                    size = path.stat().st_size
                except FileNotFoundError:
                    size = 0
                if size and size + len(line) + 1 > MAX_BYTES:
                    self._rotate(path, context_id)
                with path.open("a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError as e:
                logger.warning(f"Transcript append failed for ctx {context_id}: {e}")

    def stats(self, context_id: int) -> dict:
        """Cheap filesystem summary for the admin UI.

        Returns size + line count for the current file, plus a list of
        existing backup paths. Line count is approximate when the file
        is large (counts newlines on disk; one read pass).
        """
        path = self.path_for(context_id)
        info: dict[str, Any] = {
            "exists": path.exists(),
            "size_bytes": 0,
            "line_count": 0,
            "backups": [],
        }
        if path.exists():
            try:
                info["size_bytes"] = path.stat().st_size
                with path.open("rb") as f:
                    info["line_count"] = sum(1 for _ in f)
            except OSError as e:
                logger.debug(f"Transcript stat failed: {e}")
        for i in range(1, BACKUP_COUNT + 1):
            bp = self.backup_path(context_id, i)
            if bp.exists():
                try:
                    info["backups"].append({
                        "index": i,
                        "size_bytes": bp.stat().st_size,
                    })
                except OSError:
                    pass
        return info

    async def clear(self, context_id: int) -> int:
        """Delete the active file AND all backups. Returns number of files removed."""
        lock = await self._lock_for(context_id)
        removed = 0
        async with lock:
            for p in [self.path_for(context_id)] + [
                self.backup_path(context_id, i) for i in range(1, BACKUP_COUNT + 1)
            ]:
                try:
                    if p.exists():
                        p.unlink()
                        removed += 1
                except OSError as e:
                    logger.warning(f"Transcript clear failed for {p}: {e}")
        return removed


def _json_default(obj: Any) -> Any:
    """Fallback for non-JSON-native objects (e.g. dataclasses, bytes)."""
    if isinstance(obj, bytes):
        try:
            return obj.decode("utf-8", errors="replace")
        except Exception:
            return repr(obj)
    if hasattr(obj, "__dict__"):
        return obj.__dict__
    return repr(obj)


# Process-wide singleton. Constructed lazily so tests can patch the dir.
_GLOBAL: Optional[TranscriptLogger] = None


def get_logger() -> TranscriptLogger:
    global _GLOBAL
    if _GLOBAL is None:
        _GLOBAL = TranscriptLogger(_resolve_base_dir())
    return _GLOBAL


def _resolve_base_dir() -> Path:
    """Honor TRANSCRIPT_DIR env var so tests/dev can redirect output."""
    env = os.environ.get("TRANSCRIPT_DIR")
    if env:
        return Path(env)
    return DEFAULT_TRANSCRIPT_DIR


def build_record(
    *,
    purpose: str,
    model: str,
    messages: list[dict],
    tools: Optional[list[dict]],
    assistant_msg: dict,
    params: dict,
    latency_ms: float,
    bot_id: Optional[int],
    context_id: int,
    context_label: Optional[str],
    sender_tail: Optional[str],
    group_id: Optional[str],
    usage: Optional[dict] = None,
    cache_manifest: Optional[dict] = None,
    cache_event: Optional[dict] = None,
) -> dict:
    """Compose the JSONL line for one writer round.

    Shape matches OpenAI's chat-completions payload + response so the file
    can feed directly into SFT pipelines that expect that schema. Extra
    fields (`_meta`) are namespaced under an underscore so naive readers
    can drop them.
    """
    # Snapshot synchronously, before append() is scheduled on the event loop.
    frozen_messages = snapshot_payload(messages)
    frozen_tools = snapshot_payload(tools or [])
    frozen_assistant = snapshot_payload(assistant_msg)
    frozen_params = snapshot_payload(params)
    frozen_usage = snapshot_payload(usage or {})
    frozen_manifest = snapshot_payload(cache_manifest or {})
    frozen_event = snapshot_payload(cache_event or {})
    return {
        "ts": time.time(),
        "purpose": purpose,
        "model": model,
        "messages": frozen_messages,
        "tools": frozen_tools,
        "response": {
            "role": frozen_assistant.get("role", "assistant"),
            "content": frozen_assistant.get("content") or "",
            "tool_calls": frozen_assistant.get("tool_calls") or [],
        },
        "params": frozen_params,
        "usage": frozen_usage,
        "_meta": {
            "bot_id": bot_id,
            "context_id": context_id,
            "context_label": context_label,
            "sender_tail": sender_tail,
            # A raw Signal group id is stable identifying metadata and is not
            # needed to replay or train on the captured provider payload.
            "group_id": _redacted_identifier(group_id, "group"),
            "latency_ms": latency_ms,
            # Fingerprints and sizes only — no duplicate prompt plaintext.
            "cache_manifest": frozen_manifest,
            "cache_event": frozen_event,
        },
    }
