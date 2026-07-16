"""Shared, bounded orchestration for writer and research tool loops."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Awaitable, Callable, Optional

import aiosqlite


logger = logging.getLogger(__name__)

TOOL_RESULT_VERSION = 1
DEFAULT_MAX_TOOL_RESULT_CHARS = 6000
DEFAULT_MAX_STAGNANT_ROUNDS = 2
DEFAULT_TOOL_LOOP_SECONDS = 180.0
DEFAULT_TOOL_ARTIFACT_DIR = Path("data/tool_artifacts")
TOOL_ARTIFACT_RETENTION_SECONDS = 7 * 86400.0
_pruned_artifact_dirs: set[str] = set()


def _canonical(value: Any) -> str:
    return json.dumps(
        value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=repr,
    )


def parse_tool_arguments(call: dict) -> Any:
    raw = (call.get("function") or {}).get("arguments") or "{}"
    if not isinstance(raw, str):
        return raw
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return {}


def tool_call_name(call: dict) -> str:
    return str((call.get("function") or {}).get("name") or "")


def tool_call_fingerprint(name: str, arguments: Any) -> str:
    encoded = _canonical({"name": name, "arguments": arguments})
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ToolCallLedger:
    """Memoize tool results for the lifetime of one LLM tool loop."""

    by_call_id: dict[str, tuple[str, str]] = field(default_factory=dict)
    by_fingerprint: dict[str, str] = field(default_factory=dict)

    def lookup(
        self, *, call_id: str, name: str, arguments: Any,
    ) -> tuple[Optional[str], Optional[str]]:
        fingerprint = tool_call_fingerprint(name, arguments)
        if call_id and call_id in self.by_call_id:
            old_fingerprint, content = self.by_call_id[call_id]
            if old_fingerprint != fingerprint:
                return (
                    "duplicate_call_id_conflict",
                    "ERROR: repeated tool-call ID used with different arguments; call suppressed",
                )
            return "duplicate_call_id", content
        if fingerprint in self.by_fingerprint:
            return "duplicate_arguments", self.by_fingerprint[fingerprint]
        return None, None

    def record(
        self, *, call_id: str, name: str, arguments: Any, content: str,
    ) -> None:
        fingerprint = tool_call_fingerprint(name, arguments)
        self.by_fingerprint[fingerprint] = content
        if call_id:
            self.by_call_id[call_id] = (fingerprint, content)


# Parallelism is opt-in.  Unknown tools, generic MCP invocation, and anything
# that can change chat state remain serialized.
_READ_ONLY_TOOLS = frozenset({
    "mcp__discover",
    "recall",
    "portfolio_status",
    "portfolio_options_chain",
    "portfolio_option_quote",
    "portfolio_journal_read",
    "bot__price",
    "bot__quote",
    "bot__info",
    "bot__market",
    "bot__crypto",
    "bot__option",
    "bot__chain",
    "bot__future",
    "bot__forex",
    "bot__economy",
    "bot__chart",
    "bot__news",
    "bot__earnings",
    "bot__dividend",
    "bot__rating",
    "bot__insider",
    "bot__short",
    "bot__corr",
    "bot__ta",
    "bot__tldr",
    "bot__rsi",
    "bot__sma",
    "bot__macd",
    "bot__support",
    "bot__pnl",
    "bot__trades",
    "bot__predictions",
    "bot__leaderboard",
    "bot__kalshi",
    "bot__numerology",
})

_DURABLE_MUTATIONS = frozenset({
    "remember",
    "forget",
    "predict_self",
    "predict_for",
    "predict_update",
    "portfolio_buy",
    "portfolio_sell",
    "portfolio_place_order",
    "portfolio_place_option_order",
    "portfolio_cancel_order",
    "portfolio_buy_option",
    "portfolio_sell_option",
    "portfolio_journal_append",
    "bot__watch",
    "bot__alert",
    "bot__tip",
    "bot__predict",
    "bot__resolve",
    "bot__admin",
    "bot__cache",
})


def is_read_only_tool(name: str) -> bool:
    return name in _READ_ONLY_TOOLS


def is_durable_mutation(name: str, arguments: Any = None) -> bool:
    if name in _DURABLE_MUTATIONS or name == "mcp__invoke":
        return True
    if name == "bot__portfolio" and isinstance(arguments, dict):
        values = arguments.get("args") or []
        action = str(values[0]).lower() if values else ""
        return action in {
            "buy", "sell", "order", "place", "cancel",
            "buy-option", "sell-option", "buy_option", "sell_option",
        }
    return False


def _parse_structured_data(content: Any) -> Any:
    if not isinstance(content, str):
        return content
    stripped = content.strip()
    if stripped[:1] in {"{", "["}:
        try:
            return json.loads(stripped)
        except (TypeError, ValueError):
            pass
    return content


def _persist_tool_artifact(data: Any, artifact_dir: Optional[str | Path]) -> Optional[str]:
    raw = _canonical(data)
    artifact_id = hashlib.sha256(raw.encode("utf-8")).hexdigest()
    try:
        base_dir = Path(artifact_dir) if artifact_dir is not None else DEFAULT_TOOL_ARTIFACT_DIR
        base_dir.mkdir(parents=True, exist_ok=True)
        resolved = str(base_dir.resolve())
        if resolved not in _pruned_artifact_dirs:
            cutoff = time.time() - TOOL_ARTIFACT_RETENTION_SECONDS
            for old_path in base_dir.glob("*.json"):
                try:
                    if old_path.stat().st_mtime < cutoff:
                        old_path.unlink()
                except OSError:
                    continue
            _pruned_artifact_dirs.add(resolved)
        artifact_path = base_dir / f"{artifact_id}.json"
        if not artifact_path.exists():
            artifact_path.write_text(raw, encoding="utf-8")
        return artifact_id
    except OSError as exc:
        logger.warning("Could not persist oversized tool artifact: %s", exc)
        return None


def ensure_tool_result_envelope(
    name: str,
    content: Any,
    *,
    ok: Optional[bool] = None,
    reused: Optional[str] = None,
    max_chars: int = DEFAULT_MAX_TOOL_RESULT_CHARS,
    artifact_dir: Optional[str | Path] = None,
) -> str:
    """Return a compact, bounded tool result understood by every loop."""
    if isinstance(content, str):
        try:
            existing = json.loads(content)
        except (TypeError, ValueError):
            existing = None
        if isinstance(existing, dict) and existing.get("_tool_result") == 1:
            envelope = dict(existing)
            if reused:
                envelope["reused"] = reused
            encoded = _canonical(envelope)
            if len(encoded) <= max_chars:
                return encoded

    max_chars = max(256, int(max_chars))
    text = str(content if content is not None else "")
    inferred_ok = not text.lstrip().upper().startswith(("ERROR", "◇ ERROR"))
    data = _parse_structured_data(content)
    truncated = False
    artifact_id = None
    if len(_canonical(data)) > max_chars:
        artifact_id = _persist_tool_artifact(data, artifact_dir)
        data = text[: max(0, max_chars - 700)].rstrip()
        truncated = len(data) < len(text)
    envelope = {
        "_tool_result": TOOL_RESULT_VERSION,
        "ok": inferred_ok if ok is None else bool(ok),
        "tool": name,
        "source": name,
        "observed_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "freshness": "live" if is_read_only_tool(name) else "not_applicable",
        "data": data,
        "truncated": truncated,
        "warning": "full result stored out of band" if truncated else None,
        "artifact_id": artifact_id,
    }
    if reused:
        envelope["reused"] = reused
    encoded = _canonical(envelope)
    while len(encoded) > max_chars and envelope["data"]:
        # The metadata itself is small; shrink only data and always re-encode
        # complete JSON rather than slicing the serialized object.
        over = len(encoded) - max_chars
        data_text = str(envelope["data"])
        envelope["data"] = data_text[:max(0, len(data_text) - over - 8)] + "…"
        envelope["truncated"] = True
        encoded = _canonical(envelope)
    if len(encoded) > max_chars:
        # Extremely small caller-provided caps can be smaller than the normal
        # metadata. Retain the contract fields in a minimal valid object.
        encoded = _canonical({
            "_tool_result": TOOL_RESULT_VERSION,
            "ok": envelope["ok"],
            "tool": str(name)[:40],
            "data": "…",
            "truncated": True,
            "artifact_id": artifact_id,
        })
    return encoded


def tool_result_ok(content: str) -> bool:
    try:
        parsed = json.loads(content)
        if isinstance(parsed, dict) and parsed.get("_tool_result") == 1:
            return bool(parsed.get("ok"))
    except (TypeError, ValueError):
        pass
    return not str(content).lstrip().upper().startswith("ERROR")


def tool_information_hash(content: str) -> str:
    """Hash evidence while ignoring timestamps and replay annotations."""
    try:
        parsed = json.loads(content)
    except (TypeError, ValueError):
        parsed = content
    if isinstance(parsed, dict) and parsed.get("_tool_result") == 1:
        parsed = {
            "ok": parsed.get("ok"),
            "tool": parsed.get("tool"),
            "data": parsed.get("data"),
            "truncated": parsed.get("truncated"),
        }
    return hashlib.sha256(_canonical(parsed).encode("utf-8")).hexdigest()


@dataclass
class ToolExecution:
    message: dict
    attachments: list[Any] = field(default_factory=list)
    ok: bool = True


@dataclass
class ToolLoopOutcome:
    content: str
    tool_history: list[str]
    tokens_in: int = 0
    tokens_out: int = 0
    rounds: int = 0
    calls: int = 0
    parallel_batches: int = 0
    stop_reason: str = "complete"


class DurableToolLedger:
    """SQLite-backed claims preventing replayed mutating operations."""

    def __init__(self, db_path: str | Path = "data/watchlist.db", lease_seconds: float = 300):
        self.db_path = Path(db_path)
        self.lease_seconds = max(1.0, float(lease_seconds))
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """CREATE TABLE IF NOT EXISTS llm_tool_operations (
                           idempotency_key TEXT PRIMARY KEY,
                           source_key TEXT NOT NULL,
                           tool_name TEXT NOT NULL,
                           arguments_hash TEXT NOT NULL,
                           status TEXT NOT NULL,
                           result TEXT,
                           created_at REAL NOT NULL,
                           updated_at REAL NOT NULL
                       )"""
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_tool_ops_source "
                    "ON llm_tool_operations(source_key, updated_at)"
                )
                await db.execute(
                    "DELETE FROM llm_tool_operations WHERE updated_at < ?",
                    (time.time() - 30 * 86400.0,),
                )
                await db.commit()
            self._initialized = True

    @staticmethod
    def operation_key(source_key: str, name: str, arguments: Any) -> str:
        raw = _canonical({"source": source_key, "name": name, "arguments": arguments})
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    async def claim(
        self, source_key: str, name: str, arguments: Any,
    ) -> tuple[str, str, Optional[str]]:
        """Return (state, operation_key, prior_result)."""
        await self._ensure_initialized()
        key = self.operation_key(source_key, name, arguments)
        args_hash = tool_call_fingerprint(name, arguments)
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            cursor = await db.execute(
                """INSERT OR IGNORE INTO llm_tool_operations
                   (idempotency_key, source_key, tool_name, arguments_hash,
                    status, result, created_at, updated_at)
                   VALUES (?, ?, ?, ?, 'running', NULL, ?, ?)""",
                (key, source_key, name, args_hash, now, now),
            )
            await db.commit()
            if cursor.rowcount == 1:
                return "claimed", key, None
            cursor = await db.execute(
                "SELECT status, result, updated_at FROM llm_tool_operations "
                "WHERE idempotency_key = ?",
                (key,),
            )
            row = await cursor.fetchone()
            if row and row[0] == "completed":
                return "completed", key, row[1]
            if row and now - float(row[2] or 0) < self.lease_seconds:
                return "in_progress", key, None
            await db.execute(
                "UPDATE llm_tool_operations SET status='running', result=NULL, "
                "updated_at=? WHERE idempotency_key=?",
                (now, key),
            )
            await db.commit()
            return "claimed", key, None

    async def complete(self, operation_key: str, result: str) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "UPDATE llm_tool_operations SET status='completed', result=?, "
                "updated_at=? WHERE idempotency_key=?",
                (result, time.time(), operation_key),
            )
            await db.commit()

    async def release(self, operation_key: str) -> None:
        await self._ensure_initialized()
        async with aiosqlite.connect(self.db_path) as db:
            await db.execute(
                "DELETE FROM llm_tool_operations WHERE idempotency_key=? "
                "AND status='running'",
                (operation_key,),
            )
            await db.commit()


ChatCallback = Callable[[list[dict], Optional[list[dict]]], Awaitable[dict]]
ExecuteCallback = Callable[[dict, ToolCallLedger], Awaitable[ToolExecution]]


class ToolLoopRuntime:
    """One state machine shared by writer and deep-research clients."""

    def __init__(
        self,
        *,
        max_rounds: int,
        max_calls: Optional[int] = None,
        max_stagnant_rounds: int = DEFAULT_MAX_STAGNANT_ROUNDS,
        wall_timeout_seconds: float = DEFAULT_TOOL_LOOP_SECONDS,
    ):
        self.max_rounds = max(1, int(max_rounds))
        self.max_calls = max(1, int(max_calls or min(60, self.max_rounds * 4)))
        self.max_stagnant_rounds = max(1, int(max_stagnant_rounds))
        self.wall_timeout_seconds = max(1.0, float(wall_timeout_seconds))

    async def _execute_one(
        self,
        call: dict,
        *,
        execute: ExecuteCallback,
        ledger: ToolCallLedger,
        durable_ledger: Optional[DurableToolLedger],
        source_key: Optional[str],
    ) -> ToolExecution:
        name = tool_call_name(call)
        call_id = str(call.get("id") or "")
        args = parse_tool_arguments(call)
        operation_key: Optional[str] = None
        if durable_ledger and source_key and is_durable_mutation(name, args):
            try:
                state, operation_key, prior = await durable_ledger.claim(
                    source_key, name, args,
                )
            except Exception as exc:
                logger.exception("Durable tool claim failed for %s", name)
                content = ensure_tool_result_envelope(
                    name,
                    f"ERROR: idempotency ledger unavailable; mutation suppressed ({type(exc).__name__})",
                    ok=False,
                )
                return ToolExecution({
                    "role": "tool", "tool_call_id": call_id,
                    "name": name, "content": content,
                }, ok=False)
            if state == "completed":
                content = ensure_tool_result_envelope(
                    name, prior or "(prior operation completed)",
                    ok=True, reused="durable_replay",
                )
                return ToolExecution({
                    "role": "tool", "tool_call_id": call_id,
                    "name": name, "content": content,
                })
            if state == "in_progress":
                content = ensure_tool_result_envelope(
                    name, "ERROR: identical operation is already in progress",
                    ok=False, reused="in_progress",
                )
                return ToolExecution({
                    "role": "tool", "tool_call_id": call_id,
                    "name": name, "content": content,
                }, ok=False)

        try:
            result = await execute(call, ledger)
        except Exception as exc:
            logger.exception("Unhandled tool callback failure for %s", name)
            if operation_key and durable_ledger:
                try:
                    await durable_ledger.release(operation_key)
                except Exception:
                    logger.exception("Failed to release tool claim for %s", name)
            content = ensure_tool_result_envelope(
                name, f"ERROR: tool runtime failure ({type(exc).__name__})", ok=False,
            )
            return ToolExecution({
                "role": "tool", "tool_call_id": call_id,
                "name": name, "content": content,
            }, ok=False)
        result.message = dict(result.message)
        result.message["content"] = ensure_tool_result_envelope(
            name, result.message.get("content", ""), ok=result.ok,
        )
        result.ok = tool_result_ok(result.message["content"])
        if operation_key and durable_ledger:
            if result.ok:
                try:
                    await durable_ledger.complete(
                        operation_key, result.message["content"],
                    )
                except Exception:
                    # The side effect already happened. Keep the running claim
                    # fail-closed for its lease rather than suggesting a retry.
                    logger.exception("Failed to complete tool claim for %s", name)
            else:
                try:
                    await durable_ledger.release(operation_key)
                except Exception:
                    logger.exception("Failed to release tool claim for %s", name)
        return result

    async def _execute_batch(
        self,
        calls: list[dict],
        *,
        execute: ExecuteCallback,
        ledger: ToolCallLedger,
        durable_ledger: Optional[DurableToolLedger],
        source_key: Optional[str],
    ) -> tuple[list[ToolExecution], bool]:
        fingerprints = [
            tool_call_fingerprint(tool_call_name(call), parse_tool_arguments(call))
            for call in calls
        ]
        parallel = (
            len(calls) > 1
            and len(set(fingerprints)) == len(fingerprints)
            and all(is_read_only_tool(tool_call_name(call)) for call in calls)
        )
        if parallel:
            results = await asyncio.gather(*(
                self._execute_one(
                    call, execute=execute, ledger=ledger,
                    durable_ledger=durable_ledger, source_key=source_key,
                )
                for call in calls
            ))
            return list(results), True
        results = []
        for call in calls:
            results.append(await self._execute_one(
                call, execute=execute, ledger=ledger,
                durable_ledger=durable_ledger, source_key=source_key,
            ))
        return results, False

    async def _finalize(
        self,
        *,
        messages: list[dict],
        chat: ChatCallback,
        reason: str,
        fallback: str,
    ) -> tuple[str, int, int]:
        messages.append({
            "role": "user",
            "content": (
                "<harness_budget>\n"
                f"Tool execution stopped because {reason}. Do not call more tools. "
                "Using only the completed tool results above, give the most useful "
                "honest answer possible. Clearly state anything that remains unknown "
                "or incomplete; do not infer missing live facts.\n"
                "</harness_budget>"
            ),
        })
        try:
            final = await chat(messages, None)
        except Exception:
            logger.exception("Budgeted no-tools synthesis failed")
            return fallback, 0, 0
        tokens_in = int(final.pop("_dt_tokens_in", 0) or 0)
        tokens_out = int(final.pop("_dt_tokens_out", 0) or 0)
        content = str(final.get("content") or "").strip()
        if not content or final.get("tool_calls"):
            return fallback, tokens_in, tokens_out
        return content, tokens_in, tokens_out

    async def run(
        self,
        *,
        messages: list[dict],
        tools: Optional[list[dict]],
        chat: ChatCallback,
        execute: ExecuteCallback,
        attachments: list,
        source_key: Optional[str] = None,
        durable_ledger: Optional[DurableToolLedger] = None,
        empty_response: str = "",
        incomplete_fallback: str = "Task stopped before the tool work completed.",
    ) -> ToolLoopOutcome:
        ledger = ToolCallLedger()
        tool_history: list[str] = []
        seen_evidence: set[str] = set()
        stagnant_rounds = 0
        total_in = 0
        total_out = 0
        call_count = 0
        parallel_batches = 0
        started = time.monotonic()

        for round_idx in range(self.max_rounds):
            if time.monotonic() - started >= self.wall_timeout_seconds:
                reason = f"the {self.wall_timeout_seconds:.0f}s wall-clock budget expired"
                content, t_in, t_out = await self._finalize(
                    messages=messages, chat=chat, reason=reason,
                    fallback=incomplete_fallback,
                )
                return ToolLoopOutcome(
                    content, tool_history, total_in + t_in, total_out + t_out,
                    round_idx, call_count, parallel_batches, "wall_timeout",
                )

            assistant = await chat(messages, tools)
            total_in += int(assistant.pop("_dt_tokens_in", 0) or 0)
            total_out += int(assistant.pop("_dt_tokens_out", 0) or 0)
            messages.append(assistant)
            calls = list(assistant.get("tool_calls") or [])
            if not calls:
                content = str(assistant.get("content") or "").strip()
                return ToolLoopOutcome(
                    content or empty_response, tool_history, total_in, total_out,
                    round_idx + 1, call_count, parallel_batches, "complete",
                )

            remaining = self.max_calls - call_count
            budget_exhausted = remaining <= 0 or len(calls) > remaining
            executable = calls[:max(0, remaining)]
            skipped = calls[len(executable):]
            tool_history.extend(tool_call_name(call) or "?" for call in calls)
            call_count += len(executable)
            results: list[ToolExecution] = []
            if executable:
                results, parallel = await self._execute_batch(
                    executable, execute=execute, ledger=ledger,
                    durable_ledger=durable_ledger, source_key=source_key,
                )
                parallel_batches += int(parallel)
            for call in skipped:
                name = tool_call_name(call)
                results.append(ToolExecution({
                    "role": "tool",
                    "tool_call_id": str(call.get("id") or ""),
                    "name": name,
                    "content": ensure_tool_result_envelope(
                        name, "ERROR: per-request tool-call budget exhausted", ok=False,
                    ),
                }, ok=False))

            round_hashes: set[str] = set()
            for result in results:
                messages.append(result.message)
                attachments.extend(result.attachments)
                round_hashes.add(tool_information_hash(result.message.get("content", "")))
            new_evidence = round_hashes - seen_evidence
            seen_evidence.update(round_hashes)
            stagnant_rounds = 0 if new_evidence else stagnant_rounds + 1

            if stagnant_rounds and results:
                last = results[-1].message
                last["content"] = ensure_tool_result_envelope(
                    tool_call_name(calls[-1]),
                    last.get("content", ""),
                    ok=tool_result_ok(last.get("content", "")),
                    reused="no_new_evidence; synthesize_now",
                )

            if budget_exhausted or stagnant_rounds >= self.max_stagnant_rounds:
                reason = (
                    "the tool-call budget was exhausted"
                    if budget_exhausted else
                    f"{stagnant_rounds} consecutive rounds produced no new evidence"
                )
                content, t_in, t_out = await self._finalize(
                    messages=messages, chat=chat, reason=reason,
                    fallback=incomplete_fallback,
                )
                return ToolLoopOutcome(
                    content, tool_history, total_in + t_in, total_out + t_out,
                    round_idx + 1, call_count, parallel_batches,
                    "call_budget" if budget_exhausted else "stagnation",
                )

        reason = f"the {self.max_rounds}-round tool budget was exhausted"
        content, t_in, t_out = await self._finalize(
            messages=messages, chat=chat, reason=reason,
            fallback=incomplete_fallback,
        )
        return ToolLoopOutcome(
            content, tool_history, total_in + t_in, total_out + t_out,
            self.max_rounds, call_count, parallel_batches, "round_budget",
        )
