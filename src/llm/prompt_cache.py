"""Prompt-cache contracts and privacy-safe payload fingerprints.

The provider cache is prefix based, so two parts of a request deserve special
treatment: the system message and the tool schemas.  This module lets prompt
builders declare named blocks as stable or volatile, then fingerprints the
actual outbound payload without retaining prompt text.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )


def _digest(value: Any) -> str:
    return hashlib.sha256(_canonical(value).encode("utf-8")).hexdigest()


def _content_chars(value: Any) -> int:
    if isinstance(value, str):
        return len(value)
    return len(_canonical(value))


@dataclass(frozen=True)
class PromptBlock:
    """One named prompt fragment and its cache-locality contract."""

    name: str
    content: Any
    stability: str

    def summary(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hash": _digest(self.content),
            "chars": _content_chars(self.content),
        }


@dataclass
class PromptCachePlan:
    """Named stable/volatile fragments used to assemble one request.

    The plan is diagnostic metadata only; it is never included in the API
    payload.  ``snapshot`` hashes the rendered messages as a second line of
    defense, so an undeclared mutation of the system message is still visible.
    """

    stable_blocks: list[PromptBlock] = field(default_factory=list)
    volatile_blocks: list[PromptBlock] = field(default_factory=list)

    @classmethod
    def from_blocks(
        cls,
        *,
        stable: Iterable[tuple[str, Any]] = (),
        volatile: Iterable[tuple[str, Any]] = (),
    ) -> "PromptCachePlan":
        return cls(
            stable_blocks=[
                PromptBlock(name, content, "stable")
                for name, content in stable
                if content not in (None, "", [], {})
            ],
            volatile_blocks=[
                PromptBlock(name, content, "volatile")
                for name, content in volatile
                if content not in (None, "", [], {})
            ],
        )

    def with_stable(self, name: str, content: Any) -> "PromptCachePlan":
        if content not in (None, "", [], {}):
            self.stable_blocks.append(PromptBlock(name, content, "stable"))
        return self

    def with_volatile(self, name: str, content: Any) -> "PromptCachePlan":
        if content not in (None, "", [], {}):
            self.volatile_blocks.append(PromptBlock(name, content, "volatile"))
        return self

    def snapshot(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        *,
        purpose: str,
        context_id: Optional[int] = None,
        bot_id: Optional[int] = None,
    ) -> dict[str, Any]:
        systems = [m for m in messages if m.get("role") == "system"]
        tool_payload = tools or []
        tail = messages[-1] if messages else {}
        history = messages[1:-1] if len(messages) > 2 else []
        stable = [block.summary() for block in self.stable_blocks]
        volatile = [block.summary() for block in self.volatile_blocks]
        system_chars = sum(_content_chars(m.get("content")) for m in systems)
        tool_chars = len(_canonical(tool_payload)) if tool_payload else 0
        history_chars = sum(_content_chars(m.get("content")) for m in history)
        return {
            "version": 1,
            "purpose": purpose,
            "context_id": context_id,
            "bot_id": bot_id,
            "system_hash": _digest(systems),
            "tools_hash": _digest(tool_payload),
            "tail_hash": _digest(tail),
            "stable_blocks_hash": _digest(stable),
            "volatile_blocks_hash": _digest(volatile),
            "system_chars": system_chars,
            "tool_schema_chars": tool_chars,
            "history_chars": history_chars,
            "cacheable_prefix_chars": system_chars + tool_chars + history_chars,
            "message_count": len(messages),
            "tool_count": len(tool_payload),
            "stable_blocks": stable,
            "volatile_blocks": volatile,
        }


def automatic_cache_plan(messages: list[dict]) -> PromptCachePlan:
    """Fallback contract for callers that do not own a structured builder."""

    stable = []
    volatile = []
    for index, message in enumerate(messages):
        role = str(message.get("role") or "unknown")
        content = message.get("content")
        if role == "system":
            stable.append((f"system_message_{index}", content))
        elif index == len(messages) - 1:
            volatile.append(("rendered_user_tail", content))
    return PromptCachePlan.from_blocks(stable=stable, volatile=volatile)
