"""Typed prompt assembly with an enforceable cache-locality boundary."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from .prompt_cache import PromptCachePlan


def _present(value: Any) -> bool:
    return value not in (None, "", [], {})


@dataclass(frozen=True)
class StablePromptBlock:
    """Configuration/persona content permitted in the system prefix."""

    name: str
    content: Any


@dataclass(frozen=True)
class VolatilePromptBlock:
    """Request, recalled, or model-derived content confined to the tail."""

    name: str
    content: Any


@dataclass(frozen=True)
class CompiledPrompt:
    messages: list[dict]
    tools: Optional[list[dict]]
    cache_plan: PromptCachePlan

    def cache_prefix_bytes(self) -> bytes:
        """Canonical bytes for cache-canary tests and runtime assertions."""
        system = [m for m in self.messages if m.get("role") == "system"]
        payload = {"system": system, "tools": self.tools or []}
        return json.dumps(
            payload, ensure_ascii=False, sort_keys=True,
            separators=(",", ":"), default=repr,
        ).encode("utf-8")

    def assert_same_cache_prefix(self, other: "CompiledPrompt") -> None:
        if self.cache_prefix_bytes() != other.cache_prefix_bytes():
            raise AssertionError(
                "volatile prompt mutation changed the system/tool cache prefix"
            )


@dataclass
class PromptCompiler:
    """The only assembly path from named blocks to provider messages.

    Callers cannot accidentally pass a volatile block to ``add_stable`` (or
    vice versa): the methods accept concrete wrapper types and reject the
    wrong class at runtime as defense in depth for untyped call sites.
    """

    base_system: StablePromptBlock
    stable_blocks: list[StablePromptBlock] = field(default_factory=list)
    volatile_blocks: list[VolatilePromptBlock] = field(default_factory=list)

    @classmethod
    def with_base_system(cls, content: str) -> "PromptCompiler":
        return cls(StablePromptBlock("base_system", content))

    def add_stable(self, block: StablePromptBlock) -> "PromptCompiler":
        if not isinstance(block, StablePromptBlock):
            raise TypeError("system prefix accepts StablePromptBlock only")
        if _present(block.content):
            self.stable_blocks.append(block)
        return self

    def add_volatile(self, block: VolatilePromptBlock) -> "PromptCompiler":
        if not isinstance(block, VolatilePromptBlock):
            raise TypeError("prompt tail accepts VolatilePromptBlock only")
        if _present(block.content):
            self.volatile_blocks.append(block)
        return self

    def extend_stable(
        self, blocks: Iterable[StablePromptBlock],
    ) -> "PromptCompiler":
        for block in blocks:
            self.add_stable(block)
        return self

    def extend_volatile(
        self, blocks: Iterable[VolatilePromptBlock],
    ) -> "PromptCompiler":
        for block in blocks:
            self.add_volatile(block)
        return self

    def compile(
        self,
        *,
        history: Iterable[dict] = (),
        user_content: Any,
        tools: Optional[list[dict]] = None,
    ) -> CompiledPrompt:
        if not _present(self.base_system.content):
            raise ValueError("base system prompt cannot be empty")
        system_parts = [str(self.base_system.content)]
        system_parts.extend(
            str(block.content) for block in self.stable_blocks
            if _present(block.content)
        )
        system_prompt = "\n\n".join(system_parts)
        messages = [{"role": "system", "content": system_prompt}]
        messages.extend(dict(turn) for turn in history)
        messages.append({"role": "user", "content": user_content})
        cache_plan = PromptCachePlan.from_blocks(
            stable=[
                (self.base_system.name, self.base_system.content),
                *((block.name, block.content) for block in self.stable_blocks),
            ],
            volatile=[
                (block.name, block.content) for block in self.volatile_blocks
            ],
        )
        return CompiledPrompt(messages, tools, cache_plan)
