"""Per-invocation tool-call idempotency helpers."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Optional


def tool_call_fingerprint(name: str, arguments: Any) -> str:
    encoded = json.dumps(
        {"name": name, "arguments": arguments},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=repr,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


@dataclass
class ToolCallLedger:
    """Memoize tool results for the lifetime of one LLM tool loop."""

    by_call_id: dict[str, tuple[str, str]] = field(default_factory=dict)
    by_fingerprint: dict[str, str] = field(default_factory=dict)

    def lookup(
        self,
        *,
        call_id: str,
        name: str,
        arguments: Any,
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
        self,
        *,
        call_id: str,
        name: str,
        arguments: Any,
        content: str,
    ) -> None:
        fingerprint = tool_call_fingerprint(name, arguments)
        self.by_fingerprint[fingerprint] = content
        if call_id:
            self.by_call_id[call_id] = (fingerprint, content)
