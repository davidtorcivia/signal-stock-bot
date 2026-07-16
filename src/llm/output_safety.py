"""Shared cleanup for visible LLM output and training transcripts."""

from __future__ import annotations

import re


_META_LEAK_PATTERNS = (
    # Current turn-pointer form: `[turn h12; to David, ...]` or a group row.
    re.compile(r"^\s*\[turn\s+[hg]\d+;[^\]\n]{0,100}\]\s*", re.IGNORECASE),
    # Legacy assistant-turn addressee label: `[to David, 2m ago]`.
    re.compile(r"^\s*\[to [^\]\n]{1,80}\]\s*", re.IGNORECASE),
    # Legacy speaker labels with absolute or relative times.
    re.compile(
        r"^\s*\["
        r"[^\]\n,]{1,40},\s*"
        r"(?:just now|a moment ago|"
        r"\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}\s*UTC|"
        r"\d+\s*(?:[smhdw]|min(?:ute)?|sec(?:ond)?|hour|day|week)s?"
        r"(?:\s*ago)?)"
        r"\s*\]\s*",
        re.IGNORECASE,
    ),
    re.compile(r"^\s*Spontaneous[- ]reply[ -]?path?:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Spontaneous reply:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Reflex note:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Identity note:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Attribution rules?[^\n]*\n?", re.IGNORECASE),
)


_TOOL_CALL_LEAK_RE = re.compile(
    r"</?\|?(?:function[=\s>]|parameter=|tool[_▁\s]?call|invoke[\s=>])[\s\S]*$",
    re.IGNORECASE,
)


def strip_meta_leak(text: str) -> str:
    """Remove replay/system scaffolding copied at the start of a reply."""
    if not text:
        return text
    for _ in range(4):
        new = text
        for pattern in _META_LEAK_PATTERNS:
            new = pattern.sub("", new, count=1)
        if new == text:
            break
        text = new
    return text


def strip_tool_call_leak(text: str) -> str:
    """Cut malformed pseudo-tool markup off the visible end of a reply."""
    if not text or "<" not in text:
        return text
    return _TOOL_CALL_LEAK_RE.sub("", text).rstrip()


def sanitize_assistant_message(message: dict) -> dict:
    """Copy an assistant message with its visible text made training-safe.

    Tool-call structure is preserved exactly. Only string ``content`` is
    cleaned, matching the final Signal-delivery cleanup in AskCommand.
    """
    cleaned = dict(message)
    content = cleaned.get("content")
    if isinstance(content, str):
        cleaned["content"] = strip_tool_call_leak(strip_meta_leak(content))
    return cleaned
