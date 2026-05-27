"""Tests for the assistant-message reasoning_content mirror.

The mirror exists because OpenRouter exposes thinking-mode reasoning as
`reasoning` / `reasoning_details` on response, but DeepSeek's API rejects
round-tripped tool-call chains when the same assistant turn comes back
without `reasoning_content` ("reasoning_content in the thinking mode
must be passed back to the API"). The helper runs once per outbound
chat and back-fills the field from whichever shape we got.
"""

from src.llm.client import _ensure_reasoning_content


def test_passthrough_when_no_reasoning():
    messages = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "hello"},
    ]
    out = _ensure_reasoning_content(messages)
    assert out == messages
    # Same object identity for untouched messages so the in-memory
    # tool-loop state isn't needlessly rebuilt.
    assert out[2] is messages[2]


def test_mirrors_reasoning_string_into_reasoning_content():
    msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning": "step-by-step thought",
    }
    out = _ensure_reasoning_content([msg])
    assert out[0]["reasoning_content"] == "step-by-step thought"
    # Original untouched (shallow-copy on write).
    assert "reasoning_content" not in msg


def test_mirrors_reasoning_details_array():
    msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [
            {"type": "reasoning.text", "text": "first chunk"},
            {"type": "reasoning.text", "text": "second chunk"},
        ],
    }
    out = _ensure_reasoning_content([msg])
    assert out[0]["reasoning_content"] == "first chunk\nsecond chunk"


def test_existing_reasoning_content_is_preserved():
    """When DeepSeek native returns `reasoning_content` directly, don't
    clobber it with whatever's in `reasoning`."""
    msg = {
        "role": "assistant",
        "content": "",
        "reasoning_content": "original native field",
        "reasoning": "normalized copy",
    }
    out = _ensure_reasoning_content([msg])
    assert out[0]["reasoning_content"] == "original native field"
    assert out[0] is msg  # untouched


def test_non_assistant_messages_untouched():
    msg = {"role": "tool", "tool_call_id": "x", "content": "result", "reasoning": "ignored"}
    out = _ensure_reasoning_content([msg])
    assert out[0] is msg
    assert "reasoning_content" not in out[0]


def test_empty_reasoning_skipped():
    msg = {"role": "assistant", "content": "answer", "reasoning": ""}
    out = _ensure_reasoning_content([msg])
    assert out[0] is msg
    assert "reasoning_content" not in out[0]


def test_reasoning_details_with_content_field_fallback():
    """Some providers stash the text under `content` instead of `text`
    inside each detail object — accept either."""
    msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [
            {"type": "reasoning", "content": "via content key"},
        ],
    }
    out = _ensure_reasoning_content([msg])
    assert out[0]["reasoning_content"] == "via content key"


def test_reasoning_details_malformed_items_skipped():
    msg = {
        "role": "assistant",
        "content": "answer",
        "reasoning_details": [
            "not a dict",
            {"type": "reasoning.text", "text": "kept"},
            {"type": "reasoning.encrypted", "data": "no text key"},
        ],
    }
    out = _ensure_reasoning_content([msg])
    assert out[0]["reasoning_content"] == "kept"
