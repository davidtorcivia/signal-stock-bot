"""
Regression tests for DeepThinkClient._run_tool_loop's content extraction.

The final answer is always in `content`. Reasoning models put their
chain-of-thought in a sibling `reasoning`/`reasoning_content` field — that is
the thinking, never the deliverable. A turn with empty content but populated
reasoning means the model stopped before producing a real answer; surfacing the
trace would publish raw chain-of-thought (the WSB digest bug: a thought trace
posted instead of an article). The loop must return a `(deep_think ...` stub
instead, which downstream callers already treat as a failure.
"""

import pytest

from src.llm.deep_think import DeepThinkClient


_CFG = {"max_tool_rounds": 3}


def _client():
    # _run_tool_loop never touches self.store; a bare sentinel is enough.
    return DeepThinkClient(settings_store=object())


@pytest.mark.asyncio
async def test_reasoning_only_returns_stub_not_trace():
    client = _client()

    async def fake_chat_call(messages, tools, cfg, sender_tail, group_id, **kwargs):
        return {
            "role": "assistant",
            "content": "",
            "reasoning_content": "Let me think... the article should say X and Y",
            "_dt_tokens_in": 10,
            "_dt_tokens_out": 20,
        }

    client._chat_call = fake_chat_call
    text, history, _, _ = await client._run_tool_loop(
        messages=[], tools=None, caller_ctx=None, attachments=[],
        cfg=_CFG, sender_tail="????", group_id=None,
    )
    assert text.startswith("(deep_think ")
    assert "think" not in text or "stopped while still reasoning" in text
    assert "article should say" not in text  # the trace never leaks
    assert history == []


@pytest.mark.asyncio
async def test_real_content_is_returned_verbatim():
    client = _client()

    async def fake_chat_call(messages, tools, cfg, sender_tail, group_id, **kwargs):
        return {
            "role": "assistant",
            "content": "Here is the finished article body.",
            "reasoning": "(some thinking that should be ignored)",
            "_dt_tokens_in": 5,
            "_dt_tokens_out": 9,
        }

    client._chat_call = fake_chat_call
    text, _, _, _ = await client._run_tool_loop(
        messages=[], tools=None, caller_ctx=None, attachments=[],
        cfg=_CFG, sender_tail="????", group_id=None,
    )
    assert text == "Here is the finished article body."
