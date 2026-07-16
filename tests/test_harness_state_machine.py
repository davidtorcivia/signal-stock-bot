"""Focused tests for the typed, persistent LLM harness state machine."""

import asyncio
import json

import pytest

from src.llm.history import ConversationHistory
from src.llm.summarizer import (
    Summarizer,
    parse_structured_summary,
    render_summary_for_prompt,
    serialize_structured_summary,
)
from src.llm.tool_runtime import (
    DurableToolLedger,
    ToolExecution,
    ToolLoopRuntime,
    ensure_tool_result_envelope,
)


def _call(call_id: str, name: str, args: dict) -> dict:
    return {
        "id": call_id,
        "function": {"name": name, "arguments": json.dumps(args)},
    }


def test_tool_result_envelope_is_structured_and_strictly_bounded(tmp_path):
    encoded = ensure_tool_result_envelope(
        "bot__price", "x" * 10_000, max_chars=500,
        artifact_dir=tmp_path / "artifacts",
    )
    parsed = json.loads(encoded)
    assert len(encoded) <= 500
    assert parsed["_tool_result"] == 1
    assert parsed["tool"] == "bot__price"
    assert parsed["ok"] is True
    assert parsed["truncated"] is True
    assert parsed["artifact_id"]
    artifact = tmp_path / "artifacts" / f"{parsed['artifact_id']}.json"
    assert artifact.exists()
    assert len(artifact.read_text(encoding="utf-8")) > 10_000


@pytest.mark.asyncio
async def test_runtime_parallelizes_independent_read_only_calls_in_order():
    chat_count = 0
    both_started = asyncio.Event()
    active = 0

    async def chat(_messages, _tools):
        nonlocal chat_count
        chat_count += 1
        if chat_count == 1:
            return {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    _call("a", "bot__price", {"args": ["AAPL"]}),
                    _call("b", "bot__quote", {"args": ["MSFT"]}),
                ],
            }
        return {"role": "assistant", "content": "finished"}

    async def execute(call, _ledger):
        nonlocal active
        active += 1
        if active == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=1)
        active -= 1
        name = call["function"]["name"]
        return ToolExecution({
            "role": "tool", "tool_call_id": call["id"],
            "name": name, "content": f"result:{call['id']}",
        })

    messages = [{"role": "user", "content": "compare"}]
    outcome = await ToolLoopRuntime(max_rounds=3).run(
        messages=messages, tools=[{}], chat=chat, execute=execute,
        attachments=[],
    )
    assert outcome.content == "finished"
    assert outcome.parallel_batches == 1
    tool_ids = [m["tool_call_id"] for m in messages if m["role"] == "tool"]
    assert tool_ids == ["a", "b"]


@pytest.mark.asyncio
async def test_durable_mutation_reuses_result_across_separate_loops(tmp_path):
    durable = DurableToolLedger(tmp_path / "ops.db")
    executions = 0

    async def run_once():
        chat_count = 0

        async def chat(_messages, _tools):
            nonlocal chat_count
            chat_count += 1
            if chat_count == 1:
                return {
                    "role": "assistant", "content": "",
                    "tool_calls": [_call("remember-1", "remember", {"fact": "blue"})],
                }
            return {"role": "assistant", "content": "done"}

        async def execute(call, _ledger):
            nonlocal executions
            executions += 1
            return ToolExecution({
                "role": "tool", "tool_call_id": call["id"],
                "name": "remember", "content": "memory stored",
            })

        return await ToolLoopRuntime(max_rounds=3).run(
            messages=[{"role": "user", "content": "remember"}],
            tools=[{}], chat=chat, execute=execute, attachments=[],
            source_key="group:g:bot:1:message:99",
            durable_ledger=durable,
        )

    assert (await run_once()).content == "done"
    assert (await run_once()).content == "done"
    assert executions == 1


@pytest.mark.asyncio
async def test_runtime_stops_when_rounds_add_no_new_evidence():
    normal_round = 0

    async def chat(_messages, tools):
        nonlocal normal_round
        if tools is None:
            return {"role": "assistant", "content": "partial, with caveat"}
        normal_round += 1
        return {
            "role": "assistant", "content": "",
            "tool_calls": [_call(str(normal_round), "bot__price", {"args": ["AAPL"]})],
        }

    async def execute(call, _ledger):
        return ToolExecution({
            "role": "tool", "tool_call_id": call["id"],
            "name": "bot__price", "content": "AAPL 200.00",
        })

    outcome = await ToolLoopRuntime(max_rounds=10).run(
        messages=[{"role": "user", "content": "price"}],
        tools=[{}], chat=chat, execute=execute, attachments=[],
    )
    assert outcome.stop_reason == "stagnation"
    assert outcome.rounds == 3
    assert outcome.content == "partial, with caveat"


@pytest.mark.asyncio
async def test_turn_graph_and_retrieval_reservoir_survive_small_recent_window(tmp_path):
    history = ConversationHistory(
        db_path=str(tmp_path / "history.db"), turns_per_user=1,
    )
    first_id = await history.append("group:x", "user", "the cobalt project")
    second_id = await history.append(
        "group:x", "assistant", "cobalt ships Friday",
        parent_turn_ref=f"h{first_id}",
    )
    for index in range(4):
        user_id = await history.append("group:x", "user", f"unrelated chatter {index}")
        await history.append(
            "group:x", "assistant", f"ack {index}",
            parent_turn_ref=f"h{user_id}",
        )

    recent = await history.load(
        "group:x", include_internal=True, include_turn_ids=True,
    )
    assert all(turn["_turn_id"] not in {f"h{first_id}", f"h{second_id}"} for turn in recent)
    retrieved = await history.retrieve_relevant(
        "group:x", "when does cobalt ship?",
        exclude_turn_ids={turn["_turn_id"] for turn in recent},
        limit=3,
    )
    assert {turn["turn_id"] for turn in retrieved} >= {f"h{first_id}", f"h{second_id}"}
    assistant = next(turn for turn in retrieved if turn["turn_id"] == f"h{second_id}")
    assert assistant["parent_turn_ref"] == f"h{first_id}"


def test_structured_summary_is_valid_bounded_and_cited():
    summary = {
        "facts": [{
            "text": "David prefers cobalt.",
            "source_turn_ids": ["h12"],
            "last_confirmed": "2026-07-16 12:00 UTC",
        }],
        "decisions": [],
        "open_questions": [{
            "text": "When will it ship?",
            "source_turn_ids": ["h13"],
            "last_confirmed": None,
        }],
        "topics": [{
            "text": "Cobalt " + ("x" * 800),
            "source_turn_ids": ["h12"],
            "last_confirmed": None,
        }],
    }
    encoded = serialize_structured_summary(summary, 500)
    parsed = parse_structured_summary(encoded, allow_legacy=False)
    assert parsed is not None
    assert len(encoded) <= 500
    rendered = render_summary_for_prompt(encoded)
    assert "David prefers cobalt. [sources: h12]" in rendered
    assert "When will it ship? [sources: h13]" in rendered


@pytest.mark.asyncio
async def test_summarizer_persists_only_valid_cited_json(tmp_path):
    history = ConversationHistory(
        db_path=str(tmp_path / "summary.db"), turns_per_user=20,
    )
    for index in range(8):
        user_id = await history.append("group:s", "user", f"fact {index}")
        await history.append(
            "group:s", "assistant", f"answer {index}",
            parent_turn_ref=f"h{user_id}",
        )

    class Store:
        def get(self, key, default=None):
            values = {
                "summary_enabled": True,
                "summary_keep_recent": 2,
                "summary_min_new_turns": 2,
                "summary_max_chars": 1000,
            }
            return values.get(key, default)

    class LLM:
        async def chat_messages(self, messages, **_kwargs):
            assert "[turn h" in messages[-1]["content"]
            return {
                "role": "assistant",
                "content": json.dumps({
                    "facts": [{
                        "text": "The group discussed fact zero.",
                        "source_turn_ids": ["h1"],
                        "last_confirmed": "2026-07-16 12:00 UTC",
                    }],
                    "decisions": [], "open_questions": [], "topics": [],
                }),
            }

    summarizer = Summarizer(LLM(), history, Store())
    assert await summarizer.maybe_summarize("group:s") is True
    stored = await history.get_summary("group:s")
    assert stored is not None
    parsed = parse_structured_summary(stored["summary"], allow_legacy=False)
    assert parsed["facts"][0]["source_turn_ids"] == ["h1"]
