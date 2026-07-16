"""
Tests for the per-context transcript logger.

Covers: append writes a line, ContextVar isolation across tasks, rotation
when crossing MAX_BYTES, clear removes file + backups, build_record shape.
"""

import asyncio
import json
import tempfile
from pathlib import Path

import pytest

from src.llm import transcript as tx
from src.llm.output_safety import sanitize_assistant_message


@pytest.fixture
def tmp_logger():
    with tempfile.TemporaryDirectory() as d:
        yield tx.TranscriptLogger(base_dir=Path(d))


@pytest.mark.asyncio
async def test_append_writes_jsonl_line(tmp_logger):
    await tmp_logger.append(42, {"hello": "world", "n": 1})
    await tmp_logger.append(42, {"hello": "world", "n": 2})
    path = tmp_logger.path_for(42)
    assert path.exists()
    lines = path.read_text().splitlines()
    assert len(lines) == 2
    assert json.loads(lines[0]) == {"hello": "world", "n": 1}
    assert json.loads(lines[1]) == {"hello": "world", "n": 2}


@pytest.mark.asyncio
async def test_separate_contexts_dont_mix(tmp_logger):
    await tmp_logger.append(1, {"who": "ctx1"})
    await tmp_logger.append(2, {"who": "ctx2"})
    assert "ctx1" in tmp_logger.path_for(1).read_text()
    assert "ctx2" not in tmp_logger.path_for(1).read_text()
    assert "ctx2" in tmp_logger.path_for(2).read_text()


@pytest.mark.asyncio
async def test_rotation_when_size_exceeded(monkeypatch, tmp_logger):
    # Force a low rotation threshold so the test stays small.
    monkeypatch.setattr(tx, "MAX_BYTES", 200)
    big_payload = {"x": "z" * 80}
    for _ in range(5):
        await tmp_logger.append(99, big_payload)
    # The current file exists and at least one backup was created.
    current = tmp_logger.path_for(99)
    backup1 = tmp_logger.backup_path(99, 1)
    assert current.exists()
    assert backup1.exists(), "rotation should have produced .1 backup"


@pytest.mark.asyncio
async def test_clear_removes_everything(monkeypatch, tmp_logger):
    monkeypatch.setattr(tx, "MAX_BYTES", 200)
    big = {"x": "y" * 80}
    for _ in range(6):
        await tmp_logger.append(7, big)
    assert tmp_logger.path_for(7).exists()
    assert tmp_logger.backup_path(7, 1).exists()
    removed = await tmp_logger.clear(7)
    assert removed >= 2
    assert not tmp_logger.path_for(7).exists()
    assert not tmp_logger.backup_path(7, 1).exists()


def test_active_context_var_isolation():
    """Each set/reset roundtrip should restore the previous value."""
    assert tx.get_active() is None
    token = tx.set_active(tx.ActiveContext(
        context_id=1, context_label="t", transcript_logging_enabled=True,
        bot_id=None,
    ))
    assert tx.get_active() is not None
    assert tx.get_active().context_id == 1
    tx.reset_active(token)
    assert tx.get_active() is None


@pytest.mark.asyncio
async def test_active_context_does_not_leak_across_tasks():
    """asyncio.Task copies contextvars at create time — but mutating the
    var inside a task must not leak into siblings."""
    results = {}

    async def worker(name, ctx_id):
        tx.set_active(tx.ActiveContext(
            context_id=ctx_id, context_label=name,
            transcript_logging_enabled=True, bot_id=None,
        ))
        await asyncio.sleep(0)  # yield to scheduler
        active = tx.get_active()
        results[name] = active.context_id if active else None

    await asyncio.gather(worker("a", 1), worker("b", 2), worker("c", 3))
    assert results == {"a": 1, "b": 2, "c": 3}


def test_build_record_shape():
    rec = tx.build_record(
        purpose="ask",
        model="gpt-test",
        messages=[{"role": "user", "content": "hi"}],
        tools=[{"type": "function", "function": {"name": "x"}}],
        assistant_msg={
            "role": "assistant", "content": "yo",
            "tool_calls": [{"id": "1", "function": {"name": "x", "arguments": "{}"}}],
        },
        params={"temperature": 0.5, "max_tokens": 200},
        latency_ms=123.4,
        bot_id=5,
        context_id=10,
        context_label="test ctx",
        sender_tail="1234",
        group_id="grp-abc",
        usage={"prompt_tokens": 12, "completion_tokens": 3},
    )
    assert rec["purpose"] == "ask"
    assert rec["model"] == "gpt-test"
    assert rec["messages"] == [{"role": "user", "content": "hi"}]
    assert rec["response"]["content"] == "yo"
    assert rec["response"]["tool_calls"][0]["function"]["name"] == "x"
    assert rec["params"] == {"temperature": 0.5, "max_tokens": 200}
    assert rec["usage"]["prompt_tokens"] == 12
    assert rec["_meta"]["bot_id"] == 5
    assert rec["_meta"]["context_id"] == 10
    assert rec["_meta"]["latency_ms"] == 123.4


def test_build_record_freezes_inputs_and_redacts_inline_images():
    messages = [{
        "role": "user",
        "content": [
            {"type": "text", "text": "inspect this"},
            {
                "type": "image_url",
                "image_url": {"url": "data:image/png;base64,QUJDREVGRw=="},
            },
        ],
    }]
    tools = [{"type": "function", "function": {"name": "price"}}]
    response = {"role": "assistant", "content": "done"}
    rec = tx.build_record(
        purpose="ask", model="gpt-test", messages=messages, tools=tools,
        assistant_msg=response, params={}, latency_ms=1, bot_id=1,
        context_id=2, context_label=None, sender_tail=None, group_id=None,
    )

    messages.append({"role": "assistant", "content": "later"})
    tools[0]["function"]["name"] = "mutated"
    response["content"] = "mutated"

    assert len(rec["messages"]) == 1
    assert rec["tools"][0]["function"]["name"] == "price"
    assert rec["response"]["content"] == "done"
    image_url = rec["messages"][0]["content"][1]["image_url"]["url"]
    assert image_url.startswith("data:image/png;base64,[inline binary omitted:")
    assert "QUJDREVGRw" not in image_url


def test_logged_purposes_excludes_reactor():
    """Reactor/summary/healthcheck calls must NOT be logged."""
    assert "ask" in tx.LOGGED_PURPOSES
    assert "augment" in tx.LOGGED_PURPOSES
    assert "reactor" not in tx.LOGGED_PURPOSES
    assert "summary" not in tx.LOGGED_PURPOSES
    assert "healthcheck" not in tx.LOGGED_PURPOSES


def test_transcript_assistant_cleanup_keeps_tool_calls():
    original = {
        "role": "assistant",
        "content": "[to David] clean answer",
        "tool_calls": [{"id": "1", "function": {"name": "x"}}],
    }
    cleaned = sanitize_assistant_message(original)
    assert cleaned["content"] == "clean answer"
    assert cleaned["tool_calls"] == original["tool_calls"]
    assert original["content"].startswith("[to David]")


@pytest.mark.asyncio
async def test_stats_reports_current_and_backups(monkeypatch, tmp_logger):
    monkeypatch.setattr(tx, "MAX_BYTES", 200)
    for _ in range(4):
        await tmp_logger.append(33, {"x": "y" * 80})
    stats = tmp_logger.stats(33)
    assert stats["exists"]
    assert stats["line_count"] >= 1
    assert stats["size_bytes"] > 0
    # At least one backup exists post-rotation.
    assert len(stats["backups"]) >= 1


def test_admin_clear_button_does_not_nest_a_form():
    """The clear action must override the settings form's endpoint.

    Browsers do not support nested forms. The old markup silently submitted
    the surrounding context-settings form, leaving transcript files intact.
    """
    template_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "admin"
        / "templates"
        / "context_edit.html"
    )
    template = template_path.read_text(encoding="utf-8")
    settings_start = template.index('<form method="post" autocomplete="off">')
    settings_end = template.index("</form>", settings_start)
    settings_form = template[settings_start:settings_end]

    assert settings_form.count("<form") == 1
    assert "admin.context_transcript_clear" in settings_form
    assert 'formmethod="post"' in settings_form
    assert "formnovalidate" in settings_form
