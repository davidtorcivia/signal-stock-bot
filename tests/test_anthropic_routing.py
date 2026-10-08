"""Claude routing: only named group contexts spend Anthropic credits, and
any Claude failure falls back to the OpenRouter path."""

import pytest

from src.llm import anthropic_backend
from src.llm.client import LLMClient
from src.llm.transcript import ActiveContext, reset_active, set_active
from tests.test_llm_persona import FakeStore

KEYED = {(1, "writer", "anthropic_api_key"): "sk-test"}


def _ctx(*, named=True, group_id="g1"):
    return ActiveContext(
        context_id=3, context_label="x", transcript_logging_enabled=False,
        bot_id=1, group_id=group_id, named=named,
    )


async def _route(monkeypatch, ctx, *, bot_settings=KEYED, purpose="ask",
                 overrides=None, fail=False):
    seen = {}

    async def fake_chat(**kw):
        if fail:
            raise RuntimeError("Your credit balance is too low")
        seen.update(kw)
        return ({"role": "assistant", "content": "hi"},
                {"prompt_tokens": 10, "completion_tokens": 2,
                 "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 10,
                 "stop_reason": "end_turn"})

    monkeypatch.setattr(anthropic_backend, "chat", fake_chat)
    llm = LLMClient(FakeStore(bot_settings=bot_settings), bot_id=1)
    token = set_active(ctx) if ctx else None
    try:
        out = await llm._try_anthropic(
            purpose=purpose, overrides=overrides or {},
            messages=[{"role": "user", "content": "q"}], tools=None,
            payload={"max_tokens": 1000}, cfg={"timeout": 30, "model": "glm"},
            cache_manifest={},
        )
    finally:
        if token:
            reset_active(token)
    return out, seen.get("model")


@pytest.mark.parametrize("ctx", [
    None,                                   # no context resolved
    _ctx(group_id=None),                    # DM
    _ctx(named=False),                      # default:group catch-all
])
async def test_unnamed_or_dm_never_uses_claude(monkeypatch, ctx):
    assert (await _route(monkeypatch, ctx))[0] is None


async def test_needs_per_bot_key_and_unpinned_model(monkeypatch):
    assert (await _route(monkeypatch, _ctx(), bot_settings={}))[0] is None
    assert (await _route(monkeypatch, _ctx(), overrides={"model": "jev"}))[0] is None


async def test_ask_gets_opus_everything_else_haiku(monkeypatch):
    out, model = await _route(monkeypatch, _ctx())
    assert out["content"] == "hi" and model == anthropic_backend.WORKHORSE_MODEL
    _, model = await _route(monkeypatch, _ctx(), purpose="summary")
    assert model == anthropic_backend.CHEAP_MODEL


async def test_claude_failure_falls_back(monkeypatch):
    assert (await _route(monkeypatch, _ctx(), fail=True))[0] is None


def test_tool_loop_translation():
    raw = [{"type": "thinking", "thinking": "", "signature": "s"},
           {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]
    system, msgs = anthropic_backend.to_anthropic([
        {"role": "system", "content": "SYS"},
        {"role": "user", "content": [
            {"type": "text", "text": "look"},
            {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}},
        ]},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "t1"}],
         anthropic_backend.RAW_KEY: raw},
        {"role": "tool", "tool_call_id": "t1", "content": "r1"},
        {"role": "tool", "tool_call_id": "t2", "content": "r2"},
    ])
    assert system == "SYS"
    assert msgs[0]["content"][1]["source"] == {
        "type": "base64", "media_type": "image/png", "data": "AAA"}
    assert msgs[1]["content"] is raw  # thinking echoed back unchanged
    assert [b["tool_use_id"] for b in msgs[2]["content"]] == ["t1", "t2"]
    with pytest.raises(ValueError):
        anthropic_backend.to_anthropic([{"role": "user", "content": [
            {"type": "input_audio", "input_audio": {}}]}])


async def test_alerts_cooldown_mute_and_recovery(monkeypatch):
    import asyncio
    sent = []

    async def sender(text):
        sent.append(text)

    monkeypatch.setattr(anthropic_backend, "alert_sender", sender)
    monkeypatch.setattr(anthropic_backend, "_down", False)
    monkeypatch.setattr(anthropic_backend, "_last_alert", 0.0)
    store = FakeStore()
    err = RuntimeError("credit balance is too low")

    anthropic_backend.report_health(store, err)
    anthropic_backend.report_health(store, err)       # inside cooldown
    anthropic_backend.report_health(store, None)      # recovered
    anthropic_backend.report_health(store, None)      # already healthy
    await asyncio.sleep(0)
    assert [t[0] for t in sent] == ["⚠", "✅"]

    sent.clear()
    store._global[anthropic_backend.MUTE_KEY] = 9e12  # muted
    monkeypatch.setattr(anthropic_backend, "_last_alert", 0.0)
    anthropic_backend.report_health(store, err)
    await asyncio.sleep(0)
    assert sent == []


def test_admin_claude_mute_command(tmp_path):
    from src.commands.admin_commands import AdminCommand
    from src.settings_store import SettingsStore

    store = SettingsStore(str(tmp_path / "s.db"))
    cmd = AdminCommand(["+1"], settings_store=store)
    assert cmd._claude_alerts(["mute", "6h"]).success
    assert store.get(anthropic_backend.MUTE_KEY) > 0
    assert "muted" in cmd._claude_alerts([]).text
    assert not cmd._claude_alerts(["mute", "soon"]).success
    cmd._claude_alerts(["unmute"])
    assert store.get(anthropic_backend.MUTE_KEY) == 0


async def test_deep_think_uses_opus_high_only_for_deep_think_role(monkeypatch):
    from src.llm.deep_think import DeepThinkClient
    seen = {}

    async def fake_chat(**kw):
        seen.update(kw)
        return ({"role": "assistant", "content": "deep"},
                {"prompt_tokens": 5, "completion_tokens": 1,
                 "prompt_cache_hit_tokens": 0, "prompt_cache_miss_tokens": 5,
                 "stop_reason": "end_turn"})

    monkeypatch.setattr(anthropic_backend, "chat", fake_chat)
    cfg = {"model": "deepseek", "max_tokens": 30000, "timeout": 60}
    token = set_active(_ctx())
    try:
        dt = DeepThinkClient(FakeStore(bot_settings=KEYED), bot_id=1)
        msg = await dt._try_claude([{"role": "user", "content": "q"}], None, cfg)
        assert msg["content"] == "deep" and msg["_dt_tokens_in"] == 5
        assert seen["model"] == anthropic_backend.WORKHORSE_MODEL
        assert seen["effort"] == "high"
        tool_bot = DeepThinkClient(FakeStore(bot_settings=KEYED), bot_id=1, role="tool_bot")
        assert await tool_bot._try_claude([], None, cfg) is None
    finally:
        reset_active(token)


def test_spend_estimate(tmp_path):
    import sqlite3
    db = tmp_path / "m.db"
    with sqlite3.connect(db) as c:
        c.execute("""CREATE TABLE metric_events (occurred_at REAL, kind TEXT,
            model TEXT, tokens_in INTEGER, tokens_out INTEGER,
            cache_hit_tokens INTEGER, cache_miss_tokens INTEGER)""")
        c.executemany("INSERT INTO metric_events VALUES (?,?,?,?,?,?,?)", [
            (100, "llm_success", "claude-opus-5-5", 2_000_000, 100_000, 1_000_000, 1_000_000),
            (100, "dt_error", "claude-opus-5-5", None, None, None, None),
            (100, "llm_success", "z-ai/glm-5.3-flash", 9, 9, 0, 9),   # not Claude
            (1, "llm_success", "claude-opus-5-5", 9, 9, 9, 9),          # too old
        ])
    out = anthropic_backend.spend(db, since=50)
    (opus,) = out["models"]
    # 1M read @0.20 + 1M write @5.00 + 0.1M out @20 = 0.20 + 5.00 + 2.00
    assert round(out["total"], 2) == 7.20
    assert opus["calls"] == 1 and opus["fallbacks"] == 1 and opus["hit_rate"] == 0.5


def test_cycle_start_is_last_10th():
    from datetime import datetime, timezone
    utc = timezone.utc
    assert anthropic_backend.cycle_start(datetime(2026, 10, 8, tzinfo=utc)) == \
        datetime(2026, 9, 10, tzinfo=utc).timestamp()
    assert anthropic_backend.cycle_start(datetime(2026, 10, 10, 1, tzinfo=utc)) == \
        datetime(2026, 10, 10, tzinfo=utc).timestamp()
    assert anthropic_backend.cycle_start(datetime(2026, 1, 3, tzinfo=utc)) == \
        datetime(2025, 12, 10, tzinfo=utc).timestamp()


async def test_synthesis_round_keeps_tools_cached(monkeypatch):
    """No-tools synthesis still sends the schemas (cache prefix) with
    tool_choice none, plus the explicit system cache marker."""
    from types import SimpleNamespace
    sent = {}

    class Stream:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

        async def get_final_message(self):
            return SimpleNamespace(
                stop_reason="end_turn", content=[],
                usage=SimpleNamespace(input_tokens=1, output_tokens=1,
                                      cache_read_input_tokens=0,
                                      cache_creation_input_tokens=0))

    class Client:
        def with_options(self, **kw):
            return self

        @property
        def messages(self):
            return SimpleNamespace(stream=lambda **kw: (sent.update(kw), Stream())[1])

    monkeypatch.setattr(anthropic_backend, "_client", lambda *a: Client())
    tools = [{"type": "function", "function": {"name": "f", "parameters": {}}}]
    await anthropic_backend.chat(
        api_key="k", workspace_id="w", model=anthropic_backend.WORKHORSE_MODEL,
        messages=[{"role": "system", "content": "S"}, {"role": "user", "content": "q"}],
        tools=tools, max_tokens=100, timeout=5, allow_tools=False,
    )
    assert sent["tool_choice"] == {"type": "none"} and sent["tools"][0]["name"] == "f"
    assert sent["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert sent["max_tokens"] == 16000


def test_strip_raw_gives_deepseek_reasoning_content():
    raw = [{"type": "thinking", "thinking": "hmm", "signature": "s"},
           {"type": "tool_use", "id": "t1", "name": "f", "input": {}}]
    msgs = [{"role": "assistant", "content": "", anthropic_backend.RAW_KEY: raw},
            {"role": "user", "content": "q"}]
    out = anthropic_backend.strip_raw(msgs)
    assert anthropic_backend.RAW_KEY not in out[0]
    assert out[0]["reasoning_content"] == "hmm"
    assert out[1] is msgs[1] and anthropic_backend.RAW_KEY in msgs[0]  # caller untouched


def test_only_api_trouble_pages_admins():
    import httpx2
    import anthropic
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")

    def bad(msg):
        return anthropic.BadRequestError(msg, response=httpx2.Response(400, request=req), body=None)

    assert anthropic_backend._pages_admins(bad("Your credit balance is too low"))
    assert not anthropic_backend._pages_admins(bad("image exceeds 5 MB"))
    assert anthropic_backend._pages_admins(anthropic.APIConnectionError(request=req))
    assert not anthropic_backend._pages_admins(ValueError("audio"))


def test_mute_key_is_writable(tmp_path):
    from src.settings_store import SettingsStore
    store = SettingsStore(str(tmp_path / "s.db"))
    store.set(anthropic_backend.MUTE_KEY, 123.0)
    assert store.get(anthropic_backend.MUTE_KEY) == 123.0


def test_dashboard_renders(tmp_path, monkeypatch):
    """The dashboard (with the Claude spend panel) loads, as in production:
    a background loop for the windowed metrics, a real metrics DB."""
    import asyncio
    import threading
    from flask import Flask
    from src import metrics_log
    from src.admin.blueprint import create_admin_blueprint
    from src.settings_store import SettingsStore

    db = str(tmp_path / "s.db")
    log = metrics_log.MetricsLog(db)
    monkeypatch.setattr(metrics_log, "_log_instance", log)
    loop = asyncio.new_event_loop()
    threading.Thread(target=loop.run_forever, daemon=True).start()
    try:
        asyncio.run_coroutine_threadsafe(log._ensure_initialized(), loop).result(10)
        app = Flask(__name__)
        app.secret_key = "t"
        create_admin_blueprint(
            app=app, password_hash=b"x", settings_store=SettingsStore(db),
            signal_handler=None, loop=loop, admin_phone="+1",
        )
        client = app.test_client()
        with client.session_transaction() as s:
            s["authenticated"] = True
        resp = client.get("/admin/")
        assert resp.status_code == 200, resp.data[:500]
        assert b"Claude spend" in resp.data
    finally:
        loop.call_soon_threadsafe(loop.stop)


def test_mute_key_not_on_settings_form():
    from src.settings_store import LIVE_KEYS, RESTART_KEYS
    assert anthropic_backend.MUTE_KEY not in LIVE_KEYS | RESTART_KEYS


def test_time_stamp_is_pinned_per_inbound_message():
    """Every tool round of one message renders the same time line, so a
    minute rollover mid-loop can't change the cached user turn."""
    from src.llm.client import _inject_current_time
    msgs = [{"role": "user", "content": "q"}]
    a = _inject_current_time(msgs, now=1_791_500_000.0)
    b = _inject_current_time(msgs, now=1_791_500_000.0)
    assert a == b and "[Current time:" in a[0]["content"]
    assert _inject_current_time(msgs, now=1_791_500_100.0) != a


async def test_handle_webhook_clears_a_stale_context(monkeypatch):
    """A long-lived poller task must not carry the last named group's
    context into the next envelope (a DM poll would pass the Claude gate)."""
    from src.llm.transcript import get_active
    from src.signal.handler import SignalHandler

    handler = SignalHandler.__new__(SignalHandler)
    handler.served_bot_ids = [1]
    monkeypatch.setattr(SignalHandler, "_all_served_disabled", lambda self: True)
    set_active(_ctx())
    await handler.handle_webhook({"envelope": {}})
    assert get_active() is None
