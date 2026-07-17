"""Tests for deep_think_mode='tool_bot' — the gated tool-executor handoff.

Covers:
- ToolBotClient config chain (tool_bot_* -> deep_think_* -> llm_*) and the
  default-enabled / self-gating-prompt behaviour
- NO_TOOLS sentinel detection (`is_no_tools`)
- LLMClientFactory.get_tool_bot caching + late-bound tool propagation
- AskCommand._run_tool_bot_handoff:
    * NOTOOLS gate → writer answers directly, no notes injected, no tools
    * tools-needed → notes folded into the volatile tail, writer gets no tools
    * per-bot handoff template {notes} substitution
    * caller's `messages` list is not mutated
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots import Bot
from src.commands.base import CommandContext
from src.llm.factory import LLMClientFactory
from src.llm.tool_bot import (
    ToolBotClient,
    DEFAULT_TOOL_BOT_PROMPT,
    NO_TOOLS_SENTINEL,
)
from src.settings_store import SettingsStore


@pytest.fixture
def tmpdb(tmp_path: Path) -> str:
    return str(tmp_path / "tool_bot_test.db")


@pytest.fixture
def store(tmpdb: str) -> SettingsStore:
    s = SettingsStore(tmpdb)
    s.set("llm_base_url", "https://llm.example/v1")
    s.set("llm_api_key", "sk-llm")
    s.set("llm_model", "writer-model")
    return s


# ──────────────────────────────────────────────────────────────────
# is_no_tools
# ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "value,expected",
    [
        ("NOTOOLS", True),
        ("notools", True),
        ("  NOTOOLS\n", True),
        ("`NOTOOLS`", True),
        ('"NOTOOLS".', True),
        (None, False),
        ("", False),
        ("The price is 5; no tools were needed", False),
        ("NOTOOLS because the answer is 42", False),
    ],
)
def test_is_no_tools(value, expected):
    assert ToolBotClient.is_no_tools(value) is expected


def test_sentinel_constant_stable():
    assert NO_TOOLS_SENTINEL == "NOTOOLS"


# ──────────────────────────────────────────────────────────────────
# ToolBotClient config chain
# ──────────────────────────────────────────────────────────────────


def test_tool_bot_defaults_enabled_and_prompt(store: SettingsStore):
    """Selecting the mode is the opt-in: with only a per-bot model set the
    client is enabled and uses the self-gating default prompt."""
    store.set_bot(5, "tool_bot", "model", "tb-model")
    tb = ToolBotClient(store, bot_id=5)
    cfg = tb._config()
    assert cfg["enabled"] is True
    assert cfg["model"] == "tb-model"
    assert cfg["system_prompt"] == DEFAULT_TOOL_BOT_PROMPT
    assert NO_TOOLS_SENTINEL in cfg["system_prompt"]  # gate instruction present


def test_tool_bot_falls_back_deep_think_then_llm(store: SettingsStore):
    """base_url resolves tool_bot_ -> deep_think_ -> llm_; api_key only
    set at llm_ tier is still picked up."""
    store.set("deep_think_base_url", "https://dt.example/v1")
    store.set("deep_think_model", "dt-model")
    tb = ToolBotClient(store, bot_id=5)
    cfg = tb._config()
    # tool_bot_* unset → deep_think_* wins over llm_* for base_url/model
    assert cfg["base_url"] == "https://dt.example/v1"
    assert cfg["model"] == "dt-model"
    # api_key only at llm_ tier
    assert cfg["api_key"] == "sk-llm"


def test_tool_bot_per_bot_override_wins(store: SettingsStore):
    # deep_think global is the inherited tier; the per-bot tool_bot row
    # must win over it, and per-bot 'enabled' can turn the mode off.
    store.set("deep_think_model", "global-dt")
    store.set_bot(5, "tool_bot", "model", "perbot-tb")
    store.set_bot(5, "tool_bot", "enabled", False)
    tb = ToolBotClient(store, bot_id=5)
    cfg = tb._config()
    assert cfg["model"] == "perbot-tb"
    assert cfg["enabled"] is False


def test_tool_bot_status_ready_requires_endpoint(store: SettingsStore):
    tb = ToolBotClient(store, bot_id=5)
    # No model/base_url/api_key configured beyond llm globals: base_url +
    # api_key inherit from llm, but model inherits writer-model too, so it
    # is ready off the llm tier. Assert the ready contract explicitly.
    st = tb.status()
    assert st["ready"] == (
        st["enabled"] and bool(st["base_url"]) and bool(st["model"])
    )


# ──────────────────────────────────────────────────────────────────
# Direct MCP exposure (no broker) — the weak-model fix
# ──────────────────────────────────────────────────────────────────


class _FakeMCPTool:
    def __init__(self, server_name, name, description="", input_schema=None):
        self.server_name = server_name
        self.name = name
        self.description = description
        self.input_schema = input_schema or {"type": "object", "properties": {}}

    @property
    def qualified_name(self):
        return f"{self.server_name}__{self.name}"

    def to_openai_tool(self):
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description or self.name,
                "parameters": self.input_schema,
            },
        }


class _FakeMCPManager:
    def __init__(self, tools):
        self._tools = tools

    def all_tools(self):
        return list(self._tools)


def test_tool_bot_exposes_mcp_directly_not_broker(store: SettingsStore):
    """_collect_tools must return expanded MCP tool schemas (callable by
    name), NOT the mcp__discover/mcp__invoke broker pair."""
    mgr = _FakeMCPManager([
        _FakeMCPTool("brave-search", "brave_news_search", "Search news."),
        _FakeMCPTool("brave-search", "brave_web_search", "Web search."),
    ])
    tb = ToolBotClient(store, bot_id=2, mcp_manager=mgr)
    schemas = tb._collect_tools(policy=None)
    names = {(s.get("function") or {}).get("name") for s in schemas}
    assert "brave-search__brave_news_search" in names
    assert "brave-search__brave_web_search" in names
    # The broker must NOT be present — that's the whole point.
    assert "mcp__discover" not in names
    assert "mcp__invoke" not in names


def test_tool_bot_invalid_fn_name_keeps_broker(store: SettingsStore):
    """A tool whose qualified name isn't a valid function name can't be
    exposed directly, so the broker stays as a reachable fallback."""
    bad = "x" * 70  # exceeds the 64-char function-name limit
    mgr = _FakeMCPManager([_FakeMCPTool("srv", bad, "too long")])
    tb = ToolBotClient(store, bot_id=2, mcp_manager=mgr)
    schemas = tb._collect_tools(policy=None)
    names = {(s.get("function") or {}).get("name") for s in schemas}
    assert f"srv__{bad}" not in names          # not directly exposed
    assert "mcp__discover" in names            # broker fallback present
    assert "mcp__invoke" in names


@pytest.mark.asyncio
async def test_tool_bot_direct_mcp_call_routes_to_invoke(store, monkeypatch):
    """A direct `server__tool` call must invoke the MCP tool, not return
    the parent's 'use mcp__invoke' rejection."""
    mgr = _FakeMCPManager([_FakeMCPTool("brave-search", "brave_news_search")])
    tb = ToolBotClient(store, bot_id=2, mcp_manager=mgr)

    called = {}

    async def fake_invoke(mcp_manager, policy, *, name, arguments):
        called["name"] = name
        called["arguments"] = arguments
        return "NEWS RESULTS: headline 1; headline 2"

    monkeypatch.setattr("src.llm.tool_bot.invoke_mcp_tool", fake_invoke)
    out = await tb._handle_direct_mcp_call(
        "brave-search__brave_news_search", {"query": "today"}, None,
    )
    assert called["name"] == "brave-search__brave_news_search"
    assert called["arguments"] == {"query": "today"}
    assert "NEWS RESULTS" in out


def test_deep_think_still_rejects_direct_mcp(store: SettingsStore):
    """Regression: base DeepThinkClient must still reject direct MCP calls
    (it only reaches MCP through the broker)."""
    import asyncio
    from src.llm.deep_think import DeepThinkClient

    dt = DeepThinkClient(store, bot_id=2)
    out = asyncio.get_event_loop().run_until_complete(
        dt._handle_direct_mcp_call("brave-search__x", {}, None)
    )
    assert "mcp__invoke" in out and "unavailable" in out


# ──────────────────────────────────────────────────────────────────
# Factory
# ──────────────────────────────────────────────────────────────────


def test_factory_get_tool_bot_caches(store: SettingsStore):
    f = LLMClientFactory(store)
    a = f.get_tool_bot(3)
    b = f.get_tool_bot(3)
    assert a is b
    assert isinstance(a, ToolBotClient)
    assert f.get_tool_bot(4) is not a


def test_factory_attach_propagates_to_tool_bot(store: SettingsStore):
    f = LLMClientFactory(store)
    tb = f.get_tool_bot(1)
    assert tb.bot_tools is None
    tools_sentinel = object()
    mcp_sentinel = object()
    f.attach_bot_tools(tools_sentinel)
    f.attach_mcp_manager(mcp_sentinel)
    assert tb.bot_tools is tools_sentinel
    assert tb.mcp_manager is mcp_sentinel
    # New lookups also inherit.
    tb2 = f.get_tool_bot(2)
    assert tb2.bot_tools is tools_sentinel
    assert tb2.mcp_manager is mcp_sentinel


def test_factory_forget_drops_tool_bot(store: SettingsStore):
    f = LLMClientFactory(store)
    a = f.get_tool_bot(9)
    f.forget(9)
    assert f.get_tool_bot(9) is not a


# ──────────────────────────────────────────────────────────────────
# _run_tool_bot_handoff
# ──────────────────────────────────────────────────────────────────


def _make_ask(writer_reply="composed reply"):
    from src.commands.ask_command import AskCommand

    ask = AskCommand.__new__(AskCommand)
    ask.signal_handler = None
    ask.llm_factory = None
    captured: dict = {}

    async def chat(messages, tools=None, **kwargs):
        captured["messages"] = messages
        captured["tools"] = tools
        return {"role": "assistant", "content": writer_reply}

    ask.llm = MagicMock()
    ask.llm.chat_messages = AsyncMock(side_effect=chat)
    return ask, captured


def _ctx(bot):
    return CommandContext(
        sender="+1", group_id="!G", raw_message="!ask",
        command="ask", args=[], policy=None, bot=bot,
    )


@pytest.mark.asyncio
async def test_notools_gate_writer_answers_directly():
    """NOTOOLS sentinel → writer composes from the original messages with
    no injected handoff notes and no tools."""
    ask, captured = _make_ask()
    tool_bot = MagicMock()
    tool_bot.think = AsyncMock(return_value="NOTOOLS")

    bot = Bot(id=42, slug="artaud", display_name="Artaud", aliases=["Artaud"],
              deep_think_mode="tool_bot")
    answer = await ask._run_tool_bot_handoff(
        ctx=_ctx(bot), question="how are you?",
        research_input="user said hi",
        messages=[{"role": "system", "content": "base"},
                  {"role": "user", "content": "how are you?"}],
        attachments=[], user_hash="abc", tool_bot=tool_bot,
    )
    assert answer == "composed reply"
    assert captured["tools"] is None
    # No research_handoff block injected on the NOTOOLS path.
    joined = "".join(str(m.get("content")) for m in captured["messages"])
    assert "research_handoff" not in joined
    assert "NOTOOLS" not in joined


@pytest.mark.asyncio
async def test_tools_needed_notes_injected():
    """Non-sentinel notes → folded into the volatile user tail; writer
    still gets no tools."""
    ask, captured = _make_ask()
    tool_bot = MagicMock()
    tool_bot.think = AsyncMock(return_value="AAPL last 214.30 (live)")

    bot = Bot(id=42, slug="artaud", display_name="Artaud", aliases=["Artaud"],
              deep_think_mode="tool_bot")
    answer = await ask._run_tool_bot_handoff(
        ctx=_ctx(bot), question="aapl price?",
        research_input="user asked price",
        messages=[{"role": "system", "content": "base"},
                  {"role": "user", "content": "aapl price?"}],
        attachments=[], user_hash="abc", tool_bot=tool_bot,
    )
    assert answer == "composed reply"
    assert captured["tools"] is None
    tail = captured["messages"][-1]["content"]
    assert "AAPL last 214.30" in str(tail)
    assert "research_handoff" in str(tail)


@pytest.mark.asyncio
async def test_tool_bot_handoff_template_substitutes_notes():
    ask, captured = _make_ask()
    tool_bot = MagicMock()
    tool_bot.think = AsyncMock(return_value="FINDINGS")

    bot = Bot(id=42, slug="artaud", display_name="Artaud", aliases=["Artaud"],
              deep_think_mode="tool_bot",
              deep_think_handoff_prompt="Notes:\n{notes}\nCompose.")
    await ask._run_tool_bot_handoff(
        ctx=_ctx(bot), question="?", research_input="x",
        messages=[{"role": "system", "content": "base"},
                  {"role": "user", "content": "?"}],
        attachments=[], user_hash="", tool_bot=tool_bot,
    )
    tail = str(captured["messages"][-1]["content"])
    assert "FINDINGS" in tail
    assert "Compose." in tail


@pytest.mark.asyncio
async def test_tool_bot_handoff_does_not_mutate_callers_messages():
    ask, _ = _make_ask()
    tool_bot = MagicMock()
    tool_bot.think = AsyncMock(return_value="LEAKED NOTES")

    bot = Bot(id=1, slug="x", display_name="X", aliases=["X"],
              deep_think_mode="tool_bot")
    original = [{"role": "system", "content": "base"},
                {"role": "user", "content": "hi"}]
    await ask._run_tool_bot_handoff(
        ctx=_ctx(bot), question="?", research_input="x",
        messages=original, attachments=[], user_hash="", tool_bot=tool_bot,
    )
    assert original == [{"role": "system", "content": "base"},
                        {"role": "user", "content": "hi"}]


@pytest.mark.asyncio
async def test_tool_bot_unavailable_stub_is_injected_not_gated():
    """An unavailable/error stub is NOT the sentinel, so it must be folded
    into the notes (so the writer can be honest), not silently dropped."""
    ask, captured = _make_ask()
    tool_bot = MagicMock()
    tool_bot.think = AsyncMock(
        return_value="(tool_bot unavailable: not configured)"
    )

    bot = Bot(id=42, slug="artaud", display_name="Artaud", aliases=["Artaud"],
              deep_think_mode="tool_bot")
    await ask._run_tool_bot_handoff(
        ctx=_ctx(bot), question="?", research_input="x",
        messages=[{"role": "system", "content": "base"},
                  {"role": "user", "content": "?"}],
        attachments=[], user_hash="", tool_bot=tool_bot,
    )
    tail = str(captured["messages"][-1]["content"])
    assert "tool_bot unavailable" in tail
    assert "research_handoff" in tail
