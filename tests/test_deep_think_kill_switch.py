"""
Tests for the per-bot deep_think kill switch.

Covers:
  * Bot.deep_think_enabled defaults to True (backwards compatibility)
  * Round-trip through BotRegistry preserves both states
  * AskCommand._collect_tools hides the deep_think schema when off
  * AskCommand source enforces the flag on the research-mode branch
"""

import tempfile
from pathlib import Path

import pytest

from src.bots import Bot, BotRegistry
from src.contexts.policy import ContextPolicy, MODE_ALLOW_LIST
from src.llm.deep_think import DeepThinkClient
from src.mcp_integration import MCPTool


# ── Model defaults & registry round-trip ───────────────────────────────────

@pytest.fixture
async def registry():
    with tempfile.TemporaryDirectory() as d:
        reg = BotRegistry(db_path=str(Path(d) / "bots.db"))
        yield reg


def test_model_default_is_on():
    """New Bot instances default to deep_think_enabled=True so the flag
    is opt-out, not opt-in — preserves prior behavior for existing bots."""
    b = Bot(id=None, slug="x", display_name="X")
    assert b.deep_think_enabled is True


@pytest.mark.asyncio
async def test_registry_round_trip_on(registry):
    b = Bot(id=None, slug="bot-on", display_name="On", aliases=["o"])
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.deep_think_enabled is True


@pytest.mark.asyncio
async def test_registry_round_trip_off(registry):
    b = Bot(
        id=None, slug="bot-off", display_name="Off", aliases=["o"],
        deep_think_enabled=False,
    )
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.deep_think_enabled is False


@pytest.mark.asyncio
async def test_registry_toggle_back_on(registry):
    b = Bot(
        id=None, slug="bot-toggle", display_name="Toggle", aliases=["t"],
        deep_think_enabled=False,
    )
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    reloaded.deep_think_enabled = True
    await registry.upsert(reloaded)
    again = await registry.get(bid)
    assert again is not None
    assert again.deep_think_enabled is True


# ── _collect_tools schema filtering ────────────────────────────────────────

class FakeDeepThink:
    """Minimal DeepThinkClient stand-in for _collect_tools tests."""

    def __init__(self, ready=True):
        self.ready = ready

    def status(self):
        return {"ready": self.ready}


class FakeFactory:
    def __init__(self, clients):
        self.clients = clients

    def get_deep_think(self, bot_id):
        return self.clients[bot_id]


class FakeMCPManager:
    def __init__(self, tools):
        self.tools = tools

    def all_tools(self):
        # Deliberately return reverse lexical order. Both harnesses must
        # canonicalize it so restart/insertion order cannot rewrite the cache.
        return list(self.tools)


class FakePolicy:
    """Permissive policy: lets deep_think through at the policy layer."""
    id = 99
    kind = "group"

    def allows_deep_think(self):
        return True

    def allows_mcp(self, *_):
        return True


def _probe(deep_think=None):
    """Build a minimal AskCommand-shaped object that only supports
    _collect_tools. Cheaper than instantiating the real class."""
    from src.commands.ask_command import AskCommand

    class Probe:
        pass

    Probe._collect_tools = AskCommand._collect_tools
    p = Probe()
    p.bot_tools = None
    p.mcp_manager = None
    p.deep_think = deep_think
    p.llm_factory = None
    p.memory_store = None
    p.prediction_store = None
    p.portfolio_executor = None
    p.portfolio_journal = None
    return p


def _schema_names(tools):
    if not tools:
        return set()
    return {(t.get("function") or {}).get("name") for t in tools}


def test_deep_think_schema_included_when_bot_flag_on():
    probe = _probe(deep_think=FakeDeepThink())
    bot = Bot(id=1, slug="x", display_name="X", deep_think_enabled=True)
    tools = probe._collect_tools(policy=FakePolicy(), bot=bot)
    assert "deep_think" in _schema_names(tools)


def test_deep_think_schema_hidden_when_bot_flag_off():
    """The kill switch must hide the tool schema entirely so the writer
    can't even attempt to invoke it. Suppressing here also avoids the
    writer wasting a tool-call round on a guaranteed-stub call."""
    probe = _probe(deep_think=FakeDeepThink())
    bot = Bot(id=1, slug="x", display_name="X", deep_think_enabled=False)
    tools = probe._collect_tools(policy=FakePolicy(), bot=bot)
    assert "deep_think" not in _schema_names(tools)


def test_deep_think_schema_included_when_no_bot_passed():
    """Legacy callers that don't pass `bot=...` keep prior behavior:
    the tool is exposed if deep_think client is wired + policy allows."""
    probe = _probe(deep_think=FakeDeepThink())
    tools = probe._collect_tools(policy=FakePolicy(), bot=None)
    assert "deep_think" in _schema_names(tools)


def test_deep_think_schema_uses_active_bot_client():
    """A non-default bot's schema gate must follow its factory client.

    The default client is deliberately unready here; the active bot has a
    ready override and should still receive the tool.
    """
    probe = _probe(deep_think=FakeDeepThink(ready=False))
    probe.llm_factory = FakeFactory({7: FakeDeepThink(ready=True)})
    bot = Bot(id=7, slug="artaud", display_name="Artaud")
    tools = probe._collect_tools(policy=FakePolicy(), bot=bot)
    assert "deep_think" in _schema_names(tools)


def test_deep_think_schema_does_not_leak_from_default_bot():
    """A ready default client cannot advertise deep-think for an active
    bot whose own role client is disabled or unconfigured."""
    probe = _probe(deep_think=FakeDeepThink(ready=True))
    probe.llm_factory = FakeFactory({7: FakeDeepThink(ready=False)})
    bot = Bot(id=7, slug="artaud", display_name="Artaud")
    tools = probe._collect_tools(policy=FakePolicy(), bot=bot)
    assert "deep_think" not in _schema_names(tools)


def _mcp_tool(server, name):
    return MCPTool(
        server_name=server,
        name=name,
        description=f"{server} {name}",
        input_schema={"type": "object", "properties": {}},
    )


def test_writer_sends_stable_mcp_broker_for_nonempty_allowlist():
    probe = _probe()
    probe.mcp_manager = FakeMCPManager([
        _mcp_tool("blocked", "zeta"),
        _mcp_tool("web", "zeta"),
        _mcp_tool("web", "alpha"),
    ])
    policy = ContextPolicy(
        id=1,
        kind="group",
        key="g",
        mcp_mode=MODE_ALLOW_LIST,
        mcp_servers=["web"],
    )

    tools = probe._collect_tools(policy=policy)
    assert [t["function"]["name"] for t in tools] == [
        "mcp__discover",
        "mcp__invoke",
    ]


def test_deep_think_uses_same_stable_mcp_broker():
    client = DeepThinkClient.__new__(DeepThinkClient)
    client.bot_tools = None
    client.memory_store = None
    client.mcp_manager = FakeMCPManager([
        _mcp_tool("blocked", "zeta"),
        _mcp_tool("web", "zeta"),
        _mcp_tool("web", "alpha"),
    ])
    policy = ContextPolicy(
        id=1,
        kind="group",
        key="g",
        mcp_mode=MODE_ALLOW_LIST,
        mcp_servers=["web"],
    )

    tools = client._collect_tools(policy=policy)
    assert [t["function"]["name"] for t in tools] == [
        "mcp__discover",
        "mcp__invoke",
    ]


# ── Research-mode gating in ask source ─────────────────────────────────────

def test_research_mode_gates_on_deep_think_enabled():
    """Verify the `use_research_mode` predicate includes the per-bot
    flag. A test that runs the full ask path needs an LLM mock stack
    that doesn't exist; checking the source is a cheap regression
    guard against the predicate silently losing the gate.
    """
    import src.commands.ask_command as ask_mod
    src_text = open(ask_mod.__file__, encoding="utf-8").read()
    # Walk the predicate by paren-balance instead of first ")", which
    # would stop inside getattr(ctx.bot, "deep_think_mode", ...).
    start = src_text.find("use_research_mode = (")
    assert start >= 0, "use_research_mode predicate not found"
    depth = 0
    end = start
    for i in range(start, len(src_text)):
        c = src_text[i]
        if c == "(":
            depth += 1
        elif c == ")":
            depth -= 1
            if depth == 0:
                end = i
                break
    block = src_text[start:end + 1]
    assert 'deep_think_mode' in block
    assert 'deep_think_enabled' in block


def test_dispatch_rejects_call_when_bot_flag_off():
    """Defense-in-depth: even if a stale schema leaks through, the tool
    dispatcher rejects deep_think calls when the bot has it disabled."""
    import src.commands.ask_command as ask_mod
    src_text = open(ask_mod.__file__, encoding="utf-8").read()
    assert "deep_think unavailable: disabled for this bot" in src_text
