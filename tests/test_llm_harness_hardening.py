"""Regression tests for cache, MCP, idempotency, and retry hardening."""

import json

import aiohttp
import pytest

from src.cache import ProviderMetrics, get_metrics
from src.commands.ask_command import AskCommand
from src.commands.base import CommandContext
from src.contexts.policy import ContextPolicy, MODE_ALLOW_LIST
from src.llm.deep_think import DeepThinkClient
from src.llm.mcp_broker import (
    MCP_BROKER_TOOLS,
    broker_should_be_exposed,
    discover_mcp_tools,
    invoke_mcp_tool,
)
from src.llm.prompt_cache import PromptCachePlan
from src.llm.prompt_compiler import (
    PromptCompiler,
    StablePromptBlock,
    VolatilePromptBlock,
)
from src.llm.resilience import LLMHTTPFailure, resilient_chat_post
from src.llm.tool_runtime import ToolCallLedger
from src.mcp_integration import MCPTool
from src.predictions_resolver import PredictionResolver


class _MCPManager:
    def __init__(self, tools):
        self.tools = list(tools)
        self.calls = []

    def all_tools(self):
        return list(reversed(self.tools))

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return f"result:{name}:{arguments.get('query', '')}"


def _tool(server, name, description):
    return MCPTool(
        server_name=server,
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    )


def _web_policy(servers=None):
    return ContextPolicy(
        id=91,
        kind="group",
        key="g",
        mcp_mode=MODE_ALLOW_LIST,
        mcp_servers=list(servers if servers is not None else ["web"]),
    )


def test_prompt_plan_separates_stable_prefix_from_volatile_tail():
    stable = "You are Artaud."
    first_plan = PromptCachePlan.from_blocks(
        stable=[("base_system", stable)],
        volatile=[("recent_reactions", "👍")],
    )
    second_plan = PromptCachePlan.from_blocks(
        stable=[("base_system", stable)],
        volatile=[("recent_reactions", "💀")],
    )
    first = first_plan.snapshot(
        [{"role": "system", "content": stable}, {"role": "user", "content": "👍"}],
        list(MCP_BROKER_TOOLS),
        purpose="ask", context_id=99101, bot_id=1,
    )
    second = second_plan.snapshot(
        [{"role": "system", "content": stable}, {"role": "user", "content": "💀"}],
        list(MCP_BROKER_TOOLS),
        purpose="ask", context_id=99101, bot_id=1,
    )
    assert first["system_hash"] == second["system_hash"]
    assert first["tools_hash"] == second["tools_hash"]
    assert first["stable_blocks_hash"] == second["stable_blocks_hash"]
    assert first["volatile_blocks_hash"] != second["volatile_blocks_hash"]
    assert "Artaud" not in json.dumps(first)


def test_typed_prompt_compiler_enforces_cache_boundary():
    first_builder = PromptCompiler.with_base_system("You are Artaud.")
    first_builder.add_stable(StablePromptBlock("identity", "Name: Artaud"))
    first_builder.add_volatile(VolatilePromptBlock("reaction", "👍"))
    first = first_builder.compile(user_content="reaction: 👍", tools=list(MCP_BROKER_TOOLS))

    second_builder = PromptCompiler.with_base_system("You are Artaud.")
    second_builder.add_stable(StablePromptBlock("identity", "Name: Artaud"))
    second_builder.add_volatile(VolatilePromptBlock("reaction", "🚂"))
    second = second_builder.compile(user_content="reaction: 🚂", tools=list(MCP_BROKER_TOOLS))

    first.assert_same_cache_prefix(second)
    assert first.messages[-1] != second.messages[-1]
    with pytest.raises(TypeError):
        first_builder.add_stable(VolatilePromptBlock("bad", "dynamic"))


def test_prompt_metrics_flag_stable_changes_and_unchanged_prefix_misses():
    metrics = get_metrics()
    key = {"purpose": "ask", "context_id": 99102, "bot_id": 2}
    base = PromptCachePlan.from_blocks(stable=[("base_system", "A")])
    first = base.snapshot(
        [{"role": "system", "content": "A"}, {"role": "user", "content": "one"}],
        list(MCP_BROKER_TOOLS), **key,
    )
    second = base.snapshot(
        [{"role": "system", "content": "A"}, {"role": "user", "content": "two"}],
        list(MCP_BROKER_TOOLS), **key,
    )
    metrics.record_prompt_cache_observation(first, cache_miss_tokens=2000)
    unchanged = metrics.record_prompt_cache_observation(second, cache_miss_tokens=2000)
    assert unchanged["unexpected_miss"] is True
    assert unchanged["system_changed"] is False

    changed_plan = PromptCachePlan.from_blocks(stable=[("base_system", "B")])
    changed = changed_plan.snapshot(
        [{"role": "system", "content": "B"}, {"role": "user", "content": "three"}],
        list(MCP_BROKER_TOOLS), **key,
    )
    event = metrics.record_prompt_cache_observation(changed)
    assert event["system_changed"] is True
    assert event["stable_changed"] == ["base_system"]


@pytest.mark.asyncio
async def test_mcp_broker_discovers_only_allowed_tools_and_invokes_exact_match():
    manager = _MCPManager([
        _tool("web", "search", "Search current web pages"),
        _tool("web", "fetch", "Fetch a URL"),
        _tool("finance", "quote", "Get a stock quote"),
    ])
    policy = _web_policy()
    result = json.loads(discover_mcp_tools(manager, policy, query="search"))
    assert [row["name"] for row in result["matches"]] == ["web__search"]
    assert result["matches"][0]["parameters"]["required"] == ["query"]

    content = await invoke_mcp_tool(
        manager, policy, name="web__search", arguments={"query": "markets"},
    )
    assert content == "result:web__search:markets"
    blocked = await invoke_mcp_tool(
        manager, policy, name="finance__quote", arguments={"query": "AAPL"},
    )
    assert "blocked" in blocked
    assert manager.calls == [("web__search", {"query": "markets"})]


def test_mcp_broker_presence_is_stable_across_server_restart():
    policy = _web_policy()
    assert broker_should_be_exposed(_MCPManager([]), policy) is True
    assert broker_should_be_exposed(_MCPManager([]), _web_policy([])) is False


def test_tool_call_ledger_detects_ids_and_argument_duplicates():
    ledger = ToolCallLedger()
    ledger.record(
        call_id="call-1", name="portfolio_buy",
        arguments={"symbol": "AAPL", "qty": 1}, content="bought",
    )
    reason, content = ledger.lookup(
        call_id="call-2", name="portfolio_buy",
        arguments={"qty": 1, "symbol": "AAPL"},
    )
    assert reason == "duplicate_arguments"
    assert content == "bought"
    reason, content = ledger.lookup(
        call_id="call-1", name="portfolio_buy",
        arguments={"symbol": "AAPL", "qty": 2},
    )
    assert reason == "duplicate_call_id_conflict"
    assert "suppressed" in content


@pytest.mark.asyncio
async def test_writer_broker_invocation_is_idempotent_within_tool_loop():
    manager = _MCPManager([_tool("web", "search", "Search")])
    ask = AskCommand(llm=object(), history=object(), mcp_manager=manager)
    ctx = CommandContext(
        sender="+1", group_id="g", raw_message="", command="ask", args=[],
        policy=_web_policy(),
    )
    call = {
        "id": "first",
        "function": {
            "name": "mcp__invoke",
            "arguments": json.dumps({
                "name": "web__search", "arguments": {"query": "markets"},
            }),
        },
    }
    second = json.loads(json.dumps(call))
    second["id"] = "second"
    messages = []
    ledger = ToolCallLedger()
    await ask._execute_tool_call(
        call, messages, ctx, [], tools=list(MCP_BROKER_TOOLS), ledger=ledger,
    )
    await ask._execute_tool_call(
        second, messages, ctx, [], tools=list(MCP_BROKER_TOOLS), ledger=ledger,
    )
    assert len(manager.calls) == 1
    assert "duplicate tool call suppressed" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_writer_unknown_tool_returns_bounded_error_envelope():
    ask = AskCommand(llm=object(), history=object())
    ctx = CommandContext(
        sender="+1", group_id=None, raw_message="", command="ask", args=[],
    )
    messages = []
    await ask._execute_tool_call(
        {
            "id": "bad",
            "function": {"name": "invented__tool", "arguments": "{}"},
        },
        messages,
        ctx,
        [],
        tools=[{"type": "function", "function": {"name": "real_tool"}}],
    )
    envelope = json.loads(messages[-1]["content"])
    assert envelope["_tool_result"] == 1
    assert envelope["ok"] is False
    assert envelope["tool"] == "invented__tool"


@pytest.mark.asyncio
async def test_deep_think_broker_invocation_is_idempotent():
    manager = _MCPManager([_tool("web", "search", "Search")])
    client = DeepThinkClient.__new__(DeepThinkClient)
    client.bot_tools = None
    client.mcp_manager = manager
    ctx = CommandContext(
        sender="+1", group_id="g", raw_message="", command="ask", args=[],
        policy=_web_policy(),
    )
    call = {
        "id": "one",
        "function": {
            "name": "mcp__invoke",
            "arguments": json.dumps({
                "name": "web__search", "arguments": {"query": "news"},
            }),
        },
    }
    duplicate = json.loads(json.dumps(call))
    duplicate["id"] = "two"
    messages = []
    ledger = ToolCallLedger()
    await client._dispatch_tool_call(
        call, messages, ctx, [], tools=list(MCP_BROKER_TOOLS), ledger=ledger,
    )
    await client._dispatch_tool_call(
        duplicate, messages, ctx, [], tools=list(MCP_BROKER_TOOLS), ledger=ledger,
    )
    assert len(manager.calls) == 1
    assert "duplicate tool call suppressed" in messages[-1]["content"]


@pytest.mark.asyncio
async def test_prediction_resolver_uses_broker_and_idempotent_dispatch():
    manager = _MCPManager([_tool("web", "search", "Search")])
    resolver = PredictionResolver.__new__(PredictionResolver)
    resolver.bot_tools = None
    resolver.mcp_manager = manager
    policy = _web_policy()
    assert [
        row["function"]["name"]
        for row in resolver._collect_resolver_tools(policy)
    ] == ["mcp__discover", "mcp__invoke"]

    ctx = CommandContext(
        sender="+1", group_id="g", raw_message="", command="resolver",
        args=[], policy=policy,
    )
    call = {
        "id": "resolver-1",
        "function": {
            "name": "mcp__invoke",
            "arguments": json.dumps({
                "name": "web__search", "arguments": {"query": "outcome"},
            }),
        },
    }
    duplicate = json.loads(json.dumps(call))
    duplicate["id"] = "resolver-2"
    messages = []
    ledger = ToolCallLedger()
    await resolver._dispatch_tool_call(call, messages, ctx, ledger=ledger)
    await resolver._dispatch_tool_call(duplicate, messages, ctx, ledger=ledger)
    assert len(manager.calls) == 1
    assert "duplicate tool call suppressed" in messages[-1]["content"]


class _Response:
    def __init__(self, status, data=None, body="", headers=None):
        self.status = status
        self.data = data or {}
        self.body = body
        self.headers = headers or {}

    async def __aenter__(self):
        return self

    async def __aexit__(self, *args):
        return False

    async def text(self):
        return self.body

    async def json(self, content_type=None):
        return self.data


class _Session:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0

    def post(self, *args, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


@pytest.mark.asyncio
async def test_resilience_honors_retry_after_and_succeeds_within_bound():
    session = _Session([
        _Response(429, body="slow down", headers={"Retry-After": "0"}),
        _Response(200, data={"choices": []}),
    ])
    provider = ProviderMetrics("test-retry")
    retries = []

    async def no_sleep(delay):
        retries.append(delay)

    data = await resilient_chat_post(
        session=session, url="https://example.test", payload={}, headers={},
        request_timeout=2, hard_timeout=5, provider_metrics=provider,
        retry_attempts=2, on_retry=lambda reason, attempt, delay: retries.append(reason),
        sleep=no_sleep,
    )
    assert data == {"choices": []}
    assert session.calls == 2
    assert retries == ["http_429"]
    assert provider.successes == 1
    assert provider.errors == 1
    assert provider.consecutive_errors == 0


@pytest.mark.asyncio
async def test_resilience_does_not_retry_nontransient_http_error():
    session = _Session([_Response(401, body="bad key")])
    with pytest.raises(LLMHTTPFailure) as exc:
        await resilient_chat_post(
            session=session, url="https://example.test", payload={}, headers={},
            request_timeout=2, hard_timeout=5,
            provider_metrics=ProviderMetrics("test-auth"), retry_attempts=2,
        )
    assert exc.value.status == 401
    assert session.calls == 1


@pytest.mark.asyncio
async def test_resilience_opens_circuit_after_fifth_transient_failure():
    provider = ProviderMetrics("test-circuit")
    provider.consecutive_errors = 4
    with pytest.raises(LLMHTTPFailure):
        await resilient_chat_post(
            session=_Session([_Response(503, body="unavailable")]),
            url="https://example.test", payload={}, headers={},
            request_timeout=2, hard_timeout=5, provider_metrics=provider,
            retry_attempts=0,
        )
    assert provider.circuit_open is True
    assert provider.is_healthy() is False


@pytest.mark.asyncio
async def test_resilience_retries_network_disconnect_and_circuit_fails_fast():
    session = _Session([
        aiohttp.ClientConnectionError("reset"),
        _Response(200, data={"ok": True}),
    ])
    provider = ProviderMetrics("test-network")

    async def no_sleep(_delay):
        return None

    data = await resilient_chat_post(
        session=session, url="https://example.test", payload={}, headers={},
        request_timeout=2, hard_timeout=5, provider_metrics=provider,
        sleep=no_sleep,
    )
    assert data == {"ok": True}
    assert session.calls == 2

    provider.open_circuit(60)
    with pytest.raises(RuntimeError, match="circuit is open"):
        await resilient_chat_post(
            session=_Session([]), url="https://example.test", payload={}, headers={},
            request_timeout=2, hard_timeout=5, provider_metrics=provider,
        )
