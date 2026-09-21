"""JEV transport, bounded decisions, and fallback behavior across all four uses."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from aioresponses import aioresponses

from src.admin.blueprint import _apply_llm_form
from src.bots.models import Bot
from src.cache import MetricsCollector
from src.contexts.policy import ContextPolicy
from src.llm import jev as jev_module
from src.llm.jev import JEV_URL, JevClient
from src.llm.mcp_broker import discover_mcp_tools
from src.llm.reactor import EmojiReactor
from src.mcp_integration.models import MCPTool
from src.settings_store import SettingsStore
from src.signal.poll_voter import PollVoter
from src.wsb.digest import build_tally, classify_sentiment


@pytest.fixture
def store(tmp_path, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    monkeypatch.setattr(MetricsCollector, "_instance", None)
    metrics = MetricsCollector()
    monkeypatch.setattr(jev_module, "get_metrics", lambda: metrics)
    store = SettingsStore(str(tmp_path / "settings.db"))
    store.set("llm_base_url", "https://openrouter.ai/api/v1")
    store.set("llm_api_key", "test-key")
    return store


QUESTION = {"route": {"type": "choice", "instructions": "Choose a capability.",
                      "criteria": {"weather": "Weather", "finance": "Stock quotes"}}}


async def test_transport_uses_decisions_endpoint_and_validates_choices(store):
    client = JevClient(store)
    with aioresponses() as http:
        http.post(JEV_URL, payload={"answers": {"route": {
            "type": "choice", "choice": "weather", "confidence": 0.95,
        }}, "usage": {"input_tokens": 123, "output_tokens": 20}})
        try:
            assert await client.choose(state="weather please", questions=QUESTION, purpose="test") == {"route": "weather"}
            request = next(iter(http.requests.values()))[0].kwargs
            assert request["json"] == {
                "model": "~typesafe/jev-latest", "state": "weather please", "questions": QUESTION,
            }
            assert request["headers"]["Authorization"] == "Bearer test-key"
            assert jev_module.get_metrics()._llm.tokens_in == 123
        finally:
            await client.close()
    assert client._session is None


@pytest.mark.parametrize("answer", [
    {"type": "choice", "choice": "weather", "confidence": 0.79},
    {"type": "choice", "choice": "unauthorized", "confidence": 1},
    {"type": "choice", "choice": "weather", "confidence": float("nan")},
    {"type": "choice", "choice": "weather", "confidence": True},
    {"type": "choice", "choice": "weather", "confidence": "1"},
    {"type": "choice", "choice": "weather", "confidence": 1.1},
    {"type": "score", "choice": "weather", "confidence": 1},
    None,
])
async def test_uncertain_or_malformed_choice_falls_back(store, answer):
    client = JevClient(store)
    with aioresponses() as http:
        http.post(JEV_URL, payload={"answers": {"route": answer}})
        try:
            assert await client.choose(state="x", questions=QUESTION, purpose="test") == {}
        finally:
            await client.close()


async def test_errors_open_circuit_and_cancellation_propagates(store):
    client = JevClient(store)
    with aioresponses() as http:
        http.post(JEV_URL, status=429)
        http.post(JEV_URL, exception=asyncio.TimeoutError())
        http.post(JEV_URL, payload=[])
        try:
            for _ in range(4):
                assert await client.choose(state="x", questions=QUESTION, purpose="test") == {}
            assert sum(map(len, http.requests.values())) == 3
            jev_module.get_metrics().get_provider_metrics("jev:~typesafe/jev-latest").close_circuit()
            http.post(JEV_URL, exception=asyncio.CancelledError())
            with pytest.raises(asyncio.CancelledError):
                await client.choose(state="x", questions=QUESTION, purpose="test")
        finally:
            await client.close()


async def test_disabled_oversized_and_non_openrouter_credentials_never_send(store):
    client = JevClient(store)
    assert client.enabled()
    store.set("llm_base_url", "https://openrouter.ai.evil.example/api/v1")
    assert not client.enabled()
    store.set("jev_api_key", "dedicated")
    assert client.enabled()
    with aioresponses() as http:
        assert await client.choose(state="x" * 100_000, questions=QUESTION, purpose="test") == {}
        store.set("jev_enabled", False)
        assert await client.choose(state="x", questions=QUESTION, purpose="test") == {}
        assert not http.requests


def test_admin_form_retains_secret_and_enables_defaults(store):
    _apply_llm_form(store, {"jev_enabled": "true", "jev_api_key": "dedicated", "jev_min_confidence": "0.9"})
    _apply_llm_form(store, {"jev_api_key": ""})
    assert store.get("jev_api_key") == "dedicated"
    assert store.get_bool("jev_enabled")
    assert store.get_float("jev_min_confidence", 0) == 0.9


@pytest.mark.parametrize("selection, legacy_calls, reacts, responds", [
    ({"action": "ignore"}, 0, 0, 0),
    ({"action": "respond"}, 0, 0, 1),
    ({"action": "react"}, 1, 1, 0),
    ({}, 1, 1, 0),
])
async def test_reactor_routes_with_compact_prompt_and_fallback(store, selection, legacy_calls, reacts, responds):
    store.set("reactor_enabled", True)
    store.set("natural_response_enabled", True)
    llm = SimpleNamespace(chat_messages=AsyncMock(return_value={"tool_calls": [{"function": {
        "name": "emoji_react", "arguments": {"emoji": "🔥", "score": 8},
    }}]}))
    jev = SimpleNamespace(choose=AsyncMock(return_value=selection))
    signal = SimpleNamespace(send_reaction=AsyncMock(return_value=True))
    reactor = EmojiReactor(store, llm, signal, jev=jev)
    reactor.implicit_response_handler = AsyncMock()
    policy = ContextPolicy(id=1, kind="group", key="g", natural_response=True)
    await reactor.maybe_react(sender="alice", message="I finally got promoted!", group_id="g",
                             target_timestamp=123, policy=policy)
    assert llm.chat_messages.await_count == legacy_calls
    assert signal.send_reaction.await_count == reacts
    assert reactor.implicit_response_handler.await_count == responds
    question = jev.choose.await_args.kwargs["questions"]["action"]
    assert len(question["instructions"]) < 800
    assert "tool" not in question["instructions"]
    if selection.get("action") == "react":
        assert len(llm.chat_messages.await_args.kwargs["tools"]) == 1


async def test_reactor_respects_reply_suppression_and_custom_rules(store):
    store.set("reactor_enabled", True)
    store.set("reactor_system_prompt", "React only to promotions.")
    jev = SimpleNamespace(choose=AsyncMock(return_value={"action": "ignore"}))
    reactor = EmojiReactor(store, None, None, jev=jev)
    kwargs = dict(sender="alice", message="promotion!", group_id="g", target_timestamp=123)
    await reactor.maybe_react(**kwargs, bot_will_reply=True)
    jev.choose.assert_not_awaited()
    await reactor.maybe_react(**kwargs)
    question = jev.choose.await_args.kwargs["questions"]["action"]
    assert "React only to promotions." in question["instructions"]
    assert set(question["criteria"]) == {"ignore", "react"}


async def test_reactor_selects_eligible_bot_and_keeps_media(store):
    store.set("reactor_enabled", True)
    store.set("natural_response_enabled", True)
    bots = [Bot(id=1, slug="finance", display_name="Finance", routing_blurb="Stocks and markets"),
            Bot(id=2, slug="music", display_name="Music", routing_blurb="Music and recordings")]
    jev = SimpleNamespace(choose=AsyncMock(return_value={"action": "respond:music"}))
    reactor = EmojiReactor(store, None, None, jev=jev)
    reactor.implicit_response_handler = AsyncMock()
    policy = ContextPolicy(id=1, kind="group", key="g", natural_response=True)
    audio = [{"data": "unchanged-media"}]
    await reactor.maybe_react(sender="alice", message="What do you think of this recording?",
                             group_id="g", target_timestamp=123, policy=policy,
                             bot=bots[0], candidate_bots=bots, inbound_audio=audio)
    args = reactor.implicit_response_handler.await_args.kwargs
    assert args["bot_override"] is bots[1]
    assert args["inbound_audio"] is audio
    payload = jev.choose.await_args.kwargs
    assert "unchanged-media" not in json.dumps(payload)
    reactor.implicit_response_handler.reset_mock()
    jev.choose.return_value = {"action": "ignore"}
    await reactor.maybe_react(sender="alice", message="And this recording?", group_id="g",
                             target_timestamp=124, policy=policy, bot=bots[0], candidate_bots=bots)
    assert "respond:music" not in jev.choose.await_args.kwargs["questions"]["action"]["criteria"]


async def test_reply_only_prompt_omits_emoji_rules_without_losing_followup(store):
    store.set("reactor_enabled", True)
    store.set("natural_response_enabled", True)
    store.set("reactor_min_length", 40)
    store.set("reactor_system_prompt", "Do not react to short messages. EMOJI_ONLY_RULE")
    jev = SimpleNamespace(choose=AsyncMock(return_value={"action": "respond"}))
    reactor = EmojiReactor(store, None, None, jev=jev)
    reactor.implicit_response_handler = AsyncMock()
    await reactor.maybe_react(sender="alice", message="Why?", group_id="g", target_timestamp=123,
                             policy=ContextPolicy(id=1, kind="group", key="g", natural_response=True))
    question = jev.choose.await_args.kwargs["questions"]["action"]
    assert set(question["criteria"]) == {"ignore", "respond"}
    assert "EMOJI_ONLY_RULE" not in question["instructions"]
    assert "short questions" in question["instructions"]
    reactor.implicit_response_handler.assert_awaited_once()


async def test_emoji_selection_cannot_turn_into_a_spontaneous_reply(store):
    store.set("reactor_enabled", True)
    store.set("natural_response_enabled", True)
    llm = SimpleNamespace(chat_messages=AsyncMock(return_value={"tool_calls": [{"function": {
        "name": "should_respond", "arguments": {"reason": "Unexpected tool call"},
    }}]}))
    jev = SimpleNamespace(choose=AsyncMock(return_value={"action": "react"}))
    reactor = EmojiReactor(store, llm, None, jev=jev)
    reactor.implicit_response_handler = AsyncMock()
    await reactor.maybe_react(sender="alice", message="I got promoted!", group_id="g", target_timestamp=123,
                             policy=ContextPolicy(id=1, kind="group", key="g", natural_response=True))
    reactor.implicit_response_handler.assert_not_awaited()


@pytest.mark.parametrize("multiple, choices, expected, fallback", [
    (False, {"vote": "0"}, [0], False),
    (False, {}, [1], True),
    (True, {"0": "yes", "1": "no", "2": "yes"}, [0, 2], False),
    (True, {"0": "yes"}, [1], True),
    (True, {"0": "no", "1": "no", "2": "no"}, None, False),
])
async def test_poll_choices_and_partial_fallback(store, multiple, choices, expected, fallback):
    llm = SimpleNamespace(status=lambda: {"ready": True}, chat_messages=AsyncMock(return_value={"content": "[1]"}))
    signal = SimpleNamespace(send_poll_vote=AsyncMock(return_value=True), _known_uuids=())
    jev = SimpleNamespace(choose=AsyncMock(return_value=choices))
    voter = PollVoter(llm_client=llm, signal_handler=signal, jev=jev)
    await voter.handle_poll({"sourceNumber": "alice", "sourceUuid": "uuid"}, {
        "pollCreate": {"question": "Choose a color", "options": ["red", "green", "blue"], "allowMultiple": multiple},
        "groupInfo": {"groupId": "g"}, "timestamp": 123,
    })
    assert llm.chat_messages.await_count == int(fallback)
    if expected is None:
        signal.send_poll_vote.assert_not_awaited()
    else:
        assert signal.send_poll_vote.await_args.kwargs["selected_answers"] == expected
        assert signal.send_poll_vote.await_args.kwargs["poll_author"] == "uuid"
    payload = jev.choose.await_args.kwargs
    if not multiple:
        assert "options" not in payload["state"]
        assert payload["questions"]["vote"]["criteria"] == {"0": "red", "1": "green", "2": "blue"}


async def test_sentiment_is_per_ticker_and_preserves_uncertain_votes():
    sources = [("Buy $NVDA, short $TSLA", 10, "paired trade"), ("$NVDA to the moon", 3, None)]
    tickers = build_tally(sources)
    original = [(t.symbol, t.mentions, t.cashtags, t.weight, t.samples[:]) for t in tickers]
    jev = SimpleNamespace(choose=AsyncMock(return_value={"0:NVDA": "bullish", "0:TSLA": "bearish"}))
    await classify_sentiment(tickers, sources, jev)
    stats = {t.symbol: t for t in tickers}
    assert (stats["NVDA"].bull, stats["NVDA"].bear) == (2, 0)
    assert (stats["TSLA"].bull, stats["TSLA"].bear) == (0, 1)
    assert [(t.symbol, t.mentions, t.cashtags, t.weight, t.samples) for t in tickers] == original
    payload = jev.choose.await_args.kwargs
    assert len(payload["state"]["posts"]) == 2
    assert all("Classify" in q["instructions"] for q in payload["questions"].values())


async def test_discovery_recovers_semantic_matches_without_exposing_blocked_tools():
    tools = [MCPTool(server_name="finance", name="quote", description="Current equity price", input_schema={}),
             MCPTool(server_name="private", name="secrets", description="Private portfolio", input_schema={})]
    jev = SimpleNamespace(choose=AsyncMock(return_value={"finance__quote": "direct"}))
    manager = SimpleNamespace(all_tools=lambda: tools, jev=jev)
    policy = ContextPolicy(id=1, kind="group", key="g", mcp_mode="allow_list", mcp_servers=["finance"])
    result = json.loads(await discover_mcp_tools(manager, policy, query="stock valuation"))
    assert [t["name"] for t in result["matches"]] == ["finance__quote"]
    payload = jev.choose.await_args.kwargs
    assert "private" not in json.dumps(payload)
    assert "input_schema" not in json.dumps(payload)
    jev.choose.return_value = {}
    fallback = json.loads(await discover_mcp_tools(manager, policy, query="equity"))
    assert fallback["returned"] == 1
    jev.choose.return_value = {"finance__quote": "irrelevant"}
    rejected = json.loads(await discover_mcp_tools(manager, policy, query="equity"))
    assert rejected["returned"] == 0
    jev.choose.reset_mock()
    await discover_mcp_tools(manager, policy, query="")
    jev.choose.assert_not_awaited()
