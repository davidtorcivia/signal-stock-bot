"""Per-bot reactor settings: bot_llm_settings(bot.id, 'reactor', key)
overrides the global reactor_<key> for cooldowns, prompts, model, etc.;
the actual LLM call routes through factory.get_reactor(bot.id) so each
bot can use its own api_key/base_url.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots.models import Bot
from src.llm.factory import LLMClientFactory
from src.llm.reactor import DEFAULT_REACTOR_PROMPT, EmojiReactor
from src.settings_store import SettingsStore


@pytest.fixture
def store(tmp_path):
    return SettingsStore(str(tmp_path / "settings.db"))


def _ctx_bot(bot_id: int = 1, slug: str = "artaud") -> Bot:
    return Bot(
        id=bot_id,
        slug=slug,
        display_name=slug.capitalize(),
        aliases=[slug],
    )


# ── Factory.get_reactor ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_factory_caches_reactor_client_per_bot(store):
    factory = LLMClientFactory(store)
    c1 = factory.get_reactor(7)
    c2 = factory.get_reactor(7)
    c3 = factory.get_reactor(8)
    assert c1 is c2
    assert c1 is not c3
    assert c1.role == "reactor"
    assert c1.bot_id == 7


@pytest.mark.asyncio
async def test_factory_forget_drops_reactor_client(store):
    factory = LLMClientFactory(store)
    c1 = factory.get_reactor(7)
    factory.forget(7)
    c2 = factory.get_reactor(7)
    assert c1 is not c2


# ── Reactor._config(bot) ───────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_config_falls_back_to_global_when_no_bot_override(store):
    store.set("reactor_enabled", "1")
    store.set("reactor_sender_cooldown", "45")
    store.set("reactor_model", "global-model")
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    cfg = reactor._config(_ctx_bot())
    assert cfg["enabled"] is True
    assert cfg["sender_cooldown"] == 45
    assert cfg["model"] == "global-model"


@pytest.mark.asyncio
async def test_config_bot_override_wins_over_global(store):
    store.set("reactor_sender_cooldown", "30")
    store.set("reactor_model", "global-model")
    store.set("reactor_system_prompt", "global prompt")
    store.set_bot(7, "reactor", "sender_cooldown", "120")
    store.set_bot(7, "reactor", "model", "bot-specific-model")
    store.set_bot(7, "reactor", "system_prompt", "be very quiet")

    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    cfg = reactor._config(_ctx_bot(bot_id=7))
    assert cfg["sender_cooldown"] == 120
    assert cfg["model"] == "bot-specific-model"
    assert cfg["system_prompt"] == "be very quiet"


@pytest.mark.asyncio
async def test_config_other_bots_unaffected(store):
    store.set("reactor_sender_cooldown", "30")
    store.set_bot(7, "reactor", "sender_cooldown", "120")
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    cfg_other = reactor._config(_ctx_bot(bot_id=99))
    # Bot 99 has no override, picks up the global.
    assert cfg_other["sender_cooldown"] == 30


@pytest.mark.asyncio
async def test_config_no_bot_uses_global_only(store):
    store.set("reactor_enabled", "1")
    store.set_bot(7, "reactor", "enabled", "0")
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    # Passing None bot → bot_llm_settings ignored, only the global is consulted.
    cfg = reactor._config(None)
    assert cfg["enabled"] is True


@pytest.mark.asyncio
async def test_config_natural_response_per_bot(store):
    # Default: nothing set anywhere → off.
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    assert reactor._config(_ctx_bot(bot_id=99))["natural_response_enabled"] is False

    # Bot 7 opts in with a custom cooldown — global stays at default-off,
    # so other bots remain off, only bot 7 sees it on.
    store.set_bot(7, "reactor", "natural_response_enabled", "1")
    store.set_bot(7, "reactor", "natural_response_cooldown", "600")
    cfg_bot = reactor._config(_ctx_bot(bot_id=7))
    cfg_other = reactor._config(_ctx_bot(bot_id=99))
    assert cfg_bot["natural_response_enabled"] is True
    assert cfg_bot["natural_response_cooldown"] == 600
    assert cfg_other["natural_response_enabled"] is False
    # Other bot still sees the default cooldown — it had no override.
    assert cfg_other["natural_response_cooldown"] == 300


@pytest.mark.asyncio
async def test_config_falls_back_to_default_prompt(store):
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    cfg = reactor._config(_ctx_bot())
    assert cfg["system_prompt"] == DEFAULT_REACTOR_PROMPT


@pytest.mark.asyncio
async def test_string_zero_is_treated_as_false(store):
    """Admin types "0" in the per-bot form to disable a bool — the
    coercer must NOT see `bool("0") == True` (Python default). Without
    this fix the form input would silently invert."""
    store.set_bot(7, "reactor", "enabled", "0")
    reactor = EmojiReactor(settings_store=store, llm_client=None, signal_handler=None)
    cfg = reactor._config(_ctx_bot(bot_id=7))
    assert cfg["enabled"] is False

    # And "false" / "no" / "off" all read False too.
    for val in ("false", "False", "no", "off", ""):
        store.set_bot(7, "reactor", "enabled", val)
        cfg = reactor._config(_ctx_bot(bot_id=7))
        assert cfg["enabled"] is False, f"{val!r} should coerce to False"

    for val in ("1", "true", "True", "yes", "on"):
        store.set_bot(7, "reactor", "enabled", val)
        cfg = reactor._config(_ctx_bot(bot_id=7))
        assert cfg["enabled"] is True, f"{val!r} should coerce to True"


@pytest.mark.asyncio
async def test_is_enabled_uses_bot_override_and_context_gate(store):
    """Writer prompt gating and reactor execution share the active bot's
    override, then apply the chat policy gate."""
    store.set("reactor_enabled", False)
    store.set_bot(7, "reactor", "enabled", True)
    reactor = EmojiReactor(
        settings_store=store, llm_client=None, signal_handler=None,
    )
    bot = _ctx_bot(bot_id=7)
    assert reactor.is_enabled(bot, policy=None) is True

    policy = MagicMock(reactor_enabled=False)
    assert reactor.is_enabled(bot, policy=policy) is False
    assert reactor.is_enabled(_ctx_bot(bot_id=99), policy=None) is False


# ── Reactor._llm_for ───────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_llm_for_picks_factory_reactor_client(store):
    factory = LLMClientFactory(store)
    default_llm = MagicMock()
    reactor = EmojiReactor(
        settings_store=store,
        llm_client=default_llm,
        signal_handler=None,
        llm_factory=factory,
    )
    bot = _ctx_bot(bot_id=7)
    c = reactor._llm_for(bot)
    assert c.bot_id == 7
    assert c.role == "reactor"
    # Cached: a second call returns the same instance.
    assert reactor._llm_for(bot) is c


@pytest.mark.asyncio
async def test_llm_for_falls_back_when_no_factory(store):
    default_llm = MagicMock()
    reactor = EmojiReactor(
        settings_store=store,
        llm_client=default_llm,
        signal_handler=None,
    )
    assert reactor._llm_for(_ctx_bot()) is default_llm


@pytest.mark.asyncio
async def test_llm_for_falls_back_when_bot_is_none(store):
    factory = LLMClientFactory(store)
    default_llm = MagicMock()
    reactor = EmojiReactor(
        settings_store=store,
        llm_client=default_llm,
        signal_handler=None,
        llm_factory=factory,
    )
    assert reactor._llm_for(None) is default_llm


# ── End-to-end: maybe_react routes through per-bot client ──────────────────


@pytest.mark.asyncio
async def test_maybe_react_calls_per_bot_client(store):
    """Per-bot reactor client is what makes the chat_messages call."""
    store.set("reactor_enabled", "1")
    store.set("reactor_min_length", "0")
    store.set("reactor_sender_cooldown", "0")
    store.set("reactor_group_cooldown", "0")

    factory = LLMClientFactory(store)
    # Replace get_reactor with a sentinel-returning mock.
    sentinel_client = MagicMock()
    sentinel_client.chat_messages = AsyncMock(
        return_value={"content": "", "tool_calls": []}
    )
    factory.get_reactor = MagicMock(return_value=sentinel_client)

    default_llm = MagicMock()
    default_llm.chat_messages = AsyncMock(
        return_value={"content": "", "tool_calls": []}
    )

    reactor = EmojiReactor(
        settings_store=store,
        llm_client=default_llm,
        signal_handler=None,
        llm_factory=factory,
    )

    bot = _ctx_bot(bot_id=7)
    await reactor.maybe_react(
        sender="+15551234567",
        message="this is a normal-length test message",
        group_id="group:abc",
        target_timestamp=1700000000,
        bot=bot,
    )

    # Per-bot reactor client was used; default was NOT.
    factory.get_reactor.assert_called_with(7)
    sentinel_client.chat_messages.assert_awaited_once()
    default_llm.chat_messages.assert_not_called()
