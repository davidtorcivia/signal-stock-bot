"""Tests for the multi-bot subsystem (PR1-4 + audit fixes).

Covers:
- BotRegistry seeding, upsert with default-flag invariants, delete guards
- resolve_setting bot -> global -> default chain (incl. deep_think
  fallback to writer)
- _resolve_addressed_bot context-gated alias matching
- _run_research_handoff format-fallback + per-bot client routing
- LLMClientFactory cache identity + forget()
"""

from __future__ import annotations

import asyncio
import tempfile
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots import (
    Bot,
    BotRegistry,
    SIGIL_SLUG,
    resolve_bool,
    resolve_float,
    resolve_int,
    resolve_setting,
)
from src.commands.base import CommandContext
from src.contexts.policy import ContextPolicy
from src.llm.factory import LLMClientFactory
from src.settings_store import SettingsStore
from src.signal.handler import SignalConfig, SignalHandler


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmpdb(tmp_path: Path) -> str:
    return str(tmp_path / "bots_test.db")


@pytest.fixture
def store(tmpdb: str) -> SettingsStore:
    s = SettingsStore(tmpdb)
    s.set("llm_enabled", True)
    s.set("llm_base_url", "https://api.openai.com/v1")
    s.set("llm_api_key", "sk-global")
    s.set("llm_model", "gpt-4")
    s.set("llm_temperature", 0.7)
    s.set("llm_max_tokens", 1000)
    return s


@pytest.fixture
def registry(tmpdb: str) -> BotRegistry:
    r = BotRegistry(tmpdb)
    r.warm_sync(default_display_name="Sigil")
    return r


# ──────────────────────────────────────────────────────────────────
# BotRegistry
# ──────────────────────────────────────────────────────────────────


def test_warm_sync_seeds_sigil_with_correct_defaults(registry: BotRegistry):
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)
    assert sigil is not None
    assert sigil.id == 1
    assert sigil.display_name == "Sigil"
    assert sigil.aliases == ["Sigil"]
    assert sigil.default_for_dm is True
    assert sigil.default_for_group is True
    assert sigil.enabled is True
    assert sigil.deep_think_mode == "replace"


def test_warm_sync_idempotent(registry: BotRegistry, tmpdb: str):
    """Re-calling warm_sync on an already-seeded DB doesn't create duplicates."""
    BotRegistry(tmpdb).warm_sync(default_display_name="ShouldBeIgnored")
    bots = registry.list_sync()
    assert len(bots) == 1
    assert bots[0].display_name == "Sigil"


@pytest.mark.asyncio
async def test_upsert_clears_default_flags_on_other_bots(
    registry: BotRegistry,
):
    """When a new bot claims default_for_dm, every OTHER bot's flag clears."""
    new_id = await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud",
        aliases=["Artaud"],
        default_for_dm=True, default_for_group=True,
    ))
    bots = {b.slug: b for b in await registry.list()}
    assert bots["artaud"].default_for_dm is True
    assert bots["artaud"].default_for_group is True
    assert bots["sigil"].default_for_dm is False
    assert bots["sigil"].default_for_group is False
    assert new_id == bots["artaud"].id


@pytest.mark.asyncio
async def test_upsert_refuses_to_leave_kind_with_no_default(
    registry: BotRegistry,
):
    """Toggling Sigil's defaults off when no other bot has them set
    must be rejected — otherwise inbound messages would have nobody
    to answer them."""
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)
    sigil.default_for_dm = False
    sigil.default_for_group = False
    with pytest.raises(ValueError, match="default bot"):
        await registry.upsert(sigil)
    # Original flags preserved (transaction rolled back).
    refreshed = await registry.get_by_slug(SIGIL_SLUG)
    assert refreshed.default_for_dm is True
    assert refreshed.default_for_group is True


@pytest.mark.asyncio
async def test_delete_refuses_sigil_seed_anchor(registry: BotRegistry):
    """The sigil row is the migration anchor — deletion would orphan
    backfilled bot_id columns. Deletion should refuse."""
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)
    ok = await registry.delete(sigil.id)
    assert ok is False
    assert registry.get_by_slug_sync(SIGIL_SLUG) is not None


@pytest.mark.asyncio
async def test_delete_refuses_when_contexts_reference_bot(
    registry: BotRegistry, tmpdb: str,
):
    """A bot pinned via contexts.default_bot_id can't be deleted —
    callers must reassign first."""
    from src.contexts import ContextRegistry
    from src.contexts.policy import ContextPolicy

    ctx_reg = ContextRegistry(tmpdb)
    artaud_id = await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud", aliases=["Artaud"],
    ))
    await ctx_reg.upsert(ContextPolicy(
        id=None, kind="group", key="!REF_TEST", label="ref",
        default_bot_id=artaud_id,
    ))
    ok = await registry.delete(artaud_id)
    assert ok is False


@pytest.mark.asyncio
async def test_match_alias_is_whole_word_case_insensitive(
    registry: BotRegistry,
):
    artaud_id = await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud",
        aliases=["Artaud", "Art"],
    ))
    artaud = registry.get_sync(artaud_id)
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)

    assert registry.match_alias("hey Artaud what's AAPL?").slug == "artaud"
    assert registry.match_alias("ARTAUD!").slug == "artaud"
    # Whole-word: "art" matches the alias, "artist" does not
    assert registry.match_alias("yo art, what's the call?").slug == "artaud"
    assert registry.match_alias("i love that artist") is None
    # Restrict candidates: with only sigil eligible, "Artaud" doesn't match
    assert registry.match_alias("hey Artaud", candidates=[sigil]) is None


def test_default_for_kind_falls_back_when_no_default_flagged(
    registry: BotRegistry,
):
    """If the cache somehow holds rows with no default_for_X flag,
    default_for_kind_sync falls back to first enabled bot rather
    than returning None."""
    # Manually mutate cache to simulate the corner case
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)
    sigil.default_for_dm = False
    sigil.default_for_group = False
    bot = registry.default_for_kind_sync("dm")
    assert bot is not None  # Falls back to first enabled
    assert bot.slug == SIGIL_SLUG


# ──────────────────────────────────────────────────────────────────
# resolve_setting
# ──────────────────────────────────────────────────────────────────


def test_resolve_chain_bot_then_global_then_default(store: SettingsStore):
    # No bot override, no llm_unset_key set, default applies
    assert resolve_int(
        store, 1, "writer", "unset_key",
        global_keys=["llm_unset_key"], default=99,
    ) == 99
    # Global is set: returns global
    store.set("llm_max_tokens", 5000)
    assert resolve_int(
        store, 1, "writer", "max_tokens",
        global_keys=["llm_max_tokens"], default=99,
    ) == 5000
    # Bot override wins
    store.set_bot(1, "writer", "max_tokens", 250)
    assert resolve_int(
        store, 1, "writer", "max_tokens",
        global_keys=["llm_max_tokens"], default=99,
    ) == 250


def test_resolve_deep_think_falls_back_to_writer(store: SettingsStore):
    """deep_think uses [deep_think_X, llm_X] chain — when both bot
    and deep_think_global are unset, the writer global covers it."""
    store.set("llm_temperature", 0.7)
    # Only llm_temperature set; deep_think falls through to it
    assert resolve_float(
        store, 1, "deep_think", "temperature",
        global_keys=["deep_think_temperature", "llm_temperature"],
        default=0.0,
    ) == 0.7
    # deep_think_global takes precedence over llm_global
    store.set("deep_think_temperature", 0.3)
    assert resolve_float(
        store, 1, "deep_think", "temperature",
        global_keys=["deep_think_temperature", "llm_temperature"],
        default=0.0,
    ) == 0.3
    # Bot deep_think override wins over both
    store.set_bot(1, "deep_think", "temperature", 1.1)
    assert resolve_float(
        store, 1, "deep_think", "temperature",
        global_keys=["deep_think_temperature", "llm_temperature"],
        default=0.0,
    ) == 1.1


def test_resolve_with_bot_id_none_skips_bot_lookup(store: SettingsStore):
    """Legacy callers (no bot_id) get global-only behavior — same as
    pre-PR2 reads."""
    store.set("llm_temperature", 0.7)
    store.set_bot(1, "writer", "temperature", 0.9)  # would mask if bot_id were 1
    assert resolve_float(
        store, None, "writer", "temperature",
        global_keys=["llm_temperature"], default=0.0,
    ) == 0.7


# ──────────────────────────────────────────────────────────────────
# _resolve_addressed_bot (mention routing)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_alias_routing_is_context_gated(
    registry: BotRegistry,
):
    """Sigil-pinned context never reacts to 'Artaud'; Artaud-pinned
    never reacts to 'Sigil'. This is the core "not co-tenant"
    invariant from the user's plan."""
    artaud_id = await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud", aliases=["Artaud"],
    ))
    sigil = registry.get_by_slug_sync(SIGIL_SLUG)

    pol_sigil = ContextPolicy(
        id=1, kind="group", key="!SIGIL_GROUP", default_bot_id=sigil.id,
    )
    pol_artaud = ContextPolicy(
        id=2, kind="group", key="!ARTAUD_GROUP", default_bot_id=artaud_id,
    )

    from src.commands.dispatcher import CommandDispatcher
    disp = CommandDispatcher(bot_registry=registry)
    cfg = SignalConfig(api_url="http://x", phone_number="+15551111111")
    h = SignalHandler(cfg, disp)

    # Artaud's name in Artaud's context: matches
    bot, mentioned = await h._resolve_addressed_bot(
        {"message": "hey Artaud help"}, "!ARTAUD_GROUP", pol_artaud,
    )
    assert mentioned is True
    assert bot.slug == "artaud"

    # Sigil's name in Artaud's context: NO match
    bot, mentioned = await h._resolve_addressed_bot(
        {"message": "hey Sigil help"}, "!ARTAUD_GROUP", pol_artaud,
    )
    assert mentioned is False
    assert bot is None

    # Sigil's name in Sigil's context: matches
    bot, mentioned = await h._resolve_addressed_bot(
        {"message": "Sigil?"}, "!SIGIL_GROUP", pol_sigil,
    )
    assert mentioned is True
    assert bot.slug == "sigil"


@pytest.mark.asyncio
async def test_alias_substring_does_not_trigger(registry: BotRegistry):
    """'Art' as a whole word fires; 'artist' does not."""
    artaud_id = await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud", aliases=["Art"],
    ))
    pol = ContextPolicy(
        id=1, kind="group", key="!G", default_bot_id=artaud_id,
    )
    from src.commands.dispatcher import CommandDispatcher
    disp = CommandDispatcher(bot_registry=registry)
    h = SignalHandler(
        SignalConfig(api_url="http://x", phone_number="+1"), disp,
    )
    _, mentioned = await h._resolve_addressed_bot(
        {"message": "i love that artist"}, "!G", pol,
    )
    assert mentioned is False


# ──────────────────────────────────────────────────────────────────
# LLMClientFactory
# ──────────────────────────────────────────────────────────────────


def test_factory_caches_clients_per_bot(store: SettingsStore):
    f = LLMClientFactory(store)
    a1 = f.get_writer(1)
    a2 = f.get_writer(1)
    b = f.get_writer(2)
    assert a1 is a2  # same bot, same instance (session reuse)
    assert a1 is not b


def test_factory_resolves_per_bot_overrides(store: SettingsStore):
    f = LLMClientFactory(store)
    store.set_bot(1, "writer", "model", "model-A")
    store.set_bot(2, "writer", "model", "model-B")
    assert f.get_writer(1)._config()["model"] == "model-A"
    assert f.get_writer(2)._config()["model"] == "model-B"
    # bot_id=None reads globals
    assert f.get_writer(None)._config()["model"] == "gpt-4"


def test_factory_forget_drops_cached_client(store: SettingsStore):
    f = LLMClientFactory(store)
    a1 = f.get_writer(1)
    f.forget(1)
    a2 = f.get_writer(1)
    assert a1 is not a2


def test_factory_attach_bot_tools_propagates(store: SettingsStore):
    """Late-binding bot_tools should reach already-cached deep_think clients."""
    f = LLMClientFactory(store)
    dt = f.get_deep_think(1)
    assert dt.bot_tools is None
    sentinel = object()
    f.attach_bot_tools(sentinel)
    assert dt.bot_tools is sentinel


# ──────────────────────────────────────────────────────────────────
# _run_research_handoff (PR4 deep_think → writer flow)
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_research_handoff_substitutes_notes_placeholder():
    """Handoff template with {notes} placeholder gets the real notes."""
    from src.commands.ask_command import AskCommand

    ask = AskCommand.__new__(AskCommand)
    ask.signal_handler = None
    ask.llm_factory = None
    captured = {}

    async def chat(messages, tools=None):
        captured["messages"] = messages
        captured["tools"] = tools
        return {"role": "assistant", "content": "composed reply"}

    ask.llm = MagicMock()
    ask.llm.chat_messages = AsyncMock(side_effect=chat)
    ask.deep_think = MagicMock()
    ask.deep_think.think = AsyncMock(return_value="RESEARCH FINDINGS")

    bot = Bot(
        id=42, slug="artaud", display_name="Artaud", aliases=["Artaud"],
        deep_think_mode="research",
        deep_think_handoff_prompt="Notes follow:\n{notes}\nCompose.",
    )
    ctx = CommandContext(
        sender="+1", group_id="!G", raw_message="!ask",
        command="ask", args=[], policy=None, bot=bot,
    )
    answer = await ask._run_research_handoff(
        ctx=ctx, question="?", research_input="user said hi",
        messages=[{"role": "system", "content": "base"}],
        attachments=[], user_hash="abc",
    )
    assert answer == "composed reply"
    sys_msg = captured["messages"][0]["content"]
    assert "RESEARCH FINDINGS" in sys_msg
    assert "Compose." in sys_msg
    assert captured["tools"] is None  # no tools in research mode


@pytest.mark.asyncio
async def test_research_handoff_handles_malformed_format_template():
    """Stray '{' in the admin-supplied handoff template raises ValueError
    from str.format. The handler must catch it (audit fix H2) — without
    that catch, the whole !ask in chat would crash with a ValueError."""
    from src.commands.ask_command import AskCommand

    ask = AskCommand.__new__(AskCommand)
    ask.signal_handler = None
    ask.llm_factory = None
    ask.llm = MagicMock()
    ask.llm.chat_messages = AsyncMock(
        return_value={"role": "assistant", "content": "ok"}
    )
    ask.deep_think = MagicMock()
    ask.deep_think.think = AsyncMock(return_value="notes")

    # Stray '{' — would crash str.format with ValueError
    bot = Bot(
        id=42, slug="x", display_name="X", aliases=["X"],
        deep_think_mode="research",
        deep_think_handoff_prompt="prefix { stray brace",
    )
    ctx = CommandContext(
        sender="+1", group_id=None, raw_message="!ask",
        command="ask", args=[], policy=None, bot=bot,
    )
    # Should NOT raise — fallback is `<template>\n\n<notes>`
    answer = await ask._run_research_handoff(
        ctx=ctx, question="?", research_input="x",
        messages=[{"role": "system", "content": "base"}],
        attachments=[], user_hash="",
    )
    assert answer == "ok"


@pytest.mark.asyncio
async def test_research_handoff_does_not_mutate_callers_messages():
    """L5: the handler builds a research-augmented system message in
    place; the caller's `messages` list reuse must not see the
    mutation (defensive copy at function entry)."""
    from src.commands.ask_command import AskCommand

    ask = AskCommand.__new__(AskCommand)
    ask.signal_handler = None
    ask.llm_factory = None
    ask.llm = MagicMock()
    ask.llm.chat_messages = AsyncMock(
        return_value={"role": "assistant", "content": "ok"}
    )
    ask.deep_think = MagicMock()
    ask.deep_think.think = AsyncMock(return_value="LEAKED NOTES")

    bot = Bot(id=1, slug="x", display_name="X", aliases=["X"],
              deep_think_mode="research")
    ctx = CommandContext(
        sender="+1", group_id=None, raw_message="!ask",
        command="ask", args=[], policy=None, bot=bot,
    )
    original_messages = [{"role": "system", "content": "base"}]
    await ask._run_research_handoff(
        ctx=ctx, question="?", research_input="x",
        messages=original_messages,
        attachments=[], user_hash="",
    )
    # Caller's list and its first dict must be untouched.
    assert original_messages == [{"role": "system", "content": "base"}]


# ──────────────────────────────────────────────────────────────────
# Backfill idempotency
# ──────────────────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_backfill_is_idempotent(registry: BotRegistry, tmpdb: str):
    """Re-running backfill on a populated DB updates 0 rows."""
    import aiosqlite

    # Create a consumer table the registry knows about
    async with aiosqlite.connect(tmpdb) as db:
        await db.execute(
            "CREATE TABLE conversation_turns "
            "(id INTEGER PRIMARY KEY, content TEXT, bot_id INTEGER)"
        )
        await db.execute("INSERT INTO conversation_turns (content) VALUES ('hi')")
        await db.commit()

    # First pass: stamps the row
    await registry.backfill_consumer_tables()
    async with aiosqlite.connect(tmpdb) as db:
        cur = await db.execute("SELECT bot_id FROM conversation_turns")
        rows = await cur.fetchall()
    assert all(r[0] == 1 for r in rows)

    # Second pass: 0 NULL rows → UPDATE is a no-op
    await registry.backfill_consumer_tables()
    # No exceptions, same state
    async with aiosqlite.connect(tmpdb) as db:
        cur = await db.execute("SELECT bot_id FROM conversation_turns")
        rows = await cur.fetchall()
    assert all(r[0] == 1 for r in rows)
