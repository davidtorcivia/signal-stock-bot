"""Telling a bot to shut up: one response, then an escalating timeout on
its unprompted reactions and replies in that group."""

from __future__ import annotations

import time
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots.models import Bot
from src.llm import reactor as reactor_mod
from src.llm.reactor import EmojiReactor, is_hush
from src.settings_store import SettingsStore


def _reactor(tmp_path):
    store = SettingsStore(str(tmp_path / "settings.db"))
    store.set("reactor_enabled", "true")
    signal = MagicMock()
    signal.send_reaction = AsyncMock(return_value=True)
    llm = MagicMock()
    llm.chat_with_tools = AsyncMock()
    return EmojiReactor(store, llm, signal), signal, llm


def test_is_hush():
    for t in ["shut up sigil", "Sigil, stop talking", "stfu", "pause",
              "artaud time out", "ok enough", "go away bot"]:
        assert is_hush(t), t
    for t in ["what happens when trading is paused for the day",
              "why did the stock stop going up", "can you be quiet?",
              "TSLA looks quiet today, nothing moving at all"]:
        assert not is_hush(t), t


def test_hush_escalates_and_resets(tmp_path, monkeypatch):
    r, _, _ = _reactor(tmp_path)
    base = reactor_mod.HUSH_BASE_SECONDS
    assert [r.hush("g", 1) for _ in range(3)] == [base, base * 2, base * 4]
    assert r.is_hushed("g", 1)
    assert not r.is_hushed("g", 2)
    later = time.time() + reactor_mod.HUSH_RESET_SECONDS + 10
    monkeypatch.setattr(reactor_mod.time, "time", lambda: later)
    assert r.hush("g", 1) == base


@pytest.mark.asyncio
async def test_addressed_hush_mutes_without_reacting(tmp_path):
    r, signal, llm = _reactor(tmp_path)
    bot = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"])
    await r.maybe_react(sender="+1", message="sigil shut up", group_id="g",
                        target_timestamp=1, bot_will_reply=True, bot=bot)
    assert r.is_hushed("g", 1)
    signal.send_reaction.assert_not_called()
    # While muted, nothing reaches the model.
    await r.maybe_react(sender="+1", message="a long enough message to react to",
                        group_id="g", target_timestamp=2, bot=bot)
    llm.chat_with_tools.assert_not_called()


@pytest.mark.asyncio
async def test_unaddressed_hush_only_hits_bot_that_just_chimed_in(tmp_path):
    r, signal, _ = _reactor(tmp_path)
    sigil = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"])
    artaud = Bot(id=2, slug="artaud", display_name="Artaud", aliases=["artaud"])
    await r.maybe_react(sender="+1", message="shut up", group_id="g",
                        target_timestamp=1, bot=sigil, candidate_bots=[sigil, artaud])
    assert not r.is_hushed("g", 1)  # nobody chimed in; aimed at a human
    r.mark_implicit_response("g", bot_id=2)
    await r.maybe_react(sender="+1", message="shut up", group_id="g",
                        target_timestamp=2, bot=sigil, candidate_bots=[sigil, artaud])
    assert r.is_hushed("g", 2) and not r.is_hushed("g", 1)
    assert signal.send_reaction.await_args.kwargs["emoji"] == "🤐"
