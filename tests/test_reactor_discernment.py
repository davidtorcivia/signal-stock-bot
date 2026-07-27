"""Reactor discernment gates.

The reactor used to react to ~half of every message it evaluated, because
the only brakes were three cheap pre-LLM gates and whatever the model felt
like. These tests pin the two structural changes that fixed it:

  1. The cheap gates (`bot_will_reply` / `min_length` / cooldowns) suppress
     the emoji_react TOOL rather than abandoning the whole LLM call. That
     matters because should_respond and note_memory ride on the same
     request with their own independent gating — an early return silently
     coupled emoji throttling to the natural-response feature, and since
     cooldowns are recorded only on a real reaction, every reaction muted
     that sender's spontaneous-reply path for a full cooldown window.

  2. Three post-LLM brakes (score threshold / no-repeat / rolling budget)
     that run AFTER the model has picked an emoji, so the model's judgement
     ranks candidates and the brakes ration how many actually land. All of
     them must run before `_record_cooldowns`: a suppressed pick is a
     non-event the user never saw and must not start a cooldown.
"""

from __future__ import annotations

import json
import time
from collections import deque
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots.models import Bot
from src.contexts.policy import ContextPolicy
from src.llm.reactor import EmojiReactor
from src.settings_store import SettingsStore


@pytest.fixture
def store(tmp_path):
    s = SettingsStore(str(tmp_path / "settings.db"))
    s.set("reactor_enabled", "1")
    s.set("reactor_min_length", "0")
    s.set("reactor_sender_cooldown", "0")
    s.set("reactor_group_cooldown", "0")
    # Post-LLM brakes off unless a test opts in, so each one is exercised
    # in isolation rather than tripping over the defaults.
    s.set("reactor_hourly_budget", "0")
    s.set("reactor_daily_budget", "0")
    s.set("reactor_repeat_window", "0")
    s.set("reactor_min_score", "0")
    return s


def _bot(bot_id: int = 1, slug: str = "artaud") -> Bot:
    return Bot(id=bot_id, slug=slug, display_name=slug.capitalize(), aliases=[slug])


def _policy(**kw) -> ContextPolicy:
    return ContextPolicy(
        id=42, kind="group", key="group:abc", label="Test", **kw,
    )


def _react_call(emoji: str = "🔥", score=None) -> dict:
    args: dict = {"emoji": emoji}
    if score is not None:
        args["score"] = score
    return {
        "function": {"name": "emoji_react", "arguments": json.dumps(args)},
    }


def _respond_call(reason: str = "asked a question") -> dict:
    return {
        "function": {
            "name": "should_respond",
            "arguments": json.dumps({"reason": reason}),
        },
    }


def _reactor(store, *, tool_calls=None, react_ok=True):
    """EmojiReactor wired to mocks. Returns (reactor, llm, signal)."""
    llm = MagicMock()
    llm.chat_messages = AsyncMock(
        return_value={"content": "", "tool_calls": list(tool_calls or [])}
    )
    signal = MagicMock()
    signal.send_reaction = AsyncMock(return_value=react_ok)
    reactor = EmojiReactor(
        settings_store=store, llm_client=llm, signal_handler=signal,
    )
    return reactor, llm, signal


async def _fire(reactor, **kw):
    defaults = dict(
        sender="+15551234567",
        message="a normal-length message worth evaluating here",
        group_id="group:abc",
        target_timestamp=1700000000,
        bot=_bot(),
    )
    defaults.update(kw)
    await reactor.maybe_react(**defaults)


def _tool_names(llm) -> list[str]:
    tools = llm.chat_messages.await_args.kwargs["tools"]
    return [t["function"]["name"] for t in tools]


# ── Gate 1: cheap gates suppress the tool, not the call ────────────────────


@pytest.mark.asyncio
async def test_bot_will_reply_skips_call_when_nothing_else_offered(store):
    """The common case: a !command or @mention with natural-response off.
    No tool would survive, so the call is abandoned outright."""
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call()])
    await _fire(reactor, bot_will_reply=True)
    llm.chat_messages.assert_not_called()
    signal.send_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_bot_will_reply_does_not_react(store):
    """Even if the model somehow returns an emoji_react it wasn't offered,
    a message the bot is already answering must not also get decorated."""
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call()])
    store.set("natural_response_enabled", "1")
    reactor.implicit_response_handler = AsyncMock()
    await _fire(
        reactor, bot_will_reply=True,
        policy=_policy(natural_response=True),
    )
    # note_memory is off and should_respond is suppressed by bot_will_reply,
    # so there is nothing to ask — no call, and certainly no reaction.
    signal.send_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_short_message_still_reaches_should_respond(store):
    """Regression: `min_length` used to return early, which throttled the
    natural-response path along with emoji. A too-short message must still
    be able to trigger a spontaneous reply."""
    store.set("reactor_min_length", "40")
    store.set("natural_response_enabled", "1")
    store.set("natural_response_cooldown", "0")
    reactor, llm, signal = _reactor(store, tool_calls=[_respond_call()])
    handler = AsyncMock()
    reactor.implicit_response_handler = handler

    await _fire(
        reactor, message="short one",
        policy=_policy(natural_response=True),
    )

    llm.chat_messages.assert_awaited_once()
    names = _tool_names(llm)
    assert "should_respond" in names
    assert "emoji_react" not in names
    handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_cooldown_still_reaches_should_respond(store):
    """Same regression via the cooldown gate: a recent reaction must not
    mute this sender's spontaneous replies."""
    store.set("reactor_sender_cooldown", "300")
    store.set("natural_response_enabled", "1")
    store.set("natural_response_cooldown", "0")
    reactor, llm, signal = _reactor(store, tool_calls=[_respond_call()])
    reactor.implicit_response_handler = AsyncMock()
    # Simulate "this sender was just reacted to".
    reactor._record_cooldowns("+15551234567", "group:abc", bot_id=1)

    await _fire(reactor, policy=_policy(natural_response=True))

    llm.chat_messages.assert_awaited_once()
    assert "emoji_react" not in _tool_names(llm)
    reactor.implicit_response_handler.assert_awaited_once()


@pytest.mark.asyncio
async def test_cooldown_alone_still_abandons_call(store):
    """With natural response off, a cooldown leaves no tools — the call is
    abandoned, preserving the original cost-saving behaviour."""
    store.set("reactor_sender_cooldown", "300")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call()])
    reactor._record_cooldowns("+15551234567", "group:abc", bot_id=1)
    await _fire(reactor)
    llm.chat_messages.assert_not_called()


@pytest.mark.asyncio
async def test_normal_message_offers_emoji_tool(store):
    """Sanity: with all gates clear, emoji_react is offered and lands."""
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("💀")])
    await _fire(reactor)
    assert "emoji_react" in _tool_names(llm)
    signal.send_reaction.assert_awaited_once()
    assert signal.send_reaction.await_args.kwargs["emoji"] == "💀"


# ── Gate 3a: rolling budget ────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_hourly_budget_blocks_reaction(store):
    store.set("reactor_hourly_budget", "2")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥")])
    now = time.time()
    reactor._recent[("group:abc", 1)] = deque([
        (now - 60, "A", "x", "😬", None),
        (now - 120, "A", "x", "💀", None),
    ])

    await _fire(reactor)
    signal.send_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_budget_does_not_start_a_cooldown(store):
    """A suppressed pick is invisible to users, so it must not mute this
    sender — otherwise an unseen non-event throttles the natural-response
    path for a full cooldown window."""
    store.set("reactor_hourly_budget", "1")
    store.set("reactor_sender_cooldown", "300")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥")])
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )

    await _fire(reactor)

    signal.send_reaction.assert_not_called()
    cfg = reactor._config(_bot())
    assert reactor._within_cooldown(
        "+15551234567", "group:abc", cfg, bot_id=1,
    ) is False


@pytest.mark.asyncio
async def test_budget_allows_reaction_under_cap(store):
    store.set("reactor_hourly_budget", "3")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥")])
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


def test_budget_ignores_entries_outside_the_window(store):
    """Yesterday's reactions must not count against today's hourly cap."""
    reactor, _, _ = _reactor(store)
    cfg = dict(hourly_budget=1, daily_budget=0)
    old = time.time() - 7200  # two hours ago
    reactor._recent[("group:abc", 1)] = deque([(old, "A", "x", "😬", None)])
    assert reactor._budget_exceeded("group:abc", cfg, bot_id=1) is None


def test_daily_budget_counts_across_hours(store):
    reactor, _, _ = _reactor(store)
    cfg = dict(hourly_budget=0, daily_budget=2)
    now = time.time()
    reactor._recent[("group:abc", 1)] = deque([
        (now - 7200, "A", "x", "😬", None),
        (now - 3700, "A", "x", "💀", None),
    ])
    assert reactor._budget_exceeded("group:abc", cfg, bot_id=1) == "2/2 today"


def test_budget_disabled_when_both_zero(store):
    reactor, _, _ = _reactor(store)
    cfg = dict(hourly_budget=0, daily_budget=0)
    now = time.time()
    reactor._recent[("group:abc", 1)] = deque([(now, "A", "x", "😬", None)] * 50)
    assert reactor._budget_exceeded("group:abc", cfg, bot_id=1) is None


def test_budget_is_scoped_per_bot(store):
    """Bot A spending its budget must not mute bot B in the same group."""
    reactor, _, _ = _reactor(store)
    cfg = dict(hourly_budget=1, daily_budget=0)
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )
    assert reactor._budget_exceeded("group:abc", cfg, bot_id=1) is not None
    assert reactor._budget_exceeded("group:abc", cfg, bot_id=2) is None


# ── Gate 3b: no-repeat window ──────────────────────────────────────────────


@pytest.mark.asyncio
async def test_repeat_emoji_blocked(store):
    store.set("reactor_repeat_window", "3")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("😬")])
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )
    await _fire(reactor)
    signal.send_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_different_emoji_allowed(store):
    store.set("reactor_repeat_window", "3")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥")])
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


def test_repeat_window_only_looks_back_n(store):
    """An emoji that has aged out of the window is available again."""
    reactor, _, _ = _reactor(store)
    cfg = dict(repeat_window=2)
    for emoji in ("😬", "💀", "🔥"):
        reactor._record_recent(
            group_id="group:abc", sender_label="A", target_text="x",
            emoji=emoji, bot_id=1,
        )
    # Last two are 💀 and 🔥, so 😬 has fallen out of the window.
    assert reactor._is_repeat("group:abc", "😬", cfg, bot_id=1) is False
    assert reactor._is_repeat("group:abc", "🔥", cfg, bot_id=1) is True


def test_repeat_normalizes_variation_selector(store):
    """❤️ (with U+FE0F) and ❤ (without) are one emoji, not two — Signal and
    the models disagree about the selector."""
    reactor, _, _ = _reactor(store)
    cfg = dict(repeat_window=3)
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="❤️", bot_id=1,
    )
    assert reactor._is_repeat("group:abc", "❤", cfg, bot_id=1) is True


def test_repeat_window_zero_disables(store):
    reactor, _, _ = _reactor(store)
    cfg = dict(repeat_window=0)
    reactor._record_recent(
        group_id="group:abc", sender_label="A", target_text="x",
        emoji="😬", bot_id=1,
    )
    assert reactor._is_repeat("group:abc", "😬", cfg, bot_id=1) is False


# ── Gate 3c: self-reported score ───────────────────────────────────────────


@pytest.mark.asyncio
async def test_low_score_dropped_when_threshold_set(store):
    store.set("reactor_min_score", "7")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥", score=4)])
    await _fire(reactor)
    signal.send_reaction.assert_not_called()


@pytest.mark.asyncio
async def test_high_score_passes_threshold(store):
    store.set("reactor_min_score", "7")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥", score=8)])
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_score_is_log_only_at_zero(store):
    """The calibration phase: scores are recorded, nothing is enforced."""
    store.set("reactor_min_score", "0")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥", score=1)])
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_missing_score_is_not_treated_as_low(store):
    """Models drop required fields. An unscored pick must pass rather than
    be silently suppressed by a threshold it never reported against."""
    store.set("reactor_min_score", "7")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥")])
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_unparseable_score_is_not_treated_as_low(store):
    import json
    store.set("reactor_min_score", "7")
    bad = {
        "function": {
            "name": "emoji_react",
            "arguments": json.dumps({"emoji": "🔥", "score": "high"}),
        },
    }
    reactor, llm, signal = _reactor(store, tool_calls=[bad])
    await _fire(reactor)
    signal.send_reaction.assert_awaited_once()


@pytest.mark.asyncio
async def test_score_threshold_does_not_start_a_cooldown(store):
    store.set("reactor_min_score", "7")
    store.set("reactor_sender_cooldown", "300")
    reactor, llm, signal = _reactor(store, tool_calls=[_react_call("🔥", score=2)])
    await _fire(reactor)
    signal.send_reaction.assert_not_called()
    cfg = reactor._config(_bot())
    assert reactor._within_cooldown(
        "+15551234567", "group:abc", cfg, bot_id=1,
    ) is False


# ── Tool schema ────────────────────────────────────────────────────────────


def test_react_tool_requires_score():
    from src.llm.reactor import REACT_TOOL
    params = REACT_TOOL["function"]["parameters"]
    assert set(params["required"]) == {"emoji", "score"}
    assert params["properties"]["score"]["type"] == "integer"


def test_default_prompt_states_the_score_rubric():
    """A required score field with no rubric in the prompt produces a
    distribution bunched at 7-8, which defeats the calibration phase."""
    from src.llm.reactor import DEFAULT_REACTOR_PROMPT
    assert "score" in DEFAULT_REACTOR_PROMPT
    assert "9-10" in DEFAULT_REACTOR_PROMPT
