"""Multi-phone routing: SignalHandlerPool + per-handler filter.

The pool is the wiring that lets two bots share one signal-cli-rest-api
container while each bot sends from its own number. These tests cover
the routing rules without spinning up a real signal-cli or hitting
sockets — pool construction, handler resolution by bot, and the
per-handler dispatch filter that prevents two phones from
double-answering the same envelope.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots.models import Bot
from src.signal.pool import SignalHandlerPool


@dataclass
class _FakeBotRegistry:
    bots: list[Bot] = field(default_factory=list)

    def list_sync(self) -> list[Bot]:
        return list(self.bots)

    def get_sync(self, bot_id: int) -> Optional[Bot]:
        for b in self.bots:
            if b.id == bot_id:
                return b
        return None

    def default_for_kind_sync(self, kind: str) -> Optional[Bot]:
        flag = "default_for_dm" if kind == "dm" else "default_for_group"
        for b in self.bots:
            if b.enabled and getattr(b, flag):
                return b
        return self.bots[0] if self.bots else None


def _bot(
    bot_id: int,
    slug: str,
    *,
    signal_phone: Optional[str] = None,
    default_group: bool = False,
    default_dm: bool = False,
) -> Bot:
    return Bot(
        id=bot_id,
        slug=slug,
        display_name=slug.capitalize(),
        aliases=[slug],
        signal_phone=signal_phone,
        default_for_dm=default_dm,
        default_for_group=default_group,
    )


def _make_pool(bots: list[Bot]) -> SignalHandlerPool:
    dispatcher = MagicMock()
    dispatcher.bot_registry = _FakeBotRegistry(bots)
    pool = SignalHandlerPool(
        default_api_url="http://signal-api:8080",
        default_phone="+15550000001",
        dispatcher=dispatcher,
        bot_registry=dispatcher.bot_registry,
    )
    pool.build()
    return pool


class TestPoolBuild:
    def test_single_bot_install_has_one_handler_on_default_phone(self):
        # No signal_phone override → bot rides the default phone.
        bots = [_bot(1, "sigil", default_group=True, default_dm=True)]
        pool = _make_pool(bots)
        assert pool.phones() == ["+15550000001"]
        h = pool.default()
        assert h.config.phone_number == "+15550000001"
        assert h.is_default_phone is True
        # Sigil's id ends up in served_bot_ids of the default handler.
        assert h.served_bot_ids == {1}

    def test_two_bots_with_overrides_get_two_handlers(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        pool = _make_pool(bots)
        assert set(pool.phones()) == {"+15550000001", "+15550000002"}
        artaud_h = pool.for_phone("+15550000002")
        assert artaud_h is not None
        assert artaud_h.served_bot_ids == {2}
        assert artaud_h.is_default_phone is False
        # Default handler keeps Sigil.
        sigil_h = pool.for_phone("+15550000001")
        assert sigil_h is not None
        assert sigil_h.served_bot_ids == {1}

    def test_for_bot_routes_by_signal_phone(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        pool = _make_pool(bots)
        artaud = bots[1]
        sigil = bots[0]
        assert pool.for_bot(artaud).config.phone_number == "+15550000002"
        assert pool.for_bot(sigil).config.phone_number == "+15550000001"
        # Unknown / None bot → default handler.
        assert pool.for_bot(None).config.phone_number == "+15550000001"

    def test_for_bot_falls_back_to_default_when_phone_unregistered(self):
        # Admin pinned a bot to a phone the pool hasn't built a handler for.
        # The pool should return the default handler rather than crash.
        bots = [_bot(1, "sigil", default_group=True, default_dm=True)]
        pool = _make_pool(bots)
        orphan = _bot(99, "ghost", signal_phone="+15559999999")
        h = pool.for_bot(orphan)
        assert h.config.phone_number == "+15550000001"

    def test_known_phones_is_populated_on_every_handler(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        pool = _make_pool(bots)
        expected = {"+15550000001", "+15550000002"}
        for h in pool.handlers():
            assert h._known_phones == expected


class TestOwnsBot:
    """The per-handler filter that prevents two phones from
    double-answering the same envelope."""

    def _handlers(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        pool = _make_pool(bots)
        return pool, bots, pool.for_phone("+15550000001"), pool.for_phone("+15550000002")

    def test_default_handler_owns_no_resolved_bot(self):
        _, _, sigil_h, artaud_h = self._handlers()
        assert sigil_h._owns_bot(None) is True
        assert artaud_h._owns_bot(None) is False

    def test_handler_owns_its_own_bot(self):
        _, bots, sigil_h, artaud_h = self._handlers()
        sigil, artaud = bots
        assert sigil_h._owns_bot(sigil) is True
        assert artaud_h._owns_bot(artaud) is True

    def test_handler_drops_other_handlers_bot(self):
        _, bots, sigil_h, artaud_h = self._handlers()
        sigil, artaud = bots
        assert sigil_h._owns_bot(artaud) is False
        assert artaud_h._owns_bot(sigil) is False

    def test_orphaned_phone_falls_back_to_default(self):
        # Bot pinned to a phone we have no handler for: default claims it.
        _, _, sigil_h, artaud_h = self._handlers()
        orphan = _bot(99, "ghost", signal_phone="+15559999999")
        assert sigil_h._owns_bot(orphan) is True  # default
        assert artaud_h._owns_bot(orphan) is False  # non-default doesn't


class TestLookupBotByPhone:
    """Structured @-mention routing: a mention's target phone uniquely
    names a bot, regardless of which handler is processing."""

    def test_lookup_returns_bot_for_known_phone(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        pool = _make_pool(bots)
        sigil_h = pool.for_phone("+15550000001")
        # Sigil's handler resolves Artaud from Artaud's phone — the
        # filter then routes the message to Artaud's handler.
        assert sigil_h._lookup_bot_by_phone("+15550000002").slug == "artaud"

    def test_lookup_returns_none_for_unknown_phone(self):
        bots = [_bot(1, "sigil", default_group=True, default_dm=True)]
        pool = _make_pool(bots)
        h = pool.default()
        assert h._lookup_bot_by_phone("+19999999999") is None

    def test_lookup_skips_disabled_bots(self):
        bots = [
            _bot(1, "sigil", default_group=True, default_dm=True),
            _bot(2, "artaud", signal_phone="+15550000002"),
        ]
        bots[1].enabled = False
        pool = _make_pool(bots)
        h = pool.default()
        # Disabled bots don't get a handler in build(), but the registry
        # might still hold their row — _lookup must filter them out.
        assert h._lookup_bot_by_phone("+15550000002") is None


def _wire_dispatch_test(pool, bots):
    """Set up the shared MagicMock dispatcher with a working
    _resolve_bot, the registry/policy stubs each handler reaches for,
    and a pool reference so the cross-handler claim cache engages.
    Returns (dispatch_mock, sigil_h, artaud_h).
    """
    sigil_h = pool.for_phone("+15550000001")
    artaud_h = pool.for_phone("+15550000002")
    dispatch_mock = AsyncMock(return_value=None)
    sigil, _ = bots

    def _resolve(group_id, policy=None, addressed_bot=None):
        return addressed_bot or sigil

    # Both handlers share the same MagicMock dispatcher (the pool
    # wired them up that way); set the real fakes on it once.
    dispatcher = sigil_h.dispatcher
    dispatcher.dispatch = dispatch_mock
    dispatcher._resolve_bot = _resolve
    dispatcher.context_registry = None
    dispatcher.bot_registry = _FakeBotRegistry(bots)
    # Critical for the new takeover logic: hand the dispatcher the
    # real pool so `pool.claim(...)` is the actual cross-handler
    # dedup. Without this, both handlers would see a MagicMock for
    # `signal_pool.claim` (truthy in all cases) and double-dispatch.
    dispatcher.signal_pool = pool
    return dispatch_mock, sigil_h, artaud_h


@pytest.mark.asyncio
async def test_handle_webhook_dedups_when_both_handlers_receive(monkeypatch):
    """Common multi-phone path: signal-cli delivers a copy of the
    envelope to both linked accounts. The first handler to claim wins;
    the second finds the claim and drops. Exactly one dispatch."""
    # Speed the takeover sleep so the test doesn't wait the real
    # POOL_TAKEOVER_DELAY_SEC — the dedup logic doesn't depend on the
    # actual duration.
    monkeypatch.setattr(
        "src.signal.handler.POOL_TAKEOVER_DELAY_SEC", 0.01, raising=True,
    )

    bots = [
        _bot(1, "sigil", default_group=True, default_dm=True),
        _bot(2, "artaud", signal_phone="+15550000002"),
    ]
    pool = _make_pool(bots)
    dispatch_mock, sigil_h, artaud_h = _wire_dispatch_test(pool, bots)

    envelope = {
        "envelope": {
            "source": "+15551112222",
            "sourceUuid": "uuid-1111-2222",
            "timestamp": 1234567890,
            "dataMessage": {
                "message": "Hey artaud",
                "timestamp": 1234567890,
                "mentions": [{"number": "+15550000002"}],
            },
        }
    }

    # Both pollers receive — owner-of-record (Artaud's handler) claims
    # immediately; the non-owner sleeps briefly and finds the claim.
    # Run sequentially: artaud first (claims), then sigil (drops).
    await artaud_h.handle_webhook(envelope)
    await sigil_h.handle_webhook(envelope)

    assert dispatch_mock.call_count == 1
    assert dispatch_mock.call_args.kwargs["addressed_bot"].slug == "artaud"


@pytest.mark.asyncio
async def test_handle_webhook_takes_over_when_owner_misses_delivery(monkeypatch):
    """Recovery path: signal-cli delivers the envelope only to Sigil's
    account (the owner's poller missed it). After the grace period the
    non-owner takes over and dispatches as Artaud anyway — the message
    is no longer silently lost."""
    monkeypatch.setattr(
        "src.signal.handler.POOL_TAKEOVER_DELAY_SEC", 0.01, raising=True,
    )

    bots = [
        _bot(1, "sigil", default_group=True, default_dm=True),
        _bot(2, "artaud", signal_phone="+15550000002"),
    ]
    pool = _make_pool(bots)
    dispatch_mock, sigil_h, _artaud_h = _wire_dispatch_test(pool, bots)

    envelope = {
        "envelope": {
            "source": "+15551112222",
            "sourceUuid": "uuid-1111-2222",
            "timestamp": 1234567891,
            "dataMessage": {
                "message": "Hey artaud",
                "timestamp": 1234567891,
                "mentions": [{"number": "+15550000002"}],
            },
        }
    }

    # Only Sigil's handler receives — Artaud's poller never claims.
    await sigil_h.handle_webhook(envelope)

    # Takeover dispatched on Sigil's side, still as Artaud.
    assert dispatch_mock.call_count == 1
    assert dispatch_mock.call_args.kwargs["addressed_bot"].slug == "artaud"


# --------------------------------------------------------------------------
# Multi-bot fan-out: one message addressing >=2 bots → each replies.
# --------------------------------------------------------------------------

class _FakePolicy:
    """Minimal policy stub for fan-out gating."""
    def __init__(self, llm_intent: bool = True):
        self.llm_intent = llm_intent


def _phoned_bots() -> list[Bot]:
    """Both bots with explicit signal_phone overrides — mirrors the real
    deployment (Sigil and Artaud each on their own number), so @-mention
    phones resolve to a bot from EITHER handler."""
    return [
        _bot(1, "sigil", signal_phone="+15550000001",
             default_group=True, default_dm=True),
        _bot(2, "artaud", signal_phone="+15550000002"),
    ]


@pytest.mark.asyncio
async def test_resolve_addressed_bot_set_both_mentioned():
    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, _ = _wire_dispatch_test(pool, bots)
    data_message = {
        "message": "what do you two think?",
        "mentions": [{"number": "+15550000001"}, {"number": "+15550000002"}],
    }
    found = await sigil_h._resolve_addressed_bot_set(
        data_message, "g1", _FakePolicy(),
    )
    assert [b.slug for b in found] == ["sigil", "artaud"]


@pytest.mark.asyncio
async def test_resolve_addressed_bot_set_single_mention():
    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, _ = _wire_dispatch_test(pool, bots)
    data_message = {
        "message": "hi",
        "mentions": [{"number": "+15550000002"}],
    }
    found = await sigil_h._resolve_addressed_bot_set(
        data_message, "g1", _FakePolicy(),
    )
    assert [b.slug for b in found] == ["artaud"]


@pytest.mark.asyncio
async def test_resolve_addressed_bot_set_typed_names_fallback():
    """No structured @-mentions → fall back to typed aliases; both names
    present → both bots, active (sigil) first."""
    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, _ = _wire_dispatch_test(pool, bots)
    data_message = {"message": "artaud and sigil, weigh in"}
    found = await sigil_h._resolve_addressed_bot_set(
        data_message, "g1", _FakePolicy(),
    )
    assert [b.slug for b in found] == ["sigil", "artaud"]


@pytest.mark.asyncio
async def test_fanout_schedules_secondary_for_non_primary(monkeypatch):
    """With both bots addressed and Sigil the primary, the fan-out
    schedules exactly one secondary answer — for Artaud."""
    import asyncio
    from unittest.mock import AsyncMock as _AM

    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, _ = _wire_dispatch_test(pool, bots)
    sigil_h.dispatcher.ask_command = _AM()
    sigil_h.dispatcher.prefix = "!"
    sigil_h._answer_secondary = _AM()

    data_message = {
        "message": "thoughts?",
        "mentions": [{"number": "+15550000001"}, {"number": "+15550000002"}],
    }
    await sigil_h._fanout_secondary_bots(
        data_message=data_message, group_id="g1", policy=_FakePolicy(),
        sender="+15551112222", message_text="thoughts?",
        quote_text=None, quote_author=None, primary_bot=bots[0],
    )
    await asyncio.sleep(0)  # let the scheduled task run
    sigil_h._answer_secondary.assert_called_once()
    assert sigil_h._answer_secondary.call_args.kwargs["bot"].slug == "artaud"


@pytest.mark.asyncio
async def test_fanout_skips_explicit_commands(monkeypatch):
    """A `!`-prefixed command is single-bot — no fan-out even if two
    bots are named."""
    from unittest.mock import AsyncMock as _AM

    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, _ = _wire_dispatch_test(pool, bots)
    sigil_h.dispatcher.ask_command = _AM()
    sigil_h.dispatcher.prefix = "!"
    sigil_h._answer_secondary = _AM()

    data_message = {
        "message": "!price AAPL",
        "mentions": [{"number": "+15550000001"}, {"number": "+15550000002"}],
    }
    await sigil_h._fanout_secondary_bots(
        data_message=data_message, group_id="g1", policy=_FakePolicy(),
        sender="+15551112222", message_text="!price AAPL",
        quote_text=None, quote_author=None, primary_bot=bots[0],
    )
    sigil_h._answer_secondary.assert_not_called()


@pytest.mark.asyncio
async def test_answer_secondary_skips_user_turn_and_sends_from_own_phone():
    """The secondary answer goes through AskCommand with
    persist_user_turn=False (the primary already stored the shared user
    turn) and is sent from the secondary bot's own phone."""
    from unittest.mock import AsyncMock as _AM, MagicMock as _MM
    from src.commands.base import CommandResult

    bots = _phoned_bots()
    pool = _make_pool(bots)
    _, sigil_h, artaud_h = _wire_dispatch_test(pool, bots)
    sigil_h.dispatcher.prefix = "!"
    ask_obj = _MM()
    ask_obj.execute = _AM(return_value=CommandResult(text="the artist speaks"))
    artaud_h.send_message = _AM()

    await sigil_h._answer_secondary(
        bot=bots[1], ask_command=ask_obj, pool=pool,
        sender="+15551112222", group_id="g1", policy=_FakePolicy(),
        cleaned="thoughts?", quote_text=None, quote_author=None,
    )

    ask_obj.execute.assert_awaited_once()
    ctx = ask_obj.execute.call_args.args[0]
    assert ctx.persist_user_turn is False
    assert ctx.bot.slug == "artaud"
    assert ctx.command == "ask"
    # Sent from Artaud's own handler (its phone), not Sigil's.
    artaud_h.send_message.assert_awaited_once()
    assert artaud_h.send_message.call_args.kwargs["message"] == "the artist speaks"
