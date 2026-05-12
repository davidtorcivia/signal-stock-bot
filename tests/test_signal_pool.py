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


@pytest.mark.asyncio
async def test_handle_webhook_drops_message_routed_to_other_handler():
    """End-to-end of the routing filter: a message addressed to Artaud
    arrives at Sigil's handler (signal-cli broadcasts to both linked
    accounts); Sigil drops, Artaud dispatches."""
    from src.signal.handler import SignalHandler, SignalConfig

    bots = [
        _bot(1, "sigil", default_group=True, default_dm=True),
        _bot(2, "artaud", signal_phone="+15550000002"),
    ]
    pool = _make_pool(bots)

    # Hand-build a Sigil-side handler so we can interrogate its dispatch.
    sigil_h = pool.for_phone("+15550000001")
    artaud_h = pool.for_phone("+15550000002")

    # Both handlers share the same MagicMock dispatcher (the pool wired
    # them up that way) — install one AsyncMock for dispatch and inspect
    # the call list after both handlers have run.
    dispatch_mock = AsyncMock(return_value=None)
    sigil_h.dispatcher.dispatch = dispatch_mock
    # _resolve_bot returns the addressed bot if set, else default.
    sigil, artaud = bots

    def _resolve(group_id, policy=None, addressed_bot=None):
        return addressed_bot or sigil

    sigil_h.dispatcher._resolve_bot = _resolve
    artaud_h.dispatcher._resolve_bot = _resolve
    sigil_h.dispatcher.context_registry = None
    artaud_h.dispatcher.context_registry = None
    sigil_h.dispatcher.bot_registry = _FakeBotRegistry(bots)
    artaud_h.dispatcher.bot_registry = _FakeBotRegistry(bots)

    envelope = {
        "envelope": {
            "source": "+15551112222",
            "timestamp": 1234567890,
            "dataMessage": {
                "message": "Hey artaud",
                "timestamp": 1234567890,
                "mentions": [{"number": "+15550000002"}],
            },
        }
    }

    await sigil_h.handle_webhook(envelope)
    await artaud_h.handle_webhook(envelope)

    # Exactly one dispatch — from Artaud. Sigil saw the envelope,
    # recognized Artaud as the addressee, and dropped before dispatching.
    assert dispatch_mock.call_count == 1
    assert dispatch_mock.call_args.kwargs["addressed_bot"].slug == "artaud"
