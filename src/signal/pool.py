"""SignalHandlerPool — one SignalHandler per registered Signal phone.

Each bot in the registry has an optional `signal_phone`. Bots that leave it
NULL fall back to the global `SIGNAL_PHONE_NUMBER`. The pool indexes
handlers by phone and exposes `for_bot(bot)` so workers can send "as" a
specific bot without knowing the wiring details.

Construction is deliberately eager: the set of registered phones is taken
from the bot registry at startup. Adding a new bot phone at runtime
(via the admin UI) requires a restart to spin up its poller — the
hot-add path isn't worth the complexity until we have more than two
bots.
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any, Iterable, Optional

from .handler import SignalConfig, SignalHandler

if TYPE_CHECKING:
    from ..bots.models import Bot
    from ..bots.registry import BotRegistry
    from ..commands.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


# How long a pool-level claim stays "fresh". Long enough to cover any
# realistic websocket/dedup window across both handlers, short enough
# that the dict never grows unbounded under steady traffic.
_CLAIM_TTL_SEC = 120.0
# Soft cap before GC. Each entry is ~120 bytes; 4096 is well under 1 MB.
_CLAIM_MAX = 4096


class SignalHandlerPool:
    """Routes outbound sends to the right SignalHandler by bot.

    Inbound side: each handler runs its own poller and filters messages
    to bots it serves (see SignalHandler.served_bot_ids), so the pool
    doesn't sit in the receive path.
    """

    def __init__(
        self,
        *,
        default_api_url: str,
        default_phone: str,
        dispatcher: "CommandDispatcher",
        bot_registry: "BotRegistry",
    ):
        self._default_api_url = default_api_url.rstrip("/")
        self._default_phone = default_phone
        self._dispatcher = dispatcher
        self._bot_registry = bot_registry
        # phone -> SignalHandler. Built lazily on the first call so
        # callers can construct the pool before the registry is warm
        # (e.g. during wiring in main.py — the registry's warm_sync
        # has already run by the time main constructs the pool, but
        # tests sometimes inject a registry that isn't loaded yet).
        self._handlers: dict[str, SignalHandler] = {}
        self._built = False
        # Cross-handler claim cache used by `claim()` below. Keyed by
        # (group_id_or_"dm", source_uuid, dataMessage.timestamp). First
        # handler to claim an inbound envelope wins; the other handler
        # (if it also received a copy) sees the claim and drops.
        self._claims: dict[Any, float] = {}
        # Pool-shared set of every bot UUID we've discovered. Aliased
        # into each handler's `_known_uuids` at build() so a UUID
        # learned on one handler's first send is immediately visible to
        # every other handler's inbound self-check. Bootstrapped eagerly
        # via `bootstrap_uuids()` so the first inbound has UUIDs to
        # compare against — without that, a bot's own outbound echoes
        # back through the OTHER bot's poller with envelope.source set
        # to a UUID (not a phone), slipping past the phone-only filter
        # and letting the reactor evaluate it as user input.
        self._known_uuids: set[str] = set()

    # ------------------------------------------------------------------
    # Build
    # ------------------------------------------------------------------

    def build(self) -> None:
        """Construct one handler per distinct phone the registry knows about.

        Phones are: the global default phone (always present, so single-bot
        installs keep working when bot_registry is cold or empty) plus
        every enabled bot's `signal_phone` override.
        """
        if self._built:
            return
        seen: dict[str, tuple[str, set[int]]] = {}
        # Start with the global default phone so non-overriding bots
        # still have a handler. Bots that share this phone get added
        # to its served set below.
        if self._default_phone:
            seen[self._default_phone] = (self._default_api_url, set())

        for bot in self._bot_registry.list_sync():
            if not bot.enabled or bot.id is None:
                continue
            phone = (bot.signal_phone or "").strip() or self._default_phone
            if not phone:
                continue
            api_url = (bot.signal_api_url or "").strip() or self._default_api_url
            existing = seen.get(phone)
            if existing is None:
                seen[phone] = (api_url, {bot.id})
            else:
                # If two bots disagree on api_url for the same phone the
                # first registration wins; this is an admin misconfig and
                # we log so it's visible.
                if existing[0] != api_url:
                    logger.warning(
                        f"Bot {bot.slug!r} signal_api_url={api_url!r} conflicts with "
                        f"earlier registration for {phone!r} ({existing[0]!r}); "
                        f"keeping the first."
                    )
                existing[1].add(bot.id)

        all_phones = set(seen.keys())
        for phone, (api_url, served) in seen.items():
            handler = SignalHandler(
                SignalConfig(api_url=api_url, phone_number=phone),
                self._dispatcher,
            )
            # served_bot_ids lets the handler filter inbound dispatch
            # to its own bot(s). Empty set means "no override" — the
            # default-phone handler accepts everything (single-bot
            # installs and bots that don't override signal_phone).
            handler.served_bot_ids = served
            handler.is_default_phone = (phone == self._default_phone)
            # `_known_phones` is the pool-wide set of registered phones;
            # the handler uses it to detect when a resolved bot is
            # pinned to a phone no handler can serve, and steps in
            # (when it's the default) so the chat doesn't go silent.
            handler._known_phones = set(all_phones)
            # Share the SAME set object across handlers so a UUID learned
            # on one handler's send is instantly visible everywhere.
            handler._known_uuids = self._known_uuids
            self._handlers[phone] = handler
            tail = phone[-4:] if len(phone) >= 4 else phone
            logger.info(
                f"SignalHandlerPool: registered handler for ...{tail} "
                f"(served_bots={sorted(served) or 'default'})"
            )
        self._built = True

    # ------------------------------------------------------------------
    # Lookups
    # ------------------------------------------------------------------

    def for_bot(self, bot: Optional["Bot"]) -> SignalHandler:
        """Return the handler that should send "as" this bot. None / no
        override falls back to the default handler so callers without a
        resolved bot still get a working channel."""
        if not self._built:
            self.build()
        if bot is not None and bot.signal_phone:
            handler = self._handlers.get(bot.signal_phone)
            if handler is not None:
                return handler
            logger.warning(
                f"SignalHandlerPool.for_bot({bot.slug!r}): no handler for "
                f"phone {bot.signal_phone!r}, falling back to default"
            )
        return self.default()

    def for_phone(self, phone: str) -> Optional[SignalHandler]:
        if not self._built:
            self.build()
        return self._handlers.get(phone)

    def bot_for_uuid(self, uuid: Optional[str]):
        """Return the bot whose Signal account UUID matches `uuid`, or None.

        Resolves @-mentions that carry only a UUID (modern Signal hides
        phone numbers, so a tap-mention often has no `number`). We map
        UUID -> the handler that owns it -> the single bot it serves. This
        is GLOBAL (any handler can resolve any bot's mention to the same
        bot), which the multi-phone claim filter relies on — otherwise the
        two handlers disagree on who was addressed and the wrong one claims
        the envelope. Ambiguous (a shared-phone handler serving several
        bots) returns None so the caller falls back."""
        if not uuid:
            return None
        if not self._built:
            self.build()
        for handler in self._handlers.values():
            if handler._bot_uuid and handler._bot_uuid == uuid:
                served = list(handler.served_bot_ids)
                if len(served) == 1 and self._bot_registry is not None:
                    try:
                        return self._bot_registry.get_sync(served[0])
                    except Exception:
                        return None
                return None
        return None

    def default(self) -> SignalHandler:
        if not self._built:
            self.build()
        handler = self._handlers.get(self._default_phone)
        if handler is None:
            # Pathological: pool was built with no default phone and no
            # bot overrides. Return any handler so callers don't crash.
            if self._handlers:
                return next(iter(self._handlers.values()))
            raise RuntimeError("SignalHandlerPool: no handlers registered")
        return handler

    def refresh_known_phones(self) -> int:
        """Rebuild every handler's `_known_phones` set from the current
        bot registry. Call this after the admin adds/removes/edits a
        bot at runtime — without it, the anti-bot-to-bot loop guard
        and the "is the resolved bot's phone reachable?" check both
        use a stale snapshot from boot. Returns the size of the new
        set."""
        if not self._built:
            self.build()
            return len(self._handlers)
        # Refresh from the registry rather than just our own handlers:
        # a bot might be registered with a phone we DON'T have a
        # handler for yet (e.g. partially-configured), and we still
        # want the existing handlers to NOT dispatch as that bot's
        # outbound and not loop on it. Pull the live registry phone
        # list from the dispatcher's bot_registry.
        bot_registry = getattr(self._dispatcher, "bot_registry", None)
        phones: set[str] = set(self._handlers.keys())
        if bot_registry is not None:
            try:
                for b in bot_registry.list_sync():
                    phone = (getattr(b, "signal_phone", None) or "").strip()
                    if phone:
                        phones.add(phone)
            except Exception as e:
                logger.warning(
                    f"SignalHandlerPool.refresh_known_phones: registry "
                    f"read failed: {e}"
                )
        for handler in self._handlers.values():
            handler._known_phones = phones
        return len(phones)

    async def bootstrap_uuids(self) -> int:
        """Eagerly resolve every handler's bot UUID so the inbound
        self-check has a populated `_known_uuids` set on the first
        envelope.

        Without this, a bot's outbound message can echo back through a
        sibling handler's poller with envelope.source set to a UUID
        rather than a phone number — the phone-only filter at
        `handle_webhook` lets it through and the reactor evaluates the
        bot's own message as user input (in multi-bot chats this lets
        the LLM pick a bot to "reply" to its own post).

        Failures are non-fatal — the handler's lazy fetch_bot_uuid will
        still try later. Returns the count of UUIDs resolved.
        """
        if not self._built:
            self.build()
        resolved = 0
        for handler in self._handlers.values():
            try:
                uuid = await handler.fetch_bot_uuid()
                if uuid:
                    resolved += 1
            except Exception as e:
                logger.warning(
                    f"SignalHandlerPool.bootstrap_uuids: "
                    f"{handler.config.phone_number!r} failed: {e}"
                )
        logger.info(
            f"SignalHandlerPool: bootstrapped {resolved}/{len(self._handlers)} "
            f"bot UUIDs ({len(self._known_uuids)} known)"
        )
        return resolved

    def handlers(self) -> Iterable[SignalHandler]:
        if not self._built:
            self.build()
        return self._handlers.values()

    def phones(self) -> list[str]:
        if not self._built:
            self.build()
        return list(self._handlers.keys())

    @property
    def default_phone(self) -> str:
        return self._default_phone

    # ------------------------------------------------------------------
    # Cross-handler envelope claim
    # ------------------------------------------------------------------

    def claim(self, key: Any) -> bool:
        """Atomically claim an inbound envelope by `key`.

        Returns True if this caller is the first to claim, False if a
        sibling handler already did. Used by the handler's takeover
        path: the owner-of-record handler claims immediately on receipt;
        the non-owner sleeps briefly and then tries to claim — if the
        owner's poller actually delivered the envelope it will already
        be claimed and the non-owner drops; if it didn't (signal-cli
        delivery gap), the non-owner takes over and dispatches as the
        addressed bot, sending the reply through the right phone via
        `for_bot(...)`.

        Idempotent within the TTL window: a second claim for the same
        key returns False. GC'd lazily — every call drops entries older
        than `_CLAIM_TTL_SEC`, with a hard cap that triggers a full
        sweep when the dict grows past `_CLAIM_MAX`.
        """
        now = time.time()
        cutoff = now - _CLAIM_TTL_SEC
        if len(self._claims) > _CLAIM_MAX:
            # Hard sweep: rebuild keeping only fresh entries. Cheap
            # relative to the in-handler 2.5s sleep so we don't care
            # about doing it on the hot path.
            self._claims = {k: v for k, v in self._claims.items() if v >= cutoff}
        existing = self._claims.get(key)
        if existing is not None and existing >= cutoff:
            return False
        self._claims[key] = now
        return True
