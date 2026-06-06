"""
Signal message handler - interfaces with signal-cli-rest-api.
"""

import asyncio
import base64
import logging
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import aiohttp

from ..admin.events import get_bus
from ..commands.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


# Vision-payload guardrails. Vision models charge per image and per
# resolution; passing arbitrary inbound media without a cap would let
# any user trivially burn budget. 5 MB matches the largest single image
# Anthropic/OpenAI ingest without server-side downscaling; 4 images per
# message covers the realistic "share a screenshot collage" case
# without becoming a vector for spam-prompts.
VISION_MAX_PER_IMAGE_BYTES = 5 * 1024 * 1024
VISION_MAX_IMAGES = 4
VISION_ALLOWED_MIMES = frozenset({
    "image/jpeg", "image/png", "image/webp", "image/gif",
})

# Path inside the container where signal-cli stores inbound media. Set
# by the SIGNAL_ATTACHMENTS_DIR env var so tests can redirect.
_DEFAULT_ATTACHMENTS_DIR = "/app/data/signal-cli/attachments"


# Grace period before a non-owning handler claims an envelope addressed
# to a bot on another phone. Long enough that the owner's poller
# almost always claims first under normal conditions; short enough
# that recovery latency on the (rare) takeover path doesn't feel like
# a hang. The common path (both pollers receive — owner claims at
# t=0, non-owner finds the claim at t=2.5s and drops) just adds a
# silent coroutine sleep; user-visible latency is unaffected.
POOL_TAKEOVER_DELAY_SEC = 2.5


def _attachments_dir() -> Path:
    return Path(os.environ.get("SIGNAL_ATTACHMENTS_DIR", _DEFAULT_ATTACHMENTS_DIR))


def _read_inbound_image_attachments(
    data_message: dict, *, max_images: int = VISION_MAX_IMAGES,
) -> list[dict]:
    """Extract image attachments referenced by an inbound dataMessage.

    Returns a list of `{mime, data_b64, filename}` dicts ready to be
    handed to a multimodal LLM. Silently skips entries that:
      * are missing an `id`
      * have a non-image (or unsupported) mime type
      * point at a file that doesn't exist on disk yet (signal-cli
        race) or that exceeds VISION_MAX_PER_IMAGE_BYTES.

    Capped at `max_images` to bound payload size per round.
    """
    raw = data_message.get("attachments") or []
    if not raw:
        return []
    out: list[dict] = []
    base = _attachments_dir()
    for att in raw:
        if len(out) >= max_images:
            break
        if not isinstance(att, dict):
            continue
        mime = (att.get("contentType") or "").strip().lower()
        if mime not in VISION_ALLOWED_MIMES:
            continue
        att_id = (att.get("id") or "").strip()
        if not att_id:
            continue
        # signal-cli stores attachments by id; in some configs the
        # filename includes the original extension, in others it's
        # just the id. Try the id first, then glob for it as a
        # fallback (rare path that costs one directory listing).
        path = base / att_id
        if not path.exists():
            try:
                matches = list(base.glob(f"{att_id}.*"))
            except OSError:
                matches = []
            if not matches:
                logger.debug(
                    f"inbound image not on disk yet: {att_id} "
                    f"(mime={mime}); skipping"
                )
                continue
            path = matches[0]
        try:
            size = path.stat().st_size
        except OSError as e:
            logger.debug(f"inbound image stat failed: {att_id}: {e}")
            continue
        if size > VISION_MAX_PER_IMAGE_BYTES:
            logger.info(
                f"inbound image {att_id} skipped: {size} bytes exceeds "
                f"{VISION_MAX_PER_IMAGE_BYTES} cap"
            )
            continue
        try:
            with path.open("rb") as f:
                payload = f.read()
        except OSError as e:
            logger.debug(f"inbound image read failed: {att_id}: {e}")
            continue
        out.append({
            "mime": mime,
            "data_b64": base64.b64encode(payload).decode("ascii"),
            "filename": att.get("filename") or att_id,
        })
    return out


@dataclass
class SignalConfig:
    """Configuration for Signal API connection"""
    api_url: str
    phone_number: str
    
    def __post_init__(self):
        # Ensure no trailing slash
        self.api_url = self.api_url.rstrip("/")


class SignalHandler:
    """
    Handles Signal message sending/receiving via signal-cli-rest-api.
    
    Webhook format from signal-cli-rest-api:
    {
        "envelope": {
            "source": "+15551234567",
            "sourceDevice": 1,
            "timestamp": 1234567890,
            "dataMessage": {
                "message": "@StockBot what's AAPL?",
                "mentions": [
                    {
                        "uuid": "abc123...",
                        "start": 0,
                        "length": 9
                    }
                ],
                "groupInfo": {
                    "groupId": "abc123..."
                }
            }
        }
    }
    """
    
    def __init__(self, config: SignalConfig, dispatcher: CommandDispatcher):
        self.config = config
        self.dispatcher = dispatcher
        self._session: Optional[aiohttp.ClientSession] = None
        self._bot_uuid: Optional[str] = None  # Fetched on first use
        self._group_id_map: dict[str, str] = {}
        self._group_map_lock = asyncio.Lock()
        # (sender, timestamp) -> seen_at — drops duplicate webhook
        # deliveries. signal-cli can re-emit messages on reconnect; we
        # don't want the bot to dispatch the same `!ask` twice. Bounded
        # in size to cap memory under sustained traffic.
        self._seen_messages: dict[tuple, float] = {}
        self._seen_messages_max = 1024
        # PollVoter is injected post-construction (avoids circular deps with
        # the LLM client / group log). When None, inbound polls are ignored.
        self.poll_voter = None
        # Set by SignalHandlerPool at construction. Empty set + is_default_phone=True
        # means "this is the global-default handler, accept everything." Otherwise
        # the handler dispatches only messages whose resolved bot belongs to it
        # (so two phones in the same group don't both answer). `_known_phones`
        # is the union of all phones the pool has handlers for — used in the
        # filter's fallback rule (see `_owns_bot`).
        self.served_bot_ids: set[int] = set()
        self.is_default_phone: bool = True
        self._known_phones: set[str] = set()
        # Pool-shared set of every bot UUID known to the deployment.
        # SignalHandlerPool aliases its own `_known_uuids` here at
        # build() so a UUID resolved on any handler is instantly visible
        # to every handler's inbound self-check — without this, signal-
        # cli's UUID-form `envelope.source` slips past the phone-only
        # filter and lets the reactor evaluate a bot's own outbound as
        # user input.
        self._known_uuids: set[str] = set()
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _resolve_group_id(self, group_id: str) -> str:
        """Resolve internal group ID to V2 group ID (required for sending)."""
        if group_id.startswith("group."):
            return group_id

        if group_id in self._group_id_map:
            return self._group_id_map[group_id]

        async with self._group_map_lock:
            # Re-check after acquiring the lock
            if group_id in self._group_id_map:
                return self._group_id_map[group_id]
            await self._refresh_group_map_locked()

        return self._group_id_map.get(group_id, group_id)

    async def _refresh_group_map_locked(self):
        """Fetch groups from API and update ID map. Caller must hold the lock."""
        try:
            session = await self._get_session()
            url = f"{self.config.api_url}/v1/groups/{self.config.phone_number}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    groups = await resp.json()
                    for group in groups:
                        internal_id = group.get("internal_id")
                        v2_id = group.get("id")
                        if internal_id and v2_id:
                            self._group_id_map[internal_id] = v2_id
                    logger.info(f"Updated group ID map with {len(groups)} groups")
                else:
                    logger.error(f"Failed to fetch groups: {resp.status}")
        except Exception as e:
            logger.error(f"Error refreshing group map: {e}")

    async def send_message(
        self,
        recipient: str,
        message: str,
        group_id: Optional[str] = None,
        attachments: Optional[list[str]] = None,
        styled: bool = False,
    ):
        """
        Send a message to a recipient or group.

        Args:
            recipient: Phone number or group ID
            message: Message text
            group_id: If set, sends to this group instead of recipient
            attachments: Optional list of base64-encoded images
            styled: If True, the text is treated as markdown and converted to
                    Signal styled-text syntax (`*bold*`, `_italic_`, etc.) and
                    sent with text_mode=styled. LLM-produced messages set this.
        """
        session = await self._get_session()

        # Em-dash normalisation runs unconditionally — LLMs love them, but
        # they read awkwardly in chat and the user wants them out everywhere.
        if message:
            message = message.replace("—", " - ").replace("–", "-")

        if styled:
            from .markdown import to_signal_styled
            message = to_signal_styled(message)

        # Surface outbound text to the live admin viewer.
        try:
            get_bus().publish(
                "outbound",
                recipient_tail=(recipient or "")[-4:],
                group_id=group_id,
                styled=styled,
                attachments=len(attachments) if attachments else 0,
                text=(message or "")[:240],
            )
        except Exception:
            pass

        # Build payload for v2 API
        payload = {
            "number": self.config.phone_number,
            "message": message,
        }
        if styled:
            payload["text_mode"] = "styled"

        if group_id:
            # Resolve group ID to V2 ID
            resolved_id = await self._resolve_group_id(group_id)
            payload["recipients"] = [resolved_id]
        else:
            payload["recipients"] = [recipient]

        # Add base64 attachments if provided
        if attachments:
            # Signal API format: "data:<mime>;filename=<name>;base64,<data>"
            payload["base64_attachments"] = [
                f"data:image/png;filename=chart.png;base64,{att}"
                for att in attachments
            ]
        
        url = f"{self.config.api_url}/v2/send"

        try:
            async with session.post(url, json=payload) as resp:
                if resp.status not in (200, 201):
                    error = await resp.text()
                    # Log payload for debugging (truncate attachments)
                    debug_payload = payload.copy()
                    if "base64_attachments" in debug_payload:
                        debug_payload["base64_attachments"] = [
                            f"{att[:30]}..." for att in debug_payload["base64_attachments"]
                        ]
                    logger.error(f"Failed to send message: {resp.status} - {error} - Payload: {debug_payload}")
                    # Stale-group invalidation: if a send to a group fails
                    # with a 4xx (group renamed, bot removed-readded, v2 ID
                    # rotated), drop the cached resolution so the next send
                    # re-fetches. Without this, every subsequent send to
                    # this internal id silently fails until the bot
                    # restarts.
                    if group_id and 400 <= resp.status < 500:
                        self._group_id_map.pop(group_id, None)
                    raise Exception(f"Send failed: {resp.status}")

                tail = recipient[-4:] if recipient and len(recipient) >= 4 else (group_id or "?")
                logger.debug(f"Message sent successfully to {tail}")

        except Exception as e:
            logger.error(f"Failed to send response: {e}")
            raise
    
    async def send_reaction(
        self,
        recipient: str,
        target_author: str,
        target_timestamp: int,
        emoji: str,
        group_id: Optional[str] = None,
        remove: bool = False,
    ) -> bool:
        """Add (or remove) an emoji reaction to a specific message.

        Reactions attach to the original envelope — they do NOT appear as
        new messages. The signal-cli-rest-api endpoint is
        /v1/reactions/{number}.

        Args:
            recipient: phone number to address (ignored when group_id is set)
            target_author: phone number of the message author being reacted to
            target_timestamp: envelope timestamp of the target message
            emoji: single Unicode emoji
            group_id: send reaction in this group (recommended for groups)
            remove: if True, remove an existing reaction with this emoji
        """
        session = await self._get_session()
        payload: dict[str, object] = {
            "reaction": emoji,
            "target_author": target_author,
            "timestamp": int(target_timestamp),
        }
        if remove:
            payload["remove"] = True
        if group_id:
            payload["recipient"] = await self._resolve_group_id(group_id)
        else:
            payload["recipient"] = recipient

        url = f"{self.config.api_url}/v1/reactions/{self.config.phone_number}"
        try:
            async with session.post(url, json=payload) as resp:
                if resp.status not in (200, 201, 204):
                    text = await resp.text()
                    logger.warning(
                        f"Reaction failed: {resp.status} {text[:200]}"
                    )
                    return False
                return True
        except Exception as e:
            logger.error(f"Reaction send error: {e}")
            return False

    async def send_poll_vote(
        self,
        *,
        poll_author: str,
        poll_timestamp: int,
        selected_answers: list[int],
        group_id: Optional[str] = None,
        recipient: Optional[str] = None,
    ) -> bool:
        """Cast a vote on a poll the bot has seen.

        signal-cli-rest-api endpoint: POST /v1/polls/{number}/vote
        Body shape:
          {
            "poll_author": "<phone or uuid>",
            "poll_timestamp": "<stringified ms>",
            "recipient": "<phone or group id>",
            "selected_answers": [int, ...]
          }

        Note `poll_timestamp` is a STRING in the REST schema even though it
        represents a millisecond integer; sending it as an int returns
        "invalid request".
        """
        if not selected_answers:
            return False
        target = (
            await self._resolve_group_id(group_id) if group_id else (recipient or "")
        )
        if not target:
            logger.warning("send_poll_vote called with no recipient/group")
            return False

        # signal-cli-rest-api expects 1-INDEXED option numbers (we hit a 400
        # "index needs to be >= 1" otherwise). Callers pass 0-indexed values
        # because that's how the inbound poll defines them; we shift here at
        # the wire boundary so the rest of the codebase stays consistent.
        wire_indices = [i + 1 for i in selected_answers]

        session = await self._get_session()
        payload = {
            "poll_author": poll_author,
            "poll_timestamp": str(int(poll_timestamp)),
            "recipient": target,
            "selected_answers": wire_indices,
        }
        url = f"{self.config.api_url}/v1/polls/{self.config.phone_number}/vote"
        try:
            async with session.post(url, json=payload) as resp:
                body = await resp.text()
                if resp.status not in (200, 201, 204):
                    logger.warning(
                        f"Poll vote failed: {resp.status} {body[:200]} payload={payload}"
                    )
                    return False
                logger.info(
                    f"Poll vote sent: options={selected_answers} "
                    f"author={poll_author[-6:]} ts={poll_timestamp}"
                )
                return True
        except Exception as e:
            logger.error(f"Poll vote send error: {e}")
            return False

    async def send_typing(
        self,
        recipient: str,
        group_id: Optional[str] = None,
        stop: bool = False,
    ) -> bool:
        """Show or clear the typing indicator for a recipient or group.

        Signal clients clear the indicator on their own after ~15 seconds,
        so for long-running work the caller is responsible for refreshing
        — see `typing_indicator()` for an async context manager that
        handles the refresh loop.
        """
        session = await self._get_session()
        target = await self._resolve_group_id(group_id) if group_id else recipient
        payload = {"recipient": target}
        url = f"{self.config.api_url}/v1/typing-indicator/{self.config.phone_number}"
        method = "DELETE" if stop else "PUT"
        try:
            async with session.request(method, url, json=payload) as resp:
                if resp.status not in (200, 204):
                    body = await resp.text()
                    logger.debug(
                        f"Typing indicator {method} returned {resp.status}: "
                        f"{body[:120]}"
                    )
                    return False
                return True
        except Exception as e:
            logger.debug(f"Typing indicator error ({method}): {e}")
            return False

    def typing_indicator(
        self,
        recipient: str,
        group_id: Optional[str] = None,
        refresh_interval: float = 10.0,
    ):
        """Async context manager that keeps the typing indicator visible.

        Usage:
            async with handler.typing_indicator(sender, group_id):
                await long_running_work()

        Refreshes every `refresh_interval` seconds because Signal clients
        auto-clear the indicator after ~15s. Errors are swallowed so the
        wrapped work is never affected by indicator failures.
        """
        return _TypingIndicator(self, recipient, group_id, refresh_interval)

    def _lookup_bot_by_phone(self, phone: str):
        """Return the bot whose signal_phone matches `phone`, or None.

        Used when a structured @-mention names this handler's number —
        with multi-phone routing, that mention uniquely identifies a
        bot (the one that owns the phone) and overrides the context's
        default pick.
        """
        if not phone:
            return None
        if (
            self.dispatcher is None
            or self.dispatcher.bot_registry is None
        ):
            return None
        for bot in self.dispatcher.bot_registry.list_sync():
            if bot.enabled and (bot.signal_phone or "") == phone:
                return bot
        return None

    @staticmethod
    def _mention_ids(mention: dict) -> tuple:
        """Normalize a mention's (number, uuid) across signal-cli shapes.

        signal-cli mentions carry `number` and/or `uuid`; on number-privacy
        accounts only the UUID is present, sometimes under `name`/`aci`.
        """
        number = (mention.get("number") or "").strip()
        uuid = (
            mention.get("uuid") or mention.get("aci") or mention.get("author")
            or ""
        ).strip()
        if not uuid:
            # `name` holds the UUID when the number is hidden.
            nm = (mention.get("name") or "").strip()
            if "-" in nm and " " not in nm:  # looks like a UUID, not a label
                uuid = nm
        return number, uuid

    def _mention_targets_self(self, mention: dict) -> bool:
        """True if this mention names THIS handler's own Signal account."""
        number, uuid = self._mention_ids(mention)
        return bool(
            (number and number == self.config.phone_number)
            or (self._bot_uuid and uuid == self._bot_uuid)
        )

    def _resolve_mention_bot(self, mention: dict):
        """Resolve a structured @-mention to the bot it addresses, or None.

        Order: by phone (global) → by UUID (global, via the pool) → by THIS
        handler's own identity → the single bot this handler serves.

        Crucially this returns the SERVED bot, never the pinned `active_bot`
        — a tap-mention of Sigil in an Artaud-pinned group must summon
        Sigil. And it resolves the SAME bot on every handler (phone/UUID
        lookups are global), so the multi-phone claim filter routes the
        envelope to the right phone instead of two handlers disagreeing and
        the wrong one claiming it (which dropped the message — the bot you
        tagged stayed silent). Returns None only when the mention names no
        known bot, or names a shared-phone account serving several bots
        (ambiguous — caller falls back to the pinned bot).
        """
        number, uuid = self._mention_ids(mention)
        bot = self._lookup_bot_by_phone(number)
        if bot is not None:
            return bot
        pool = getattr(self.dispatcher, "signal_pool", None) if self.dispatcher else None
        if uuid and pool is not None and hasattr(pool, "bot_for_uuid"):
            bot = pool.bot_for_uuid(uuid)
            if bot is not None:
                return bot
        # Mention names this handler's own account → the bot it serves.
        if self._mention_targets_self(mention):
            served = list(self.served_bot_ids)
            if (
                len(served) == 1
                and self.dispatcher is not None
                and self.dispatcher.bot_registry is not None
            ):
                return self.dispatcher.bot_registry.get_sync(served[0])
        return None

    @staticmethod
    def _build_claim_key(envelope: dict, data_message: dict, group_id):
        """Pool-shared key for cross-handler envelope deduplication.

        Both handlers must compute the SAME key when they receive a
        copy of the same envelope; otherwise dedup fails and we double-
        respond. We prefer `sourceUuid` because it's stable across
        accounts — signal-cli's `source` field can be a phone number on
        one account's view of a contact and a UUID on another's, which
        would split the key. Falls back to `source` only when the UUID
        is missing.

        `dataMessage.timestamp` is the canonical Signal "message id" and
        is identical across both receiving accounts. Returns None when
        we can't build a stable key — caller should fall through to the
        legacy drop-silently path so we don't accidentally double-
        dispatch.
        """
        ts = data_message.get("timestamp") or envelope.get("timestamp")
        if not ts:
            return None
        source_uuid = (envelope.get("sourceUuid") or "").strip()
        if not source_uuid:
            # Fallback: `source` is less stable but better than nothing.
            source_uuid = (envelope.get("source") or "").strip()
        if not source_uuid:
            return None
        return (group_id or "dm", source_uuid, int(ts))

    def _owns_bot(self, bot) -> bool:
        """True if this handler should send-as the given bot.

        Rules, in order:
          * No bot resolved: only the default-phone handler proceeds —
            otherwise an unaddressed message in a context without a
            pinned bot would go silent on multi-phone installs.
          * Bot has no signal_phone override: belongs to the default
            handler.
          * Bot's signal_phone matches this handler: claim it.
          * Bot's signal_phone is set to a phone no handler in the pool
            recognizes (admin pinned a bot whose number isn't deployed
            yet): the default handler steps in as a fallback so the
            chat isn't silent — better to answer in the wrong voice than
            not at all.
          * Otherwise: another handler owns this bot — drop.
        """
        if bot is None:
            return self.is_default_phone
        bot_phone = (getattr(bot, "signal_phone", None) or "").strip()
        if not bot_phone:
            return self.is_default_phone
        if bot_phone == self.config.phone_number:
            return True
        if bot_phone not in self._known_phones and self.is_default_phone:
            return True
        return False

    async def fetch_bot_uuid(self) -> Optional[str]:
        """Fetch and cache the bot's UUID from signal-cli API.

        Uses the per-account identities listing — signal-cli always
        records a self-trust entry for the account itself, so the row
        where `number == self.config.phone_number` carries our own
        UUID. (`/v1/about` only returns version info, which is why the
        old implementation always failed silently.)

        On success, also registers the UUID into the pool-shared
        `_known_uuids` set so every sibling handler can recognize this
        bot's outbound echoes on the next inbound — see the self-check
        in `handle_webhook`.
        """
        if self._bot_uuid:
            return self._bot_uuid

        try:
            session = await self._get_session()
            # signal-cli-rest-api quirk: the path segment is the phone
            # number with the leading `+` URL-encoded as `%2B`.
            encoded = self.config.phone_number.replace("+", "%2B")
            url = f"{self.config.api_url}/v1/identities/{encoded}"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    rows = data if isinstance(data, list) else [data]
                    for row in rows:
                        if row.get("number") == self.config.phone_number:
                            self._bot_uuid = row.get("uuid")
                            if self._bot_uuid:
                                self._known_uuids.add(self._bot_uuid)
                                logger.info(
                                    f"Fetched bot UUID for "
                                    f"...{self.config.phone_number[-4:]}: "
                                    f"...{self._bot_uuid[-8:]}"
                                )
                            return self._bot_uuid
                else:
                    logger.debug(
                        f"fetch_bot_uuid HTTP {resp.status} for "
                        f"{self.config.phone_number}"
                    )
        except Exception as e:
            logger.debug(f"Could not fetch bot UUID: {e}")

        return None

    # Filler words allowed before a leading vocative name without breaking
    # it ("hey Sigil …", "ok Artaud …"). Lowercased.
    _VOCATIVE_GREETINGS = frozenset({
        "hey", "hi", "hello", "yo", "ok", "okay", "so", "um", "uh", "well",
        "hmm", "guys", "everyone", "folks", "both", "also", "please",
    })
    # When one of these sits immediately before a TRAILING name, the name is
    # an object/topic ("what do you think of Artaud?"), not a vocative — so
    # it must NOT be treated as addressing that bot.
    _TOPIC_MARKERS = frozenset({
        "of", "about", "regarding", "re", "abt", "like", "than",
    })
    _CONNECTORS = frozenset({",", "&", "and"})

    @staticmethod
    def _alias_bot_pairs(bots) -> list:
        """(lowercased alias, bot) for every enabled bot's alias_set."""
        pairs: list = []
        for b in bots:
            try:
                for a in b.alias_set():
                    a = (a or "").strip().lower()
                    if a:
                        pairs.append((a, b))
            except Exception:
                continue
        return pairs

    def _mentions_any_alias(self, text: str, bots) -> bool:
        """True if any bot alias appears anywhere as a whole word."""
        low = (text or "").lower()
        for alias, _ in self._alias_bot_pairs(bots):
            if re.search(rf"\b{re.escape(alias)}\b", low):
                return True
        return False

    def _addressed_bots(self, text: str, bots) -> list:
        """Bots ADDRESSED by name as a vocative — leading OR trailing — in
        order of appearance, deduped. A name buried mid-sentence as a topic
        is NOT addressed.

        Leading: name(s) at the start (after greeting fillers), joined by
        and/&/comma — "Sigil, …", "hey Sigil and Artaud …".
        Trailing: name(s) at the very end (ignoring punctuation), joined by
        connectors — "what about you, sigil?", "thoughts sigil?" — UNLESS
        the token right before the run is a topic marker like "of"/"about"
        (which makes it an object, not an addressee).

        Examples:
          "Sigil and Artaud, thoughts?"   -> [Sigil, Artaud]
          "what about you sigil?"          -> [Sigil]
          "Sigil what do you make of Artaud?" -> [Sigil]  (Artaud = topic)
          "what do you think of Artaud?"   -> []          (topic only)
        """
        alias_map: dict = {}
        for alias, b in self._alias_bot_pairs(bots):
            alias_map.setdefault(alias, b)
        if not alias_map:
            return []
        tokens = re.findall(r"[a-z0-9'’]+|,|&", (text or "").lower())
        if not tokens:
            return []

        found: list = []

        def _add(bot) -> None:
            bid = getattr(bot, "id", None)
            if all(getattr(x, "id", None) != bid for x in found):
                found.append(bot)

        # Leading vocative: greeting fillers, then names + connectors, until
        # the first token that is neither.
        seen_name = False
        for tok in tokens:
            if tok in self._CONNECTORS:
                continue
            bot = alias_map.get(tok)
            if bot is not None:
                _add(bot)
                seen_name = True
                continue
            if not seen_name and tok in self._VOCATIVE_GREETINGS:
                continue
            break

        # Trailing vocative: walk backward over a run of names + connectors
        # at the very end; reject if a topic marker sits just before it.
        trailing: list = []
        run_first = None  # lowest index of the trailing name run
        i = len(tokens) - 1
        while i >= 0:
            tok = tokens[i]
            if tok in self._CONNECTORS:
                i -= 1
                continue
            bot = alias_map.get(tok)
            if bot is not None:
                trailing.append(bot)
                run_first = i
                i -= 1
                continue
            break
        if trailing and run_first is not None and run_first - 1 >= 0:
            if tokens[run_first - 1] in self._TOPIC_MARKERS:
                trailing = []
        for bot in reversed(trailing):  # restore appearance order
            _add(bot)

        return found

    async def _resolve_addressed_bot(
        self, data_message: dict, group_id: Optional[str], policy
    ):
        """Return the bot that the inbound message is addressing, or None.

        Mention sources, in order of strength:
          1. Signal structured @-mention matching the bot's phone/UUID.
             When the Signal phone is shared (initial Artaud rollout),
             this resolves to the same bot for every receiver, so we
             defer to alias-matching to disambiguate.
          2. Whole-word, case-insensitive alias match in the message
             text against the *active* bot for this context. "Active"
             means: the bot that `_resolve_bot(policy)` would pick
             — the pinned `default_bot_id`, or the registry's
             default-for-kind. This enforces the "Sigil-only context
             never answers to 'Artaud'" rule by gating alias-matching
             on context membership rather than the global bot list.
          3. Quote-reply to one of our previous messages. With
             multi-phone, the quoted author's phone uniquely
             identifies the bot that wrote the quoted message, so the
             reply routes to that same bot — quoting Sigil's message
             summons Sigil even in an Artaud-pinned context. With a
             shared phone we can't tell which bot wrote the quoted
             message, so the active bot for the context is the safe
             pick.

        Returns the matched Bot (or None), and a `mentioned` boolean
        for callers that just need a yes/no signal."""
        # Pull the candidate bot from the dispatcher's resolver. If
        # bot_registry isn't wired or the registry returns nothing
        # (early boot, tests), fall back to legacy bot_name matching
        # so this code stays useful in single-bot setups too.
        active_bot = None
        if self.dispatcher is not None:
            active_bot = self.dispatcher._resolve_bot(group_id, policy=policy)

        # 1) Structured @-mention — phone/UUID match means the message
        # was directed at a Signal account we recognize. With multi-
        # phone, the mention's target phone uniquely identifies the bot
        # the user wants. Look it up globally (not just against this
        # handler's phone) so the SAME mention resolves to the SAME bot
        # in every handler; the downstream multi-phone filter then
        # decides which handler actually dispatches. This is how a user
        # @-mentions Artaud's number in a Sigil-pinned context to
        # summon Artaud anyway.
        mentions = data_message.get("mentions") or []
        if mentions:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            targets_self = False
            for mention in mentions:
                mb = self._resolve_mention_bot(mention)
                if mb is not None:
                    return mb, True
                if self._mention_targets_self(mention):
                    targets_self = True
            # The mention named this handler's account but it serves
            # several bots (shared-phone) — can't disambiguate, so the
            # pinned bot fields it. (Multi-phone never reaches here: the
            # served-bot resolution above handles it.)
            if targets_self and active_bot is not None:
                return active_bot, True
            # Mentions present but none mapped to a bot we serve — log the
            # raw ids so a misrouted tag is debuggable instead of silent.
            logger.info(
                "Mention(s) matched no bot: %s",
                [self._mention_ids(m) for m in mentions],
            )

        # 2) Typed-name addressing. A bot is ADDRESSED when its name LEADS
        # the message (a vocative): "Sigil, …", "hey Artaud …", or a
        # compound "Sigil and Artaud, …". A name buried later as a TOPIC
        # ("what do you think of Artaud?") must NOT summon it — that was
        # the old bug: matching a name anywhere + biasing to the pinned
        # bot meant the bot being talked ABOUT answered instead of the
        # one being talked TO. Primary = the first-addressed (left-most)
        # bot. When a name appears only as a topic, the pinned/active bot
        # fields the reply (the topic mention doesn't redirect it).
        message_text = data_message.get("message") or ""
        bot_registry = getattr(self.dispatcher, "bot_registry", None)
        if message_text and bot_registry is not None:
            all_bots = [
                b for b in bot_registry.list_sync()
                if getattr(b, "enabled", True)
            ]
            addressed = self._addressed_bots(message_text, all_bots)
            if addressed:
                return addressed[0], True
            # Name present only as a topic → engage, but the pinned bot
            # answers; the bot being discussed isn't the addressee.
            if active_bot is not None and self._mentions_any_alias(
                message_text, all_bots
            ):
                return active_bot, True
        elif active_bot is not None and message_text:
            # Registry not wired (early-boot / tests) — single-bot path.
            # Leading vs. topic doesn't matter when there's one bot: any
            # mention of its name engages it.
            if self._mentions_any_alias(message_text, [active_bot]):
                return active_bot, True
        elif active_bot is None and message_text:
            # Pre-PR4 single-bot fallback when neither registry nor
            # active bot is available.
            bot_name = (getattr(self.dispatcher, "bot_name", "") or "").strip()
            if bot_name and re.search(
                rf"\b{re.escape(bot_name)}\b", message_text, re.IGNORECASE
            ):
                return None, True

        # 3) Quote-reply to a previous bot message.
        quote = data_message.get("quote") or {}
        quote_author_number = quote.get("authorNumber") or ""
        quote_author_uuid = quote.get("authorUuid") or quote.get("author") or ""
        if quote_author_number or quote_author_uuid:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            # Multi-phone: the quoted author's phone names a specific
            # bot. Look it up globally (not just against this handler's
            # phone) so the SAME quote resolves to the SAME bot in every
            # handler — the downstream multi-phone filter then picks who
            # actually answers. Without this, quoting bot B in a context
            # where bot A is the active default either summons A
            # (single-phone) or silently drops because no handler
            # recognized B's phone as its own (multi-phone).
            quoted_bot = self._lookup_bot_by_phone(quote_author_number)
            if quoted_bot is not None:
                return quoted_bot, True
            # Shared-phone fallback: quote author matches this handler's
            # own phone/UUID but we can't tell which persona wrote it —
            # defer to the active bot for the context.
            if quote_author_number and quote_author_number == self.config.phone_number:
                return active_bot, True
            if self._bot_uuid and quote_author_uuid == self._bot_uuid:
                return active_bot, True

        return None, False

    async def _resolve_addressed_bot_set(
        self, data_message: dict, group_id: Optional[str], policy
    ) -> list:
        """Every bot a message EXPLICITLY addresses, in priority order.

        Companion to `_resolve_addressed_bot` (which returns just the
        primary). This collects the full set so handle_webhook can fan out
        when a single message names more than one bot — e.g. "Sigil and
        Artaud, thoughts?" — and have each named bot reply in its own voice
        instead of only the pinned/first one.

        Resolution rule ("mentions, leading names as fallback"):
          1. Structured @-mentions → every mention whose phone/UUID maps to
             a bot. If that yields >=1 bot, it IS the set (typed names are
             not consulted — the user tapped specific accounts).
          2. Only when no bot was @-mentioned: LEADING-vocative typed names
             (see `_addressed_bots` — leading or trailing), in order. A name
             buried later as a topic is ignored, so "Sigil, what about
             Artaud?" addresses only Sigil — it never fans out to the bot
             being talked about.

        Quote-reply targeting is intentionally excluded — a quote addresses
        one prior message and is handled by the single-bot path. Returns a
        list deduped by bot id (empty when nothing is explicitly addressed).
        """
        active_bot = None
        if self.dispatcher is not None and hasattr(self.dispatcher, "_resolve_bot"):
            try:
                active_bot = self.dispatcher._resolve_bot(group_id, policy=policy)
            except Exception:
                active_bot = None

        found: list = []

        def _add(bot) -> None:
            if bot is None:
                return
            bid = getattr(bot, "id", None)
            if all(getattr(b, "id", None) != bid for b in found):
                found.append(bot)

        # 1) Structured @-mentions — resolve each to the bot it names
        # (phone or UUID, globally), returning the served bot, never the
        # pinned default. See `_resolve_mention_bot`.
        mentions = data_message.get("mentions") or []
        if mentions:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            targets_self = False
            for mention in mentions:
                mb = self._resolve_mention_bot(mention)
                if mb is not None:
                    _add(mb)
                elif self._mention_targets_self(mention):
                    targets_self = True
            if not found and targets_self and active_bot is not None:
                _add(active_bot)
        if found:
            return found

        # 2) Fallback: typed aliases (only when no bot was @-mentioned).
        message_text = data_message.get("message") or ""
        bot_registry = getattr(self.dispatcher, "bot_registry", None)
        if message_text and bot_registry is not None:
            all_bots = [
                b for b in bot_registry.list_sync()
                if getattr(b, "enabled", True)
            ]
            # Only vocative names count (leading or trailing), in order of
            # appearance — so "Sigil and Artaud, …" fans out to both, but
            # "Sigil, what about Artaud?" addresses only Sigil (Artaud is
            # the topic).
            for bot in self._addressed_bots(message_text, all_bots):
                _add(bot)
        return found

    async def handle_webhook(self, data: dict):
        """
        Handle incoming webhook from signal-cli-rest-api.

        Parses the webhook payload, extracts message info,
        dispatches to command handler, and sends response.
        """
        envelope = data.get("envelope", {})
        sender = envelope.get("source")
        target_timestamp = envelope.get("timestamp")

        # Handle data message
        data_message = envelope.get("dataMessage", {})
        message_text = data_message.get("message", "")
        # dataMessage.timestamp is the canonical "message id" — prefer it when
        # present; otherwise fall back to the envelope timestamp.
        message_ts = data_message.get("timestamp") or target_timestamp

        # Polls arrive with `pollCreate` populated and `message` = null. The
        # bot's normal command path can't reach them, so we branch off into a
        # dedicated path that calls the LLM to choose option(s) and casts a
        # vote. Don't return immediately though — a `pollCreate` envelope
        # doesn't carry text to dispatch, so falling through to the normal
        # path is fine (it'll be filtered by the empty-text check below).
        if data_message.get("pollCreate") and self.poll_voter is not None:
            try:
                asyncio.create_task(
                    self.poll_voter.handle_poll(envelope, data_message)
                )
            except Exception as e:
                logger.error(f"Poll handler launch failed: {e}")

        # Skip empty messages or non-text messages
        if not sender or not message_text:
            logger.debug("Skipping message: no sender or empty text")
            return

        # Self-message guard: drop envelopes whose sender is any phone
        # OR UUID this pool serves. Three flavors:
        #   * Our own outbound echo (signal-cli has been observed to
        #     replay outbound on reconnect; without the guard the
        #     dispatcher would treat it as user input and could loop).
        #   * The OTHER bot's outbound. With multi-phone, Sigil's
        #     handler receives Artaud's group messages as regular
        #     "user" input from `+16467699190`. Treating bot-on-bot
        #     traffic as user input would let two bots have a
        #     conversation with each other (token burn, chat noise).
        #   * UUID-form sender. signal-cli sometimes populates
        #     `envelope.source` with a UUID rather than a phone (the
        #     post-PNI Signal account state, or contacts that haven't
        #     been resolved to a number on this account yet). The
        #     phone-only checks above silently fail in that case, so we
        #     also match against the pool-shared `_known_uuids` set —
        #     populated eagerly at startup by
        #     `SignalHandlerPool.bootstrap_uuids()` and lazily on every
        #     `fetch_bot_uuid()` success.
        source_uuid = (envelope.get("sourceUuid") or "").strip()
        is_bot_echo = (
            sender == self.config.phone_number
            or sender in self._known_phones
            or (source_uuid and (
                source_uuid == self._bot_uuid
                or source_uuid in self._known_uuids
            ))
        )
        if is_bot_echo:
            tail = (sender or source_uuid or "????")[-4:]
            logger.debug(f"Skipping bot-side echo from ...{tail}")
            return

        # Idempotency: signal-cli replays buffered messages on websocket
        # reconnect. Without this, a transient disconnect during an
        # `!ask` causes the same prompt to be processed twice — the user
        # sees two responses and the LLM history records two turns.
        # Dedup on (sender, dataMessage.timestamp) with a 60s TTL.
        if message_ts:
            key = (sender, message_ts)
            now = time.time()
            if key in self._seen_messages and now - self._seen_messages[key] < 60.0:
                logger.debug(f"Skipping duplicate message {key}")
                return
            self._seen_messages[key] = now
            # Bounded LRU-ish: when full, drop the oldest 10% in one pass.
            if len(self._seen_messages) > self._seen_messages_max:
                cutoff = sorted(self._seen_messages.values())[
                    self._seen_messages_max // 10
                ]
                self._seen_messages = {
                    k: v for k, v in self._seen_messages.items() if v >= cutoff
                }

        # Extract group info if present
        group_info = data_message.get("groupInfo")
        group_id = None
        if group_info:
            group_id = group_info.get("groupId")

        # Quoted-message info — present when the user is replying to another
        # message. signal-cli puts the original text/author here so the LLM
        # can see what's being replied to without us looking it up.
        quote = data_message.get("quote") or {}
        quote_text = (quote.get("text") or "").strip() or None
        quote_author = (
            quote.get("authorNumber")
            or quote.get("author")
            or None
        )

        # Resolve the addressed bot before dispatch. The dispatcher
        # uses `_resolve_bot` to pick a default; what mention-routing
        # adds is the alias path — when the user says "Artaud" in an
        # Artaud-pinned context that fact is already self-consistent,
        # but in PR5's shared-context scenario this is what lets the
        # right bot answer.
        policy_for_routing = None
        if self.dispatcher is not None and self.dispatcher.context_registry is not None:
            try:
                policy_for_routing = await self.dispatcher.context_registry.resolve(
                    group_id, sender
                )
            except Exception as e:
                logger.debug(f"context resolve for mention routing failed: {e}")
        addressed_bot, is_mentioned = await self._resolve_addressed_bot(
            data_message, group_id, policy_for_routing
        )

        # Multi-phone routing with takeover. Each registered phone runs
        # its own SignalHandler/poller; signal-cli usually delivers a
        # copy of every envelope to every linked account in the group,
        # but we've observed it occasionally miss delivery to one phone
        # (Signal session/key gap on the sender side). The old behavior
        # was to silently drop here when the addressed bot lives on
        # another phone and trust the OTHER handler to pick it up — if
        # that handler's poller missed delivery, the message was lost.
        #
        # New behavior: claim the envelope through the pool. Owner-of-
        # record handler claims at t=0. Non-owner sleeps
        # POOL_TAKEOVER_DELAY_SEC and then tries to claim — if the owner
        # actually received its copy it has already claimed and the
        # non-owner drops; otherwise the non-owner takes over and
        # dispatches as the addressed bot (response goes out via
        # `pool.for_bot(effective_bot)` below so it still comes from
        # the right phone).
        effective_bot = None
        took_over = False
        if (
            self.dispatcher is not None
            and self.dispatcher.bot_registry is not None
        ):
            effective_bot = self.dispatcher._resolve_bot(
                group_id, policy=policy_for_routing, addressed_bot=addressed_bot,
            )
            pool = getattr(self.dispatcher, "signal_pool", None)
            claim_key = self._build_claim_key(
                envelope, data_message, group_id,
            )
            if claim_key is not None and pool is not None and hasattr(pool, "claim"):
                if self._owns_bot(effective_bot):
                    if not pool.claim(claim_key):
                        logger.debug(
                            f"Handler ...{self.config.phone_number[-4:]}: "
                            f"sibling already claimed {claim_key!r}"
                        )
                        return
                else:
                    # Wait for the owner to claim. If they don't within
                    # the grace period, the owner's poller likely missed
                    # delivery — take over.
                    try:
                        await asyncio.sleep(POOL_TAKEOVER_DELAY_SEC)
                    except asyncio.CancelledError:
                        raise
                    if not pool.claim(claim_key):
                        logger.debug(
                            f"Handler ...{self.config.phone_number[-4:]}: "
                            f"owner claimed during takeover wait "
                            f"{claim_key!r}"
                        )
                        return
                    bot_slug = (
                        effective_bot.slug if effective_bot is not None else None
                    )
                    logger.warning(
                        f"Handler ...{self.config.phone_number[-4:]}: taking "
                        f"over for bot={bot_slug!r} after "
                        f"{POOL_TAKEOVER_DELAY_SEC}s — owner's poller "
                        f"never claimed (signal-cli delivery gap)"
                    )
                    took_over = True
            elif not self._owns_bot(effective_bot):
                # No pool or no claim key (degenerate envelope): fall back
                # to the legacy drop-silently behavior so we don't double-
                # respond on the common case where both handlers have a
                # copy.
                logger.debug(
                    f"Handler ...{self.config.phone_number[-4:]}: "
                    f"dropping envelope routed to bot="
                    f"{effective_bot.slug if effective_bot else None!r} "
                    f"(no pool/claim key available)"
                )
                return

        # Pull inbound images when the bot answering this message has
        # vision_enabled. Bound to the resolved bot (not all bots) so
        # vision payload is paid for only when the receiving model can
        # actually consume it. When no bot is resolved yet — falls back
        # to the registry's default-for-kind via the dispatcher's
        # _resolve_bot — we still pre-extract so the dispatcher can use
        # them once it knows which bot will answer. Cheap: skipped
        # entirely when the message has no attachments at all.
        inbound_images: list[dict] = []
        vision_candidate = addressed_bot
        if (
            vision_candidate is None
            and self.dispatcher is not None
            and self.dispatcher.bot_registry is not None
        ):
            kind = "group" if group_id else "dm"
            try:
                vision_candidate = self.dispatcher.bot_registry.default_for_kind_sync(kind)
            except Exception as e:
                logger.debug(f"vision: default bot lookup failed: {e}")
        if (
            vision_candidate is not None
            and getattr(vision_candidate, "vision_enabled", False)
            and data_message.get("attachments")
        ):
            inbound_images = _read_inbound_image_attachments(data_message)
            if inbound_images:
                logger.info(
                    f"Inbound vision: {len(inbound_images)} image(s) for "
                    f"bot={vision_candidate.slug}"
                )

        bot_label = f" [→{addressed_bot.slug}]" if addressed_bot else ""
        img_label = f" [+{len(inbound_images)} img]" if inbound_images else ""
        takeover_label = " [takeover]" if took_over else ""
        logger.info(
            f"Received message from {sender[-4:]}:{takeover_label} "
            f"{'[group] ' if group_id else ''}"
            f"{'[@mentioned]' + bot_label + ' ' if is_mentioned else ''}"
            f"{img_label} "
            f"{message_text[:50]}..."
        )

        # Dispatch to command handler. Policy already resolved for
        # mention routing — pass it down so the dispatcher reuses it
        # rather than running the same SQLite query a second time.
        result = await self.dispatcher.dispatch(
            sender=sender,
            message=message_text,
            group_id=group_id,
            mentioned=is_mentioned,
            target_timestamp=message_ts,
            quote_text=quote_text,
            quote_author=quote_author,
            addressed_bot=addressed_bot,
            policy=policy_for_routing,
            inbound_images=inbound_images,
        )

        # Send response if command was processed. Route through the
        # answering bot's own handler so multi-phone installs always
        # send from the bot's number — required when we took over an
        # envelope for a bot on another phone, but harmless in the
        # common (owner-self) case where `for_bot` returns `self`.
        if result:
            response_handler = self
            if effective_bot is not None:
                pool = getattr(self.dispatcher, "signal_pool", None)
                if pool is not None and hasattr(pool, "for_bot"):
                    try:
                        candidate = pool.for_bot(effective_bot)
                        if candidate is not None:
                            response_handler = candidate
                    except Exception as e:
                        logger.debug(f"pool.for_bot fallback to self: {e}")
            try:
                # If dm_only, send directly to user regardless of group context
                target_group = None if result.dm_only else group_id
                await response_handler.send_message(
                    recipient=sender,
                    message=result.text,
                    group_id=target_group,
                    attachments=result.attachments,
                    styled=result.styled,
                )
            except Exception as e:
                logger.error(f"Failed to send response: {e}")
                # Fallback: try sending directly to user if group send failed
                if group_id:
                    try:
                        logger.info(f"Attempting fallback DM to {sender[-4:]}")
                        await self.send_message(
                            recipient=sender,
                            message=f"{result.text}\n\n(Replied privately due to group send error)",
                            group_id=None,
                            attachments=result.attachments,
                            styled=result.styled,
                        )
                    except Exception as fallback_e:
                        logger.error(f"Fallback DM failed: {fallback_e}")

        # Multi-bot fan-out: when ONE message explicitly addresses more
        # than one bot ("@Sigil @Artaud thoughts?"), the block above
        # answered as the PRIMARY bot only. Trigger the other addressed
        # bots here so each replies in its own voice. This runs only on
        # the handler that won the per-message claim (the others returned
        # early), so one handler orchestrates every reply and routes each
        # through its own bot's phone via the pool — no cross-handler
        # coordination, and the shared writes (inbound group-log line,
        # user history turn) stay single-copy because the secondaries
        # skip them.
        await self._fanout_secondary_bots(
            data_message=data_message,
            group_id=group_id,
            policy=policy_for_routing,
            sender=sender,
            message_text=message_text,
            quote_text=quote_text,
            quote_author=quote_author,
            primary_bot=effective_bot,
        )

    async def _fanout_secondary_bots(
        self, *, data_message, group_id, policy, sender, message_text,
        quote_text, quote_author, primary_bot,
    ) -> None:
        """Fire replies from every addressed bot OTHER than the primary.

        No-op unless this is a group chat in conversational (llm_intent)
        mode and the message explicitly addresses >=2 bots. Plain commands
        (`!price ...`) are skipped — multi-bot addressing is a chat
        affordance, not a command one. Each secondary answer is dispatched
        as a background task so the bots reply in parallel.
        """
        if not group_id or self.dispatcher is None:
            return
        ask_command = getattr(self.dispatcher, "ask_command", None)
        if ask_command is None:
            return
        if policy is None or not getattr(policy, "llm_intent", False):
            return
        prefix = getattr(self.dispatcher, "prefix", "!")
        if message_text.strip().startswith(prefix):
            return  # explicit command — single bot handles it
        try:
            addressed = await self._resolve_addressed_bot_set(
                data_message, group_id, policy,
            )
        except Exception as e:
            logger.debug(f"multi-bot set resolve failed: {e}")
            return
        primary_id = getattr(primary_bot, "id", None)
        secondaries = [
            b for b in addressed if getattr(b, "id", None) != primary_id
        ]
        if not secondaries:
            return
        # Drop @-mention tokens; each bot reads the remaining text.
        cleaned = re.sub(r"@\S+\s*", "", message_text).strip() or message_text.strip()
        if not cleaned:
            return
        pool = getattr(self.dispatcher, "signal_pool", None)
        for bot in secondaries:
            logger.info(
                f"Multi-bot fan-out: also answering as {bot.slug} "
                f"(primary={getattr(primary_bot, 'slug', None)})"
            )
            asyncio.create_task(
                self._answer_secondary(
                    bot=bot, ask_command=ask_command, pool=pool,
                    sender=sender, group_id=group_id, policy=policy,
                    cleaned=cleaned, quote_text=quote_text,
                    quote_author=quote_author,
                )
            )

    async def _answer_secondary(
        self, *, bot, ask_command, pool, sender, group_id, policy, cleaned,
        quote_text, quote_author,
    ) -> None:
        """Produce + send one secondary bot's reply in a multi-bot fan-out.

        Bypasses dispatcher.dispatch on purpose: the primary already logged
        the inbound message to the group log and stored the shared user
        turn. Calls AskCommand directly with `persist_user_turn=False` so
        the secondary stores only its OWN assistant turn, then sends from
        the secondary bot's phone via `pool.for_bot`.
        """
        from ..commands.base import CommandContext
        try:
            ctx = CommandContext(
                sender=sender,
                group_id=group_id,
                raw_message=f"{getattr(self.dispatcher, 'prefix', '!')}ask {cleaned}",
                command="ask",
                args=cleaned.split(),
                policy=policy,
                bot=bot,
                quote_text=quote_text,
                quote_author=quote_author,
                persist_user_turn=False,
            )
            result = await ask_command.execute(ctx)
        except Exception as e:
            logger.error(
                f"Multi-bot fan-out: ask failed for {getattr(bot, 'slug', None)}: {e}"
            )
            return
        if not result or not getattr(result, "text", None):
            return
        response_handler = self
        if pool is not None and hasattr(pool, "for_bot"):
            try:
                candidate = pool.for_bot(bot)
                if candidate is not None:
                    response_handler = candidate
            except Exception as e:
                logger.debug(f"fan-out for_bot fallback to self: {e}")
        try:
            target_group = None if result.dm_only else group_id
            await response_handler.send_message(
                recipient=sender,
                message=result.text,
                group_id=target_group,
                attachments=result.attachments,
                styled=result.styled,
            )
        except Exception as e:
            logger.error(
                f"Multi-bot fan-out send failed for {getattr(bot, 'slug', None)}: {e}"
            )

    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()


class _TypingIndicator:
    """Context manager that pings the typing indicator on a refresh loop."""

    def __init__(
        self,
        handler: "SignalHandler",
        recipient: str,
        group_id: Optional[str],
        refresh_interval: float,
    ):
        self.handler = handler
        self.recipient = recipient
        self.group_id = group_id
        self.refresh_interval = refresh_interval
        self._task: Optional[asyncio.Task] = None

    async def __aenter__(self):
        self._task = asyncio.create_task(self._loop())
        return self

    async def __aexit__(self, exc_type, exc, tb):
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            except Exception as e:
                logger.debug(f"Typing indicator loop teardown: {e}")
        # Best-effort stop. If this fails the indicator clears itself in
        # ~15 seconds anyway.
        await self.handler.send_typing(self.recipient, self.group_id, stop=True)

    async def _loop(self):
        try:
            while True:
                await self.handler.send_typing(self.recipient, self.group_id)
                await asyncio.sleep(self.refresh_interval)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.debug(f"Typing indicator loop error: {e}")

