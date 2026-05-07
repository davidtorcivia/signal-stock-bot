"""
Signal message handler - interfaces with signal-cli-rest-api.
"""

import asyncio
import logging
import re
import time
from dataclasses import dataclass
from typing import Optional

import aiohttp

from ..admin.events import get_bus
from ..commands.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)


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

    async def fetch_bot_uuid(self) -> Optional[str]:
        """Fetch and cache the bot's UUID from signal-cli API."""
        if self._bot_uuid:
            return self._bot_uuid
        
        try:
            session = await self._get_session()
            url = f"{self.config.api_url}/v1/about"
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    # Check if our phone's info is available
                    for account in data if isinstance(data, list) else [data]:
                        if account.get("number") == self.config.phone_number:
                            self._bot_uuid = account.get("uuid")
                            if self._bot_uuid:
                                logger.info(f"Fetched bot UUID: {self._bot_uuid[:8]}...")
                            return self._bot_uuid
        except Exception as e:
            logger.debug(f"Could not fetch bot UUID: {e}")
        
        return None
    
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
          3. Quote-reply to one of our previous messages (whoever the
             phone says it was). When phone is shared and we can't tell
             which bot wrote the quoted message, the active bot for the
             context is the safe pick.

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
        # was directed at this Signal account. With a shared phone the
        # mention can't disambiguate Sigil-vs-Artaud, so we treat it as
        # "addressing the active bot" rather than failing closed.
        mentions = data_message.get("mentions") or []
        if mentions:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            for mention in mentions:
                if mention.get("number", "") == self.config.phone_number:
                    return active_bot, True
                mentioned_uuid = mention.get("uuid", "")
                if self._bot_uuid and mentioned_uuid == self._bot_uuid:
                    return active_bot, True

        # 2) Alias match against the active bot's alias set.
        message_text = data_message.get("message") or ""
        if active_bot is not None and message_text:
            for alias in active_bot.alias_set():
                if re.search(rf"\b{re.escape(alias)}\b", message_text, re.IGNORECASE):
                    return active_bot, True
        elif active_bot is None and message_text:
            # Legacy fallback when bot_registry isn't wired: use the
            # dispatcher's bot_name. Behaves identically to pre-PR4.
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
            if quote_author_number and quote_author_number == self.config.phone_number:
                return active_bot, True
            if self._bot_uuid and quote_author_uuid == self._bot_uuid:
                return active_bot, True

        return None, False

    async def _is_bot_mentioned(self, data_message: dict) -> bool:
        """Bool-only wrapper for callers that don't need bot identity.
        Defers to `_resolve_addressed_bot` with no context info — used
        by legacy paths that route mention=True/False without
        propagating which bot."""
        _, mentioned = await self._resolve_addressed_bot(
            data_message, group_id=None, policy=None
        )
        return mentioned
    
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

        # Self-message guard: if signal-cli ever echoes our own outbound
        # message back through the receive endpoint (has happened on
        # certain reconnect paths), the dispatcher would treat it as
        # input and could loop. Drop it before any further work.
        if sender == self.config.phone_number:
            logger.debug("Skipping own outbound echo")
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

        bot_label = f" [→{addressed_bot.slug}]" if addressed_bot else ""
        logger.info(
            f"Received message from {sender[-4:]}: "
            f"{'[group] ' if group_id else ''}"
            f"{'[@mentioned]' + bot_label + ' ' if is_mentioned else ''}"
            f"{message_text[:50]}..."
        )

        # Dispatch to command handler
        result = await self.dispatcher.dispatch(
            sender=sender,
            message=message_text,
            group_id=group_id,
            mentioned=is_mentioned,
            target_timestamp=message_ts,
            quote_text=quote_text,
            quote_author=quote_author,
            addressed_bot=addressed_bot,
        )
        
        # Send response if command was processed
        if result:
            try:
                # If dm_only, send directly to user regardless of group context
                target_group = None if result.dm_only else group_id
                await self.send_message(
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

