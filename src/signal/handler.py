"""
Signal message handler - interfaces with signal-cli-rest-api.
"""

import asyncio
import logging
import re
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
                    raise Exception(f"Send failed: {resp.status}")
                
                logger.debug(f"Message sent successfully to {recipient[-4:] if recipient else group_id}")
                
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

        session = await self._get_session()
        payload = {
            "poll_author": poll_author,
            "poll_timestamp": str(int(poll_timestamp)),
            "recipient": target,
            "selected_answers": list(selected_answers),
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
    
    async def _is_bot_mentioned(self, data_message: dict) -> bool:
        """
        Check if the bot is mentioned in the message.

        Three paths count as a mention:
          1. Signal's structured @-mention (matched by phone or UUID).
          2. The configured bot_name appearing as a whole word in the
             message text — e.g. "hey Sigil, what's AAPL?". Whole-word,
             case-insensitive so casual references like "Sigil!" or
             "Sigil," still match without grabbing substrings.
          3. A quote-reply to one of the bot's own messages. If a user
             swipes-to-reply on something the bot said, that's a direct
             address even without an @-mention or name mention — treat
             it as if the bot were mentioned so conversation flows
             naturally. Without this, follow-ups like "expand on that"
             or "are you sure?" would silently fall to the floor in a
             group chat.
        """
        # 1) Structured @-mention
        mentions = data_message.get("mentions") or []
        if mentions:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            for mention in mentions:
                if mention.get("number", "") == self.config.phone_number:
                    return True
                mentioned_uuid = mention.get("uuid", "")
                if self._bot_uuid and mentioned_uuid == self._bot_uuid:
                    return True

        # 2) Plain-text name reference. Pull the live bot_name off the
        # dispatcher; it refreshes from settings on every dispatch, so
        # the value is at most one message stale after an admin rename.
        bot_name = (getattr(self.dispatcher, "bot_name", "") or "").strip()
        message_text = data_message.get("message") or ""
        if bot_name and message_text:
            if re.search(
                rf"\b{re.escape(bot_name)}\b", message_text, re.IGNORECASE
            ):
                return True

        # 3) Quote-reply to a bot message
        quote = data_message.get("quote") or {}
        quote_author_number = quote.get("authorNumber") or ""
        quote_author_uuid = quote.get("authorUuid") or quote.get("author") or ""
        if quote_author_number or quote_author_uuid:
            if not self._bot_uuid:
                await self.fetch_bot_uuid()
            if quote_author_number and quote_author_number == self.config.phone_number:
                return True
            if self._bot_uuid and quote_author_uuid == self._bot_uuid:
                return True

        return False
    
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

        # Check if bot is mentioned
        is_mentioned = await self._is_bot_mentioned(data_message)

        logger.info(
            f"Received message from {sender[-4:]}: "
            f"{'[group] ' if group_id else ''}"
            f"{'[@mentioned] ' if is_mentioned else ''}"
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

