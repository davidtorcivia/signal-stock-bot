"""
Signal message handler - interfaces with signal-cli-rest-api.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Optional

import aiohttp

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
        
        Signal mentions include the phone number or UUID of mentioned users.
        We check if any mention matches our bot's phone number or UUID.
        """
        mentions = data_message.get("mentions", [])
        if not mentions:
            return False
        
        # Ensure we have our UUID for matching
        if not self._bot_uuid:
            await self.fetch_bot_uuid()
        
        # Check each mention - signal-cli provides the mentioned number/uuid
        for mention in mentions:
            # Check phone number match
            mentioned_number = mention.get("number", "")
            if mentioned_number == self.config.phone_number:
                return True
            
            # Check UUID match
            mentioned_uuid = mention.get("uuid", "")
            if self._bot_uuid and mentioned_uuid == self._bot_uuid:
                return True
        
        # No matching mention found
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

        # Skip empty messages or non-text messages
        if not sender or not message_text:
            logger.debug("Skipping message: no sender or empty text")
            return

        # Extract group info if present
        group_info = data_message.get("groupInfo")
        group_id = None
        if group_info:
            group_id = group_info.get("groupId")

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

