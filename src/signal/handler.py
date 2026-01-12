"""
Signal message handler - interfaces with signal-cli-rest-api.
"""

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
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def send_message(
        self,
        recipient: str,
        message: str,
        group_id: Optional[str] = None,
        attachments: Optional[list[str]] = None
    ):
        """
        Send a message to a recipient or group.
        
        Args:
            recipient: Phone number or group ID
            message: Message text
            group_id: If set, sends to this group instead of recipient
            attachments: Optional list of base64-encoded images
        """
        session = await self._get_session()
        
        # Build payload for v2 API
        payload = {
            "number": self.config.phone_number,
            "message": message,
        }
        
        if group_id:
            # For groups, use group_id directly
            payload["recipients"] = [group_id]
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
                    logger.error(f"Failed to send message: {resp.status} - {error}")
                    raise Exception(f"Send failed: {resp.status}")
                
                logger.debug(f"Message sent successfully to {recipient[-4:] if recipient else group_id}")
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error sending message: {e}")
            raise
    
    def _is_bot_mentioned(self, data_message: dict) -> bool:
        """
        Check if the bot is mentioned in the message.
        
        Signal mentions include the phone number or UUID of mentioned users.
        We check if any mention matches our bot's phone number.
        """
        mentions = data_message.get("mentions", [])
        if not mentions:
            return False
        
        # Check each mention - signal-cli provides the mentioned number/uuid
        for mention in mentions:
            # The mention might have a 'number' field matching our phone
            mentioned_number = mention.get("number", "")
            if mentioned_number == self.config.phone_number:
                return True
            
            # Also check by UUID if we have one stored (future enhancement)
            # For now, any mention in a group triggers if we can't identify
            # This is a simple heuristic - in practice, you'd want to verify
        
        # If there are mentions but we can't verify, check if message
        # starts with @ followed by text (common pattern)
        return len(mentions) > 0
    
    async def handle_webhook(self, data: dict):
        """
        Handle incoming webhook from signal-cli-rest-api.
        
        Parses the webhook payload, extracts message info,
        dispatches to command handler, and sends response.
        """
        envelope = data.get("envelope", {})
        sender = envelope.get("source")
        
        # Handle data message
        data_message = envelope.get("dataMessage", {})
        message_text = data_message.get("message", "")
        
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
        is_mentioned = self._is_bot_mentioned(data_message)
        
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
        )
        
        # Send response if command was processed
        if result:
            try:
                await self.send_message(
                    recipient=sender,
                    message=result.text,
                    group_id=group_id,
                    attachments=result.attachments,
                )
            except Exception as e:
                logger.error(f"Failed to send response: {e}")
    
    async def close(self):
        """Close the HTTP session"""
        if self._session and not self._session.closed:
            await self._session.close()

