"""
Message poller for signal-cli-rest-api in native mode.

Polls the /v1/receive endpoint to fetch incoming messages
and forwards them to the message handler.
"""

import asyncio
import logging
import threading
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SignalPoller:
    """
    Polls signal-cli-rest-api for incoming messages.
    
    In native mode, the /v1/receive/<number> endpoint returns
    any pending messages. This class polls that endpoint and
    forwards messages to the handler.
    """
    
    def __init__(
        self,
        api_url: str,
        phone_number: str,
        on_message: Callable[[dict], None],
        poll_interval: float = 2.0,
        jsonrpc_port: int = 6001,  # Unused in native mode, kept for compatibility
    ):
        """
        Initialize the poller.
        
        Args:
            api_url: Base URL of signal-cli-rest-api (e.g., http://signal-api:8080)
            phone_number: Bot's phone number for receiving messages
            on_message: Async callback function to handle incoming messages
            poll_interval: Seconds between poll attempts
        """
        self.api_url = api_url.rstrip("/")
        self.phone_number = phone_number
        self.on_message = on_message
        self.poll_interval = poll_interval
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the poller in a background thread."""
        if self._running:
            logger.warning("Poller already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Signal poller started (interval: {self.poll_interval}s)")
    
    def stop(self):
        """Stop the poller."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Signal poller stopped")
    
    def _run_loop(self):
        """Run the async event loop in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._poll_loop())
        except Exception as e:
            logger.error(f"Poller loop error: {e}")
        finally:
            loop.close()
    
    async def _poll_loop(self):
        """Main polling loop."""
        # URL encode the phone number (+ becomes %2B)
        encoded_number = self.phone_number.replace("+", "%2B")
        receive_url = f"{self.api_url}/v1/receive/{encoded_number}"
        
        logger.info(f"Polling endpoint: {receive_url}")
        
        timeout = aiohttp.ClientTimeout(total=30)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while self._running:
                try:
                    await self._poll_once(session, receive_url)
                except aiohttp.ClientError as e:
                    logger.debug(f"Poll request failed: {e}")
                except Exception as e:
                    logger.error(f"Poll error: {e}")
                
                await asyncio.sleep(self.poll_interval)
    
    async def _poll_once(self, session: aiohttp.ClientSession, url: str):
        """Perform a single poll request."""
        async with session.get(url) as resp:
            if resp.status == 200:
                messages = await resp.json()
                
                if messages:
                    logger.info(f"Received {len(messages)} message(s) via polling")
                    
                    for msg in messages:
                        try:
                            # Convert to webhook format
                            webhook_data = self._convert_to_webhook_format(msg)
                            if webhook_data:
                                await self.on_message(webhook_data)
                        except Exception as e:
                            logger.error(f"Error processing polled message: {e}")
            
            elif resp.status == 204:
                # No new messages - this is normal
                pass
            else:
                text = await resp.text()
                logger.warning(f"Poll returned {resp.status}: {text[:200]}")
    
    def _convert_to_webhook_format(self, msg: dict) -> Optional[dict]:
        """
        Convert signal-cli receive format to webhook envelope format.
        
        The receive endpoint returns messages in a slightly different format
        than webhooks. This normalizes them.
        """
        # Already in envelope format
        if "envelope" in msg:
            return msg
        
        # Raw message format - wrap in envelope
        if "source" in msg or "sourceNumber" in msg:
            return {"envelope": msg}
        
        # Unknown format - log and return as-is wrapped
        logger.debug(f"Unknown message format, wrapping: {list(msg.keys())}")
        return {"envelope": msg}
