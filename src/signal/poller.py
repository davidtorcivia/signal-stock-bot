"""
Message poller for signal-cli-rest-api.

Fallback mechanism when webhooks are not working in json-rpc mode.
Periodically polls the receive endpoint and forwards messages to the handler.
"""

import asyncio
import logging
import threading
import time
from typing import Callable, Optional

import aiohttp

logger = logging.getLogger(__name__)


class SignalPoller:
    """
    Polls signal-cli-rest-api for incoming messages.
    
    This is a fallback for when RECEIVE_WEBHOOK_URL doesn't work.
    Runs in a background thread and calls the message handler for each message.
    """
    
    def __init__(
        self,
        api_url: str,
        phone_number: str,
        on_message: Callable[[dict], None],
        poll_interval: float = 1.0,
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
        
        async with aiohttp.ClientSession() as session:
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
        async with session.get(url, timeout=aiohttp.ClientTimeout(total=10)) as resp:
            if resp.status == 200:
                messages = await resp.json()
                
                if messages:
                    logger.debug(f"Received {len(messages)} message(s) via polling")
                    
                    for msg in messages:
                        try:
                            # Convert to webhook format if needed
                            webhook_data = self._convert_to_webhook_format(msg)
                            if webhook_data:
                                await self.on_message(webhook_data)
                        except Exception as e:
                            logger.error(f"Error processing polled message: {e}")
            
            elif resp.status == 204:
                # No new messages
                pass
            else:
                text = await resp.text()
                logger.debug(f"Poll returned {resp.status}: {text[:100]}")
    
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
        
        # Unknown format
        logger.warning(f"Unknown message format: {list(msg.keys())}")
        return None
