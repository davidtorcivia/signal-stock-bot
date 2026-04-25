"""
WebSocket listener for signal-cli-rest-api in json-rpc mode.

Connects to the WebSocket endpoint /v1/receive/<number> to receive
incoming messages in real-time.
"""

import asyncio
import concurrent.futures
import json
import logging
import threading
from typing import Awaitable, Callable, Optional, Union

import websockets
from websockets.exceptions import ConnectionClosed, WebSocketException

logger = logging.getLogger(__name__)


class SignalPoller:
    """
    WebSocket client for signal-cli-rest-api in json-rpc mode.
    
    In json-rpc mode, the /v1/receive/<number> endpoint is a WebSocket
    that streams incoming messages. This class connects and forwards
    messages to the handler.
    """
    
    def __init__(
        self,
        api_url: str,
        phone_number: str,
        on_message: Callable[[dict], Awaitable[None]],
        poll_interval: float = 5.0,
        loop: Optional[asyncio.AbstractEventLoop] = None,
    ):
        """
        Initialize the WebSocket listener.

        Args:
            api_url: Base URL of signal-cli-rest-api (e.g., http://signal-api:8080)
            phone_number: Bot's phone number for receiving messages
            on_message: Async callback function to handle incoming messages
            poll_interval: Reconnect delay on failure
            loop: External asyncio loop to run on. If provided, the poller
                  schedules its listen loop on it instead of owning a thread.
        """
        self.ws_url = api_url.replace("http://", "ws://").replace("https://", "wss://")
        self.phone_number = phone_number
        self.on_message = on_message
        self.poll_interval = poll_interval
        self._running = False
        self._loop = loop
        self._thread: Optional[threading.Thread] = None
        self._task: Optional[concurrent.futures.Future] = None

    def start(self):
        """Start the WebSocket listener."""
        if self._running:
            logger.warning("Listener already running")
            return

        self._running = True
        if self._loop is not None:
            # Schedule on the shared loop
            self._task = asyncio.run_coroutine_threadsafe(
                self._listen_loop(), self._loop
            )
            logger.info("Signal WebSocket listener scheduled on shared loop")
        else:
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()
            logger.info("Signal WebSocket listener started on dedicated thread")

    def stop(self):
        """Stop the WebSocket listener."""
        self._running = False
        if self._task is not None:
            self._task.cancel()
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Signal WebSocket listener stopped")

    def _run_loop(self):
        """Run the async event loop in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen_loop())
        except Exception as e:
            logger.error(f"WebSocket loop error: {e}")
        finally:
            loop.close()
    
    async def _listen_loop(self):
        """Main listening loop with reconnection logic."""
        # URL encode the phone number
        encoded_number = self.phone_number.replace("+", "%2B")
        ws_endpoint = f"{self.ws_url}/v1/receive/{encoded_number}"
        
        logger.info(f"WebSocket endpoint: {ws_endpoint}")
        
        while self._running:
            try:
                await self._connect_and_listen(ws_endpoint)
            except ConnectionClosed as e:
                logger.warning(f"WebSocket connection closed: {e}")
            except WebSocketException as e:
                logger.warning(f"WebSocket error: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
            
            if self._running:
                logger.info(f"Reconnecting in {self.poll_interval}s...")
                await asyncio.sleep(self.poll_interval)
    
    async def _connect_and_listen(self, ws_endpoint: str):
        """Connect to WebSocket and process messages."""
        logger.info(f"Connecting to WebSocket: {ws_endpoint}")
        
        async with websockets.connect(
            ws_endpoint,
            ping_interval=20,
            ping_timeout=10,
        ) as websocket:
            logger.info("Connected to signal-cli WebSocket")
            
            async for message in websocket:
                if not self._running:
                    break
                
                try:
                    data = json.loads(message)
                    await self._handle_message(data)
                except json.JSONDecodeError as e:
                    logger.debug(f"Failed to parse message: {e}")
                except Exception as e:
                    logger.error(f"Error handling message: {e}")
    
    async def _handle_message(self, msg: dict):
        """Handle a message received via WebSocket."""
        # The WebSocket sends envelope data directly
        envelope = msg.get("envelope", msg)
        
        # Skip empty or sync messages. Use `or {}` because dataMessage may be
        # explicitly null (sync/typing/edit/sticker envelopes), and the
        # default-arg form of .get() doesn't coerce a present-but-null value.
        data_msg = envelope.get("dataMessage") or {}
        if not data_msg:
            logger.debug(f"Skipping non-data message")
            return

        # Same null-vs-missing trap: a "message": null field would slice into
        # NoneType and raise, dropping the whole envelope before dispatch.
        text = data_msg.get("message") or ""
        source = (envelope.get("source") or "")[-4:]
        
        logger.info(f"Received message from ...{source}: {text[:50]}")
        
        # Forward as webhook format
        webhook_data = {"envelope": envelope}
        
        try:
            await self.on_message(webhook_data)
        except Exception as e:
            logger.error(f"Error processing message: {e}")
