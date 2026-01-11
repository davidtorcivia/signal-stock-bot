"""
JSON-RPC socket listener for signal-cli daemon.

Connects directly to signal-cli's JSON-RPC socket to receive messages
in real-time. This bypasses the broken webhook mechanism in signal-cli-rest-api.
"""

import asyncio
import json
import logging
import socket
import threading
import time
from typing import Callable, Optional

logger = logging.getLogger(__name__)


class SignalPoller:
    """
    Connects to signal-cli's JSON-RPC socket and listens for incoming messages.
    
    In json-rpc mode, signal-cli runs as a daemon and exposes a JSON-RPC
    interface on TCP port 6001. This class connects to that socket and
    forwards received messages to the handler.
    """
    
    def __init__(
        self,
        api_url: str,
        phone_number: str,
        on_message: Callable[[dict], None],
        poll_interval: float = 1.0,
        jsonrpc_port: int = 6001,
    ):
        """
        Initialize the listener.
        
        Args:
            api_url: Base URL of signal-cli-rest-api (used to extract host)
            phone_number: Bot's phone number for filtering messages
            on_message: Async callback function to handle incoming messages
            poll_interval: Reconnect delay on failure
            jsonrpc_port: TCP port for JSON-RPC socket (default 6001)
        """
        # Extract host from URL (e.g., "http://signal-api:8080" -> "signal-api")
        self.host = api_url.replace("http://", "").replace("https://", "").split(":")[0]
        self.phone_number = phone_number
        self.on_message = on_message
        self.poll_interval = poll_interval
        self.jsonrpc_port = jsonrpc_port
        self._running = False
        self._thread: Optional[threading.Thread] = None
    
    def start(self):
        """Start the listener in a background thread."""
        if self._running:
            logger.warning("Listener already running")
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()
        logger.info(f"Signal listener started (connecting to {self.host}:{self.jsonrpc_port})")
    
    def stop(self):
        """Stop the listener."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        logger.info("Signal listener stopped")
    
    def _run_loop(self):
        """Run the async event loop in the background thread."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._listen_loop())
        except Exception as e:
            logger.error(f"Listener loop error: {e}")
        finally:
            loop.close()
    
    async def _listen_loop(self):
        """Main listening loop with reconnection logic."""
        while self._running:
            try:
                await self._connect_and_listen()
            except ConnectionRefusedError:
                logger.debug(f"Connection refused to {self.host}:{self.jsonrpc_port}, retrying...")
            except Exception as e:
                logger.error(f"Socket error: {e}")
            
            if self._running:
                await asyncio.sleep(self.poll_interval)
    
    async def _connect_and_listen(self):
        """Connect to JSON-RPC socket and process events."""
        logger.info(f"Connecting to JSON-RPC socket at {self.host}:{self.jsonrpc_port}")
        
        reader, writer = await asyncio.open_connection(self.host, self.jsonrpc_port)
        logger.info("Connected to signal-cli JSON-RPC socket")
        
        try:
            buffer = ""
            while self._running:
                # Read data with timeout
                try:
                    data = await asyncio.wait_for(reader.read(4096), timeout=30.0)
                except asyncio.TimeoutError:
                    # Send keepalive / check connection
                    continue
                
                if not data:
                    logger.warning("Socket closed by server")
                    break
                
                buffer += data.decode("utf-8")
                
                # Process complete JSON objects (newline-delimited)
                while "\n" in buffer:
                    line, buffer = buffer.split("\n", 1)
                    line = line.strip()
                    if not line:
                        continue
                    
                    try:
                        msg = json.loads(line)
                        await self._handle_message(msg)
                    except json.JSONDecodeError as e:
                        logger.debug(f"Failed to parse JSON: {e}")
        finally:
            writer.close()
            await writer.wait_closed()
    
    async def _handle_message(self, msg: dict):
        """Handle a JSON-RPC message from signal-cli."""
        # JSON-RPC notification format: {"jsonrpc": "2.0", "method": "receive", "params": {...}}
        method = msg.get("method")
        
        if method == "receive":
            params = msg.get("params", {})
            envelope = params.get("envelope", {})
            account = params.get("account", "")
            
            # Check if this message is for our account
            if account and account != self.phone_number:
                logger.debug(f"Ignoring message for account {account[-4:]}")
                return
            
            # Convert to webhook format and forward
            webhook_data = {"envelope": envelope}
            
            data_msg = envelope.get("dataMessage", {})
            text = data_msg.get("message", "")
            source = envelope.get("source", "")[-4:]
            
            logger.info(f"Received message from ...{source}: {text[:50]}")
            
            try:
                await self.on_message(webhook_data)
            except Exception as e:
                logger.error(f"Error processing message: {e}")
        
        elif method:
            logger.debug(f"Ignoring JSON-RPC method: {method}")
