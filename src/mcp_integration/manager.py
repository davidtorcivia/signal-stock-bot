"""
MCPManager — owns live client sessions for configured MCP servers.

One session per server. Each session holds a long-running asyncio task that
keeps the transport + ClientSession open; the manager hands out tool calls
to that task via direct method invocation on the live session object.

Lifecycle task ownership matters: the mcp SDK's transports (stdio in
particular) use anyio task groups that must be entered and exited from the
same task. We run the session's entire lifetime inside a dedicated task
and tear it down by setting an asyncio.Event it's awaiting.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Optional

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

try:
    from mcp.client.sse import sse_client
except ImportError:  # older SDKs
    sse_client = None  # type: ignore

try:
    from mcp.client.streamable_http import streamablehttp_client
except ImportError:
    streamablehttp_client = None  # type: ignore

from .models import MCPServerConfig, MCPTool
from .registry import MCPRegistry

logger = logging.getLogger(__name__)


def _field(obj, *names, default=None):
    """Read the first attribute that exists, by any of its historical names.

    The MCP Python SDK renamed its pydantic model fields from the wire's
    camelCase to snake_case in 2.0 (`Tool.inputSchema` → `Tool.input_schema`,
    `CallToolResult.isError` → `is_error`). Both spellings are accepted here
    so the client works against either SDK major version — the servers we
    launch are third-party and pin their own `mcp`, so a single hard-coded
    spelling would break on whichever side moved first.

    This mattered silently: the old `getattr(result, "isError", False)` kept
    returning False against a 2.x SDK, so every failed tool call was reported
    to the model as a successful one.
    """
    for name in names:
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


TOOL_CALL_TIMEOUT = 30.0
# 60s allows a first-run `npx -y @some/package` to download its deps.
STARTUP_TIMEOUT = 60.0


class MCPSessionError(Exception):
    """Something went wrong with an MCP session."""


class _MCPSession:
    """Holds one running client session for one MCP server."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.tools: list[MCPTool] = []
        self.started_at: Optional[float] = None
        self.last_error: Optional[str] = None

        self._session: Optional[ClientSession] = None
        self._task: Optional[asyncio.Task] = None
        self._ready: Optional[asyncio.Event] = None
        self._stop: Optional[asyncio.Event] = None

    @property
    def running(self) -> bool:
        return self._session is not None and self._task is not None and not self._task.done()

    async def start(self) -> None:
        if self.running:
            return

        self._ready = asyncio.Event()
        self._stop = asyncio.Event()
        startup_error: list[BaseException] = []
        self._task = asyncio.create_task(
            self._run(self._ready, self._stop, startup_error),
            name=f"mcp-session-{self.config.name}",
        )

        # Wait for the task to either become ready or die during startup.
        ready_task = asyncio.create_task(self._ready.wait())
        try:
            done, _ = await asyncio.wait(
                {ready_task, self._task},
                timeout=STARTUP_TIMEOUT,
                return_when=asyncio.FIRST_COMPLETED,
            )
        finally:
            if not ready_task.done():
                ready_task.cancel()

        if self._task.done():
            # Startup failed — surface the error
            try:
                self._task.result()
            except BaseException as e:
                self.last_error = str(e)
                raise MCPSessionError(f"{self.config.name}: startup failed: {e}") from e
            raise MCPSessionError(f"{self.config.name}: session exited before ready")

        if not self._ready.is_set():
            # Timed out — stop the task and bail
            self._stop.set()
            raise MCPSessionError(f"{self.config.name}: startup timed out")

    async def stop(self) -> None:
        if self._stop:
            self._stop.set()
        if self._task:
            try:
                await asyncio.wait_for(self._task, timeout=10.0)
            except asyncio.TimeoutError:
                self._task.cancel()
                try:
                    await self._task
                except (asyncio.CancelledError, Exception):
                    pass
        self._task = None
        self._session = None
        self._ready = None
        self._stop = None

    async def call_tool(self, tool_name: str, arguments: dict) -> str:
        if not self._session:
            raise MCPSessionError(f"{self.config.name}: not running")
        try:
            result = await asyncio.wait_for(
                self._session.call_tool(tool_name, arguments),
                timeout=TOOL_CALL_TIMEOUT,
            )
        except asyncio.TimeoutError:
            raise MCPSessionError(f"Tool {tool_name} timed out after {TOOL_CALL_TIMEOUT}s")
        return _flatten_result(result)

    async def _open_transport(self):
        """Return the async context manager for this server's transport."""
        cfg = self.config
        if cfg.transport == "stdio":
            if not cfg.command:
                raise MCPSessionError("stdio server is missing command")
            # The SDK passes only a safe allowlist of env vars to the child
            # when `env` is None, so UV_CONSTRAINT — which pins the MCP SDK
            # version that uvx-launched servers resolve — gets stripped and
            # those servers silently resolve an SDK they can't import. Thread
            # it through explicitly rather than inheriting the whole
            # environment, which would hand every third-party server process
            # the bot's API keys.
            passthrough = {
                k: v for k in ("UV_CONSTRAINT",)
                if (v := os.environ.get(k))
            }
            merged_env = (
                {**os.environ, **cfg.env} if cfg.env
                else (passthrough or None)
            )
            params = StdioServerParameters(
                command=cfg.command,
                args=list(cfg.args or []),
                env=merged_env,
            )
            return stdio_client(params)
        if cfg.transport == "sse":
            if sse_client is None:
                raise MCPSessionError("sse_client unavailable in installed mcp SDK")
            if not cfg.url:
                raise MCPSessionError("sse server is missing url")
            return sse_client(cfg.url, headers=cfg.headers or None)
        if cfg.transport == "http":
            if streamablehttp_client is None:
                raise MCPSessionError("streamablehttp_client unavailable in installed mcp SDK")
            if not cfg.url:
                raise MCPSessionError("http server is missing url")
            return streamablehttp_client(cfg.url, headers=cfg.headers or None)
        raise MCPSessionError(f"Unknown transport: {cfg.transport}")

    async def _run(
        self,
        ready: asyncio.Event,
        stop: asyncio.Event,
        startup_error: list[BaseException],
    ) -> None:
        """Long-running task that holds the session open until stopped."""
        try:
            transport_cm = await self._open_transport()
            async with transport_cm as transport:
                # stdio/sse: (read, write); http: (read, write, maybe session_id_fn)
                read, write = transport[0], transport[1]
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    listing = await session.list_tools()
                    self.tools = [
                        MCPTool(
                            server_name=self.config.name,
                            name=t.name,
                            description=(t.description or ""),
                            input_schema=(
                                _field(t, "input_schema", "inputSchema")
                                or {"type": "object", "properties": {}}
                            ),
                        )
                        for t in listing.tools
                    ]
                    self._session = session
                    self.started_at = time.time()
                    self.last_error = None
                    ready.set()
                    await stop.wait()
        except asyncio.CancelledError:
            raise
        except BaseException as e:
            self.last_error = str(e)
            startup_error.append(e)
            logger.error(f"MCP session {self.config.name} exited: {e}")
            raise


def _flatten_result(result) -> str:
    """Collapse an mcp CallToolResult into a plain-text string."""
    # `result.content` is a list of content parts. For phase 3 we flatten
    # text parts and stringify the rest.
    out_parts: list[str] = []
    content = getattr(result, "content", None) or []
    for part in content:
        text = getattr(part, "text", None)
        if text:
            out_parts.append(text)
            continue
        data = getattr(part, "data", None)
        if data:
            out_parts.append(f"<binary: {len(data)} bytes>")
            continue
        out_parts.append(json.dumps(_coerce(part)))
    if _field(result, "is_error", "isError", default=False):
        return "ERROR: " + "\n".join(out_parts) if out_parts else "ERROR (no detail)"
    return "\n".join(out_parts) if out_parts else "(no result)"


def _coerce(obj) -> Any:
    try:
        return obj.model_dump() if hasattr(obj, "model_dump") else str(obj)
    except Exception:
        return str(obj)


class MCPManager:
    """Coordinates multiple MCP sessions. All async ops run on the same loop."""

    def __init__(self, registry: MCPRegistry):
        self.registry = registry
        self._sessions: dict[int, _MCPSession] = {}
        self._lock = asyncio.Lock()

    async def start_server(self, server_id: int) -> _MCPSession:
        async with self._lock:
            existing = self._sessions.get(server_id)
            if existing and existing.running:
                return existing

            cfg = await self.registry.get(server_id)
            if not cfg or cfg.id is None:
                raise MCPSessionError(f"Server {server_id} not found")
            session = _MCPSession(cfg)
            try:
                await session.start()
            except Exception:
                self._sessions.pop(server_id, None)
                raise
            self._sessions[server_id] = session
            logger.info(f"MCP {cfg.name} started with {len(session.tools)} tool(s)")
            return session

    async def stop_server(self, server_id: int) -> None:
        async with self._lock:
            session = self._sessions.pop(server_id, None)
        if session:
            await session.stop()
            logger.info(f"MCP {session.config.name} stopped")

    async def restart_server(self, server_id: int) -> _MCPSession:
        await self.stop_server(server_id)
        return await self.start_server(server_id)

    async def start_all_enabled(self) -> None:
        configs = await self.registry.list()
        for cfg in configs:
            if cfg.enabled and cfg.id is not None:
                try:
                    await self.start_server(cfg.id)
                except Exception as e:
                    logger.error(f"MCP {cfg.name} failed to start: {e}")

    async def stop_all(self) -> None:
        for sid in list(self._sessions.keys()):
            try:
                await self.stop_server(sid)
            except Exception as e:
                logger.error(f"MCP stop error for {sid}: {e}")

    def get_session(self, server_id: int) -> Optional[_MCPSession]:
        return self._sessions.get(server_id)

    def all_tools(self) -> list[MCPTool]:
        tools: list[MCPTool] = []
        for s in self._sessions.values():
            if s.running:
                tools.extend(s.tools)
        return tools

    def status(self) -> dict[int, dict]:
        """Snapshot of per-server state."""
        out = {}
        for sid, s in self._sessions.items():
            out[sid] = {
                "running": s.running,
                "tool_count": len(s.tools),
                "started_at": s.started_at,
                "last_error": s.last_error,
            }
        return out

    async def call_tool(self, qualified_name: str, arguments: dict) -> str:
        server_name, _, tool_name = qualified_name.partition("__")
        if not tool_name:
            raise MCPSessionError(f"Invalid qualified tool name: {qualified_name!r}")
        for s in self._sessions.values():
            if s.config.name == server_name and s.running:
                return await s.call_tool(tool_name, arguments)
        raise MCPSessionError(f"MCP server '{server_name}' is not running")
