"""
Per-bot LLMClient + DeepThinkClient cache.

Without this, ask_command would speak in the default bot's voice for
every context — even contexts pinned to another bot — because
`self.llm` and `self.deep_think` are wired once at construction with
the default bot's id. PR2 added `bot_id` resolution to LLMClient /
DeepThinkClient, but PR1-4 left the wiring single-bot. The factory
caches one client per (bot_id, role) so:

  * each bot's per-bot writer / deep_think overrides actually apply,
  * the long-lived aiohttp sessions are shared across calls (no
    TCP+TLS reconnect per !ask), and
  * deep-think tool plumbing (bot_tools, mcp_manager) is wired once
    rather than re-resolved on every lookup.

Cache invalidation isn't needed at runtime: Bot rows can be edited
freely, and LLMClient reads its config live from SettingsStore on
every call. Only when a bot is *deleted* do we want to drop its
cached client — `forget(bot_id)` covers that.
"""

from __future__ import annotations

import logging
import threading
from typing import Optional

from .client import LLMClient
from .deep_think import DeepThinkClient

logger = logging.getLogger(__name__)


class LLMClientFactory:
    """Thread-safe cache of bot-scoped LLMClient and DeepThinkClient.

    Use `get_writer(bot_id)` for the writer role, `get_deep_think(bot_id)`
    for the research role. Pass `bot_id=None` to get a legacy
    no-bot-scoping client that mirrors the pre-PR2 behavior (reads
    raw global llm_*/deep_think_* keys).

    `bot_tools` and `mcp_manager` are wired once on the factory and
    propagated to every DeepThinkClient instance. They're late-bound
    in main.py because they depend on the dispatcher being constructed
    first (chicken-and-egg with ask_command). Setters update both
    already-cached clients and any future ones.
    """

    def __init__(self, settings_store, bot_registry=None):
        self.store = settings_store
        # Optional BotRegistry — when wired, the writer's system-prompt
        # resolver falls back to bots.persona between the explicit
        # bot_llm_settings override and the global llm_system_prompt.
        # Without this, a non-default bot whose identity lives only in
        # `bots.persona` would inherit the global prompt (Sigil's text)
        # and leak persona across bots.
        self.bot_registry = bot_registry
        self._writers: dict[Optional[int], LLMClient] = {}
        self._reactors: dict[Optional[int], LLMClient] = {}
        self._deep_thinks: dict[Optional[int], DeepThinkClient] = {}
        # Late-bound; updated via attach_bot_tools / attach_mcp_manager.
        self._bot_tools = None
        self._mcp_manager = None
        self._lock = threading.RLock()

    def _persona_for(self, bot_id: int) -> Optional[str]:
        """BotRegistry-backed persona lookup. Returns None silently when
        the registry isn't wired or the bot row is missing — the writer
        then falls through to the global prompt as before."""
        if self.bot_registry is None:
            return None
        try:
            bot = self.bot_registry.get_sync(bot_id)
        except Exception as e:
            logger.debug(f"persona lookup failed for bot {bot_id}: {e}")
            return None
        if bot is None:
            return None
        return (bot.persona or "").strip() or None

    def get_writer(self, bot_id: Optional[int]) -> LLMClient:
        # The cache key includes None so legacy callers (no bot_id)
        # share a session too. RLock makes this safe under the
        # signal-handler thread + webhook thread + admin thread.
        with self._lock:
            client = self._writers.get(bot_id)
            if client is None:
                client = LLMClient(
                    self.store,
                    bot_id=bot_id,
                    persona_provider=self._persona_for if bot_id is not None else None,
                )
                self._writers[bot_id] = client
            return client

    def get_reactor(self, bot_id: Optional[int]) -> LLMClient:
        """Reactor-role client per bot. Used as the HTTP plumbing for the
        emoji/should_respond decision call. Reactor-specific values
        (model, temperature, prompt, cooldowns) are owned by EmojiReactor
        and passed in via `overrides`; this client only contributes
        api_key / base_url / timeout from the bot_llm_settings(bot,
        'reactor', *) → admin_settings.llm_* chain, so a bot that points
        its reactor at a different provider transparently works.
        """
        with self._lock:
            client = self._reactors.get(bot_id)
            if client is None:
                client = LLMClient(
                    self.store,
                    bot_id=bot_id,
                    role="reactor",
                )
                self._reactors[bot_id] = client
            return client

    def get_deep_think(self, bot_id: Optional[int]) -> DeepThinkClient:
        with self._lock:
            client = self._deep_thinks.get(bot_id)
            if client is None:
                client = DeepThinkClient(
                    self.store,
                    bot_tools=self._bot_tools,
                    mcp_manager=self._mcp_manager,
                    bot_id=bot_id,
                )
                self._deep_thinks[bot_id] = client
            return client

    def attach_bot_tools(self, bot_tools) -> None:
        """Late-bound after dispatcher exists. Backfills every cached
        DeepThinkClient so already-constructed clients pick up the
        tools, and remembers it for future lookups."""
        with self._lock:
            self._bot_tools = bot_tools
            for client in self._deep_thinks.values():
                client.bot_tools = bot_tools

    def attach_mcp_manager(self, mcp_manager) -> None:
        with self._lock:
            self._mcp_manager = mcp_manager
            for client in self._deep_thinks.values():
                client.mcp_manager = mcp_manager

    def forget(self, bot_id: int) -> None:
        """Drop cached clients for a bot — called when the bot is
        deleted. The aiohttp sessions on the dropped clients become
        garbage-collected; existing in-flight calls finish on the old
        client."""
        with self._lock:
            self._writers.pop(bot_id, None)
            self._reactors.pop(bot_id, None)
            self._deep_thinks.pop(bot_id, None)

    async def close_all(self) -> None:
        """Close every cached client's aiohttp session. Called from
        shutdown paths."""
        with self._lock:
            clients = (
                list(self._writers.values())
                + list(self._reactors.values())
            )
            self._writers.clear()
            self._reactors.clear()
            self._deep_thinks.clear()
        for client in clients:
            try:
                await client.close()
            except Exception as e:
                logger.debug(f"LLMClient.close failed during shutdown: {e}")
