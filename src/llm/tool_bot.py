"""
ToolBotClient — the tool-executor half of `deep_think_mode='tool_bot'`.

This is the sibling that runs the agentic tool loop on behalf of a writer
model with poor native tool-calling (e.g. a locally-trained persona model).
It is a thin specialization of `DeepThinkClient`: same tool loop, same
policy-filtered tool kit, same bounded runtime — but it reads its config
from the bot's dedicated `tool_bot` role
(`bot_llm_settings(bot_id, 'tool_bot', *)`) and falls back
`tool_bot_* -> deep_think_* -> llm_*`, so an admin can point it at a
different, more capable endpoint without touching deep_think.

The one behavioural addition is the GATE. Where research mode delegates on
every turn, tool_bot mode instructs this client to answer with the bare
sentinel `NOTOOLS` when the message needs no live data or tools. The caller
(ask_command) sees that sentinel and skips the handoff entirely, letting the
writer answer directly — so pure conversation never pays for a research
pass, and the writer never receives misleading "notes" for a turn that
needed none.

Like its parent, `think()` NEVER raises: every failure resolves to a string
the writer can integrate or discard.
"""

from __future__ import annotations

import logging
import re
from typing import Optional

from .deep_think import DeepThinkClient
from .mcp_broker import allowed_mcp_tools, invoke_mcp_tool

logger = logging.getLogger(__name__)

# OpenAI/OpenRouter function-name constraint. MCP tools whose qualified
# name violates it can't be exposed directly (they'd be rejected by the
# API), so they stay reachable only via the broker fallback.
_VALID_FN_NAME = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# Emitted by the tool-bot (verbatim, alone) when the turn needs no tools.
# Kept short and unmistakable so `is_no_tools` can detect it cheaply even if
# the model wraps it in a little punctuation or a trailing newline.
NO_TOOLS_SENTINEL = "NOTOOLS"


DEFAULT_TOOL_BOT_PROMPT = """\
You are the tool-running half of a two-model assistant. A separate writer
model composes the actual chat reply in its own voice; YOUR only job is to
do any live-data or tool work it can't, and hand back concise notes.

You have the assistant's full tool kit — bot commands (prices, charts,
news, fundamentals, calendars) and any configured MCP servers (web/news
search, fetch, filings, etc.). Every tool is callable DIRECTLY by its
name; just call it. There is no discover/invoke step.

FIRST decide whether this turn needs tools at all:

- If the message is small talk, opinion, wordplay, an emotional beat, a
  question answerable from general knowledge, or anything that does NOT
  depend on a live number, a current event, or a specific fetched fact —
  reply with EXACTLY the single token:
      NOTOOLS
  Nothing else. No explanation, no punctuation. The writer will handle it.

- Otherwise, USE THE TOOLS. Don't guess at a price, P/E, or recent event —
  fetch it. Chain calls when one finding leads to the next. Then write
  compact notes: the facts you gathered, each with its source/number, and
  any caveat about freshness or uncertainty. Do NOT write a chat reply,
  a greeting, or persona voice — just the findings. The writer turns your
  notes into the reply.

Be honest about failure: if a tool errored or you couldn't get the data,
say so plainly in the notes rather than inventing a value. Never emit
NOTOOLS as a way to avoid work on a question that genuinely needs data —
the sentinel is only for turns that truly need none."""


class ToolBotClient(DeepThinkClient):
    """DeepThinkClient specialized for `deep_think_mode='tool_bot'`.

    Reads the `tool_bot` role config chain and defaults to enabled (the
    mode selection on the bot is itself the opt-in), with a self-gating
    system prompt. All tool-loop mechanics are inherited unchanged.
    """

    def __init__(
        self,
        settings_store,
        bot_tools=None,
        mcp_manager=None,
        bot_id: Optional[int] = None,
    ):
        super().__init__(
            settings_store,
            bot_tools=bot_tools,
            mcp_manager=mcp_manager,
            bot_id=bot_id,
            role="tool_bot",
            role_label="tool_bot",
            # Per-bot overrides live under the `tool_bot` role namespace
            # (bot_llm_settings(bot_id, 'tool_bot', *)) — that's the real
            # config surface, set from the bots form. GLOBAL inheritance
            # falls to the deep_think endpoint, then the writer (llm_*):
            # there is deliberately no global `tool_bot_*` settings block,
            # so a bot with no per-bot overrides reuses whatever capable
            # endpoint deep_think already points at.
            own_prefixes=("deep_think",),
            shared_fallback=("llm",),
            # Selecting tool_bot mode is the opt-in; readiness then hinges
            # only on base_url/api_key/model being present.
            default_enabled=True,
            default_system_prompt=DEFAULT_TOOL_BOT_PROMPT,
        )

    def _collect_tools(self, policy=None):
        """Expose MCP tools DIRECTLY (expanded schemas) instead of via the
        discover/invoke broker.

        The broker keeps the writer's tool catalog small, but tool_bot mode
        exists for models with weak tool-calling, and those models reliably
        fail the broker's two-step indirection — they call the discovered
        `server__tool` name directly instead of wrapping it in mcp__invoke.
        Handing them the tools already expanded matches what they try to do
        and removes the step they can't perform. Only the tool_bot's own
        prompt grows; the writer is untouched.

        Bot tools come from the shared allow-list exactly as in the parent.
        MCP tools with API-invalid names fall back to the broker so they
        stay reachable.
        """
        schemas: list[dict] = []
        if self.bot_tools is not None:
            schemas.extend(self.bot_tools.list_tools(policy=policy))

        expandable = []
        skipped = []
        if self.mcp_manager is not None:
            for tool in allowed_mcp_tools(self.mcp_manager, policy):
                if _VALID_FN_NAME.match(tool.qualified_name):
                    expandable.append(tool)
                else:
                    skipped.append(tool.qualified_name)
        for tool in expandable:
            schemas.append(tool.to_openai_tool())
        if skipped:
            # Names too long / invalid for a function name — keep the broker
            # so they remain reachable via discover/invoke.
            from .mcp_broker import MCP_BROKER_TOOLS
            logger.info(
                "tool_bot: %d MCP tool(s) not directly exposable "
                "(invalid fn name), keeping broker fallback: %s",
                len(skipped), ", ".join(skipped[:5]),
            )
            schemas.extend(MCP_BROKER_TOOLS)

        schemas.sort(
            key=lambda schema: str(
                (schema.get("function") or {}).get("name") or ""
            )
        )
        return schemas or None

    async def _handle_direct_mcp_call(self, name, args, caller_ctx) -> str:
        """Invoke an MCP tool directly by its qualified name.

        Because `_collect_tools` exposes MCP tools expanded, the model calls
        `server__tool` names directly — route those straight to the MCP
        manager instead of the parent's 'use mcp__invoke' rejection."""
        policy = getattr(caller_ctx, "policy", None) if caller_ctx else None
        return await invoke_mcp_tool(
            self.mcp_manager, policy, name=name, arguments=args,
        )

    @staticmethod
    def is_no_tools(notes: Optional[str]) -> bool:
        """True when the tool-bot signalled the turn needs no delegation.

        Tolerant of trailing whitespace/newlines and a stray wrapping
        character or two, but requires the sentinel to be essentially the
        whole reply — a notes block that merely mentions the word in prose
        must not be mistaken for the gate.
        """
        if not notes:
            return False
        stripped = notes.strip().strip("`*_.\"' \t\r\n")
        return stripped.upper() == NO_TOOLS_SENTINEL
