"""
!ask — pass a question to the configured LLM, with optional tool use via MCP.

Flow per request:
  1. Build messages (system prompt + optional group-chat suffix + history + user).
  2. If MCP servers expose tools, pass them to the LLM.
  3. Loop up to MAX_TOOL_ROUNDS: if the assistant returns tool_calls, run them
     through the MCP manager and feed the results back as tool messages.
  4. Persist the final user question + assistant answer to per-user history.

The command's registered name is always "ask"; an admin-chosen alias from
`ask_command_name` is added at dispatch time so users can rename it live.
"""

import json
import logging
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..database import hash_phone
from ..llm import LLMClient, LLMDisabled, LLMError, LLMNotConfigured, ConversationHistory

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ROUNDS = 25


class AskCommand(BaseCommand):
    name = "ask"
    aliases = ["a"]
    description = "Ask the configured LLM"
    usage = "!ask <question>"
    help_explanation = (
        "Send a question to the LLM configured in /admin/llm. "
        "Conversations are per-user and remember a rolling window of recent turns. "
        "Use '!ask reset' to wipe your history."
    )

    def __init__(
        self,
        llm: LLMClient,
        history: ConversationHistory,
        group_log=None,
        mcp_manager=None,
        bot_tools=None,
        enricher=None,
    ):
        self.llm = llm
        self.history = history
        self.group_log = group_log
        self.mcp_manager = mcp_manager
        self.bot_tools = bot_tools
        # Optional message-text enricher (e.g. TwitterExpander) — called on the
        # user's question so pasted links carry their content into the prompt.
        self.enricher = enricher

    def _live_turns(self) -> int:
        try:
            return max(0, int(self.llm.store.get("llm_history_turns") or 6))
        except (TypeError, ValueError):
            return 6

    def _live_group_ctx(self) -> int:
        try:
            return max(0, int(self.llm.store.get("group_context_messages") or 0))
        except (TypeError, ValueError):
            return 0

    def _live_max_tool_rounds(self) -> int:
        try:
            v = int(self.llm.store.get("llm_max_tool_rounds") or DEFAULT_MAX_TOOL_ROUNDS)
            return max(1, v)
        except (TypeError, ValueError):
            return DEFAULT_MAX_TOOL_ROUNDS

    async def _build_group_context(self, ctx) -> str:
        limit = self._live_group_ctx()
        if limit <= 0 or not ctx.is_group or self.group_log is None:
            return ""
        msgs = await self.group_log.recent(ctx.group_id, limit=limit, exclude_last=1)
        if not msgs:
            return ""
        lines = ["Recent group chat (oldest first, senders shown by last 4 digits):"]
        for m in msgs:
            tail = (m["sender"] or "")[-4:] or "????"
            text = (m["text"] or "").replace("\n", " ").strip()
            if text:
                lines.append(f"  [...{tail}] {text}")
        return "\n".join(lines)

    def _collect_tools(self, policy=None) -> Optional[list[dict]]:
        schemas: list[dict] = []
        if self.bot_tools is not None:
            schemas.extend(self.bot_tools.list_tools(policy=policy))
        if self.mcp_manager is not None:
            mcp_tools = self.mcp_manager.all_tools()
            if policy is not None:
                mcp_tools = [t for t in mcp_tools if policy.allows_mcp(t.server_name)]
            schemas.extend(t.to_openai_tool() for t in mcp_tools)
        return schemas or None

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        caller_ctx: CommandContext,
        attachments: list,
    ) -> str:
        """Drive the assistant/tool back-and-forth until we have final text.

        Attachments produced by bot-tool calls are appended to `attachments`
        so the caller can include them in the final Signal response.

        If the round cap is hit we DO NOT fabricate a summary from partial
        results — accuracy matters more than appearing helpful. Instead we
        return an honest error noting the cap and the tools the LLM tried.
        """
        max_rounds = self._live_max_tool_rounds()
        tool_call_history: list[str] = []

        for round_idx in range(max_rounds):
            logger.info(
                f"LLM round {round_idx + 1}/{max_rounds}: "
                f"requesting completion ({len(messages)} msgs in context)"
            )
            assistant_msg = await self.llm.chat_messages(messages, tools=tools)
            messages.append(assistant_msg)
            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                content = (assistant_msg.get("content") or "").strip()
                logger.info(
                    f"LLM round {round_idx + 1}: final answer "
                    f"({len(content)} chars, no tool calls)"
                )
                return content
            for call in tool_calls:
                fn_name = (call.get("function") or {}).get("name") or "?"
                tool_call_history.append(fn_name)
                await self._execute_tool_call(call, messages, caller_ctx, attachments)

        # Cap hit — log the full sequence and return an honest error. We
        # deliberately do NOT make a final no-tools call: the LLM would
        # produce a summary based on incomplete work, which can be
        # confidently wrong on factual queries.
        logger.warning(
            f"Tool loop hit cap ({max_rounds} rounds) without resolution. "
            f"Tool sequence: {' -> '.join(tool_call_history)}"
        )

        # Last few tools tried, for the user-facing message
        recent = tool_call_history[-6:]
        recent_str = " -> ".join(recent)
        return (
            f"Task didn't complete after {max_rounds} tool-call rounds. "
            f"The model was working through: {recent_str}. "
            f"Either ask a more focused question, or raise "
            f"`Max tool-call rounds per !ask` in /admin/llm."
        )

    async def _execute_tool_call(
        self,
        call: dict,
        messages: list[dict],
        caller_ctx: CommandContext,
        attachments: list,
    ) -> None:
        call_id = call.get("id") or ""
        fn = call.get("function") or {}
        name = fn.get("name") or ""
        raw_args = fn.get("arguments") or "{}"
        try:
            args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
        except json.JSONDecodeError:
            args = {}

        if isinstance(raw_args, str):
            log_args = raw_args[:200]
        else:
            log_args = json.dumps(args)[:200]
        logger.info(f"LLM tool call: {name} args={log_args}")

        is_bot_tool = (
            self.bot_tools is not None
            and name.startswith(self.bot_tools.NAMESPACE + "__")
        )

        try:
            if is_bot_tool:
                result = await self.bot_tools.call(name, args, caller_ctx)
                content = result.text if result else "(no result)"
                if result and result.attachments:
                    attachments.extend(result.attachments)
            elif self.mcp_manager is not None:
                content = await self.mcp_manager.call_tool(name, args)
            else:
                content = f"ERROR: unknown tool {name}"
        except Exception as e:
            logger.warning(f"Tool {name} failed: {e}")
            content = f"ERROR: {e}"

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })

    def _context_key(self, ctx: CommandContext) -> str:
        """Group chats share a thread via group_id. DMs use the hashed phone."""
        if ctx.group_id:
            return f"group:{ctx.group_id}"
        return f"dm:{hash_phone(ctx.sender)}"

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                "Ask something: !ask what sectors are rotating this week?"
            )

        user_hash = hash_phone(ctx.sender)
        sender_tail = (ctx.sender or "")[-4:]
        context_key = self._context_key(ctx)
        is_group = ctx.group_id is not None

        if ctx.args[0].lower() in ("reset", "clear", "forget"):
            removed = await self.history.clear(context_key)
            return CommandResult.ok(f"Cleared {removed} conversation turn(s).")

        parts = ctx.raw_message.split(maxsplit=1)
        question = parts[1].strip() if len(parts) > 1 else " ".join(ctx.args)
        if not question:
            return CommandResult.error("Ask something: !ask <question>")

        # Inline-expand any tweet/URL links the user pasted so the LLM
        # gets the substance, not just an opaque URL.
        if self.enricher is not None:
            try:
                question = await self.enricher.expand(question)
            except Exception as e:
                logger.debug(f"Question enrichment failed: {e}")

        try:
            prior = await self.history.load(
                context_key,
                turns_per_user=self._live_turns(),
                attribute_senders=is_group,
            )
            group_ctx = await self._build_group_context(ctx)

            prompt_override = None
            if ctx.policy is not None and ctx.policy.system_prompt:
                prompt_override = ctx.policy.system_prompt
            system_prompt = self.llm._resolve_system_prompt(prompt_override, group_ctx or None)

            # Current question gets the same attribution prefix in group mode
            current_user_content = (
                f"[...{sender_tail}] {question}" if is_group else question
            )

            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            messages.extend(prior)
            messages.append({"role": "user", "content": current_user_content})

            tools = self._collect_tools(policy=ctx.policy)
            attachments: list = []
            answer = await self._run_tool_loop(messages, tools, ctx, attachments)
        except LLMDisabled:
            return CommandResult.error(
                "LLM is not enabled. An admin can turn it on at /admin/llm."
            )
        except LLMNotConfigured as e:
            return CommandResult.error(f"LLM not configured: {e}")
        except LLMError as e:
            logger.warning(f"LLM error for {ctx.sender[-4:]}: {e}")
            return CommandResult.error(str(e))

        if not answer:
            return CommandResult.error("LLM returned no answer.")

        try:
            # Store the raw question (no prefix) + sender_tail separately; the
            # load path re-adds the prefix when replaying in a group context.
            await self.history.append(
                context_key, "user", question,
                user_hash=user_hash, sender_tail=sender_tail,
            )
            await self.history.append(
                context_key, "assistant", answer, user_hash=user_hash,
            )
        except Exception as e:
            logger.error(f"Failed to persist ask history: {e}")

        return CommandResult(
            text=answer,
            success=True,
            attachments=attachments or None,
            styled=True,
        )
