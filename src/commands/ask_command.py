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

import asyncio
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
        signal_handler=None,
        name_registry=None,
        summarizer=None,
        reactor=None,
    ):
        self.llm = llm
        self.history = history
        self.group_log = group_log
        self.mcp_manager = mcp_manager
        self.bot_tools = bot_tools
        # Optional message-text enricher (e.g. TwitterExpander) — called on the
        # user's question so pasted links carry their content into the prompt.
        self.enricher = enricher
        # Optional Signal handler — used only to drive the typing indicator
        # while the LLM tool loop runs. None is fine; the indicator is a UX
        # nicety, not a correctness requirement.
        self.signal_handler = signal_handler
        # Optional name registry — when set, group-context lines and the
        # current user's message are prefixed with `[Name]` instead of
        # `[...4137]` for known users.
        self.name_registry = name_registry
        # Optional rolling-summary writer. When set, each successful !ask
        # fires a fire-and-forget call that may (re)compress older turns
        # into a paragraph injected into future system prompts.
        self.summarizer = summarizer
        # Optional reactor — used here only as a read-only source for the
        # in-memory log of recent reactions, so the writing LLM can answer
        # "why did you react with X?" without confabulating.
        self.reactor = reactor

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

    async def _enrich(self, text: str) -> str:
        """Best-effort, idempotent re-enrichment at LLM-feed boundaries.

        We never want the model to receive a bare URL it can't open. Every
        text fragment routed into the LLM context goes through this helper,
        even fragments stored before the enricher existed or fragments
        whose original write-time enrichment failed. The expanders are
        idempotent (skip URLs whose snippet is already inline) and cached,
        so repeated calls are cheap.
        """
        if self.enricher is None or not text:
            return text
        try:
            return await self.enricher.expand(text)
        except Exception as e:
            logger.debug(f"Read-time enrichment failed: {e}")
            return text

    async def _build_group_context(self, ctx) -> str:
        limit = self._live_group_ctx()
        if limit <= 0 or not ctx.is_group or self.group_log is None:
            return ""
        msgs = await self.group_log.recent(ctx.group_id, limit=limit, exclude_last=1)
        if not msgs:
            return ""
        if self.name_registry is not None and any(self.name_registry._cache):
            header = "Recent group chat (oldest first; senders by name when known, last-4 otherwise):"
        else:
            header = "Recent group chat (oldest first, senders shown by last 4 digits):"

        # Enrich messages concurrently. _enrich short-circuits on empty
        # input so empty cells cost ~nothing; gathering avoids 30 sequential
        # awaits stacking up on the LLM hot path.
        raw_texts = [(m["text"] or "").strip() for m in msgs]
        enriched = await asyncio.gather(*(self._enrich(t) for t in raw_texts))

        lines = [header]
        for m, text in zip(msgs, enriched):
            if not text:
                continue
            label = self._sender_label(m["sender"])
            lines.append(f"  [{label}] {text.replace(chr(10), ' ')}")
        return "\n".join(lines)

    def _sender_label(self, phone: Optional[str]) -> str:
        if self.name_registry is None:
            return f"...{(phone or '')[-4:] or '????'}"
        return self.name_registry.label_for(phone)

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

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                "Ask something: !ask what sectors are rotating this week?"
            )

        user_hash = hash_phone(ctx.sender)
        sender_tail = (ctx.sender or "")[-4:]
        context_key = ctx.context_key()
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
        question = await self._enrich(question)

        try:
            prior = await self.history.load(
                context_key,
                turns_per_user=self._live_turns(),
                attribute_senders=is_group,
            )
            # Re-enrich each prior turn — bot/user messages stored before
            # the enricher existed (or stored with a failed enrichment)
            # would otherwise reach the LLM as bare URLs it can't open.
            # Gathered: with 30-turn history this matters on the LLM hot path.
            text_msgs = [
                m for m in prior if isinstance(m.get("content"), str)
            ]
            if text_msgs:
                enriched = await asyncio.gather(
                    *(self._enrich(m["content"]) for m in text_msgs)
                )
                for m, new_content in zip(text_msgs, enriched):
                    m["content"] = new_content

            group_ctx = await self._build_group_context(ctx)

            # Long-term rolling summary, if one exists. Prepended to the
            # group-context block so both end up in the system message in
            # a sensible reading order: persona → long memory → recent
            # group chat → current time → response style.
            summary_block = ""
            try:
                summary = await self.history.get_summary(context_key)
                if summary and summary.get("summary"):
                    summary_text = await self._enrich(summary["summary"])
                    summary_block = (
                        "Conversation memory (rolling summary of older "
                        "turns):\n" + summary_text
                    )
            except Exception as e:
                logger.debug(f"Failed to load summary for ask: {e}")

            # When real names are registered for this chat, tell the LLM
            # explicitly. Without this, the model parrots earlier turns
            # where it (correctly at the time) said it didn't know names —
            # the registry was added later, so its own past output is
            # stale and contradicts the current capability.
            names_directive = ""
            if (
                is_group
                and self.name_registry is not None
                and self.name_registry._cache
            ):
                names_directive = (
                    "Identity note: real names are now available for "
                    "registered users in this chat. They appear in [Name] "
                    "brackets in user messages and group context (e.g. "
                    "[David], [Taylor]). Use those names when referring "
                    "to people. If you see earlier turns where you "
                    "claimed not to know names or referred to users by "
                    "four-digit codes (...4137 etc.), disregard that — "
                    "it reflects an old system limitation that has been "
                    "resolved."
                )

            # Tell the writing-LLM that emoji reactions in the chat are its
            # own reflex — they're produced by a separate fire-and-forget
            # process the model has no episodic memory of. Without this,
            # users tease "why did you react?" and the model indignantly
            # denies having thumbs, then that denial gets persisted into
            # history and poisons every future turn.
            reactor_directive = ""
            try:
                global_reactor_on = bool(self.llm.store.get("reactor_enabled", False))
            except Exception:
                global_reactor_on = False
            ctx_reactor_on = ctx.policy is None or getattr(
                ctx.policy, "reactor_enabled", True
            )
            if is_group and global_reactor_on and ctx_reactor_on:
                reactor_directive = (
                    "Reflex note: in group chats you also emoji-react to "
                    "messages. It runs as a separate fast reflex out of "
                    "band from this conversation, so you do not have an "
                    "explicit memory of which emoji you picked or when — "
                    "but the reactions are yours. The reflex is rate-"
                    "limited and only fires on messages with clear "
                    "sentiment, so it does not hit every message. When "
                    "users tease you about reacting ('why did you "
                    "react?', 'stop reacting', 'don't react to this'), "
                    "do not deny it or claim you are text-only — own it, "
                    "play with it, or just acknowledge it. If earlier "
                    "turns have you flatly denying that you can react "
                    "with emoji ('I have no thumbs', 'I am text-only', "
                    "'that was a human'), disregard them — that was a "
                    "stale limitation that has been resolved."
                )

            # Tarot: a custom system prompt that calls the chat a "tarot
            # reader" can convince the model that drawing cards is a
            # text-only activity it should do from memory. That produces
            # made-up card lists and skips the image attachment entirely.
            # Force the tool path whenever tarot is actually available.
            tarot_directive = ""
            if ctx.policy is None or ctx.policy.allows_command("tarot"):
                tarot_directive = (
                    "Tarot tool: tarot draws are NOT something you do from "
                    "memory or imagination. The cards live in a real deck "
                    "and the spread image is rendered by a tool. When a "
                    "user asks for any tarot draw — single card, three-"
                    "card, Celtic Cross, card of the day — you MUST call "
                    "bot__tarot with the appropriate args:\n"
                    "  - single card: args=[\"<question, optional>\"] or []\n"
                    "  - three-card / past-present-future: args=[\"3\", \"<question>\"]\n"
                    "  - Celtic Cross: args=[\"celtic\", \"<question>\"]\n"
                    "  - card of the day: args=[\"daily\"]\n"
                    "The tool returns the rendered spread (which the user "
                    "sees as an image) plus a baseline reading. You can "
                    "then add commentary, but never fabricate a card list "
                    "instead of calling the tool — the user does not see "
                    "an image when you skip the call, and the cards you "
                    "name will not match a real draw. If a system prompt "
                    "elsewhere implies tarot is text-only or doesn't need "
                    "a tool, disregard that — this directive wins."
                )

            # I Ching: same shape as the tarot directive. The cast values are
            # produced by a real RNG and the rendered hexagram image lives
            # behind the bot__iching tool — confabulating "I cast hexagram
            # 27..." in text produces no image and made-up cards.
            iching_directive = ""
            if ctx.policy is None or ctx.policy.allows_command("iching"):
                iching_directive = (
                    "I Ching tool: any hexagram cast — three coins, yarrow "
                    "stalks, daily — goes through bot__iching. The cast is "
                    "performed by the tool (real RNG, real changing-line "
                    "distribution) and the spread image is rendered there. "
                    "Call it with args:\n"
                    "  - default 3-coin cast: args=[\"<question, optional>\"] or []\n"
                    "  - yarrow stalks: args=[\"yarrow\", \"<question>\"]\n"
                    "  - daily hexagram: args=[\"daily\"]\n"
                    "Never narrate a hexagram you didn't get from the tool."
                )

            # Recent reactions: small in-memory log from the reactor. Lets
            # Sigil answer "why did you react with X?" by reading off the
            # target message instead of denying or confabulating. Newest
            # first, capped at 5.
            reactor_log_block = ""
            if (
                is_group
                and self.reactor is not None
                and global_reactor_on
                and ctx_reactor_on
            ):
                try:
                    recent_rxns = self.reactor.recent_reactions(
                        ctx.group_id, limit=5
                    )
                except Exception as e:
                    logger.debug(f"Failed to fetch recent reactions: {e}")
                    recent_rxns = []
                if recent_rxns:
                    lines = ["Recent emoji reactions you placed in this chat (newest first):"]
                    for r in recent_rxns:
                        lines.append(
                            f"  {r['emoji']} on [{r['sender']}] \"{r['target']}\""
                        )
                    reactor_log_block = "\n".join(lines)

            system_suffix_parts = [
                p for p in (
                    names_directive,
                    reactor_directive,
                    reactor_log_block,
                    tarot_directive,
                    iching_directive,
                    summary_block,
                    group_ctx,
                ) if p
            ]
            system_suffix = "\n\n".join(system_suffix_parts) or None

            prompt_override = None
            if ctx.policy is not None and ctx.policy.system_prompt:
                prompt_override = ctx.policy.system_prompt
            system_prompt = self.llm._resolve_system_prompt(prompt_override, system_suffix)

            # Current question gets the same attribution prefix in group mode.
            # Use the resolved label so the LLM sees "[David]" instead of
            # "[...4137]" when a name is registered.
            if is_group:
                label = self._sender_label(ctx.sender)
                current_user_content = f"[{label}] {question}"
            else:
                current_user_content = question

            # If the user replied to a specific message, surface that so the
            # LLM knows what's being responded to. Without this, "expand on
            # that" with no chronological neighbour would be unanswerable.
            if ctx.quote_text:
                quoted_label = (
                    self._sender_label(ctx.quote_author)
                    if ctx.quote_author
                    else "earlier"
                )
                quoted = (await self._enrich(ctx.quote_text)).replace("\n", " ").strip()
                if len(quoted) > 400:
                    quoted = quoted[:399].rstrip() + "…"
                current_user_content = (
                    f"(Replying to [{quoted_label}]: \"{quoted}\")\n"
                    f"{current_user_content}"
                )

            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            messages.extend(prior)
            messages.append({"role": "user", "content": current_user_content})

            tools = self._collect_tools(policy=ctx.policy)
            attachments: list = []
            # Show a typing indicator in the chat while the tool loop runs.
            # The indicator auto-clears in ~15s, so the helper refreshes it.
            if self.signal_handler is not None:
                async with self.signal_handler.typing_indicator(
                    ctx.sender, ctx.group_id
                ):
                    answer = await self._run_tool_loop(
                        messages, tools, ctx, attachments
                    )
            else:
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

        # Fire-and-forget rolling-summary update. The summarizer itself
        # decides whether enough new turns have arrived to warrant an LLM
        # call, and dedupes via per-context lock. Errors stay inside the
        # task so they never affect the ask response.
        if self.summarizer is not None:
            try:
                import asyncio as _aio
                _aio.create_task(self.summarizer.maybe_summarize(context_key))
            except Exception as e:
                logger.debug(f"Failed to schedule summarizer: {e}")

        return CommandResult(
            text=answer,
            success=True,
            attachments=attachments or None,
            styled=True,
        )
