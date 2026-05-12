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
import math
import re
import time
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from .predict_command import (
    PREDICT_FOR_TOOL,
    PREDICT_SELF_TOOL,
    PREDICT_UPDATE_TOOL,
    _format_deadline,
    extract_prediction,
)
from .portfolio_command import (
    PORTFOLIO_BUY_TOOL,
    PORTFOLIO_CANCEL_ORDER_TOOL,
    PORTFOLIO_JOURNAL_APPEND_TOOL,
    PORTFOLIO_JOURNAL_READ_TOOL,
    PORTFOLIO_PLACE_ORDER_TOOL,
    PORTFOLIO_SELL_TOOL,
    PORTFOLIO_STATUS_TOOL,
    render_status,
)
from ..charts.portfolio import render_portfolio_image
from ..database import hash_phone
from ..group_log import BOT_SENDER
from ..paper_portfolio import (
    SOURCE_CRON,
    SOURCE_REACTIVE,
    VALID_SOURCES,
)
from ..paper_portfolio_executor import PaperPortfolioExecutor
from ..predictions import PredictionStore
from ..llm import (
    LLMClient,
    LLMDisabled,
    LLMError,
    LLMNotConfigured,
    ConversationHistory,
    format_relative_age,
)
from ..memory import (
    FORGET_TOOL,
    KINDS,
    MemoryStore,
    RECALL_TOOL,
    REMEMBER_TOOL,
    SOURCE_EXPLICIT,
    SubjectResolver,
    build_preamble,
    compute_explicit_confidence,
    render_recall_results,
)

logger = logging.getLogger(__name__)

DEFAULT_MAX_TOOL_ROUNDS = 25

# Threshold above which we add an explicit "prior conversation is stale" hint
# to the system prompt. Tuned conservatively: anything older than half a day
# is likely a fresh topic, not a continuation.
STALENESS_THRESHOLD_SECONDS = 6 * 3600


# Phrases that almost always mean "use the smarter model" — when these appear
# in the user's message, we inject a strong hint pushing the writer toward
# deep_think instead of trusting the model's own judgement of "is this hard?"
# Writers under-call deep_think because plausible-sounding answers come easily
# from training; explicit user intent is a much more reliable trigger than
# difficulty self-assessment.
_DEEP_THINK_TRIGGER_RE = re.compile(
    r"\b("
    r"think\s+(?:hard|carefully|deeply|long|really)"
    r"|deep[\s-]?think"
    r"|really\s+think"
    r"|carefully\s+think"
    r"|think\s+(?:about|on)\s+(?:this|that|it)\s+(?:hard|carefully|really)"
    r"|take\s+your\s+time"
    r"|dig\s+(?:in|into|deep)"
    r"|do\s+(?:some\s+)?(?:real\s+)?research"
    r"|give\s+(?:it|this|that)\s+(?:real|some)\s+thought"
    r"|put\s+some\s+thought\s+into"
    r"|don'?t\s+(?:just\s+)?(?:guess|hand[\s-]?wave)"
    r")\b",
    re.IGNORECASE,
)


def _user_explicitly_asked_to_think(text: str) -> bool:
    return bool(_DEEP_THINK_TRIGGER_RE.search(text or ""))


# Distinctive openers that ONLY ever appear in the system prompt's directive
# blocks — never as a legitimate user-facing reply. The model occasionally
# pattern-matches on these labels and copies them into its visible output
# (we've seen `[to David]`, `Spontaneous reply:`, `Reflex note:`...). The
# system prompt tells it not to; these regexes strip the leak as belt-and-
# braces. Patterns are anchored to the start of the answer because mid-reply
# uses of these phrases (e.g. quoting them in an explanation) are fine.
_META_LEAK_PATTERNS = (
    # Assistant-turn addressee label: `[to David, 2m ago]`
    re.compile(r"^\s*\[to [^\]\n]{1,80}\]\s*", re.IGNORECASE),
    # User-turn speaker label: `[J, just now]`, `[David, 2m ago]`,
    # `[...4137, 5h ago]`. Same shape the history load path produces
    # for user messages in groups; some models mimic it as a header
    # for their own reply. Tightly constrained — must contain a
    # comma + a time-like phrase — so generic bracket-bullets
    # ("[1] foo", "[note] bar") survive untouched.
    re.compile(
        r"^\s*\["
        r"[^\]\n,]{1,40},\s*"
        r"(?:just now|a moment ago|"
        r"\d+\s*(?:[smhdw]|min(?:ute)?|sec(?:ond)?|hour|day|week)s?"
        r"(?:\s*ago)?)"
        r"\s*\]\s*",
        re.IGNORECASE,
    ),
    # Implicit-ask path system block opener
    re.compile(r"^\s*Spontaneous[- ]reply[ -]?path?:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Spontaneous reply:[^\n]*\n?", re.IGNORECASE),
    # Other directive labels that the model has been observed mimicking
    re.compile(r"^\s*Reflex note:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Identity note:[^\n]*\n?", re.IGNORECASE),
    re.compile(r"^\s*Attribution rules?[^\n]*\n?", re.IGNORECASE),
)


def _strip_meta_leak(text: str) -> str:
    """Remove directive/system-prompt text the model copied into output.

    Some models start their reply with the heading of the directive block
    they were just given (`Spontaneous reply: ...`, `[to David] ...`). The
    response-style rule tells them not to, but this is the safety net so
    users never see internal scaffolding as literal output text — and it
    prevents stacked leaks from round-tripping through history.
    """
    if not text:
        return text
    # Loop because leaks can stack (e.g. `[to David] Spontaneous reply: ...`)
    # and a single pass would only catch the outermost. Bounded so a degenerate
    # model output can't burn cycles here.
    for _ in range(4):
        new = text
        for pat in _META_LEAK_PATTERNS:
            new = pat.sub("", new, count=1)
        if new == text:
            break
        text = new
    return text


# Backwards-compat shim — older tests and callers may reference the
# narrower addressee-only stripper. The general stripper is a strict
# superset, so aliasing is safe.
_strip_addressee_leak = _strip_meta_leak


# Tool exposed to the writer LLM when DeepThinkClient is configured + ready.
# Single tool, no namespace prefix — distinguishes itself by literal name in
# the dispatch path inside _execute_tool_call.
_DEEP_THINK_TOOL_SCHEMA = {
    "type": "function",
    "function": {
        "name": "deep_think",
        "description": (
            "Delegate ONE focused hard sub-problem to a smarter, slower "
            "model that ALSO HAS THE SAME TOOL KIT YOU DO (price, chart, "
            "news, MCP servers — everything except deep_think itself). "
            "Use when a question is genuinely hard: multi-step reasoning, "
            "careful comparisons, research that needs to chain several "
            "tool calls, or anything where you'd hand-wave. The deep "
            "model will fetch its own data — you don't need to pre-load "
            "results into context unless they're already in the chat. "
            "Do NOT use for simple lookups you can do directly, tarot/"
            "iching draws, or banter. Returns the smart model's text "
            "which you weave into your reply (don't paste verbatim). "
            "Slow (10-90s typical) — the bot will send your "
            "`status_message` to the chat immediately when you invoke "
            "this so the user knows you're working on it. On "
            "'(unavailable: ...)' or '(rate-limited: ...)' stubs, just "
            "answer without it."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "question": {
                    "type": "string",
                    "description": (
                        "The precise sub-question to think hard about. "
                        "Be specific — this is sent verbatim to a fresh "
                        "model with no other context unless you supply it."
                    ),
                },
                "context": {
                    "type": "string",
                    "description": (
                        "Optional supporting context (history snippets, "
                        "the user's framing, etc.) the deep model should "
                        "consider. Capped to 8000 chars by the bot."
                    ),
                },
                "status_message": {
                    "type": "string",
                    "description": (
                        "A short message sent to the chat IMMEDIATELY "
                        "when this tool fires, before the deep model "
                        "runs (which takes 10-90s). Tell the user you're "
                        "thinking hard so they don't think the bot is "
                        "stuck. Write it in YOUR OWN voice for THIS "
                        "specific chat — match the persona, tone, and "
                        "language you're using. Vary the wording each "
                        "time, don't copy-paste a stock phrase. Keep it "
                        "under 100 chars and casual. Examples (English "
                        "neutral — adapt to your context): 'gimme a sec "
                        "to dig into this', 'hold on, this one needs "
                        "real thought', 'lemme actually work this out — "
                        "back in a minute'."
                    ),
                },
            },
            "required": ["question", "status_message"],
        },
    },
}


# Patterns that strip identifying tokens from tool-result content before
# the model sees them. The fetch MCP echoes our outbound user-agent in
# robots.txt errors — without scrubbing, every blocked fetch leaks the
# legacy "Sigil" persona into in-context learning and the writer flips
# identity. Anything resembling a useragent / version / repo tag gets
# nuked. Keep this list narrow — overzealous scrubbing destroys
# legitimate tool output too.
_USERAGENT_TAG_RE = re.compile(
    r"<useragent>[^<]*</useragent>", re.IGNORECASE
)
_HEADER_UA_LINE_RE = re.compile(
    r"^[ \t]*User-Agent[ \t]*:[ \t]*[^\r\n]+",
    re.IGNORECASE | re.MULTILINE,
)


def _scrub_tool_content(content: str) -> str:
    """Remove persona-revealing identifiers from a tool result.

    Currently strips `<useragent>...</useragent>` blocks (emitted by
    `mcp-server-fetch` on robots.txt blocks) and bare `User-Agent: ...`
    header echoes. The bot's name appearing inside a tool error
    response is interpreted by the writer as "this is who I am" via
    in-context learning, so even small leaks compound across turns.
    """
    if not isinstance(content, str) or not content:
        return content
    out = _USERAGENT_TAG_RE.sub("(useragent redacted)", content)
    out = _HEADER_UA_LINE_RE.sub("User-Agent: (redacted)", out)
    return out


def _wrap_xml(tag: str, body: str) -> str:
    """Wrap a block of text in semantic XML tags for the model.

    Empty/whitespace bodies short-circuit to "" so callers can compose
    suffix blocks unconditionally and rely on the surrounding `if` filter
    to drop the empty ones.
    """
    inner = (body or "").strip()
    if not inner:
        return ""
    return f"<{tag}>\n{inner}\n</{tag}>"


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
        deep_think=None,
        memory_store: Optional[MemoryStore] = None,
        prediction_store: Optional[PredictionStore] = None,
        portfolio_executor: Optional[PaperPortfolioExecutor] = None,
        portfolio_journal=None,
        llm_factory=None,
    ):
        self.llm = llm
        # Per-bot writer/deep_think router. When set (always in the real
        # app; None only in older tests that hand-roll AskCommand),
        # `_llm_for(ctx)` and `_deep_think_for(ctx)` resolve to the
        # bot-scoped clients so per-bot model/temperature/extra_body
        # actually take effect. Falls back to the constructor-supplied
        # `llm`/`deep_think` when ctx.bot is None.
        self.llm_factory = llm_factory
        self.history = history
        self.group_log = group_log
        self.mcp_manager = mcp_manager
        self.bot_tools = bot_tools
        # Optional MemoryStore — when set, the writer LLM gets remember/
        # recall/forget tools (gated per-context via memory_writes_enabled)
        # and stored memories about active speakers in the chat are
        # auto-injected into the system suffix.
        self.memory_store = memory_store
        # Optional PredictionStore — when set and the per-context policy
        # allows the !predict command, the writer LLM gets a `predict_self`
        # tool letting Sigil log its own forecasts under a bot-author row
        # so its calls appear on the leaderboard alongside humans.
        self.prediction_store = prediction_store
        # Optional paper-portfolio executor — when set and the per-context
        # policy allows the !portfolio command, the writer LLM gets
        # portfolio_buy / portfolio_sell / portfolio_status tools so
        # Sigil can paper-trade reactively in chat. The cron worker
        # holds its own executor reference for scheduled decision points.
        self.portfolio_executor = portfolio_executor
        # Per-portfolio markdown journal — bot's free-form notebook for
        # narrative reflection on its trading. Surfaced as
        # portfolio_journal_append / portfolio_journal_read tools when
        # both the journal is wired AND the chat allows portfolio. Kept
        # separate from MemoryStore: the journal is paragraph-style
        # narrative, MemoryStore is structured key-value facts.
        self.portfolio_journal = portfolio_journal
        self.subject_resolver: Optional[SubjectResolver] = None
        # Optional DeepThinkClient — when set and the per-context policy
        # allows it, exposed to the writer LLM as a `deep_think` tool.
        # The client itself reads its own enabled flag at call time, so a
        # disabled or unconfigured client just returns "(unavailable)" stubs
        # the writer can integrate or discard.
        self.deep_think = deep_think
        # Optional message-text enricher (e.g. TwitterExpander) — called on the
        # user's question so pasted links carry their content into the prompt.
        self.enricher = enricher
        # Optional Signal handler — used only to drive the typing indicator
        # while the LLM tool loop runs. None is fine; the indicator is a UX
        # nicety, not a correctness requirement.
        self.signal_handler = signal_handler
        # Set by main.py post-construction. When the pool is wired, the
        # typing indicator and tool-loop status messages route through
        # `pool.for_bot(ctx.bot)` so multi-phone installs do this from
        # the right number. Falls back to `self.signal_handler` when None.
        from typing import Any as _Any
        self.signal_pool: _Any = None
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
        # Late-bound after construction (NameRegistry shares the same
        # late-binding pattern as bot_tools / signal_handler).
        if name_registry is not None:
            self.subject_resolver = SubjectResolver(name_registry)

    def _llm_for(self, ctx) -> "LLMClient":
        """Resolve the writer LLMClient for this command's active bot.
        Falls back to the constructor-supplied default when no factory
        is wired (older tests) or no bot is set on the context. The
        cached client persists its aiohttp session across calls."""
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        if self.llm_factory is None or bot is None or bot.id is None:
            return self.llm
        return self.llm_factory.get_writer(bot.id)

    def _deep_think_for(self, ctx):
        """Same as _llm_for but for the deep_think role. Returns the
        constructor-supplied default when no factory is wired or no
        bot is set."""
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        if self.llm_factory is None or bot is None or bot.id is None:
            return self.deep_think
        return self.llm_factory.get_deep_think(bot.id)

    def _live_turns(self, ctx=None) -> int:
        """Resolve turns-per-user, preferring the per-context override.

        `history_turns_override = 0` is an explicit "no history" — useful
        for one-shot chats. Only `None` means "inherit global". Bounded at
        0 so a negative override can't crash the LIMIT clause downstream.
        """
        policy = getattr(ctx, "policy", None) if ctx is not None else None
        if policy is not None:
            override = getattr(policy, "history_turns_override", None)
            if override is not None:
                return max(0, int(override))
        return self.llm.store.get_int("llm_history_turns", 6, min_value=0)

    def _live_group_ctx(self) -> int:
        return self.llm.store.get_int("group_context_messages", 0, min_value=0)

    def _live_max_tool_rounds(self) -> int:
        return self.llm.store.get_int(
            "llm_max_tool_rounds", DEFAULT_MAX_TOOL_ROUNDS, min_value=1
        )

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

    async def _build_group_context(self, ctx, now: float) -> str:
        """Render recent group chat, oldest-first, wrapped for the model.

        Each line gets `[Sender, 5m ago] text` so the model can tell which
        bits of context are seconds-old vs. hours-old. Returns the full
        `<group_context>...</group_context>` block, or "" if there's nothing
        to show.
        """
        limit = self._live_group_ctx()
        if limit <= 0 or not ctx.is_group or self.group_log is None:
            return ""
        msgs = await self.group_log.recent(ctx.group_id, limit=limit, exclude_last=1)
        if not msgs:
            return ""

        # Enrich messages concurrently. _enrich short-circuits on empty
        # input so empty cells cost ~nothing; gathering avoids 30 sequential
        # awaits stacking up on the LLM hot path.
        raw_texts = [(m["text"] or "").strip() for m in msgs]
        enriched = await asyncio.gather(*(self._enrich(t) for t in raw_texts))

        # Resolve the active bot's display name once so every BOT_SENDER
        # line in this block is labeled with the CURRENT persona — not
        # whatever bot wrote the row (the group_log only stores a
        # BOT_SENDER sentinel, not which specific bot it was). Without
        # this, Artaud answering in a chat that previously had Sigil
        # would see prior bot turns labeled `[Sigil, Nm ago]` and
        # contradict its own identity via in-context learning. The
        # active-persona label also future-proofs against renames.
        active_bot_name = self._active_bot_display(ctx)

        lines: list[str] = []
        for m, text in zip(msgs, enriched):
            if not text:
                continue
            if m["sender"] == BOT_SENDER:
                label = active_bot_name
            else:
                label = self._sender_label(m["sender"])
            ts = m.get("created_at")
            ago = format_relative_age(now - ts) if ts is not None else None
            bracket = f"{label}, {ago}" if ago else label
            lines.append(f"[{bracket}] {text.replace(chr(10), ' ')}")
        if not lines:
            return ""
        return _wrap_xml("group_context", "\n".join(lines))

    def _active_bot_display(self, ctx) -> str:
        """Display name to attribute the bot's own past turns to.

        Priority: the bot currently answering (`ctx.bot`) — so the model
        sees its history in its own voice. Falls back to the seed
        `name_registry.bot_name` (legacy single-bot path) or "Bot".
        """
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        if bot is not None and getattr(bot, "display_name", None):
            return bot.display_name
        if self.name_registry is not None:
            return self.name_registry.bot_name
        return "Bot"

    def _sender_label(self, phone: Optional[str]) -> str:
        if phone == BOT_SENDER:
            return (
                self.name_registry.bot_name
                if self.name_registry is not None
                else "Bot"
            )
        if self.name_registry is None:
            return f"...{(phone or '')[-4:] or '????'}"
        return self.name_registry.label_for(phone)

    def _collect_tools(self, policy=None, bot=None) -> Optional[list[dict]]:
        schemas: list[dict] = []
        if self.bot_tools is not None:
            schemas.extend(self.bot_tools.list_tools(policy=policy))
        if self.mcp_manager is not None:
            mcp_tools = self.mcp_manager.all_tools()
            if policy is not None:
                mcp_tools = [t for t in mcp_tools if policy.allows_mcp(t.server_name)]
            schemas.extend(t.to_openai_tool() for t in mcp_tools)
        # deep_think is exposed only when the client is wired AND the global
        # flag is on AND the per-context policy permits AND the per-bot
        # `deep_think_enabled` is True. The client returns "(unavailable)"
        # for disabled/unconfigured calls, but suppressing the schema
        # entirely keeps the writer from wasting tool-call rounds on a
        # guaranteed-stub when we already know it's off. The per-bot flag
        # is a master kill — admins flip it off when they want deep_think
        # completely out of one bot's repertoire.
        if self.deep_think is not None:
            dt_status = self.deep_think.status()
            policy_ok = policy is None or policy.allows_deep_think()
            bot_ok = bot is None or getattr(bot, "deep_think_enabled", True)
            if dt_status.get("ready") and policy_ok and bot_ok:
                schemas.append(_DEEP_THINK_TOOL_SCHEMA)
        # Memory tools: recall is always exposed when a store is wired and
        # the policy resolves to a real (non-default) row; remember/forget
        # require the per-context memory_writes_enabled flag. Default rows
        # are excluded so passive-learning writes don't bleed across
        # unregistered DMs sharing the default:dm policy.
        if (
            self.memory_store is not None
            and policy is not None
            and policy.id is not None
            and policy.kind != "default"
        ):
            schemas.append(RECALL_TOOL)
            if getattr(policy, "memory_writes_enabled", True):
                schemas.append(REMEMBER_TOOL)
                schemas.append(FORGET_TOOL)
        # predict_self: the bot's own prediction logging tool. Distinct from
        # `bot__predict` (which the writer can already invoke to log a
        # prediction on the asker's behalf) — this one authors as the bot
        # under a sentinel hash so Sigil shows up on the leaderboard. Only
        # exposed when the prediction store is wired AND the per-context
        # policy allows the !predict command at all.
        if self.prediction_store is not None and (
            policy is None or policy.allows_command("predict")
        ):
            schemas.append(PREDICT_SELF_TOOL)
            schemas.append(PREDICT_FOR_TOOL)
            schemas.append(PREDICT_UPDATE_TOOL)
        # Paper-portfolio tools — exposed when the executor is wired AND
        # the per-context policy allows the !portfolio command. Same
        # gating shape as predict tools above: a chat that hasn't opted
        # into the portfolio feature shouldn't see the trade tools.
        if self.portfolio_executor is not None and (
            policy is None or policy.allows_command("portfolio")
        ):
            schemas.append(PORTFOLIO_STATUS_TOOL)
            schemas.append(PORTFOLIO_BUY_TOOL)
            schemas.append(PORTFOLIO_SELL_TOOL)
            schemas.append(PORTFOLIO_PLACE_ORDER_TOOL)
            schemas.append(PORTFOLIO_CANCEL_ORDER_TOOL)
            # Journal tools are gated separately because the journal
            # store is wired independently — a deploy could have
            # portfolio enabled but no journal directory configured,
            # in which case we hide the tools rather than expose
            # broken endpoints.
            if self.portfolio_journal is not None:
                schemas.append(PORTFOLIO_JOURNAL_APPEND_TOOL)
                schemas.append(PORTFOLIO_JOURNAL_READ_TOOL)
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
        # Resolve the writer client per-bot. _llm_for handles the
        # ctx.bot=None fallback so legacy callers (no factory wired)
        # still hit the singleton self.llm. Each call re-resolves so a
        # late-changed ctx.bot would still pick up the right client,
        # but in practice ctx.bot is fixed for the lifetime of one
        # command.
        llm = self._llm_for(caller_ctx)
        max_rounds = self._live_max_tool_rounds()
        tool_call_history: list[str] = []

        for round_idx in range(max_rounds):
            logger.info(
                f"LLM round {round_idx + 1}/{max_rounds}: "
                f"requesting completion ({len(messages)} msgs in context)"
            )
            assistant_msg = await llm.chat_messages(messages, tools=tools)
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
                await self._execute_tool_call(
                    call, messages, caller_ctx, attachments, tools=tools,
                )

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

    _DEFAULT_RESEARCH_HANDOFF = (
        "RESEARCH NOTES (from your research model — the slower, smarter "
        "deeper-thinking sibling that just ran the tool loop on your "
        "behalf):\n\n{notes}\n\nUse these notes to compose your reply. "
        "Do not call tools; the research is already done. Stay in your "
        "voice — the notes are scaffolding, not the response. If the "
        "notes admit they couldn't answer, say so honestly rather than "
        "fabricating."
    )

    async def _run_research_handoff(
        self,
        *,
        ctx: CommandContext,
        question: str,
        research_input: str,
        messages: list[dict],
        attachments: list,
        user_hash: str,
    ) -> str:
        """Two-LLM research-mode handoff: deep_think does the tool work
        and produces structured notes, then the writer composes the
        final reply with those notes injected. Used when
        `ctx.bot.deep_think_mode == 'research'`. The writer gets no
        tools — research is already done."""
        # Resolve per-bot deep_think + writer. For Artaud the deep_think
        # is the cloud research model with the toolset, the writer is
        # the local MLX model that owns the persona; routing through
        # the factory means each invocation hits the bot's own clients.
        deep_think = self._deep_think_for(ctx)
        writer = self._llm_for(ctx)
        # Defensive copy: we mutate `messages[0]` below (replacing the
        # system message with one that includes the research notes).
        # If the caller passed a list it intends to reuse, mutating
        # in place would leak notes into a future call.
        messages = list(messages)

        # Run deep_think against the same context the writer would have
        # seen. We pass the user-message block as `context` so deep_think
        # has the full situational frame (group context, replying_to,
        # current message). Failure modes return placeholder strings
        # rather than raising — the writer will see them and report
        # back honestly.
        #
        # Images are NOT routed through deep_think — the writer is the
        # persona doing the talking, and when its model is vision-capable
        # the user wants it to read pixels natively. The multimodal payload
        # already lives on `messages[-1]` and reaches the writer below.
        notes = await deep_think.think(
            question,
            context=research_input,
            user_hash=user_hash,
            group_id=ctx.group_id,
            caller_ctx=ctx,
            attachments=attachments,
        )

        # Inject notes into the system message via a final sub-block.
        # The handoff prompt is per-bot (set in /admin/bots) so each
        # bot can phrase the handoff in its own voice; an empty value
        # falls back to the default scaffolding above.
        handoff_template = (
            getattr(ctx.bot, "deep_think_handoff_prompt", None)
            or self._DEFAULT_RESEARCH_HANDOFF
        )
        # Allow either {notes} or a literal handoff prompt followed by
        # the notes — the {notes} form gives admins control over where
        # the notes appear; falling back to "<prompt>\n\n<notes>"
        # avoids forcing them to remember the placeholder. Catch broad
        # because str.format raises ValueError on stray '{' too, and
        # we don't want a malformed admin-supplied template to crash
        # the entire !ask in chat.
        try:
            handoff_block = handoff_template.format(notes=notes)
        except (KeyError, IndexError, ValueError):
            handoff_block = f"{handoff_template}\n\n{notes}"

        # Prepend to the system message in-place. Using a system role
        # (not user) keeps the writer's framing as "compose a reply"
        # rather than "respond to the notes."
        if messages and messages[0].get("role") == "system":
            existing = (messages[0].get("content") or "").rstrip()
            messages[0] = dict(messages[0])
            messages[0]["content"] = (
                f"{existing}\n\n{handoff_block}" if existing else handoff_block
            )
        else:
            messages.insert(0, {"role": "system", "content": handoff_block})

        # Single-shot: no tools. The writer's only job is to compose.
        assistant_msg = await writer.chat_messages(messages, tools=None)
        return (assistant_msg.get("content") or "").strip()

    async def _handle_predict_self_tool(
        self,
        args: dict,
        caller_ctx: CommandContext,
    ) -> str:
        """Log a prediction authored by the bot itself.

        Uses the BOT_SENDER sentinel as the user_hash source so the bot's
        forecasts land on the leaderboard as a distinct row, separate from
        whichever human asked. The same parsing pipeline that backs
        !predict is reused so structured stock-shape claims still
        auto-resolve via live price.
        """
        store = self.prediction_store
        if store is None:
            return "(predict_self unavailable: no prediction store wired)"
        # Defence-in-depth policy check (also gated when filtering tool
        # schemas). Only applies when a policy is present.
        policy = caller_ctx.policy
        if policy is not None and not policy.allows_command("predict"):
            return "(predict_self unavailable: !predict not allowed in this chat)"

        claim_text = str(args.get("claim") or "").strip()
        if not claim_text:
            return "ERROR: predict_self requires a non-empty claim with a deadline."

        parsed, err = await extract_prediction(claim_text, llm_client=self.llm)
        if err is not None:
            return f"ERROR: {err}"
        assert parsed is not None  # extract_prediction post-condition

        bot_label = (
            self.name_registry.bot_name
            if self.name_registry is not None
            else "Bot"
        )
        try:
            pred_id = await store.create(
                user_hash=hash_phone(BOT_SENDER),
                user_label=bot_label,
                group_id=caller_ctx.group_id,
                context_key=caller_ctx.context_key(),
                claim=parsed["claim"],
                deadline_utc=parsed["deadline_utc"],
                ticker=parsed.get("ticker"),
                threshold=parsed.get("threshold"),
                direction=parsed.get("direction"),
            )
        except Exception as e:
            logger.exception(f"predict_self: store.create failed: {e}")
            return "ERROR: couldn't save the prediction."

        deadline_str = _format_deadline(parsed["deadline_utc"])
        kind_note = ""
        if parsed.get("ticker"):
            kind_note = (
                f" (auto-resolves: {parsed['ticker']} "
                f"{parsed['direction']} ${parsed['threshold']:g})"
            )
        return (
            f"Logged your own prediction #{pred_id}: "
            f"\"{parsed['claim']}\" due {deadline_str}{kind_note}. "
            f"It's on the leaderboard under {bot_label}."
        )

    async def _handle_predict_update_tool(
        self,
        args: dict,
        caller_ctx: CommandContext,
    ) -> str:
        """Revise a still-pending prediction's claim/deadline within the
        15-minute grace window. Re-uses extract_prediction so the new
        claim goes through the same parser as the original."""
        store = self.prediction_store
        if store is None:
            return "(predict_update unavailable: no prediction store wired)"
        policy = caller_ctx.policy
        if policy is not None and not policy.allows_command("predict"):
            return "(predict_update unavailable: !predict not allowed in this chat)"

        try:
            pred_id = int(args.get("id") or 0)
        except (TypeError, ValueError):
            return "ERROR: predict_update requires an integer prediction id."
        if pred_id <= 0:
            return "ERROR: predict_update requires a positive id."
        claim_text = str(args.get("claim") or "").strip()
        if not claim_text:
            return "ERROR: predict_update requires a non-empty new claim."

        parsed, err = await extract_prediction(claim_text, llm_client=self.llm)
        if err is not None:
            return f"ERROR: {err}"
        assert parsed is not None

        # Caller-as-predictor enforcement: the LLM tool can only revise
        # the asker's own pending prediction, not someone else's. Without
        # this, a prompt-injected message in a group chat could coax
        # Sigil into rewriting another member's claim within their
        # 15-min grace window.
        caller_user_hash = (
            hash_phone(caller_ctx.sender) if caller_ctx.sender else None
        )

        try:
            status = await store.update_pending(
                pred_id,
                claim=parsed["claim"],
                deadline_utc=parsed["deadline_utc"],
                ticker=parsed.get("ticker"),
                threshold=parsed.get("threshold"),
                direction=parsed.get("direction"),
                expected_user_hash=caller_user_hash,
            )
        except Exception as e:
            logger.exception(f"predict_update: store.update_pending failed: {e}")
            return "ERROR: couldn't update the prediction."

        if status == "not_found":
            return f"ERROR: no prediction #{pred_id}."
        if status == "not_owner":
            return (
                f"ERROR: prediction #{pred_id} belongs to a different chat "
                f"member. You can only update your own predictions."
            )
        if status == "not_pending":
            return (
                f"ERROR: #{pred_id} is already resolved or expired. "
                f"Admin can override the verdict from the dashboard if "
                f"the auto-resolution was wrong."
            )
        if status == "stale":
            return (
                f"ERROR: #{pred_id} is past the 15-minute edit window. "
                f"Predictions are locked after that to keep the "
                f"leaderboard honest. Ask an admin to fix it from the "
                f"dashboard if it's genuinely wrong."
            )
        if status != "ok":
            return f"ERROR: unexpected store status {status!r}."

        deadline_str = _format_deadline(parsed["deadline_utc"])
        kind_note = ""
        if parsed.get("ticker"):
            kind_note = (
                f" (auto-resolves: {parsed['ticker']} "
                f"{parsed['direction']} ${parsed['threshold']:g})"
            )
        return (
            f"Updated prediction #{pred_id}: "
            f"\"{parsed['claim']}\" due {deadline_str}{kind_note}."
        )

    async def _handle_portfolio_tool(
        self,
        name: str,
        args: dict,
        caller_ctx: CommandContext,
        attachments: list,
    ) -> str:
        """Dispatch portfolio_buy / portfolio_sell / portfolio_status.

        Returns a human-readable result string fed back to the LLM so it
        can reference the fill (or rejection) in its reply. All three
        tools are defence-in-depth gated by `policy.allows_command(
        "portfolio")`; the schema filter in `_collect_tools` already
        hides them from the writer when the policy disallows.

        For `portfolio_status` we also append a rendered dashboard PNG
        to `attachments` so the user sees the portfolio as an image
        (Signal renders monospace tables badly). The text returned to
        the LLM still includes the full structured data so the model
        can discuss specific positions without needing vision.
        """
        executor = self.portfolio_executor
        if executor is None:
            return f"({name} unavailable: no portfolio executor wired)"
        policy = caller_ctx.policy
        if policy is not None and not policy.allows_command("portfolio"):
            return f"({name} unavailable: portfolio not allowed in this chat)"

        ctx_key = caller_ctx.context_key()
        # Trade provenance: trust the cron's tag if set, otherwise
        # default to "reactive" (a real chat message triggered this).
        # `automation_source` is enumerated against VALID_SOURCES so
        # an unrecognized tag silently downgrades to reactive instead
        # of crashing inside the store's strict source check.
        source_tag = caller_ctx.automation_source or SOURCE_REACTIVE
        if source_tag not in VALID_SOURCES:
            logger.warning(
                f"unknown automation_source {source_tag!r}; "
                f"falling back to {SOURCE_REACTIVE!r}"
            )
            source_tag = SOURCE_REACTIVE

        if name == "portfolio_status":
            try:
                snap = await executor.status(ctx_key)
            except Exception as e:
                logger.exception(f"portfolio_status failed: {e}")
                return f"ERROR: couldn't load portfolio status: {type(e).__name__}"

            # Attach the portfolio dashboard image so the user sees the
            # numbers in a clean PNG. Failures are non-fatal — the LLM
            # still gets the text data and can reply without the image.
            try:
                image_b64 = render_portfolio_image(
                    snap, bot_name=self._bot_label(),
                )
                attachments.append(image_b64)
                image_note = (
                    "\n\n[NOTE TO MODEL: a portfolio dashboard image is "
                    "being attached to your reply automatically — the "
                    "user will see the holdings, equity, and PnL "
                    "rendered as a polished card. Do NOT reproduce "
                    "this as a markdown table, monospace block, or "
                    "bulleted list in your text. Reply in plain prose: "
                    "comment on what's interesting, what you're "
                    "watching, or what you'd change. If the user just "
                    "asked to see the portfolio with no follow-up, a "
                    "single short sentence is enough.]"
                )
            except Exception as e:
                logger.warning(
                    f"portfolio_status: image render failed, "
                    f"continuing without attachment: {e}"
                )
                image_note = ""

            return render_status(snap, bot_name=self._bot_label()) + image_note

        if name == "portfolio_buy":
            ticker = str(args.get("ticker") or "").strip().upper()
            reason = str(args.get("reason") or "").strip()
            if not ticker:
                return "ERROR: portfolio_buy requires a ticker."
            if not reason:
                return "ERROR: portfolio_buy requires a one-sentence reason."
            dollars = args.get("dollars")
            qty = args.get("qty")
            try:
                dollars_f = float(dollars) if dollars is not None else None
                qty_f = float(qty) if qty is not None else None
            except (TypeError, ValueError):
                return "ERROR: dollars/qty must be numeric."
            if dollars_f is not None and not math.isfinite(dollars_f):
                return "ERROR: dollars must be a finite number."
            if qty_f is not None and not math.isfinite(qty_f):
                return "ERROR: qty must be a finite number."
            try:
                result = await executor.execute_buy(
                    ctx_key,
                    ticker=ticker,
                    dollars=dollars_f,
                    qty=qty_f,
                    reason=reason,
                    source=source_tag,
                )
            except Exception as e:
                logger.exception(f"portfolio_buy failed: {e}")
                return f"ERROR: buy failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'buy rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            return (
                f"Bought {result['qty_after']:.4f} {ticker} "
                f"@ ${result['price']:.2f} (cost ${result['proceeds']:,.2f}). "
                f"Position avg ${result['avg_cost_after']:.2f}. "
                f"Cash now ${result['cash_after']:,.2f}. "
                f"Reason: {reason}"
            )

        if name == "portfolio_sell":
            ticker = str(args.get("ticker") or "").strip().upper()
            reason = str(args.get("reason") or "").strip()
            if not ticker:
                return "ERROR: portfolio_sell requires a ticker."
            if not reason:
                return "ERROR: portfolio_sell requires a one-sentence reason."
            qty_arg = args.get("qty")
            try:
                result = await executor.execute_sell(
                    ctx_key,
                    ticker=ticker,
                    qty=qty_arg,
                    reason=reason,
                    source=source_tag,
                )
            except Exception as e:
                logger.exception(f"portfolio_sell failed: {e}")
                return f"ERROR: sell failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'sell rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            pnl = result.get("realized_pnl") or 0.0
            pnl_part = (
                f" Realized P/L ${pnl:+,.2f}." if abs(pnl) > 0.005 else ""
            )
            remaining = result.get("qty_after") or 0.0
            remaining_part = (
                f" Remaining: {remaining:.4f} shares."
                if remaining > 0 else " Position closed."
            )
            return (
                f"Sold {ticker} @ ${result['price']:.2f} for "
                f"${result['proceeds']:,.2f}.{pnl_part}{remaining_part} "
                f"Cash now ${result['cash_after']:,.2f}. Reason: {reason}"
            )

        if name == "portfolio_place_order":
            ticker = str(args.get("ticker") or "").strip().upper()
            side = str(args.get("side") or "").strip().lower()
            kind = str(args.get("kind") or "").strip().lower()
            reason = str(args.get("reason") or "").strip()
            if not ticker:
                return "ERROR: portfolio_place_order requires a ticker."
            if not reason:
                return "ERROR: portfolio_place_order requires a one-sentence reason."
            trigger_raw = args.get("trigger_price")
            if trigger_raw is None:
                return "ERROR: trigger_price is required."
            try:
                trigger_price = float(trigger_raw)
            except (TypeError, ValueError):
                return "ERROR: trigger_price must be a number."
            qty = args.get("qty")
            dollars = args.get("dollars")
            close_position = bool(args.get("close_position") or False)
            try:
                qty_f = float(qty) if qty is not None else None
                dollars_f = float(dollars) if dollars is not None else None
            except (TypeError, ValueError):
                return "ERROR: qty/dollars must be numeric."
            expires_in_days_raw = args.get("expires_in_days")
            try:
                expires_in_days = (
                    float(expires_in_days_raw)
                    if expires_in_days_raw is not None else 30.0
                )
            except (TypeError, ValueError):
                return "ERROR: expires_in_days must be a number."
            # Clamp to a sane window — keep the LLM honest about how
            # long stale orders should hang around.
            expires_in_days = max(1.0, min(90.0, expires_in_days))
            try:
                result = await executor.place_order(
                    ctx_key,
                    ticker=ticker, side=side, kind=kind,
                    trigger_price=trigger_price,
                    qty=qty_f, dollars=dollars_f,
                    close_position=close_position,
                    reason=reason,
                    expires_in_days=expires_in_days,
                )
            except Exception as e:
                logger.exception(f"portfolio_place_order failed: {e}")
                return f"ERROR: place_order failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'order rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            warn_part = ""
            if result.get("warning"):
                warn_part = f" WARNING: {result['warning']}"
            current_part = ""
            if result.get("current_price") is not None:
                current_part = f" (current ${result['current_price']:.2f})"
            return (
                f"Placed order #{result['order_id']}: {kind}-{side} "
                f"{ticker} @ ${result['trigger_price']:.2f}{current_part}. "
                f"Will fire automatically when triggered (~5 min poll "
                f"during market hours).{warn_part} Reason: {reason}"
            )

        if name == "portfolio_cancel_order":
            order_raw = args.get("order_id")
            if order_raw is None:
                return "ERROR: order_id is required."
            try:
                order_id = int(order_raw)
            except (TypeError, ValueError):
                return "ERROR: order_id must be an integer."
            if order_id <= 0:
                return "ERROR: order_id must be positive."
            try:
                result = await executor.cancel_order(ctx_key, order_id)
            except Exception as e:
                logger.exception(f"portfolio_cancel_order failed: {e}")
                return f"ERROR: cancel_order failed: {type(e).__name__}"
            if not result.get("ok"):
                err = result.get("error") or "rejected"
                # Translate the store's status codes into something the
                # LLM can react to without having to know the
                # vocabulary.
                friendly = {
                    "not_found": f"no order #{order_id} exists",
                    "wrong_context": (
                        f"order #{order_id} belongs to a different chat"
                    ),
                    "not_pending": (
                        f"order #{order_id} is already filled, cancelled, "
                        f"or expired — nothing to cancel"
                    ),
                }.get(err, err)
                return f"ERROR: {friendly}"
            caller_ctx.portfolio_mutation_count += 1
            return f"Cancelled order #{order_id}."

        return f"ERROR: unknown portfolio tool {name!r}"

    async def _handle_journal_tool(
        self,
        name: str,
        args: dict,
        caller_ctx: CommandContext,
    ) -> str:
        """Dispatch portfolio_journal_append / portfolio_journal_read.

        Same policy gate as the portfolio tools: a chat that isn't
        opted into the portfolio feature shouldn't have a journal
        either. Defense-in-depth — _collect_tools already filters
        these out when the policy disallows.
        """
        journal = self.portfolio_journal
        if journal is None:
            return f"({name} unavailable: journal not configured)"
        policy = caller_ctx.policy
        if policy is not None and not policy.allows_command("portfolio"):
            return f"({name} unavailable: portfolio not allowed in this chat)"

        ctx_key = caller_ctx.context_key()

        if name == "portfolio_journal_append":
            entry = str(args.get("entry") or "").strip()
            if not entry:
                return "ERROR: portfolio_journal_append requires a non-empty entry."
            try:
                result = await journal.append(ctx_key, entry)
            except Exception as e:
                logger.exception(f"portfolio_journal_append failed: {e}")
                return f"ERROR: journal append failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'append rejected')}"
            return (
                f"Journal entry saved at {result['ts']} "
                f"(file size: {result['file_size']} bytes)."
            )

        if name == "portfolio_journal_read":
            limit_raw = args.get("limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 10
            except (TypeError, ValueError):
                return "ERROR: limit must be an integer."
            try:
                result = await journal.read_recent(ctx_key, limit=limit)
            except Exception as e:
                logger.exception(f"portfolio_journal_read failed: {e}")
                return f"ERROR: journal read failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'read failed')}"
            entries = result.get("entries") or []
            total = result.get("total_entries", 0)
            if not entries:
                return (
                    "(journal is empty — no entries yet. This is your "
                    "blank notebook; start writing.)"
                )
            # Render entries as plain markdown so the LLM sees them in
            # the same shape it wrote them. Total count gives it
            # context about what it's NOT seeing.
            blocks = [
                f"## {e['ts']}\n\n{e['body']}" for e in entries
            ]
            header = (
                f"(showing {len(entries)} most recent of {total} total entries)"
            )
            return header + "\n\n" + "\n\n".join(blocks)

        return f"ERROR: unknown journal tool {name!r}"

    def _bot_label(self) -> str:
        if self.name_registry is not None:
            return self.name_registry.bot_name
        return "Sigil"

    async def _handle_predict_for_tool(
        self,
        args: dict,
        caller_ctx: CommandContext,
    ) -> str:
        """Log a prediction on behalf of a third-party chat member.

        Subject can be a registered name or a phone-tail (`...4810`).
        We resolve in that order:
          1. Tail format `...XXXX` → most recent sender in the group
             whose phone ends in those 4 digits.
          2. Registered name → SubjectResolver against NameRegistry.
          3. Reject anything that resolves to the bot, the room, or
             a free-text label — predictions need a real chat member.
        """
        store = self.prediction_store
        if store is None:
            return "(predict_for unavailable: no prediction store wired)"
        policy = caller_ctx.policy
        if policy is not None and not policy.allows_command("predict"):
            return "(predict_for unavailable: !predict not allowed in this chat)"

        subject_hint = str(args.get("subject") or "").strip()
        claim_text = str(args.get("claim") or "").strip()
        if not subject_hint:
            return "ERROR: predict_for requires a non-empty subject."
        if not claim_text:
            return "ERROR: predict_for requires a non-empty claim with a deadline."

        # 1) Tail-format lookup: covers users who aren't in the name
        # registry. The LLM sees them in group_context as `[...4810, ...]`
        # and can pass that through verbatim.
        user_hash: Optional[str] = None
        user_label: Optional[str] = None
        tail_match = re.match(r"^\s*\.\.\.\s*([A-Za-z0-9]{4})\s*$", subject_hint)
        if tail_match and self.group_log is not None and caller_ctx.group_id:
            tail = tail_match.group(1)
            try:
                phone = await self.group_log.find_recent_sender_by_tail(
                    caller_ctx.group_id, tail,
                )
            except Exception as e:
                logger.debug(f"predict_for: tail lookup failed: {e}")
                phone = None
            if phone:
                user_hash = hash_phone(phone)
                user_label = self._sender_label(phone)
            else:
                return (
                    f"ERROR: no recent sender in this group ends in {tail!r}. "
                    f"If they're registered by name, pass the name instead."
                )

        # 2) Name resolution via the existing SubjectResolver.
        if user_hash is None:
            resolver = self.subject_resolver
            if resolver is None:
                return "(predict_for unavailable: subject resolver not wired)"
            from ..memory import SUBJECT_CONTEXT, SUBJECT_SELF, is_user_hash
            key, label = resolver.resolve(
                subject_hint, sender_phone=caller_ctx.sender,
            )
            if not key:
                return f"(could not resolve subject {subject_hint!r})"
            if key == SUBJECT_SELF:
                return (
                    "ERROR: predict_for is for OTHER people. Use "
                    "predict_self for your own forecasts."
                )
            if key == SUBJECT_CONTEXT:
                return (
                    "ERROR: predict_for needs a real chat member. The "
                    "room itself can't have a leaderboard row."
                )
            if not is_user_hash(key):
                return (
                    f"ERROR: {subject_hint!r} resolved to a free-text "
                    f"subject ({label!r}). Predictions need a registered "
                    f"chat member or the user's tail (e.g. '...4810')."
                )
            user_hash = key
            user_label = label or subject_hint

        # 3) Parse the claim — same pipeline that backs !predict so
        # stock-shape claims auto-resolve at the deadline.
        parsed, err = await extract_prediction(claim_text, llm_client=self.llm)
        if err is not None:
            return f"ERROR: {err}"
        assert parsed is not None

        try:
            pred_id = await store.create(
                user_hash=user_hash,
                user_label=user_label or subject_hint,
                group_id=caller_ctx.group_id,
                context_key=caller_ctx.context_key(),
                claim=parsed["claim"],
                deadline_utc=parsed["deadline_utc"],
                ticker=parsed.get("ticker"),
                threshold=parsed.get("threshold"),
                direction=parsed.get("direction"),
            )
        except Exception as e:
            logger.exception(f"predict_for: store.create failed: {e}")
            return "ERROR: couldn't save the prediction."

        deadline_str = _format_deadline(parsed["deadline_utc"])
        kind_note = ""
        if parsed.get("ticker"):
            kind_note = (
                f" (auto-resolves: {parsed['ticker']} "
                f"{parsed['direction']} ${parsed['threshold']:g})"
            )
        return (
            f"Logged prediction #{pred_id} for {user_label}: "
            f"\"{parsed['claim']}\" due {deadline_str}{kind_note}."
        )

    async def _handle_memory_tool(
        self,
        name: str,
        args: dict,
        caller_ctx: CommandContext,
    ) -> str:
        """Dispatch remember/recall/forget. Returns the tool result text."""
        store = self.memory_store
        policy = caller_ctx.policy
        if store is None or policy is None or policy.id is None:
            return "(memory unavailable in this chat)"
        if policy.kind == "default":
            return "(memory unavailable: this chat has no explicit context row)"

        resolver = self.subject_resolver
        sender_phone = caller_ctx.sender

        sender_user_hash = hash_phone(sender_phone) if sender_phone else ""

        if name == "recall":
            subject_hint = (args.get("subject") or "").strip()
            query = (args.get("query") or "").strip()
            if subject_hint and resolver is not None:
                key, _ = resolver.resolve(
                    subject_hint, sender_phone=sender_phone
                )
                if not key:
                    return "(could not resolve subject)"
                rows = await store.list_for_subject(
                    context_id=policy.id,
                    subject_key=key,
                )
                if query:
                    ql = query.lower()
                    rows = [r for r in rows if ql in r["content"].lower()]
            elif query:
                rows = await store.search(
                    context_id=policy.id, query=query, limit=12,
                )
            else:
                rows = await store.list_for_context(
                    policy.id, limit=20,
                )
            return render_recall_results(rows, name_registry=self.name_registry)

        if name == "remember":
            if not getattr(policy, "memory_writes_enabled", True):
                return "(memory writes disabled for this chat)"
            subject_hint = (args.get("subject") or "").strip()
            kind = (args.get("kind") or "").strip().lower()
            content = (args.get("content") or "").strip()
            if not subject_hint or kind not in KINDS or not content:
                return (
                    "ERROR: remember requires non-empty subject, content, "
                    f"and kind in {sorted(KINDS)}."
                )
            if resolver is None:
                return "(subject resolver not configured)"
            key, label = resolver.resolve(
                subject_hint, sender_phone=sender_phone
            )
            if not key:
                return "(could not resolve subject)"
            # Third-party memories (about anyone other than the speaker or
            # the room) start at lower confidence — the bot was just told
            # what to store via prompt and can't independently verify it.
            # Corroboration from a second speaker promotes them to full.
            initial_conf = compute_explicit_confidence(
                subject_key=key, sender_user_hash=sender_user_hash,
            )
            mem_id = await store.add(
                context_id=policy.id,
                subject_key=key,
                subject_label=label,
                kind=kind,
                content=content,
                confidence=initial_conf,
                source=SOURCE_EXPLICIT,
                source_user_hash=sender_user_hash,
                source_message_at=time.time(),
            )
            if mem_id is None:
                return "(memory not saved — invalid input)"
            return f"saved memory #{mem_id} about {label or subject_hint}"

        if name == "forget":
            if not getattr(policy, "memory_writes_enabled", True):
                return "(memory writes disabled for this chat)"
            try:
                memory_id = int(args.get("memory_id") or 0)
            except (TypeError, ValueError):
                return "ERROR: forget requires an integer memory_id"
            if memory_id <= 0:
                return "ERROR: memory_id must be a positive integer"
            existing = await store.get(memory_id)
            if not existing or existing.get("context_id") != policy.id:
                return f"(no memory #{memory_id} in this chat)"
            ok = await store.delete(memory_id)
            return f"forgot memory #{memory_id}" if ok else "(forget failed)"

        return f"(unknown memory tool: {name})"

    async def _execute_tool_call(
        self,
        call: dict,
        messages: list[dict],
        caller_ctx: CommandContext,
        attachments: list,
        tools: Optional[list[dict]] = None,
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

        # Hallucinated-name guard: if the writer invents a tool not in the
        # schemas we just sent, don't let the call hit MCP — `who__who`
        # surfacing as "MCP server 'who' is not running" reads like real
        # infrastructure failure to the model and derails it into apology
        # mode. Return a corrective signal that nudges it back to plain
        # text. Skipped when `tools=None` (legacy callers that didn't
        # pass the schemas) so behavior stays compatible.
        if tools:
            valid_names = {
                (t.get("function") or {}).get("name")
                for t in tools
                if isinstance(t, dict)
            }
            valid_names.discard(None)
            if name and name not in valid_names:
                logger.info(
                    f"Rejecting hallucinated tool call {name!r}; "
                    f"valid={sorted(n for n in valid_names if n)[:10]}..."
                )
                content = (
                    f"No tool named '{name}' exists. This message is "
                    f"conversational — no tool is needed. Respond to the "
                    f"user directly in your own voice."
                )
                messages.append({
                    "role": "tool",
                    "tool_call_id": call_id,
                    "name": name,
                    "content": content,
                })
                return

        is_bot_tool = (
            self.bot_tools is not None
            and name.startswith(self.bot_tools.NAMESPACE + "__")
        )
        is_deep_think = name == "deep_think" and self.deep_think is not None
        is_memory = name in ("remember", "recall", "forget") and (
            self.memory_store is not None
        )
        is_predict_self = (
            name == "predict_self" and self.prediction_store is not None
        )
        is_predict_for = (
            name == "predict_for" and self.prediction_store is not None
        )
        is_predict_update = (
            name == "predict_update" and self.prediction_store is not None
        )
        is_portfolio_tool = (
            name in (
                "portfolio_buy",
                "portfolio_sell",
                "portfolio_status",
                "portfolio_place_order",
                "portfolio_cancel_order",
            )
            and self.portfolio_executor is not None
        )
        is_journal_tool = (
            name in ("portfolio_journal_append", "portfolio_journal_read")
            and self.portfolio_journal is not None
        )

        try:
            if is_predict_self:
                content = await self._handle_predict_self_tool(args, caller_ctx)
            elif is_predict_for:
                content = await self._handle_predict_for_tool(args, caller_ctx)
            elif is_predict_update:
                content = await self._handle_predict_update_tool(args, caller_ctx)
            elif is_portfolio_tool:
                content = await self._handle_portfolio_tool(
                    name, args, caller_ctx, attachments,
                )
            elif is_journal_tool:
                content = await self._handle_journal_tool(
                    name, args, caller_ctx,
                )
            elif is_memory:
                content = await self._handle_memory_tool(
                    name, args, caller_ctx
                )
            elif is_deep_think:
                # Policy gate is also checked when filtering schemas in
                # _collect_tools, but a writer that hallucinates the tool
                # call (or holds an in-flight schema across a policy
                # change) shouldn't bypass enforcement. Same logic applies
                # to the per-bot kill switch — if the admin turned it off
                # mid-conversation, defense-in-depth here rejects any
                # in-flight invocation rather than running it.
                policy = caller_ctx.policy
                bot = caller_ctx.bot
                if bot is not None and not getattr(bot, "deep_think_enabled", True):
                    content = "(deep_think unavailable: disabled for this bot)"
                elif policy is not None and not policy.allows_deep_think():
                    content = "(deep_think unavailable: not allowed in this chat)"
                else:
                    # Send the writer's status message to the chat right
                    # now so the user sees something happen before the
                    # 10-90s wait. Best-effort — failure to send doesn't
                    # block the deep_think call; a typing indicator is a
                    # decent fallback.
                    status_msg = str(args.get("status_message") or "").strip()
                    placeholder_handler = self._handler_for_ctx(caller_ctx)
                    if status_msg and placeholder_handler is not None:
                        try:
                            await placeholder_handler.send_message(
                                recipient=caller_ctx.sender,
                                message=status_msg,
                                group_id=caller_ctx.group_id,
                                styled=True,
                            )
                            logger.info(
                                f"DeepThink: sent placeholder to "
                                f"...{caller_ctx.sender[-4:]}: {status_msg!r}"
                            )
                        except Exception as e:
                            logger.warning(f"DeepThink placeholder send failed: {e}")

                    user_hash = hash_phone(caller_ctx.sender)
                    # Pass caller_ctx + attachments so the deep model gets
                    # the same tool kit (filtered by the same policy) and
                    # any attachments it produces (charts, etc.) bubble up
                    # to the writer's attachment list. Per-bot deep_think
                    # routing here too — Artaud's deep_think config is
                    # different from Sigil's even when called as a tool.
                    content = await self._deep_think_for(caller_ctx).think(
                        question=str(args.get("question") or ""),
                        context=str(args.get("context") or ""),
                        user_hash=user_hash,
                        group_id=caller_ctx.group_id,
                        caller_ctx=caller_ctx,
                        attachments=attachments,
                    )
            elif is_bot_tool:
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

        # Scrub identifying tokens before the result re-enters the
        # model's context window. Tools (especially the fetch MCP) can
        # echo back the bot's user-agent verbatim — "Sigil signal-stock-bot
        # ..." in the UA leaks the legacy persona into in-context
        # learning and the writer starts answering as Sigil. Hygiene
        # generally: don't pass raw tool errors verbatim.
        content = _scrub_tool_content(content)

        messages.append({
            "role": "tool",
            "tool_call_id": call_id,
            "name": name,
            "content": content,
        })

    def _handler_for_ctx(self, ctx):
        """Route outbound sends to the bot's phone when the pool is wired.

        Returns the SignalHandler that should send "as" this context's bot
        (for typing indicators and tool-loop status messages). Falls back
        to the legacy singleton when the pool isn't available — tests and
        single-bot installs continue to work unchanged.
        """
        if self.signal_pool is not None and ctx is not None:
            return self.signal_pool.for_bot(getattr(ctx, "bot", None))
        return self.signal_handler

    # How many of the most-recent user turns should keep their image
    # attachments visible to the writer. Tuned by feel: 5 covers normal
    # follow-up ("what's the price label on that chart?", "how about the
    # next one?") without ballooning the prompt with image tokens that
    # the user has long since moved past. Each image is paid for per
    # writer round, so this trades cost for coherence.
    VISION_HISTORY_USER_TURNS = 5

    def _inflate_image_history(self, prior: list[dict], ctx) -> None:
        """Re-attach persisted images to the last N user turns in-place.

        Walks `prior` (chronological order) and:
          1. Keeps `image_refs` on the last N user turns; drops them on
             older user turns so the writer doesn't carry stale pictures
             round after round.
          2. Converts kept image_refs into OpenAI multimodal `content`
             parts (`[{type: text, ...}, {type: image_url, ...}, ...]`).
          3. Removes the `image_refs` key from every turn so the dicts
             are clean OpenAI message shape when extended onto `messages`.

        No-op when the active bot has vision disabled — even if old
        rows have image_refs persisted (vision was on previously), we
        don't replay pictures to a text-only model.
        """
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        vision_on = bool(bot and getattr(bot, "vision_enabled", False))
        # Walk in reverse so we keep refs on the MOST RECENT N user turns,
        # not the FIRST N. Older user turns get their refs cleared
        # whether vision is on or off — the column stays populated in
        # SQLite until age-prune, but the LLM only ever sees the
        # freshest N.
        seen_user = 0
        for turn in reversed(prior):
            if turn.get("role") != "user":
                continue
            if not vision_on or seen_user >= self.VISION_HISTORY_USER_TURNS:
                turn.pop("image_refs", None)
            seen_user += 1

        if not vision_on:
            return

        for turn in prior:
            refs = turn.pop("image_refs", None)
            if not refs:
                continue
            text_content = turn.get("content") or ""
            parts: list[dict] = [{"type": "text", "text": text_content}]
            for img in refs:
                mime = (img.get("mime") or "image/jpeg") if isinstance(img, dict) else ""
                b64 = (img.get("data_b64") or "") if isinstance(img, dict) else ""
                if not b64:
                    continue
                parts.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:{mime};base64,{b64}"},
                })
            if len(parts) > 1:
                turn["content"] = parts

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                "Ask something: !ask what sectors are rotating this week?"
            )

        user_hash = hash_phone(ctx.sender)
        sender_tail = (ctx.sender or "")[-4:]
        context_key = ctx.context_key()
        is_group = ctx.group_id is not None

        # The reset/clear/forget subcommand is only valid when the user
        # explicitly invoked !ask. For implicit asks (reactor-triggered
        # spontaneous replies), the user's message could legitimately
        # start with any of these words ("Reset the alarms", "Clear the
        # table") — we must not interpret it as a history wipe.
        if (
            ctx.args[0].lower() in ("reset", "clear", "forget")
            and not getattr(ctx, "implicit_reason", None)
        ):
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
            now_ts = time.time()
            prior = await self.history.load(
                context_key,
                turns_per_user=self._live_turns(ctx),
                attribute_senders=is_group,
                now=now_ts,
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

            # Image-history replay: re-attach images on the last
            # `VISION_HISTORY_USER_TURNS` user messages so follow-ups
            # ("what was the gap on that chart?") still see the picture.
            # Older user turns drop their refs — once the conversation
            # has moved on a few rounds the images become dead weight
            # in the prompt. Gated on the active bot still being
            # vision-enabled; if vision was turned off, we don't
            # re-surface old images regardless of what's persisted.
            self._inflate_image_history(prior, ctx)

            group_ctx_block = await self._build_group_context(ctx, now_ts)

            # Long-term rolling summary, if one exists. Lives in the system
            # suffix so it bookends the persona; the XML tag tells the
            # model what it is (the inner text no longer needs a prose label).
            summary_block = ""
            try:
                summary = await self.history.get_summary(context_key)
                if summary and summary.get("summary"):
                    summary_text = await self._enrich(summary["summary"])
                    summary_block = summary_text
            except Exception as e:
                logger.debug(f"Failed to load summary for ask: {e}")

            # Staleness advisory: if the most recent stored turn is hours old,
            # warn the model so it doesn't assume the new message is a
            # continuation of an old thread. Skipped when the history is
            # empty — there's nothing to be stale about.
            staleness_block = ""
            try:
                last_ts = await self.history.latest_turn_timestamp(context_key)
            except Exception as e:
                logger.debug(f"Failed to read latest turn timestamp: {e}")
                last_ts = None
            if last_ts is not None and prior:
                age = now_ts - last_ts
                if age >= STALENESS_THRESHOLD_SECONDS:
                    staleness_block = (
                        f"The most recent prior turn was {format_relative_age(age)}. "
                        f"Treat the conversation history as background — the new "
                        f"message may be a fresh topic, unrelated to those older turns."
                    )

            # Multi-speaker group rules. The conversation history and the
            # group_context block both label every turn with a `[Name, time
            # ago]` bracket; without an explicit contract, the model has
            # been mis-attributing statements between speakers and even
            # confusing its own past replies with users'. Spelled out as
            # ground rules so the model treats brackets as the source of
            # truth for who said what.
            attribution_rules = ""
            if is_group:
                attribution_rules = (
                    "Attribution rules — this is a multi-speaker group chat:\n"
                    "- Every line in <group_context> and every user message "
                    "in the conversation history begins with `[Name, time ago]` "
                    "showing WHO spoke and WHEN. Different `[Name]` brackets "
                    "are different people. Never combine, conflate, or "
                    "transfer statements between speakers.\n"
                    "- Your own past replies (assistant turns) are tagged "
                    "`[to Name, time ago]` showing WHO you were answering. "
                    "Those are your words, not the addressee's. Untagged "
                    "assistant turns are still yours — they just predate this "
                    "labeling.\n"
                    "- BOTH bracket forms — `[Name, time ago]` on user "
                    "turns AND `[to Name, time ago]` on your assistant "
                    "turns — are INTERNAL METADATA the system adds at "
                    "replay time. They exist so YOU can tell speakers "
                    "apart in history; they are NEVER part of a visible "
                    "reply. Do NOT begin your output with `[J, just now]`, "
                    "`[David, 2m ago]`, `[to Sarah]`, or any other "
                    "bracket-prefix that mimics those formats. If you "
                    "want to address someone by name, use natural prose "
                    "(`David — yes, …`) — never the bracket form.\n"
                    "- The current message is wrapped in "
                    "`<current_message from=\"Name\">`. Reply to that speaker; "
                    "do not assume the prior turn's asker is still in front "
                    "of you.\n"
                    "- When asked \"what did X say\" or \"what did I say\", "
                    "trust the bracket labels — they are the source of truth "
                    "for attribution. If a label is missing or you can't "
                    "tell, say you don't know rather than guessing."
                )

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

            # Journal directive: gentle nudge to use the markdown
            # notebook at the moments where reflective writing actually
            # helps. Surfaced only when both the journal is wired AND
            # the chat allows portfolio (matching the gating in
            # _collect_tools — tools the model can't call shouldn't
            # show up in the prompt).
            journal_directive = ""
            if self.portfolio_journal is not None and (
                ctx.policy is None or ctx.policy.allows_command("portfolio")
            ):
                journal_directive = (
                    "Journal: you have a private markdown notebook for "
                    "this chat's portfolio (portfolio_journal_append / "
                    "portfolio_journal_read). It's YOUR space — chat "
                    "members never see it. Treat it like a real "
                    "trader's journal: paragraph-style narrative, "
                    "honest, written for future-you. Strong moments "
                    "to write:\n"
                    "  - After placing a trade: thesis, what would "
                    "    invalidate it, exit plan.\n"
                    "  - When the order watcher pings you about a "
                    "    fill: did the thesis work, what would you "
                    "    do differently.\n"
                    "  - End of trading day (cron close window): "
                    "    summarize patterns you noticed.\n"
                    "  - When you change your mind on a position or a "
                    "    broader market read.\n"
                    "Read recent entries (portfolio_journal_read) "
                    "BEFORE checking the portfolio or making a fresh "
                    "trade — that's how you stay coherent across "
                    "sessions instead of restarting your reasoning "
                    "every time. Don't journal every tick; quality "
                    "over volume. If you have nothing real to say, "
                    "skip."
                )

            # Deep think: heavyweight delegation tool. Surface only when
            # both the client is ready AND the per-context policy allows
            # it, so the writer doesn't get told about a tool it can't
            # actually call. The tool itself is pre-filtered out of
            # _collect_tools when these checks fail.
            deep_think_directive = ""
            # Whether the user's current message contains an explicit
            # ask-to-think-hard phrase. Initialized False so the trigger
            # block below remains safe even when the directive isn't built
            # (deep_think disabled / unconfigured).
            deep_think_user_trigger = False
            if (
                self.deep_think is not None
                and self.deep_think.status().get("ready")
                and (ctx.policy is None or ctx.policy.allows_deep_think())
            ):
                # Strong hint — appended after the directive — when the
                # user's CURRENT message contains an explicit ask-to-think-
                # hard phrase. The base directive lives in persona space;
                # this is situational and goes near the end of the system
                # suffix so it has recency-bias weight against any
                # contradicting framing the writer model might pattern-
                # match on (e.g. "easy question, just answer it").
                deep_think_user_trigger = _user_explicitly_asked_to_think(question)

                deep_think_directive = (
                    "Deep-think tool: a separate, slower, smarter model is "
                    "available via the deep_think(question, context, "
                    "status_message) tool. It HAS THE SAME TOOL KIT YOU "
                    "DO — every bot command and every MCP server you can "
                    "call, the deep model can call too. So when you "
                    "delegate, you're handing off a research task, not "
                    "just a thinking task: it will fetch its own data, "
                    "chain its own tool calls, and return a finished "
                    "answer.\n\n"
                    "WHEN TO CALL IT — be liberal, not stingy. Lean "
                    "toward calling deep_think for any of these:\n"
                    "  * The user explicitly asks you to think hard, "
                    "think carefully, take your time, dig deep, really "
                    "think about it, give it real thought, etc. THIS IS "
                    "AN UNAMBIGUOUS TRIGGER — honor it every time, even "
                    "if you feel you could answer directly. The user is "
                    "asking for the smart model.\n"
                    "  * Multi-step reasoning, careful comparisons, "
                    "synthesis across multiple sources.\n"
                    "  * Research that needs several chained tool calls "
                    "to reach a confident answer.\n"
                    "  * Open-ended judgment calls (recommendations, "
                    "rankings, predictions) where confidence matters.\n"
                    "  * Anywhere you'd otherwise hand-wave, hedge, or "
                    "guess. If your draft answer would start with "
                    "\"probably\" or \"I'd guess\", call deep_think.\n\n"
                    "WHEN NOT TO CALL IT: trivial lookups you can do "
                    "with one tool (a single price quote, a single news "
                    "fetch), tarot/iching draws (their own tools), or "
                    "pure banter.\n\n"
                    "Pass a precise question (not a topic) and any "
                    "context the deep model needs (the user's actual "
                    "framing, constraints, prior turns) — but you don't "
                    "need to pre-load tool results, the deep model can "
                    "fetch fresh data itself.\n\n"
                    "IMPORTANT — status_message: the tool is SLOW (10-90s "
                    "typical), so when you call it the bot will "
                    "immediately send your `status_message` to the chat "
                    "as a real message the user sees. Use this to tell "
                    "the user you're thinking — write it in YOUR OWN "
                    "voice for THIS chat (match the persona, language, "
                    "tone). Vary the phrasing every time, don't reuse a "
                    "stock line. Keep it under 100 chars and casual. "
                    "Examples (adapt to your voice): 'gimme a sec to dig "
                    "into this', 'hold on, this one needs real thought', "
                    "'lemme actually work this out — back in a minute'.\n\n"
                    "After the tool returns, weave its text into your "
                    "own final reply (don't paste verbatim, don't repeat "
                    "the status message). On '(unavailable: ...)' or "
                    "'(rate-limited: ...)' stubs, just answer the user "
                    "directly using what you know."
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

            # Python sandbox: when the Pyodide MCP server is running, the
            # writer LLM gets `pyodide__pyodide_execute(code, timeout)` and
            # `pyodide__pyodide_install-packages(package)` exposed
            # automatically. The directive tells Sigil what's pre-loaded,
            # what to use it for (real computation, not paraphrasing
            # data), and to bump the default 5s timeout — finance work
            # routinely overruns that.
            python_tool_directive = ""
            if self.mcp_manager is not None and any(
                t.server_name == "pyodide"
                for t in self.mcp_manager.all_tools()
            ):
                python_tool_directive = (
                    "Python sandbox (Pyodide): you have a real Python "
                    "interpreter via `pyodide__pyodide_execute(code, "
                    "timeout)`. State persists across calls in the "
                    "same conversation — variables stick. "
                    "Pre-loaded: numpy, pandas, scipy, matplotlib, "
                    "scikit-learn (Pyodide bundle), plus yfinance and "
                    "statsmodels (pre-cached). For other PyPI packages "
                    "call `pyodide__pyodide_install-packages(package)` "
                    "first.\n\n"
                    "USE IT for actual computation: correlations, "
                    "regressions, NPV/IRR, Black-Scholes from formula, "
                    "volatility estimation, custom indicators, "
                    "portfolio metrics — anything where a number is "
                    "the answer and you'd otherwise guess.\n\n"
                    "Pass `timeout=30000` (30s) for non-trivial work — "
                    "the default is 5s which is often too tight for "
                    "yfinance fetches or matrix math. The hard ceiling "
                    "from the bot side is 30s per call.\n\n"
                    "Pattern: either fetch data via existing tools "
                    "(bot__price, bot__chart) and paste it into your "
                    "script as a Python literal, OR import yfinance "
                    "inside the sandbox and fetch directly. After "
                    "computing, weave the result into your reply in "
                    "plain prose — DO NOT dump raw stdout / DataFrame "
                    "reprs / matplotlib output to the user."
                )

            # Implicit / spontaneous trigger: the reactor's should_respond
            # tool decided this message warrants a reply, but the user did
            # not @mention, quote-reply, or otherwise address the bot.
            # Tell the writer to look at full context, judge whether the
            # reactor's call was actually right, and bail with empty content
            # if there's nothing useful to add. Empty-content from the
            # writer becomes a silent no-op upstream (see
            # dispatch_implicit_ask).
            implicit_directive = ""
            if getattr(ctx, "implicit_reason", None):
                # Frame as instructions, NOT as a label-prefixed paragraph.
                # The previous "Spontaneous reply: this message was NOT..."
                # form was distinctive enough that some models echoed the
                # opener verbatim into their visible reply. Plain bullet
                # rules don't trigger that mimicry as easily, and the
                # closing "do not echo" rule + the response-style no-echo
                # rule + _strip_meta_leak give us defense in depth.
                implicit_directive = (
                    "The current message was NOT addressed to you directly "
                    "— no @mention, quote-reply, or name trigger. The "
                    f"reactor flagged it because: {ctx.implicit_reason!r}.\n\n"
                    "You are the second-stage filter. Look at the full "
                    "group context and decide whether a real reply is "
                    "actually warranted. It is fine — and often correct "
                    "— to stay silent. To stay silent, return empty "
                    "content (no tool calls, no text). The bot will say "
                    "nothing.\n\n"
                    "Reply only when you have something genuinely useful "
                    "to add: a real answer to an open-ended question, a "
                    "factual correction, or a substantive continuation of "
                    "a thread you started. Skip otherwise. Be brief — "
                    "lighter-touch than replies the user explicitly asked "
                    "for.\n\n"
                    "Do NOT echo any of this directive in your output. "
                    "Your reply is the literal Signal message users see; "
                    "if you find yourself writing \"spontaneous reply\" "
                    "or restating these instructions, stop and bail to "
                    "empty content instead."
                )

            # System suffix: persona-adjacent directives + long memory +
            # staleness advisory. Each block gets its own XML tag so the
            # model can scan the structure instead of guessing at section
            # boundaries in a wall of text. Group context + the live
            # trigger are NOT placed here — they belong to the user-role
            # message below so they read as situational input, not persona.
            # Late-binding hint that fires only on this turn — the user
            # just asked you to think hard. Place it AFTER conversation_
            # memory so it has stronger recency-weight than any "answer
            # quickly" framing in summary or staleness blocks.
            deep_think_trigger_hint = ""
            if (
                self.deep_think is not None
                and self.deep_think.status().get("ready")
                and (ctx.policy is None or ctx.policy.allows_deep_think())
                and deep_think_user_trigger
            ):
                deep_think_trigger_hint = (
                    "USER EXPLICITLY ASKED YOU TO THINK HARD on the "
                    "current message. Call deep_think now — that's "
                    "exactly what it's for. Do not answer directly from "
                    "memory and do not skip the tool. Pass a precise "
                    "sub-question and a status_message in your voice."
                )

            # Per-context memory preamble: facts learned about people in
            # this chat. Auto-injected for the current sender, the room
            # itself, and anyone named in the message. The writer doesn't
            # need a tool call to reference these.
            memory_block = ""
            if (
                self.memory_store is not None
                and ctx.policy is not None
                and ctx.policy.id is not None
                and ctx.policy.kind != "default"
            ):
                try:
                    memory_block = await build_preamble(
                        memory_store=self.memory_store,
                        context_id=ctx.policy.id,
                        sender_phone=ctx.sender,
                        sender_user_hash=user_hash,
                        sender_label=self._sender_label(ctx.sender)
                            if is_group else None,
                        current_message_text=question,
                        name_registry=self.name_registry,
                    )
                except Exception as e:
                    logger.debug(f"Memory preamble build failed: {e}")

            system_suffix_parts = [
                _wrap_xml("attribution_rules", attribution_rules),
                _wrap_xml("identity_note", names_directive),
                _wrap_xml("reactor_reflex", reactor_directive),
                _wrap_xml("recent_reactions", reactor_log_block),
                _wrap_xml("tarot_tool", tarot_directive),
                _wrap_xml("iching_tool", iching_directive),
                _wrap_xml("portfolio_journal", journal_directive),
                _wrap_xml("deep_think_tool", deep_think_directive),
                _wrap_xml("python_tool", python_tool_directive),
                _wrap_xml("spontaneous_reply", implicit_directive),
                _wrap_xml("conversation_memory", summary_block),
                _wrap_xml("context_memories", memory_block),
                _wrap_xml("conversation_status", staleness_block),
                _wrap_xml("deep_think_trigger", deep_think_trigger_hint),
            ]
            system_suffix_parts = [p for p in system_suffix_parts if p]
            system_suffix = "\n\n".join(system_suffix_parts) or None

            prompt_override = None
            if ctx.policy is not None and ctx.policy.system_prompt:
                prompt_override = ctx.policy.system_prompt
            # Per-bot system prompt: route through the bot-scoped writer
            # so a bot's writer-role override (e.g. Artaud's persona
            # baked into bot_llm_settings) replaces the global default.
            system_prompt = self._llm_for(ctx)._resolve_system_prompt(
                prompt_override, system_suffix
            )

            # Trigger user-message: optional group context, optional reply
            # target, then the live message wrapped in <current_message>
            # with an explicit "respond to this" instruction. The wrapper
            # disambiguates the trigger from history + group context, which
            # otherwise just look like more user turns to the model.
            trigger_parts: list[str] = []

            if group_ctx_block:
                trigger_parts.append(group_ctx_block)

            if ctx.quote_text:
                quoted_label = (
                    self._sender_label(ctx.quote_author)
                    if ctx.quote_author
                    else "earlier"
                )
                quoted = (await self._enrich(ctx.quote_text)).replace("\n", " ").strip()
                if len(quoted) > 400:
                    quoted = quoted[:399].rstrip() + "…"
                trigger_parts.append(
                    f'<replying_to from="{quoted_label}">\n{quoted}\n</replying_to>'
                )

            sender_label = self._sender_label(ctx.sender) if is_group else "user"
            # For implicit / spontaneous asks the closing instruction softens
            # — the user didn't ask the bot anything, so "Respond to this"
            # would override the bail-freely guidance in the system suffix.
            closing = (
                "Decide whether to respond. Empty output = stay silent."
                if getattr(ctx, "implicit_reason", None)
                else "Respond to this message."
            )
            trigger_parts.append(
                f'<current_message from="{sender_label}" sent="just now">\n'
                f"{question}\n"
                f"</current_message>\n"
                f"{closing}"
            )
            current_user_content = "\n\n".join(trigger_parts)

            # Vision: when the resolved bot is vision-enabled and the
            # inbound message carried image attachments, swap the user
            # message from a plain string to the OpenAI multimodal array
            # form so the writer model receives the bytes. One-shot: we
            # never persist the images into conversation history below,
            # so follow-up turns won't re-see them (cheaper and usually
            # the right semantics for "describe this picture" flows).
            inbound_images = getattr(ctx, "inbound_images", None) or []
            vision_active = (
                bool(inbound_images)
                and ctx.bot is not None
                and getattr(ctx.bot, "vision_enabled", False)
            )
            if vision_active:
                parts: list[dict] = [
                    {"type": "text", "text": current_user_content}
                ]
                for img in inbound_images:
                    mime = img.get("mime") or "image/jpeg"
                    b64 = img.get("data_b64") or ""
                    if not b64:
                        continue
                    parts.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    })
                logger.info(
                    f"Vision: attaching {len(parts) - 1} image(s) to "
                    f"writer round for bot={ctx.bot.slug}"
                )
                user_message_content = parts
            else:
                user_message_content = current_user_content

            messages: list[dict] = [{"role": "system", "content": system_prompt}]
            messages.extend(prior)
            messages.append({"role": "user", "content": user_message_content})

            tools = self._collect_tools(policy=ctx.policy, bot=ctx.bot)
            attachments: list = []

            # Research mode: when the active bot is configured for it
            # (Artaud), deep_think runs the entire tool loop first and
            # returns notes; the writer LLM then composes the final
            # reply from those notes without tool access. This pairs a
            # locally-trained writer (which carries the persona) with a
            # cloud research model (which has the toolset). When the
            # bot has no per-bot LLMClient yet (PR2 single-bot path),
            # we still go through this branch — the same llm_client
            # composes; only the tool-vs-no-tool split changes.
            use_research_mode = (
                ctx.bot is not None
                and getattr(ctx.bot, "deep_think_mode", "replace") == "research"
                and getattr(ctx.bot, "deep_think_enabled", True)
                and self.deep_think is not None
                and self.deep_think.status().get("ready")
                and (ctx.policy is None or ctx.policy.allows_deep_think())
            )
            # Vision short-circuit: when the writer has images on this
            # turn, bypass research mode and let the writer run its own
            # tool loop. deep_think wouldn't see the pixels (its model is
            # text-context only here), and its notes would actively
            # mislead the writer — typical failure mode is the notes
            # apologize "I can't see the image" and the writer parrots
            # that even though it has the bytes in its multimodal payload.
            # Image turns get writer-direct routing; subsequent text-only
            # turns continue using research mode normally.
            if use_research_mode and vision_active:
                logger.info(
                    "Vision active — bypassing research mode for this turn "
                    f"(bot={ctx.bot.slug})"
                )
                use_research_mode = False

            typing_handler = self._handler_for_ctx(ctx)
            if use_research_mode and typing_handler is not None:
                async with typing_handler.typing_indicator(
                    ctx.sender, ctx.group_id
                ):
                    answer = await self._run_research_handoff(
                        ctx=ctx,
                        question=question,
                        research_input=current_user_content,
                        messages=messages,
                        attachments=attachments,
                        user_hash=user_hash,
                    )
            elif use_research_mode:
                answer = await self._run_research_handoff(
                    ctx=ctx,
                    question=question,
                    research_input=current_user_content,
                    messages=messages,
                    attachments=attachments,
                    user_hash=user_hash,
                )
            elif typing_handler is not None:
                # Show a typing indicator in the chat while the tool loop runs.
                # The indicator auto-clears in ~15s, so the helper refreshes it.
                async with typing_handler.typing_indicator(
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

        # Strip any directive/meta text the writer copied from the system
        # prompt into its visible reply (`[to David]`, "Spontaneous reply:",
        # "Reflex note:", etc.). The system prompt tells it not to, but
        # this is the safety net so users never see internal scaffolding
        # as literal output — and it prevents leaks from round-tripping
        # through history.
        answer = _strip_meta_leak(answer)

        try:
            # Store the raw question (no prefix) + sender_tail separately; the
            # load path re-adds the prefix when replaying in a group context.
            # Pass the same turns-per-user value the load path uses so a
            # context-override of 30 doesn't get clipped back to the global
            # 6 on every write.
            live_turns = self._live_turns(ctx)
            # Persist inbound images on the user turn so follow-ups can
            # re-see them (see _inflate_image_history). Only when vision
            # was actually active this round — if the bot doesn't have
            # vision the bytes aren't on inbound_images anyway, but we
            # also gate explicitly so a misconfigured handler can't
            # accidentally bloat history with bytes no one will read.
            persisted_images = inbound_images if vision_active else None
            await self.history.append(
                context_key, "user", question,
                user_hash=user_hash, sender_tail=sender_tail,
                turns_per_user=live_turns,
                image_refs=persisted_images,
            )
            # Persist the addressee on the assistant row too: the load path
            # uses it to render `[to Name, time ago]` in group playback so
            # the model can pair its prior answers with the right asker.
            await self.history.append(
                context_key, "assistant", answer,
                user_hash=user_hash, sender_tail=sender_tail,
                turns_per_user=live_turns,
            )
        except Exception as e:
            logger.error(f"Failed to persist ask history: {e}")

        # Mirror the bot's own reply into the group_log so subsequent !ask
        # calls see it in `<group_context>` alongside human messages —
        # otherwise the bot's voice is invisible in the time-ordered group
        # view and only shows up in the user/assistant alternation. Gated
        # the same way as inbound logging in dispatcher.py: only when the
        # group_context_messages feature is on.
        if (
            is_group
            and self.group_log is not None
            and self._live_group_ctx() > 0
        ):
            try:
                await self.group_log.append_bot(ctx.group_id, answer)
            except Exception as e:
                logger.error(f"Failed to append bot reply to group log: {e}")

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
