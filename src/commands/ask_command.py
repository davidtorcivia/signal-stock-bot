"""
!ask — pass a question to the configured LLM, with optional tool use via MCP.

Flow per request:
  1. Build messages (system prompt + optional group-chat suffix + history + user).
  2. If policy permits MCP, pass the fixed discovery/invocation broker.
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
    PORTFOLIO_BUY_OPTION_TOOL,
    PORTFOLIO_BUY_TOOL,
    PORTFOLIO_CANCEL_ORDER_TOOL,
    PORTFOLIO_JOURNAL_APPEND_TOOL,
    PORTFOLIO_JOURNAL_READ_TOOL,
    PORTFOLIO_OPTION_QUOTE_TOOL,
    PORTFOLIO_OPTIONS_CHAIN_TOOL,
    PORTFOLIO_PLACE_OPTION_ORDER_TOOL,
    PORTFOLIO_PLACE_ORDER_TOOL,
    PORTFOLIO_SELL_OPTION_TOOL,
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
from ..options_symbols import (
    friendly_name as _opt_friendly_name,
    normalize_contract as _opt_normalize_contract,
)
from ..predictions import PredictionStore
from ..llm import (
    LLMClient,
    LLMDisabled,
    LLMError,
    LLMNotConfigured,
    ConversationHistory,
    format_history_timestamp,
)
from ..llm.client import DEFAULT_SYSTEM_PROMPT
from ..llm.mcp_broker import (
    MCP_BROKER_TOOLS,
    MCP_DISCOVER_NAME,
    MCP_INVOKE_NAME,
    broker_should_be_exposed,
    discover_mcp_tools,
    invoke_mcp_tool,
)
from ..llm.prompt_cache import PromptCachePlan
from ..llm.summarizer import render_summary_for_prompt
from ..llm.prompt_compiler import (
    PromptCompiler,
    StablePromptBlock,
    VolatilePromptBlock,
)
from ..llm.tool_runtime import (
    DurableToolLedger,
    ToolCallLedger,
    ToolExecution,
    ToolLoopRuntime,
    ensure_tool_result_envelope,
    tool_result_ok,
)
from ..llm.output_safety import (
    strip_meta_leak as _strip_meta_leak,
    strip_tool_call_leak as _strip_tool_call_leak,
)
from ..signal.message_text import normalize_signal_text
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


# Requests that normally need more room than the writer's terse conversational
# budget. Tool results also select the extended budget at runtime, even when
# the original wording was short ("price?", "fetch this").
_EXTENDED_RESPONSE_RE = re.compile(
    r"\b("
    r"summary|summaries|summari[sz](?:e|ed|ing|ation)?"
    r"|analy[sz](?:e|ed|ing|is)"
    r"|compar(?:e|ed|ing|ison)"
    r"|explain|explanation"
    r"|break\s+down|step[\s-]+by[\s-]+step"
    r"|detail(?:ed|s)?|thorough(?:ly)?|comprehensive"
    r"|long(?:er|form)?|essay|report|pros\s+and\s+cons"
    r")\b",
    re.IGNORECASE,
)


def _user_explicitly_asked_to_think(text: str) -> bool:
    return bool(_DEEP_THINK_TRIGGER_RE.search(text or ""))


def _wants_extended_response(text: str) -> bool:
    return bool(_EXTENDED_RESPONSE_RE.search(text or ""))


def _adaptive_response_overrides(
    writer, question: str, messages: list[dict],
) -> Optional[dict]:
    """Select the extended writer budget for long-form or evidence rounds.

    ``max_tokens`` remains the cheap conversational default. The separately
    configurable extended ceiling is used when the user asks for depth or a
    tool has returned evidence that needs synthesis. Test doubles and legacy
    clients without ``_config`` simply retain their normal behavior.
    """
    has_tool_results = any(
        message.get("role") == "tool" for message in messages
    )
    if not has_tool_results and not _wants_extended_response(question):
        return None
    try:
        cfg = writer._config()
    except Exception:
        return None
    if not isinstance(cfg, dict):
        return None
    try:
        normal = max(1, int(cfg.get("max_tokens", 1000)))
        extended = max(
            normal, int(cfg.get("extended_max_tokens", normal)),
        )
    except (TypeError, ValueError):
        return None
    return {"max_tokens": extended} if extended > normal else None


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
        bot_registry=None,
    ):
        self.llm = llm
        # Per-bot writer/deep_think router. When set (always in the real
        # app; None only in older tests that hand-roll AskCommand),
        # `_llm_for(ctx)` and `_deep_think_for(ctx)` resolve to the
        # bot-scoped clients so per-bot model/temperature/extra_body
        # actually take effect. Falls back to the constructor-supplied
        # `llm`/`deep_think` when ctx.bot is None.
        self.llm_factory = llm_factory
        # Bot registry — used to resolve other bots' display names when
        # rendering multi-bot group context, and to enumerate enabled
        # bots' aliases for cross-bot addressing. Optional: when None
        # (legacy tests) the renderer falls back to the active bot's
        # display name for all bot turns.
        self.bot_registry = bot_registry
        self.history = history
        self._durable_tool_ledger = DurableToolLedger(
            getattr(history, "db_path", "data/watchlist.db")
        )
        self.group_log = group_log
        self.mcp_manager = mcp_manager
        self.bot_tools = bot_tools
        # Optional MemoryStore — when set, the writer LLM gets remember/
        # recall/forget tools (gated per-context via memory_writes_enabled)
        # and stored memories about active speakers in the chat are
        # auto-injected into the volatile user tail.
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
        # into a cited structured summary for future user-tail context.
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

    @staticmethod
    def _canonical_turn_text(text: object) -> str:
        """Comparison form used only for legacy history/group deduplication."""
        value = normalize_signal_text(text).strip()
        # History stores the question without the explicit command token;
        # group_log stores the original inbound line. Signal timestamps are
        # authoritative for new rows, but this makes pre-migration rows match.
        value = re.sub(r"^!\S+\s+", "", value)
        return " ".join(value.split()).casefold()

    @staticmethod
    def _history_turn_candidates(prior: list[dict]) -> list[dict]:
        return [
            {
                "turn_id": turn.get("_turn_id"),
                "role": turn.get("role"),
                "raw_content": turn.get("_raw_content") or "",
                "created_at": turn.get("_created_at"),
                "source_message_ts": turn.get("_source_message_ts"),
                "sender_tail": turn.get("_sender_tail"),
                "sender": None,
                "bot_id": None,
            }
            for turn in prior
            if turn.get("_turn_id")
        ]

    @staticmethod
    def _render_turn_graph(
        prior: list[dict],
        retrieved_turns: list[dict],
        visible_turns: list[dict],
    ) -> str:
        """Render only useful, resolvable non-linear reply edges.

        A normal alternating history already says h2 follows h1, so emitting
        every h2->h1 edge duplicates ordering and makes the graph look more
        important than it is. Keep only branches whose parent differs from
        the preceding history row, and only when both endpoints are visible.
        Retrieved rows have no reliable adjacent ordering, so their visible
        parent edges are retained.
        """
        visible_ids = {
            str(turn.get("turn_id"))
            for turn in visible_turns
            if turn.get("turn_id")
        }
        visible_ids.update(
            str(turn.get("turn_id"))
            for turn in retrieved_turns
            if turn.get("turn_id")
        )

        edges: list[str] = []
        previous_id: Optional[str] = None
        for turn in prior:
            turn_id = turn.get("_turn_id")
            parent = turn.get("_parent_turn_ref")
            if (
                turn_id
                and parent
                and str(parent) in visible_ids
                and str(parent) != previous_id
            ):
                edges.append(f"{turn_id} parent={parent}")
            if turn_id:
                previous_id = str(turn_id)

        for turn in retrieved_turns:
            turn_id = turn.get("turn_id")
            parent = turn.get("parent_turn_ref")
            if turn_id and parent and str(parent) in visible_ids:
                edges.append(f"{turn_id} parent={parent}")
        return "\n".join(dict.fromkeys(edges))

    async def _build_group_context(
        self,
        ctx,
        prior: Optional[list[dict]] = None,
    ) -> tuple[str, list[dict]]:
        """Render recent group chat, oldest-first, wrapped for the model.

        Each line gets `[turn g12; Sender, 2026-06-06 18:30 UTC] text` so the
        model can tell which bits of context are seconds-old vs. hours-old
        and exact quotes can point back to one row. The stamp
        is absolute (derived from each message's own `created_at`, not from
        "now") so the block stays byte-stable across requests and lands in
        the cached prompt prefix — see `format_history_timestamp`. Returns
        the block plus its visible turn registry, or two empty values if
        there's nothing to show.
        """
        limit = self._live_group_ctx()
        if limit <= 0 or not ctx.is_group or self.group_log is None:
            return "", []
        floor_at = (
            getattr(ctx.policy, "purge_floor_at", None)
            if ctx.policy is not None else None
        )
        msgs = await self.group_log.recent(
            ctx.group_id, limit=limit, exclude_last=1,
            floor_at=floor_at,
        )
        if not msgs:
            return "", []

        # Ambient context is a delta, not a rolling replay. Once this bot has
        # persisted a newer conversation turn, older room chatter has already
        # served its purpose and repeating it on every request both wastes
        # tokens and re-anchors the model on a stale aside. This timestamp
        # cursor is per-bot because `prior` is already bot-scoped.
        history_cutoff = max(
            (
                float(turn.get("_created_at"))
                for turn in (prior or [])
                if turn.get("_created_at") is not None
            ),
            default=None,
        )
        if history_cutoff is not None:
            msgs = [
                msg for msg in msgs
                if float(msg.get("created_at") or 0) > history_cutoff
            ]
        if not msgs:
            return "", []

        # A user post that triggered the bot exists in both stores: history
        # preserves the role alternation, while group_log preserves the room
        # timeline. Never send both copies. New rows match on Signal's stable
        # source timestamp; old rows fall back to sender-tail + normalized
        # content, consuming matches one-for-one so repeated messages survive.
        history_signal_ids = {
            str(t.get("_source_message_ts"))
            for t in (prior or [])
            if t.get("role") == "user" and t.get("_source_message_ts") is not None
        }
        legacy_history_counts: dict[tuple[str, str], int] = {}
        for turn in prior or []:
            if turn.get("role") != "user" or turn.get("_source_message_ts") is not None:
                continue
            sig = (
                str(turn.get("_sender_tail") or ""),
                self._canonical_turn_text(turn.get("_raw_content")),
            )
            legacy_history_counts[sig] = legacy_history_counts.get(sig, 0) + 1

        deduped: list[dict] = []
        for msg in msgs:
            if msg.get("sender") == BOT_SENDER:
                deduped.append(msg)
                continue
            message_ts = msg.get("message_ts")
            if message_ts is not None and str(message_ts) in history_signal_ids:
                continue
            # Synthetic/implicit asks may lack a source timestamp even while
            # their dispatcher-written group row has one. Use the legacy
            # signature whenever exact-id matching did not resolve the row.
            sig = (
                str(msg.get("sender") or "")[-4:],
                self._canonical_turn_text(msg.get("text")),
            )
            remaining = legacy_history_counts.get(sig, 0)
            if remaining:
                legacy_history_counts[sig] = remaining - 1
                continue
            deduped.append(msg)
        msgs = deduped
        if not msgs:
            return "", []

        # Enrich messages concurrently. _enrich short-circuits on empty
        # input so empty cells cost ~nothing; gathering avoids 30 sequential
        # awaits stacking up on the LLM hot path.
        raw_texts = [
            normalize_signal_text(m.get("text") or "").strip()
            for m in msgs
        ]
        enriched = await asyncio.gather(*(self._enrich(t) for t in raw_texts))

        # Multi-bot attribution: each BOT_SENDER row carries the writing
        # bot's id (or NULL on legacy rows). The renderer looks up that
        # bot's display_name and prefixes it with `Bot:` — co-bot rows
        # read `[Bot: Sigil, ...]`. Hiding the bot-ness (the original
        # design) made the writer treat sibling bots as humans: it
        # attributed their lines to people, claimed authorship of their
        # replies, and argued with users about who said what (the
        # 2026-06-06 Bot Town incident). The marker plus the matching
        # attribution rule lets the model keep humans, itself, and
        # sibling bots in three distinct buckets.
        active_bot_name = self._active_bot_display(ctx)
        active_bot_id = getattr(getattr(ctx, "bot", None), "id", None)

        def _bot_row_label(row_bot_id) -> str:
            if row_bot_id is None:
                # Legacy bot row (pre-multi-bot column). Attribute to
                # the active bot for back-compat — those rows pre-date
                # any co-bot in this context.
                return active_bot_name
            if self.bot_registry is not None:
                try:
                    bot = self.bot_registry.get_sync(row_bot_id)
                except Exception:
                    bot = None
                if bot is not None and getattr(bot, "display_name", None):
                    return bot.display_name
            return active_bot_name

        lines: list[str] = []
        candidates: list[dict] = []
        for m, text in zip(msgs, enriched):
            if not text:
                continue
            if m["sender"] == BOT_SENDER:
                # Skip the ACTIVE bot's own prior replies. They're already
                # replayed in the user/assistant alternation as this bot's
                # assistant turns (history.load is bot_id-scoped to include
                # them), so rendering them here too made the writer see its
                # last message twice — and the group_context copy sits right
                # before the new question, priming it to repeat itself. The
                # absolute UTC stamp on every line + assistant turn lets the
                # model reconstruct the interleaved timeline without this
                # copy. Legacy NULL-bot rows are treated as the active bot's
                # own (matching history.load, which replays NULL assistant
                # turns as this bot's), so they're skipped too. CO-bots'
                # rows (a different bot_id) stay — they're filtered out of
                # THIS bot's history, so group_context is their only home.
                row_bot_id = m.get("bot_id")
                if row_bot_id == active_bot_id or row_bot_id is None:
                    continue
                label = f"Bot: {_bot_row_label(row_bot_id)}"
                role = "assistant"
            else:
                label = self._sender_label(m["sender"])
                role = "user"
            ts = m.get("created_at")
            stamp = format_history_timestamp(ts) if ts is not None else None
            bracket = f"{label}, {stamp}" if stamp else label
            turn_id = f"g{m['id']}"
            lines.append(
                f"[turn {turn_id}; {bracket}] {text.replace(chr(10), ' ')}"
            )
            candidates.append({
                "turn_id": turn_id,
                "role": role,
                "raw_content": m.get("text") or "",
                "created_at": ts,
                "source_message_ts": m.get("message_ts"),
                "sender_tail": (
                    str(m.get("sender") or "")[-4:]
                    if m.get("sender") != BOT_SENDER else None
                ),
                "sender": m.get("sender"),
                "bot_id": m.get("bot_id"),
            })
        if not lines:
            return "", []
        return _wrap_xml("group_context", "\n".join(lines)), candidates

    def _candidate_matches_quote_author(self, ctx, candidate: dict) -> bool:
        author = getattr(ctx, "quote_author", None)
        if not author:
            return True
        if candidate.get("sender") and candidate.get("sender") != BOT_SENDER:
            return candidate["sender"] == author
        if candidate.get("role") == "user":
            return str(author)[-4:] == str(candidate.get("sender_tail") or "")

        active_bot = getattr(ctx, "bot", None)
        if candidate.get("turn_id", "").startswith("h"):
            return (
                active_bot is not None
                and (getattr(active_bot, "signal_phone", None) or "") == author
            )
        if self.bot_registry is not None and candidate.get("bot_id") is not None:
            try:
                bot = self.bot_registry.get_sync(candidate["bot_id"])
            except Exception:
                bot = None
            return bool(bot and (getattr(bot, "signal_phone", None) or "") == author)
        return True

    def _find_reply_turn(self, ctx, candidates: list[dict]) -> Optional[str]:
        """Resolve a Signal quote to one visible history/group turn id."""
        if not getattr(ctx, "quote_text", None):
            return None
        quote_ts = getattr(ctx, "quote_timestamp", None)
        if quote_ts is not None:
            for candidate in reversed(candidates):
                source_ts = candidate.get("source_message_ts")
                if source_ts is not None and str(source_ts) == str(quote_ts):
                    return candidate.get("turn_id")

        wanted = self._canonical_turn_text(ctx.quote_text)
        if not wanted:
            return None
        for candidate in reversed(candidates):
            if not self._candidate_matches_quote_author(ctx, candidate):
                continue
            actual = self._canonical_turn_text(candidate.get("raw_content"))
            if actual == wanted:
                return candidate.get("turn_id")
        # Some Signal quote payloads truncate long source text.  A bounded
        # containment fallback remains safer than dropping the reference.
        if len(wanted) >= 8:
            for candidate in reversed(candidates):
                if not self._candidate_matches_quote_author(ctx, candidate):
                    continue
                actual = self._canonical_turn_text(candidate.get("raw_content"))
                if wanted in actual or actual in wanted:
                    return candidate.get("turn_id")
        return None

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

    def _build_identity_block(
        self,
        ctx,
        authoritative_prompt: Optional[str] = None,
    ) -> str:
        """Build a routing/identity reminder without fighting the persona.

        The resolved writer system prompt is authoritative for identity.
        Bot ``display_name`` and aliases are also routing handles used by
        Signal, but they are not necessarily the character's name (a legacy
        ``sigil`` bot row may, for example, host an Artaud-trained writer).
        Treating a routing handle as a name produced the contradictory pair
        ``You are Artaud`` / ``Your name is Sigil`` in the same system
        message.

        With a generic prompt there is no persona identity, so a direct name
        reminder remains useful. When the resolved prompt actually declares a
        character, the block describes display names as chat routing handles
        instead and leaves identity to that prompt. If the prompt already
        contains the display name, the shorter legacy wording is safe.

        Other bots in the room are deliberately NOT mentioned — they
        appear in <group_context> as labeled participant utterances,
        and the writer should treat them like any other speaker.
        """
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        if bot is None or not getattr(bot, "display_name", None):
            return ""
        aliases = getattr(bot, "aliases", None) or []
        # Display name first, then any non-empty aliases distinct from it.
        names = [bot.display_name]
        for a in aliases:
            a = (a or "").strip()
            if a and a.lower() != bot.display_name.lower() and a not in names:
                names.append(a)
        also = (
            f" (you also answer to: {', '.join(names[1:])})"
            if len(names) > 1 else ""
        )
        # Check the ACTUAL resolved base prompt, not only bots.persona.
        # Per-bot bot_llm_settings.system_prompt and per-context overrides
        # take precedence over that field and were the source of the
        # production Artaud/Sigil contradiction.
        resolved = (authoritative_prompt or "").strip()
        persona = (getattr(bot, "persona", None) or "").strip()
        identity_text = resolved or persona
        prompt_names_bot = bool(identity_text) and re.search(
            rf"\b{re.escape(bot.display_name)}\b",
            identity_text,
            re.IGNORECASE,
        )
        if prompt_names_bot:
            return (
                f"You also answer to: {', '.join(names)}. "
                f"When someone in chat addresses a different name, "
                f"that's a different participant — do not respond as "
                f"them or assume their identity. Speak only as yourself."
            )

        # A prompt/persona may intentionally name a character that differs
        # from the registry display name. Detect explicit identity framing,
        # rather than assuming every custom instruction is a persona: a
        # harmless override such as "Keep answers concise" should still get
        # the configured display name.
        declares_identity = bool(persona)
        if resolved and resolved != DEFAULT_SYSTEM_PROMPT:
            declares_identity = declares_identity or bool(re.search(
                r"(?im)^\s*(?:you are|your name is|you answer as|"
                r"you(?:'re| are) called|i am)\s+"
                r"(?!an?\b|the\s+(?:assistant|bot)\b)[A-Z][\w'.-]*\b",
                resolved,
            ))
        if declares_identity:
            return (
                f"Chat routing handles for this bot: {', '.join(names)}. "
                f"Messages addressed to those handles are routed to you, "
                f"but your identity and voice come from the primary system "
                f"prompt; do not adopt a different identity from a routing "
                f"handle. When someone addresses any other name, that's a "
                f"different participant — do not respond as them or assume "
                f"their identity. Speak only as yourself."
            )
        return (
            f"Your name is {bot.display_name}{also}. "
            f"When someone in chat addresses a different name, that's a "
            f"different participant — do not respond as them or assume "
            f"their identity. Speak only as yourself."
        )

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

    def _reply_target_label(self, ctx) -> str:
        """Label for the author of the message being quote-replied to.

        Active bot's own phone → "you" — reads naturally from the
        writer's own perspective ("you're being replied to") and keeps
        attribution unambiguous when the user quotes the bot.
        Another known bot's phone → that bot's display_name, matching
        what <group_context> uses for that bot's rows so the writer
        sees the same name in both places.
        Anyone else → the regular sender label.
        """
        phone = ctx.quote_author if ctx is not None else None
        if not phone:
            return "earlier message"
        active_bot = getattr(ctx, "bot", None) if ctx is not None else None
        if (
            active_bot is not None
            and (getattr(active_bot, "signal_phone", None) or "") == phone
        ):
            return "you"
        if self.bot_registry is not None:
            try:
                for bot in self.bot_registry.list_sync():
                    if (getattr(bot, "signal_phone", None) or "") == phone:
                        name = getattr(bot, "display_name", None)
                        if name:
                            return name
            except Exception:
                pass
        return self._sender_label(phone)

    def _collect_tools(self, policy=None, bot=None) -> Optional[list[dict]]:
        schemas: list[dict] = []
        if self.bot_tools is not None:
            schemas.extend(self.bot_tools.list_tools(policy=policy))
        # MCP uses a fixed two-schema broker. Detailed schemas are returned
        # only by mcp__discover after the model identifies a need, avoiding a
        # permanent multi-thousand-token catalog on every chat turn.
        if broker_should_be_exposed(self.mcp_manager, policy):
            schemas.extend(MCP_BROKER_TOOLS)
        # deep_think is exposed only when the client is wired AND the global
        # flag is on AND the per-context policy permits AND the per-bot
        # `deep_think_enabled` is True. The client returns "(unavailable)"
        # for disabled/unconfigured calls, but suppressing the schema
        # entirely keeps the writer from wasting tool-call rounds on a
        # guaranteed-stub when we already know it's off. The per-bot flag
        # is a master kill — admins flip it off when they want deep_think
        # completely out of one bot's repertoire.
        # Resolve readiness from the ACTIVE bot's deep-think client. The
        # constructor field belongs to the default bot; using it here made
        # non-default bots advertise or hide tools according to another
        # bot's configuration even though dispatch later used the factory.
        deep_think = self.deep_think
        factory = getattr(self, "llm_factory", None)
        if (
            factory is not None
            and bot is not None
            and getattr(bot, "id", None) is not None
        ):
            deep_think = factory.get_deep_think(bot.id)
        if deep_think is not None:
            dt_status = deep_think.status()
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
            # Options tools — long-only buy/sell + chain/quote lookup.
            # Same gating as the equity tools; the executor handles
            # OCC normalization so the LLM can pass friendly strings.
            schemas.append(PORTFOLIO_OPTIONS_CHAIN_TOOL)
            schemas.append(PORTFOLIO_OPTION_QUOTE_TOOL)
            schemas.append(PORTFOLIO_BUY_OPTION_TOOL)
            schemas.append(PORTFOLIO_SELL_OPTION_TOOL)
            schemas.append(PORTFOLIO_PLACE_OPTION_ORDER_TOOL)
            # Journal tools are gated separately because the journal
            # store is wired independently — a deploy could have
            # portfolio enabled but no journal directory configured,
            # in which case we hide the tools rather than expose
            # broken endpoints.
            if self.portfolio_journal is not None:
                schemas.append(PORTFOLIO_JOURNAL_APPEND_TOOL)
                schemas.append(PORTFOLIO_JOURNAL_READ_TOOL)
        # MCP sessions can be restarted in any order. Canonical ordering keeps
        # an unchanged effective tool set byte-identical for prefix caching.
        schemas.sort(
            key=lambda schema: str(
                (schema.get("function") or {}).get("name") or ""
            )
        )
        return schemas or None

    async def _run_tool_loop(
        self,
        messages: list[dict],
        tools: Optional[list[dict]],
        caller_ctx: CommandContext,
        attachments: list,
        cache_plan: Optional[PromptCachePlan] = None,
        question: str = "",
    ) -> str:
        """Drive the assistant/tool back-and-forth until we have final text.

        Attachments produced by bot-tool calls are appended to `attachments`
        so the caller can include them in the final Signal response.

        If a round/call/progress budget is hit, the runtime makes one final
        no-tools synthesis constrained to completed evidence. If that fails,
        it returns an honest incomplete-work message.
        """
        # Resolve the writer client per-bot. _llm_for handles the
        # ctx.bot=None fallback so legacy callers (no factory wired)
        # still hit the singleton self.llm. Each call re-resolves so a
        # late-changed ctx.bot would still pick up the right client,
        # but in practice ctx.bot is fixed for the lifetime of one
        # command.
        llm = self._llm_for(caller_ctx)
        max_rounds = self._live_max_tool_rounds()

        async def chat_round(
            working_messages: list[dict], active_tools: Optional[list[dict]],
        ) -> dict:
            logger.info(
                "LLM tool runtime: requesting completion (%d msgs)",
                len(working_messages),
            )
            return await llm.chat_messages(
                working_messages, tools=active_tools, cache_plan=cache_plan,
                overrides=_adaptive_response_overrides(
                    llm, question, working_messages,
                ),
            )

        async def execute_call(
            call: dict, ledger: ToolCallLedger,
        ) -> ToolExecution:
            local_messages: list[dict] = []
            local_attachments: list = []
            await self._execute_tool_call(
                call, local_messages, caller_ctx, local_attachments,
                tools=tools, ledger=ledger,
            )
            name = (call.get("function") or {}).get("name") or ""
            message = local_messages[-1] if local_messages else {
                "role": "tool", "tool_call_id": call.get("id") or "",
                "name": name,
                "content": ensure_tool_result_envelope(
                    name, "ERROR: tool returned no result", ok=False,
                ),
            }
            return ToolExecution(
                message=message,
                attachments=local_attachments,
                ok=tool_result_ok(message.get("content", "")),
            )

        source_key = None
        if caller_ctx.message_timestamp is not None:
            source_key = (
                f"{caller_ctx.context_key()}:{caller_ctx.bot_id or 0}:"
                f"{caller_ctx.message_timestamp}"
            )
        outcome = await ToolLoopRuntime(max_rounds=max_rounds).run(
            messages=messages,
            tools=tools,
            chat=chat_round,
            execute=execute_call,
            attachments=attachments,
            source_key=source_key,
            durable_ledger=self._durable_tool_ledger,
            incomplete_fallback=(
                f"Task did not complete within the {max_rounds}-round tool "
                "budget. The completed results were insufficient for a safe answer."
            ),
        )
        logger.info(
            "LLM tool runtime stopped reason=%s rounds=%d calls=%d parallel=%d",
            outcome.stop_reason, outcome.rounds, outcome.calls,
            outcome.parallel_batches,
        )
        return outcome.content

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
        cache_plan: Optional[PromptCachePlan] = None,
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
        # Defensive copy: we mutate the tail user message below when adding
        # research notes.
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

        # Inject notes into the volatile user tail. Research output changes on
        # every request; putting it in the system message invalidated the whole
        # provider prefix immediately before the final writer call.
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

        rendered_handoff = _wrap_xml("research_handoff", handoff_block)
        if messages and messages[-1].get("role") == "user":
            messages[-1] = dict(messages[-1])
            existing = messages[-1].get("content")
            if isinstance(existing, list):
                messages[-1]["content"] = list(existing) + [
                    {"type": "text", "text": rendered_handoff}
                ]
            else:
                current = str(existing or "").rstrip()
                messages[-1]["content"] = (
                    f"{current}\n\n{rendered_handoff}"
                    if current else rendered_handoff
                )
        else:
            messages.append({"role": "user", "content": rendered_handoff})
        if cache_plan is not None:
            cache_plan.with_volatile("research_handoff", rendered_handoff)

        # Single-shot: no tools. The writer's only job is to compose.
        assistant_msg = await writer.chat_messages(
            messages, tools=None, cache_plan=cache_plan,
            overrides=_adaptive_response_overrides(
                writer, question, messages,
            ),
        )
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

        # Multi-bot: each bot gets its own leaderboard identity. Without
        # this, Sigil and Artaud both write predictions with
        # user_hash=hash_phone(BOT_SENDER) and label=name_registry's
        # single bot_name — collapsing onto one leaderboard row. Per-bot
        # sentinel derives a distinct hash, and we prefer the bot's
        # display_name as the label so the leaderboard reads correctly.
        active_bot = getattr(caller_ctx, "bot", None)
        if active_bot is not None:
            bot_label = (
                getattr(active_bot, "display_name", None)
                or (self.name_registry.bot_name if self.name_registry else "Bot")
            )
            bot_user_hash = hash_phone(f"{BOT_SENDER}:{active_bot.id}")
        else:
            bot_label = (
                self.name_registry.bot_name
                if self.name_registry is not None else "Bot"
            )
            bot_user_hash = hash_phone(BOT_SENDER)
        try:
            pred_id = await store.create(
                user_hash=bot_user_hash,
                user_label=bot_label,
                group_id=caller_ctx.group_id,
                context_key=caller_ctx.context_key(),
                claim=parsed["claim"],
                deadline_utc=parsed["deadline_utc"],
                ticker=parsed.get("ticker"),
                threshold=parsed.get("threshold"),
                direction=parsed.get("direction"),
                bot_id=caller_ctx.bot_id,
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

        # Multi-bot scoping: each bot in a multi-bot group has its own
        # portfolio so they can compete. `ctx.portfolio_key()` wraps
        # the chat context with the responding bot's id.
        ctx_key = caller_ctx.portfolio_key()
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
                snap = await executor.status(
                    ctx_key,
                    label_hint=(
                        caller_ctx.bot.display_name
                        if getattr(caller_ctx, "bot", None) is not None
                           and getattr(caller_ctx.bot, "display_name", None)
                        else None
                    ),
                )
            except Exception as e:
                logger.exception(f"portfolio_status failed: {e}")
                return f"ERROR: couldn't load portfolio status: {type(e).__name__}"

            # Attach the portfolio dashboard image so the user sees the
            # numbers in a clean PNG. Failures are non-fatal — the LLM
            # still gets the text data and can reply without the image.
            try:
                image_b64 = render_portfolio_image(
                    snap, bot_name=self._bot_label(caller_ctx),
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

            return render_status(snap, bot_name=self._bot_label(caller_ctx)) + image_note

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

        # ---- Options tools ----

        if name == "portfolio_options_chain":
            underlying = str(args.get("underlying") or "").strip().upper()
            if not underlying:
                return "ERROR: portfolio_options_chain requires an underlying ticker."
            expiration_raw = args.get("expiration")
            expiration = (
                str(expiration_raw).strip() if expiration_raw else None
            )
            limit_raw = args.get("limit")
            try:
                limit = int(limit_raw) if limit_raw is not None else 50
            except (TypeError, ValueError):
                return "ERROR: limit must be an integer."
            limit = max(1, min(250, limit))
            try:
                result = await executor.options_chain(
                    underlying, expiration=expiration, limit=limit,
                )
            except Exception as e:
                logger.exception(f"portfolio_options_chain failed: {e}")
                return f"ERROR: chain lookup failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'chain unavailable')}"
            rows = result.get("rows") or []
            if not rows:
                exp_part = f" expiring {expiration}" if expiration else ""
                return f"No contracts found for {underlying}{exp_part}."
            # Render compact list for the LLM. Sort by (expiration,
            # type, strike) so the model sees a coherent layout.
            rows.sort(key=lambda r: (
                r.get("expiration") or "", r.get("option_type") or "",
                r.get("strike") or 0.0,
            ))
            lines = [
                f"Options chain for {underlying} "
                f"({len(rows)} contract{'s' if len(rows) != 1 else ''}):"
            ]
            for r in rows[:limit]:
                exp = r.get("expiration") or "?"
                strike = r.get("strike") or 0.0
                otype = (r.get("option_type") or "?").upper()[:1]
                prem = r.get("premium") or 0.0
                vol = r.get("volume") or 0
                oi = r.get("open_interest") or 0
                iv = r.get("iv")
                iv_part = f" IV={iv:.2f}" if isinstance(iv, (int, float)) else ""
                delta = r.get("delta")
                delta_part = (
                    f" Δ={delta:.2f}" if isinstance(delta, (int, float)) else ""
                )
                lines.append(
                    f"  {r.get('contract', '?'):<25} {otype} ${strike:<7.2f} "
                    f"{exp}  prem=${prem:.2f}  vol={vol} oi={oi}"
                    f"{iv_part}{delta_part}"
                )
            return "\n".join(lines)

        if name == "portfolio_option_quote":
            contract_raw = str(args.get("contract") or "").strip()
            if not contract_raw:
                return "ERROR: portfolio_option_quote requires a contract."
            try:
                occ = _opt_normalize_contract(contract_raw)
            except ValueError as e:
                return f"ERROR: {e}"
            try:
                quote = await executor.providers.get_option_quote(occ)
            except Exception as e:
                logger.warning(f"portfolio_option_quote({occ}): {e}")
                return f"ERROR: quote failed: {type(e).__name__}: {e}"
            if quote is None:
                return f"ERROR: no quote for {occ}"
            greeks = getattr(quote, "greeks", None) or {}
            iv = getattr(quote, "implied_volatility", None)
            iv_part = f"  IV={iv:.2f}" if isinstance(iv, (int, float)) else ""
            delta_part = ""
            if isinstance(greeks.get("delta"), (int, float)):
                delta_part = f"  Δ={greeks['delta']:.3f}"
            lines = [
                f"⊡ {_opt_friendly_name(occ)}  (OCC: {occ})",
                f"  premium: ${getattr(quote, 'price', 0.0) or 0.0:.2f}/sh  "
                f"vol={getattr(quote, 'volume', 0)}  "
                f"oi={getattr(quote, 'open_interest', 0)}{iv_part}{delta_part}",
                f"  estimated cost: $"
                f"{(getattr(quote, 'price', 0.0) or 0.0) * 100:.2f} per contract",
            ]
            return "\n".join(lines)

        if name == "portfolio_buy_option":
            contract_raw = str(args.get("contract") or "").strip()
            reason = str(args.get("reason") or "").strip()
            qty_raw = args.get("qty")
            if not contract_raw:
                return "ERROR: portfolio_buy_option requires a contract."
            if not reason:
                return "ERROR: portfolio_buy_option requires a one-sentence reason."
            try:
                qty_int = int(qty_raw) if qty_raw is not None else 0
            except (TypeError, ValueError):
                return "ERROR: qty must be a positive whole number of contracts."
            if qty_int <= 0:
                return "ERROR: qty must be a positive whole number of contracts."
            try:
                result = await executor.execute_buy_option(
                    ctx_key,
                    contract=contract_raw,
                    qty=qty_int,
                    reason=reason,
                    source=source_tag,
                )
            except Exception as e:
                logger.exception(f"portfolio_buy_option failed: {e}")
                return f"ERROR: option buy failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'buy rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            return (
                f"Bought {qty_int} {result['friendly']} "
                f"@ ${result['premium']:.2f}/sh (cost "
                f"${result['proceeds']:,.2f}, multiplier "
                f"{result.get('multiplier', 100)}). "
                f"Position avg ${result['avg_premium_after']:.2f}/sh. "
                f"Cash now ${result['cash_after']:,.2f}. "
                f"Reason: {reason}"
            )

        if name == "portfolio_place_option_order":
            contract_raw = str(args.get("contract") or "").strip()
            side = str(args.get("side") or "").strip().lower()
            kind = str(args.get("kind") or "").strip().lower()
            reason = str(args.get("reason") or "").strip()
            if not contract_raw:
                return "ERROR: portfolio_place_option_order requires a contract."
            if not reason:
                return "ERROR: portfolio_place_option_order requires a one-sentence reason."
            trigger_raw = args.get("trigger_premium")
            if trigger_raw is None:
                return "ERROR: trigger_premium is required."
            try:
                trigger_premium = float(trigger_raw)
            except (TypeError, ValueError):
                return "ERROR: trigger_premium must be a number."
            qty = args.get("qty")
            close_position = bool(args.get("close_position") or False)
            try:
                qty_f = float(qty) if qty is not None else None
            except (TypeError, ValueError):
                return "ERROR: qty must be numeric."
            expires_in_days_raw = args.get("expires_in_days")
            try:
                expires_in_days = (
                    float(expires_in_days_raw)
                    if expires_in_days_raw is not None else 30.0
                )
            except (TypeError, ValueError):
                return "ERROR: expires_in_days must be a number."
            expires_in_days = max(1.0, min(90.0, expires_in_days))
            try:
                result = await executor.place_order(
                    ctx_key,
                    contract=contract_raw, side=side, kind=kind,
                    trigger_price=trigger_premium,
                    qty=qty_f, close_position=close_position,
                    reason=reason,
                    expires_in_days=expires_in_days,
                )
            except Exception as e:
                logger.exception(f"portfolio_place_option_order failed: {e}")
                return f"ERROR: place_option_order failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'order rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            warn_part = ""
            if result.get("warning"):
                warn_part = f" WARNING: {result['warning']}"
            current_part = ""
            if result.get("current_price") is not None:
                current_part = (
                    f" (current premium ${result['current_price']:.2f}/sh)"
                )
            try:
                friendly = _opt_friendly_name(result.get("contract") or contract_raw)
            except Exception:
                friendly = result.get("contract") or contract_raw
            return (
                f"Placed option order #{result['order_id']}: "
                f"{kind}-{side} {friendly} @ premium "
                f"${result['trigger_price']:.2f}/sh{current_part}. "
                f"Will fire automatically when the contract's premium "
                f"crosses the trigger (~5 min poll during market hours)."
                f"{warn_part} Reason: {reason}"
            )

        if name == "portfolio_sell_option":
            contract_raw = str(args.get("contract") or "").strip()
            reason = str(args.get("reason") or "").strip()
            qty_arg = args.get("qty")
            if not contract_raw:
                return "ERROR: portfolio_sell_option requires a contract."
            if not reason:
                return "ERROR: portfolio_sell_option requires a one-sentence reason."
            try:
                result = await executor.execute_sell_option(
                    ctx_key,
                    contract=contract_raw,
                    qty=qty_arg,
                    reason=reason,
                    source=source_tag,
                )
            except Exception as e:
                logger.exception(f"portfolio_sell_option failed: {e}")
                return f"ERROR: option sell failed: {type(e).__name__}"
            if not result.get("ok"):
                return f"ERROR: {result.get('error', 'sell rejected')}"
            caller_ctx.portfolio_mutation_count += 1
            pnl = result.get("realized_pnl") or 0.0
            pnl_part = (
                f" Realized P/L ${pnl:+,.2f}." if abs(pnl) > 0.005 else ""
            )
            remaining = result.get("qty_after") or 0.0
            remaining_part = (
                f" Remaining: {remaining:g} contracts."
                if remaining > 0 else " Position closed."
            )
            return (
                f"Sold {result['friendly']} @ ${result['premium']:.2f}/sh "
                f"for ${result['proceeds']:,.2f}.{pnl_part}{remaining_part} "
                f"Cash now ${result['cash_after']:,.2f}. Reason: {reason}"
            )

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

        # Same per-bot scoping as the portfolio tools above — each bot
        # in a multi-bot group has its own journal file so they don't
        # cross-contaminate reflections.
        ctx_key = caller_ctx.portfolio_key()

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

    def _bot_label(self, ctx=None) -> str:
        """Display name to render for the bot in user-visible output
        (portfolio caption, image header, etc). Prefer the ctx.bot's
        display_name in a multi-bot group; fall back to the registry's
        seed name for single-bot installs."""
        bot = getattr(ctx, "bot", None) if ctx is not None else None
        if bot is not None and getattr(bot, "display_name", None):
            return bot.display_name
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
            # Multi-bot scoping: each bot recalls only its own memories
            # (plus legacy NULL-bot rows). Other bots' impressions of
            # the same chat/people stay private to them.
            recall_bot_id = (
                caller_ctx.bot_id
            )
            if subject_hint and resolver is not None:
                key, _ = resolver.resolve(
                    subject_hint, sender_phone=sender_phone
                )
                if not key:
                    return "(could not resolve subject)"
                rows = await store.list_for_subject(
                    context_id=policy.id,
                    subject_key=key,
                    bot_id=recall_bot_id,
                )
                if query:
                    ql = query.lower()
                    rows = [r for r in rows if ql in r["content"].lower()]
            elif query:
                rows = await store.search(
                    context_id=policy.id, query=query, limit=12,
                    bot_id=recall_bot_id,
                )
            else:
                # Per-bot scope so the wildcard browse doesn't cross
                # into another bot's mental model.
                rows = await store.list_for_context(
                    policy.id, limit=20,
                    bot_id=recall_bot_id,
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
                bot_id=(
                    caller_ctx.bot_id
                ),
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
        ledger: Optional[ToolCallLedger] = None,
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
                    "content": ensure_tool_result_envelope(name, content, ok=False),
                })
                return

        if ledger is not None:
            duplicate_reason, duplicate_content = ledger.lookup(
                call_id=call_id, name=name, arguments=args,
            )
            if duplicate_reason is not None:
                logger.warning(
                    "Suppressing duplicate LLM tool call name=%s reason=%s",
                    name, duplicate_reason,
                )
                if duplicate_reason == "duplicate_call_id_conflict":
                    content = ensure_tool_result_envelope(
                        name,
                        duplicate_content or "ERROR: duplicate tool call suppressed",
                        ok=False,
                        reused="duplicate tool call suppressed: call-id conflict",
                    )
                else:
                    content = ensure_tool_result_envelope(
                        name, duplicate_content or "(no result)", ok=True,
                        reused=f"duplicate tool call suppressed: {duplicate_reason}",
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
        deep_think_client = self._deep_think_for(caller_ctx)
        is_deep_think = name == "deep_think" and deep_think_client is not None
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
                "portfolio_place_option_order",
                "portfolio_cancel_order",
                "portfolio_options_chain",
                "portfolio_option_quote",
                "portfolio_buy_option",
                "portfolio_sell_option",
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
                    content = await deep_think_client.think(
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
                if result is not None and not result.success:
                    content = f"ERROR: {content}"
                if result and result.attachments:
                    attachments.extend(result.attachments)
            elif name == MCP_DISCOVER_NAME:
                content = discover_mcp_tools(
                    self.mcp_manager,
                    caller_ctx.policy,
                    query=str(args.get("query") or ""),
                    server=str(args.get("server") or ""),
                    limit=args.get("limit", 5),
                )
            elif name == MCP_INVOKE_NAME:
                content = await invoke_mcp_tool(
                    self.mcp_manager,
                    caller_ctx.policy,
                    name=str(args.get("name") or ""),
                    arguments=args.get("arguments"),
                )
            elif self.mcp_manager is not None:
                # Direct MCP schemas are no longer exposed. Refuse a stale or
                # hallucinated direct call instead of bypassing the broker's
                # policy validation.
                content = (
                    f"ERROR: direct MCP call {name!r} is not available. "
                    f"Use {MCP_DISCOVER_NAME}, then {MCP_INVOKE_NAME}."
                )
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
        content = ensure_tool_result_envelope(
            name, _scrub_tool_content(content),
        )

        if ledger is not None:
            ledger.record(
                call_id=call_id,
                name=name,
                arguments=args,
                content=str(content),
            )

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
        question = normalize_signal_text(question)
        if not question:
            return CommandResult.error("Ask something: !ask <question>")

        # Inline-expand any tweet/URL links the user pasted so the LLM
        # gets the substance, not just an opaque URL.
        question = await self._enrich(question)

        try:
            now_ts = time.time()
            floor_at = (
                getattr(ctx.policy, "purge_floor_at", None)
                if ctx.policy is not None else None
            )
            # Multi-bot scoping: pass ctx.bot.id so the load filter
            # returns user turns shared with co-bots + this bot's own
            # assistant turns. Other bots' replies surface separately
            # via group_log → <group_context>, attributed by writer.
            history_bot_id = (
                ctx.bot_id
            )
            live_turns = self._live_turns(ctx)
            prior = await self.history.load(
                context_key,
                turns_per_user=live_turns,
                attribute_senders=is_group,
                now=now_ts,
                floor_at=floor_at,
                bot_id=history_bot_id,
                include_turn_ids=True,
                include_internal=True,
            )
            # Re-enrich each prior turn — bot/user messages stored before
            # the enricher existed (or stored with a failed enrichment)
            # would otherwise reach the LLM as bare URLs it can't open.
            # Gathered: with 30-turn history this matters on the LLM hot path.
            text_msgs = [
                m for m in prior if isinstance(m.get("content"), str)
            ]
            if text_msgs:
                normalized = [
                    normalize_signal_text(m["content"]) for m in text_msgs
                ]
                enriched = await asyncio.gather(
                    *(self._enrich(text) for text in normalized)
                )
                for m, new_content in zip(text_msgs, enriched):
                    m["content"] = new_content
                    m["_raw_content"] = normalize_signal_text(
                        m.get("_raw_content") or ""
                    )

            # Image-history replay: re-attach images on the last
            # `VISION_HISTORY_USER_TURNS` user messages so follow-ups
            # ("what was the gap on that chart?") still see the picture.
            # Older user turns drop their refs — once the conversation
            # has moved on a few rounds the images become dead weight
            # in the prompt. Gated on the active bot still being
            # vision-enabled; if vision was turned off, we don't
            # re-surface old images regardless of what's persisted.
            self._inflate_image_history(prior, ctx)

            group_ctx_block, group_turns = await self._build_group_context(
                ctx, prior,
            )
            history_turns = self._history_turn_candidates(prior)
            visible_turns = history_turns + group_turns
            reply_turn_id = self._find_reply_turn(ctx, visible_turns)
            latest_turn_id = None
            dated_turns = [
                t for t in visible_turns if t.get("created_at") is not None
            ]
            if dated_turns:
                latest_turn_id = max(
                    dated_turns, key=lambda t: t["created_at"]
                ).get("turn_id")
            current_parent_ref = reply_turn_id or latest_turn_id

            # Hybrid selection: exact recent history remains in role order;
            # a few older lexical matches are added only to the volatile tail.
            # This recovers relevant older details without widening the cached
            # prefix or repeating rows already visible verbatim.
            retrieved_history_block = ""
            retrieved_turns: list[dict] = []
            if live_turns > 0:
                try:
                    retrieved_turns = await self.history.retrieve_relevant(
                        context_key,
                        question,
                        exclude_turn_ids={
                            str(turn.get("_turn_id")) for turn in prior
                            if turn.get("_turn_id")
                        },
                        limit=4,
                        floor_at=floor_at,
                        bot_id=history_bot_id,
                        attribute_senders=is_group,
                    )
                    if retrieved_turns:
                        retrieved_history_block = "\n".join(
                            f"[turn {turn['turn_id']}; {turn['speaker']}; "
                            f"{format_history_timestamp(turn['created_at'])}] "
                            f"{turn['content']}"
                            for turn in retrieved_turns
                        )
                except Exception as e:
                    logger.debug(f"Relevant-history retrieval failed: {e}")

            # History order already carries the ordinary chain. Surface only
            # visible branch edges, with an unambiguous parent= representation.
            turn_graph_block = self._render_turn_graph(
                prior, retrieved_turns, visible_turns,
            )
            # Snapshot the active bot's deep-think client once for every
            # prompt/execution gate in this request. The default client on
            # self.deep_think may have different readiness and settings.
            deep_think_client = self._deep_think_for(ctx)

            # Long-term rolling summary, if one exists. Lives in the system
            # suffix so it bookends the persona; the XML tag tells the
            # model what it is (the inner text no longer needs a prose label).
            summary_block = ""
            if live_turns > 0:
                try:
                    summary_bot_id = (
                        ctx.bot.id if getattr(ctx, "bot", None) is not None
                        else None
                    )
                    summary = await self.history.get_summary(
                        context_key,
                        floor_at=floor_at,
                        bot_id=summary_bot_id,
                    )
                    if summary and summary.get("summary"):
                        summary_text = render_summary_for_prompt(summary["summary"])
                        summary_text = await self._enrich(summary_text)
                        summary_block = summary_text
                except Exception as e:
                    logger.debug(f"Failed to load summary for ask: {e}")

            # Staleness advisory: if the most recent stored turn is hours old,
            # warn the model so it doesn't assume the new message is a
            # continuation of an old thread. Skipped when the history is
            # empty — there's nothing to be stale about.
            staleness_block = ""
            last_ts = None
            if live_turns > 0:
                try:
                    last_ts = await self.history.latest_turn_timestamp(
                        context_key, floor_at=floor_at
                    )
                except Exception as e:
                    logger.debug(f"Failed to read latest turn timestamp: {e}")
            if last_ts is not None and prior:
                age = now_ts - last_ts
                # Gate on `age` (presence flips once, when the thread crosses
                # the threshold) but render the stamp absolutely so the block
                # text doesn't drift every request — the model reads recency
                # off the current-time anchor in the last user message.
                if age >= STALENESS_THRESHOLD_SECONDS:
                    staleness_block = (
                        f"The most recent prior turn was at "
                        f"{format_history_timestamp(last_ts)}. "
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
                    "- Every line in <group_context> and every conversation "
                    "history turn begins with `[turn ID; Name, "
                    "YYYY-MM-DD HH:MM UTC]` showing WHO spoke and WHEN (the "
                    "timestamp is UTC; compare it against the current time "
                    "given in your latest message to judge how long ago it "
                    "was). Different `[Name]` brackets are different people. "
                    "Never combine, conflate, or transfer statements between "
                    "speakers.\n"
                    "- Your own past replies (assistant turns) are tagged "
                    "`[turn ID; to Name, YYYY-MM-DD HH:MM UTC]` showing WHO you were "
                    "answering. Those are your words, not the addressee's. "
                    "Untagged assistant turns are still yours — they just "
                    "predate this labeling.\n"
                    "- BOTH bracket forms — `[turn ID; Name, ...UTC]` on user "
                    "turns AND `[turn ID; to Name, ...UTC]` on your assistant "
                    "turns — are INTERNAL METADATA the system adds at "
                    "replay time. They exist so YOU can tell speakers "
                    "apart in history; they are NEVER part of a visible "
                    "reply. Do NOT begin your output with `[turn h12; David, "
                    "2026-06-06 18:30 UTC]`, `[turn h13; to Sarah]`, or any other "
                    "bracket-prefix that mimics those formats. If you "
                    "want to address someone by name, use natural prose "
                    "(`David — yes, …`) — never the bracket form.\n"
                    "- Lines in <group_context> labeled `[turn ID; Bot: Name, "
                    "...UTC]` are messages from OTHER automated bots in "
                    "this chat. They are not humans and they are not you. "
                    "Never claim their statements as your own, never "
                    "attribute them to a human, and never answer on their "
                    "behalf.\n"
                    "- `<current_message ... follows=\"ID\">` points to the "
                    "most recent visible turn. For unquoted follow-ups such "
                    "as 'what does that mean?', resolve 'that' against the "
                    "follows turn before reaching back to older topics. "
                    "`<replying_to turn=\"ID\">`, when present, is an explicit "
                    "Signal quote and is the authoritative target. The "
                    "`parent` attribute and <turn_graph> record durable reply "
                    "edges. A graph row `child parent=target` means child "
                    "replied to target; follow it when topics interleave.\n"
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
            # process with only a short explicit event log. Without this,
            # users tease "why did you react?" and the model indignantly
            # denies having thumbs, then that denial gets persisted into
            # history and poisons every future turn.
            reactor_directive = ""
            try:
                if self.reactor is not None and hasattr(
                    self.reactor, "is_enabled"
                ):
                    reactor_on = self.reactor.is_enabled(
                        ctx.bot, ctx.policy,
                    )
                else:
                    # Backward-compatible path for older test doubles and
                    # deployments constructed without EmojiReactor.
                    reactor_on = bool(
                        self.llm.store.get("reactor_enabled", False)
                    ) and (
                        ctx.policy is None
                        or getattr(ctx.policy, "reactor_enabled", True)
                    )
            except Exception:
                reactor_on = False
            if is_group and reactor_on:
                reactor_directive = (
                    "Reflex note: in group chats you also emoji-react to "
                    "messages. It runs as a separate fast reflex out of "
                    "band from this conversation, but the reactions are "
                    "yours. When a <recent_reactions> block is present in "
                    "the current user turn, it is your short explicit log "
                    "of which emoji you placed and on what; use it when "
                    "answering questions about a specific reaction. The "
                    "reflex is rate-"
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
                deep_think_client is not None
                and deep_think_client.status().get("ready")
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

            # Recent reactions: volatile in-memory state from the reactor.
            # It MUST live in the tail user turn, not the system suffix:
            # every reaction changes this text, and putting it in the static
            # prefix invalidates the provider's ~40K-token prompt cache.
            # Newest first, capped at 5.
            reactor_log_block = ""
            if (
                is_group
                and self.reactor is not None
                and reactor_on
            ):
                try:
                    recent_rxns = self.reactor.recent_reactions(
                        ctx.group_id, limit=5,
                        bot_id=(
                            ctx.bot.id
                            if getattr(ctx, "bot", None) is not None
                            else None
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Failed to fetch recent reactions: {e}")
                    recent_rxns = []
                if recent_rxns:
                    reaction_turns = {
                        str(turn.get("source_message_ts")): turn.get("turn_id")
                        for turn in [*visible_turns, *retrieved_turns]
                        if turn.get("source_message_ts") is not None
                        and turn.get("turn_id")
                    }
                    lines = ["Recent reactions you placed (newest first):"]
                    for r in recent_rxns[:5]:
                        if not isinstance(r, dict):
                            continue
                        # The target is user-controlled historical text.
                        # Flatten/cap it so a malformed reactor row cannot
                        # balloon or break the XML-shaped prompt block.
                        emoji = re.sub(
                            r"[\r\n<>\[\]\"]", "",
                            str(r.get("emoji") or ""),
                        )[:16]
                        target_turn = reaction_turns.get(
                            str(r.get("target_timestamp"))
                        )
                        if target_turn:
                            # The referenced turn already carries author and
                            # text, so an id is both clearer and much smaller.
                            lines.append(f"  {emoji} turn={target_turn}")
                        else:
                            # The target fell outside visible history. Retain a
                            # bounded snippet so questions about it remain
                            # answerable without widening history.
                            sender = re.sub(
                                r"[\r\n<>\[\]\"]", "",
                                str(r.get("sender") or "unknown"),
                            )[:80]
                            target = re.sub(
                                r"\s+", " ", str(r.get("target") or "")
                            ).replace("<", "‹").replace(">", "›").replace(
                                '"', "”"
                            )[:160]
                            lines.append(
                                f"  {emoji} on [{sender}] \"{target}\""
                            )
                    if len(lines) > 1:
                        reactor_log_block = "\n".join(lines)

            # Python sandbox: when the Pyodide MCP server is allowed, its
            # execution tools are available through the compact MCP broker.
            # The directive tells the writer what's pre-loaded,
            # what to use it for (real computation, not paraphrasing
            # data), and to bump the default 5s timeout — finance work
            # routinely overruns that.
            python_tool_directive = ""
            if (
                self.mcp_manager is not None
                and (
                    ctx.policy is None
                    or ctx.policy.allows_mcp("pyodide")
                )
            ):
                python_tool_directive = (
                    "Python sandbox (Pyodide): you have a real Python "
                    "interpreter. Discover its exact schema with "
                    "`mcp__discover(query=\"pyodide execute python\")`, "
                    "then run it through `mcp__invoke`. State persists across calls in the "
                    "same conversation — variables stick. "
                    "Pre-loaded: numpy, pandas, scipy, matplotlib, "
                    "scikit-learn (Pyodide bundle), plus yfinance and "
                    "statsmodels (pre-cached). For other PyPI packages "
                    "discover and invoke the Pyodide package installer first.\n\n"
                    "USE IT for actual computation: correlations, "
                    "regressions, NPV/IRR, Black-Scholes from formula, "
                    "volatility estimation, custom indicators, "
                    "portfolio metrics — anything where a number is "
                    "the answer and you'd otherwise guess.\n\n"
                    "Pass `timeout=30000` (30s) for non-trivial work — "
                    "the default is 5s which is often too tight for "
                    "yfinance fetches or matrix math. The hard ceiling "
                    "from the bot side is 30s per call.\n\n"
                    "For market data, import yfinance inside the sandbox "
                    "or discover an allowed data capability through the "
                    "MCP broker. After "
                    "computing, weave the result into your reply in "
                    "plain prose — DO NOT dump raw stdout / DataFrame "
                    "reprs / matplotlib output to the user."
                )

            mcp_broker_directive = ""
            if broker_should_be_exposed(self.mcp_manager, ctx.policy):
                mcp_broker_directive = (
                    "External MCP capabilities use a compact catalog. When "
                    "live or specialized data may help, call `mcp__discover` "
                    "with the capability you need. It returns a small set of "
                    "exact tool names and argument schemas. Then call "
                    "`mcp__invoke` with one returned name and matching "
                    "arguments. Do not invent or directly call hidden MCP "
                    "tool names."
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
                # The reason is reactor-LLM output derived from raw user
                # text — sanitize before it lands in the volatile prompt tail:
                # cap the length and drop angle brackets so it can't
                # open/close XML-ish blocks or smuggle directive-shaped
                # markup into the prompt. It's informational scaffolding;
                # lossy is fine.
                reason_safe = re.sub(
                    r"[<>]", "", str(ctx.implicit_reason)
                ).strip()[:200] or "(no reason given)"
                implicit_directive = (
                    "The current message was NOT addressed to you directly "
                    "— no @mention, quote-reply, or name trigger. The "
                    f"reactor flagged it because: {reason_safe!r}.\n\n"
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

            # Prompt blocks keep their own XML boundaries so the model can
            # distinguish configuration from situational state. Only the
            # former is admitted to the cacheable system prefix below.
            # Late-binding hint that fires only on this turn — the user
            # just asked you to think hard. Place it AFTER conversation_
            # memory so it has stronger recency-weight than any "answer
            # quickly" framing in summary or staleness blocks.
            deep_think_trigger_hint = ""
            if (
                deep_think_client is not None
                and deep_think_client.status().get("ready")
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
                        bot_id=(
                            ctx.bot.id
                            if getattr(ctx, "bot", None) is not None
                            else None
                        ),
                    )
                except Exception as e:
                    logger.debug(f"Memory preamble build failed: {e}")

            prompt_override = None
            if ctx.policy is not None and ctx.policy.system_prompt:
                prompt_override = ctx.policy.system_prompt
            writer = self._llm_for(ctx)
            base_system_prompt = writer._resolve_system_prompt(
                prompt_override, None
            )

            # Identity block: clarifies which chat handles route to THIS bot so
            # a multi-bot group doesn't confuse the model into answering
            # to another bot's name. Other bots in the room appear as
            # regular participants in <group_context> — the writer
            # should NOT treat them differently from human speakers.
            # The resolved base prompt is passed in so a per-bot/context
            # persona remains authoritative and cannot be contradicted by
            # a stale registry display name.
            identity_block = self._build_identity_block(
                ctx, authoritative_prompt=base_system_prompt,
            )

            # Typed blocks are compiled below.  Request-derived content has a
            # different type from cacheable configuration, so it cannot enter
            # the system prefix through this assembly path.
            stable_system_blocks = [
                StablePromptBlock("your_identity", _wrap_xml("your_identity", identity_block)),
                StablePromptBlock(
                    "turn_pointer_rules",
                    _wrap_xml(
                        "turn_pointer_rules",
                        "Conversation-history rows may begin with a stable "
                        "`[turn h<ID>; ...]` pointer. The current message's "
                        "`parent`/`follows` attributes and <turn_graph> refer "
                        "to those IDs. Graph rows use `child parent=target`: "
                        "the child replied to the target. Linear history "
                        "edges are omitted. Use pointers for reference "
                        "resolution, but never copy them into your reply.",
                    ),
                ),
                StablePromptBlock("attribution_rules", _wrap_xml("attribution_rules", attribution_rules)),
                StablePromptBlock("identity_note", _wrap_xml("identity_note", names_directive)),
                StablePromptBlock("reactor_reflex", _wrap_xml("reactor_reflex", reactor_directive)),
                StablePromptBlock("tarot_tool", _wrap_xml("tarot_tool", tarot_directive)),
                StablePromptBlock("iching_tool", _wrap_xml("iching_tool", iching_directive)),
                StablePromptBlock("portfolio_journal", _wrap_xml("portfolio_journal", journal_directive)),
                StablePromptBlock("deep_think_tool", _wrap_xml("deep_think_tool", deep_think_directive)),
                StablePromptBlock("mcp_broker", _wrap_xml("mcp_broker", mcp_broker_directive)),
                StablePromptBlock("python_tool", _wrap_xml("python_tool", python_tool_directive)),
            ]

            # Trigger user-message: optional group context, optional reply
            # target, then the live message wrapped in <current_message>
            # with an explicit "respond to this" instruction. The wrapper
            # disambiguates the trigger from history + group context, which
            # otherwise just look like more user turns to the model.
            trigger_parts: list[str] = []

            if group_ctx_block:
                trigger_parts.append(group_ctx_block)

            # Volatile reflex state belongs beside the other situational
            # context at the tail of the prompt. This preserves the static
            # system/history cache prefix when a new emoji reaction lands.
            if reactor_log_block:
                trigger_parts.append(
                    _wrap_xml("recent_reactions", reactor_log_block)
                )

            volatile_prompt_blocks = [
                VolatilePromptBlock("spontaneous_reply", _wrap_xml("spontaneous_reply", implicit_directive)),
                VolatilePromptBlock("conversation_memory", _wrap_xml("conversation_memory", summary_block)),
                VolatilePromptBlock("retrieved_history", _wrap_xml("retrieved_history", retrieved_history_block)),
                VolatilePromptBlock("turn_graph", _wrap_xml("turn_graph", turn_graph_block)),
                VolatilePromptBlock("context_memories", _wrap_xml("context_memories", memory_block)),
                VolatilePromptBlock("conversation_status", _wrap_xml("conversation_status", staleness_block)),
                VolatilePromptBlock("deep_think_trigger", _wrap_xml("deep_think_trigger", deep_think_trigger_hint)),
            ]
            trigger_parts.extend(
                block.content for block in volatile_prompt_blocks if block.content
            )

            if ctx.quote_text:
                # Flag this turn as a quote-reply but DON'T embed the
                # quoted text. The original message is already in
                # <group_context> above with its real author/timestamp
                # label; embedding it here a second time duplicated
                # content and — worse — used a different label format
                # (phone-tail for bot quotes via _sender_label, vs.
                # display_name in group_context) so the writer saw the
                # same line twice attributed to two different speakers.
                # That scrambled who-said-what and sometimes caused the
                # writer to decide the thread was already resolved and
                # stay silent. The label resolver below uses the same
                # naming scheme as group_context.
                quoted_label = self._reply_target_label(ctx)
                if reply_turn_id:
                    trigger_parts.append(
                        f'<replying_to turn="{reply_turn_id}" '
                        f'from="{quoted_label}"/>'
                    )
                else:
                    # The source may be older than both retention windows.
                    # In that case preserve Signal's quote payload once,
                    # rather than leaving the model an author-only pointer.
                    trigger_parts.append(
                        f'<replying_to from="{quoted_label}">\n'
                        f'{normalize_signal_text(ctx.quote_text)}\n'
                        f'</replying_to>'
                    )

            sender_label = self._sender_label(ctx.sender) if is_group else "user"
            # For implicit / spontaneous asks the closing instruction softens
            # — the user didn't ask the bot anything, so "Respond to this"
            # would override the bail-freely guidance in the volatile tail.
            closing = (
                "Decide whether to respond. Empty output = stay silent."
                if getattr(ctx, "implicit_reason", None)
                else "Respond to this message."
            )
            follows_attr = (
                f' follows="{latest_turn_id}"' if latest_turn_id else ""
            )
            parent_attr = (
                f' parent="{current_parent_ref}"' if current_parent_ref else ""
            )
            trigger_parts.append(
                f'<current_message from="{sender_label}" sent="just now"'
                f'{follows_attr}{parent_attr}>\n'
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

            # Internal ids/timestamps power deduplication and reply pointers,
            # but provider message objects must contain only API fields.
            provider_prior = [
                {k: v for k, v in turn.items() if not k.startswith("_")}
                for turn in prior
            ]
            tools = self._collect_tools(policy=ctx.policy, bot=ctx.bot)
            compiler = PromptCompiler.with_base_system(base_system_prompt)
            compiler.extend_stable(stable_system_blocks)
            compiler.extend_volatile([
                VolatilePromptBlock("group_context", group_ctx_block),
                VolatilePromptBlock(
                    "recent_reactions",
                    _wrap_xml("recent_reactions", reactor_log_block),
                ),
                *volatile_prompt_blocks,
                VolatilePromptBlock("reply_target", ctx.quote_text or ""),
                VolatilePromptBlock("current_message", question),
            ])
            compiled_prompt = compiler.compile(
                history=provider_prior,
                user_content=user_message_content,
                tools=tools,
            )
            messages = compiled_prompt.messages
            cache_plan = compiled_prompt.cache_plan
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
                and deep_think_client is not None
                and deep_think_client.status().get("ready")
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
                        cache_plan=cache_plan,
                    )
            elif use_research_mode:
                answer = await self._run_research_handoff(
                    ctx=ctx,
                    question=question,
                    research_input=current_user_content,
                    messages=messages,
                    attachments=attachments,
                    user_hash=user_hash,
                    cache_plan=cache_plan,
                )
            elif typing_handler is not None:
                # Show a typing indicator in the chat while the tool loop runs.
                # The indicator auto-clears in ~15s, so the helper refreshes it.
                async with typing_handler.typing_indicator(
                    ctx.sender, ctx.group_id
                ):
                    answer = await self._run_tool_loop(
                        messages, tools, ctx, attachments,
                        cache_plan=cache_plan,
                        question=question,
                    )
            else:
                answer = await self._run_tool_loop(
                    messages, tools, ctx, attachments,
                    cache_plan=cache_plan,
                    question=question,
                )
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
        # through history. Same for pseudo-tool-call markup the model
        # wrote as text instead of a real tool_calls payload.
        answer = _strip_tool_call_leak(_strip_meta_leak(answer))
        # Re-check: if scaffolding was the ENTIRE output, the strip just
        # emptied the answer. Without this guard an empty assistant turn
        # would be persisted (breaking the user/assistant alternation on
        # replay) and the explicit path would send a blank Signal message.
        if not answer:
            return CommandResult.error("LLM returned no answer.")

        try:
            # Store the raw question (no prefix) + sender_tail separately; the
            # load path re-adds the prefix when replaying in a group context.
            # Pass the same turns-per-user value the load path uses so a
            # context-override of 30 doesn't get clipped back to the global
            # 6 on every write.
            # Persist inbound images on the user turn so follow-ups can
            # re-see them (see _inflate_image_history). Only when vision
            # was actually active this round — if the bot doesn't have
            # vision the bytes aren't on inbound_images anyway, but we
            # also gate explicitly so a misconfigured handler can't
            # accidentally bloat history with bytes no one will read.
            persisted_images = inbound_images if vision_active else None
            # bot_id pins multi-bot history: user turns get tagged with
            # the responding bot's id so load() can scope a bot's view to
            # its own conversation arc. Assistant turns get the same tag
            # so each bot's API alternation only includes its own
            # replies — other bots' replies surface in <group_context>
            # via the group_log path, labeled with the writer's name.
            bot_id_for_turn = (
                ctx.bot_id
            )
            # Skip the shared user-turn write for a secondary bot in a
            # multi-bot fan-out — the primary already stored it, and a
            # second copy (role='user' matches every bot's load regardless
            # of bot_id) would replay the human's message twice. The
            # assistant turn below is always persisted: it's this bot's own.
            # Hold the per-context lock across the pair so two concurrent
            # asks in the same context can't interleave their writes
            # (user-A, user-B, assistant-B, assistant-A) and garble the
            # replayed alternation.
            if live_turns > 0:
                async with self.history.lock_for(context_key):
                    current_user_turn_ref = None
                    if getattr(ctx, "persist_user_turn", True):
                        inserted_user_id = await self.history.append(
                            context_key, "user", question,
                            user_hash=user_hash, sender_tail=sender_tail,
                            turns_per_user=live_turns,
                            image_refs=persisted_images,
                            bot_id=bot_id_for_turn,
                            source_message_ts=ctx.message_timestamp,
                            parent_turn_ref=current_parent_ref,
                        )
                        if inserted_user_id is not None:
                            current_user_turn_ref = f"h{inserted_user_id}"
                    else:
                        current_user_turn_ref = await self.history.find_turn_ref_by_source(
                            context_key, ctx.message_timestamp,
                        )
                    # Persist the addressee on the assistant row too: the load
                    # path uses it to render `[to Name, time ago]` in group
                    # playback so the model can pair its prior answers with
                    # the right asker.
                    await self.history.append(
                        context_key, "assistant", answer,
                        user_hash=user_hash, sender_tail=sender_tail,
                        turns_per_user=live_turns,
                        bot_id=bot_id_for_turn,
                        parent_turn_ref=current_user_turn_ref,
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
                # Stamp the row with the writing bot's id so the renderer
                # can attribute the line to the correct bot in a multi-
                # bot group. Other bots reading group_context later see
                # this as a participant utterance under that bot's
                # display_name.
                bot_id_for_log = (
                    ctx.bot.id if getattr(ctx, "bot", None) is not None
                    else None
                )
                await self.group_log.append_bot(
                    ctx.group_id, answer, bot_id=bot_id_for_log,
                )
            except Exception as e:
                logger.error(f"Failed to append bot reply to group log: {e}")

        # Fire-and-forget rolling-summary update. The summarizer itself
        # decides whether enough new turns have arrived to warrant an LLM
        # call, and dedupes via per-context lock. Errors stay inside the
        # task so they never affect the ask response.
        if self.summarizer is not None and live_turns > 0:
            try:
                import asyncio as _aio
                _aio.create_task(
                    self.summarizer.maybe_summarize(
                        context_key,
                        floor_at=floor_at,
                        bot_id=(
                            ctx.bot.id
                            if getattr(ctx, "bot", None) is not None
                            else None
                        ),
                    )
                )
            except Exception as e:
                logger.debug(f"Failed to schedule summarizer: {e}")

        return CommandResult(
            text=answer,
            success=True,
            attachments=attachments or None,
            styled=True,
        )
