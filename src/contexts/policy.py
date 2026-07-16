"""
ContextPolicy — the resolved access rules for a given chat (group or DM).

Three "modes" per gate:
  * allow_all  — no restriction
  * allow_list — only the listed items pass
  * deny_list  — everything except the listed items passes
"""

from dataclasses import dataclass, field
from typing import Optional


MODE_ALLOW_ALL = "allow_all"
MODE_ALLOW_LIST = "allow_list"
MODE_DENY_LIST = "deny_list"
MODES = (MODE_ALLOW_ALL, MODE_ALLOW_LIST, MODE_DENY_LIST)


@dataclass
class ContextPolicy:
    id: Optional[int]
    kind: str                      # 'group' | 'dm' | 'default'
    key: str                       # group_id, phone, or 'default:group'/'default:dm'
    label: str = ""
    command_mode: str = MODE_ALLOW_ALL
    commands: list[str] = field(default_factory=list)
    # MCP schemas are unusually expensive: a handful of enabled servers can
    # add tens of thousands of tokens to every writer request. New contexts
    # therefore start with an empty, stable allow-list and must opt servers in
    # deliberately. Existing database rows keep their stored mode.
    mcp_mode: str = MODE_ALLOW_LIST
    mcp_servers: list[str] = field(default_factory=list)
    system_prompt: Optional[str] = None   # None / empty => use global LLM prompt
    llm_intent: bool = False              # Route non-command messages through the LLM with bot+MCP tools
    # Emoji reactor: per-context override of the global setting + dedicated prompt
    reactor_enabled: bool = True          # Only matters when reactor is globally enabled
    reactor_prompt: Optional[str] = None  # None / empty => use global reactor prompt
    # Natural response: when True, the reactor LLM also gets a should_respond
    # tool so it can decide to write a reply (not just react) for messages that
    # weren't directed at the bot via mention/quote. Requires reactor_enabled.
    natural_response: bool = False
    # Deep think: when True, the writer LLM gets a deep_think tool that
    # delegates hard sub-problems to a configurable smarter model. Default
    # on — the global kill switch is enough to disable it everywhere; per-
    # context flag exists for cost containment in cheap chats.
    deep_think_enabled: bool = True
    # Memory: per-context gates for the bot's per-context memory store.
    # `memory_writes_enabled` controls whether the writer LLM can call
    # `remember`/`forget` in this chat (default on — explicit user-driven
    # writes are always wanted). `reactor_memory_writes` controls whether
    # the reactor passively learns from messages even when the main bot
    # isn't called (default off — opt-in, since passive learning has
    # privacy implications).
    memory_writes_enabled: bool = True
    reactor_memory_writes: bool = False
    # Multi-bot scoping: which bot answers in this context when no bot
    # is named in the message. NULL = use the registry's
    # default_for_kind. Set to a specific bot.id to pin (e.g. an
    # Artaud-only group). PR4 mention routing can still let a non-
    # default bot answer when explicitly addressed by alias.
    default_bot_id: Optional[int] = None
    # Transcript logging: when True, every writer LLM round in this chat
    # is appended to data/transcripts/ctx_<id>.jsonl as one OpenAI-shape
    # JSON object (messages sent + assistant reply + tool calls + params).
    # Off by default; only writable on explicit (non-default) rows so the
    # catch-all rows don't accumulate mixed-context training data. Used
    # to harvest fine-tuning corpora for models trained against this
    # harness.
    transcript_logging_enabled: bool = False
    # History-turns override: when set, this context uses its own value
    # for "how many prior turn-pairs to send to the writer LLM" instead
    # of the global `llm_history_turns`. None = inherit global. 0 = no
    # history at all (one-shot mode — useful for oracle-only chats).
    history_turns_override: Optional[int] = None
    # Purge floor: a hard cutoff timestamp. Conversation turns, summaries,
    # and ALL group_log rows (bot AND user) older than this are invisible
    # to every read path that assembles LLM context — even if the
    # underlying rows haven't been deleted yet. Set by the admin "Purge
    # context" button; the purge action also DELETEs pre-floor rows for
    # cleanup, but the floor is what guarantees the bot can't see the
    # pre-purge conversation forever going forward.
    # None = no floor (the default). context_memories are NOT gated by the
    # floor — they're durable by design.
    purge_floor_at: Optional[float] = None

    def storage_key(self) -> str:
        """Key that conversation_turns / summaries are stored under for
        this context — `group:<id>` or `dm:<hash>` — mirroring
        CommandContext.context_key().

        This is NOT `self.key`: the context ROW is indexed by the raw
        group_id (groups) or phone (DMs), but history is keyed with a
        `group:` / `dm:` prefix (and DMs hash the phone). The purge action
        needs this form to clear the right rows — passing `self.key`
        directly matched nothing, so purges never freed history.
        """
        if self.kind == "group":
            return f"group:{self.key}"
        if self.kind == "dm":
            from ..database import hash_phone
            return f"dm:{hash_phone(self.key)}"
        return self.key

    def allows_command(self, name: str) -> bool:
        name = (name or "").lower()
        listed = {c.lower() for c in self.commands}
        if self.command_mode == MODE_ALLOW_LIST:
            return name in listed
        if self.command_mode == MODE_DENY_LIST:
            return name not in listed
        return True

    def allows_mcp(self, server_name: str) -> bool:
        listed = set(self.mcp_servers)
        if self.mcp_mode == MODE_ALLOW_LIST:
            return server_name in listed
        if self.mcp_mode == MODE_DENY_LIST:
            return server_name not in listed
        return True

    def allows_deep_think(self) -> bool:
        return self.deep_think_enabled


# Fully-open fallback used when the registry can't be consulted for any reason.
# Keep this explicit now that normal ContextPolicy instances default MCP access
# to an empty allow-list.
PERMISSIVE = ContextPolicy(
    id=None,
    kind="default",
    key="__fallback__",
    mcp_mode=MODE_ALLOW_ALL,
)
