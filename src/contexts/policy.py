"""
ContextPolicy — the resolved access rules for a given chat (group or DM).

Three "modes" per gate:
  * allow_all  — no restriction (default)
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
    mcp_mode: str = MODE_ALLOW_ALL
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


# Fully-open fallback used when the registry can't be consulted for any reason.
PERMISSIVE = ContextPolicy(id=None, kind="default", key="__fallback__")
