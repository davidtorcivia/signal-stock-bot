"""
Bot identity registry.

A "bot" is the unit that owns a persona, a display name, an alias list
that triggers it from chat text, and (optionally) its own Signal phone
number. It also owns three LLM role configs (writer, reactor,
deep_think) stored in `bot_llm_settings` and resolved with a
bot -> global -> default fallback chain.

Schema and seeding live in `registry.py`. Existing single-bot installs
get a `sigil` row inserted on first boot with `aliases=[bot_name]` and
`default_for_dm=default_for_group=1`; consumer tables (contexts,
conversation_turns, predictions, portfolios, context_memories,
metric_events) get a nullable `bot_id` column backfilled to that row.
"""

from .models import Bot
from .registry import BotRegistry, SIGIL_SLUG
from .settings import (
    resolve_setting,
    resolve_int,
    resolve_float,
    resolve_bool,
    resolve_str,
    resolve_stripped,
)

__all__ = [
    "Bot",
    "BotRegistry",
    "SIGIL_SLUG",
    "resolve_setting",
    "resolve_int",
    "resolve_float",
    "resolve_bool",
    "resolve_str",
    "resolve_stripped",
]
