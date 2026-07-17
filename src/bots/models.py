"""Bot dataclass — the identity record stored in the `bots` table."""

from dataclasses import dataclass, field
from typing import Optional


# deep_think_mode values:
#   'replace'  — current behavior: deep_think writes the user-facing reply
#                directly when invoked (used for Sigil).
#   'research' — deep_think runs the tool loop and returns notes; the
#                writer LLM composes the final reply with those notes
#                injected as a system suffix (used for Artaud, whose
#                voice is a locally-trained model). ALWAYS delegates —
#                every turn pays for a deep_think pass.
#   'tool_bot' — like 'research', but the tool loop runs on the bot's
#                dedicated `tool_bot`-role client (its own model/key/url,
#                falling back deep_think_* -> llm_*) AND it self-gates:
#                the tool-bot answers with the sentinel `NOTOOLS` when the
#                turn needs no live data, so the writer composes directly
#                without a wasted handoff. For writer models with poor
#                native tool-calling: a capable sibling does the tools, the
#                persona model only ever writes.
DEEP_THINK_MODE_REPLACE = "replace"
DEEP_THINK_MODE_RESEARCH = "research"
DEEP_THINK_MODE_TOOL_BOT = "tool_bot"
DEEP_THINK_MODES = (
    DEEP_THINK_MODE_REPLACE,
    DEEP_THINK_MODE_RESEARCH,
    DEEP_THINK_MODE_TOOL_BOT,
)


@dataclass
class Bot:
    id: Optional[int]
    slug: str                           # stable identifier; never user-facing
    display_name: str                   # shown to chat ("Sigil", "Artaud")
    aliases: list[str] = field(default_factory=list)
    # Optional persona/system-prompt prefix. Empty/None means the
    # writer's global llm_system_prompt applies. For models trained on
    # their persona (Artaud's local MLX) this can stay short or empty —
    # the weights already know who they are.
    persona: Optional[str] = None
    # NULL on either field means "use the global value from Config."
    # Once a bot gets its own Signal number, populate signal_phone (and
    # signal_api_url if it lives on a different signal-cli-rest-api).
    signal_phone: Optional[str] = None
    signal_api_url: Optional[str] = None
    enabled: bool = True
    # Default-bot flags: which bot answers in a context that hasn't been
    # explicitly assigned. Exactly one bot should be the default per
    # kind; the registry enforces this on upsert.
    default_for_dm: bool = False
    default_for_group: bool = False
    # PR4 fields, declared in the schema now to avoid a later ALTER.
    deep_think_mode: str = DEEP_THINK_MODE_REPLACE
    deep_think_handoff_prompt: Optional[str] = None
    # Vision: when True, inbound image attachments are base64-encoded and
    # passed to the writer LLM in OpenAI multimodal shape (text + image_url
    # parts). Off by default — only flip on for bots whose writer model
    # actually supports vision (e.g. claude-sonnet-4-5, gpt-5-vision, etc.).
    # Images are NOT persisted into conversation_turns (one-shot per
    # message): the next round sees only the text.
    vision_enabled: bool = False
    # Per-bot deep_think kill switch. When False, deep_think is fully
    # disabled FOR THIS BOT regardless of global / per-context settings:
    # the tool schema is hidden from the writer (so it can't invoke it)
    # AND `deep_think_mode='research'` is treated as `'replace'` (no
    # handoff). Default True preserves existing behavior; flip off for
    # bots whose writer model already handles long reasoning natively or
    # for cost containment on a single bot without changing globals.
    deep_think_enabled: bool = True
    # Short "when to pick this bot" hint surfaced to the reactor LLM in
    # multi-bot chats so it can route an unaddressed message to the
    # bot whose remit best fits ("Artaud — poetry, art, philosophy;
    # Sigil — markets, technical questions"). Kept separate from
    # `persona` because persona is a system-prompt fragment the bot
    # speaks WITH, while this is a one-liner the reactor reads ABOUT
    # the bot to decide who should answer. Empty/None means the
    # reactor falls back to the persona's first sentence.
    routing_blurb: Optional[str] = None
    created_at: float = 0.0
    updated_at: float = 0.0

    def alias_set(self) -> set[str]:
        """Lowercased alias set for case-insensitive trigger matching."""
        out = {self.display_name.strip().lower()} if self.display_name else set()
        for a in self.aliases:
            a = (a or "").strip().lower()
            if a:
                out.add(a)
        out.discard("")
        return out
