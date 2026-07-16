"""
Base classes for bot commands.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from ..bots import Bot


@dataclass
class CommandContext:
    """Context passed to command handlers"""
    sender: str              # Phone number of sender
    group_id: Optional[str]  # Group ID if in group chat
    raw_message: str         # Original message text
    command: str             # The command (e.g., "price")
    args: list[str]          # Arguments after command
    policy: Optional[object] = None   # ContextPolicy resolved for this chat
    # Bot whose voice this command is speaking with. Resolved by the
    # dispatcher from the inbound message — explicit mention/quote
    # routing in PR4, default-bot-for-context until then. AskCommand
    # uses this in PR3 to pick the per-bot LLMClient/DeepThinkClient
    # via the factory; commands that don't speak as the bot can
    # ignore it. Optional only because background workers and tests
    # synthesize CommandContexts before bot resolution exists.
    bot: Optional["Bot"] = None
    # Quoted message info, when the user is replying to another message in
    # Signal. Surfaced as text context for the LLM so it knows what's being
    # responded to. None when the message isn't a reply.
    quote_text: Optional[str] = None
    quote_author: Optional[str] = None  # phone number, when known
    # Signal's dataMessage timestamp is its stable message identifier.  Keep
    # it with the command so group-log rows, history rows, and quote replies
    # can refer to the same source turn without fuzzy text matching.
    message_timestamp: Optional[int] = None
    quote_timestamp: Optional[int] = None
    # Whether this turn should persist the *inbound user message* into
    # conversation history. False for a secondary bot in a multi-bot
    # fan-out (when one message addresses both Sigil and Artaud): the
    # primary bot already stored the shared user turn, so the secondary
    # must skip it or the human's message would appear twice in every
    # bot's replayed history. The secondary still persists its OWN
    # assistant turn (its reply is unique to it).
    persist_user_turn: bool = True
    # Set when this !ask was triggered spontaneously by the reactor's
    # should_respond tool — not by an explicit mention/quote/command. Carries
    # the reactor's one-sentence justification so the writer model can decide
    # whether the inferred intent was actually right (and bail if not).
    implicit_reason: Optional[str] = None
    # Set when this command was invoked by an automated worker rather than
    # a real user message — currently the trading cron uses "cron". The
    # portfolio tool handler reads this to tag trades with their actual
    # provenance (cron vs reactive) instead of always claiming "reactive".
    automation_source: Optional[str] = None
    # Incremented by the portfolio tool dispatcher each time a mutating
    # tool (buy/sell/place_order/cancel_order) succeeds during this
    # command. The trading cron reads this to suppress the chat post
    # when the writer chose to sit out — no moves means no message.
    portfolio_mutation_count: int = 0
    # Image attachments captured from the inbound Signal message. Each
    # entry: {"mime": "image/jpeg", "data_b64": "<base64>", "filename": "..."}.
    # Populated by the signal handler when the resolved bot has
    # vision_enabled=True. Consumed by ask_command's writer-message
    # builder, then dropped: images are one-shot per message and not
    # written into conversation history.
    inbound_images: list[dict] = field(default_factory=list)

    @property
    def is_group(self) -> bool:
        return self.group_id is not None

    def context_key(self) -> str:
        """Stable key identifying the conversation thread.

        Group chats share a thread by group_id; DMs are scoped to the
        hashed phone so phone-number changes don't fragment history.
        """
        from ..database import hash_phone
        if self.group_id:
            return f"group:{self.group_id}"
        return f"dm:{hash_phone(self.sender)}"

    def portfolio_key(self) -> str:
        """`context_key` scoped to the responding bot — so each bot in
        a multi-bot group has its own portfolio/journal/cron arc.

        Centralizes the pattern repeated across every portfolio
        entry point (LLM tool dispatch, chat commands, cron worker).
        Returns the raw context_key when ctx.bot is None — that path
        is used by legacy tests / single-bot DM contexts where the
        scoping is a no-op anyway.
        """
        from ..paper_portfolio import portfolio_context_key
        bot_id = getattr(self.bot, "id", None) if self.bot is not None else None
        return portfolio_context_key(self.context_key(), bot_id)

    @property
    def bot_id(self) -> Optional[int]:
        """Convenience: ctx.bot.id with the None guard pre-applied.
        Most multi-bot scoping sites need just the integer."""
        return getattr(self.bot, "id", None) if self.bot is not None else None


@dataclass
class CommandResult:
    """Result from command execution"""
    text: str
    success: bool = True
    attachments: Optional[list[str]] = None  # Base64-encoded images
    dm_only: bool = False  # If True, send as DM even in group chat
    styled: bool = False   # If True, text contains markdown to be Signal-styled

    @classmethod
    def error(cls, message: str) -> "CommandResult":
        return cls(text=f"◇ {message}", success=False)

    @classmethod
    def ok(cls, message: str) -> "CommandResult":
        return cls(text=message, success=True)

    @classmethod
    def with_chart(cls, message: str, chart_base64: str) -> "CommandResult":
        """Create result with chart attachment."""
        return cls(text=message, success=True, attachments=[chart_base64])


class BaseCommand(ABC):
    """
    Base class for all commands.
    
    Subclasses must define:
    - name: primary command name
    - description: help text
    - usage: example usage
    
    Optional:
    - aliases: alternative command names
    """
    
    name: str
    aliases: list[str] = []
    description: str
    usage: str
    
    help_explanation: str = "No simplified explanation available."

    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute the command and return a result"""
        pass
    
    def matches(self, command: str) -> bool:
        """Check if this handler matches the command"""
        command = command.lower()
        return command == self.name.lower() or command in [a.lower() for a in self.aliases]

    def has_help_flag(self, ctx: CommandContext) -> bool:
        """Check if -help argument is present"""
        return "-help" in [a.lower() for a in ctx.args] or "--help" in [a.lower() for a in ctx.args]

    def get_help_result(self) -> CommandResult:
        """Return simplified help message"""
        return CommandResult.ok(
            f"◈ HELP: !{self.name}\n\n"
            f"{self.help_explanation}\n\n"
            f"Usage: {self.usage}"
        )
