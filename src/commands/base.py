"""
Base classes for bot commands.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional


@dataclass
class CommandContext:
    """Context passed to command handlers"""
    sender: str              # Phone number of sender
    group_id: Optional[str]  # Group ID if in group chat
    raw_message: str         # Original message text
    command: str             # The command (e.g., "price")
    args: list[str]          # Arguments after command
    policy: Optional[object] = None   # ContextPolicy resolved for this chat
    # Quoted message info, when the user is replying to another message in
    # Signal. Surfaced as text context for the LLM so it knows what's being
    # responded to. None when the message isn't a reply.
    quote_text: Optional[str] = None
    quote_author: Optional[str] = None  # phone number, when known
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
