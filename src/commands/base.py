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
    
    @property
    def is_group(self) -> bool:
        return self.group_id is not None


@dataclass
class CommandResult:
    """Result from command execution"""
    text: str
    success: bool = True
    attachments: Optional[list[str]] = None  # Base64-encoded images
    
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
