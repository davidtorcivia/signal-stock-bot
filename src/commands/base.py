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
    
    @classmethod
    def error(cls, message: str) -> "CommandResult":
        return cls(text=f"❌ {message}", success=False)
    
    @classmethod
    def ok(cls, message: str) -> "CommandResult":
        return cls(text=message, success=True)


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
    
    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute the command and return a result"""
        pass
    
    def matches(self, command: str) -> bool:
        """Check if this handler matches the command"""
        command = command.lower()
        return command == self.name.lower() or command in [a.lower() for a in self.aliases]
