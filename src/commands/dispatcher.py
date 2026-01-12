"""
Command dispatcher - routes messages to appropriate handlers.

Supports:
- Standard command prefix (e.g., !price AAPL)
- Inline symbol mentions (e.g., $AAPL in natural text)
"""

import logging
import re
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..cache import get_metrics

logger = logging.getLogger(__name__)

# Pattern for inline symbol mentions like $AAPL, $BTC-USD
SYMBOL_MENTION_PATTERN = re.compile(r'\$([A-Z]{1,5}(?:[-\.][A-Z]{1,5})?)', re.IGNORECASE)


class CommandDispatcher:
    """
    Routes incoming messages to appropriate command handlers.
    
    Supports:
    - Configurable command prefix
    - Command aliases
    - Inline symbol mentions ($AAPL)
    - Unknown command handling
    """
    
    def __init__(self, prefix: str = "!", enable_inline_symbols: bool = True, bot_name: str = "Stock Bot"):
        self.prefix = prefix
        self.enable_inline_symbols = enable_inline_symbols
        self.bot_name = bot_name
        self.commands: dict[str, BaseCommand] = {}
        self._pattern = re.compile(
            rf"^{re.escape(prefix)}([\w?]+)(?:\s+(.*))?$",
            re.IGNORECASE | re.DOTALL
        )
    
    def register(self, command: BaseCommand):
        """Register a command handler"""
        self.commands[command.name.lower()] = command
        for alias in command.aliases:
            self.commands[alias.lower()] = command
        logger.info(f"Registered command: {command.name} (aliases: {command.aliases})")
    
    def parse_message(self, text: str) -> Optional[tuple[str, list[str]]]:
        """
        Parse message into command and args.
        Returns None if not a command.
        """
        text = text.strip()
        match = self._pattern.match(text)
        
        if not match:
            return None
        
        command = match.group(1).lower()
        args_str = match.group(2) or ""
        
        # Split args, preserving quoted strings
        args = args_str.split() if args_str.strip() else []
        
        return command, args
    
    def extract_inline_symbols(self, text: str) -> list[str]:
        """
        Extract $SYMBOL mentions from natural text.
        Returns list of symbols without the $ prefix.
        """
        matches = SYMBOL_MENTION_PATTERN.findall(text)
        # Deduplicate while preserving order
        seen = set()
        symbols = []
        for symbol in matches:
            upper = symbol.upper()
            if upper not in seen:
                seen.add(upper)
                symbols.append(upper)
        return symbols[:10]  # Limit to 10 symbols
    
    async def dispatch(
        self,
        sender: str,
        message: str,
        group_id: Optional[str] = None,
        mentioned: bool = False
    ) -> Optional[CommandResult]:
        """
        Dispatch a message to the appropriate command handler.
        
        Args:
            sender: Phone number of sender
            message: Message text
            group_id: Group ID if in group chat
            mentioned: True if bot was @mentioned
        
        Returns:
            CommandResult if message was a command or contained inline symbols
            None if message was not a command
        """
        # Record request metric
        get_metrics().record_request()

        # First try standard command parsing
        parsed = self.parse_message(message)
        
        if parsed:
            command, args = parsed
            return await self._execute_command(command, args, sender, message, group_id)
        
        # Try inline symbol detection if enabled
        if self.enable_inline_symbols:
            symbols = self.extract_inline_symbols(message)
            if symbols:
                logger.info(f"Detected inline symbols: {symbols}")
                # Route to price command
                return await self._execute_command("price", symbols, sender, message, group_id)
        
        # If bot was @mentioned but no command or symbols found, provide help
        if mentioned:
            logger.info("Bot mentioned without command, providing help intro")
            return CommandResult.ok(
                f"» Hey! I'm {self.bot_name}.\n\n"
                "Try these:\n"
                "• !price AAPL - Get stock price\n"
                "• !crypto - Top cryptocurrencies\n"
                "• $AAPL - Quick lookup\n"
                "• !help - All commands"
            )
        
        return None
    
    async def _execute_command(
        self,
        command: str,
        args: list[str],
        sender: str,
        raw_message: str,
        group_id: Optional[str]
    ) -> CommandResult:
        """Execute a command with the given arguments."""
        handler = self.commands.get(command)
        if not handler:
            available = ", ".join(
                sorted(set(cmd.name for cmd in self.commands.values()))
            )
            return CommandResult.error(
                f"Unknown command: {command}\n"
                f"Available: {available}\n"
                f"Type {self.prefix}help for help"
            )
        
        ctx = CommandContext(
            sender=sender,
            group_id=group_id,
            raw_message=raw_message,
            command=command,
            args=args,
        )
        
        try:
            logger.info(f"Executing {command} from {sender[-4:]}: args={args}")
            result = await handler.execute(ctx)
            logger.debug(f"Command {command} completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.exception(f"Error executing command {command}")
            return CommandResult.error(f"Internal error: {type(e).__name__}")
    
    def get_commands(self) -> list[BaseCommand]:
        """Get unique list of registered commands"""
        seen = set()
        commands = []
        for cmd in self.commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd)
        return commands
