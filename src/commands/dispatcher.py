"""
Command dispatcher - routes messages to appropriate handlers.

Supports:
- Standard command prefix (e.g., !price AAPL)
- Inline symbol mentions (e.g., $AAPL in natural text)
"""

import logging
import re
import time
from typing import Optional
from collections import defaultdict
from pathlib import Path

from .base import BaseCommand, CommandContext, CommandResult
from ..cache import get_metrics

logger = logging.getLogger(__name__)

# Audit logger - separate file
_audit_logger: Optional[logging.Logger] = None

def get_audit_logger() -> logging.Logger:
    """Get or create audit logger."""
    global _audit_logger
    if _audit_logger is None:
        _audit_logger = logging.getLogger("audit")
        _audit_logger.setLevel(logging.INFO)
        # Ensure logs directory exists
        log_dir = Path("logs")
        log_dir.mkdir(exist_ok=True)
        # File handler for audit log
        handler = logging.FileHandler(log_dir / "audit.log")
        handler.setFormatter(logging.Formatter(
            "%(asctime)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        ))
        _audit_logger.addHandler(handler)
    return _audit_logger


class UserRateLimiter:
    """Per-user rate limiting."""
    
    def __init__(self, limit: int = 30, window: int = 60):
        self.limit = limit
        self.window = window  # seconds
        self._requests: dict[str, list[float]] = defaultdict(list)
    
    def check(self, user: str) -> tuple[bool, int]:
        """
        Check if user is within rate limit.
        
        Returns:
            (allowed, retry_after_seconds)
        """
        now = time.time()
        window_start = now - self.window
        
        # Clean old requests
        self._requests[user] = [t for t in self._requests[user] if t > window_start]
        
        if len(self._requests[user]) >= self.limit:
            # Calculate when oldest request expires
            oldest = min(self._requests[user])
            retry_after = int(oldest + self.window - now) + 1
            return False, retry_after
        
        # Record this request
        self._requests[user].append(now)
        return True, 0


def levenshtein_distance(s1: str, s2: str) -> int:
    """Calculate Levenshtein distance between two strings."""
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    
    prev_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        curr_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = prev_row[j + 1] + 1
            deletions = curr_row[j] + 1
            substitutions = prev_row[j] + (c1 != c2)
            curr_row.append(min(insertions, deletions, substitutions))
        prev_row = curr_row
    return prev_row[-1]


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
    
    def __init__(self, prefix: str = "!", enable_inline_symbols: bool = True, 
                 bot_name: str = "Stock Bot", rate_limit: int = 30):
        self.prefix = prefix
        self.enable_inline_symbols = enable_inline_symbols
        self.bot_name = bot_name
        self.commands: dict[str, BaseCommand] = {}
        self._rate_limiter = UserRateLimiter(limit=rate_limit)
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
        # Check rate limit
        allowed, retry_after = self._rate_limiter.check(sender)
        if not allowed:
            return CommandResult.error(
                f"Slow down! Try again in {retry_after} seconds."
            )
        
        # Record request metric
        get_metrics().record_request()
        
        # Check for command chaining (multiple commands in one message)
        # e.g., "!price AAPL !ta AAPL"
        # We need to be careful not to split inside quoted strings (though simplified splitting is likely okay for now)
        if message.count(self.prefix) > 1 and message.strip().startswith(self.prefix):
            # Split by prefix, but filter empty strings (e.g. leading prefix)
            commands = [f"{self.prefix}{cmd.strip()}" for cmd in message.split(self.prefix) if cmd.strip()]
            
            if len(commands) > 1:
                results = []
                for cmd_str in commands:
                    parsed = self.parse_message(cmd_str)
                    if parsed:
                        command, args = parsed
                        result = await self._execute_command(command, args, sender, message, group_id)
                        if result:
                            results.append(result)
                
                if results:
                    # Merge results
                    merged_text = "\n\n━━━━━━━━━━━━━━━━━━━━━━\n\n".join([r.text for r in results if r])
                    merged_attachments = []
                    for r in results:
                        if r and r.attachments:
                            merged_attachments.extend(r.attachments)
                    
                    return CommandResult(
                        text=merged_text,
                        success=all(r.success for r in results if r),
                        attachments=merged_attachments
                    )

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
        
        # Try natural language intent parsing (for mentions or direct queries)
        if mentioned or self._looks_like_query(message):
            from .intent_parser import parse_intent
            
            intent = parse_intent(message)
            if intent and intent.confidence >= 0.5:
                logger.info(f"Intent parsed: {intent.command} {intent.symbols} (confidence: {intent.confidence:.2f})")
                return await self._execute_command(intent.command, intent.args, sender, message, group_id)
            
            # If mentioned but no intent, show intro
            if mentioned:
                logger.info("Bot mentioned without command, providing help intro")
                return CommandResult.ok(
                    f"» Hey! I'm {self.bot_name}.\n\n"
                    "Try these:\n"
                    "• !price AAPL - Get stock price\n"
                    "• Chart Apple - Natural language\n"
                    "• $AAPL - Quick lookup\n"
                    "• !help - All commands"
                )
        
        return None
    
    def _looks_like_query(self, text: str) -> bool:
        """Check if text looks like a stock-related query."""
        # Skip short messages
        if len(text) < 5:
            return False
        
        # Check for question indicators
        question_words = ['what', 'how', 'show', 'get', 'tell', 'can', 'give']
        text_lower = text.lower()
        
        for word in question_words:
            if text_lower.startswith(word) or f" {word} " in text_lower:
                return True
        
        # Check for finance keywords
        finance_keywords = [
            'chart', 'price', 'rsi', 'macd', 'earnings', 'dividend',
            'news', 'rating', 'insider', 'short', 'correlation',
        ]
        return any(kw in text_lower for kw in finance_keywords)
    
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
            # Try to suggest closest match
            suggestion = self._find_closest_command(command)
            msg = f"Unknown command: {command}"
            if suggestion:
                msg += f"\n  Did you mean: {self.prefix}{suggestion}"
            msg += f"\nType {self.prefix}help for commands"
            return CommandResult.error(msg)
        
        ctx = CommandContext(
            sender=sender,
            group_id=group_id,
            raw_message=raw_message,
            command=command,
            args=args,
        )
        
        # universal -help flag check
        if handler.has_help_flag(ctx):
             return handler.get_help_result()
        
        # Audit log
        audit = get_audit_logger()
        audit.info(f"{sender[-4:]} | {command} {' '.join(args)}")
        
        try:
            logger.info(f"Executing {command} from {sender[-4:]}: args={args}")
            result = await handler.execute(ctx)
            logger.debug(f"Command {command} completed: success={result.success}")
            return result
            
        except Exception as e:
            logger.exception(f"Error executing command {command}")
            return CommandResult.error(f"Internal error: {type(e).__name__}")
    
    def _find_closest_command(self, typo: str) -> Optional[str]:
        """Find closest command name using Levenshtein distance."""
        unique_commands = set(cmd.name for cmd in self.commands.values())
        best_match = None
        best_distance = float('inf')
        
        for cmd_name in unique_commands:
            distance = levenshtein_distance(typo.lower(), cmd_name.lower())
            if distance < best_distance and distance <= 2:  # Max 2 edits
                best_distance = distance
                best_match = cmd_name
        
        return best_match
    
    def get_commands(self) -> list[BaseCommand]:
        """Get unique list of registered commands"""
        seen = set()
        commands = []
        for cmd in self.commands.values():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd)
        return commands
