"""
Command dispatcher - routes messages to appropriate handlers.

Supports:
- Standard command prefix (e.g., !price AAPL)
- Inline symbol mentions (e.g., $AAPL in natural text)
- Context awareness (pronouns)
- Multi-intent parsing ("Chart Apple and show RSI")
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

# Pattern for "corn" easter egg (Bitcoin insider joke) - whole word only
CORN_PATTERN = re.compile(r'\bcorn\b', re.IGNORECASE)


class CommandDispatcher:
    """
    Routes incoming messages to appropriate command handlers.
    
    Supports:
    - Configurable command prefix
    - Command aliases
    - Inline symbol mentions ($AAPL)
    - Unknown command handling
    - Context awareness
    """
    
    def __init__(self, prefix: str = "!", enable_inline_symbols: bool = True, 
                 bot_name: str = "Stock Bot", rate_limit: int = 30, context_manager=None):
        self.prefix = prefix
        self.enable_inline_symbols = enable_inline_symbols
        self.bot_name = bot_name
        self.context_manager = context_manager
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
        """
        # Check rate limit
        allowed, retry_after = self._rate_limiter.check(sender)
        if not allowed:
            return CommandResult.error(
                f"Slow down! Try again in {retry_after} seconds."
            )
        
        # Record request metric
        get_metrics().record_request()
        
        # Easter egg: Check for "corn" (Bitcoin inside joke)
        if CORN_PATTERN.search(message):
            corn_result = await self._handle_corn_easter_egg()
            if corn_result:
                return corn_result
        
        # Check for command chaining (multiple commands in one message)
        if message.count(self.prefix) > 1 and message.strip().startswith(self.prefix):
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
                    return self._merge_results(results)

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
            
            # Strip bot mentions before NLP parsing (e.g., "@Sigil chart AAPL" -> "chart AAPL")
            cleaned = message
            if mentioned:
                # Remove @mentions (Signal format) and bot name references
                cleaned = re.sub(r'@\S+\s*', '', cleaned, count=1)  # Remove first @mention
                # Also remove standalone bot name references
                cleaned = re.sub(rf'\b{re.escape(self.bot_name)}\b', '', cleaned, flags=re.IGNORECASE)
                cleaned = cleaned.strip()
            
            # Multi-intent support: Split by punctuation or "and"
            # e.g., "Chart Apple. Show me its RSI" or "Chart AAPL and RSI"
            
            # Protect common abbreviations from being split
            ABBREVIATIONS = ['U.S.', 'U.K.', 'e.g.', 'i.e.', 'Inc.', 'Corp.', 'Ltd.', 'Dr.', 'Mr.', 'Mrs.', 'vs.']
            abbrev_placeholders = {}
            for i, abbr in enumerate(ABBREVIATIONS):
                placeholder = f"__ABBR{i}__"
                if abbr in cleaned:
                    cleaned = cleaned.replace(abbr, placeholder)
                    abbrev_placeholders[placeholder] = abbr
            
            # Smart splitting on " and " - only split if it separates commands, not parameters
            # Heuristic: split on " and " only if what follows looks like a command
            # (contains a verb like 'show', 'chart', 'get', 'what', etc.)
            command_verbs = {'show', 'chart', 'get', 'what', 'give', 'tell', 'price', 'check', 'find', 'do'}
            
            def should_split_on_and(text):
                """Determine if ' and ' should split this text."""
                parts = re.split(r'\s+and\s+', text, maxsplit=1, flags=re.IGNORECASE)
                if len(parts) < 2:
                    return False
                after_and = parts[1].lower().split()[0] if parts[1].split() else ""
                # Split if what follows starts with a command verb or looks like new content
                return after_and in command_verbs or any(c.isupper() for c in after_and)
            
            if should_split_on_and(cleaned):
                cleaned = re.sub(r'\s+and\s+', ' <SEP> ', cleaned, flags=re.IGNORECASE)
            
            # Split regex: Matches .?! NOT followed by a digit (to protect 1.5)
            split_pattern = r'[.?!]+(?!\d)'
            raw_segments = re.split(split_pattern, cleaned)
            
            segments = []
            for s in raw_segments:
                # Handle <SEP> from "and" normalization
                parts = s.split(' <SEP> ')
                for p in parts:
                    # Restore abbreviations
                    for placeholder, abbr in abbrev_placeholders.items():
                        p = p.replace(placeholder, abbr)
                    if p.strip():
                        segments.append(p.strip())
            
            logger.debug(f"Dispatch segments: {segments}")
            
            results = []
            # Context chaining: track symbol from previous segment for pronoun resolution
            chained_symbol = None
            
            for i, segment in enumerate(segments):
                intent = parse_intent(segment)
                
                # Apply confidence decay for later segments (less certain)
                if intent and i > 0:
                    intent.confidence *= 0.95  # 5% decay per segment
                
                if intent and intent.confidence >= 0.5:
                    # Context chaining: if intent has no symbols but uses pronouns, inject chained symbol
                    if not intent.symbols and chained_symbol:
                        # Check if intent args contain pronouns
                        if any(p in intent.args for p in ('it', 'that', 'this', 'its')):
                            intent.args = [chained_symbol if a in ('it', 'that', 'this', 'its') else a for a in intent.args]
                            intent.symbols = [chained_symbol]
                    
                    # Update chained symbol for next segment
                    if intent.symbols:
                        chained_symbol = intent.symbols[0]
                    
                    logger.info(f"Intent parsed: {intent.command} {intent.symbols} (confidence: {intent.confidence:.2f})")
                    result = await self._execute_command(intent.command, intent.args, sender, message, group_id)
                    if result:
                        results.append(result)
            
            if results:
                if len(results) == 1:
                    return results[0]
                return self._merge_results(results)
            
            # If mentioned but no intent, show intro
            if mentioned and not results:
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
    
    def _merge_results(self, results: list[CommandResult]) -> CommandResult:
        """Merge multiple command results into one."""
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

    def _looks_like_query(self, text: str) -> bool:
        """Check if text looks like a stock-related query."""
        if len(text) < 3:
            return False
            
        text_lower = text.lower()
        
        # Question starters
        question_words = ['what', 'how', 'show', 'get', 'tell', 'can', 'give', 'is', 'should', 'would', 'could', 'do', 'does', 'will']
        for word in question_words:
            if text_lower.startswith(word) or f" {word} " in text_lower:
                return True
        
        # Sentiment/advice patterns
        if any(p in text_lower for p in ['buy', 'sell', 'hold', 'invest', 'bullish', 'bearish', 'good investment', 'bad investment']):
            return True
        
        # Finance keywords
        finance_keywords = [
            'chart', 'price', 'rsi', 'macd', 'earnings', 'dividend',
            'news', 'rating', 'insider', 'short', 'correlation',
            'stock', 'share', 'market', 'crypto', 'bitcoin', 'analysis',
            'candlestick', 'candlesticks', 'bollinger', 'sma', 'ema',
        ]
        return any(kw in text_lower for kw in finance_keywords)

    async def _resolve_context(self, sender: str, args: list[str]) -> list[str]:
        """Resolve context-dependent arguments."""
        if not self.context_manager or not args:
            return args
        
        first_arg = args[0].lower()
        if first_arg in ('it', 'that', 'this', 'its'):
            import hashlib
            user_hash = hashlib.sha256(sender.encode()).hexdigest()
            ctx = await self.context_manager.get_context(user_hash)
            
            if ctx.last_symbol:
                logger.info(f"Resolved pronoun '{first_arg}' to '{ctx.last_symbol}' for {sender[-4:]}")
                args[0] = ctx.last_symbol
                
        return args

    async def _update_context(self, sender: str, command: str, args: list[str], success: bool):
        """Update context after command execution."""
        if not self.context_manager or not success:
            return
            
        symbol = None
        if args:
            candidate = args[0].upper()
            if candidate.isalpha() and 2 <= len(candidate) <= 6:
                symbol = candidate
        
        if symbol:
            import hashlib
            user_hash = hashlib.sha256(sender.encode()).hexdigest()
            await self.context_manager.update_context(user_hash, symbol=symbol, intent=command)
    
    async def _execute_command(
        self,
        command: str,
        args: list[str],
        sender: str,
        raw_message: str,
        group_id: Optional[str]
    ) -> CommandResult:
        """Execute a command with the given arguments."""
        # Resolve context (pronouns)
        args = await self._resolve_context(sender, args)

        handler = self.commands.get(command)
        if not handler:
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
        
        if handler.has_help_flag(ctx):
             return handler.get_help_result()
        
        audit = get_audit_logger()
        audit.info(f"{sender[-4:]} | {command} {' '.join(args)}")
        
        try:
            logger.info(f"Executing {command} from {sender[-4:]}: args={args}")
            result = await handler.execute(ctx)
            logger.debug(f"Command {command} completed: success={result.success}")
            
            # Update context if successful
            await self._update_context(sender, command, args, result.success)
            
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
    
    async def _handle_corn_easter_egg(self) -> Optional[CommandResult]:
        """
        Handle 'corn' easter egg - return Bitcoin price and chart.
        
        Inside joke: 'corn' = Bitcoin in the group chat.
        """
        try:
            # Get the chart command to access providers and chart generator
            chart_cmd = self.commands.get("chart")
            if not chart_cmd:
                # Fallback to price command if chart not available
                price_cmd = self.commands.get("price")
                if price_cmd:
                    from .base import CommandContext
                    ctx = CommandContext(
                        sender="corn_easter_egg",
                        group_id=None,
                        raw_message="corn",
                        command="price",
                        args=["BTC-USD"],
                    )
                    return await price_cmd.execute(ctx)
                return None
            
            # Get BTC quote
            quote = await chart_cmd.providers.get_quote("BTC-USD")
            
            # Get historical bars for last 24h candlestick chart
            # Crypto trades 24/7, so use "5d" period with 15m bars and slice to ~24h
            # (96 bars × 15min = 24 hours)
            bars = await chart_cmd.providers.get_historical("BTC-USD", period="5d", interval="15m")
            if len(bars) > 96:
                bars = bars[-96:]  # Last 24 hours
            
            # Calculate actual 24h change from historical bars
            if bars and len(bars) >= 2:
                start_price = bars[0].open
                end_price = bars[-1].close
                change_24h = end_price - start_price
                change_24h_pct = ((end_price / start_price) - 1) * 100 if start_price > 0 else 0
            else:
                change_24h = quote.change
                change_24h_pct = quote.change_percent
            
            # Generate candlestick chart with corrected 24h change
            from ..charts import ChartOptions
            options = ChartOptions(
                chart_type="candle",
                show_volume=True,
            )
            
            generator = chart_cmd._get_generator()
            chart_b64 = generator.generate(
                symbol="BTC-USD",
                bars=bars,
                period="24h",
                current_price=quote.price,
                change_percent=change_24h_pct,  # Use calculated 24h change
                options=options,
            )
            
            # Calculate fun stats
            sats_per_dollar = int(100_000_000 / quote.price) if quote.price > 0 else 0
            
            # Price direction indicator (based on 24h change)
            if change_24h_pct > 3:
                mood = "Absolutely SHUCKING it"
                indicator = ">>>"
            elif change_24h_pct > 0:
                mood = "Growing nicely"
                indicator = ">"
            elif change_24h_pct > -3:
                mood = "Bit wilted"
                indicator = "v"
            else:
                mood = "Getting harvested"
                indicator = "vvv"
            
            # Format response with fun stats (no emojis, only unicode)
            lines = [
                "* * * CORN REPORT * * *",
                "",
                f"{indicator} ${quote.price:,.2f}",
                f"   {'+' if change_24h >= 0 else ''}{change_24h:,.2f} ({change_24h_pct:+.2f}% 24h)",
                "",
                f"Mood: {mood}",
                f"Sats per dollar: {sats_per_dollar:,}",
            ]
            
            # Add volume if available
            if quote.volume:
                vol_billions = quote.volume * quote.price / 1_000_000_000
                lines.append(f"24h Volume: ${vol_billions:.1f}B")
            
            lines.append("")
            lines.append("~~ stack sats ~~")
            
            logger.info("Corn easter egg triggered - returning BTC price and chart")
            
            return CommandResult(
                text="\n".join(lines),
                success=True,
                attachments=[chart_b64] if chart_b64 else None,
            )
            
        except Exception as e:
            logger.error(f"Corn easter egg error: {e}")
            # Silent failure - don't spam the chat with errors
            return None
