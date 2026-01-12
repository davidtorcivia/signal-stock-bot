"""
Watchlist command for stock bot.

Provides: !watch, !watch add, !watch remove, !watch clear
"""

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager
from ..database import WatchlistDB, hash_phone


class WatchCommand(BaseCommand):
    """Manage personal stock watchlist."""
    name = "watch"
    aliases = ["w", "watchlist"]
    description = "Manage your watchlist"
    usage = "!watch [add|remove|clear] [symbols]"
    help_explanation = """Track your favorite stocks with a personal watchlist.

**Commands:**
• !watch — View your watchlist with live prices
• !watch add AAPL MSFT — Add symbols to your list
• !watch remove TSLA — Remove a symbol
• !watch clear — Clear your entire watchlist

**Pro Tips:**
• Add stocks you're watching for entries or exits.
• Your watchlist persists across sessions.
• Limit: 50 symbols per user."""
    
    def __init__(self, provider_manager: ProviderManager, watchlist_db: WatchlistDB):
        self.providers = provider_manager
        self.db = watchlist_db
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        user_hash = hash_phone(ctx.sender)
        
        if not ctx.args:
            return await self._show_watchlist(user_hash)
        
        subcommand = ctx.args[0].lower()
        
        if subcommand == "add":
            return await self._add_symbols(user_hash, ctx.args[1:])
        elif subcommand == "remove" or subcommand == "rm":
            return await self._remove_symbol(user_hash, ctx.args[1:])
        elif subcommand == "clear":
            return await self._clear_watchlist(user_hash)
        else:
            # Treat as symbols to add if they look like symbols
            # e.g., "!watch AAPL" should add AAPL
            return await self._add_symbols(user_hash, ctx.args)
    
    async def _show_watchlist(self, user_hash: str) -> CommandResult:
        """Display watchlist with live prices."""
        symbols = await self.db.get_watchlist(user_hash)
        
        if not symbols:
            return CommandResult.ok(
                "◈ Your Watchlist\n\n"
                "No symbols yet.\n"
                "Add with: !watch add AAPL MSFT"
            )
        
        try:
            quotes = await self.providers.get_quotes(symbols)
        except Exception:
            quotes = {}
        
        lines = ["◈ Your Watchlist", ""]
        
        for symbol in symbols:
            if symbol in quotes:
                q = quotes[symbol]
                indicator = "▲" if q.change >= 0 else "▼"
                sign = "+" if q.change >= 0 else ""
                lines.append(
                    f"{indicator} {symbol}: ${q.price:.2f} ({sign}{q.change_percent:.2f}%)"
                )
            else:
                lines.append(f"◇ {symbol}: N/A")
        
        # Add timestamp
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        lines.append(f"\n◷ as of {now.strftime('%-I:%M %p ET')}")
        
        return CommandResult.ok("\n".join(lines))
    
    async def _add_symbols(self, user_hash: str, symbols: list[str]) -> CommandResult:
        """Add symbols to watchlist."""
        if not symbols:
            return CommandResult.error("Specify symbols to add: !watch add AAPL MSFT")
        
        # Validate and resolve symbols
        from ..utils import resolve_symbol
        from .stock_commands import validate_symbol
        
        valid_symbols = []
        invalid = []
        
        for s in symbols[:20]:  # Limit batch size
            try:
                resolved, _ = await resolve_symbol(s)
                is_valid, result = validate_symbol(resolved)
                if is_valid:
                    valid_symbols.append(result)
                else:
                    invalid.append(s)
            except Exception:
                invalid.append(s)
        
        if not valid_symbols:
            return CommandResult.error(f"No valid symbols: {', '.join(invalid)}")
        
        added, skipped = await self.db.add_symbols(user_hash, valid_symbols)
        
        lines = []
        if added:
            lines.append(f"Added {added} symbol(s) to watchlist")
        if skipped:
            lines.append(f"Skipped (limit reached): {', '.join(skipped)}")
        if invalid:
            lines.append(f"Invalid: {', '.join(invalid)}")
        if added == 0 and not skipped and not invalid:
            lines.append("Symbols already in watchlist")
        
        return CommandResult.ok("\n".join(lines))
    
    async def _remove_symbol(self, user_hash: str, symbols: list[str]) -> CommandResult:
        """Remove symbol from watchlist."""
        if not symbols:
            return CommandResult.error("Specify symbol to remove: !watch remove TSLA")
        
        symbol = symbols[0].upper()
        removed = await self.db.remove_symbol(user_hash, symbol)
        
        if removed:
            return CommandResult.ok(f"Removed {symbol} from watchlist")
        else:
            return CommandResult.error(f"{symbol} not in your watchlist")
    
    async def _clear_watchlist(self, user_hash: str) -> CommandResult:
        """Clear entire watchlist."""
        count = await self.db.clear(user_hash)
        
        if count:
            return CommandResult.ok(f"Cleared {count} symbol(s) from watchlist")
        else:
            return CommandResult.ok("Watchlist already empty")
