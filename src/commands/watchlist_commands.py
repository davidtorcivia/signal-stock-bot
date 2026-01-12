"""
Watchlist command for stock bot.

Provides: !watch, !watch add, !watch remove, !watch clear, !watch sort, !watch export
"""

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager
from ..database import WatchlistDB, hash_phone


class WatchCommand(BaseCommand):
    """Manage personal stock watchlist."""
    name = "watch"
    aliases = ["w", "watchlist"]
    description = "Manage your watchlist"
    usage = "!watch [add|remove|clear|sort|export] [symbols]"
    help_explanation = """Track your favorite stocks with a personal watchlist.

**Commands:**
• !watch — View your watchlist with live prices
• !watch add AAPL MSFT — Add symbols to your list
• !watch remove TSLA — Remove a symbol
• !watch clear — Clear your entire watchlist
• !watch sort [alpha|change] — Sort by name or % change
• !watch export — Get list as CSV (sent as DM)

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
        elif subcommand == "sort":
            return await self._show_watchlist(user_hash, sort_by=ctx.args[1] if len(ctx.args) > 1 else "change")
        elif subcommand == "export":
            return await self._export_watchlist(user_hash)
        else:
            # Treat as symbols to add if they look like symbols
            # e.g., "!watch AAPL" should add AAPL
            return await self._add_symbols(user_hash, ctx.args)
    
    async def _show_watchlist(self, user_hash: str, sort_by: str = None) -> CommandResult:
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
        
        # Build quote data for sorting
        quote_data = []
        for symbol in symbols:
            if symbol in quotes:
                q = quotes[symbol]
                quote_data.append((symbol, q.price, q.change_percent, True))
            else:
                quote_data.append((symbol, 0, 0, False))
        
        # Sort if requested
        if sort_by:
            sort_by = sort_by.lower()
            if sort_by in ("alpha", "name", "a"):
                quote_data.sort(key=lambda x: x[0])
            elif sort_by in ("change", "percent", "pct", "%"):
                quote_data.sort(key=lambda x: x[2], reverse=True)
        
        lines = ["◈ Your Watchlist", ""]
        
        up_count = 0
        down_count = 0
        
        for symbol, price, change_pct, has_quote in quote_data:
            if has_quote:
                indicator = "▲" if change_pct >= 0 else "▼"
                sign = "+" if change_pct >= 0 else ""
                lines.append(f"{indicator} {symbol}: ${price:.2f} ({sign}{change_pct:.2f}%)")
                if change_pct >= 0:
                    up_count += 1
                else:
                    down_count += 1
            else:
                lines.append(f"◇ {symbol}: N/A")
        
        # Add change summary
        if up_count > 0 or down_count > 0:
            lines.append(f"\n▲{up_count} ▼{down_count}")
        
        # Add timestamp
        from datetime import datetime
        from zoneinfo import ZoneInfo
        now = datetime.now(ZoneInfo("America/New_York"))
        lines.append(f"◷ as of {now.strftime('%-I:%M %p ET')}")
        
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
    
    async def _export_watchlist(self, user_hash: str) -> CommandResult:
        """Export watchlist as CSV - sent as DM."""
        symbols = await self.db.get_watchlist(user_hash)
        
        if not symbols:
            return CommandResult.ok("No symbols to export")
        
        csv_list = ", ".join(symbols)
        result = CommandResult.ok(
            f"◈ Watchlist Export\n\n"
            f"{csv_list}\n\n"
            f"({len(symbols)} symbols)"
        )
        result.dm_only = True  # Send as DM, not to group
        return result
