"""
Stock-related command implementations.
"""

import re
from typing import Optional

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager, SymbolNotFoundError, ProviderError

# Valid symbol pattern: alphanumeric, dots, hyphens, carets (for indices)
# Examples: AAPL, BRK.B, BTC-USD, ^GSPC
SYMBOL_PATTERN = re.compile(r'^[\^]?[A-Z0-9]{1,10}(?:[.-][A-Z0-9]{1,5})?$', re.IGNORECASE)


def validate_symbol(symbol: str) -> tuple[bool, str]:
    """
    Validate and sanitize a stock symbol.
    Returns (is_valid, sanitized_symbol or error_message).
    """
    symbol = symbol.strip().upper()
    
    if not symbol:
        return False, "Empty symbol"
    
    if len(symbol) > 15:
        return False, f"Symbol too long: {symbol[:10]}..."
    
    if not SYMBOL_PATTERN.match(symbol):
        return False, f"Invalid symbol format: {symbol}"
    
    return True, symbol


def format_number(n: Optional[float | int], prefix: str = "") -> str:
    """Format large numbers with K/M/B/T suffixes"""
    if n is None:
        return "N/A"
    
    abs_n = abs(n)
    if abs_n >= 1_000_000_000_000:
        return f"{prefix}{n/1_000_000_000_000:.2f}T"
    if abs_n >= 1_000_000_000:
        return f"{prefix}{n/1_000_000_000:.2f}B"
    if abs_n >= 1_000_000:
        return f"{prefix}{n/1_000_000:.2f}M"
    if abs_n >= 1_000:
        return f"{prefix}{n/1_000:.2f}K"
    if isinstance(n, float):
        return f"{prefix}{n:.2f}"
    return f"{prefix}{n}"


def format_change(change: float, pct: float) -> str:
    """Format price change with emoji indicator"""
    arrow = "📈" if change >= 0 else "📉"
    sign = "+" if change >= 0 else ""
    return f"{arrow} {sign}{change:.2f} ({sign}{pct:.2f}%)"


def format_price(price: float) -> str:
    """Format price with appropriate decimal places"""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"


class PriceCommand(BaseCommand):
    name = "price"
    aliases = ["p", "pr", "$"]
    description = "Get current stock price"
    usage = "!price AAPL [MSFT GOOGL ...]"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                f"Usage: {self.usage}\n"
                f"💡 Tip: You can also just type $AAPL in any message"
            )
        
        # Validate and sanitize symbols
        symbols = []
        invalid = []
        for s in ctx.args[:10]:  # Limit to 10
            valid, result = validate_symbol(s)
            if valid:
                symbols.append(result)
            else:
                invalid.append(s)
        
        if not symbols:
            return CommandResult.error(
                f"No valid symbols provided.\n"
                f"Valid formats: AAPL, BRK.B, BTC-USD, ^GSPC"
            )
        
        try:
            if len(symbols) == 1:
                quote = await self.providers.get_quote(symbols[0])
                name_display = f"{quote.name} ({quote.symbol})" if quote.name else quote.symbol
                
                return CommandResult.ok(
                    f"{name_display}\n"
                    f"💵 {format_price(quote.price)}\n"
                    f"{format_change(quote.change, quote.change_percent)}\n"
                    f"📊 Vol: {format_number(quote.volume)}"
                )
            else:
                quotes = await self.providers.get_quotes(symbols)
                if not quotes:
                    return CommandResult.error("No quotes found")
                
                lines = []
                for symbol in symbols:
                    if symbol in quotes:
                        q = quotes[symbol]
                        sign = "+" if q.change >= 0 else ""
                        emoji = "🟢" if q.change >= 0 else "🔴"
                        lines.append(
                            f"{emoji} {q.symbol}: {format_price(q.price)} "
                            f"({sign}{q.change_percent:.2f}%)"
                        )
                    else:
                        lines.append(f"⚪ {symbol}: Not found")
                
                return CommandResult.ok("\n".join(lines))
                
        except SymbolNotFoundError:
            hint = ""
            sym = symbols[0]
            # Suggest crypto format if it looks like crypto
            if sym in ("BTC", "ETH", "SOL", "DOGE", "XRP"):
                hint = f"\n💡 Try {sym}-USD for crypto"
            return CommandResult.error(f"Symbol not found: {sym}{hint}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")


class QuoteCommand(BaseCommand):
    name = "quote"
    aliases = ["q", "detail"]
    description = "Get detailed stock quote"
    usage = "!quote AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            quote = await self.providers.get_quote(symbol)
            name_display = quote.name or symbol
            
            lines = [
                f"📊 {name_display} ({quote.symbol})",
                "",
                f"Price: {format_price(quote.price)}",
                format_change(quote.change, quote.change_percent),
                "",
            ]
            
            if quote.open is not None:
                lines.append(f"Open: {format_price(quote.open)}")
            if quote.high is not None:
                lines.append(f"High: {format_price(quote.high)}")
            if quote.low is not None:
                lines.append(f"Low: {format_price(quote.low)}")
            if quote.prev_close is not None:
                lines.append(f"Prev Close: {format_price(quote.prev_close)}")
            
            lines.append(f"Volume: {format_number(quote.volume)}")
            
            if quote.market_cap:
                lines.append(f"Market Cap: {format_number(quote.market_cap, '$')}")
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")


class FundamentalsCommand(BaseCommand):
    name = "info"
    aliases = ["i", "fundamentals", "fund"]
    description = "Get company fundamentals"
    usage = "!info AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            fund = await self.providers.get_fundamentals(symbol)
            
            lines = [
                f"📋 {fund.name} ({fund.symbol})",
                "",
            ]
            
            if fund.sector:
                lines.append(f"Sector: {fund.sector}")
            if fund.industry:
                lines.append(f"Industry: {fund.industry}")
            
            lines.append("")
            
            if fund.pe_ratio is not None:
                lines.append(f"P/E Ratio: {fund.pe_ratio:.2f}")
            else:
                lines.append("P/E Ratio: N/A")
            
            if fund.eps is not None:
                lines.append(f"EPS: ${fund.eps:.2f}")
            else:
                lines.append("EPS: N/A")
            
            if fund.market_cap:
                lines.append(f"Market Cap: {format_number(fund.market_cap, '$')}")
            
            if fund.dividend_yield is not None:
                lines.append(f"Dividend Yield: {fund.dividend_yield*100:.2f}%")
            
            if fund.fifty_two_week_high or fund.fifty_two_week_low:
                lines.append("")
                if fund.fifty_two_week_high:
                    lines.append(f"52W High: {format_price(fund.fifty_two_week_high)}")
                if fund.fifty_two_week_low:
                    lines.append(f"52W Low: {format_price(fund.fifty_two_week_low)}")
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")


class MarketCommand(BaseCommand):
    name = "market"
    aliases = ["m", "indices"]
    description = "Get major market indices"
    usage = "!market"
    
    INDICES = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^RUT": "Russell 2000",
        "^VIX": "VIX",
    }
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            quotes = await self.providers.get_quotes(list(self.INDICES.keys()))
            
            lines = ["📈 Market Overview", ""]
            
            for symbol, name in self.INDICES.items():
                if symbol in quotes:
                    q = quotes[symbol]
                    emoji = "🟢" if q.change >= 0 else "🔴"
                    sign = "+" if q.change >= 0 else ""
                    
                    # VIX is typically displayed differently
                    if symbol == "^VIX":
                        lines.append(f"😰 {name}: {q.price:.2f} ({sign}{q.change_percent:.2f}%)")
                    else:
                        lines.append(f"{emoji} {name}: {q.price:,.2f} ({sign}{q.change_percent:.2f}%)")
                else:
                    lines.append(f"⚪ {name}: N/A")
            
            return CommandResult.ok("\n".join(lines))
            
        except ProviderError as e:
            return CommandResult.error(f"Market data unavailable: {e}")


class HelpCommand(BaseCommand):
    name = "help"
    aliases = ["h", "?", "commands"]
    description = "Show available commands"
    usage = "!help [command]"
    
    def __init__(self, commands: list[BaseCommand]):
        self._commands = {cmd.name: cmd for cmd in commands}
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[0].lower()
            
            for cmd in self._commands.values():
                if cmd.matches(cmd_name):
                    aliases = f"\nAliases: {', '.join(cmd.aliases)}" if cmd.aliases else ""
                    return CommandResult.ok(
                        f"📖 !{cmd.name}{aliases}\n\n"
                        f"{cmd.description}\n\n"
                        f"Usage: {cmd.usage}"
                    )
            
            return CommandResult.error(f"Unknown command: {cmd_name}")
        
        # General help
        lines = ["📖 Stock Bot Commands", ""]
        
        for cmd in self._commands.values():
            if cmd.name != "help":
                lines.append(f"!{cmd.name} - {cmd.description}")
        
        lines.append(f"!help - {self.description}")
        lines.append("")
        lines.append("💡 Tip: Type $AAPL in any message for quick lookup")
        lines.append("Type !help <command> for detailed usage")
        
        return CommandResult.ok("\n".join(lines))


class StatusCommand(BaseCommand):
    name = "status"
    aliases = ["providers", "health"]
    description = "Show provider status"
    usage = "!status"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        status = self.providers.get_status()
        health = await self.providers.health_check()
        
        lines = ["🔧 Provider Status", ""]
        
        for name, info in status.items():
            healthy = health.get(name, False)
            health_emoji = "✅" if healthy else "❌"
            
            if info["rate_limited"]:
                remaining = info["rate_limit_remaining_seconds"]
                lines.append(f"{health_emoji} {name}: ⏳ Rate limited ({remaining}s)")
            else:
                lines.append(f"{health_emoji} {name}: Ready")
        
        return CommandResult.ok("\n".join(lines))


class CryptoCommand(BaseCommand):
    """Command for quick crypto overview"""
    name = "crypto"
    aliases = ["c", "coins"]
    description = "Get top cryptocurrency prices"
    usage = "!crypto"
    
    # Top cryptos by market cap
    CRYPTO_SYMBOLS = {
        "BTC-USD": "Bitcoin",
        "ETH-USD": "Ethereum",
        "SOL-USD": "Solana",
        "XRP-USD": "XRP",
        "DOGE-USD": "Dogecoin",
    }
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            quotes = await self.providers.get_quotes(list(self.CRYPTO_SYMBOLS.keys()))
            
            lines = ["🪙 Crypto Overview", ""]
            
            for symbol, name in self.CRYPTO_SYMBOLS.items():
                if symbol in quotes:
                    q = quotes[symbol]
                    emoji = "🟢" if q.change >= 0 else "🔴"
                    sign = "+" if q.change >= 0 else ""
                    lines.append(
                        f"{emoji} {name}: {format_price(q.price)} "
                        f"({sign}{q.change_percent:.2f}%)"
                    )
                else:
                    lines.append(f"⚪ {name}: N/A")
            
            return CommandResult.ok("\n".join(lines))
            
        except ProviderError as e:
            return CommandResult.error(f"Crypto data unavailable: {e}")

