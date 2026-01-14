"""
Stock-related command implementations.
"""

import re
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager, SymbolNotFoundError, ProviderError


# Eastern Time zone for US market hours
ET = ZoneInfo("America/New_York")


class Symbols:
    """
    Unicode symbols for polished output.
    
    Options for esoteric unicode instead of emojis:
    
    ARROWS:
        ▲ ▼  (triangles - clean, minimal)
        △ ▽  (outline triangles)
        ↑ ↓  (simple arrows)
        ⬆ ⬇  (bold arrows)
        ⇧ ⇩  (shift-style arrows)
        
    INDICATORS:
        ● ○  (filled/empty circles)
        ◆ ◇  (diamonds)
        ■ □  (squares)
        ★ ☆  (stars)
        
    CURRENCY:
        ◈ ◉  (fancy circles)
        ❖ ✦  (decorative)
        ⬢ ⬡  (hexagons)
        
    CURRENT SELECTION (esoteric unicode):
    """
    # Price/Value indicators
    PRICE = "◈"      # Diamond in circle - for price
    UP = "▲"         # Up triangle - for positive change
    DOWN = "▼"       # Down triangle - for negative change
    NEUTRAL = "◆"    # Diamond - for no change
    
    # Status indicators
    VOLUME = "⊡"     # Squared plus - for volume
    TIME = "◷"       # Clock face - for timestamp
    BULL = "●"       # Filled circle - bullish/positive
    BEAR = "○"       # Empty circle - bearish/negative
    
    # Multi-quote indicators
    GREEN = "▲"      # Up - for positive
    RED = "▼"        # Down - for negative  
    GRAY = "◇"       # Diamond outline - for not found
    
    # Separators
    DOT = "·"        # Middle dot separator
    DASH = "─"       # Horizontal line


def get_timestamp() -> str:
    """Get current timestamp in ET timezone with market-appropriate format."""
    now = datetime.now(ET)
    # Format: "3:45 PM ET" or "10:30 AM ET"
    # Use %I (12-hour with leading zero) and manually strip the leading zero
    # This is cross-platform (%-I is Unix-only, %#I is Windows-only)
    hour = now.strftime("%I").lstrip("0") or "12"  # Handle midnight edge case
    return f"{hour}:{now.strftime('%M %p')} ET"


def format_timestamp() -> str:
    """Get formatted timestamp line for responses."""
    return f"{Symbols.TIME} as of {get_timestamp()}"


# Valid symbol pattern: alphanumeric, dots, hyphens, carets (for indices), = (for futures/currencies)
# Examples: AAPL, BRK.B, BTC-USD, ^GSPC, GC=F, EURUSD=X
SYMBOL_PATTERN = re.compile(r'^[\^]?[A-Z0-9]{1,10}(?:[.\-=][A-Z0-9]{1,5})?$', re.IGNORECASE)


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
    """Format price change with unicode indicator"""
    arrow = Symbols.UP if change >= 0 else Symbols.DOWN
    sign = "+" if change >= 0 else ""
    return f"{arrow} {sign}{change:.2f} ({sign}{pct:.2f}% 1d)"


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
    description = "Get real-time stock price"
    usage = "!price AAPL [MSFT GOOGL]"
    help_explanation = """Shows the current trading price and daily change for one or more stocks.

**What You See:**
• Current Price: The last traded price.
• Change: How much the price moved today ($ and %).
• Volume: Number of shares traded. High volume = high interest.

**Pro Tips:**
• Green (▲) means the stock is up today.
• Red (▼) means the stock is down today.
• Use smart names: "!price apple" or "!price gold" works!"""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                f"Usage: {self.usage}\n"
                f"› Tip: You can also just type $AAPL in any message"
            )
        
        # Import symbol resolver
        from ..utils import resolve_symbol
        
        # Resolve and validate symbols
        symbols = []
        resolved_names = {}  # Track what was resolved
        for s in ctx.args[:10]:  # Limit to 10
            resolved, name = await resolve_symbol(s)
            valid, result = validate_symbol(resolved)
            if valid:
                symbols.append(result)
                if name:
                    resolved_names[result] = name
        
        if not symbols:
            return CommandResult.error(
                f"No valid symbols provided.\n"
                f"Valid formats: AAPL, apple, btc, BTC-USD, ^GSPC"
            )
        
        try:
            if len(symbols) == 1:
                quote = await self.providers.get_quote(symbols[0])
                name_display = f"{quote.name} ({quote.symbol})" if quote.name else quote.symbol
                
                return CommandResult.ok(
                    f"{name_display}\n"
                    f"{Symbols.PRICE} {format_price(quote.price)}\n"
                    f"{format_change(quote.change, quote.change_percent)}\n"
                    f"{Symbols.VOLUME} Vol: {format_number(quote.volume)}\n"
                    f"{format_timestamp()}"
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
                        indicator = Symbols.GREEN if q.change >= 0 else Symbols.RED
                        lines.append(
                            f"{indicator} {q.symbol}: {format_price(q.price)} "
                            f"({sign}{q.change_percent:.2f}% 1d)"
                        )
                    else:
                        lines.append(f"{Symbols.GRAY} {symbol}: Not found")
                
                lines.append(f"\n{format_timestamp()}")
                return CommandResult.ok("\n".join(lines))
                
        except SymbolNotFoundError:
            hint = ""
            sym = symbols[0]
            # Suggest crypto format if it looks like crypto
            if sym in ("BTC", "ETH", "SOL", "DOGE", "XRP"):
                hint = f"\n› Try {sym}-USD for crypto"
            return CommandResult.error(f"Symbol not found: {sym}{hint}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")


class QuoteCommand(BaseCommand):
    name = "quote"
    aliases = ["q", "detail"]
    description = "Get detailed stock quote"
    usage = "!quote AAPL"
    help_explanation = """A detailed market snapshot with more context than !price.

**What You See:**
• Open: Price at market open (9:30 AM ET).
• High/Low: Today's trading range.
• Previous Close: Yesterday's closing price.
• Volume: Shares traded today.
• Market Cap: Total company value (shares × price).

**What It Tells You:**
• If High/Low range is wide, it's a volatile day.
• If Volume is much higher than average, something is happening.
• Market Cap helps compare companies by size."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        # Resolve symbol (e.g., "apple" → "AAPL")
        from ..utils import resolve_symbol
        symbol, resolved_name = await resolve_symbol(ctx.args[0])
        
        try:
            quote = await self.providers.get_quote(symbol)
            name_display = quote.name or resolved_name or symbol
            
            lines = [
                f"⊞ {name_display} ({quote.symbol})",
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
    help_explanation = """Fundamental data about the company's financial health.

**Key Metrics:**
• P/E Ratio: Price ÷ Earnings. Lower = cheaper relative to profits.
  - Under 15: Possibly undervalued.
  - Over 30: Expensive, or high growth expected.
• EPS: Earnings Per Share. How much profit per share.
• Dividend Yield: Annual dividend ÷ stock price. Income investors love this.
• 52-Week High/Low: Shows how the stock has traded over the past year.

**What to Watch For:**
• A stock near its 52-week low may be oversold (or in trouble).
• A stock near its 52-week high is showing strength (or may be overbought)."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            fund = await self.providers.get_fundamentals(symbol)
            
            lines = [
                f"⊟ {fund.name} ({fund.symbol})",
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
    help_explanation = "Shows an overview of major US indices (S&P 500, Dow, Nasdaq) and the VIX (Volatility Index)."
    
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
            
            lines = ["⊞ Market Overview", ""]
            
            for symbol, name in self.INDICES.items():
                if symbol in quotes:
                    q = quotes[symbol]
                    indicator = "●" if q.change >= 0 else "○"
                    sign = "+" if q.change >= 0 else ""
                    
                    # VIX is typically displayed differently
                    if symbol == "^VIX":
                        lines.append(f"⚡ {name}: {q.price:.2f} ({sign}{q.change_percent:.2f}% 1d)")
                    else:
                        lines.append(f"{indicator} {name}: {q.price:,.2f} ({sign}{q.change_percent:.2f}% 1d)")
                else:
                    lines.append(f"◇ {name}: N/A")
            
            return CommandResult.ok("\n".join(lines))
            
        except ProviderError as e:
            return CommandResult.error(f"Market data unavailable: {e}")


class HelpCommand(BaseCommand):
    name = "help"
    aliases = ["h", "?", "commands"]
    description = "Show available commands"
    usage = "!help [command]"
    
    def __init__(self, commands: list[BaseCommand], bot_name: str = "Stock Bot"):
        self._commands = {cmd.name: cmd for cmd in commands}
        self.bot_name = bot_name
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[0].lower()
            
            for cmd in self._commands.values():
                if cmd.matches(cmd_name):
                    aliases = f"\nAliases: {', '.join(cmd.aliases)}" if cmd.aliases else ""
                    return CommandResult.ok(
                        f"⌘ !{cmd.name}{aliases}\n\n"
                        f"{cmd.description}\n\n"
                        f"Usage: {cmd.usage}"
                    )
            
            return CommandResult.error(f"Unknown command: {cmd_name}")
        
        # General help
        lines = [f"⌘ {self.bot_name} Commands", ""]
        
        for cmd in self._commands.values():
            if cmd.name != "help":
                lines.append(f"!{cmd.name} - {cmd.description}")
        
        lines.append(f"!help - {self.description}")
        lines.append("")
        lines.append("› Tip: Type $AAPL in any message for quick lookup")
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
        
        lines = ["⚙ Provider Status", ""]
        
        for name, info in status.items():
            healthy = health.get(name, False)
            health_indicator = "◆" if healthy else "◇"
            
            if info["rate_limited"]:
                remaining = info["rate_limit_remaining_seconds"]
                lines.append(f"{health_indicator} {name}: ↻ Rate limited ({remaining}s)")
            else:
                lines.append(f"{health_indicator} {name}: Ready")
        
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
            
            lines = ["◎ Crypto Overview", ""]
            
            for symbol, name in self.CRYPTO_SYMBOLS.items():
                if symbol in quotes:
                    q = quotes[symbol]
                    indicator = "●" if q.change >= 0 else "○"
                    sign = "+" if q.change >= 0 else ""
                    lines.append(
                        f"{indicator} {name}: {format_price(q.price)} "
                        f"({sign}{q.change_percent:.2f}% 1d)"
                    )
                else:
                    lines.append(f"◇ {name}: N/A")
            
            return CommandResult.ok("\n".join(lines))
            
        except ProviderError as e:
            return CommandResult.error(f"Crypto data unavailable: {e}")


class OptionCommand(BaseCommand):
    """Command for options quotes"""
    name = "option"
    aliases = ["opt", "o"]
    description = "Get option quote"
    usage = "!opt TSLA230120C00150000"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(
                f"Usage: {self.usage}\n"
                "Provide the full OCC symbol (e.g. AAPL230616C00150000)"
            )
        
        symbol = ctx.args[0].upper()
        # Ensure O: prefix for Massive if using standard strings or pass as is? 
        # Manually prepending O: if not present might be safer for MassiveProvider logic.
        # But MassiveProvider logic tries to handle it.
        # Let's pass as is.
        
        try:
            q = await self.providers.get_option_quote(symbol)
            
            lines = [
                f"⊡ {q.symbol}",
                f"{q.type.capitalize()} on {q.underlying}",
                f"Strike: ${q.strike:.2f} | Exp: {q.expiration.strftime('%Y-%m-%d')}",
                "",
                f"Price: {format_price(q.price)}",
                f"{format_change(q.change, q.change_percent)}",
                f"Vol: {q.volume} | OI: {q.open_interest}",
                "",
            ]
            
            if q.greeks:
                g = q.greeks
                lines.append(f"Delta: {g.get('delta', 'N/A')}")
                # Add other greeks if desired
            
            if q.implied_volatility:
                lines.append(f"IV: {q.implied_volatility:.2f}")

            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
             return CommandResult.error(f"Option not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Option data unavailable: {e}")


class FuturesCommand(BaseCommand):
    """Command for futures quotes"""
    name = "future"
    aliases = ["fut", "f"]
    description = "Get futures quote"
    usage = "!future ES"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            q = await self.providers.get_future_quote(symbol)
            
            indicator = "▼"
            sign = ""
            if q.change >= 0:
                indicator = "▲"
                sign = "+"
            
            lines = [
                f"⊠ {q.symbol}",
                f"{indicator} {format_price(q.price)}",
                f"{sign}{q.change:.2f} ({sign}{q.change_percent:.2f}% 1d)",
                f"Vol: {format_number(q.volume)}"
            ]
            return CommandResult.ok("\n".join(lines))
        except SymbolNotFoundError:
             return CommandResult.error(f"Future not found: {symbol}")
        except ProviderError as e:
             return CommandResult.error(f"Futures data unavailable: {e}")


class ForexCommand(BaseCommand):
    """Command for forex rates"""
    name = "forex"
    aliases = ["fx", "curr"]
    description = "Get forex rate"
    usage = "!fx EUR/USD"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            q = await self.providers.get_forex_quote(symbol)
            
            lines = [
                f"⇆ {q.symbol}",
                f"{format_price(q.rate)}",
                f"{format_change(q.change, q.change_percent)}"
            ]
            return CommandResult.ok("\n".join(lines))
        except SymbolNotFoundError:
             return CommandResult.error(f"Pair not found: {symbol}")
        except ProviderError as e:
             return CommandResult.error(f"Forex data unavailable: {e}")


class EconomyCommand(BaseCommand):
    """Command for economic indicators"""
    name = "economy"
    aliases = ["eco", "macro"]
    description = "Get economic data"
    usage = "!eco CPI"
    
    def __init__(self, provider_manager: ProviderManager, bot_name: str = "Stock Bot"):
        self.providers = provider_manager
        self.bot_name = bot_name

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        # Parse arguments
        args = [a.upper() for a in ctx.args]
        indicator = args[0]
        
        # Check for chart request
        # valid periods: 1y, 2y, 5y, 10y, max
        valid_periods = {"1Y", "2Y", "5Y", "10Y", "MAX"}
        period = "5Y"  # default
        show_chart = False
        
        if len(args) > 1:
            if "CHART" in args:
                show_chart = True
                args.remove("CHART")
            
            # Check for period in remaining args
            for arg in args[1:]:
                if arg in valid_periods:
                    period = arg
                    show_chart = True
                    break
        
        try:
            if show_chart:
                # Get historical data
                points, name, unit = await self.providers.get_economy_historical(indicator, period)
                
                # Convert to HistoricalBar objects for ChartGenerator
                from ..providers.base import HistoricalBar
                from ..charts import ChartGenerator, ChartOptions
                
                bars = []
                for date, value in points:
                    bars.append(HistoricalBar(
                        timestamp=date,
                        open=value,
                        high=value,
                        low=value,
                        close=value,
                        volume=0
                    ))
                
                # Determine formatting based on unit/indicator
                # Default
                value_format = "${:,.2f}"
                y_label = "Value"
                
                # Percent cases
                if "%" in unit or "RATE" in name.upper() or "PERCENT" in name.upper():
                    value_format = "{:.2f}%"
                    y_label = "Percent"
                
                # Index cases (CPI, Sentiment)
                elif "INDEX" in name.upper() or "CPI" in name.upper() or "(IND" in unit.upper():
                     value_format = "{:,.1f}"
                     y_label = "Index Value"
                
                # Trillions/Billions (Debt is M USD, GDP is B USD)
                elif "M USD" in unit.upper():
                    # Check if value is large enough for Trillions (1,000,000 M = 1 T)
                    if points[-1][1] > 900000:
                         # Divide by 1,000,000 for display
                         # We need to hack this since ChartGenerator uses format() on raw value
                         # But ChartGenerator formatting string logic is limited.
                         # BETTER: Let ChartGenerator handle raw value, we just update y_label.
                         # Actually, ChartGenerator format string "{:.2f}T" expects raw value to be scaled already?
                         # No, python format string can't scale.
                         # We should scale the DATA if we change the unit label.
                         pass 
                         
                    # For now, just handle formatting assuming raw data
                    # If data is in Millions, 35,000,000 is 35T.
                    # We can't scale data easily inside this logic block without modifying 'bars'.
                    
                    # SCALING LOGIC:
                    scaling_factor = 1.0
                    wrapper_format = "${:,.2f}"
                    
                    if points[-1][1] > 900000: # > 900B (since input is M) -> Trillions
                        scaling_factor = 1_000_000
                        value_format = "${:,.2f} T"
                        y_label = "Trillions of Dollars"
                    else: # Billions
                        scaling_factor = 1_000
                        value_format = "${:,.2f} B"
                        y_label = "Billions of Dollars"
                        
                    # Apply scaling to bars
                    for bar in bars:
                        bar.open /= scaling_factor
                        bar.high /= scaling_factor
                        bar.low /= scaling_factor
                        bar.close /= scaling_factor
                        
                elif "BILLION" in unit.upper() or "B USD" in unit.upper():
                    value_format = "${:,.0f} B"
                    y_label = "Billions of Dollars"
                    
                # Thousands (Jobs)
                elif "K " in unit.upper():
                    value_format = "{:,.0f}K"
                    y_label = "Thousands"
                
                # Generate chart
                options = ChartOptions(
                    chart_type="line",
                    show_volume=False,
                    value_format=value_format,
                    y_label=y_label,
                    fill_area=True,
                    line_color="#FF9800", # Bloomberg Terminal Amber
                )
                
                generator = ChartGenerator(bot_name=self.bot_name)
                
                # Use name provided by FredProvider (which is usually descriptive)
                chart_b64 = generator.generate(
                    symbol=name.split(":")[0], # Truncate if too long? No, use full name if reasonable
                    bars=bars,
                    period=period,
                    current_price=points[-1][1],
                    change_percent=None, 
                    options=options,
                )
                
                return CommandResult(
                    text=f"⌂ {name} ({points[-1][0].strftime('%Y-%m-%d')})\nValue: {points[-1][1]}{unit}",
                    success=True,
                    attachments=[chart_b64]
                )
            
            else:
                # Standard data request
                data = await self.providers.get_economy_data(indicator)
                
                lines = [
                    f"⌂ {data.name}",
                    f"Value: {data.value}{data.unit}",
                    f"Date: {data.date.strftime('%Y-%m-%d')}"
                ]
                if data.previous:
                     lines.append(f"Prev: {data.previous}{data.unit}")
                
                return CommandResult.ok("\n".join(lines))
                
        except ProviderError as e:
             return CommandResult.error(f"Economy data unavailable: {e}")


class ProRequiredCommand(BaseCommand):
    """
    Stub command for features requiring Polygon Pro plan.
    Returns a helpful message directing users to upgrade.
    """
    
    def __init__(self, name: str, aliases: list[str], description: str, usage: str):
        self._name = name
        self._aliases = aliases
        self._description = description
        self._usage = usage
    
    @property
    def name(self) -> str:
        return self._name
    
    @property
    def aliases(self) -> list[str]:
        return self._aliases
    
    @property
    def description(self) -> str:
        return self._description
    
    @property
    def usage(self) -> str:
        return self._usage
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        return CommandResult.error(
            f"The !{self._name} command requires a Polygon.io paid plan.\n\n"
            f"› Upgrade at polygon.io/pricing\n"
            f"› Set MASSIVE_PRO=true in your .env after upgrading"
        )


class ChartCommand(BaseCommand):
    """
    Command for generating stock price charts.
    
    Supports flags:
      -c             Candlestick chart
      -sma20/50/200  Add SMA overlay
      -bb            Add Bollinger Bands
      -rsi           Add RSI panel
      -compare MSFT  Overlay comparison symbol
    """
    name = "chart"
    aliases = ["ch", "graph"]
    description = "Generate stock price chart"
    usage = "!chart AAPL [1m] [-c] [-sma20] [-bb] [-rsi] [-compare MSFT]"
    help_explanation = """Generates a visual price chart with optional technical indicators.

**Periods:** 1d (intraday), 5d, 1m, 3m, 6m, 1y, 5y, max.

**Options:**
• -c: Candlestick chart. Shows open/high/low/close per bar.
• -sma20/50/200: Overlay moving averages to see trends.
• -bb: Bollinger Bands. Shows volatility and potential reversals.
• -rsi: Add RSI indicator below the chart.
• -compare MSFT: Overlay another stock to compare performance.

**Reading Candlesticks:**
• Green candle: Closed higher than open (bullish).
• Red candle: Closed lower than open (bearish).
• Long wicks: Rejection of prices (reversal signs).

**Pro Tip:** Use "!chart AAPL 1y -sma50 -sma200" to see Golden/Death Cross signals."""

    # Extended help for specific flags
    help_flags = """◈ CHART OPTIONS EXPLAINED

━━━ -c (CANDLESTICK) ━━━
Each bar shows:
• Body: Open to Close price.
• Wicks: High and Low of the period.
• Green = Bullish (close > open). Red = Bearish (close < open).
• Patterns to watch: Doji (indecision), Hammer (reversal), Engulfing (momentum).

━━━ -sma (MOVING AVERAGES) ━━━
• SMA20: Short-term trend (1 month).
• SMA50: Medium-term. Institutions watch this.
• SMA200: Long-term. THE most important level.
• GOLDEN CROSS: SMA50 crosses ABOVE SMA200. Major BUY signal.
• DEATH CROSS: SMA50 crosses BELOW SMA200. Major SELL signal.

━━━ -bb (BOLLINGER BANDS) ━━━
• Upper/Lower bands show 2 standard deviations from 20-SMA.
• Price near UPPER band = potentially overbought.
• Price near LOWER band = potentially oversold.
• Band SQUEEZE (narrow) = volatility coming. Breakout expected.

━━━ -rsi (RSI PANEL) ━━━
• RSI > 70: Overbought. May be expensive.
• RSI < 30: Oversold. May be cheap.
• RSI divergence from price = potential reversal.

━━━ -compare (COMPARISON) ━━━
• Overlays another symbol to compare performance.
• Shows relative strength between two stocks.
• Example: !chart AAPL 1y -compare SPY (AAPL vs market)."""
    
    def __init__(self, provider_manager: ProviderManager, bot_name: str = "Stock Bot"):
        self.providers = provider_manager
        self.bot_name = bot_name
        self._generator = None
    
    def _get_generator(self):
        """Lazy-load chart generator to avoid matplotlib import at startup."""
        if self._generator is None:
            from ..charts import ChartGenerator
            self._generator = ChartGenerator(
                theme="dark",
                width=800,
                height=500,
                bot_name=self.bot_name
            )
        return self._generator
    
    def _parse_args(self, args: list[str]) -> tuple[str, str, dict]:
        """
        Parse command arguments into symbol, period, and options.
        
        Returns:
            (symbol, period, options_dict)
        """
        symbol = None
        period = "1m"  # Default
        options = {
            "chart_type": "line",
            "sma_periods": [],
            "bollinger": False,
            "rsi": False,
            "compare": None,  # Comparison symbol
        }
        
        valid_periods = {"1d", "5d", "1w", "1m", "3m", "6m", "1y", "ytd", "5y", "max"}
        
        i = 0
        while i < len(args):
            arg = args[i]
            arg_lower = arg.lower()
            
            # Flags
            if arg_lower == "-c" or arg_lower == "--candle":
                options["chart_type"] = "candle"
            elif arg_lower.startswith("-sma"):
                # Parse SMA period: -sma20, -sma50, -sma200
                try:
                    period_num = int(arg_lower.replace("-sma", ""))
                    if period_num > 0:
                        options["sma_periods"].append(period_num)
                except ValueError:
                    pass
            elif arg_lower == "-bb" or arg_lower == "--bollinger":
                options["bollinger"] = True
            elif arg_lower == "-rsi":
                options["rsi"] = True
            elif arg_lower == "-compare" or arg_lower == "--compare":
                # Next arg is the comparison symbol
                if i + 1 < len(args) and not args[i + 1].startswith("-"):
                    options["compare"] = args[i + 1]  # Don't uppercase - resolver handles it
                    i += 1
            # Period
            elif arg_lower in valid_periods:
                period = arg_lower
            # Symbol (first non-flag, non-period argument)
            elif symbol is None and not arg.startswith("-"):
                symbol = arg  # Don't uppercase - resolver handles it
            
            i += 1
        
        return symbol, period, options
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Check for -help flag first
        if self.has_help_flag(ctx):
            # Check if any specific flags are present to give detailed help
            args_lower = [a.lower() for a in ctx.args]
            has_specific = any(f in args_lower for f in ["-c", "-sma", "-bb", "-rsi", "-compare"])
            if has_specific or any(a.startswith("-sma") for a in args_lower):
                return CommandResult.ok(self.help_flags)
            return self.get_help_result()
        
        if not ctx.args:
            return CommandResult.error(
                f"Usage: {self.usage}\n"
                f"Periods: 1d, 5d, 1m, 3m, 6m, 1y, ytd\n"
                f"Flags: -c (candle), -sma20/50/200, -bb, -rsi"
            )
        
        # Parse arguments
        symbol, period, opts = self._parse_args(ctx.args)
        
        if not symbol:
            return CommandResult.error("Symbol required. Example: !chart AAPL 1m -c")
        
        # Resolve symbol (e.g., "apple" → "AAPL")
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(symbol)
        
        # Also resolve comparison symbol if provided
        if opts["compare"]:
            opts["compare"], _ = await resolve_symbol(opts["compare"])
        
        # Validate symbol
        valid, result = validate_symbol(symbol)
        if not valid:
            return CommandResult.error(result)
        
        # Map period to provider parameters
        from ..charts.generator import get_period_params
        from ..charts import ChartOptions
        provider_period, interval = get_period_params(period)
        
        # Build ChartOptions
        chart_options = ChartOptions(
            chart_type=opts["chart_type"],
            sma_periods=opts["sma_periods"],
            bollinger=opts["bollinger"],
            rsi=opts["rsi"],
            show_volume=True,
            comparison_symbol=opts["compare"],
        )
        
        try:
            # Get historical data
            bars = await self.providers.get_historical(
                symbol=symbol,
                period=provider_period,
                interval=interval
            )
            
            if not bars:
                return CommandResult.error(f"No chart data available for {symbol}")
            
            # Fetch comparison data if requested
            if opts["compare"]:
                try:
                    comp_bars = await self.providers.get_historical(
                        symbol=opts["compare"],
                        period=provider_period,
                        interval=interval
                    )
                    chart_options.comparison_bars = comp_bars
                except Exception as e:
                    # Continue without comparison if fetch fails
                    chart_options.comparison_symbol = None
            
            # Get current quote for title
            try:
                quote = await self.providers.get_quote(symbol)
                current_price = quote.price
                name = quote.name or symbol
            except Exception:
                current_price = bars[-1].close
                name = symbol
            
            # Calculate period change from chart data
            period_change_pct = ((bars[-1].close - bars[0].open) / bars[0].open) * 100
            
            # Period label
            period_labels = {
                "1d": "1d", "5d": "5d", "1w": "1w", "1m": "1m",
                "3m": "3m", "6m": "6m", "1y": "1y", "ytd": "ytd",
                "5y": "5y", "max": "all"
            }
            period_label = period_labels.get(period, period)
            
            # Generate chart
            generator = self._get_generator()
            chart_base64 = generator.generate(
                symbol=symbol,
                bars=bars,
                period=period,
                current_price=current_price,
                change_percent=period_change_pct,
                options=chart_options,
            )
            
            # Build caption
            sign = "+" if period_change_pct >= 0 else ""
            indicator = "▲" if period_change_pct >= 0 else "▼"
            
            # Add indicator labels
            indicator_labels = []
            if chart_options.chart_type == "candle":
                indicator_labels.append("candle")
            if chart_options.sma_periods:
                indicator_labels.append(f"SMA{'/'.join(map(str, chart_options.sma_periods))}")
            if chart_options.bollinger:
                indicator_labels.append("BB")
            if chart_options.rsi:
                indicator_labels.append("RSI")
            if chart_options.comparison_symbol and chart_options.comparison_bars:
                indicator_labels.append(f"vs {chart_options.comparison_symbol}")
            
            indicator_str = f" [{', '.join(indicator_labels)}]" if indicator_labels else ""
            
            caption = (
                f"{name} ({symbol}){indicator_str}\n"
                f"{Symbols.PRICE} {format_price(current_price)} "
                f"{indicator} {sign}{period_change_pct:.2f}% {period_label}\n"
                f"{format_timestamp()}"
            )
            
            return CommandResult.with_chart(caption, chart_base64)
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Chart data unavailable: {e}")
        except ImportError as e:
            return CommandResult.error(
                f"Charts require dependencies. Install: pip install matplotlib mplfinance pandas\n"
                f"Error: {e}"
            )
        except Exception as e:
            return CommandResult.error(f"Chart generation failed: {type(e).__name__}")


