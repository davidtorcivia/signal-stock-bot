"""
Earnings and dividend commands for stock bot.

Provides: !earnings, !dividend
Uses yfinance for data.
"""

from datetime import datetime
from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager


def format_date(dt) -> str:
    """Format datetime to readable string."""
    if dt is None:
        return "N/A"
    if hasattr(dt, 'strftime'):
        return dt.strftime("%b %d, %Y")
    return str(dt)


def format_number(value, prefix: str = "") -> str:
    """Format large numbers."""
    if value is None:
        return "N/A"
    if abs(value) >= 1e12:
        return f"{prefix}{value/1e12:.2f}T"
    elif abs(value) >= 1e9:
        return f"{prefix}{value/1e9:.2f}B"
    elif abs(value) >= 1e6:
        return f"{prefix}{value/1e6:.2f}M"
    else:
        return f"{prefix}{value:,.2f}"


class EarningsCommand(BaseCommand):
    """Earnings date and estimates."""
    name = "earnings"
    aliases = ["earn", "er"]
    description = "Earnings date and estimates"
    usage = "!earnings AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            import yfinance as yf
            import asyncio
            
            loop = asyncio.get_event_loop()
            
            def fetch_earnings():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                calendar = ticker.calendar
                return info, calendar
            
            info, calendar = await loop.run_in_executor(None, fetch_earnings)
            
            if not info:
                return CommandResult.error(f"No data for {symbol}")
            
            name = info.get('shortName', symbol)
            
            lines = [
                f"◈ {name} ({symbol}) Earnings",
                ""
            ]
            
            # Next earnings date
            if calendar is not None and not calendar.empty:
                if 'Earnings Date' in calendar.columns:
                    earnings_dates = calendar['Earnings Date']
                    if len(earnings_dates) > 0:
                        next_date = earnings_dates.iloc[0]
                        lines.append(f"Next Earnings: {format_date(next_date)}")
            
            # EPS estimates
            if 'trailingEps' in info and info['trailingEps']:
                lines.append(f"Trailing EPS: ${info['trailingEps']:.2f}")
            
            if 'forwardEps' in info and info['forwardEps']:
                lines.append(f"Forward EPS: ${info['forwardEps']:.2f}")
            
            # Revenue
            if 'totalRevenue' in info and info['totalRevenue']:
                lines.append(f"Revenue (TTM): {format_number(info['totalRevenue'], '$')}")
            
            # Profit margin
            if 'profitMargins' in info and info['profitMargins']:
                lines.append(f"Profit Margin: {info['profitMargins']*100:.1f}%")
            
            # P/E ratios
            if 'trailingPE' in info and info['trailingPE']:
                lines.append(f"P/E (TTM): {info['trailingPE']:.1f}")
            
            if 'forwardPE' in info and info['forwardPE']:
                lines.append(f"P/E (FWD): {info['forwardPE']:.1f}")
            
            if len(lines) <= 2:
                return CommandResult.error(f"No earnings data for {symbol}")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Earnings lookup failed: {type(e).__name__}")


class DividendCommand(BaseCommand):
    """Dividend information."""
    name = "dividend"
    aliases = ["div", "yield"]
    description = "Dividend information"
    usage = "!dividend AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            import yfinance as yf
            import asyncio
            
            loop = asyncio.get_event_loop()
            
            def fetch_dividend():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                dividends = ticker.dividends
                return info, dividends
            
            info, dividends = await loop.run_in_executor(None, fetch_dividend)
            
            if not info:
                return CommandResult.error(f"No data for {symbol}")
            
            name = info.get('shortName', symbol)
            
            # Check if company pays dividends
            div_yield = info.get('dividendYield')
            div_rate = info.get('dividendRate')
            
            if not div_yield and not div_rate:
                return CommandResult.ok(
                    f"◈ {name} ({symbol})\n\n"
                    f"This stock does not pay dividends."
                )
            
            lines = [
                f"◈ {name} ({symbol}) Dividend",
                ""
            ]
            
            if div_yield:
                lines.append(f"Yield: {div_yield*100:.2f}%")
            
            if div_rate:
                lines.append(f"Annual Rate: ${div_rate:.2f}")
            
            # Ex-dividend date
            ex_date = info.get('exDividendDate')
            if ex_date:
                ex_dt = datetime.fromtimestamp(ex_date)
                lines.append(f"Ex-Dividend Date: {format_date(ex_dt)}")
            
            # Payout ratio
            payout = info.get('payoutRatio')
            if payout:
                lines.append(f"Payout Ratio: {payout*100:.1f}%")
            
            # 5-year avg yield
            avg_yield = info.get('fiveYearAvgDividendYield')
            if avg_yield:
                lines.append(f"5Y Avg Yield: {avg_yield:.2f}%")
            
            # Recent dividend history
            if dividends is not None and len(dividends) > 0:
                recent = dividends.tail(4)
                lines.append("")
                lines.append("Recent Dividends:")
                for date, amount in recent.items():
                    lines.append(f"  {date.strftime('%Y-%m-%d')}: ${amount:.4f}")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Dividend lookup failed: {type(e).__name__}")
