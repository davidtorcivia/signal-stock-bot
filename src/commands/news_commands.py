"""
News command for stock bot.

Provides: !news
Uses yfinance for headlines.
"""

from datetime import datetime
from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager


class NewsCommand(BaseCommand):
    """Recent news headlines for a symbol."""
    name = "news"
    aliases = ["headlines", "n"]
    description = "Recent news headlines"
    usage = "!news AAPL [count]"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        # Parse arguments
        symbol = None
        count = 5  # Default
        
        for arg in ctx.args:
            try:
                n = int(arg)
                if 1 <= n <= 20:
                    count = n
            except ValueError:
                if symbol is None:
                    symbol = arg
        
        if not symbol:
            # Market-wide news
            symbol = "^GSPC"  # S&P 500 as proxy
            is_market = True
        else:
            is_market = False
        
        from ..utils import resolve_symbol
        symbol, resolved_name = await resolve_symbol(symbol)
        
        try:
            import yfinance as yf
            import asyncio
            
            loop = asyncio.get_event_loop()
            
            def fetch_news():
                ticker = yf.Ticker(symbol)
                return ticker.news
            
            news = await loop.run_in_executor(None, fetch_news)
            
            if not news:
                if is_market:
                    return CommandResult.error("No market news available")
                return CommandResult.error(f"No news for {symbol}")
            
            # Limit to requested count
            news = news[:count]
            
            if is_market:
                lines = ["◈ Market News", ""]
            else:
                name = resolved_name or symbol
                lines = [f"◈ {name} ({symbol}) News", ""]
            
            for i, item in enumerate(news, 1):
                title = item.get('title', 'No title')
                publisher = item.get('publisher', '')
                
                # Format: 1. Title (Publisher)
                if publisher:
                    lines.append(f"{i}. {title}")
                    lines.append(f"   ↳ {publisher}")
                else:
                    lines.append(f"{i}. {title}")
                
                # Add link if available
                link = item.get('link', '')
                if link and len(link) < 60:
                    lines.append(f"   {link}")
                
                if i < len(news):
                    lines.append("")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"News lookup failed: {type(e).__name__}")
