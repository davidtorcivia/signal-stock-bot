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
    help_explanation = """Fetches recent news headlines for a stock.

**What You See:**
• Headline: The title of the news article.
• Publisher: The source (Reuters, Yahoo Finance, etc.).

**How News Affects Stocks:**
• Positive news (earnings beat, new product) usually pushes price UP.
• Negative news (lawsuit, missed earnings) usually pushes price DOWN.
• "Buy the rumor, sell the news" — prices often move BEFORE news is official.

**Pro Tip:** If a stock drops on good news, institutions may be selling. Be careful."""
    
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
            symbol = "SPY"  # Use SPY instead of ^GSPC for better news
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
                # Handle yfinance's nested content structure (changed in recent versions)
                # New structure: item['content']['title'], item['content']['provider']
                content = item.get('content', item)  # Fallback to item itself for old format
                
                # Get title from various possible locations
                title = (
                    content.get('title') or 
                    content.get('headline') or 
                    item.get('title') or  # Old format fallback
                    content.get('summary', '')[:100] or 
                    'Untitled'
                )
                
                # Clean title
                if title and len(title) > 100:
                    title = title[:97] + "..."
                
                # Get publisher - new format uses nested provider object
                provider = content.get('provider', {})
                if isinstance(provider, dict):
                    publisher = provider.get('displayName') or provider.get('name', '')
                else:
                    publisher = item.get('publisher') or item.get('source', '')
                
                # Format: 1. Title
                lines.append(f"{i}. {title}")
                if publisher:
                    lines.append(f"   ↳ {publisher}")
                
                if i < len(news):
                    lines.append("")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            import traceback
            return CommandResult.error(f"News lookup failed: {type(e).__name__}: {str(e)[:50]}")

