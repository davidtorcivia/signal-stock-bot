"""
News command for stock bot.

Provides: !news
Uses yfinance for headlines.
"""

from datetime import datetime, timezone
from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager


# Simple sentiment keywords for headline analysis
POSITIVE_KEYWORDS = {
    'beats', 'surges', 'soars', 'rallies', 'jumps', 'gains', 'rises', 'climbs',
    'upgrade', 'upgraded', 'buy', 'bullish', 'profit', 'growth', 'record',
    'breakthrough', 'outperform', 'strong', 'positive', 'success', 'win', 'wins',
    'launch', 'launches', 'deal', 'partnership', 'innovation', 'expands'
}
NEGATIVE_KEYWORDS = {
    'falls', 'drops', 'plunges', 'tumbles', 'sinks', 'declines', 'slides',
    'downgrade', 'downgraded', 'sell', 'bearish', 'loss', 'losses', 'miss',
    'misses', 'warning', 'warns', 'cuts', 'layoffs', 'lawsuit', 'investigation',
    'recall', 'fraud', 'bankruptcy', 'crash', 'crashes', 'weak', 'negative', 'fails'
}


def get_sentiment(title: str) -> str:
    """Analyze headline for simple sentiment indicator."""
    words = set(title.lower().split())
    
    pos_count = len(words & POSITIVE_KEYWORDS)
    neg_count = len(words & NEGATIVE_KEYWORDS)
    
    if pos_count > neg_count:
        return "●"  # Positive (filled circle)
    elif neg_count > pos_count:
        return "○"  # Negative (empty circle)
    return ""  # Neutral (no indicator)


def format_relative_time(timestamp: int) -> str:
    """Format Unix timestamp as relative time (e.g., '2h ago')."""
    if not timestamp:
        return ""
    
    try:
        now = datetime.now(timezone.utc)
        then = datetime.fromtimestamp(timestamp, tz=timezone.utc)
        delta = now - then
        
        seconds = delta.total_seconds()
        if seconds < 0:
            return ""
        
        if seconds < 60:
            return "just now"
        elif seconds < 3600:
            mins = int(seconds / 60)
            return f"{mins}m ago"
        elif seconds < 86400:
            hours = int(seconds / 3600)
            return f"{hours}h ago"
        elif seconds < 172800:
            return "Yesterday"
        elif seconds < 604800:
            days = int(seconds / 86400)
            return f"{days}d ago"
        else:
            return then.strftime("%b %d")
    except Exception:
        return ""


class NewsCommand(BaseCommand):
    """Recent news headlines for a symbol."""
    name = "news"
    aliases = ["headlines", "n"]
    description = "Recent news headlines"
    usage = "!news AAPL [count] [-sentiment]"
    help_explanation = """Fetches recent news headlines for a stock.

**What You See:**
• Headline: The title of the news article.
• Publisher: The source (Reuters, Yahoo Finance, etc.).
• Time: How long ago it was published.

**Flags:**
• -sentiment: Add positive/negative indicators (● bullish, ○ bearish)

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
        show_sentiment = False
        
        for arg in ctx.args:
            lower_arg = arg.lower()
            if lower_arg in ('-sentiment', '--sentiment', '-s'):
                show_sentiment = True
                continue
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
                
                # Get timestamp
                pub_time = content.get('pubDate') or item.get('providerPublishTime', 0)
                if isinstance(pub_time, str):
                    # Try parsing ISO format
                    try:
                        dt = datetime.fromisoformat(pub_time.replace('Z', '+00:00'))
                        pub_time = int(dt.timestamp())
                    except Exception:
                        pub_time = 0
                
                relative_time = format_relative_time(pub_time)
                
                # Get URL - prefer original article (clickThroughUrl) over Yahoo redirect
                click_through = content.get('clickThroughUrl', {})
                if isinstance(click_through, dict):
                    url = click_through.get('url', '')
                else:
                    url = click_through or ''
                
                # Fallback to canonical/link if no original URL
                if not url:
                    canonical_url = content.get('canonicalUrl', {})
                    if isinstance(canonical_url, dict):
                        url = canonical_url.get('url', '')
                    else:
                        url = canonical_url or ''
                if not url:
                    url = item.get('link', '')
                
                # Build headline line with optional sentiment
                sentiment = get_sentiment(title) if show_sentiment else ""
                if sentiment:
                    lines.append(f"{sentiment} {i}. {title}")
                else:
                    lines.append(f"{i}. {title}")
                
                # Publisher and time on same line if both available
                meta_parts = []
                if publisher:
                    meta_parts.append(publisher)
                if relative_time:
                    meta_parts.append(relative_time)
                if meta_parts:
                    lines.append(f"   ↳ {' · '.join(meta_parts)}")
                
                if url:
                    lines.append(f"   → {url}")
                
                if i < len(news):
                    lines.append("")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            import traceback
            return CommandResult.error(f"News lookup failed: {type(e).__name__}: {str(e)[:50]}")
