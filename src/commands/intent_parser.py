"""
Natural language intent parser for the stock bot.

Parses natural language queries and maps them to bot commands.
Examples:
- "chart apple" → !chart AAPL
- "what's the rsi for tesla" → !rsi TSLA
- "any news on microsoft?" → !news MSFT
- "show me google earnings" → !earnings GOOGL
"""

import re
import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Intent:
    """Parsed intent from natural language."""
    command: str
    symbols: List[str]
    args: List[str]
    confidence: float  # 0.0 to 1.0


# Command patterns: (regex pattern, command name, requires_symbol)
INTENT_PATTERNS = [
    # Chart/graph requests
    (r'\b(chart|graph|plot|show\s+chart)\b.*', 'chart', True),
    
    # Price queries
    (r'\b(price|how\s+much|what.?s|current|quote)\b.*', 'price', True),
    (r'\b(stock|share)\s+price\b.*', 'price', True),
    
    # Technical analysis
    (r'\brsi\b', 'rsi', True),
    (r'\bmacd\b', 'macd', True),
    (r'\b(sma|moving\s+average)\b', 'sma', True),
    (r'\b(support|resistance|levels)\b', 'sr', True),
    (r'\b(ta|technical\s+analysis|technicals)\b', 'ta', True),
    (r'\btldr\b', 'tldr', True),
    
    # News
    (r'\b(news|headlines|article)\b.*', 'news', True),
    
    # Earnings/dividends
    (r'\b(earnings|quarter|eps)\b.*', 'earnings', True),
    (r'\b(dividend|yield|payout)\b.*', 'dividend', True),
    
    # Fundamentals
    (r'\b(pe|p/e|valuation|fundamentals|info|about)\b.*', 'info', True),
    
    # Analytics
    (r'\b(rating|analyst|upgrade|downgrade)\b.*', 'rating', True),
    (r'\b(insider|insiders|buying|selling)\b.*', 'insider', True),
    (r'\b(short\s+interest|shorts|squeeze)\b.*', 'short', True),
    (r'\b(correlation|correlate|compare)\b.*', 'corr', True),
    
    # Market overview
    (r'\b(market|markets|indices|index)\b(?!.*\w)', 'market', False),
    (r'\b(crypto|bitcoin|ethereum)\b(?!.*\w)', 'crypto', False),
    (r'\b(forex|currency|currencies|fx)\b(?!.*\w)', 'fx', False),
    (r'\b(futures|commodities|oil|gold)\b(?!.*\w)', 'futures', False),
    
    # Watchlist
    (r'\b(watchlist|watching|my\s+stocks)\b', 'watch', False),
    
    # Help
    (r'\b(help|commands|what\s+can\s+you)\b', 'help', False),
]

# Symbol extraction patterns
STOCK_SYMBOL_PATTERN = re.compile(r'\$([A-Z]{1,5})', re.IGNORECASE)
TICKER_IN_TEXT = re.compile(r'\b([A-Z]{1,5})\b')


def extract_symbols_from_text(text: str) -> List[str]:
    """
    Extract stock symbols from natural language text.
    
    Handles:
    - Explicit: $AAPL, $TSLA
    - Company names: Apple, Tesla, Microsoft
    - Tickers in context: "chart AAPL"
    """
    from ..utils.symbols import SYMBOL_ALIASES, resolve_alias
    
    symbols = []
    seen = set()
    text_lower = text.lower()
    
    # 1. Extract explicit $SYMBOL mentions
    for match in STOCK_SYMBOL_PATTERN.findall(text):
        sym = match.upper()
        if sym not in seen:
            symbols.append(sym)
            seen.add(sym)
    
    # 2. Check for company name aliases
    for alias, symbol in SYMBOL_ALIASES.items():
        # Match as whole word
        pattern = rf'\b{re.escape(alias)}\b'
        if re.search(pattern, text_lower):
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
    
    # 3. Look for uppercase tickers (1-5 chars) if nothing found
    if not symbols:
        # Find capitalized words that look like tickers
        words = text.split()
        for word in words:
            clean = re.sub(r'[^\w]', '', word)
            if clean.isupper() and 1 <= len(clean) <= 5 and clean.isalpha():
                if clean not in seen:
                    symbols.append(clean)
                    seen.add(clean)
    
    return symbols[:5]  # Limit to 5


def parse_intent(text: str) -> Optional[Intent]:
    """
    Parse a natural language query into a command intent.
    
    Returns None if no clear intent is detected.
    """
    text_lower = text.lower().strip()
    
    # Skip if it's already a command
    if text.startswith('!') or text.startswith('/'):
        return None
    
    # Try each intent pattern
    for pattern, command, requires_symbol in INTENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            symbols = extract_symbols_from_text(text)
            
            # If command requires a symbol but none found, skip
            if requires_symbol and not symbols:
                continue
            
            # Calculate confidence based on match quality
            confidence = 0.7  # Base confidence for pattern match
            if symbols:
                confidence += 0.2  # Boost if we found symbols
            
            logger.debug(f"Intent parsed: {command} {symbols} (confidence: {confidence})")
            
            return Intent(
                command=command,
                symbols=symbols,
                args=symbols,  # Symbols become args
                confidence=confidence,
            )
    
    # Check if text contains a potential symbol we should look up
    symbols = extract_symbols_from_text(text)
    if symbols:
        # Default to price lookup if we found symbols but no specific intent
        return Intent(
            command='price',
            symbols=symbols,
            args=symbols,
            confidence=0.5,  # Lower confidence for default
        )
    
    return None


def is_question_about_stocks(text: str) -> bool:
    """Check if text appears to be asking about stocks/finance."""
    keywords = [
        'stock', 'share', 'price', 'chart', 'earnings', 'dividend',
        'news', 'market', 'trade', 'trading', 'buy', 'sell', 'invest',
        'rsi', 'macd', 'technical', 'analysis', 'fundamentals',
        'pe', 'eps', 'revenue', 'profit', 'quarter',
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)
