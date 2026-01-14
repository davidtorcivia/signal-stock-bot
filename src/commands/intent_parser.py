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
    negated_terms: List[str] = None  # Terms explicitly negated by user


# Negation patterns - words that negate the following term
NEGATION_WORDS = {'not', "don't", "dont", 'without', 'no', 'except', 'excluding', 'skip', 'hide'}

# Ticker collision list - common English words that are also tickers
# These require explicit $SYMBOL notation or strong context
AMBIGUOUS_TICKERS = {'NOW', 'OPEN', 'GO', 'ON', 'IT', 'ALL', 'ARE', 'BE', 'CAN', 'FOR', 'HAS', 'NEW', 'ONE', 'OUT', 'SEE', 'TWO', 'WAY', 'BIG', 'KEY', 'CAT', 'DOG', 'FUN', 'RUN', 'SUN', 'DAY', 'ITD', 'TRULY', 'AWFUL', 'LATE', 'HARD', 'FAST', 'LOST', 'LOW', 'HIGH', 'MAN', 'BOY', 'ART', 'EAT', 'RED', 'SEE', 'SKY', 'TOP', 'WIN', 'YES', 'ZIP'}


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
    (r'\b(correlation|correlate)\b.*', 'corr', True),
    
    # Market overview
    (r'\b(market|markets|indices|index)\b(?!.*\w)', 'market', False),
    (r'\b(crypto|bitcoin|ethereum)\b(?!.*\w)', 'crypto', False),
    (r'\b(forex|currency|currencies|fx)\b(?!.*\w)', 'fx', False),
    (r'\b(futures|commodities|oil|gold)\b(?!.*\w)', 'futures', False),
    
    # Watchlist
    (r'\b(watchlist|watching|my\s+stocks)\b', 'watch', False),
    
    # Help
    (r'\b(help|commands|what\s+can\s+you)\b', 'help', False),
    
    # Economy
    (r'\b(economy|economic|macro|fed|indicator)\b', 'economy', False),
]

# Build economy keyword map (lowercase keyword -> indicator key)
# Detects: "cpi", "unemployment", "gdp", "inflation", etc.
from ..providers.fred import INDICATOR_MAPPING
ECONOMY_KEYWORDS = {}
for key in INDICATOR_MAPPING:
    ECONOMY_KEYWORDS[key.lower()] = key
    # Add common variations if needed, e.g. "unemployment rate" -> UNEMPLOYMENT
    if key == "UNEMPLOYMENT":
        ECONOMY_KEYWORDS["unemployment rate"] = key
    elif key == "FEDFUNDS":
        ECONOMY_KEYWORDS["fed funds"] = key
        ECONOMY_KEYWORDS["interest rates"] = key
    elif key == "INFLATION":
        ECONOMY_KEYWORDS["inflation rate"] = key


# Symbol extraction patterns
STOCK_SYMBOL_PATTERN = re.compile(r'\$([A-Z]{1,5})', re.IGNORECASE)
TICKER_IN_TEXT = re.compile(r'\b([A-Z]{1,5})\b')


def extract_symbols_from_text(text: str) -> List[str]:
    """
    Extract stock symbols from natural language text.
    
    Handles:
    - Explicit: $AAPL, $TSLA
    - Company names: Apple, Tesla, Microsoft
    - Fuzzy matches: Nvidea -> NVDA
    - Tickers in context: "chart AAPL"
    - Safe lowercase: "chart apple" (if not stopword)
    """
    from ..utils.symbols import SYMBOL_ALIASES
    from ..utils.stopwords import STOPWORDS
    try:
        from thefuzz import process, fuzz
        HAS_FUZZ = True
    except ImportError:
        HAS_FUZZ = False
        logger.warning("thefuzz not installed, fuzzy matching disabled")
    
    symbols = []
    seen = set()
    text_lower = text.lower()
    
    # 1. Extract explicit $SYMBOL mentions
    for match in STOCK_SYMBOL_PATTERN.findall(text):
        sym = match.upper()
        if sym not in seen:
            symbols.append(sym)
            seen.add(sym)
    
    # 2. Check for company name aliases (Exact & Fuzzy)
    # We want to match names in text. Simple iteration is O(N*M).
    # For fuzzy, we check words in text against aliases?
    # Or check if any alias is in text?
    
    # Fast path: Check exact aliases
    for alias, symbol in SYMBOL_ALIASES.items():
        # Match as whole word
        pattern = rf'\b{re.escape(alias)}\b'
        if re.search(pattern, text_lower):
            if symbol not in seen:
                symbols.append(symbol)
                seen.add(symbol)
    
    # Fuzzy path: If we found nothing yet, check words for typos of aliases
    # Only if we have no clear symbols
    if not symbols and HAS_FUZZ:
        words = text_lower.split()
        for word in words:
            if len(word) < 4: continue # Skip short words
            if word in STOPWORDS: continue
            
            # Check against aliases
            # process.extractOne returns (match, score)
            # aliases keys are the choices
            match, score = process.extractOne(word, SYMBOL_ALIASES.keys(), scorer=fuzz.ratio)
            if score >= 80: # Lowered threshold to catch nvidea -> nvidia (83%)
                symbol = SYMBOL_ALIASES[match]
                logger.debug(f"Fuzzy match: '{word}' -> '{match}' ({symbol}) score={score}")
                if symbol not in seen:
                    symbols.append(symbol)
                    seen.add(symbol)
    
    # 3. Look for tickers (Uppercase OR Safe Lowercase)
    # Always check this to catch mixed queries like "Apple and MSFT"
    words = text.split()
    for word in words:
        # Skip words with apostrophes (it'd, don't, etc.)
        if "'" in word or "’" in word:
            continue
            
        # Clean punctuation
        clean = re.sub(r'[^\w-]', '', word)
        clean_lower = clean.lower()
        
        # Skip empty or if it's a known alias (handled in Step 2)
        if not clean or clean_lower in SYMBOL_ALIASES:
            continue
        
        # Criteria for valid potential ticker:
        # - Length 2-5 chars
        # - Not a stopword
        # - Alpha only (or dot/dash)
        # - Not an ambiguous ticker (unless explicit $SYMBOL)
        if 2 <= len(clean) <= 5 and clean.replace('-','').replace('.','').isalpha():
            sym = clean.upper()
            # Skip ambiguous tickers unless they were explicit ($SYMBOL)
            if sym in AMBIGUOUS_TICKERS and not word.startswith('$'):
                continue
            if clean_lower not in STOPWORDS:
                if sym not in seen:
                    symbols.append(sym)
                    seen.add(sym)
    
    return symbols[:5]  # Limit to 5


# Time period pattern (e.g., 6m, 1y, 5d)
PERIOD_PATTERN = re.compile(r'\b(\d+)([dDwWmMyY])\b')

# Chart params patterns
CHART_PARAMS = [
    (re.compile(r'\b(candle|candlestick|candlesticks|candles)\b', re.IGNORECASE), "-c"),
    (re.compile(r'\b(rsi)\b', re.IGNORECASE), "-rsi"),
    (re.compile(r'\b(bollinger|bands|bb)\b', re.IGNORECASE), "-bb"),
    (re.compile(r'\b(sma|moving average)\s*(\d+)?\b', re.IGNORECASE), "-sma"),
]

def parse_intent(text: str) -> Optional[Intent]:
    """
    Parse a natural language query into a command intent.
    
    Returns None if no clear intent is detected.
    """
    text_lower = text.lower().strip()
    
    # Skip if it's already a command
    if text.startswith('!') or text.startswith('/'):
        return None
    
    # Extract symbols first
    symbols = extract_symbols_from_text(text)
    
    # Extract extra args (periods, chart options)
    args = list(symbols) # Start with symbols as args
    
    # If no symbols, check for pronouns to facilitate context resolution
    if not symbols:
        words = text_lower.split()
        for p in ('it', 'that', 'this', 'its'):
            if p in words:
                args.append(p)
                break
    
    # 1. Period parsing
    period_match = PERIOD_PATTERN.search(text)
    if period_match:
        # Normalize period (e.g. 6m)
        period = f"{period_match.group(1)}{period_match.group(2).lower()}"
        args.append(period)
    
    # 1b. Date range parsing (e.g., "from January to March", "since 2023")
    date_range_patterns = [
        (r'\bfrom\s+(\w+)\s+to\s+(\w+)\b', lambda m: f"--from={m.group(1)} --to={m.group(2)}"),
        (r'\bsince\s+(\d{4}|\w+)\b', lambda m: f"--since={m.group(1)}"),
        (r'\blast\s+(\d+)\s+(day|week|month|year)s?\b', lambda m: f"{m.group(1)}{m.group(2)[0]}"),
    ]
    for pattern, extractor in date_range_patterns:
        match = re.search(pattern, text_lower)
        if match:
            result = extractor(match)
            if result.startswith('--'):
                args.append(result)
            else:
                args.append(result)
            break
    
    # 2. Negation detection - find terms that are explicitly negated
    negated_terms = []
    words = text_lower.split()
    for i, word in enumerate(words):
        if word in NEGATION_WORDS and i + 1 < len(words):
            next_word = words[i + 1]
            # The negated term could be a symbol or a flag
            if next_word in ('rsi', 'macd', 'sma', 'bollinger', 'bb'):
                negated_terms.append(f"-{next_word}")
            elif next_word.upper() in [s.upper() for s in symbols]:
                negated_terms.append(next_word.upper())
        
    # 3. Chart options parsing (skip negated flags)
    # Special handling for SMA - find ALL occurrences
    sma_pattern = re.compile(r'\bsma\s*(\d+)\b', re.IGNORECASE)
    sma_matches = sma_pattern.findall(text_lower)
    for sma_val in sma_matches:
        flag = f"-sma{sma_val}"
        if flag not in negated_terms and 'sma' not in [t.lstrip('-') for t in negated_terms]:
            if flag not in args:
                args.append(flag)
    
    # Handle other chart params (excluding SMA since we handled it above)
    for pattern, flag in CHART_PARAMS:
        if flag == "-sma":
            continue  # Already handled above
        match = pattern.search(text_lower)
        if match:
            # Check if this flag is negated
            if flag in negated_terms or flag.lstrip('-') in [t.lstrip('-') for t in negated_terms]:
                continue
            if flag not in args:
                args.append(flag)
    
    # 4. Economy / Indicator detection
    # Check if any economy keyword is present in the text
    found_indicator = None
    for keyword, indicator in ECONOMY_KEYWORDS.items():
        # Match as whole word/phrase
        if re.search(rf'\b{re.escape(keyword)}\b', text_lower):
            found_indicator = indicator
            break
            
    if found_indicator:
        # Detected an economy query like "chart cpi" or "unemployment rate"
        # We need to construct args: [INDICATOR, optional_chart_flag, optional_period]
        
        eco_args = [found_indicator]
        
        # Check if they want a chart
        is_chart_request = re.search(r'\b(chart|graph|plot)\b', text_lower)
        if is_chart_request:
            eco_args.append("CHART")
            
        # Add period if found (parsed in step 1 or 1b)
        # We check if any args[1:] look like a period/date range
        # Periods are typically at the end of args list from previous steps
        for arg in args:
            # Check for standard periods (5y, 10y) or date flags (--since)
            if (isinstance(arg, str) and 
                (re.match(r'^\d+[dwmy]$', arg) or arg.startswith('--'))):
                eco_args.append(arg)
                break
            
        return Intent(
            command='economy',
            symbols=[],
            args=eco_args,
            confidence=0.9,
            negated_terms=None
        )

    # 5. Sentiment extraction - detect buy/sell/hold questions
    sentiment_pattern = re.search(r'\b(is|should|would)\s+\w+\s+(a\s+)?(buy|sell|hold|bullish|bearish)\b', text_lower)
    is_sentiment_query = sentiment_pattern is not None

    # 6. Comparison parsing (vs/compare) -> returns chart command directly
    is_comparison = re.search(r'\b(vs|versus|compare|compared)\b', text_lower)
    if is_comparison and len(symbols) >= 2:
        # Comparison intent detected
        # e.g. "Compare Apple to Tesla" -> chart with -compare flag
        primary = symbols[0]
        secondary = symbols[1]
        
        # Rebuild args to be explicit: [Primary, -compare, Secondary, ...others]
        current_extras = [a for a in args if a not in symbols]
        new_args = [primary, "-compare", secondary] + current_extras
        
        return Intent(
            command='chart',
            symbols=symbols,
            args=new_args,
            confidence=0.85,
            negated_terms=negated_terms if negated_terms else None,
        )
    
    # 7. Sentiment query -> route to rating command (BEFORE pattern matching)
    # This ensures "should I buy bitcoin?" routes to rating, not crypto
    if is_sentiment_query and symbols:
        return Intent(
            command='rating',
            symbols=symbols,
            args=args,
            confidence=0.8,  # High confidence for explicit sentiment queries
            negated_terms=negated_terms if negated_terms else None,
        )
    
    # Try each intent pattern
    for pattern, command, requires_symbol in INTENT_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            # If command requires a symbol but none found, skip
            if requires_symbol and not symbols:
                continue
            
            # Calculate confidence based on match quality
            confidence = 0.7  # Base confidence
            if symbols:
                confidence += 0.2  # Boost if symbols found
            
            logger.debug(f"Intent parsed: {command} {symbols} args={args} (confidence: {confidence})")
            
            return Intent(
                command=command,
                symbols=symbols,
                args=args,
                confidence=confidence,
                negated_terms=negated_terms if negated_terms else None,
            )
    
    # Check if we found chart-specific args (infer chart command)
    chart_flags = {"-c", "-rsi", "-bb"}
    found_flags = [a for a in args if isinstance(a, str) and (a in chart_flags or a.startswith("-sma"))]
    
    if found_flags:
        # If we have chart params, default to chart command
        return Intent(
            command='chart',
            symbols=symbols,
            args=args,
            confidence=0.6,
            negated_terms=negated_terms if negated_terms else None,
        )

    # Check if text contains a potential symbol we should look up
    if symbols:
        # Default to price lookup if we found symbols but no specific intent
        return Intent(
            command='price',
            symbols=symbols,
            args=args,
            confidence=0.5,
            negated_terms=negated_terms if negated_terms else None,
        )
    
    return None


def is_question_about_stocks(text: str) -> bool:
    """Check if text appears to be asking about stocks/finance."""
    keywords = [
        'stock', 'share', 'price', 'chart', 'earnings', 'dividend',
        'news', 'market', 'trade', 'trading', 'buy', 'sell', 'invest',
        'rsi', 'macd', 'technical', 'analysis', 'fundamentals',
        'pe', 'eps', 'revenue', 'profit', 'quarter',
        'economy', 'gdp', 'inflation', 'cpi', 'unemployment',  # Added economy keywords
    ]
    text_lower = text.lower()
    return any(kw in text_lower for kw in keywords)
