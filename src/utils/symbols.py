"""
Smart symbol resolution for stock/crypto symbols.

Provides:
1. Alias table for common names (instant)
2. Yahoo Finance search fallback (network)
"""

import logging
from typing import Optional, Tuple
import asyncio

logger = logging.getLogger(__name__)


# Common name → symbol mappings (instant lookup)
SYMBOL_ALIASES = {
    # Tech giants
    "apple": "AAPL",
    "microsoft": "MSFT",
    "google": "GOOGL",
    "alphabet": "GOOGL",
    "amazon": "AMZN",
    "meta": "META",
    "facebook": "META",
    "netflix": "NFLX",
    "nvidia": "NVDA",
    "tesla": "TSLA",
    "intel": "INTC",
    "amd": "AMD",
    "salesforce": "CRM",
    "adobe": "ADBE",
    "oracle": "ORCL",
    "ibm": "IBM",
    "cisco": "CSCO",
    "uber": "UBER",
    "lyft": "LYFT",
    "airbnb": "ABNB",
    "spotify": "SPOT",
    "twitter": "X",
    "snap": "SNAP",
    "snapchat": "SNAP",
    "pinterest": "PINS",
    "zoom": "ZM",
    "slack": "WORK",
    "dropbox": "DBX",
    "paypal": "PYPL",
    "square": "SQ",
    "block": "SQ",
    "shopify": "SHOP",
    "coinbase": "COIN",
    "robinhood": "HOOD",
    "palantir": "PLTR",
    "broadcom": "AVGO",
    "arm": "ARM",
    "supermicro": "SMCI",
    "gamestop": "GME",
    "gme": "GME",
    "amc": "AMC",
    "blackberry": "BB",
    
    # Finance
    "jpmorgan": "JPM",
    "chase": "JPM",
    "goldman": "GS",
    "morgan stanley": "MS",
    "bank of america": "BAC",
    "bofa": "BAC",
    "wells fargo": "WFC",
    "citi": "C",
    "citibank": "C",
    "visa": "V",
    "mastercard": "MA",
    "amex": "AXP",
    "american express": "AXP",
    "berkshire": "BRK-B",
    "blackrock": "BLK",
    
    # Retail / Consumer
    "walmart": "WMT",
    "target": "TGT",
    "costco": "COST",
    "home depot": "HD",
    "lowes": "LOW",
    "nike": "NKE",
    "starbucks": "SBUX",
    "mcdonalds": "MCD",
    "coca cola": "KO",
    "coke": "KO",
    "pepsi": "PEP",
    "disney": "DIS",
    
    # Auto
    "ford": "F",
    "gm": "GM",
    "general motors": "GM",
    "rivian": "RIVN",
    "lucid": "LCID",
    
    # Healthcare
    "johnson": "JNJ",
    "jnj": "JNJ",
    "pfizer": "PFE",
    "moderna": "MRNA",
    "merck": "MRK",
    "unitedhealth": "UNH",
    
    # Crypto
    "bitcoin": "BTC-USD",
    "btc": "BTC-USD",
    "ethereum": "ETH-USD",
    "eth": "ETH-USD",
    "solana": "SOL-USD",
    "sol": "SOL-USD",
    "cardano": "ADA-USD",
    "ada": "ADA-USD",
    "dogecoin": "DOGE-USD",
    "doge": "DOGE-USD",
    "xrp": "XRP-USD",
    "ripple": "XRP-USD",
    "polkadot": "DOT-USD",
    "dot": "DOT-USD",
    "avalanche": "AVAX-USD",
    "avax": "AVAX-USD",
    "chainlink": "LINK-USD",
    "link": "LINK-USD",
    "polygon": "MATIC-USD",
    "matic": "MATIC-USD",
    "litecoin": "LTC-USD",
    "ltc": "LTC-USD",
    "sui": "SUI-USD",
    "pepe": "PEPE-USD",
    "shiba": "SHIB-USD",
    "shib": "SHIB-USD",
    "bonk": "BONK-USD",
    
    # Commodities & Futures
    "gold": "GC=F",
    "silver": "SI=F",
    "oil": "CL=F",
    "crude": "CL=F",
    "crude oil": "CL=F",
    "wti": "CL=F",
    "brent": "BZ=F",
    "natural gas": "NG=F",
    "natgas": "NG=F",
    "gas": "NG=F",
    "copper": "HG=F",
    "platinum": "PL=F",
    "palladium": "PA=F",
    "corn": "ZC=F",
    "wheat": "ZW=F",
    "soybeans": "ZS=F",
    "coffee": "KC=F",
    "sugar": "SB=F",
    "cotton": "CT=F",
    
    # Bonds & Treasuries
    "bonds": "^TNX",
    "10 year": "^TNX",
    "10y": "^TNX",
    "tnotes": "^TNX",
    "t-notes": "^TNX",
    "treasury": "^TNX",
    "2 year": "^IRX",
    "2y": "^IRX",
    "30 year": "^TYX",
    "30y": "^TYX",
    "tlt": "TLT",
    
    # Indices & ETFs
    "spy": "SPY",
    "qqq": "QQQ",
    "dia": "DIA",
    "iwm": "IWM",
    "voo": "VOO",
    "vti": "VTI",
    "sp500": "^GSPC",
    "s&p": "^GSPC",
    "s&p 500": "^GSPC",
    "dow": "^DJI",
    "dow jones": "^DJI",
    "nasdaq": "^IXIC",
    "russell": "^RUT",
    "russell 2000": "^RUT",
    "vix": "^VIX",
    "volatility": "^VIX",
    "fear": "^VIX",
    
    # Currencies
    "euro": "EURUSD=X",
    "yen": "JPYUSD=X",
    "pound": "GBPUSD=X",
    "dollar": "DX-Y.NYB",
    "usd": "DX-Y.NYB",
    "dxy": "DX-Y.NYB",
}


def resolve_alias(query: str) -> Optional[str]:
    """
    Check if query matches a known alias.
    
    Returns:
        Symbol if found, None otherwise
    """
    query_lower = query.lower().strip()
    return SYMBOL_ALIASES.get(query_lower)


async def search_yahoo(query: str) -> Optional[Tuple[str, str]]:
    """
    Search Yahoo Finance for a symbol.
    
    Returns:
        (symbol, name) tuple if found, None otherwise
    """
    try:
        import yfinance as yf
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        
        def do_search():
            try:
                # yfinance search returns dict with 'quotes' key
                results = yf.search(query)
                if results and 'quotes' in results:
                    quotes = results['quotes']
                    if quotes:
                        # Return first result
                        first = quotes[0]
                        return (first.get('symbol'), first.get('shortname') or first.get('longname'))
            except Exception as e:
                logger.debug(f"Yahoo search failed for '{query}': {e}")
            return None
        
        result = await loop.run_in_executor(None, do_search)
        return result
        
    except ImportError:
        logger.warning("yfinance not installed, Yahoo search unavailable")
        return None
    except Exception as e:
        logger.debug(f"Yahoo search error: {e}")
        return None


async def resolve_symbol(query: str) -> Tuple[str, Optional[str]]:
    """
    Resolve a query to a symbol using alias table + Yahoo fallback.
    
    Args:
        query: User input (e.g., "apple", "btc", "AAPL")
    
    Returns:
        (symbol, resolved_name) tuple. resolved_name is None if it was already a valid symbol.
    """
    # Already looks like a valid symbol (uppercase, short)
    if query.isupper() and len(query) <= 6 and query.isalpha():
        return (query, None)
    
    # Check alias table first (instant)
    alias_match = resolve_alias(query)
    if alias_match:
        logger.debug(f"Resolved '{query}' → '{alias_match}' via alias")
        return (alias_match, query.title())
    
    # Fallback to Yahoo search
    yahoo_result = await search_yahoo(query)
    if yahoo_result:
        symbol, name = yahoo_result
        logger.debug(f"Resolved '{query}' → '{symbol}' ({name}) via Yahoo")
        return (symbol, name)
    
    # Return as-is if no match (let provider handle validation)
    return (query.upper(), None)


def is_valid_symbol_format(symbol: str) -> bool:
    """Check if string looks like a valid stock symbol."""
    if not symbol:
        return False
    # Allow 1-10 chars, letters, numbers, dash, dot (for BRK-B, BTC-USD)
    import re
    return bool(re.match(r'^[A-Z0-9\-\.]{1,10}$', symbol.upper()))
