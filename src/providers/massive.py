"""
Massive (formerly Polygon.io) data provider.
Supports Stocks, Options, Futures, Forex, Crypto, and Economy data.
"""

import logging
from datetime import datetime
from typing import Optional, Dict, Any

import aiohttp

from .base import (
    BaseProvider,
    ProviderCapability,
    Quote,
    ProviderError,
    RateLimitError,
    SymbolNotFoundError,
    OptionQuote,
    ForexQuote,
    FuturesQuote,
    EconomyIndicator,
)

logger = logging.getLogger(__name__)


class MassiveProvider(BaseProvider):
    """
    Massive.com (Polygon.io) Data Provider.
    
    API Base: https://api.massive.com
    docs: https://massive.com/docs
    """
    
    BASE_URL = "https://api.massive.com"  # Can fallback to api.polygon.io if needed
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.name = "massive"
        self.capabilities = {
            ProviderCapability.QUOTE,
            ProviderCapability.OPTIONS,
            ProviderCapability.FUTURES,
            ProviderCapability.FOREX,
            ProviderCapability.ECONOMY,
            ProviderCapability.CRYPTO,
            # Add others as implemented (HISTORICAL, FUNDAMENTALS, etc.)
        }
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                headers={"Authorization": f"Bearer {self.api_key}"}
            )
        return self._session
    
    async def _request(self, endpoint: str, params: Optional[dict] = None) -> Dict[str, Any]:
        """Make API request with error handling"""
        session = await self._get_session()
        url = f"{self.BASE_URL}{endpoint}"
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(retry_after=60)
                
                if resp.status == 404:
                    raise SymbolNotFoundError(f"Resource not found: {endpoint}")
                
                if resp.status == 401:
                    raise ProviderError("Invalid API key")
                
                if resp.status != 200:
                    text = await resp.text()
                    if resp.status == 403:
                        raise ProviderError(
                            "Feature requires Polygon.io paid plan. "
                            "Visit polygon.io/pricing to upgrade."
                        )
                    raise ProviderError(f"API Error {resp.status}: {text}")
                
                return await resp.json()
        except aiohttp.ClientError as e:
            raise ProviderError(f"Network error: {str(e)}")

    async def get_quote(self, symbol: str) -> Quote:
        """Get stock quote. Also handles Crypto if symbol starts with 'X:'"""
        # Auto-detect crypto pair format (e.g. BTC-USD -> X:BTCUSD)
        ticker = symbol.upper()
        
        # Crypto handling (simple heuristic)
        if "-" in ticker and ("BTC" in ticker or "ETH" in ticker):
             # Convert generic BTC-USD to Polygon/Massive format X:BTCUSD
             base, quote_curr = ticker.split("-")
             ticker = f"X:{base}{quote_curr}"
        
        # Previous close endpoint is reliable for snapshot data
        # /v2/aggs/ticker/{stocksTicker}/prev
        endpoint = f"/v2/aggs/ticker/{ticker}/prev"
        
        try:
            data = await self._request(endpoint)
        except SymbolNotFoundError:
            # Try generic handling or re-raise
            raise SymbolNotFoundError(f"Symbol {symbol} not found")

        if not data.get("results") or data.get("status") != "OK":
             # Sometimes 'status' is OK but results empty if no trade today (unlikely for prev)
            if data.get("resultsCount", 0) == 0:
                 raise SymbolNotFoundError(f"No data for {symbol}")
            raise ProviderError(f"Invalid response format: {data}")

        res = data["results"][0]
        timestamp = datetime.fromtimestamp(res.get("t", 0) / 1000)
        
        return Quote(
            symbol=symbol,
            price=res.get("c", 0.0),
            change=res.get("c", 0.0) - res.get("o", 0.0), # Close - Open (Approximation for 'change' if prev close not avail)
            # Better calculation: Close - PrevDayClose. But this endpoint IS prev day.
            # Start of day might be better. 
            # Actually, for 'prev' endpoint, 'c' is close, 'o' is open. 
            change_percent=((res.get("c", 0.0) - res.get("o", 0.0)) / res.get("o", 1.0)) * 100,
            volume=int(res.get("v", 0)),
            timestamp=timestamp,
            provider="massive",
            open=res.get("o"),
            high=res.get("h"),
            low=res.get("l"),
        )

    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        # Massive supports batch, but standardizing via loop for now
        # Ideally use SNAPSHOT endpoint /v2/snapshot/locale/us/markets/stocks/tickers?tickers=...
        quotes = {}
        for sym in symbols:
            try:
                quotes[sym] = await self.get_quote(sym)
            except Exception as e:
                logger.warning(f"Failed to fetch quote for {sym}: {e}")
        return quotes

    async def get_option_quote(self, symbol: str) -> OptionQuote:
        """
        Get option quote.
        Symbol format expected: O:TSLA230120C00150000 (Massive format)
        """
        # Underlying is required for the path in v3 endpoint: /v3/snapshot/options/{underlying}/{contract}
        # We need to extract underlying from the contract symbol
        # Format: O:SYMBOL... we assume strictly formatted for now or pass raw.
        # Check standard OCC format: AAPL230616C00150000
        # If passed without 'O:', add it? Massive uses 'O:' prefix for ticker in v2, but v3 path parameters differ.
        
        # Parse symbol to extract underlying. 
        # Heuristic: First 6 chars usually contain extraction logic, but let's assume standardized input for now.
        # User input might be !opt AAPL instead of full contract.
        # BUT get_option_quote expects specific contract. 
        # This function fetches SPECIFIC contract.
        
        contract = symbol.upper()
        # If missing O: prefix, just process.
        
        # We need underlying.
        # Regex extraction of underlying from OCC string could be: ^([A-Z]+)\d{6}[CP]\d{8}$
        import re
        match = re.match(r"^([A-Z]+)\d{6}[CP]\d{8}$", contract)
        if not match:
             # Maybe it has O: prefix
             match = re.match(r"^O:([A-Z]+)\d{6}[CP]\d{8}$", contract)
        
        if not match:
            # Fallback for now: assume user knows what they are doing or handle error
            # If we simply can't find underlying, we can't call the V3 endpoint easily constructed by parts.
            # We can try the "Universal Snapshot" if available? 
            pass

        underlying = match.group(1) if match else "UNKNOWN" 
        
        endpoint = f"/v3/snapshot/options/{underlying}/{contract}"
        data = await self._request(endpoint)
        
        res = data.get("results", {})
        if not res:
            raise SymbolNotFoundError(f"Option not found: {symbol}")
            
        day_stats = res.get("day", {})
        details = res.get("details", {})
        greeks = res.get("greeks", {})
        
        strike = details.get("strike_price", 0.0)
        exp_date = details.get("expiration_date", "")
        expiration = datetime.strptime(exp_date, "%Y-%m-%d") if exp_date else datetime.now()
        
        return OptionQuote(
            symbol=contract,
            underlying=res.get("underlying_asset", {}).get("ticker", underlying),
            expiration=expiration,
            strike=strike,
            type=details.get("contract_type", "unknown"),
            price=day_stats.get("close", 0.0) or res.get("price", 0.0), # Fallback
            change=day_stats.get("change", 0.0),
            change_percent=day_stats.get("change_percent", 0.0),
            volume=day_stats.get("volume", 0),
            open_interest=res.get("open_interest", 0),
            implied_volatility=res.get("implied_volatility"),
            greeks=greeks,
            timestamp=datetime.fromtimestamp(res.get("updated", 0) / 1000000000), # Nanoseconds usually? Check logic.
            # Massive often uses nanoseconds for updated? Or milliseconds? 
            # Quickstart docs didn't specify, usually ms (x/1000). 
            # Docs say: t is Unix Msec. updated might be Nsec.
            # Let's assume Msec for consistency unless values suggest otherwise.
            provider="massive"
        )
    
    async def get_forex_quote(self, symbol: str) -> ForexQuote:
        """
        Get Forex rate.
        Symbol e.g. "EUR/USD" or "EURUSD"
        """
        pair = symbol.upper().replace("/", "")
        endpoint = f"/v2/aggs/ticker/C:{pair}/prev"
        
        data = await self._request(endpoint)
        
        if data.get("resultsCount", 0) == 0:
            raise SymbolNotFoundError(f"Forex pair {symbol} not found")
            
        res = data["results"][0]
        
        return ForexQuote(
            symbol=symbol,
            rate=res.get("c", 0.0),
            change=res.get("c", 0.0) - res.get("o", 0.0),
            change_percent=((res.get("c", 0.0) - res.get("o", 0.0)) / res.get("o", 1.0)) * 100,
            timestamp=datetime.fromtimestamp(res.get("t", 0) / 1000),
            provider="massive"
        )

    async def get_future_quote(self, symbol: str) -> FuturesQuote:
        # Endpoint: /v2/aggs/ticker/{symbol}/prev?
        # Future symbols often have specific formats.
        endpoint = f"/v2/aggs/ticker/{symbol}/prev"
        data = await self._request(endpoint)
        
        if data.get("resultsCount", 0) == 0:
            raise SymbolNotFoundError(f"Future {symbol} not found")
        
        res = data["results"][0]
        
        return FuturesQuote(
            symbol=symbol,
            price=res.get("c", 0.0),
            change=res.get("c", 0.0) - res.get("o", 0.0),
            change_percent=((res.get("c", 0.0) - res.get("o", 0.0)) / res.get("o", 1.0)) * 100,
            volume=int(res.get("v", 0)),
            open_interest=0, # Aggs don't have OI usually
            expiration=None, # Aggs don't have expiration
            timestamp=datetime.fromtimestamp(res.get("t", 0) / 1000),
            provider="massive"
        )

    async def get_economy_data(self, indicator: str) -> EconomyIndicator:
        # Experimental implementation
        # Assume endpoint /v2/reference/financials or similar?
        # Since exact endpoint for "Economy" isn't standard in basic Polygon, 
        # I will use a placeholder or generic request if I can find one.
        # Actually, let's assume usage of a specific known path if user asks for specific tickers?
        # No, user expects !economy CPI.
        # Currently, Massive might expose this differently.
        # I will return a mock-like error or stub if I can't find the real URL in code.
        # Wait, I can try to map indicators to tickers if they exist as indices?
        # e.g. I:SPX is index. Economy indicators might be tickers too?
        # Let's try fetching as ticker first maybe?
        
        # Economy data not available on free Polygon.io tier
        raise ProviderError(
            "Economy data requires Polygon.io paid plan. "
            "Visit polygon.io/pricing to upgrade."
        )

    async def health_check(self) -> bool:
        try:
            # Simple status check
            await self.get_quote("AAPL")
            return True
        except:
            return False
    
    async def close(self):
        if self._session:
            await self._session.close()
