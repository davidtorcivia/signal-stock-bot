"""
Alpha Vantage provider - requires free API key.

Free tier: 25 requests/day (was 500/day in 2024)
Paid tier: Higher limits

Best for: fundamentals, historical data, forex
"""

import asyncio
import logging
from datetime import datetime
from typing import Optional

import aiohttp

from .base import (
    BaseProvider,
    Quote,
    HistoricalBar,
    Fundamentals,
    ProviderCapability,
    SymbolNotFoundError,
    RateLimitError,
)

logger = logging.getLogger(__name__)


class AlphaVantageProvider(BaseProvider):
    name = "alphavantage"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
        ProviderCapability.FUNDAMENTALS,
    }
    
    BASE_URL = "https://www.alphavantage.co/query"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _request(self, params: dict) -> dict:
        params['apikey'] = self.api_key
        session = await self._get_session()
        
        try:
            async with session.get(self.BASE_URL, params=params) as resp:
                data = await resp.json()
                
                # Check for rate limit
                if 'Note' in data:
                    if 'call frequency' in data['Note'].lower():
                        raise RateLimitError(retry_after=60)
                    if 'premium' in data['Note'].lower():
                        raise RateLimitError(retry_after=86400)  # Daily limit
                
                # Check for error
                if 'Error Message' in data:
                    raise SymbolNotFoundError(data['Error Message'])
                
                # Check for empty response
                if 'Information' in data:
                    logger.warning(f"Alpha Vantage info: {data['Information']}")
                    raise RateLimitError(retry_after=60)
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error from Alpha Vantage: {e}")
            raise
    
    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request({
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
        })
        
        quote_data = data.get('Global Quote', {})
        if not quote_data:
            raise SymbolNotFoundError(f"No quote data for {symbol}")
        
        try:
            price = float(quote_data.get('05. price', 0))
            change = float(quote_data.get('09. change', 0))
            change_pct_str = quote_data.get('10. change percent', '0%')
            change_pct = float(change_pct_str.rstrip('%'))
            
            return Quote(
                symbol=symbol.upper(),
                price=price,
                change=change,
                change_percent=change_pct,
                volume=int(quote_data.get('06. volume', 0)),
                timestamp=datetime.now(),
                provider=self.name,
                open=float(quote_data.get('02. open', 0)) or None,
                high=float(quote_data.get('03. high', 0)) or None,
                low=float(quote_data.get('04. low', 0)) or None,
                prev_close=float(quote_data.get('08. previous close', 0)) or None,
            )
        except (ValueError, TypeError) as e:
            logger.error(f"Error parsing quote data for {symbol}: {e}")
            raise SymbolNotFoundError(f"Invalid data for {symbol}")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """
        Alpha Vantage doesn't support batch quotes.
        We parallelize but this consumes multiple API calls.
        """
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quotes = {}
        for i, result in enumerate(results):
            if isinstance(result, Quote):
                quotes[symbols[i].upper()] = result
            elif isinstance(result, RateLimitError):
                # Stop processing if rate limited
                raise result
            else:
                logger.debug(f"Failed to get quote for {symbols[i]}: {result}")
        
        return quotes
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        # Map interval to Alpha Vantage function
        if interval in ('1m', '5m', '15m', '30m', '60m'):
            function = 'TIME_SERIES_INTRADAY'
            params = {'interval': interval}
        else:
            function = 'TIME_SERIES_DAILY'
            params = {}
        
        # Map period to outputsize
        compact_periods = {'5d', '1wk', '1mo'}
        outputsize = 'compact' if period in compact_periods else 'full'
        
        data = await self._request({
            'function': function,
            'symbol': symbol,
            'outputsize': outputsize,
            **params,
        })
        
        # Find the time series key (varies by function)
        ts_key = None
        for key in data:
            if 'Time Series' in key:
                ts_key = key
                break
        
        if not ts_key:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for date_str, values in data[ts_key].items():
            try:
                bars.append(HistoricalBar(
                    timestamp=datetime.fromisoformat(date_str.replace(' ', 'T')),
                    open=float(values['1. open']),
                    high=float(values['2. high']),
                    low=float(values['3. low']),
                    close=float(values['4. close']),
                    volume=int(values['5. volume']),
                ))
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping bar {date_str}: {e}")
        
        return sorted(bars, key=lambda b: b.timestamp)
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        data = await self._request({
            'function': 'OVERVIEW',
            'symbol': symbol,
        })
        
        if not data or 'Symbol' not in data:
            raise SymbolNotFoundError(f"No fundamental data for {symbol}")
        
        def safe_float(val):
            try:
                if val and val not in ('None', '-', 'N/A'):
                    return float(val)
            except (ValueError, TypeError):
                pass
            return None
        
        def safe_int(val):
            try:
                if val and val not in ('None', '-', 'N/A'):
                    return int(val)
            except (ValueError, TypeError):
                pass
            return None
        
        return Fundamentals(
            symbol=symbol.upper(),
            name=data.get('Name', symbol),
            pe_ratio=safe_float(data.get('PERatio')),
            eps=safe_float(data.get('EPS')),
            market_cap=safe_int(data.get('MarketCapitalization')),
            dividend_yield=safe_float(data.get('DividendYield')),
            fifty_two_week_high=safe_float(data.get('52WeekHigh')),
            fifty_two_week_low=safe_float(data.get('52WeekLow')),
            sector=data.get('Sector') if data.get('Sector') != 'None' else None,
            industry=data.get('Industry') if data.get('Industry') != 'None' else None,
            provider=self.name,
        )
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("IBM")
            return True
        except RateLimitError:
            # Rate limit means API is working, just exhausted
            return True
        except Exception as e:
            logger.error(f"Alpha Vantage health check failed: {e}")
            return False
    
    async def close(self):
        """Close the aiohttp session"""
        if self._session and not self._session.closed:
            await self._session.close()
