"""
Finnhub provider - requires free API key.

Free tier: 60 calls/minute
Premium tier: Higher limits

Best for: real-time quotes, US stocks
Get API key: https://finnhub.io
"""

import asyncio
import logging
from datetime import datetime, timedelta
from typing import Optional

import aiohttp

from .base import (
    BaseProvider,
    Quote,
    HistoricalBar,
    ProviderCapability,
    SymbolNotFoundError,
    RateLimitError,
    ProviderError,
)

logger = logging.getLogger(__name__)


class FinnhubProvider(BaseProvider):
    name = "finnhub"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
    }
    
    BASE_URL = "https://finnhub.io/api/v1"
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=30)
            self._session = aiohttp.ClientSession(timeout=timeout)
        return self._session
    
    async def _request(self, endpoint: str, params: dict = None) -> dict:
        params = params or {}
        params['token'] = self.api_key
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(retry_after=60)
                if resp.status == 401:
                    raise ProviderError("Invalid Finnhub API key")
                if resp.status == 403:
                    raise ProviderError("Finnhub API key lacks permissions")
                
                data = await resp.json()
                
                # Check for error response
                if isinstance(data, dict) and data.get('error'):
                    raise ProviderError(data['error'])
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error from Finnhub: {e}")
            raise ProviderError(f"Finnhub request failed: {e}")
    
    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request('quote', {'symbol': symbol.upper()})
        
        # Finnhub returns empty data for invalid symbols
        if not data or data.get('c') is None or data.get('c') == 0:
            raise SymbolNotFoundError(f"No quote data for {symbol}")
        
        price = data.get('c', 0)  # Current price
        prev_close = data.get('pc', price)  # Previous close
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=0,  # Finnhub quote endpoint doesn't include volume
            timestamp=datetime.now(),
            provider=self.name,
            open=data.get('o'),
            high=data.get('h'),
            low=data.get('l'),
            prev_close=prev_close,
        )
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quotes - Finnhub doesn't support batch, so we parallelize."""
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        quotes = {}
        for i, result in enumerate(results):
            if isinstance(result, Quote):
                quotes[symbols[i].upper()] = result
            elif isinstance(result, RateLimitError):
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
        """Get historical candle data from Finnhub."""
        # Map period to timestamps
        now = datetime.now()
        period_map = {
            '1d': timedelta(days=1),
            '5d': timedelta(days=5),
            '1w': timedelta(weeks=1),
            '1mo': timedelta(days=30),
            '3mo': timedelta(days=90),
            '6mo': timedelta(days=180),
            '1y': timedelta(days=365),
            'ytd': timedelta(days=(now - datetime(now.year, 1, 1)).days),
            '5y': timedelta(days=365*5),
            'max': timedelta(days=365*20),
        }
        
        delta = period_map.get(period, timedelta(days=30))
        from_ts = int((now - delta).timestamp())
        to_ts = int(now.timestamp())
        
        # Map interval to Finnhub resolution
        resolution_map = {
            '1m': '1',
            '5m': '5',
            '15m': '15',
            '30m': '30',
            '60m': '60',
            '1h': '60',
            '1d': 'D',
            '1wk': 'W',
            '1mo': 'M',
        }
        resolution = resolution_map.get(interval, 'D')
        
        data = await self._request('stock/candle', {
            'symbol': symbol.upper(),
            'resolution': resolution,
            'from': from_ts,
            'to': to_ts,
        })
        
        if data.get('s') == 'no_data' or not data.get('c'):
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        timestamps = data.get('t', [])
        opens = data.get('o', [])
        highs = data.get('h', [])
        lows = data.get('l', [])
        closes = data.get('c', [])
        volumes = data.get('v', [])
        
        for i in range(len(timestamps)):
            bars.append(HistoricalBar(
                timestamp=datetime.fromtimestamp(timestamps[i]),
                open=opens[i],
                high=highs[i],
                low=lows[i],
                close=closes[i],
                volume=int(volumes[i]) if i < len(volumes) else 0,
            ))
        
        return sorted(bars, key=lambda b: b.timestamp)
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("AAPL")
            return True
        except RateLimitError:
            return True  # Rate limit means API is working
        except Exception as e:
            logger.error(f"Finnhub health check failed: {e}")
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
