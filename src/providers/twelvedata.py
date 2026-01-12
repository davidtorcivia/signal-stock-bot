"""
Twelve Data provider - requires free API key.

Free tier: 800 calls/day, 8 calls/minute
Premium tier: Higher limits

Best for: historical data, technical indicators
Get API key: https://twelvedata.com
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


class TwelveDataProvider(BaseProvider):
    name = "twelvedata"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
    }
    
    BASE_URL = "https://api.twelvedata.com"
    
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
        params['apikey'] = self.api_key
        session = await self._get_session()
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(retry_after=60)
                if resp.status == 401:
                    raise ProviderError("Invalid Twelve Data API key")
                
                data = await resp.json()
                
                # Check for error response
                if data.get('status') == 'error':
                    code = data.get('code', 0)
                    message = data.get('message', 'Unknown error')
                    if code == 429:
                        raise RateLimitError(retry_after=60)
                    elif 'not found' in message.lower() or 'invalid' in message.lower():
                        raise SymbolNotFoundError(message)
                    else:
                        raise ProviderError(message)
                
                return data
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error from Twelve Data: {e}")
            raise ProviderError(f"Twelve Data request failed: {e}")
    
    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request('quote', {'symbol': symbol.upper()})
        
        if not data or 'close' not in data:
            raise SymbolNotFoundError(f"No quote data for {symbol}")
        
        price = float(data.get('close', 0))
        prev_close = float(data.get('previous_close', price))
        change = float(data.get('change', price - prev_close))
        change_pct = float(data.get('percent_change', 0))
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=int(data.get('volume', 0) or 0),
            timestamp=datetime.now(),
            provider=self.name,
            open=float(data.get('open', 0)) if data.get('open') else None,
            high=float(data.get('high', 0)) if data.get('high') else None,
            low=float(data.get('low', 0)) if data.get('low') else None,
            prev_close=prev_close,
            name=data.get('name'),
        )
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quotes - Twelve Data supports batch via comma-separated symbols."""
        # Limit batch size to avoid hitting limits
        batch_size = 8
        quotes = {}
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i+batch_size]
            symbol_str = ','.join(s.upper() for s in batch)
            
            try:
                data = await self._request('quote', {'symbol': symbol_str})
                
                # Single symbol returns dict, multiple returns list
                if isinstance(data, dict):
                    if 'close' in data:
                        quotes[batch[0].upper()] = await self._parse_quote(data, batch[0])
                elif isinstance(data, list):
                    for item in data:
                        if item.get('close'):
                            sym = item.get('symbol', '').upper()
                            quotes[sym] = await self._parse_quote(item, sym)
            except RateLimitError:
                raise
            except Exception as e:
                logger.warning(f"Batch quote failed: {e}")
        
        return quotes
    
    async def _parse_quote(self, data: dict, symbol: str) -> Quote:
        price = float(data.get('close', 0))
        prev_close = float(data.get('previous_close', price))
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=float(data.get('change', 0)),
            change_percent=float(data.get('percent_change', 0)),
            volume=int(data.get('volume', 0) or 0),
            timestamp=datetime.now(),
            provider=self.name,
            open=float(data.get('open', 0)) if data.get('open') else None,
            high=float(data.get('high', 0)) if data.get('high') else None,
            low=float(data.get('low', 0)) if data.get('low') else None,
            prev_close=prev_close,
        )
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        """Get historical time series data."""
        # Map period to outputsize
        period_to_size = {
            '1d': 390,   # Intraday minutes
            '5d': 5,
            '1w': 7,
            '1mo': 30,
            '3mo': 90,
            '6mo': 180,
            '1y': 365,
            'ytd': 252,
            '5y': 1260,
            'max': 5000,
        }
        
        # Map interval
        interval_map = {
            '1m': '1min',
            '5m': '5min',
            '15m': '15min',
            '30m': '30min',
            '60m': '1h',
            '1h': '1h',
            '1d': '1day',
            '1wk': '1week',
            '1mo': '1month',
        }
        
        td_interval = interval_map.get(interval, '1day')
        outputsize = period_to_size.get(period, 30)
        
        data = await self._request('time_series', {
            'symbol': symbol.upper(),
            'interval': td_interval,
            'outputsize': outputsize,
        })
        
        values = data.get('values', [])
        if not values:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for v in values:
            try:
                bars.append(HistoricalBar(
                    timestamp=datetime.fromisoformat(v['datetime']),
                    open=float(v['open']),
                    high=float(v['high']),
                    low=float(v['low']),
                    close=float(v['close']),
                    volume=int(v.get('volume', 0) or 0),
                ))
            except (KeyError, ValueError) as e:
                logger.debug(f"Skipping bar: {e}")
        
        return sorted(bars, key=lambda b: b.timestamp)
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("AAPL")
            return True
        except RateLimitError:
            return True
        except Exception as e:
            logger.error(f"Twelve Data health check failed: {e}")
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
