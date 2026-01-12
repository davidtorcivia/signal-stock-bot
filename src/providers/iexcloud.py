"""
IEX Cloud provider - requires API key.

Free tier: 50,000 credits/month (varies by endpoint)
Premium tier: Higher limits

Best for: high-quality real-time data, fundamentals
Get API key: https://iexcloud.io
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
    Fundamentals,
    ProviderCapability,
    SymbolNotFoundError,
    RateLimitError,
    ProviderError,
)

logger = logging.getLogger(__name__)


class IEXCloudProvider(BaseProvider):
    name = "iexcloud"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
        ProviderCapability.FUNDAMENTALS,
    }
    
    # Use sandbox for free testing, production for real data
    BASE_URL = "https://cloud.iexapis.com/stable"
    SANDBOX_URL = "https://sandbox.iexapis.com/stable"
    
    def __init__(self, api_key: str, use_sandbox: bool = False):
        self.api_key = api_key
        self.base_url = self.SANDBOX_URL if use_sandbox else self.BASE_URL
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
        url = f"{self.base_url}/{endpoint}"
        
        try:
            async with session.get(url, params=params) as resp:
                if resp.status == 429:
                    raise RateLimitError(retry_after=60)
                if resp.status == 401:
                    raise ProviderError("Invalid IEX Cloud API key")
                if resp.status == 402:
                    raise RateLimitError(retry_after=86400)  # Credits exhausted
                if resp.status == 403:
                    raise ProviderError("IEX Cloud API key lacks permissions")
                if resp.status == 404:
                    raise SymbolNotFoundError("Symbol not found")
                
                # IEX returns empty array for no data
                text = await resp.text()
                if not text or text == '[]' or text == 'null':
                    raise SymbolNotFoundError("No data available")
                
                return await resp.json()
                
        except aiohttp.ClientError as e:
            logger.error(f"HTTP error from IEX Cloud: {e}")
            raise ProviderError(f"IEX Cloud request failed: {e}")
    
    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request(f'stock/{symbol}/quote')
        
        if not data:
            raise SymbolNotFoundError(f"No quote data for {symbol}")
        
        price = data.get('latestPrice', 0)
        prev_close = data.get('previousClose', price)
        change = data.get('change', 0)
        change_pct = data.get('changePercent', 0) * 100 if data.get('changePercent') else 0
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=data.get('volume', 0) or 0,
            timestamp=datetime.now(),
            provider=self.name,
            open=data.get('open'),
            high=data.get('high'),
            low=data.get('low'),
            prev_close=prev_close,
            market_cap=data.get('marketCap'),
            name=data.get('companyName'),
        )
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quotes - IEX supports batch via market/batch endpoint."""
        symbol_str = ','.join(s.upper() for s in symbols)
        
        try:
            data = await self._request('stock/market/batch', {
                'symbols': symbol_str,
                'types': 'quote',
            })
            
            quotes = {}
            for sym, info in data.items():
                if 'quote' in info and info['quote']:
                    q = info['quote']
                    price = q.get('latestPrice', 0)
                    prev_close = q.get('previousClose', price)
                    
                    quotes[sym.upper()] = Quote(
                        symbol=sym.upper(),
                        price=price,
                        change=q.get('change', 0),
                        change_percent=q.get('changePercent', 0) * 100 if q.get('changePercent') else 0,
                        volume=q.get('volume', 0) or 0,
                        timestamp=datetime.now(),
                        provider=self.name,
                        open=q.get('open'),
                        high=q.get('high'),
                        low=q.get('low'),
                        prev_close=prev_close,
                        market_cap=q.get('marketCap'),
                        name=q.get('companyName'),
                    )
            
            return quotes
            
        except RateLimitError:
            raise
        except Exception as e:
            logger.warning(f"Batch quote failed: {e}")
            return {}
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        """Get historical chart data."""
        # Map period to IEX range
        range_map = {
            '1d': '1d',
            '5d': '5d',
            '1w': '5d',
            '1mo': '1m',
            '3mo': '3m',
            '6mo': '6m',
            '1y': '1y',
            'ytd': 'ytd',
            '5y': '5y',
            'max': 'max',
        }
        
        iex_range = range_map.get(period, '1m')
        
        data = await self._request(f'stock/{symbol}/chart/{iex_range}')
        
        if not data:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for item in data:
            try:
                # Handle intraday vs daily format
                if 'minute' in item:
                    ts = datetime.fromisoformat(f"{item['date']}T{item['minute']}")
                else:
                    ts = datetime.fromisoformat(item['date'])
                
                bars.append(HistoricalBar(
                    timestamp=ts,
                    open=float(item.get('open') or item.get('fOpen') or 0),
                    high=float(item.get('high') or item.get('fHigh') or 0),
                    low=float(item.get('low') or item.get('fLow') or 0),
                    close=float(item.get('close') or item.get('fClose') or 0),
                    volume=int(item.get('volume') or item.get('fVolume') or 0),
                ))
            except (KeyError, ValueError, TypeError) as e:
                logger.debug(f"Skipping bar: {e}")
        
        return sorted(bars, key=lambda b: b.timestamp)
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Get company fundamentals from key stats and company info."""
        # Fetch both endpoints
        try:
            stats = await self._request(f'stock/{symbol}/stats')
            company = await self._request(f'stock/{symbol}/company')
        except SymbolNotFoundError:
            raise
        except Exception as e:
            logger.warning(f"Failed to get fundamentals for {symbol}: {e}")
            raise SymbolNotFoundError(f"No fundamental data for {symbol}")
        
        return Fundamentals(
            symbol=symbol.upper(),
            name=company.get('companyName', symbol),
            pe_ratio=stats.get('peRatio'),
            eps=stats.get('ttmEPS'),
            market_cap=stats.get('marketcap'),
            dividend_yield=stats.get('dividendYield'),
            fifty_two_week_high=stats.get('week52high'),
            fifty_two_week_low=stats.get('week52low'),
            sector=company.get('sector'),
            industry=company.get('industry'),
            provider=self.name,
        )
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("AAPL")
            return True
        except RateLimitError:
            return True
        except Exception as e:
            logger.error(f"IEX Cloud health check failed: {e}")
            return False
    
    async def close(self):
        if self._session and not self._session.closed:
            await self._session.close()
