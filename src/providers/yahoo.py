"""
Yahoo Finance provider - no API key required.

Uses yfinance library which scrapes Yahoo Finance.
Best for: development, fallback, users without API keys.
Limitations: unofficial, may break if Yahoo changes their site.
"""

import asyncio
import logging
from datetime import datetime

import yfinance as yf

from .base import (
    BaseProvider,
    Quote,
    HistoricalBar,
    Fundamentals,
    ProviderCapability,
    SymbolNotFoundError,
)

logger = logging.getLogger(__name__)


class YahooFinanceProvider(BaseProvider):
    name = "yahoo"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
        ProviderCapability.FUNDAMENTALS,
    }
    
    async def get_quote(self, symbol: str) -> Quote:
        """Fetch quote using yfinance (runs sync code in executor)"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_quote_sync, symbol)
    
    def _get_quote_sync(self, symbol: str) -> Quote:
        ticker = yf.Ticker(symbol)
        
        # Try info first (more complete but slower)
        try:
            info = ticker.info
            if info and info.get('regularMarketPrice') is not None:
                price = info.get('regularMarketPrice', 0)
                prev_close = info.get('regularMarketPreviousClose', price)
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0
                
                return Quote(
                    symbol=symbol.upper(),
                    price=price,
                    change=change,
                    change_percent=change_pct,
                    volume=info.get('regularMarketVolume', 0) or 0,
                    timestamp=datetime.now(),
                    provider=self.name,
                    open=info.get('regularMarketOpen'),
                    high=info.get('regularMarketDayHigh'),
                    low=info.get('regularMarketDayLow'),
                    prev_close=prev_close,
                    market_cap=info.get('marketCap'),
                    name=info.get('shortName'),
                )
        except Exception as e:
            logger.debug(f"Info lookup failed for {symbol}, trying fast_info: {e}")
        
        # Fallback to fast_info
        try:
            fast = ticker.fast_info
            if fast.last_price is None:
                raise SymbolNotFoundError(f"Symbol not found: {symbol}")
            
            prev_close = fast.previous_close or fast.last_price
            change = fast.last_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return Quote(
                symbol=symbol.upper(),
                price=fast.last_price,
                change=change,
                change_percent=change_pct,
                volume=int(fast.last_volume or 0),
                timestamp=datetime.now(),
                provider=self.name,
                prev_close=prev_close,
                market_cap=getattr(fast, 'market_cap', None),
            )
        except SymbolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise SymbolNotFoundError(f"Symbol not found: {symbol}")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch fetch quotes"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_quotes_sync, symbols)
    
    def _get_quotes_sync(self, symbols: list[str]) -> dict[str, Quote]:
        results = {}
        
        # yfinance batch download for efficiency
        try:
            tickers = yf.Tickers(" ".join(symbols))
            for symbol in symbols:
                try:
                    symbol_upper = symbol.upper()
                    ticker = tickers.tickers.get(symbol_upper)
                    if ticker:
                        quote = self._get_quote_sync(symbol)
                        results[symbol_upper] = quote
                except SymbolNotFoundError:
                    logger.debug(f"Symbol not found in batch: {symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching {symbol} in batch: {e}")
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fall back to individual fetches
            for symbol in symbols:
                try:
                    results[symbol.upper()] = self._get_quote_sync(symbol)
                except Exception:
                    pass
        
        return results
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._get_historical_sync, symbol, period, interval
        )
    
    def _get_historical_sync(
        self, symbol: str, period: str, interval: str
    ) -> list[HistoricalBar]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for idx, row in hist.iterrows():
            bars.append(HistoricalBar(
                timestamp=idx.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
            ))
        
        return bars
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_fundamentals_sync, symbol)
    
    def _get_fundamentals_sync(self, symbol: str) -> Fundamentals:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'shortName' not in info:
            raise SymbolNotFoundError(f"No fundamental data for {symbol}")
        
        return Fundamentals(
            symbol=symbol.upper(),
            name=info.get('shortName', symbol),
            pe_ratio=info.get('trailingPE'),
            eps=info.get('trailingEps'),
            market_cap=info.get('marketCap'),
            dividend_yield=info.get('dividendYield'),
            fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
            fifty_two_week_low=info.get('fiftyTwoWeekLow'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            provider=self.name,
        )
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("AAPL")
            return True
        except Exception as e:
            logger.error(f"Yahoo health check failed: {e}")
            return False
