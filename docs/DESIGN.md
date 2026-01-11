# Signal Stock Bot: Design Document

A robust, multi-provider stock data bot for Signal with support for multiple financial APIs.

## Overview

**Goals**:
- Respond to stock queries in Signal chats (individual and group)
- Support multiple financial data providers with graceful fallback
- Handle rate limits, API failures, and edge cases gracefully
- Be easily extensible for new data sources and commands
- Maintain clean separation of concerns for testability

**Non-goals**:
- Trading execution
- Portfolio management (maybe v2)
- Real-time streaming alerts (maybe v2)

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        Signal Layer                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │  Webhook    │───▶│   Message    │───▶│     Command       │  │
│  │  Handler    │    │   Router     │    │     Dispatcher    │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Command Layer                              │
│  ┌─────────────┐    ┌──────────────┐    ┌───────────────────┐  │
│  │   !price    │    │   !quote     │    │     !market       │  │
│  │   Command   │    │   Command    │    │     Command       │  │
│  └─────────────┘    └──────────────┘    └───────────────────┘  │
└────────────────────────────────┬────────────────────────────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────────────────┐
│                      Provider Layer                              │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                  ProviderManager                         │   │
│  │    ┌─────────┐  ┌─────────────┐  ┌────────────────┐    │   │
│  │    │ Yahoo   │  │ AlphaVantage│  │   Polygon      │    │   │
│  │    │ Finance │  │   Provider  │  │   Provider     │    │   │
│  │    └─────────┘  └─────────────┘  └────────────────┘    │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Provider Abstraction

### Interface definition

All providers implement a common interface, enabling fallback chains and easy testing:

```python
# src/providers/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum

class ProviderCapability(Enum):
    QUOTE = "quote"              # Current price
    HISTORICAL = "historical"    # OHLCV history
    FUNDAMENTALS = "fundamentals"  # P/E, market cap, etc.
    OPTIONS = "options"          # Options chain
    NEWS = "news"                # Company news
    CRYPTO = "crypto"            # Cryptocurrency data

@dataclass
class Quote:
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    timestamp: datetime
    provider: str
    
    # Optional fields (provider-dependent)
    open: Optional[float] = None
    high: Optional[float] = None
    low: Optional[float] = None
    prev_close: Optional[float] = None
    market_cap: Optional[int] = None
    name: Optional[str] = None

@dataclass
class HistoricalBar:
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int

@dataclass 
class Fundamentals:
    symbol: str
    name: str
    pe_ratio: Optional[float]
    eps: Optional[float]
    market_cap: Optional[int]
    dividend_yield: Optional[float]
    fifty_two_week_high: Optional[float]
    fifty_two_week_low: Optional[float]
    sector: Optional[str]
    industry: Optional[str]
    provider: str

class ProviderError(Exception):
    """Base exception for provider errors"""
    pass

class RateLimitError(ProviderError):
    """Raised when rate limit is hit"""
    def __init__(self, retry_after: Optional[int] = None):
        self.retry_after = retry_after
        super().__init__(f"Rate limit exceeded. Retry after {retry_after}s")

class SymbolNotFoundError(ProviderError):
    """Raised when symbol doesn't exist"""
    pass

class BaseProvider(ABC):
    """Abstract base for all financial data providers"""
    
    name: str
    capabilities: set[ProviderCapability]
    
    @abstractmethod
    async def get_quote(self, symbol: str) -> Quote:
        """Get current quote for a symbol"""
        pass
    
    @abstractmethod
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quote retrieval"""
        pass
    
    async def get_historical(
        self, 
        symbol: str, 
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        """Get historical OHLCV data. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support historical data")
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Get fundamental data. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support fundamentals")
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is operational"""
        pass
```

### Yahoo Finance provider (no API key required)

```python
# src/providers/yahoo.py

import yfinance as yf
import asyncio
from datetime import datetime
from .base import (
    BaseProvider, Quote, HistoricalBar, Fundamentals,
    ProviderCapability, SymbolNotFoundError, ProviderError
)

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
        info = ticker.info
        
        if not info or info.get('regularMarketPrice') is None:
            # Try fast_info as fallback
            try:
                fast = ticker.fast_info
                if fast.last_price is None:
                    raise SymbolNotFoundError(f"Symbol not found: {symbol}")
                return Quote(
                    symbol=symbol.upper(),
                    price=fast.last_price,
                    change=fast.last_price - fast.previous_close,
                    change_percent=((fast.last_price - fast.previous_close) / fast.previous_close) * 100,
                    volume=int(fast.last_volume or 0),
                    timestamp=datetime.now(),
                    provider=self.name,
                    prev_close=fast.previous_close,
                    market_cap=fast.market_cap,
                )
            except Exception:
                raise SymbolNotFoundError(f"Symbol not found: {symbol}")
        
        price = info.get('regularMarketPrice', 0)
        prev_close = info.get('regularMarketPreviousClose', price)
        change = price - prev_close
        change_pct = (change / prev_close * 100) if prev_close else 0
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=info.get('regularMarketVolume', 0),
            timestamp=datetime.now(),
            provider=self.name,
            open=info.get('regularMarketOpen'),
            high=info.get('regularMarketDayHigh'),
            low=info.get('regularMarketDayLow'),
            prev_close=prev_close,
            market_cap=info.get('marketCap'),
            name=info.get('shortName'),
        )
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch fetch quotes"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_quotes_sync, symbols)
    
    def _get_quotes_sync(self, symbols: list[str]) -> dict[str, Quote]:
        results = {}
        tickers = yf.Tickers(" ".join(symbols))
        
        for symbol in symbols:
            try:
                ticker = tickers.tickers.get(symbol.upper())
                if ticker:
                    results[symbol.upper()] = self._get_quote_sync(symbol)
            except SymbolNotFoundError:
                continue
            except Exception:
                continue
        
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
                open=row['Open'],
                high=row['High'],
                low=row['Low'],
                close=row['Close'],
                volume=int(row['Volume']),
            ))
        return bars
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._get_fundamentals_sync, symbol)
    
    def _get_fundamentals_sync(self, symbol: str) -> Fundamentals:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info:
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
        except Exception:
            return False
```

### Alpha Vantage provider (API key required)

```python
# src/providers/alphavantage.py

import aiohttp
import asyncio
from datetime import datetime
from typing import Optional
from .base import (
    BaseProvider, Quote, HistoricalBar, Fundamentals,
    ProviderCapability, SymbolNotFoundError, RateLimitError, ProviderError
)

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
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def _request(self, params: dict) -> dict:
        params['apikey'] = self.api_key
        session = await self._get_session()
        
        async with session.get(self.BASE_URL, params=params) as resp:
            data = await resp.json()
            
            # Check for rate limit
            if 'Note' in data and 'call frequency' in data['Note']:
                raise RateLimitError(retry_after=60)
            
            # Check for error
            if 'Error Message' in data:
                raise SymbolNotFoundError(data['Error Message'])
            
            return data
    
    async def get_quote(self, symbol: str) -> Quote:
        data = await self._request({
            'function': 'GLOBAL_QUOTE',
            'symbol': symbol,
        })
        
        quote_data = data.get('Global Quote', {})
        if not quote_data:
            raise SymbolNotFoundError(f"No quote data for {symbol}")
        
        price = float(quote_data.get('05. price', 0))
        change = float(quote_data.get('09. change', 0))
        change_pct = float(quote_data.get('10. change percent', '0%').rstrip('%'))
        
        return Quote(
            symbol=symbol.upper(),
            price=price,
            change=change,
            change_percent=change_pct,
            volume=int(quote_data.get('06. volume', 0)),
            timestamp=datetime.now(),
            provider=self.name,
            open=float(quote_data.get('02. open', 0)),
            high=float(quote_data.get('03. high', 0)),
            low=float(quote_data.get('04. low', 0)),
            prev_close=float(quote_data.get('08. previous close', 0)),
        )
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Alpha Vantage doesn't support batch quotes, so we parallelize"""
        tasks = [self.get_quote(s) for s in symbols]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        return {
            symbols[i].upper(): result 
            for i, result in enumerate(results) 
            if isinstance(result, Quote)
        }
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        # Map period/interval to Alpha Vantage functions
        if interval in ('1m', '5m', '15m', '30m', '60m'):
            function = 'TIME_SERIES_INTRADAY'
            params = {'interval': interval}
        else:
            function = 'TIME_SERIES_DAILY'
            params = {}
        
        data = await self._request({
            'function': function,
            'symbol': symbol,
            'outputsize': 'compact',
            **params,
        })
        
        # Find the time series key
        ts_key = None
        for key in data:
            if 'Time Series' in key:
                ts_key = key
                break
        
        if not ts_key:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for date_str, values in data[ts_key].items():
            bars.append(HistoricalBar(
                timestamp=datetime.fromisoformat(date_str),
                open=float(values['1. open']),
                high=float(values['2. high']),
                low=float(values['3. low']),
                close=float(values['4. close']),
                volume=int(values['5. volume']),
            ))
        
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
                return float(val) if val and val != 'None' else None
            except:
                return None
        
        def safe_int(val):
            try:
                return int(val) if val and val != 'None' else None
            except:
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
            sector=data.get('Sector'),
            industry=data.get('Industry'),
            provider=self.name,
        )
    
    async def health_check(self) -> bool:
        try:
            await self.get_quote("IBM")
            return True
        except RateLimitError:
            return True  # Rate limit means API is working
        except Exception:
            return False
    
    async def close(self):
        if self._session:
            await self._session.close()
```

### Provider manager with fallback

```python
# src/providers/manager.py

import asyncio
import logging
from typing import Optional
from .base import (
    BaseProvider, Quote, HistoricalBar, Fundamentals,
    ProviderCapability, ProviderError, RateLimitError
)

logger = logging.getLogger(__name__)

class ProviderManager:
    """
    Manages multiple providers with automatic fallback.
    
    Priority order is determined by the order providers are added.
    If a provider fails or is rate limited, the next provider is tried.
    """
    
    def __init__(self):
        self.providers: list[BaseProvider] = []
        self._rate_limited: dict[str, float] = {}  # provider_name -> retry_after_timestamp
    
    def add_provider(self, provider: BaseProvider):
        """Add a provider to the fallback chain"""
        self.providers.append(provider)
        logger.info(f"Added provider: {provider.name} with capabilities: {provider.capabilities}")
    
    def _get_available_providers(
        self, 
        capability: ProviderCapability
    ) -> list[BaseProvider]:
        """Get providers that support a capability and aren't rate limited"""
        import time
        now = time.time()
        
        available = []
        for p in self.providers:
            if capability not in p.capabilities:
                continue
            
            rate_limit_until = self._rate_limited.get(p.name, 0)
            if now < rate_limit_until:
                logger.debug(f"Provider {p.name} is rate limited until {rate_limit_until}")
                continue
            
            available.append(p)
        
        return available
    
    def _mark_rate_limited(self, provider: BaseProvider, retry_after: int):
        """Mark a provider as rate limited"""
        import time
        self._rate_limited[provider.name] = time.time() + retry_after
        logger.warning(f"Provider {provider.name} rate limited for {retry_after}s")
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get quote with automatic fallback"""
        providers = self._get_available_providers(ProviderCapability.QUOTE)
        
        if not providers:
            raise ProviderError("No providers available for quotes")
        
        last_error = None
        for provider in providers:
            try:
                logger.debug(f"Trying {provider.name} for quote: {symbol}")
                return await provider.get_quote(symbol)
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
            except ProviderError as e:
                logger.warning(f"Provider {provider.name} failed: {e}")
                last_error = e
            except Exception as e:
                logger.exception(f"Unexpected error from {provider.name}")
                last_error = e
        
        raise last_error or ProviderError("All providers failed")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quotes with fallback"""
        providers = self._get_available_providers(ProviderCapability.QUOTE)
        
        if not providers:
            raise ProviderError("No providers available for quotes")
        
        results = {}
        remaining = set(symbols)
        
        for provider in providers:
            if not remaining:
                break
            
            try:
                logger.debug(f"Trying {provider.name} for batch quotes: {remaining}")
                batch_results = await provider.get_quotes(list(remaining))
                results.update(batch_results)
                remaining -= set(batch_results.keys())
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed batch: {e}")
        
        return results
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        """Get historical data with fallback"""
        providers = self._get_available_providers(ProviderCapability.HISTORICAL)
        
        if not providers:
            raise ProviderError("No providers available for historical data")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_historical(symbol, period, interval)
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
            except ProviderError as e:
                last_error = e
            except Exception as e:
                logger.exception(f"Unexpected error from {provider.name}")
                last_error = e
        
        raise last_error or ProviderError("All providers failed")
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Get fundamentals with fallback"""
        providers = self._get_available_providers(ProviderCapability.FUNDAMENTALS)
        
        if not providers:
            raise ProviderError("No providers available for fundamentals")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_fundamentals(symbol)
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
            except ProviderError as e:
                last_error = e
            except Exception as e:
                logger.exception(f"Unexpected error from {provider.name}")
                last_error = e
        
        raise last_error or ProviderError("All providers failed")
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers"""
        results = {}
        for provider in self.providers:
            results[provider.name] = await provider.health_check()
        return results
```

## Command Layer

### Command interface

```python
# src/commands/base.py

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
import re

@dataclass
class CommandContext:
    """Context passed to command handlers"""
    sender: str          # Phone number of sender
    group_id: Optional[str]  # Group ID if in group chat
    raw_message: str     # Original message text
    command: str         # The command (e.g., "price")
    args: list[str]      # Arguments after command
    
    @property
    def is_group(self) -> bool:
        return self.group_id is not None

@dataclass
class CommandResult:
    """Result from command execution"""
    text: str
    success: bool = True
    
    @classmethod
    def error(cls, message: str) -> "CommandResult":
        return cls(text=f"❌ {message}", success=False)
    
    @classmethod
    def ok(cls, message: str) -> "CommandResult":
        return cls(text=message, success=True)

class BaseCommand(ABC):
    """Base class for all commands"""
    
    name: str              # Primary command name (e.g., "price")
    aliases: list[str] = []  # Alternative names (e.g., ["p", "pr"])
    description: str       # Help text
    usage: str            # Usage example
    
    @abstractmethod
    async def execute(self, ctx: CommandContext) -> CommandResult:
        """Execute the command"""
        pass
    
    def matches(self, command: str) -> bool:
        """Check if this handler matches the command"""
        command = command.lower()
        return command == self.name or command in self.aliases
```

### Command implementations

```python
# src/commands/stock_commands.py

from .base import BaseCommand, CommandContext, CommandResult
from ..providers.manager import ProviderManager
from ..providers.base import SymbolNotFoundError, ProviderError

def format_number(n: float | int | None, prefix: str = "") -> str:
    """Format large numbers with K/M/B suffixes"""
    if n is None:
        return "N/A"
    if abs(n) >= 1_000_000_000:
        return f"{prefix}{n/1_000_000_000:.2f}B"
    if abs(n) >= 1_000_000:
        return f"{prefix}{n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"{prefix}{n/1_000:.2f}K"
    if isinstance(n, float):
        return f"{prefix}{n:.2f}"
    return f"{prefix}{n}"

def format_change(change: float, pct: float) -> str:
    """Format price change with arrow"""
    arrow = "📈" if change >= 0 else "📉"
    sign = "+" if change >= 0 else ""
    return f"{arrow} {sign}{change:.2f} ({sign}{pct:.2f}%)"

class PriceCommand(BaseCommand):
    name = "price"
    aliases = ["p", "pr", "$"]
    description = "Get current stock price"
    usage = "!price AAPL [MSFT GOOGL ...]"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbols = [s.upper() for s in ctx.args[:10]]  # Limit to 10
        
        try:
            if len(symbols) == 1:
                quote = await self.providers.get_quote(symbols[0])
                return CommandResult.ok(
                    f"{quote.name or quote.symbol} ({quote.symbol})\n"
                    f"💵 ${quote.price:.2f}\n"
                    f"{format_change(quote.change, quote.change_percent)}\n"
                    f"📊 Vol: {format_number(quote.volume)}"
                )
            else:
                quotes = await self.providers.get_quotes(symbols)
                if not quotes:
                    return CommandResult.error("No quotes found")
                
                lines = []
                for symbol in symbols:
                    if symbol in quotes:
                        q = quotes[symbol]
                        sign = "+" if q.change >= 0 else ""
                        lines.append(f"{q.symbol}: ${q.price:.2f} ({sign}{q.change_percent:.2f}%)")
                    else:
                        lines.append(f"{symbol}: Not found")
                
                return CommandResult.ok("\n".join(lines))
                
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbols[0]}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")

class QuoteCommand(BaseCommand):
    name = "quote"
    aliases = ["q", "detail"]
    description = "Get detailed stock quote"
    usage = "!quote AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            quote = await self.providers.get_quote(symbol)
            
            lines = [
                f"📊 {quote.name or symbol} ({quote.symbol})",
                f"",
                f"Price: ${quote.price:.2f}",
                f"Change: {format_change(quote.change, quote.change_percent)}",
                f"",
                f"Open: ${quote.open:.2f}" if quote.open else None,
                f"High: ${quote.high:.2f}" if quote.high else None,
                f"Low: ${quote.low:.2f}" if quote.low else None,
                f"Prev Close: ${quote.prev_close:.2f}" if quote.prev_close else None,
                f"Volume: {format_number(quote.volume)}",
                f"Market Cap: {format_number(quote.market_cap, '$')}" if quote.market_cap else None,
            ]
            
            return CommandResult.ok("\n".join(l for l in lines if l is not None))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")

class FundamentalsCommand(BaseCommand):
    name = "info"
    aliases = ["i", "fundamentals", "fund"]
    description = "Get company fundamentals"
    usage = "!info AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        symbol = ctx.args[0].upper()
        
        try:
            fund = await self.providers.get_fundamentals(symbol)
            
            lines = [
                f"📋 {fund.name} ({fund.symbol})",
                f"",
                f"Sector: {fund.sector or 'N/A'}",
                f"Industry: {fund.industry or 'N/A'}",
                f"",
                f"P/E Ratio: {fund.pe_ratio:.2f}" if fund.pe_ratio else "P/E: N/A",
                f"EPS: ${fund.eps:.2f}" if fund.eps else "EPS: N/A",
                f"Market Cap: {format_number(fund.market_cap, '$')}" if fund.market_cap else None,
                f"Div Yield: {fund.dividend_yield*100:.2f}%" if fund.dividend_yield else None,
                f"",
                f"52W High: ${fund.fifty_two_week_high:.2f}" if fund.fifty_two_week_high else None,
                f"52W Low: ${fund.fifty_two_week_low:.2f}" if fund.fifty_two_week_low else None,
            ]
            
            return CommandResult.ok("\n".join(l for l in lines if l is not None))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")

class MarketCommand(BaseCommand):
    name = "market"
    aliases = ["m", "indices"]
    description = "Get major market indices"
    usage = "!market"
    
    INDICES = {
        "^GSPC": "S&P 500",
        "^DJI": "Dow Jones",
        "^IXIC": "Nasdaq",
        "^RUT": "Russell 2000",
        "^VIX": "VIX",
    }
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        try:
            quotes = await self.providers.get_quotes(list(self.INDICES.keys()))
            
            lines = ["📈 Market Overview", ""]
            for symbol, name in self.INDICES.items():
                if symbol in quotes:
                    q = quotes[symbol]
                    arrow = "🟢" if q.change >= 0 else "🔴"
                    sign = "+" if q.change >= 0 else ""
                    lines.append(f"{arrow} {name}: {q.price:,.2f} ({sign}{q.change_percent:.2f}%)")
                else:
                    lines.append(f"⚪ {name}: N/A")
            
            return CommandResult.ok("\n".join(lines))
            
        except ProviderError as e:
            return CommandResult.error(f"Market data unavailable: {e}")

class HelpCommand(BaseCommand):
    name = "help"
    aliases = ["h", "?", "commands"]
    description = "Show available commands"
    usage = "!help [command]"
    
    def __init__(self, commands: list[BaseCommand]):
        self.commands = {cmd.name: cmd for cmd in commands}
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if ctx.args:
            # Help for specific command
            cmd_name = ctx.args[0].lower()
            for cmd in self.commands.values():
                if cmd.matches(cmd_name):
                    aliases = f" (aliases: {', '.join(cmd.aliases)})" if cmd.aliases else ""
                    return CommandResult.ok(
                        f"!{cmd.name}{aliases}\n\n"
                        f"{cmd.description}\n\n"
                        f"Usage: {cmd.usage}"
                    )
            return CommandResult.error(f"Unknown command: {cmd_name}")
        
        # General help
        lines = ["📖 Stock Bot Commands", ""]
        for cmd in self.commands.values():
            if cmd.name != "help":
                lines.append(f"!{cmd.name} - {cmd.description}")
        lines.append("")
        lines.append("Type !help <command> for detailed usage")
        
        return CommandResult.ok("\n".join(lines))
```

### Command dispatcher

```python
# src/commands/dispatcher.py

import re
import logging
from typing import Optional
from .base import BaseCommand, CommandContext, CommandResult

logger = logging.getLogger(__name__)

class CommandDispatcher:
    """
    Routes incoming messages to appropriate command handlers.
    Supports configurable prefix and handles unknown commands.
    """
    
    def __init__(self, prefix: str = "!"):
        self.prefix = prefix
        self.commands: dict[str, BaseCommand] = {}
        self._pattern = re.compile(
            rf"^{re.escape(prefix)}(\w+)(?:\s+(.*))?$",
            re.IGNORECASE
        )
    
    def register(self, command: BaseCommand):
        """Register a command handler"""
        self.commands[command.name] = command
        for alias in command.aliases:
            self.commands[alias] = command
        logger.info(f"Registered command: {command.name}")
    
    def parse_message(self, text: str) -> Optional[tuple[str, list[str]]]:
        """Parse message into command and args. Returns None if not a command."""
        match = self._pattern.match(text.strip())
        if not match:
            return None
        
        command = match.group(1).lower()
        args_str = match.group(2) or ""
        args = args_str.split() if args_str else []
        
        return command, args
    
    async def dispatch(
        self,
        sender: str,
        message: str,
        group_id: Optional[str] = None
    ) -> Optional[CommandResult]:
        """
        Dispatch a message to the appropriate command handler.
        Returns None if message is not a command.
        """
        parsed = self.parse_message(message)
        if not parsed:
            return None
        
        command, args = parsed
        
        handler = self.commands.get(command)
        if not handler:
            return CommandResult.error(
                f"Unknown command: {command}\nType {self.prefix}help for available commands"
            )
        
        ctx = CommandContext(
            sender=sender,
            group_id=group_id,
            raw_message=message,
            command=command,
            args=args,
        )
        
        try:
            logger.info(f"Executing {command} from {sender}: {args}")
            return await handler.execute(ctx)
        except Exception as e:
            logger.exception(f"Error executing {command}")
            return CommandResult.error(f"Internal error: {type(e).__name__}")
```

## Signal Integration

### Webhook handler

```python
# src/signal/handler.py

import aiohttp
import logging
from dataclasses import dataclass
from typing import Optional
from ..commands.dispatcher import CommandDispatcher

logger = logging.getLogger(__name__)

@dataclass
class SignalConfig:
    api_url: str
    phone_number: str

class SignalHandler:
    """Handles Signal message sending/receiving via signal-cli-rest-api"""
    
    def __init__(self, config: SignalConfig, dispatcher: CommandDispatcher):
        self.config = config
        self.dispatcher = dispatcher
        self._session: Optional[aiohttp.ClientSession] = None
    
    async def _get_session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession()
        return self._session
    
    async def send_message(
        self,
        recipient: str,
        message: str,
        group_id: Optional[str] = None
    ):
        """Send a message to a recipient or group"""
        session = await self._get_session()
        
        payload = {
            "number": self.config.phone_number,
            "message": message,
        }
        
        if group_id:
            payload["recipients"] = [group_id]
        else:
            payload["recipients"] = [recipient]
        
        async with session.post(
            f"{self.config.api_url}/v2/send",
            json=payload
        ) as resp:
            if resp.status != 201:
                error = await resp.text()
                logger.error(f"Failed to send message: {error}")
                raise Exception(f"Send failed: {resp.status}")
    
    async def handle_webhook(self, data: dict):
        """
        Handle incoming webhook from signal-cli-rest-api.
        
        Expected format:
        {
            "envelope": {
                "source": "+15551234567",
                "sourceDevice": 1,
                "timestamp": 1234567890,
                "dataMessage": {
                    "message": "!price AAPL",
                    "groupInfo": {
                        "groupId": "abc123..."
                    }
                }
            }
        }
        """
        envelope = data.get("envelope", {})
        sender = envelope.get("source")
        data_message = envelope.get("dataMessage", {})
        message_text = data_message.get("message", "")
        
        if not sender or not message_text:
            return
        
        # Extract group ID if present
        group_info = data_message.get("groupInfo")
        group_id = group_info.get("groupId") if group_info else None
        
        # Dispatch to command handler
        result = await self.dispatcher.dispatch(
            sender=sender,
            message=message_text,
            group_id=group_id,
        )
        
        if result:
            await self.send_message(
                recipient=sender,
                message=result.text,
                group_id=group_id,
            )
    
    async def close(self):
        if self._session:
            await self._session.close()
```

### Flask webhook server

```python
# src/server.py

import asyncio
import logging
from flask import Flask, request, jsonify
from .signal.handler import SignalHandler

logger = logging.getLogger(__name__)

def create_app(signal_handler: SignalHandler) -> Flask:
    app = Flask(__name__)
    
    @app.route("/webhook", methods=["POST"])
    def webhook():
        data = request.json
        logger.debug(f"Received webhook: {data}")
        
        # Run async handler in event loop
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(signal_handler.handle_webhook(data))
        finally:
            loop.close()
        
        return jsonify({"status": "ok"})
    
    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})
    
    return app
```

## Configuration

### Config file structure

```python
# src/config.py

import os
from dataclasses import dataclass, field
from typing import Optional
import yaml

@dataclass
class ProviderConfig:
    name: str
    enabled: bool = True
    api_key: Optional[str] = None
    priority: int = 0  # Lower = higher priority

@dataclass
class Config:
    # Signal settings
    signal_api_url: str = "http://localhost:8080"
    signal_phone_number: str = ""
    
    # Bot settings
    command_prefix: str = "!"
    log_level: str = "INFO"
    
    # Provider settings
    providers: list[ProviderConfig] = field(default_factory=list)
    
    @classmethod
    def from_yaml(cls, path: str) -> "Config":
        with open(path) as f:
            data = yaml.safe_load(f)
        
        providers = [
            ProviderConfig(**p) for p in data.pop("providers", [])
        ]
        
        return cls(providers=providers, **data)
    
    @classmethod
    def from_env(cls) -> "Config":
        """Load config from environment variables"""
        providers = []
        
        # Yahoo Finance (no key required)
        if os.getenv("YAHOO_ENABLED", "true").lower() == "true":
            providers.append(ProviderConfig(
                name="yahoo",
                enabled=True,
                priority=int(os.getenv("YAHOO_PRIORITY", "0")),
            ))
        
        # Alpha Vantage
        av_key = os.getenv("ALPHAVANTAGE_API_KEY")
        if av_key:
            providers.append(ProviderConfig(
                name="alphavantage",
                enabled=True,
                api_key=av_key,
                priority=int(os.getenv("ALPHAVANTAGE_PRIORITY", "10")),
            ))
        
        # Polygon
        polygon_key = os.getenv("POLYGON_API_KEY")
        if polygon_key:
            providers.append(ProviderConfig(
                name="polygon",
                enabled=True,
                api_key=polygon_key,
                priority=int(os.getenv("POLYGON_PRIORITY", "5")),
            ))
        
        return cls(
            signal_api_url=os.getenv("SIGNAL_API_URL", "http://localhost:8080"),
            signal_phone_number=os.getenv("SIGNAL_PHONE_NUMBER", ""),
            command_prefix=os.getenv("COMMAND_PREFIX", "!"),
            log_level=os.getenv("LOG_LEVEL", "INFO"),
            providers=sorted(providers, key=lambda p: p.priority),
        )
```

### Example config.yaml

```yaml
# config.yaml

signal_api_url: "http://signal-api:8080"
signal_phone_number: "+15551234567"
command_prefix: "!"
log_level: "INFO"

providers:
  - name: yahoo
    enabled: true
    priority: 0  # Primary - no rate limits
  
  - name: alphavantage
    enabled: true
    api_key: "${ALPHAVANTAGE_API_KEY}"  # Resolved from env
    priority: 10  # Fallback - has rate limits
  
  - name: polygon
    enabled: false
    api_key: "${POLYGON_API_KEY}"
    priority: 5
```

## Testing

### Provider tests

```python
# tests/test_providers.py

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from src.providers.yahoo import YahooFinanceProvider
from src.providers.alphavantage import AlphaVantageProvider
from src.providers.manager import ProviderManager
from src.providers.base import Quote, SymbolNotFoundError, RateLimitError

@pytest.fixture
def yahoo_provider():
    return YahooFinanceProvider()

@pytest.fixture
def av_provider():
    return AlphaVantageProvider(api_key="test_key")

@pytest.fixture
def provider_manager(yahoo_provider, av_provider):
    manager = ProviderManager()
    manager.add_provider(yahoo_provider)
    manager.add_provider(av_provider)
    return manager

class TestYahooProvider:
    @pytest.mark.asyncio
    async def test_get_quote_valid_symbol(self, yahoo_provider):
        quote = await yahoo_provider.get_quote("AAPL")
        
        assert quote.symbol == "AAPL"
        assert quote.price > 0
        assert quote.provider == "yahoo"
    
    @pytest.mark.asyncio
    async def test_get_quote_invalid_symbol(self, yahoo_provider):
        with pytest.raises(SymbolNotFoundError):
            await yahoo_provider.get_quote("INVALIDXYZ123")
    
    @pytest.mark.asyncio
    async def test_get_quotes_batch(self, yahoo_provider):
        quotes = await yahoo_provider.get_quotes(["AAPL", "MSFT", "GOOGL"])
        
        assert len(quotes) == 3
        assert "AAPL" in quotes
        assert all(q.price > 0 for q in quotes.values())
    
    @pytest.mark.asyncio
    async def test_get_historical(self, yahoo_provider):
        bars = await yahoo_provider.get_historical("AAPL", period="5d")
        
        assert len(bars) > 0
        assert all(b.close > 0 for b in bars)
    
    @pytest.mark.asyncio
    async def test_health_check(self, yahoo_provider):
        assert await yahoo_provider.health_check() is True

class TestAlphaVantageProvider:
    @pytest.mark.asyncio
    async def test_get_quote_valid(self, av_provider):
        # Mock the HTTP request
        mock_response = {
            "Global Quote": {
                "01. symbol": "IBM",
                "05. price": "150.00",
                "09. change": "2.50",
                "10. change percent": "1.69%",
                "06. volume": "5000000",
                "02. open": "148.00",
                "03. high": "151.00",
                "04. low": "147.50",
                "08. previous close": "147.50",
            }
        }
        
        with patch.object(av_provider, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            quote = await av_provider.get_quote("IBM")
        
        assert quote.symbol == "IBM"
        assert quote.price == 150.00
        assert quote.change == 2.50
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, av_provider):
        mock_response = {
            "Note": "Thank you for using Alpha Vantage! Our standard API call frequency..."
        }
        
        with patch.object(av_provider, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            mock_req.side_effect = RateLimitError(retry_after=60)
            
            with pytest.raises(RateLimitError):
                await av_provider.get_quote("IBM")

class TestProviderManager:
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self, provider_manager):
        # Make yahoo fail, should fall back to alphavantage
        with patch.object(
            provider_manager.providers[0], 
            'get_quote', 
            new_callable=AsyncMock
        ) as mock_yahoo:
            mock_yahoo.side_effect = Exception("Yahoo failed")
            
            with patch.object(
                provider_manager.providers[1],
                'get_quote',
                new_callable=AsyncMock
            ) as mock_av:
                mock_av.return_value = Quote(
                    symbol="AAPL",
                    price=150.0,
                    change=1.0,
                    change_percent=0.67,
                    volume=1000000,
                    timestamp=MagicMock(),
                    provider="alphavantage"
                )
                
                quote = await provider_manager.get_quote("AAPL")
                
                assert quote.provider == "alphavantage"
                mock_yahoo.assert_called_once()
                mock_av.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_skips_provider(self, provider_manager):
        import time
        
        # Mark yahoo as rate limited
        provider_manager._rate_limited["yahoo"] = time.time() + 3600
        
        with patch.object(
            provider_manager.providers[1],
            'get_quote',
            new_callable=AsyncMock
        ) as mock_av:
            mock_av.return_value = Quote(
                symbol="AAPL",
                price=150.0,
                change=1.0,
                change_percent=0.67,
                volume=1000000,
                timestamp=MagicMock(),
                provider="alphavantage"
            )
            
            quote = await provider_manager.get_quote("AAPL")
            
            # Should skip yahoo entirely
            assert quote.provider == "alphavantage"
```

### Command tests

```python
# tests/test_commands.py

import pytest
from unittest.mock import AsyncMock, MagicMock
from src.commands.stock_commands import PriceCommand, QuoteCommand, MarketCommand
from src.commands.base import CommandContext
from src.providers.base import Quote, SymbolNotFoundError

@pytest.fixture
def mock_provider_manager():
    return MagicMock()

@pytest.fixture
def price_command(mock_provider_manager):
    return PriceCommand(mock_provider_manager)

class TestPriceCommand:
    @pytest.mark.asyncio
    async def test_single_symbol(self, price_command, mock_provider_manager):
        mock_provider_manager.get_quote = AsyncMock(return_value=Quote(
            symbol="AAPL",
            price=150.00,
            change=2.50,
            change_percent=1.69,
            volume=50000000,
            timestamp=MagicMock(),
            provider="yahoo",
            name="Apple Inc.",
        ))
        
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price AAPL",
            command="price",
            args=["AAPL"],
        )
        
        result = await price_command.execute(ctx)
        
        assert result.success
        assert "Apple Inc." in result.text
        assert "$150.00" in result.text
        assert "📈" in result.text  # Positive change
    
    @pytest.mark.asyncio
    async def test_multiple_symbols(self, price_command, mock_provider_manager):
        mock_provider_manager.get_quotes = AsyncMock(return_value={
            "AAPL": Quote(
                symbol="AAPL", price=150.0, change=2.0, change_percent=1.35,
                volume=1000000, timestamp=MagicMock(), provider="yahoo"
            ),
            "MSFT": Quote(
                symbol="MSFT", price=300.0, change=-1.0, change_percent=-0.33,
                volume=2000000, timestamp=MagicMock(), provider="yahoo"
            ),
        })
        
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price AAPL MSFT",
            command="price",
            args=["AAPL", "MSFT"],
        )
        
        result = await price_command.execute(ctx)
        
        assert result.success
        assert "AAPL" in result.text
        assert "MSFT" in result.text
    
    @pytest.mark.asyncio
    async def test_no_args_error(self, price_command):
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price",
            command="price",
            args=[],
        )
        
        result = await price_command.execute(ctx)
        
        assert not result.success
        assert "Usage:" in result.text
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self, price_command, mock_provider_manager):
        mock_provider_manager.get_quote = AsyncMock(
            side_effect=SymbolNotFoundError("Not found")
        )
        
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price INVALID",
            command="price",
            args=["INVALID"],
        )
        
        result = await price_command.execute(ctx)
        
        assert not result.success
        assert "not found" in result.text.lower()
```

### Integration tests

```python
# tests/test_integration.py

import pytest
import asyncio
from src.providers.yahoo import YahooFinanceProvider
from src.providers.manager import ProviderManager
from src.commands.dispatcher import CommandDispatcher
from src.commands.stock_commands import PriceCommand, QuoteCommand, MarketCommand, HelpCommand

@pytest.fixture
def integration_setup():
    """Set up real providers and commands for integration testing"""
    provider_manager = ProviderManager()
    provider_manager.add_provider(YahooFinanceProvider())
    
    dispatcher = CommandDispatcher(prefix="!")
    
    price_cmd = PriceCommand(provider_manager)
    quote_cmd = QuoteCommand(provider_manager)
    market_cmd = MarketCommand(provider_manager)
    
    dispatcher.register(price_cmd)
    dispatcher.register(quote_cmd)
    dispatcher.register(market_cmd)
    dispatcher.register(HelpCommand([price_cmd, quote_cmd, market_cmd]))
    
    return dispatcher

class TestIntegration:
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_full_price_flow(self, integration_setup):
        """Test complete flow from message to response"""
        result = await integration_setup.dispatch(
            sender="+15551234567",
            message="!price AAPL",
            group_id=None,
        )
        
        assert result is not None
        assert result.success
        assert "$" in result.text
    
    @pytest.mark.asyncio
    @pytest.mark.integration
    async def test_market_command(self, integration_setup):
        """Test market overview command"""
        result = await integration_setup.dispatch(
            sender="+15551234567",
            message="!market",
            group_id=None,
        )
        
        assert result is not None
        assert result.success
        assert "S&P 500" in result.text
    
    @pytest.mark.asyncio
    async def test_help_command(self, integration_setup):
        """Test help command"""
        result = await integration_setup.dispatch(
            sender="+15551234567",
            message="!help",
            group_id=None,
        )
        
        assert result is not None
        assert result.success
        assert "price" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_unknown_command(self, integration_setup):
        """Test handling of unknown commands"""
        result = await integration_setup.dispatch(
            sender="+15551234567",
            message="!notacommand",
            group_id=None,
        )
        
        assert result is not None
        assert not result.success
        assert "unknown" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_non_command_ignored(self, integration_setup):
        """Test that non-commands are ignored"""
        result = await integration_setup.dispatch(
            sender="+15551234567",
            message="Just a regular message",
            group_id=None,
        )
        
        assert result is None
```

### Test configuration

```ini
# pytest.ini

[pytest]
asyncio_mode = auto
markers =
    integration: marks tests as integration tests (deselect with '-m "not integration"')
testpaths = tests
python_files = test_*.py
python_functions = test_*
addopts = -v --tb=short
```

```txt
# requirements-test.txt

pytest>=7.0
pytest-asyncio>=0.21
pytest-cov>=4.0
pytest-mock>=3.10
```

## Project Structure

```
signal-stock-bot/
├── docker-compose.yml
├── Dockerfile
├── config.yaml
├── .env.example
├── requirements.txt
├── requirements-test.txt
├── pytest.ini
├── README.md
├── docs/
│   ├── INFRASTRUCTURE.md
│   └── DESIGN.md
├── src/
│   ├── __init__.py
│   ├── main.py
│   ├── config.py
│   ├── server.py
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── manager.py
│   │   ├── yahoo.py
│   │   ├── alphavantage.py
│   │   └── polygon.py
│   ├── commands/
│   │   ├── __init__.py
│   │   ├── base.py
│   │   ├── dispatcher.py
│   │   └── stock_commands.py
│   └── signal/
│       ├── __init__.py
│       └── handler.py
└── tests/
    ├── __init__.py
    ├── conftest.py
    ├── test_providers.py
    ├── test_commands.py
    └── test_integration.py
```

## Main entrypoint

```python
# src/main.py

import logging
import sys
from .config import Config
from .providers.manager import ProviderManager
from .providers.yahoo import YahooFinanceProvider
from .providers.alphavantage import AlphaVantageProvider
from .commands.dispatcher import CommandDispatcher
from .commands.stock_commands import (
    PriceCommand, QuoteCommand, FundamentalsCommand, MarketCommand, HelpCommand
)
from .signal.handler import SignalHandler, SignalConfig
from .server import create_app

def setup_logging(level: str):
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/bot.log"),
        ]
    )

def create_provider_manager(config: Config) -> ProviderManager:
    manager = ProviderManager()
    
    for provider_config in config.providers:
        if not provider_config.enabled:
            continue
        
        if provider_config.name == "yahoo":
            manager.add_provider(YahooFinanceProvider())
        elif provider_config.name == "alphavantage" and provider_config.api_key:
            manager.add_provider(AlphaVantageProvider(provider_config.api_key))
        # Add more providers here
    
    return manager

def main():
    # Load config
    config = Config.from_env()
    setup_logging(config.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Signal Stock Bot")
    
    # Set up providers
    provider_manager = create_provider_manager(config)
    
    # Set up commands
    dispatcher = CommandDispatcher(prefix=config.command_prefix)
    
    price_cmd = PriceCommand(provider_manager)
    quote_cmd = QuoteCommand(provider_manager)
    info_cmd = FundamentalsCommand(provider_manager)
    market_cmd = MarketCommand(provider_manager)
    
    dispatcher.register(price_cmd)
    dispatcher.register(quote_cmd)
    dispatcher.register(info_cmd)
    dispatcher.register(market_cmd)
    dispatcher.register(HelpCommand([price_cmd, quote_cmd, info_cmd, market_cmd]))
    
    # Set up Signal handler
    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    
    # Create and run Flask app
    app = create_app(signal_handler)
    app.run(host="0.0.0.0", port=5000)

if __name__ == "__main__":
    main()
```

## Deployment files

### Dockerfile

```dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY config.yaml .

# Create log directory
RUN mkdir -p logs

# Run
CMD ["python", "-m", "src.main"]
```

### requirements.txt

```txt
flask>=3.0
aiohttp>=3.9
yfinance>=0.2
pyyaml>=6.0
gunicorn>=21.0
```

### .env.example

```bash
# Signal Configuration
SIGNAL_API_URL=http://signal-api:8080
SIGNAL_PHONE_NUMBER=+15551234567

# Bot Configuration
COMMAND_PREFIX=!
LOG_LEVEL=INFO

# Provider API Keys (optional)
ALPHAVANTAGE_API_KEY=your_key_here
POLYGON_API_KEY=your_key_here

# Provider Priorities (lower = higher priority)
YAHOO_PRIORITY=0
ALPHAVANTAGE_PRIORITY=10
POLYGON_PRIORITY=5
```

## Commands Reference

| Command | Aliases | Description | Example |
|---------|---------|-------------|---------|
| `!price` | `!p`, `!$` | Get current price(s) | `!price AAPL MSFT` |
| `!quote` | `!q`, `!detail` | Detailed quote | `!quote TSLA` |
| `!info` | `!i`, `!fund` | Company fundamentals | `!info NVDA` |
| `!market` | `!m`, `!indices` | Major indices overview | `!market` |
| `!help` | `!h`, `!?` | Show help | `!help price` |

## Extension Points

### Adding a new provider

1. Create `src/providers/newprovider.py` implementing `BaseProvider`
2. Add provider config handling in `src/config.py`
3. Register in `create_provider_manager()` in `src/main.py`
4. Add tests in `tests/test_providers.py`

### Adding a new command

1. Create command class in `src/commands/stock_commands.py` extending `BaseCommand`
2. Register in `main()` with `dispatcher.register()`
3. Add to `HelpCommand` list
4. Add tests in `tests/test_commands.py`

## Maintenance

- **Update signal-cli monthly** (protocol changes)
- **Monitor rate limits** via logs
- **Rotate API keys** periodically
- **Review provider health** with `/health` endpoint
