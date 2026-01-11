"""
Base provider interface for financial data sources.

All providers implement this interface, enabling:
- Consistent data structures across providers
- Automatic fallback when providers fail
- Easy testing via mocking
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from datetime import datetime
from enum import Enum


class ProviderCapability(Enum):
    """Capabilities that providers may support"""
    QUOTE = "quote"
    HISTORICAL = "historical"
    FUNDAMENTALS = "fundamentals"
    OPTIONS = "options"
    FUTURES = "futures"
    FOREX = "forex"
    ECONOMY = "economy"
    NEWS = "news"
    CRYPTO = "crypto"


@dataclass
class Quote:
    """Current quote data for a symbol"""
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
    """Single OHLCV bar"""
    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: int


@dataclass
class Fundamentals:
    """Company fundamental data"""
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


@dataclass
class OptionQuote:
    """Quote for an options contract"""
    symbol: str  # Contract symbol (e.g. O:TSLA230120C00150000)
    underlying: str
    expiration: datetime
    strike: float
    type: str  # "call" or "put"
    price: float
    change: float
    change_percent: float
    volume: int
    open_interest: int
    implied_volatility: Optional[float]
    greeks: Optional[dict[str, float]]
    timestamp: datetime
    provider: str


@dataclass
class ForexQuote:
    """Quote for a forex pair"""
    symbol: str  # e.g. EUR/USD
    rate: float
    change: float
    change_percent: float
    timestamp: datetime
    provider: str
    bid: Optional[float] = None
    ask: Optional[float] = None


@dataclass
class FuturesQuote:
    """Quote for a futures contract"""
    symbol: str
    price: float
    change: float
    change_percent: float
    volume: int
    open_interest: int
    expiration: Optional[datetime]
    timestamp: datetime
    provider: str


@dataclass
class EconomyIndicator:
    """Economic indicator data"""
    name: str  # e.g. CPI, GDP
    value: float
    unit: str
    date: datetime  # Date of the value
    period: str  # "monthly", "quarterly", etc.
    provider: str
    previous: Optional[float] = None  # Previous period value


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
    """
    Abstract base for all financial data providers.
    
    Subclasses must implement:
    - get_quote()
    - get_quotes()
    - health_check()
    
    Optional overrides:
    - get_historical()
    - get_fundamentals()
    - get_option_quote()
    - get_forex_quote()
    - get_future_quote()
    - get_economy_data()
    """
    
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

    async def get_option_quote(self, symbol: str) -> OptionQuote:
        """Get option quote. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support options")

    async def get_forex_quote(self, symbol: str) -> ForexQuote:
        """Get forex quote. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support forex")

    async def get_future_quote(self, symbol: str) -> FuturesQuote:
        """Get futures quote. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support futures")
    
    async def get_economy_data(self, indicator: str) -> EconomyIndicator:
        """Get economic data. Override if supported."""
        raise NotImplementedError(f"{self.name} doesn't support economy data")
    
    @abstractmethod
    async def health_check(self) -> bool:
        """Check if provider is operational"""
        pass
