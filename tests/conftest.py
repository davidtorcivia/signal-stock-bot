"""
Pytest fixtures for Signal Stock Bot tests.
"""

import pytest
from unittest.mock import MagicMock, AsyncMock
from datetime import datetime

from src.providers import (
    ProviderManager,
    YahooFinanceProvider,
    Quote,
)
from src.commands import (
    CommandDispatcher,
    PriceCommand,
    QuoteCommand,
    FundamentalsCommand,
    MarketCommand,
    HelpCommand,
    StatusCommand,
)


@pytest.fixture
def sample_quote():
    """Sample quote for testing"""
    return Quote(
        symbol="AAPL",
        price=185.92,
        change=2.34,
        change_percent=1.27,
        volume=52300000,
        timestamp=datetime.now(),
        provider="yahoo",
        name="Apple Inc.",
        open=184.00,
        high=186.50,
        low=183.75,
        prev_close=183.58,
        market_cap=2890000000000,
    )


@pytest.fixture
def sample_quotes():
    """Sample batch quotes for testing"""
    now = datetime.now()
    return {
        "AAPL": Quote(
            symbol="AAPL", price=185.92, change=2.34, change_percent=1.27,
            volume=52300000, timestamp=now, provider="yahoo", name="Apple Inc."
        ),
        "MSFT": Quote(
            symbol="MSFT", price=378.91, change=3.35, change_percent=0.89,
            volume=22100000, timestamp=now, provider="yahoo", name="Microsoft Corporation"
        ),
        "GOOGL": Quote(
            symbol="GOOGL", price=141.80, change=-0.45, change_percent=-0.32,
            volume=18500000, timestamp=now, provider="yahoo", name="Alphabet Inc."
        ),
    }


@pytest.fixture
def mock_provider_manager():
    """Mock provider manager for command testing"""
    manager = MagicMock(spec=ProviderManager)
    manager.get_quote = AsyncMock()
    manager.get_quotes = AsyncMock()
    manager.get_fundamentals = AsyncMock()
    manager.get_historical = AsyncMock()
    manager.health_check = AsyncMock(return_value={"yahoo": True})
    manager.get_status = MagicMock(return_value={
        "yahoo": {
            "capabilities": ["quote", "historical", "fundamentals"],
            "rate_limited": False,
            "rate_limit_remaining_seconds": 0,
        }
    })
    return manager


@pytest.fixture
def yahoo_provider():
    """Real Yahoo Finance provider for integration tests"""
    return YahooFinanceProvider()


@pytest.fixture
def provider_manager(yahoo_provider):
    """Provider manager with Yahoo Finance for integration tests"""
    manager = ProviderManager()
    manager.add_provider(yahoo_provider)
    return manager


@pytest.fixture
def dispatcher(mock_provider_manager):
    """Command dispatcher with mock provider"""
    d = CommandDispatcher(prefix="!")
    
    price_cmd = PriceCommand(mock_provider_manager)
    quote_cmd = QuoteCommand(mock_provider_manager)
    info_cmd = FundamentalsCommand(mock_provider_manager)
    market_cmd = MarketCommand(mock_provider_manager)
    status_cmd = StatusCommand(mock_provider_manager)
    
    d.register(price_cmd)
    d.register(quote_cmd)
    d.register(info_cmd)
    d.register(market_cmd)
    d.register(status_cmd)
    d.register(HelpCommand([price_cmd, quote_cmd, info_cmd, market_cmd, status_cmd]))
    
    return d


@pytest.fixture
def integration_dispatcher(provider_manager):
    """Command dispatcher with real provider for integration tests"""
    d = CommandDispatcher(prefix="!")
    
    price_cmd = PriceCommand(provider_manager)
    quote_cmd = QuoteCommand(provider_manager)
    info_cmd = FundamentalsCommand(provider_manager)
    market_cmd = MarketCommand(provider_manager)
    status_cmd = StatusCommand(provider_manager)
    
    d.register(price_cmd)
    d.register(quote_cmd)
    d.register(info_cmd)
    d.register(market_cmd)
    d.register(status_cmd)
    d.register(HelpCommand([price_cmd, quote_cmd, info_cmd, market_cmd, status_cmd]))
    
    return d
