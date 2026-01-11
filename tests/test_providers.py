"""
Tests for financial data providers.
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime

from src.providers import (
    YahooFinanceProvider,
    AlphaVantageProvider,
    ProviderManager,
    Quote,
    SymbolNotFoundError,
    RateLimitError,
    ProviderCapability,
)


class TestYahooFinanceProvider:
    """Tests for Yahoo Finance provider"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_quote_valid_symbol(self, yahoo_provider):
        """Test fetching a valid stock quote"""
        quote = await yahoo_provider.get_quote("AAPL")
        
        assert quote.symbol == "AAPL"
        assert quote.price > 0
        assert quote.provider == "yahoo"
        assert isinstance(quote.volume, int)
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_quote_with_name(self, yahoo_provider):
        """Test that quote includes company name"""
        quote = await yahoo_provider.get_quote("MSFT")
        
        assert quote.name is not None
        assert "Microsoft" in quote.name or quote.symbol == "MSFT"
    
    @pytest.mark.asyncio
    async def test_get_quote_invalid_symbol(self, yahoo_provider):
        """Test that invalid symbol raises SymbolNotFoundError"""
        with pytest.raises(SymbolNotFoundError):
            await yahoo_provider.get_quote("INVALIDXYZ123456")
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_quotes_batch(self, yahoo_provider):
        """Test batch quote fetching"""
        symbols = ["AAPL", "MSFT", "GOOGL"]
        quotes = await yahoo_provider.get_quotes(symbols)
        
        assert len(quotes) >= 2  # At least most should succeed
        for symbol, quote in quotes.items():
            assert quote.price > 0
            assert quote.provider == "yahoo"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_historical(self, yahoo_provider):
        """Test historical data fetching"""
        bars = await yahoo_provider.get_historical("AAPL", period="5d")
        
        assert len(bars) > 0
        for bar in bars:
            assert bar.close > 0
            assert bar.volume >= 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_get_fundamentals(self, yahoo_provider):
        """Test fundamentals fetching"""
        fund = await yahoo_provider.get_fundamentals("AAPL")
        
        assert fund.symbol == "AAPL"
        assert fund.name is not None
        assert fund.provider == "yahoo"
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_health_check(self, yahoo_provider):
        """Test health check"""
        result = await yahoo_provider.health_check()
        assert result is True
    
    def test_capabilities(self, yahoo_provider):
        """Test provider capabilities"""
        assert ProviderCapability.QUOTE in yahoo_provider.capabilities
        assert ProviderCapability.HISTORICAL in yahoo_provider.capabilities
        assert ProviderCapability.FUNDAMENTALS in yahoo_provider.capabilities


class TestAlphaVantageProvider:
    """Tests for Alpha Vantage provider (mocked)"""
    
    @pytest.fixture
    def av_provider(self):
        return AlphaVantageProvider(api_key="test_key")
    
    @pytest.mark.asyncio
    async def test_get_quote_success(self, av_provider):
        """Test successful quote fetch"""
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
        assert quote.change_percent == 1.69
    
    @pytest.mark.asyncio
    async def test_get_quote_not_found(self, av_provider):
        """Test symbol not found handling"""
        mock_response = {"Global Quote": {}}
        
        with patch.object(av_provider, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.return_value = mock_response
            
            with pytest.raises(SymbolNotFoundError):
                await av_provider.get_quote("INVALID")
    
    @pytest.mark.asyncio
    async def test_rate_limit_handling(self, av_provider):
        """Test rate limit detection"""
        with patch.object(av_provider, '_request', new_callable=AsyncMock) as mock_req:
            mock_req.side_effect = RateLimitError(retry_after=60)
            
            with pytest.raises(RateLimitError) as exc_info:
                await av_provider.get_quote("IBM")
            
            assert exc_info.value.retry_after == 60


class TestProviderManager:
    """Tests for provider manager"""
    
    @pytest.mark.asyncio
    async def test_single_provider_success(self):
        """Test with single provider succeeding"""
        manager = ProviderManager(enable_cache=False)
        
        mock_provider = MagicMock()
        mock_provider.name = "mock"
        mock_provider.capabilities = {ProviderCapability.QUOTE}
        mock_provider.get_quote = AsyncMock(return_value=Quote(
            symbol="AAPL",
            price=150.0,
            change=1.0,
            change_percent=0.67,
            volume=1000000,
            timestamp=datetime.now(),
            provider="mock"
        ))
        
        manager.add_provider(mock_provider)
        quote = await manager.get_quote("AAPL")
        
        assert quote.symbol == "AAPL"
        assert quote.provider == "mock"
    
    @pytest.mark.asyncio
    async def test_fallback_on_failure(self):
        """Test fallback when first provider fails"""
        manager = ProviderManager(enable_cache=False)
        
        # First provider fails
        mock_provider1 = MagicMock()
        mock_provider1.name = "failing"
        mock_provider1.capabilities = {ProviderCapability.QUOTE}
        mock_provider1.get_quote = AsyncMock(side_effect=Exception("Failed"))
        
        # Second provider succeeds
        mock_provider2 = MagicMock()
        mock_provider2.name = "working"
        mock_provider2.capabilities = {ProviderCapability.QUOTE}
        mock_provider2.get_quote = AsyncMock(return_value=Quote(
            symbol="AAPL",
            price=150.0,
            change=1.0,
            change_percent=0.67,
            volume=1000000,
            timestamp=datetime.now(),
            provider="working"
        ))
        
        manager.add_provider(mock_provider1)
        manager.add_provider(mock_provider2)
        
        quote = await manager.get_quote("AAPL")
        
        assert quote.provider == "working"
        # With retry logic, first provider may be called multiple times
        assert mock_provider1.get_quote.call_count >= 1
        mock_provider2.get_quote.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_rate_limit_skips_provider(self):
        """Test that rate-limited providers are skipped"""
        import time
        
        manager = ProviderManager(enable_cache=False)
        
        mock_provider1 = MagicMock()
        mock_provider1.name = "rate_limited"
        mock_provider1.capabilities = {ProviderCapability.QUOTE}
        
        mock_provider2 = MagicMock()
        mock_provider2.name = "available"
        mock_provider2.capabilities = {ProviderCapability.QUOTE}
        mock_provider2.get_quote = AsyncMock(return_value=Quote(
            symbol="AAPL",
            price=150.0,
            change=1.0,
            change_percent=0.67,
            volume=1000000,
            timestamp=datetime.now(),
            provider="available"
        ))
        
        manager.add_provider(mock_provider1)
        manager.add_provider(mock_provider2)
        
        # Mark first provider as rate limited
        manager._rate_limited["rate_limited"] = time.time() + 3600
        
        quote = await manager.get_quote("AAPL")
        
        assert quote.provider == "available"
        # First provider should not be called
        mock_provider1.get_quote.assert_not_called()
    
    @pytest.mark.asyncio
    async def test_no_providers_raises_error(self):
        """Test error when no providers available"""
        manager = ProviderManager(enable_cache=False)
        
        with pytest.raises(Exception) as exc_info:
            await manager.get_quote("AAPL")
        
        assert "No providers available" in str(exc_info.value)
    
    def test_get_status(self):
        """Test status reporting"""
        manager = ProviderManager()
        
        mock_provider = MagicMock()
        mock_provider.name = "test"
        mock_provider.capabilities = {ProviderCapability.QUOTE}
        
        manager.add_provider(mock_provider)
        status = manager.get_status()
        
        assert "test" in status
        assert "capabilities" in status["test"]
        assert "rate_limited" in status["test"]
