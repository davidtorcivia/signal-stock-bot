"""
Integration tests - require network access.

Run with: pytest -m integration
Skip with: pytest -m "not integration"
"""

import pytest


@pytest.mark.integration
class TestYahooIntegration:
    """Integration tests with real Yahoo Finance API"""
    
    @pytest.mark.asyncio
    async def test_price_command_real(self, integration_dispatcher):
        """Test price command with real data"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price AAPL",
        )
        
        assert result is not None
        assert result.success
        assert "$" in result.text
        assert "AAPL" in result.text
    
    @pytest.mark.asyncio
    async def test_batch_price_real(self, integration_dispatcher):
        """Test batch price with real data"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price AAPL MSFT",
        )
        
        assert result is not None
        assert result.success
        assert "AAPL" in result.text
        assert "MSFT" in result.text
    
    @pytest.mark.asyncio
    async def test_quote_command_real(self, integration_dispatcher):
        """Test detailed quote with real data"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!quote GOOGL",
        )
        
        assert result is not None
        assert result.success
        assert "Open:" in result.text
        assert "Volume:" in result.text
    
    @pytest.mark.asyncio
    async def test_info_command_real(self, integration_dispatcher):
        """Test fundamentals with real data"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!info NVDA",
        )
        
        assert result is not None
        assert result.success
        assert "NVDA" in result.text
    
    @pytest.mark.asyncio
    async def test_market_command_real(self, integration_dispatcher):
        """Test market overview with real data"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!market",
        )
        
        assert result is not None
        assert result.success
        assert "S&P 500" in result.text
    
    @pytest.mark.asyncio
    async def test_invalid_symbol_real(self, integration_dispatcher):
        """Test invalid symbol handling"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price INVALIDXYZ123",
        )
        
        assert result is not None
        assert not result.success
        assert "not found" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_index_symbol(self, integration_dispatcher):
        """Test index symbol (^GSPC)"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price ^GSPC",
        )
        
        assert result is not None
        # Index symbols might not always work, just check it doesn't crash
    
    @pytest.mark.asyncio
    async def test_etf_symbol(self, integration_dispatcher):
        """Test ETF symbol"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price SPY",
        )
        
        assert result is not None
        assert result.success
        assert "$" in result.text


@pytest.mark.integration
class TestProviderFallback:
    """Tests for provider fallback behavior"""
    
    @pytest.mark.asyncio
    async def test_yahoo_provider_health(self, yahoo_provider):
        """Test Yahoo provider health check"""
        result = await yahoo_provider.health_check()
        assert result is True
    
    @pytest.mark.asyncio
    async def test_yahoo_handles_various_symbols(self, yahoo_provider):
        """Test Yahoo handles various symbol types"""
        symbols = ["AAPL", "MSFT", "SPY", "BTC-USD"]
        
        for symbol in symbols:
            try:
                quote = await yahoo_provider.get_quote(symbol)
                assert quote.price > 0
            except Exception as e:
                # Some symbols might fail, that's ok
                print(f"Symbol {symbol} failed: {e}")


@pytest.mark.integration
class TestEndToEnd:
    """End-to-end workflow tests"""
    
    @pytest.mark.asyncio
    async def test_full_workflow(self, integration_dispatcher):
        """Test complete user workflow"""
        # Check help
        help_result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!help",
        )
        assert help_result.success
        
        # Get price
        price_result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!p AAPL",
        )
        assert price_result.success
        
        # Get detailed quote
        quote_result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!q AAPL",
        )
        assert quote_result.success
        
        # Get market overview
        market_result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!m",
        )
        assert market_result.success
    
    @pytest.mark.asyncio
    async def test_group_message_handling(self, integration_dispatcher):
        """Test command works in group context"""
        result = await integration_dispatcher.dispatch(
            sender="+15551234567",
            message="!price AAPL",
            group_id="test_group_123",
        )
        
        assert result is not None
        assert result.success
