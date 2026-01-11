"""
Tests for bot commands.
"""

import pytest
from datetime import datetime

from src.commands import (
    CommandDispatcher,
    CommandContext,
    PriceCommand,
    QuoteCommand,
    MarketCommand,
    HelpCommand,
)
from src.providers import Quote, Fundamentals, SymbolNotFoundError


class TestCommandDispatcher:
    """Tests for command dispatcher"""
    
    def test_parse_simple_command(self, dispatcher):
        """Test parsing a simple command"""
        result = dispatcher.parse_message("!price AAPL")
        
        assert result is not None
        command, args = result
        assert command == "price"
        assert args == ["AAPL"]
    
    def test_parse_command_multiple_args(self, dispatcher):
        """Test parsing command with multiple arguments"""
        result = dispatcher.parse_message("!price AAPL MSFT GOOGL")
        
        assert result is not None
        command, args = result
        assert command == "price"
        assert args == ["AAPL", "MSFT", "GOOGL"]
    
    def test_parse_command_no_args(self, dispatcher):
        """Test parsing command without arguments"""
        result = dispatcher.parse_message("!market")
        
        assert result is not None
        command, args = result
        assert command == "market"
        assert args == []
    
    def test_parse_non_command(self, dispatcher):
        """Test that non-commands return None"""
        result = dispatcher.parse_message("Just a regular message")
        assert result is None
    
    def test_parse_different_prefix(self):
        """Test command with different prefix"""
        d = CommandDispatcher(prefix="/")
        result = d.parse_message("/price AAPL")
        
        assert result is not None
        command, args = result
        assert command == "price"
    
    @pytest.mark.asyncio
    async def test_dispatch_unknown_command(self, dispatcher):
        """Test dispatching unknown command"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!unknowncommand",
        )
        
        assert result is not None
        assert not result.success
        assert "unknown" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_dispatch_non_command(self, dispatcher):
        """Test dispatching non-command returns None"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="Hello there",
        )
        
        assert result is None


class TestPriceCommand:
    """Tests for price command"""
    
    @pytest.mark.asyncio
    async def test_single_symbol(self, mock_provider_manager, sample_quote):
        """Test price command with single symbol"""
        mock_provider_manager.get_quote.return_value = sample_quote
        
        cmd = PriceCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price AAPL",
            command="price",
            args=["AAPL"],
        )
        
        result = await cmd.execute(ctx)
        
        assert result.success
        assert "Apple Inc." in result.text
        assert "$185.92" in result.text
        assert "📈" in result.text  # Positive change
    
    @pytest.mark.asyncio
    async def test_multiple_symbols(self, mock_provider_manager, sample_quotes):
        """Test price command with multiple symbols"""
        mock_provider_manager.get_quotes.return_value = sample_quotes
        
        cmd = PriceCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price AAPL MSFT GOOGL",
            command="price",
            args=["AAPL", "MSFT", "GOOGL"],
        )
        
        result = await cmd.execute(ctx)
        
        assert result.success
        assert "AAPL" in result.text
        assert "MSFT" in result.text
        assert "GOOGL" in result.text
    
    @pytest.mark.asyncio
    async def test_no_args(self, mock_provider_manager):
        """Test price command without arguments"""
        cmd = PriceCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price",
            command="price",
            args=[],
        )
        
        result = await cmd.execute(ctx)
        
        assert not result.success
        assert "Usage" in result.text
    
    @pytest.mark.asyncio
    async def test_invalid_symbol(self, mock_provider_manager):
        """Test price command with invalid symbol"""
        mock_provider_manager.get_quote.side_effect = SymbolNotFoundError("Not found")
        
        cmd = PriceCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price INVALID",
            command="price",
            args=["INVALID"],
        )
        
        result = await cmd.execute(ctx)
        
        assert not result.success
        assert "not found" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_negative_change(self, mock_provider_manager):
        """Test price display with negative change"""
        quote = Quote(
            symbol="TSLA",
            price=248.50,
            change=-5.20,
            change_percent=-2.05,
            volume=98000000,
            timestamp=datetime.now(),
            provider="yahoo",
            name="Tesla, Inc.",
        )
        mock_provider_manager.get_quote.return_value = quote
        
        cmd = PriceCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!price TSLA",
            command="price",
            args=["TSLA"],
        )
        
        result = await cmd.execute(ctx)
        
        assert result.success
        assert "📉" in result.text  # Negative change indicator


class TestQuoteCommand:
    """Tests for quote command"""
    
    @pytest.mark.asyncio
    async def test_detailed_quote(self, mock_provider_manager, sample_quote):
        """Test detailed quote display"""
        mock_provider_manager.get_quote.return_value = sample_quote
        
        cmd = QuoteCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!quote AAPL",
            command="quote",
            args=["AAPL"],
        )
        
        result = await cmd.execute(ctx)
        
        assert result.success
        assert "Open:" in result.text
        assert "High:" in result.text
        assert "Low:" in result.text
        assert "Volume:" in result.text


class TestMarketCommand:
    """Tests for market command"""
    
    @pytest.mark.asyncio
    async def test_market_overview(self, mock_provider_manager):
        """Test market overview display"""
        now = datetime.now()
        mock_provider_manager.get_quotes.return_value = {
            "^GSPC": Quote(
                symbol="^GSPC", price=5123.41, change=38.25, change_percent=0.75,
                volume=0, timestamp=now, provider="yahoo", name="S&P 500"
            ),
            "^DJI": Quote(
                symbol="^DJI", price=38654.42, change=196.00, change_percent=0.51,
                volume=0, timestamp=now, provider="yahoo", name="Dow Jones"
            ),
        }
        
        cmd = MarketCommand(mock_provider_manager)
        ctx = CommandContext(
            sender="+15551234567",
            group_id=None,
            raw_message="!market",
            command="market",
            args=[],
        )
        
        result = await cmd.execute(ctx)
        
        assert result.success
        assert "Market Overview" in result.text
        assert "S&P 500" in result.text


class TestHelpCommand:
    """Tests for help command"""
    
    @pytest.mark.asyncio
    async def test_general_help(self, dispatcher):
        """Test general help output"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!help",
        )
        
        assert result is not None
        assert result.success
        assert "price" in result.text.lower()
        assert "quote" in result.text.lower()
        assert "market" in result.text.lower()
    
    @pytest.mark.asyncio
    async def test_specific_command_help(self, dispatcher):
        """Test help for specific command"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!help price",
        )
        
        assert result is not None
        assert result.success
        assert "price" in result.text.lower()
        assert "Usage" in result.text
    
    @pytest.mark.asyncio
    async def test_help_unknown_command(self, dispatcher):
        """Test help for unknown command"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!help unknowncmd",
        )
        
        assert result is not None
        assert not result.success


class TestCommandAliases:
    """Tests for command aliases"""
    
    @pytest.mark.asyncio
    async def test_price_alias_p(self, dispatcher, mock_provider_manager, sample_quote):
        """Test !p alias for price"""
        mock_provider_manager.get_quote.return_value = sample_quote
        
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!p AAPL",
        )
        
        assert result is not None
        assert result.success
    
    @pytest.mark.asyncio
    async def test_quote_alias_q(self, dispatcher, mock_provider_manager, sample_quote):
        """Test !q alias for quote"""
        mock_provider_manager.get_quote.return_value = sample_quote
        
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!q AAPL",
        )
        
        assert result is not None
        assert result.success
    
    @pytest.mark.asyncio
    async def test_market_alias_m(self, dispatcher, mock_provider_manager):
        """Test !m alias for market"""
        mock_provider_manager.get_quotes.return_value = {}
        
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!m",
        )
        
        assert result is not None
        # Even with empty quotes, command should execute
    
    @pytest.mark.asyncio
    async def test_help_alias_question(self, dispatcher):
        """Test !? alias for help"""
        result = await dispatcher.dispatch(
            sender="+15551234567",
            message="!?",
        )
        
        assert result is not None
        assert result.success
