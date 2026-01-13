import pytest
import re
from unittest.mock import MagicMock, AsyncMock, patch
from src.commands.intent_parser import parse_intent, extract_symbols_from_text, Intent
from src.commands.dispatcher import CommandDispatcher, CommandResult

# --- Comparison Logic ---

def test_intent_comparison():
    # "Chart Apple vs Tesla"
    text = "Chart Apple vs Tesla"
    intent = parse_intent(text)
    
    assert intent.command == "chart"
    assert "AAPL" in intent.symbols
    assert "TSLA" in intent.symbols
    assert "-compare" in intent.args
    
    # "Compare AAPL to MSFT"
    text2 = "Compare AAPL to MSFT"
    intent2 = parse_intent(text2)
        
    assert intent2.command == "chart"
    assert "AAPL" in intent2.symbols
    assert "MSFT" in intent2.symbols
    assert "-compare" in intent2.args

# --- Robust Splitting Logic ---

@pytest.mark.asyncio
async def test_robust_splitting():
    dispatcher = CommandDispatcher()
    # Mock execute to return success
    dispatcher._execute_command = AsyncMock(return_value=CommandResult.ok("Done"))
    
    # Mock context
    dispatcher.context_manager = MagicMock()
    mock_ctx = MagicMock()
    mock_ctx.last_symbol = "AAPL"
    dispatcher.context_manager.get_context = AsyncMock(return_value=mock_ctx)
    dispatcher.context_manager.update_context = AsyncMock()
    
    # 1. Basic splitting (Period)
    # "Chart AAPL. Show RSI" -> Should split into 2 commands.
    msg = "Chart AAPL. Show RSI"
    await dispatcher.dispatch("u1", msg)
    assert dispatcher._execute_command.call_count >= 2, f"Expected at least 2 calls, got {dispatcher._execute_command.call_count}"

    # 2. "And" handling
    dispatcher._execute_command.reset_mock()
    msg2 = "Chart Apple and show RSI"
    await dispatcher.dispatch("u1", msg2)
    assert dispatcher._execute_command.call_count >= 2, f"Expected at least 2 calls for 'and', got {dispatcher._execute_command.call_count}"

    # 3. Multiple delimiters
    dispatcher._execute_command.reset_mock()
    msg3 = "Chart Apple! Show RSI? Do it."
    await dispatcher.dispatch("u1", msg3)
    assert dispatcher._execute_command.call_count >= 2, f"Expected at least 2 calls for multiple delimiters, got {dispatcher._execute_command.call_count}"

# --- List Handling ---

def test_symbol_list_with_commas():
    symbols = extract_symbols_from_text("Price of AAPL, MSFT, and GOOG")
    assert "AAPL" in symbols
    assert "MSFT" in symbols
    assert "GOOG" in symbols
