import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from src.commands.intent_parser import parse_intent, extract_symbols_from_text, Intent
from src.context import ContextManager
from src.commands.dispatcher import CommandDispatcher, CommandResult

# --- Context Manager Tests ---

@pytest.mark.asyncio
async def test_context_manager(tmp_path):
    db_file = tmp_path / "test_context.db"
    manager = ContextManager(str(db_file))
    
    # Update context
    user_hash = "hash123"
    await manager.update_context(user_hash, symbol="AAPL", intent="chart")
    
    # Retrieve context
    ctx = await manager.get_context(user_hash)
    assert ctx.last_symbol == "AAPL"
    assert ctx.last_intent == "chart"
    
    # Partial update (intent only)
    await manager.update_context(user_hash, intent="rsi")
    ctx2 = await manager.get_context(user_hash)
    assert ctx2.last_symbol == "AAPL" # Should persist
    assert ctx2.last_intent == "rsi"

# --- Fuzzy Logic Tests ---

def test_extract_symbols_fuzzy():
    # Exact
    assert "AAPL" in extract_symbols_from_text("chart AAPL")
    assert "TSLA" in extract_symbols_from_text("chart tesla") # Alias
    
    # Fuzzy (requires thefuzz installed)
    try:
        import thefuzz
        # "Nvidea" -> NVDA
        assert "NVDA" in extract_symbols_from_text("price of nvidea")
        # "Microsft" -> MSFT
        assert "MSFT" in extract_symbols_from_text("news for microsft")
    except ImportError:
        pass

def test_stopwords():
    # "can" is a stopword (CAN is a ticker)
    # "chart can" -> should NOT extract CAN
    assert "CAN" not in extract_symbols_from_text("chart can") 
    
    # "chart CAN" (explicit caps) -> should extract CAN?
    # extract_symbols_from_text implementation:
    # 3. Look for tickers... if clean_lower not in STOPWORDS...
    # If explicit caps, it falls into "uppercase" logic.
    # Current logic: clean.isupper() ... then checks stopwords?
    # Let's verify behavior. Ideally explicit caps $CAN or "CAN" (maybe) should count.
    # Logic: if clean.isupper(): sym = clean.upper() -> added.
    # Oh wait, my logic was:
    # if clean_lower not in STOPWORDS: sym = clean.upper()...
    # So even uppercase CAN is ignored if 'can' is in stopwords?
    # If so, that's a limitation/protection.
    pass

# --- Advanced Params Tests ---

def test_intent_parser_periods():
    # "chart aapl 6m"
    intent = parse_intent("chart aapl 6m")
    assert intent.command == "chart"
    assert "AAPL" in intent.symbols
    assert "6m" in intent.args

    # "price of btc 1y"
    intent = parse_intent("price of btc 1y")
    assert "BTC-USD" in intent.symbols
    assert "1y" in intent.args

# --- Multi-Intent Tests ---

@pytest.mark.asyncio
async def test_dispatcher_multi_intent():
    dispatcher = CommandDispatcher()
    
    # Mock execute_command to return Success
    dispatcher._execute_command = AsyncMock(return_value=CommandResult.ok("Done"))
    
    # "Chart Apple and show RSI"
    # This relies on _execute_command doing context resolution logic, which we mocked.
    # But dispatch logic splits it.
    
    # We test _looks_like_query and dispatch calls
    result = await dispatcher.dispatch("sender", "Chart Apple and show RSI")
    
    # Should call _execute_command twice?
    # First: "Chart Apple" -> intent chart AAPL
    # Second: "show RSI" -> intent rsi (no symbol).
    # Multi-intent logic in dispatch iterates and calls execute.
    
    assert dispatcher._execute_command.call_count == 2
    # Verify calls?
    # call args: (command, args, ...)
    calls = dispatcher._execute_command.call_args_list
    assert calls[0][0][0] == "chart"
    # Second call might be 'rsi' with empty symbol list?
    # Since we mocked it, it just returns "Done".
    # Result should be merged text "Done\n\n...\n\nDone"

@pytest.mark.asyncio
async def test_complex_query_robustness():
    """Test 'Chart Apple. Do it in candlesticks' flow."""
    dispatcher = CommandDispatcher()
    dispatcher.context_manager = MagicMock()
    # Mock context get/update
    mock_ctx = MagicMock()
    mock_ctx.last_symbol = "AAPL"
    dispatcher.context_manager.get_context = AsyncMock(return_value=mock_ctx)
    dispatcher.context_manager.update_context = AsyncMock()
    
    dispatcher._execute_command = AsyncMock(return_value=CommandResult.ok("Done"))
    
    # "Chart Apple. Do it in candlesticks"
    # Should split into:
    # 1. "Chart Apple" -> Command: chart, Args: [AAPL]
    # 2. "Do it in candlesticks" -> Command: chart (inferred from -c), Args: [AAPL, -c] (resolved it->AAPL)
    
    await dispatcher.dispatch("user1", "Chart Apple. Do it in candlesticks")
    
    print(f"Call count: {dispatcher._execute_command.call_count}")
    for i, call in enumerate(dispatcher._execute_command.call_args_list):
        print(f"Call {i}: {call}")

    assert dispatcher._execute_command.call_count == 2
    calls = dispatcher._execute_command.call_args_list
    
    # First call: chart AAPL
    assert calls[0][0][0] == "chart"
    assert "AAPL" in calls[0][0][1]
    
    # Second call: chart AAPL -c
    assert calls[1][0][0] == "chart"
    # Check args for resolved symbol AND flag
    args2 = calls[1][0][1]
    # Context chaining resolves 'it' to 'AAPL' within dispatcher before calling _execute_command
    assert "AAPL" in args2 
    assert "-c" in args2   # Extracted from 'candlesticks'
