import pytest
import asyncio
import time
from unittest.mock import MagicMock, patch
from src.commands.intent_parser import parse_intent, extract_symbols_from_text
from src.database import AlertsDB
from src.cache import RequestDeduplicator
from src.providers.base import CircuitBreaker

# --- Intent Parser Tests ---

def test_extract_symbols():
    assert "AAPL" in extract_symbols_from_text("Check $AAPL price")
    assert "TSLA" in extract_symbols_from_text("Chart Tesla stock")
    assert "MSFT" in extract_symbols_from_text("News for Microsoft")
    assert "GOOGL" in extract_symbols_from_text("google earnings")

def test_intent_parser_chart():
    intent = parse_intent("chart apple")
    assert intent.command == "chart"
    assert "AAPL" in intent.symbols
    assert intent.confidence > 0.6

def test_intent_parser_price():
    intent = parse_intent("what is the price of NVDA")
    assert intent.command == "price"
    assert "NVDA" in intent.symbols

def test_intent_parser_analytics():
    intent = parse_intent("insider trading for amazon")
    # Amazon is an alias for AMZN
    assert intent.command == "insider"
    assert "AMZN" in intent.symbols
    
    intent = parse_intent("rsi for TSLA")
    assert intent.command == "rsi"
    assert "TSLA" in intent.symbols

# --- Alerts DB Tests ---

@pytest.mark.asyncio
async def test_alerts_crud(tmp_path):
    db_file = tmp_path / "test_alerts.db"
    db = AlertsDB(str(db_file))
    await db._ensure_initialized()
    
    # Add alert
    alert_id = await db.add_alert(
        user_hash="hash123",
        user_phone="+123",
        symbol="AAPL",
        condition="above",
        target_value=200.0,
        group_id="group1"
    )
    assert alert_id is not None, "Failed to create alert"
    
    # List alerts
    alerts = await db.get_active_alerts("hash123")
    assert len(alerts) == 1
    assert alerts[0]["symbol"] == "AAPL"
    assert alerts[0]["target_value"] == 200.0
    
    # Trigger alert
    success = await db.trigger_alert(alert_id)
    assert success
    
    # Verify inactive
    alerts_after = await db.get_active_alerts("hash123")
    assert len(alerts_after) == 0
    
    # Remove (cleanup)
    alert_id_2 = await db.add_alert("hash123", "+123", "MSFT", "below", 100.0)
    removed = await db.remove_alert(alert_id_2, "hash123")
    assert removed

# --- Performance Tests ---

@pytest.mark.asyncio
async def test_deduplicator():
    dedup = RequestDeduplicator()
    
    # Simulate an async function
    mock_func = MagicMock()
    async def task():
        mock_func()
        await asyncio.sleep(0.05)
        return "result"
    
    # Fire two requests concurrently
    t1 = asyncio.create_task(dedup.execute("key1", task))
    t2 = asyncio.create_task(dedup.execute("key1", task))
    
    r1 = await t1
    r2 = await t2
    
    assert r1 == "result"
    assert r2 == "result"
    # Should only run once
    assert mock_func.call_count == 1

def test_circuit_breaker():
    cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1)
    
    assert cb.is_available()
    
    # Fail once
    cb.record_failure()
    assert cb.is_available()
    
    # Fail twice - trigger open
    cb.record_failure()
    assert not cb.is_available()
    
    # Wait for recovery
    time.sleep(1.1)
    
    # Now half-open
    assert cb.is_available()
    
    # Success closes it
    cb.record_success()
    assert cb.state == "closed"
    assert cb.consecutive_failures == 0
