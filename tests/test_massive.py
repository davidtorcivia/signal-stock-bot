import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from src.providers.massive import MassiveProvider
from src.providers.base import ProviderCapability, ProviderError, RateLimitError

@pytest.fixture
async def massive_provider():
    provider = MassiveProvider(api_key="test_key")
    yield provider
    await provider.close()

def create_mock_response(status=200, json_data=None):
    mock_resp = AsyncMock()
    mock_resp.status = status
    if json_data:
        mock_resp.json.return_value = json_data
    return mock_resp

@pytest.mark.asyncio
async def test_capabilities(massive_provider):
    assert ProviderCapability.QUOTE in massive_provider.capabilities
    assert ProviderCapability.OPTIONS in massive_provider.capabilities
    assert ProviderCapability.FUTURES in massive_provider.capabilities

@pytest.mark.asyncio
async def test_get_option_quote(massive_provider):
    json_data = {
        "status": "OK",
        "results": {
            "underlying_asset": {"ticker": "AAPL"},
            "details": {
                "contract_type": "call",
                "strike_price": 150.0,
                "expiration_date": "2023-01-20"
            },
            "day": {
                "close": 5.25,
                "change": 0.25,
                "change_percent": 5.0,
                "volume": 1200
            },
            "open_interest": 5000,
            "updated": 1670000000000000000
        }
    }
    
    mock_resp = create_mock_response(json_data=json_data)
    # The session.get() call returns an async context manager
    # mocking: session.get.return_value.__aenter__.return_value = mock_resp
    
    mock_get = MagicMock()
    mock_get.return_value.__aenter__.return_value = mock_resp
    
    with patch('aiohttp.ClientSession.get', new=mock_get):
        quote = await massive_provider.get_option_quote("O:AAPL230120C00150000")
        assert quote.symbol == "O:AAPL230120C00150000"
        assert quote.price == 5.25
        assert quote.provider == "massive"

@pytest.mark.asyncio
async def test_get_forex_quote(massive_provider):
    json_data = {
        "status": "OK",
        "results": [{
            "c": 1.08,
            "o": 1.07,
            "v": 100,
            "t": 1670000000000
        }],
        "resultsCount": 1
    }
    
    mock_resp = create_mock_response(json_data=json_data)
    mock_get = MagicMock()
    mock_get.return_value.__aenter__.return_value = mock_resp

    with patch('aiohttp.ClientSession.get', new=mock_get):
        quote = await massive_provider.get_forex_quote("EUR/USD")
        assert quote.symbol == "EUR/USD"
        assert quote.rate == 1.08

@pytest.mark.asyncio
async def test_get_future_quote(massive_provider):
    json_data = {
        "status": "OK",
        "results": [{
            "c": 4000.50,
            "o": 3990.00,
            "v": 1500,
            "t": 1670000000000
        }],
        "resultsCount": 1
    }

    mock_resp = create_mock_response(json_data=json_data)
    mock_get = MagicMock()
    mock_get.return_value.__aenter__.return_value = mock_resp

    with patch('aiohttp.ClientSession.get', new=mock_get):
        quote = await massive_provider.get_future_quote("ES")
        assert quote.symbol == "ES"
        assert quote.price == 4000.50

@pytest.mark.asyncio
async def test_api_errors(massive_provider):
    # Test 401
    mock_resp_401 = create_mock_response(status=401)
    mock_get_401 = MagicMock()
    mock_get_401.return_value.__aenter__.return_value = mock_resp_401
    
    with patch('aiohttp.ClientSession.get', new=mock_get_401):
        with pytest.raises(ProviderError) as exc:
            await massive_provider.get_quote("AAPL")
        assert "Invalid API key" in str(exc.value)

    # Test 429
    mock_resp_429 = create_mock_response(status=429)
    mock_get_429 = MagicMock()
    mock_get_429.return_value.__aenter__.return_value = mock_resp_429
    
    with patch('aiohttp.ClientSession.get', new=mock_get_429):
        with pytest.raises(RateLimitError):
            await massive_provider.get_quote("AAPL")
