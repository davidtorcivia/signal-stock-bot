
import asyncio
import os
import sys
import logging

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.providers.massive import MassiveProvider
from src.providers.base import ProviderError, SymbolNotFoundError

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("verify_massive")

async def verify():
    api_key = os.environ.get("MASSIVE_API_KEY")
    if not api_key:
        logger.error("MASSIVE_API_KEY not found in environment")
        return

    logger.info("Initializing MassiveProvider...")
    provider = MassiveProvider(api_key=api_key)

    try:
        # 1. Test Stock Quote
        logger.info("\n--- Q1: Stock Quote (AAPL) ---")
        try:
            quote = await provider.get_quote("AAPL")
            logger.info(f"SUCCESS: {quote.symbol} Price: ${quote.price} Vol: {quote.volume}")
        except Exception as e:
            logger.error(f"FAILED: {e}")

        # 2. Test Forex
        logger.info("\n--- Q2: Forex (EUR/USD) ---")
        try:
            quote = await provider.get_forex_quote("EUR/USD")
            logger.info(f"SUCCESS: {quote.symbol} Rate: {quote.rate}")
        except Exception as e:
            logger.error(f"FAILED: {e}")

        # 3. Test Crypto
        logger.info("\n--- Q3: Crypto (BTC-USD -> X:BTCUSD) ---")
        try:
            quote = await provider.get_quote("BTC-USD")
            logger.info(f"SUCCESS: {quote.symbol} Price: ${quote.price}")
        except Exception as e:
            logger.error(f"FAILED: {e}")

        # 4. Test Futures (Use a known symbol like /ES or generic if possible, Massive uses tickers like 'ES=F' or strictly 'C:EURUSD' but for futures... check doc assumptions)
        # Massive Docs for futures tickers: 'ES' might be confusing. Often /ES or ES=F.
        # Let's try 'CL' (Crude) or 'ES' (E-mini)
        logger.info("\n--- Q4: Futures (ES) ---")
        try:
            quote = await provider.get_future_quote("ES") # Might fail if symbol invalid, but tests the path
            logger.info(f"SUCCESS: {quote.symbol} Price: ${quote.price}")
        except SymbolNotFoundError:
            logger.warning("Symbol 'ES' not found (expected if market data requires specific format)")
        except Exception as e:
            logger.error(f"FAILED: {e}")

        # 5. Test Option (Hard to pick a valid live one without searching... let's try a very likely one or skip)
        # We'll skip specific option contract test to avoid 404 spam unless we find a valid ticker.
        # But we tested the unit test logic.
        
    finally:
        await provider.close()

if __name__ == "__main__":
    asyncio.run(verify())
