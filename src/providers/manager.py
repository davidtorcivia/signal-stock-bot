"""
Provider manager with automatic fallback.

Routes requests to available providers in priority order.
Handles rate limiting by temporarily excluding providers.
Includes caching and retry logic for improved reliability.
"""

import asyncio
import logging
import time
from typing import Optional

from .base import (
    BaseProvider,
    Quote,
    HistoricalBar,
    Fundamentals,
    OptionQuote,
    ForexQuote,
    FuturesQuote,
    EconomyIndicator,
    ProviderCapability,
    ProviderError,
    RateLimitError,
)
from ..cache import get_quote_cache, TTLCache

logger = logging.getLogger(__name__)

# Retry configuration
MAX_RETRIES = 2
BASE_DELAY_SECONDS = 0.5


class ProviderManager:
    """
    Manages multiple providers with automatic fallback.
    
    Priority order is determined by the order providers are added.
    If a provider fails or is rate limited, the next provider is tried.
    """
    
    def __init__(self, cache_ttl: int = 300, enable_cache: bool = True):
        self.providers: list[BaseProvider] = []
        self._rate_limited: dict[str, float] = {}  # provider_name -> retry_after_timestamp
        self._enable_cache = enable_cache
        self._cache: TTLCache[Quote] = get_quote_cache(cache_ttl) if enable_cache else None
    
    def add_provider(self, provider: BaseProvider):
        """Add a provider to the fallback chain"""
        self.providers.append(provider)
        logger.info(
            f"Added provider: {provider.name} "
            f"with capabilities: {[c.value for c in provider.capabilities]}"
        )
    
    def _get_available_providers(
        self,
        capability: ProviderCapability
    ) -> list[BaseProvider]:
        """Get providers that support a capability and aren't rate limited"""
        now = time.time()
        
        available = []
        for p in self.providers:
            # Check capability
            if capability not in p.capabilities:
                continue
            
            # Check rate limit
            rate_limit_until = self._rate_limited.get(p.name, 0)
            if now < rate_limit_until:
                remaining = int(rate_limit_until - now)
                logger.debug(f"Provider {p.name} rate limited for {remaining}s more")
                continue
            
            available.append(p)
        
        return available
    
    def _mark_rate_limited(self, provider: BaseProvider, retry_after: int):
        """Mark a provider as rate limited"""
        self._rate_limited[provider.name] = time.time() + retry_after
        logger.warning(f"Provider {provider.name} rate limited for {retry_after}s")
    
    def _clear_rate_limit(self, provider: BaseProvider):
        """Clear rate limit for a provider"""
        if provider.name in self._rate_limited:
            del self._rate_limited[provider.name]
    
    async def get_quote(self, symbol: str) -> Quote:
        """Get quote with caching, retry logic, and automatic fallback"""
        symbol = symbol.upper()
        
        # Check cache first
        if self._enable_cache and self._cache:
            cached = self._cache.get(symbol)
            if cached:
                logger.debug(f"Cache hit for {symbol}")
                return cached
        
        providers = self._get_available_providers(ProviderCapability.QUOTE)
        
        if not providers:
            raise ProviderError("No providers available for quotes")
        
        last_error: Optional[Exception] = None
        
        for provider in providers:
            # Retry loop with exponential backoff
            for attempt in range(MAX_RETRIES + 1):
                try:
                    logger.debug(f"Trying {provider.name} for quote: {symbol} (attempt {attempt + 1})")
                    quote = await provider.get_quote(symbol)
                    self._clear_rate_limit(provider)
                    
                    # Cache successful result
                    if self._enable_cache and self._cache:
                        self._cache.set(symbol, quote)
                    
                    return quote
                    
                except RateLimitError as e:
                    self._mark_rate_limited(provider, e.retry_after or 60)
                    last_error = e
                    break  # Don't retry rate limits, move to next provider
                    
                except ProviderError as e:
                    logger.warning(f"Provider {provider.name} failed for {symbol}: {e}")
                    last_error = e
                    break  # Provider errors are usually not transient
                    
                except Exception as e:
                    last_error = e
                    if attempt < MAX_RETRIES:
                        delay = BASE_DELAY_SECONDS * (2 ** attempt)
                        logger.warning(f"Retrying {provider.name} in {delay}s: {e}")
                        await asyncio.sleep(delay)
                    else:
                        logger.error(f"Max retries exceeded for {provider.name}: {e}")
        
        raise last_error or ProviderError("All providers failed")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch quotes with caching and fallback"""
        symbols = [s.upper() for s in symbols]
        results: dict[str, Quote] = {}
        remaining = set(symbols)
        
        # Check cache first
        if self._enable_cache and self._cache:
            cached = self._cache.get_multi(list(remaining))
            results.update(cached)
            remaining -= set(cached.keys())
            if cached:
                logger.debug(f"Cache hits for {len(cached)} symbols")
        
        if not remaining:
            return results
        
        providers = self._get_available_providers(ProviderCapability.QUOTE)
        
        if not providers:
            if results:  # Return cached results even if no providers
                return results
            raise ProviderError("No providers available for quotes")
        
        for provider in providers:
            if not remaining:
                break
            
            try:
                logger.debug(f"Trying {provider.name} for batch quotes: {remaining}")
                batch_results = await provider.get_quotes(list(remaining))
                results.update(batch_results)
                remaining -= set(batch_results.keys())
                self._clear_rate_limit(provider)
                
                # Cache successful results
                if self._enable_cache and self._cache:
                    self._cache.set_multi(batch_results)
                
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                
            except Exception as e:
                logger.warning(f"Provider {provider.name} failed batch: {e}")
        
        return results
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        """Get historical data with fallback"""
        providers = self._get_available_providers(ProviderCapability.HISTORICAL)
        
        if not providers:
            raise ProviderError("No providers available for historical data")
        
        last_error: Optional[Exception] = None
        
        for provider in providers:
            try:
                logger.debug(f"Trying {provider.name} for historical: {symbol}")
                bars = await provider.get_historical(symbol, period, interval)
                self._clear_rate_limit(provider)
                return bars
                
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
                
            except ProviderError as e:
                logger.warning(f"Provider {provider.name} failed historical for {symbol}: {e}")
                last_error = e
                
            except Exception as e:
                logger.exception(f"Unexpected error from {provider.name}")
                last_error = e
        
        raise last_error or ProviderError("All providers failed")
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        """Get fundamentals with fallback"""
        providers = self._get_available_providers(ProviderCapability.FUNDAMENTALS)
        
        if not providers:
            raise ProviderError("No providers available for fundamentals")
        
        last_error: Optional[Exception] = None
        
        for provider in providers:
            try:
                logger.debug(f"Trying {provider.name} for fundamentals: {symbol}")
                fund = await provider.get_fundamentals(symbol)
                self._clear_rate_limit(provider)
                return fund
                
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
                
            except ProviderError as e:
                logger.warning(f"Provider {provider.name} failed fundamentals for {symbol}: {e}")
                last_error = e
                
            except Exception as e:
                logger.exception(f"Unexpected error from {provider.name}")
                last_error = e
        
        raise last_error or ProviderError("All providers failed")

    async def get_option_quote(self, symbol: str) -> OptionQuote:
        """Get option quote with fallback"""
        providers = self._get_available_providers(ProviderCapability.OPTIONS)
        if not providers:
             raise ProviderError("No providers available for options")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_option_quote(symbol)
            except RateLimitError as e:
                self._mark_rate_limited(provider, e.retry_after or 60)
                last_error = e
            except Exception as e:
                logger.warning(f"Error fetching option quote from {provider.name}: {e}")
                last_error = e
        raise last_error or ProviderError("All providers failed to return option quote")

    async def get_forex_quote(self, symbol: str) -> ForexQuote:
        """Get forex quote with fallback"""
        providers = self._get_available_providers(ProviderCapability.FOREX)
        if not providers:
             raise ProviderError("No providers available for forex")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_forex_quote(symbol)
            except RateLimitError as e:
                 self._mark_rate_limited(provider, e.retry_after or 60)
                 last_error = e
            except Exception as e:
                logger.warning(f"Error fetching forex quote from {provider.name}: {e}")
                last_error = e
        raise last_error or ProviderError("All providers failed to return forex quote")

    async def get_future_quote(self, symbol: str) -> FuturesQuote:
        """Get futures quote with fallback"""
        providers = self._get_available_providers(ProviderCapability.FUTURES)
        if not providers:
             raise ProviderError("No providers available for futures")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_future_quote(symbol)
            except RateLimitError as e:
                 self._mark_rate_limited(provider, e.retry_after or 60)
                 last_error = e
            except Exception as e:
                logger.warning(f"Error fetching futures quote from {provider.name}: {e}")
                last_error = e
        raise last_error or ProviderError("All providers failed to return futures quote")

    async def get_economy_data(self, indicator: str) -> EconomyIndicator:
        """Get economic data with fallback"""
        providers = self._get_available_providers(ProviderCapability.ECONOMY)
        if not providers:
             raise ProviderError("No providers available for economy data")
        
        last_error = None
        for provider in providers:
            try:
                return await provider.get_economy_data(indicator)
            except RateLimitError as e:
                 self._mark_rate_limited(provider, e.retry_after or 60)
                 last_error = e
            except Exception as e:
                logger.warning(f"Error fetching economy data from {provider.name}: {e}")
                last_error = e
        raise last_error or ProviderError("All providers failed to return economy data")
    
    async def health_check(self) -> dict[str, bool]:
        """Check health of all providers"""
        results = {}
        for provider in self.providers:
            try:
                results[provider.name] = await provider.health_check()
            except Exception as e:
                logger.error(f"Health check failed for {provider.name}: {e}")
                results[provider.name] = False
        return results
    
    def get_status(self) -> dict:
        """Get current status of all providers"""
        now = time.time()
        status = {}
        
        for provider in self.providers:
            rate_limit_until = self._rate_limited.get(provider.name, 0)
            is_rate_limited = now < rate_limit_until
            
            status[provider.name] = {
                "capabilities": [c.value for c in provider.capabilities],
                "rate_limited": is_rate_limited,
                "rate_limit_remaining_seconds": max(0, int(rate_limit_until - now)) if is_rate_limited else 0,
            }
        
        return status
