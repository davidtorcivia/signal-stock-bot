"""
Yahoo Finance provider - no API key required.

Uses yfinance library which scrapes Yahoo Finance.
Best for: development, fallback, users without API keys.
Limitations: unofficial, may break if Yahoo changes their site.

Options: the only free chain source in this deployment. Polygon serves
chains on its paid tier only, so without this the whole options surface
(!chain, !opt, the paper portfolio's options tools, skew, flow) returns
"no providers available". Yahoo gives strike, bid/ask, last, volume,
open interest and implied vol, but no greeks.
"""

import logging
import math
import re
from datetime import datetime

import yfinance as yf

from ..executor import run_blocking

# Set yfinance cache path to avoid permission errors in Docker
try:
    yf.set_tz_cache_location("/tmp/yfinance-cache")
except Exception:
    pass  # Ignore if it fails

from ..cache import TTLCache
from ..options_symbols import parse_occ
from .base import (
    BaseProvider,
    Quote,
    HistoricalBar,
    Fundamentals,
    OptionQuote,
    ProviderCapability,
    ProviderError,
    SymbolNotFoundError,
)

logger = logging.getLogger(__name__)

# Raw yfinance chain frames, keyed by "SYMBOL:YYYY-MM-DD". Yahoo has no
# per-contract endpoint, so every single-contract quote is a whole-chain
# download; without this, pricing one portfolio's positions or drawing a
# term structure re-downloads the same expiry several times over. Short
# TTL because the paper executor fills at these marks.
_CHAIN_CACHE: TTLCache = TTLCache(ttl_seconds=60, max_size=256, name="yahoo_option_chains")


def _finite_float(value):
    """None for NaN/inf/unparseable — yfinance frames are full of NaN."""
    if value is None:
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _atm_pivot(calls_frame):
    """Spot, near enough, from a calls frame's inTheMoney flag.

    The boundary between in- and out-of-the-money calls is the
    underlying price, so the highest ITM strike locates the money
    without a second quote request. None when the column is absent.
    """
    try:
        itm = calls_frame[calls_frame["inTheMoney"].astype(bool)]
        if itm.empty:
            return None
        return float(itm["strike"].max())
    except Exception:
        return None


class YahooFinanceProvider(BaseProvider):
    name = "yahoo"
    capabilities = {
        ProviderCapability.QUOTE,
        ProviderCapability.HISTORICAL,
        ProviderCapability.FUNDAMENTALS,
        ProviderCapability.OPTIONS,
    }
    
    async def get_quote(self, symbol: str) -> Quote:
        """Fetch quote using yfinance (runs sync code in bounded executor)."""
        return await run_blocking(self._get_quote_sync, symbol)
    
    def _get_quote_sync(self, symbol: str) -> Quote:
        ticker = yf.Ticker(symbol)
        
        # Try info first (more complete but slower)
        try:
            info = ticker.info
            if info and info.get('regularMarketPrice') is not None:
                price = info.get('regularMarketPrice', 0)
                prev_close = info.get('regularMarketPreviousClose', price)
                change = price - prev_close
                change_pct = (change / prev_close * 100) if prev_close else 0
                
                return Quote(
                    symbol=symbol.upper(),
                    price=price,
                    change=change,
                    change_percent=change_pct,
                    volume=info.get('regularMarketVolume', 0) or 0,
                    timestamp=datetime.now(),
                    provider=self.name,
                    open=info.get('regularMarketOpen'),
                    high=info.get('regularMarketDayHigh'),
                    low=info.get('regularMarketDayLow'),
                    prev_close=prev_close,
                    market_cap=info.get('marketCap'),
                    name=info.get('shortName'),
                )
        except Exception as e:
            logger.debug(f"Info lookup failed for {symbol}, trying fast_info: {e}")
        
        # Fallback to fast_info
        try:
            fast = ticker.fast_info
            if fast.last_price is None:
                raise SymbolNotFoundError(f"Symbol not found: {symbol}")
            
            prev_close = fast.previous_close or fast.last_price
            change = fast.last_price - prev_close
            change_pct = (change / prev_close * 100) if prev_close else 0
            
            return Quote(
                symbol=symbol.upper(),
                price=fast.last_price,
                change=change,
                change_percent=change_pct,
                volume=int(fast.last_volume or 0),
                timestamp=datetime.now(),
                provider=self.name,
                prev_close=prev_close,
                market_cap=getattr(fast, 'market_cap', None),
            )
        except SymbolNotFoundError:
            raise
        except Exception as e:
            logger.error(f"Failed to get quote for {symbol}: {e}")
            raise SymbolNotFoundError(f"Symbol not found: {symbol}")
    
    async def get_quotes(self, symbols: list[str]) -> dict[str, Quote]:
        """Batch fetch quotes."""
        return await run_blocking(self._get_quotes_sync, symbols, timeout=30.0)
    
    def _get_quotes_sync(self, symbols: list[str]) -> dict[str, Quote]:
        results = {}
        
        # yfinance batch download for efficiency
        try:
            tickers = yf.Tickers(" ".join(symbols))
            for symbol in symbols:
                try:
                    symbol_upper = symbol.upper()
                    ticker = tickers.tickers.get(symbol_upper)
                    if ticker:
                        quote = self._get_quote_sync(symbol)
                        results[symbol_upper] = quote
                except SymbolNotFoundError:
                    logger.debug(f"Symbol not found in batch: {symbol}")
                except Exception as e:
                    logger.warning(f"Error fetching {symbol} in batch: {e}")
        except Exception as e:
            logger.error(f"Batch fetch failed: {e}")
            # Fall back to individual fetches
            for symbol in symbols:
                try:
                    results[symbol.upper()] = self._get_quote_sync(symbol)
                except Exception:
                    pass
        
        return results
    
    async def get_historical(
        self,
        symbol: str,
        period: str = "1mo",
        interval: str = "1d"
    ) -> list[HistoricalBar]:
        return await run_blocking(
            self._get_historical_sync, symbol, period, interval, timeout=25.0
        )
    
    def _get_historical_sync(
        self, symbol: str, period: str, interval: str
    ) -> list[HistoricalBar]:
        ticker = yf.Ticker(symbol)
        hist = ticker.history(period=period, interval=interval)
        
        if hist.empty:
            raise SymbolNotFoundError(f"No historical data for {symbol}")
        
        bars = []
        for idx, row in hist.iterrows():
            bars.append(HistoricalBar(
                timestamp=idx.to_pydatetime(),
                open=float(row['Open']),
                high=float(row['High']),
                low=float(row['Low']),
                close=float(row['Close']),
                volume=int(row['Volume']),
            ))
        
        return bars
    
    async def get_fundamentals(self, symbol: str) -> Fundamentals:
        return await run_blocking(self._get_fundamentals_sync, symbol)
    
    def _get_fundamentals_sync(self, symbol: str) -> Fundamentals:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        
        if not info or 'shortName' not in info:
            raise SymbolNotFoundError(f"No fundamental data for {symbol}")
        
        return Fundamentals(
            symbol=symbol.upper(),
            name=info.get('shortName', symbol),
            pe_ratio=info.get('trailingPE'),
            eps=info.get('trailingEps'),
            market_cap=info.get('marketCap'),
            dividend_yield=info.get('dividendYield'),
            fifty_two_week_high=info.get('fiftyTwoWeekHigh'),
            fifty_two_week_low=info.get('fiftyTwoWeekLow'),
            sector=info.get('sector'),
            industry=info.get('industry'),
            provider=self.name,
        )
    

    # ── Options ────────────────────────────────────────────────────────────

    async def get_options_chain(
        self,
        underlying: str,
        expiration: str = None,
        limit: int = 100,
    ) -> list[OptionQuote]:
        """Contracts for one expiry, nearest the money first.

        Differs from the Polygon implementation in two ways worth
        knowing: `expiration=None` means the FRONT expiry rather than a
        snapshot across all of them, and `limit` keeps the contracts
        closest to the money rather than an arbitrary page.
        """
        return await run_blocking(
            self._get_options_chain_sync, underlying, expiration, limit,
            timeout=30.0,
        )

    def _get_options_chain_sync(
        self, underlying: str, expiration, limit: int,
    ) -> list[OptionQuote]:
        underlying = (underlying or "").strip().upper()
        if not re.match(r"^[A-Z][A-Z.\-]{0,9}$", underlying):
            raise ProviderError(
                f"get_options_chain: invalid underlying {underlying!r}"
            )
        if expiration and not re.match(r"^\d{4}-\d{2}-\d{2}$", str(expiration)):
            raise ProviderError(
                f"get_options_chain: invalid expiration {expiration!r}; "
                f"expected YYYY-MM-DD"
            )
        ticker = yf.Ticker(underlying)
        try:
            expirations = list(ticker.options or ())
        except Exception as e:
            raise SymbolNotFoundError(f"No options for {underlying}: {e}")
        if not expirations:
            raise SymbolNotFoundError(f"No options for {underlying}")
        target = str(expiration) if expiration else expirations[0]
        if target not in expirations:
            raise SymbolNotFoundError(
                f"{underlying} has no {target} expiry; available: "
                f"{', '.join(expirations[:8])}"
            )
        cache_key = f"{underlying}:{target}"
        chain = _CHAIN_CACHE.get(cache_key)
        if chain is None:
            try:
                chain = ticker.option_chain(target)
            except Exception as e:
                raise ProviderError(f"chain fetch failed for {underlying}: {e}")
            _CHAIN_CACHE.set(cache_key, chain)

        rows: list[OptionQuote] = []
        pivot = None
        for frame, kind in ((chain.calls, "call"), (chain.puts, "put")):
            if frame is None or frame.empty:
                continue
            if kind == "call":
                pivot = _atm_pivot(frame)
            for row in frame.itertuples(index=False):
                quote = self._row_to_option_quote(row, underlying, target, kind)
                if quote is not None:
                    rows.append(quote)
        if not rows:
            return rows
        # `limit` takes the contracts nearest the money, not the lowest
        # strikes. Truncating a strike-sorted chain would hand back deep
        # ITM calls and worthless puts and never the strikes anyone
        # trades — on SPY at 759, limit=100 returned 505 through 744.
        limit = max(1, int(limit))
        if len(rows) > limit:
            if pivot is None:
                strikes = sorted(q.strike for q in rows)
                pivot = strikes[len(strikes) // 2]
            rows.sort(key=lambda q: (abs(q.strike - pivot), q.strike, q.type))
            rows = rows[:limit]
        # Strike order with calls before puts at the same strike: callers
        # that slice around the money get a symmetric window either way.
        rows.sort(key=lambda q: (q.strike, q.type))
        return rows

    def _row_to_option_quote(
        self, row, underlying: str, expiration: str, kind: str,
    ):
        symbol = str(getattr(row, "contractSymbol", "") or "")
        if not symbol:
            return None
        try:
            exp_dt = datetime.strptime(expiration, "%Y-%m-%d")
        except ValueError:
            exp_dt = datetime.now()
        # Mid beats last: a contract that last printed hours ago still
        # has a live two-sided market, and the paper executor fills at
        # this price. A bid of exactly 0.00 against a live ask is the
        # normal state of a cheap far-OTM contract, not a missing book,
        # so it must not fall through to `lastPrice` — that print can be
        # days old and well above the current offer, which would both
        # overstate the mark and fill buys above the ask.
        bid = _finite_float(getattr(row, "bid", None)) or 0.0
        ask = _finite_float(getattr(row, "ask", None)) or 0.0
        last = _finite_float(getattr(row, "lastPrice", None))
        if ask > 0:
            price = (bid + ask) / 2.0 if ask >= bid else ask
        else:
            price = last or 0.0
        return OptionQuote(
            symbol=symbol,
            underlying=underlying,
            expiration=exp_dt,
            strike=_finite_float(getattr(row, "strike", None)) or 0.0,
            type=kind,
            price=price,
            change=_finite_float(getattr(row, "change", None)) or 0.0,
            change_percent=_finite_float(getattr(row, "percentChange", None)) or 0.0,
            volume=int(_finite_float(getattr(row, "volume", None)) or 0),
            open_interest=int(_finite_float(getattr(row, "openInterest", None)) or 0),
            implied_volatility=_finite_float(
                getattr(row, "impliedVolatility", None)
            ),
            greeks=None,
            timestamp=datetime.now(),
            provider=self.name,
        )

    async def get_option_expirations(self, underlying: str) -> list[str]:
        return await run_blocking(
            self._get_option_expirations_sync, underlying, timeout=20.0,
        )

    def _get_option_expirations_sync(self, underlying: str) -> list[str]:
        underlying = (underlying or "").strip().upper()
        if not re.match(r"^[A-Z][A-Z.\-]{0,9}$", underlying):
            raise ProviderError(
                f"get_option_expirations: invalid underlying {underlying!r}"
            )
        try:
            out = [str(e) for e in (yf.Ticker(underlying).options or ())]
        except Exception as e:
            raise SymbolNotFoundError(f"No options for {underlying}: {e}")
        if not out:
            raise SymbolNotFoundError(f"No options for {underlying}")
        return out

    async def get_option_quote(self, symbol: str) -> OptionQuote:
        return await run_blocking(
            self._get_option_quote_sync, symbol, timeout=30.0,
        )

    def _get_option_quote_sync(self, symbol: str) -> OptionQuote:
        """Single contract by OCC symbol.

        Yahoo has no per-contract endpoint, so this pulls that contract's
        expiry and picks the row out. `_CHAIN_CACHE` is what keeps that
        affordable: quoting a dozen contracts on one expiry costs one
        download, not a dozen.
        """
        occ = (symbol or "").strip().upper()
        try:
            parts = parse_occ(occ)
        except Exception as e:
            raise SymbolNotFoundError(f"Not an OCC contract symbol: {symbol!r} ({e})")
        chain = self._get_options_chain_sync(
            parts.root, parts.expiration.isoformat(), limit=10_000,
        )
        for quote in chain:
            if quote.symbol.upper() == occ:
                return quote
        raise SymbolNotFoundError(f"Contract not listed: {occ}")

    async def health_check(self) -> bool:
        try:
            await self.get_quote("AAPL")
            return True
        except Exception as e:
            logger.error(f"Yahoo health check failed: {e}")
            return False
