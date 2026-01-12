"""
Technical analysis commands for stock bot.

Provides: !ta, !rsi, !sma, !macd, !support
"""

from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager, ProviderError, SymbolNotFoundError

import numpy as np


def calculate_sma(closes: list, period: int) -> float:
    """Calculate Simple Moving Average."""
    if len(closes) < period:
        return None
    return sum(closes[-period:]) / period


def calculate_ema(closes: list, period: int) -> float:
    """Calculate Exponential Moving Average."""
    if len(closes) < period:
        return None
    multiplier = 2 / (period + 1)
    ema = sum(closes[:period]) / period  # Start with SMA
    for price in closes[period:]:
        ema = (price - ema) * multiplier + ema
    return ema


def calculate_rsi(closes: list, period: int = 14) -> float:
    """Calculate Relative Strength Index."""
    if len(closes) < period + 1:
        return None
    
    gains = []
    losses = []
    for i in range(1, len(closes)):
        change = closes[i] - closes[i-1]
        if change > 0:
            gains.append(change)
            losses.append(0)
        else:
            gains.append(0)
            losses.append(abs(change))
    
    avg_gain = sum(gains[-period:]) / period
    avg_loss = sum(losses[-period:]) / period
    
    if avg_loss == 0:
        return 100
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_macd(closes: list) -> dict:
    """Calculate MACD (12, 26, 9)."""
    if len(closes) < 26:
        return None
    
    # Calculate EMAs
    ema12_values = []
    ema26_values = []
    
    # Initialize with SMA
    ema12 = sum(closes[:12]) / 12
    ema26 = sum(closes[:26]) / 26
    
    mult12 = 2 / 13
    mult26 = 2 / 27
    
    for i, price in enumerate(closes):
        if i >= 12:
            ema12 = (price - ema12) * mult12 + ema12
            ema12_values.append(ema12)
        if i >= 26:
            ema26 = (price - ema26) * mult26 + ema26
            ema26_values.append(ema26)
    
    if not ema26_values:
        return None
    
    # MACD line = EMA12 - EMA26
    macd_line = [ema12_values[i + 14] - ema26_values[i] for i in range(len(ema26_values))]
    
    if len(macd_line) < 9:
        return None
    
    # Signal line = 9-period EMA of MACD
    signal = sum(macd_line[:9]) / 9
    mult9 = 2 / 10
    for val in macd_line[9:]:
        signal = (val - signal) * mult9 + signal
    
    current_macd = macd_line[-1]
    histogram = current_macd - signal
    
    return {
        "macd": current_macd,
        "signal": signal,
        "histogram": histogram,
        "bullish": current_macd > signal
    }


def calculate_support_resistance(highs: list, lows: list, closes: list) -> dict:
    """Calculate support and resistance levels using pivot points."""
    if not highs or not lows or not closes:
        return None
    
    # Use typical pivot point method
    high = max(highs[-20:]) if len(highs) >= 20 else max(highs)
    low = min(lows[-20:]) if len(lows) >= 20 else min(lows)
    close = closes[-1]
    
    pivot = (high + low + close) / 3
    
    # Support levels
    s1 = (2 * pivot) - high
    s2 = pivot - (high - low)
    
    # Resistance levels
    r1 = (2 * pivot) - low
    r2 = pivot + (high - low)
    
    return {
        "pivot": pivot,
        "support": [s1, s2],
        "resistance": [r1, r2]
    }


def interpret_rsi(rsi: float) -> str:
    """Interpret RSI value."""
    if rsi >= 70:
        return "Overbought ⚠"
    elif rsi >= 60:
        return "Moderately High"
    elif rsi >= 40:
        return "Neutral"
    elif rsi >= 30:
        return "Moderately Low"
    else:
        return "Oversold ⚠"


def format_price(price: float) -> str:
    """Format price with appropriate decimals."""
    if price >= 1000:
        return f"${price:,.2f}"
    elif price >= 1:
        return f"${price:.2f}"
    else:
        return f"${price:.4f}"


class TechnicalAnalysisCommand(BaseCommand):
    """Technical analysis summary for a symbol."""
    name = "ta"
    aliases = ["technical", "analysis"]
    description = "Technical analysis summary"
    usage = "!ta AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            # Get 200 days of data for calculations
            bars = await self.providers.get_historical(symbol, "1y", "1d")
            if not bars or len(bars) < 50:
                return CommandResult.error(f"Insufficient data for {symbol}")
            
            closes = [b.close for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            current = closes[-1]
            
            # Calculate indicators
            sma20 = calculate_sma(closes, 20)
            sma50 = calculate_sma(closes, 50)
            sma200 = calculate_sma(closes, 200)
            rsi = calculate_rsi(closes)
            macd = calculate_macd(closes)
            levels = calculate_support_resistance(highs, lows, closes)
            
            # Determine trend
            bullish_signals = 0
            total_signals = 0
            
            if sma20 and sma50:
                total_signals += 1
                if current > sma50:
                    bullish_signals += 1
            
            if sma50 and sma200:
                total_signals += 1
                if sma50 > sma200:
                    bullish_signals += 1
            
            if rsi:
                total_signals += 1
                if 40 <= rsi <= 60:
                    bullish_signals += 0.5
                elif rsi < 30:
                    bullish_signals += 1  # Oversold = buy signal
                elif rsi > 70:
                    pass  # Overbought = sell signal
                else:
                    bullish_signals += 0.5
            
            if macd:
                total_signals += 1
                if macd["bullish"]:
                    bullish_signals += 1
            
            # Build output
            if sma50 and sma200:
                if current > sma50 > sma200:
                    trend = "▲ Bullish (above 50/200 SMA)"
                elif current < sma50 < sma200:
                    trend = "▼ Bearish (below 50/200 SMA)"
                else:
                    trend = "◇ Mixed"
            else:
                trend = "◇ Insufficient data"
            
            lines = [
                f"⊞ {symbol} Technical Analysis",
                "",
                f"Trend: {trend}",
            ]
            
            if rsi:
                lines.append(f"RSI(14): {rsi:.1f} — {interpret_rsi(rsi)}")
            
            if macd:
                macd_signal = "Bullish" if macd["bullish"] else "Bearish"
                lines.append(f"MACD: {macd_signal} (hist: {macd['histogram']:.3f})")
            
            if levels:
                s_str = " | ".join(format_price(s) for s in levels["support"])
                r_str = " | ".join(format_price(r) for r in levels["resistance"])
                lines.append("")
                lines.append(f"Support: {s_str}")
                lines.append(f"Resistance: {r_str}")
            
            # Overall signal
            if total_signals > 0:
                ratio = bullish_signals / total_signals
                if ratio >= 0.6:
                    signal = f"● Buy ({int(bullish_signals)}/{total_signals} bullish)"
                elif ratio <= 0.4:
                    signal = f"○ Sell ({int(bullish_signals)}/{total_signals} bullish)"
                else:
                    signal = f"◐ Hold ({int(bullish_signals)}/{total_signals} bullish)"
                lines.append("")
                lines.append(f"Signal: {signal}")
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except ProviderError as e:
            return CommandResult.error(f"Data unavailable: {e}")
        except Exception as e:
            return CommandResult.error(f"Analysis failed: {type(e).__name__}")


class RSICommand(BaseCommand):
    """RSI indicator with interpretation."""
    name = "rsi"
    aliases = []
    description = "RSI indicator"
    usage = "!rsi AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            bars = await self.providers.get_historical(symbol, "3mo", "1d")
            if not bars or len(bars) < 15:
                return CommandResult.error(f"Insufficient data for {symbol}")
            
            closes = [b.close for b in bars]
            rsi = calculate_rsi(closes)
            
            if rsi is None:
                return CommandResult.error(f"Cannot calculate RSI for {symbol}")
            
            # RSI interpretation
            interpretation = interpret_rsi(rsi)
            
            # Visual bar
            bar_width = 20
            filled = int(rsi / 100 * bar_width)
            bar = "█" * filled + "░" * (bar_width - filled)
            
            lines = [
                f"◈ {symbol} RSI(14)",
                "",
                f"Value: {rsi:.1f}",
                f"[{bar}]",
                f"Interpretation: {interpretation}",
                "",
                "30 = Oversold | 70 = Overbought"
            ]
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except Exception as e:
            return CommandResult.error(f"RSI calculation failed: {type(e).__name__}")


class SMACommand(BaseCommand):
    """Moving averages command."""
    name = "sma"
    aliases = ["ma", "moving"]
    description = "Simple Moving Averages"
    usage = "!sma AAPL [20 50 200]"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        # Parse periods
        periods = []
        for arg in ctx.args[1:]:
            try:
                p = int(arg)
                if 1 <= p <= 500:
                    periods.append(p)
            except ValueError:
                pass
        
        if not periods:
            periods = [20, 50, 200]  # Defaults
        
        try:
            bars = await self.providers.get_historical(symbol, "1y", "1d")
            if not bars or len(bars) < max(periods):
                return CommandResult.error(f"Insufficient data for {symbol}")
            
            closes = [b.close for b in bars]
            current = closes[-1]
            
            lines = [
                f"◈ {symbol} Moving Averages",
                f"Current: {format_price(current)}",
                ""
            ]
            
            for period in sorted(periods):
                sma = calculate_sma(closes, period)
                if sma:
                    diff_pct = ((current - sma) / sma) * 100
                    arrow = "▲" if current > sma else "▼"
                    lines.append(f"SMA{period}: {format_price(sma)} ({arrow} {diff_pct:+.1f}%)")
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except Exception as e:
            return CommandResult.error(f"SMA calculation failed: {type(e).__name__}")


class MACDCommand(BaseCommand):
    """MACD indicator."""
    name = "macd"
    aliases = []
    description = "MACD indicator"
    usage = "!macd AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            bars = await self.providers.get_historical(symbol, "3mo", "1d")
            if not bars or len(bars) < 35:
                return CommandResult.error(f"Insufficient data for {symbol}")
            
            closes = [b.close for b in bars]
            macd = calculate_macd(closes)
            
            if not macd:
                return CommandResult.error(f"Cannot calculate MACD for {symbol}")
            
            signal_text = "Bullish ▲" if macd["bullish"] else "Bearish ▼"
            
            lines = [
                f"◈ {symbol} MACD (12, 26, 9)",
                "",
                f"MACD Line: {macd['macd']:.3f}",
                f"Signal Line: {macd['signal']:.3f}",
                f"Histogram: {macd['histogram']:.3f}",
                "",
                f"Signal: {signal_text}",
            ]
            
            if macd["histogram"] > 0:
                lines.append("Momentum: Increasing ↑")
            else:
                lines.append("Momentum: Decreasing ↓")
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except Exception as e:
            return CommandResult.error(f"MACD calculation failed: {type(e).__name__}")


class SupportResistanceCommand(BaseCommand):
    """Support and resistance levels."""
    name = "support"
    aliases = ["levels", "sr"]
    description = "Support/Resistance levels"
    usage = "!support AAPL"
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error(f"Usage: {self.usage}")
        
        from ..utils import resolve_symbol
        symbol, _ = await resolve_symbol(ctx.args[0])
        
        try:
            bars = await self.providers.get_historical(symbol, "3mo", "1d")
            if not bars or len(bars) < 20:
                return CommandResult.error(f"Insufficient data for {symbol}")
            
            closes = [b.close for b in bars]
            highs = [b.high for b in bars]
            lows = [b.low for b in bars]
            current = closes[-1]
            
            levels = calculate_support_resistance(highs, lows, closes)
            
            if not levels:
                return CommandResult.error(f"Cannot calculate levels for {symbol}")
            
            lines = [
                f"◈ {symbol} Support/Resistance",
                f"Current: {format_price(current)}",
                "",
                f"Pivot: {format_price(levels['pivot'])}",
                "",
                "Resistance:",
                f"  R2: {format_price(levels['resistance'][1])}",
                f"  R1: {format_price(levels['resistance'][0])}",
                "",
                "Support:",
                f"  S1: {format_price(levels['support'][0])}",
                f"  S2: {format_price(levels['support'][1])}",
            ]
            
            return CommandResult.ok("\n".join(lines))
            
        except SymbolNotFoundError:
            return CommandResult.error(f"Symbol not found: {symbol}")
        except Exception as e:
            return CommandResult.error(f"Level calculation failed: {type(e).__name__}")
