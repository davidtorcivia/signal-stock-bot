"""
Advanced analytics commands for stock bot.

Provides: !rating, !insider, !short, !corr
Uses yfinance for data.
"""

import asyncio
from datetime import datetime
from .base import BaseCommand, CommandContext, CommandResult
from ..providers import ProviderManager


class RatingCommand(BaseCommand):
    """Analyst ratings consensus."""
    name = "rating"
    aliases = ["ratings", "analyst"]
    description = "Analyst ratings consensus"
    usage = "!rating AAPL"
    help_explanation = """Shows Wall Street analyst ratings:

**What You See:**
- Buy/Hold/Sell counts from analysts
- Average target price
- Upside/downside potential

**Pro Tip:** Analyst targets lag reality. Use as sentiment indicator, not price prediction."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error("Specify a symbol: !rating AAPL")
        
        from ..utils import resolve_symbol
        symbol, name = await resolve_symbol(ctx.args[0])
        
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()
            
            def fetch():
                ticker = yf.Ticker(symbol)
                info = ticker.info
                recommendations = ticker.recommendations
                return info, recommendations
            
            info, recs = await loop.run_in_executor(None, fetch)
            
            lines = [f"◈ {name or symbol} ({symbol}) Analyst Ratings", ""]
            
            # Ratings breakdown
            buy = info.get("numberOfAnalystOpinions", 0)
            rec = info.get("recommendationKey", "").upper()
            target = info.get("targetMeanPrice")
            target_high = info.get("targetHighPrice")
            target_low = info.get("targetLowPrice")
            current = info.get("currentPrice") or info.get("regularMarketPrice")
            
            if rec:
                lines.append(f"Consensus: {rec}")
            if buy:
                lines.append(f"Analysts covering: {buy}")
            
            lines.append("")
            lines.append("━━━ Price Targets ━━━")
            
            if target and current:
                upside = ((target - current) / current) * 100
                indicator = "▲" if upside >= 0 else "▼"
                lines.append(f"Current: ${current:.2f}")
                lines.append(f"Target: ${target:.2f} ({indicator}{abs(upside):.1f}%)")
                if target_low and target_high:
                    lines.append(f"Range: ${target_low:.2f} - ${target_high:.2f}")
            else:
                lines.append("No price targets available")
            
            # Recent recommendations
            if recs is not None and len(recs) > 0:
                lines.append("")
                lines.append("━━━ Recent Actions ━━━")
                recent = recs.tail(5)
                for idx, row in recent.iterrows():
                    # Handle different yfinance versions (column names vary)
                    firm = row.get("Firm") or row.get("firm") or "Analyst"
                    grade = row.get("To Grade") or row.get("toGrade") or row.get("strongBuy") or ""
                    if firm and grade:
                        lines.append(f"  {str(firm)[:15]}: {grade}")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Rating lookup failed: {type(e).__name__}")


class InsiderCommand(BaseCommand):
    """Insider trading activity."""
    name = "insider"
    aliases = ["insiders"]
    description = "Recent insider transactions"
    usage = "!insider AAPL"
    help_explanation = """Shows recent insider buying/selling:

**What You See:**
- Who bought or sold (executives, directors)
- Transaction size and type
- Recent activity timeline

**Pro Tip:** Heavy insider buying often precedes good news. Selling is less reliable (tax, diversification)."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error("Specify a symbol: !insider AAPL")
        
        from ..utils import resolve_symbol
        symbol, name = await resolve_symbol(ctx.args[0])
        
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()
            
            def fetch():
                ticker = yf.Ticker(symbol)
                return ticker.insider_transactions
            
            transactions = await loop.run_in_executor(None, fetch)
            
            lines = [f"◈ {name or symbol} ({symbol}) Insider Activity", ""]
            
            if transactions is None or len(transactions) == 0:
                lines.append("No recent insider transactions found.")
                return CommandResult.ok("\n".join(lines))
            
            # Recent transactions
            recent = transactions.head(8)
            
            for idx, row in recent.iterrows():
                insider = str(row.get("Insider", "Unknown"))[:20]
                trans_type = row.get("Transaction", "")
                shares = row.get("Shares", 0)
                value = row.get("Value", 0)
                
                # Format transaction type
                if "Buy" in str(trans_type) or "Purchase" in str(trans_type):
                    indicator = "▲ BUY"
                elif "Sell" in str(trans_type) or "Sale" in str(trans_type):
                    indicator = "▼ SELL"
                else:
                    indicator = "◇"
                
                # Format value
                if value and value > 0:
                    if value >= 1_000_000:
                        val_str = f"${value/1_000_000:.1f}M"
                    elif value >= 1000:
                        val_str = f"${value/1000:.0f}K"
                    else:
                        val_str = f"${value:.0f}"
                else:
                    val_str = ""
                
                lines.append(f"{indicator} {insider}")
                if shares and val_str:
                    lines.append(f"   {shares:,} shares ({val_str})")
                lines.append("")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Insider lookup failed: {type(e).__name__}")


class ShortCommand(BaseCommand):
    """Short interest data."""
    name = "short"
    aliases = ["si", "shorts"]
    description = "Short interest data"
    usage = "!short AAPL"
    help_explanation = """Shows short selling metrics:

**Metrics:**
- Short % of Float: How much is shorted
- Short Ratio: Days to cover at avg volume
- Short Interest: Total shares shorted

**Pro Tip:** High short interest (>20%) creates squeeze potential, but also indicates bearish sentiment."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not ctx.args:
            return CommandResult.error("Specify a symbol: !short AAPL")
        
        from ..utils import resolve_symbol
        symbol, name = await resolve_symbol(ctx.args[0])
        
        try:
            import yfinance as yf
            loop = asyncio.get_event_loop()
            
            def fetch():
                ticker = yf.Ticker(symbol)
                return ticker.info
            
            info = await loop.run_in_executor(None, fetch)
            
            lines = [f"◈ {name or symbol} ({symbol}) Short Interest", ""]
            
            short_pct = info.get("shortPercentOfFloat")
            short_ratio = info.get("shortRatio")
            shares_short = info.get("sharesShort")
            shares_short_prev = info.get("sharesShortPreviousMonthDate")
            float_shares = info.get("floatShares")
            
            if short_pct:
                pct = short_pct * 100
                indicator = "▲" if pct > 15 else "◇"
                lines.append(f"Short % of Float: {indicator} {pct:.1f}%")
            
            if short_ratio:
                lines.append(f"Days to Cover: {short_ratio:.1f}")
            
            if shares_short:
                if shares_short >= 1_000_000:
                    short_str = f"{shares_short/1_000_000:.1f}M"
                else:
                    short_str = f"{shares_short/1000:.0f}K"
                lines.append(f"Shares Short: {short_str}")
            
            if float_shares:
                if float_shares >= 1_000_000_000:
                    float_str = f"{float_shares/1_000_000_000:.1f}B"
                elif float_shares >= 1_000_000:
                    float_str = f"{float_shares/1_000_000:.0f}M"
                else:
                    float_str = f"{float_shares/1000:.0f}K"
                lines.append(f"Float: {float_str}")
            
            if not short_pct and not short_ratio:
                lines.append("Short data not available for this symbol.")
            else:
                # Interpretation
                lines.append("")
                if short_pct and short_pct > 0.2:
                    lines.append("» High short interest - squeeze potential")
                elif short_pct and short_pct > 0.1:
                    lines.append("» Elevated short interest")
                elif short_pct:
                    lines.append("» Normal short levels")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Short interest lookup failed: {type(e).__name__}")


class CorrelationCommand(BaseCommand):
    """Calculate correlation between two symbols."""
    name = "corr"
    aliases = ["correlation", "beta"]
    description = "30-day correlation coefficient"
    usage = "!corr AAPL SPY"
    help_explanation = """Calculates price correlation between symbols:

**Values:**
- +1.0: Move together perfectly
- 0: No relationship
- -1.0: Move opposite

**Pro Tip:** Diversify with low/negative correlations. High correlation = similar risk exposure."""
    
    def __init__(self, provider_manager: ProviderManager):
        self.providers = provider_manager
    
    async def execute(self, ctx: CommandContext) -> CommandResult:
        if len(ctx.args) < 2:
            return CommandResult.error("Specify two symbols: !corr AAPL SPY")
        
        from ..utils import resolve_symbol
        symbol1, name1 = await resolve_symbol(ctx.args[0])
        symbol2, name2 = await resolve_symbol(ctx.args[1])
        
        try:
            import yfinance as yf
            import numpy as np
            loop = asyncio.get_event_loop()
            
            def fetch():
                t1 = yf.Ticker(symbol1)
                t2 = yf.Ticker(symbol2)
                h1 = t1.history(period="1mo")
                h2 = t2.history(period="1mo")
                return h1, h2
            
            h1, h2 = await loop.run_in_executor(None, fetch)
            
            if len(h1) < 5 or len(h2) < 5:
                return CommandResult.error("Insufficient data for correlation")
            
            # Calculate returns
            r1 = h1["Close"].pct_change().dropna()
            r2 = h2["Close"].pct_change().dropna()
            
            # Align dates
            common = r1.index.intersection(r2.index)
            if len(common) < 5:
                return CommandResult.error("Insufficient overlapping data")
            
            r1 = r1.loc[common]
            r2 = r2.loc[common]
            
            # Calculate correlation
            correlation = np.corrcoef(r1, r2)[0, 1]
            
            lines = [f"◈ Correlation: {symbol1} vs {symbol2}", ""]
            
            # Visual bar
            bar_pos = int((correlation + 1) / 2 * 10)  # -1 to +1 → 0 to 10
            bar = "░" * bar_pos + "█" + "░" * (10 - bar_pos)
            lines.append(f"-1 [{bar}] +1")
            lines.append("")
            lines.append(f"Coefficient: {correlation:+.3f}")
            lines.append(f"Period: 30 days ({len(common)} trading days)")
            
            # Interpretation
            lines.append("")
            if correlation > 0.8:
                lines.append("» Very high correlation - move together")
            elif correlation > 0.5:
                lines.append("» Moderate positive correlation")
            elif correlation > -0.5:
                lines.append("» Low correlation - somewhat independent")
            elif correlation > -0.8:
                lines.append("» Moderate negative correlation")
            else:
                lines.append("» High negative correlation - move opposite")
            
            return CommandResult.ok("\n".join(lines))
            
        except Exception as e:
            return CommandResult.error(f"Correlation failed: {type(e).__name__}")
