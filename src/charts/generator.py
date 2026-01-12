"""
Chart generator for stock price visualizations.

Uses mplfinance for professional-grade financial charts.
Automatically handles market hour gaps (no weekend/overnight jumps).
"""

import io
import base64
import logging
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Headless backend

import pandas as pd
import mplfinance as mpf

from ..providers import HistoricalBar

logger = logging.getLogger(__name__)


# Period/interval mappings for user-friendly input
PERIOD_MAPPINGS = {
    "1d": ("1d", "5m"),      # 1 day, 5-min bars
    "5d": ("5d", "15m"),     # 5 days, 15-min bars
    "1w": ("5d", "15m"),     # 1 week (alias)
    "1m": ("1mo", "1h"),     # 1 month, hourly bars
    "3m": ("3mo", "1d"),     # 3 months, daily
    "6m": ("6mo", "1d"),     # 6 months, daily
    "1y": ("1y", "1d"),      # 1 year, daily
    "ytd": ("ytd", "1d"),    # Year to date
    "5y": ("5y", "1wk"),     # 5 years, weekly
    "max": ("max", "1mo"),   # All time, monthly
}


def create_dark_style(bot_name: str = "Stock Bot"):
    """Create professional dark theme for mplfinance."""
    
    # Market colors
    mc = mpf.make_marketcolors(
        up='#00C853',      # Green for up
        down='#FF1744',    # Red for down
        edge='inherit',
        wick='inherit',
        volume='inherit',
    )
    
    # Chart style
    style = mpf.make_mpf_style(
        base_mpf_style='nightclouds',
        marketcolors=mc,
        facecolor='#000000',
        edgecolor='#333333',
        figcolor='#000000',
        gridcolor='#222222',
        gridstyle=':',
        gridaxis='both',
        y_on_right=False,
        rc={
            'axes.labelcolor': '#888888',
            'axes.titlecolor': '#FFFFFF',
            'xtick.color': '#888888',
            'ytick.color': '#888888',
            'font.size': 9,
        }
    )
    
    return style


class ChartGenerator:
    """
    Generates stock charts as base64-encoded PNG images.
    
    Uses mplfinance for professional financial charts that:
    - Properly handle market hour gaps (no weekend/overnight jumps)
    - Support candlestick and line chart styles
    - Include volume overlay
    """
    
    def __init__(
        self,
        theme: str = "dark",
        width: int = 800,
        height: int = 500,
        bot_name: str = "Stock Bot"
    ):
        self.theme = theme
        self.width = width
        self.height = height
        self.bot_name = bot_name
        self.dpi = 100
        self._style = create_dark_style(bot_name)
    
    def generate(
        self,
        symbol: str,
        bars: list[HistoricalBar],
        period: str = "1d",
        show_volume: bool = True,
        current_price: Optional[float] = None,
        change_percent: Optional[float] = None,
        chart_type: str = "line",  # "line" or "candle"
    ) -> str:
        """
        Generate a chart and return as base64-encoded PNG.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            bars: Historical OHLCV data
            period: Time period label (e.g., "1d", "1m")
            show_volume: Whether to show volume bars
            current_price: Current price for title
            change_percent: Percent change for title
            chart_type: "line" for line chart, "candle" for candlestick
        
        Returns:
            Base64-encoded PNG string for Signal attachment
        """
        if not bars:
            raise ValueError("No data to chart")
        
        # Convert bars to pandas DataFrame (required by mplfinance)
        df = pd.DataFrame([
            {
                'Date': bar.timestamp,
                'Open': bar.open,
                'High': bar.high,
                'Low': bar.low,
                'Close': bar.close,
                'Volume': bar.volume,
            }
            for bar in bars
        ])
        df.set_index('Date', inplace=True)
        df.index = pd.DatetimeIndex(df.index)
        
        # Build title
        title = self._build_title(symbol, current_price, change_percent, period)
        
        # Figure size
        fig_width = self.width / self.dpi
        fig_height = self.height / self.dpi
        
        # Determine chart type
        if chart_type == "candle":
            mpf_type = "candle"
        else:
            mpf_type = "line"
        
        # Create the plot
        buf = io.BytesIO()
        
        # mplfinance automatically handles gaps with show_nontrading=False (default)
        fig, axes = mpf.plot(
            df,
            type=mpf_type,
            style=self._style,
            title=title,
            volume=show_volume,
            figsize=(fig_width, fig_height),
            returnfig=True,
            show_nontrading=False,  # Skip weekends/holidays (industry standard)
            tight_layout=True,
            scale_padding={'left': 0.1, 'right': 0.8, 'top': 0.6, 'bottom': 0.5},
        )
        
        # Add watermark
        fig.text(
            0.99, 0.01,
            self.bot_name,
            ha='right', va='bottom',
            color='#444444',
            fontsize=8,
            alpha=0.7,
        )
        
        # Save to buffer
        fig.savefig(
            buf,
            format='png',
            dpi=self.dpi,
            facecolor='#000000',
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1,
        )
        
        import matplotlib.pyplot as plt
        plt.close(fig)
        
        buf.seek(0)
        base64_data = base64.b64encode(buf.read()).decode('utf-8')
        
        logger.debug(f"Generated chart for {symbol}: {len(base64_data)} bytes")
        
        return base64_data
    
    def _build_title(
        self,
        symbol: str,
        price: Optional[float],
        change_percent: Optional[float],
        period: str
    ) -> str:
        """Build chart title string."""
        parts = [symbol.upper()]
        
        if price is not None:
            parts.append(f"${price:,.2f}" if price >= 1 else f"${price:.4f}")
        
        if change_percent is not None:
            sign = "+" if change_percent >= 0 else ""
            indicator = "▲" if change_percent >= 0 else "▼"
            parts.append(f"{indicator} {sign}{change_percent:.2f}%")
        
        # Period label
        period_labels = {
            "1d": "1D", "5d": "5D", "1w": "1W", "1m": "1M",
            "3m": "3M", "6m": "6M", "1y": "1Y", "ytd": "YTD",
            "5y": "5Y", "max": "MAX"
        }
        parts.append(f"({period_labels.get(period, period.upper())})")
        
        return " · ".join(parts)


def get_period_params(period: str) -> tuple[str, str]:
    """
    Convert user-friendly period to provider parameters.
    
    Returns:
        (period, interval) tuple for provider.get_historical()
    """
    return PERIOD_MAPPINGS.get(period.lower(), ("1mo", "1d"))
