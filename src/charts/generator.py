"""
Chart generator for stock price visualizations.

Uses mplfinance for professional-grade financial charts.
Automatically handles market hour gaps (no weekend/overnight jumps).
Supports technical indicators: SMA, Bollinger Bands, RSI.
"""

import io
import base64
import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Headless backend

import numpy as np
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


@dataclass
class ChartOptions:
    """Options for chart generation."""
    chart_type: str = "line"  # "line" or "candle"
    sma_periods: list[int] = field(default_factory=list)  # e.g., [20, 50, 200]
    bollinger: bool = False   # Add Bollinger Bands
    rsi: bool = False         # Add RSI panel
    show_volume: bool = True
    # Comparison symbol overlay
    comparison_symbol: Optional[str] = None
    comparison_bars: Optional[list] = None  # HistoricalBar list


# SMA line colors
SMA_COLORS = {
    20: '#FFD700',   # Gold
    50: '#00BFFF',   # Deep Sky Blue
    200: '#FF69B4',  # Hot Pink
}


def create_dark_style():
    """Create professional dark theme for mplfinance."""
    
    mc = mpf.make_marketcolors(
        up='#00C853',      # Green for up
        down='#FF1744',    # Red for down
        edge='inherit',
        wick='inherit',
        volume='inherit',
    )
    
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


def calculate_sma(closes: pd.Series, period: int) -> pd.Series:
    """Calculate Simple Moving Average."""
    return closes.rolling(window=period, min_periods=1).mean()


def calculate_bollinger_bands(closes: pd.Series, period: int = 20, std_dev: float = 2.0):
    """Calculate Bollinger Bands (middle, upper, lower)."""
    sma = closes.rolling(window=period, min_periods=1).mean()
    std = closes.rolling(window=period, min_periods=1).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return sma, upper, lower


def calculate_rsi(closes: pd.Series, period: int = 14) -> pd.Series:
    """Calculate Relative Strength Index."""
    delta = closes.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period, min_periods=1).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period, min_periods=1).mean()
    
    # Avoid division by zero
    rs = gain / loss.replace(0, np.inf)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)  # Default to neutral


class ChartGenerator:
    """
    Generates stock charts as base64-encoded PNG images.
    
    Uses mplfinance for professional financial charts that:
    - Properly handle market hour gaps (no weekend/overnight jumps)
    - Support candlestick and line chart styles
    - Include volume overlay
    - Support technical indicators (SMA, Bollinger, RSI)
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
        self._style = create_dark_style()
    
    def generate(
        self,
        symbol: str,
        bars: list[HistoricalBar],
        period: str = "1d",
        current_price: Optional[float] = None,
        change_percent: Optional[float] = None,
        options: Optional[ChartOptions] = None,
    ) -> str:
        """
        Generate a chart and return as base64-encoded PNG.
        
        Args:
            symbol: Stock symbol (e.g., "AAPL")
            bars: Historical OHLCV data
            period: Time period label (e.g., "1d", "1m")
            current_price: Current price for title
            change_percent: Percent change for title
            options: Chart options (indicators, chart type, etc.)
        
        Returns:
            Base64-encoded PNG string for Signal attachment
        """
        if not bars:
            raise ValueError("No data to chart")
        
        if options is None:
            options = ChartOptions()
        
        # Convert bars to pandas DataFrame
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
        
        # Build addplots for indicators
        addplots = []
        
        # SMA overlays
        for sma_period in options.sma_periods:
            sma = calculate_sma(df['Close'], sma_period)
            color = SMA_COLORS.get(sma_period, '#AAAAAA')
            addplots.append(mpf.make_addplot(
                sma,
                color=color,
                width=1.2,
                label=f'SMA{sma_period}'
            ))
        
        # Bollinger Bands
        if options.bollinger:
            bb_mid, bb_upper, bb_lower = calculate_bollinger_bands(df['Close'])
            addplots.append(mpf.make_addplot(bb_upper, color='#666666', width=0.8, linestyle='--'))
            addplots.append(mpf.make_addplot(bb_mid, color='#888888', width=0.8))
            addplots.append(mpf.make_addplot(bb_lower, color='#666666', width=0.8, linestyle='--'))
        
        # RSI panel
        panel_ratios = [4, 1]  # Price, Volume
        if options.rsi:
            rsi = calculate_rsi(df['Close'])
            addplots.append(mpf.make_addplot(
                rsi,
                panel=2,
                color='#9C27B0',
                ylabel='RSI',
                ylim=(0, 100),
            ))
            # RSI overbought/oversold lines
            rsi_70 = pd.Series([70] * len(df), index=df.index)
            rsi_30 = pd.Series([30] * len(df), index=df.index)
            addplots.append(mpf.make_addplot(rsi_70, panel=2, color='#FF5252', width=0.5, linestyle='--'))
            addplots.append(mpf.make_addplot(rsi_30, panel=2, color='#69F0AE', width=0.5, linestyle='--'))
            panel_ratios = [4, 1, 1.5]  # Price, Volume, RSI
        
        # Comparison symbol overlay (normalized percent returns)
        if options.comparison_bars and options.comparison_symbol:
            comp_df = pd.DataFrame([
                {'Date': bar.timestamp, 'Close': bar.close}
                for bar in options.comparison_bars
            ])
            comp_df.set_index('Date', inplace=True)
            comp_df.index = pd.DatetimeIndex(comp_df.index)
            
            # Align to main dataframe dates
            comp_df = comp_df.reindex(df.index, method='ffill')
            
            # Normalize both to percent return from start
            main_normalized = (df['Close'] / df['Close'].iloc[0] - 1) * 100
            comp_normalized = (comp_df['Close'] / comp_df['Close'].iloc[0] - 1) * 100
            
            # Add comparison as secondary y-axis with bright color
            addplots.append(mpf.make_addplot(
                comp_normalized,
                color='#00E5FF',  # Cyan
                width=1.5,
                linestyle='--',
                secondary_y=True,
                ylabel=f'{options.comparison_symbol} %',
            ))
        
        # Build title (will be placed above chart)
        title = self._build_title(
            symbol, current_price, change_percent, period,
            comparison_symbol=options.comparison_symbol if options.comparison_bars else None
        )
        
        # Figure size - increase height if RSI panel
        fig_width = self.width / self.dpi
        fig_height = (self.height + (100 if options.rsi else 0)) / self.dpi
        
        # Chart type
        mpf_type = "candle" if options.chart_type == "candle" else "line"
        
        # Create the plot
        buf = io.BytesIO()
        
        plot_kwargs = {
            'type': mpf_type,
            'style': self._style,
            'volume': options.show_volume,
            'figsize': (fig_width, fig_height),
            'returnfig': True,
            'show_nontrading': False,
            'tight_layout': True,
            'scale_padding': {'left': 0.1, 'right': 0.8, 'top': 0.8, 'bottom': 0.5},
        }
        
        if addplots:
            plot_kwargs['addplot'] = addplots
        
        if options.rsi:
            plot_kwargs['panel_ratios'] = panel_ratios
        
        fig, axes = mpf.plot(df, **plot_kwargs)
        
        # Add title above chart (not inside)
        fig.suptitle(
            title,
            color='#FFFFFF',
            fontsize=12,
            fontweight='bold',
            y=0.98,
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
        period: str,
        comparison_symbol: Optional[str] = None
    ) -> str:
        """Build chart title string."""
        parts = [symbol.upper()]
        
        # Add comparison symbol if present
        if comparison_symbol:
            parts[0] = f"{symbol.upper()} vs {comparison_symbol.upper()}"
        
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
