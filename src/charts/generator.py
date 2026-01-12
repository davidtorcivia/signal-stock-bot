"""
Chart generator for stock price visualizations.

Generates price charts optimized for Signal messaging.
Uses matplotlib with Agg backend for headless rendering.
"""

import io
import base64
import logging
from datetime import datetime
from typing import Optional

import matplotlib
matplotlib.use('Agg')  # Headless backend

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter

from .themes import ChartTheme, get_theme
from ..providers import HistoricalBar

logger = logging.getLogger(__name__)


# Period/interval mappings for user-friendly input
# Using higher resolution for smoother charts
PERIOD_MAPPINGS = {
    "1d": ("1d", "5m"),      # 1 day, 5-min bars (~78 points)
    "5d": ("5d", "15m"),     # 5 days, 15-min bars (~130 points)
    "1w": ("5d", "15m"),     # 1 week (alias)
    "1m": ("1mo", "60m"),    # 1 month, hourly bars (~500 points)
    "3m": ("3mo", "60m"),    # 3 months, hourly (~1500 points)
    "6m": ("6mo", "1d"),     # 6 months, daily
    "1y": ("1y", "1d"),      # 1 year, daily (~252 points)
    "ytd": ("ytd", "1d"),    # Year to date
    "5y": ("5y", "1wk"),     # 5 years, weekly
    "max": ("max", "1mo"),   # All time, monthly
}


class ChartGenerator:
    """
    Generates stock charts as base64-encoded PNG images.
    
    Usage:
        generator = ChartGenerator()
        base64_img = await generator.generate(symbol, bars, period)
    """
    
    def __init__(
        self,
        theme: str = "dark",
        width: int = 800,
        height: int = 500,
        bot_name: str = "Stock Bot"
    ):
        self.theme = get_theme(theme)
        self.width = width
        self.height = height
        self.bot_name = bot_name
        self.dpi = 100
    
    def generate(
        self,
        symbol: str,
        bars: list[HistoricalBar],
        period: str = "1d",
        show_volume: bool = True,
        current_price: Optional[float] = None,
        change_percent: Optional[float] = None,
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
        
        Returns:
            Base64-encoded PNG string for Signal attachment
        """
        if not bars:
            raise ValueError("No data to chart")
        
        # Create figure with proper sizing
        fig_height = self.height / self.dpi
        fig_width = self.width / self.dpi
        
        if show_volume:
            # Two subplots: price (80%) and volume (20%)
            fig, (ax_price, ax_vol) = plt.subplots(
                2, 1,
                figsize=(fig_width, fig_height),
                height_ratios=[4, 1],
                sharex=True,
                facecolor=self.theme.background
            )
        else:
            fig, ax_price = plt.subplots(
                figsize=(fig_width, fig_height),
                facecolor=self.theme.background
            )
            ax_vol = None
        
        # Apply theme to axes
        self._style_axis(ax_price)
        if ax_vol:
            self._style_axis(ax_vol)
        
        # Extract data
        dates = [bar.timestamp for bar in bars]
        closes = [bar.close for bar in bars]
        volumes = [bar.volume for bar in bars]
        opens = [bar.open for bar in bars]
        
        # Determine if overall movement is up or down
        is_up = closes[-1] >= opens[0]
        line_color = self.theme.line_up if is_up else self.theme.line_down
        
        # Plot price line
        ax_price.plot(
            dates, closes,
            color=line_color,
            linewidth=self.theme.line_width,
            solid_capstyle='round'
        )
        
        # Auto-scale Y-axis to data range with padding (don't start at 0!)
        price_min = min(closes)
        price_max = max(closes)
        price_range = price_max - price_min
        padding = price_range * 0.10 if price_range > 0 else price_min * 0.05
        ax_price.set_ylim(price_min - padding, price_max + padding)
        
        # Fill area under curve (subtle gradient effect)
        ax_price.fill_between(
            dates, closes,
            y2=price_min - padding,  # Fill to bottom of visible area
            alpha=0.1,
            color=line_color
        )
        
        # Add grid
        ax_price.grid(
            True,
            color=self.theme.grid_color,
            alpha=self.theme.grid_alpha,
            linestyle=self.theme.grid_style,
            linewidth=0.5
        )
        
        # Y-axis formatting (price)
        ax_price.yaxis.set_major_formatter(
            FuncFormatter(lambda x, p: f"${x:,.2f}" if x >= 1 else f"${x:.4f}")
        )
        ax_price.tick_params(
            colors=self.theme.text_color,
            labelsize=self.theme.label_size
        )
        
        # Add current price line
        if current_price:
            ax_price.axhline(
                y=current_price,
                color=line_color,
                linestyle='--',
                linewidth=0.8,
                alpha=0.7
            )
        
        # Volume bars
        if ax_vol and volumes:
            # Color volume bars by price direction
            colors = []
            for i in range(len(closes)):
                if i == 0:
                    colors.append(self.theme.volume_up if is_up else self.theme.volume_down)
                else:
                    colors.append(
                        self.theme.volume_up if closes[i] >= closes[i-1]
                        else self.theme.volume_down
                    )
            
            ax_vol.bar(
                dates, volumes,
                color=colors,
                width=0.8 * (dates[1] - dates[0]) if len(dates) > 1 else 1,
                alpha=self.theme.volume_alpha
            )
            
            # Volume Y-axis formatting
            ax_vol.yaxis.set_major_formatter(
                FuncFormatter(lambda x, p: self._format_volume(x))
            )
            ax_vol.tick_params(
                colors=self.theme.text_color,
                labelsize=self.theme.label_size - 1
            )
            ax_vol.grid(
                True,
                color=self.theme.grid_color,
                alpha=self.theme.grid_alpha * 0.5,
                linestyle=self.theme.grid_style,
                linewidth=0.5
            )
        
        # X-axis date formatting
        bottom_ax = ax_vol if ax_vol else ax_price
        self._format_dates(bottom_ax, period, dates)
        
        # Title
        title = self._build_title(symbol, current_price, change_percent, period)
        ax_price.set_title(
            title,
            color=self.theme.title_color,
            fontsize=self.theme.title_size,
            fontweight='bold',
            loc='left',
            pad=10
        )
        
        # Watermark
        fig.text(
            0.99, 0.01,
            self.bot_name,
            ha='right', va='bottom',
            color=self.theme.watermark_color,
            alpha=self.theme.watermark_alpha,
            fontsize=8
        )
        
        # Tight layout
        plt.tight_layout()
        
        # Render to base64
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format='png',
            dpi=self.dpi,
            facecolor=self.theme.background,
            edgecolor='none',
            bbox_inches='tight',
            pad_inches=0.1
        )
        plt.close(fig)
        
        buf.seek(0)
        base64_data = base64.b64encode(buf.read()).decode('utf-8')
        
        logger.debug(f"Generated chart for {symbol}: {len(base64_data)} bytes")
        
        return base64_data
    
    def _style_axis(self, ax):
        """Apply theme styling to an axis."""
        ax.set_facecolor(self.theme.figure_background)
        
        # Spine styling
        for spine in ax.spines.values():
            spine.set_color(self.theme.axis_color)
            spine.set_linewidth(self.theme.spine_width)
        
        # Hide top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    
    def _format_dates(self, ax, period: str, dates: list):
        """Configure date axis formatting based on period."""
        if period in ("1d",):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M'))
            ax.xaxis.set_major_locator(mdates.HourLocator(interval=1))
        elif period in ("5d", "1w"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.DayLocator())
        elif period in ("1m", "3m"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
            ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
        elif period in ("6m", "1y", "ytd"):
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%b'))
            ax.xaxis.set_major_locator(mdates.MonthLocator())
        else:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
            ax.xaxis.set_major_locator(mdates.YearLocator())
        
        ax.tick_params(
            axis='x',
            colors=self.theme.text_color,
            labelsize=self.theme.label_size,
            rotation=0
        )
    
    def _format_volume(self, x: float) -> str:
        """Format volume numbers with K/M/B suffixes."""
        if x >= 1_000_000_000:
            return f'{x/1_000_000_000:.1f}B'
        elif x >= 1_000_000:
            return f'{x/1_000_000:.1f}M'
        elif x >= 1_000:
            return f'{x/1_000:.0f}K'
        return str(int(x))
    
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
