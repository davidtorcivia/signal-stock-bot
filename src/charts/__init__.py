"""
Chart generation module for stock visualizations.
Uses mplfinance for professional financial charts.
"""

from .generator import ChartGenerator, get_period_params

__all__ = [
    "ChartGenerator",
    "get_period_params",
]

