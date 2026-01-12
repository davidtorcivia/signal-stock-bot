"""
Chart color themes and styling constants.
"""

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ChartTheme:
    """Base chart theme configuration."""
    
    # Canvas
    background: str
    figure_background: str
    
    # Price line
    line_up: str      # Color when price is up
    line_down: str    # Color when price is down
    line_width: float
    
    # Volume bars
    volume_up: str
    volume_down: str
    volume_alpha: float
    
    # Grid
    grid_color: str
    grid_alpha: float
    grid_style: str
    
    # Text
    text_color: str
    title_color: str
    label_size: int
    title_size: int
    
    # Axes
    axis_color: str
    spine_width: float
    
    # Watermark
    watermark_color: str
    watermark_alpha: float


# Professional dark theme - optimized for Signal
DarkTheme = ChartTheme(
    # Canvas - pure black for OLED
    background="#000000",
    figure_background="#000000",
    
    # Price line
    line_up="#00C853",      # Material Green A400
    line_down="#FF1744",    # Material Red A400
    line_width=2.0,
    
    # Volume
    volume_up="#00C85380",   # Green with alpha
    volume_down="#FF174480", # Red with alpha
    volume_alpha=0.5,
    
    # Grid - subtle
    grid_color="#333333",
    grid_alpha=0.3,
    grid_style=":",
    
    # Text
    text_color="#AAAAAA",
    title_color="#FFFFFF",
    label_size=9,
    title_size=12,
    
    # Axes
    axis_color="#444444",
    spine_width=0.5,
    
    # Watermark
    watermark_color="#333333",
    watermark_alpha=0.5,
)


# Light theme for preference
LightTheme = ChartTheme(
    background="#FFFFFF",
    figure_background="#FAFAFA",
    
    line_up="#2E7D32",      # Green 800
    line_down="#C62828",    # Red 800
    line_width=2.0,
    
    volume_up="#2E7D3260",
    volume_down="#C6282860",
    volume_alpha=0.4,
    
    grid_color="#E0E0E0",
    grid_alpha=0.5,
    grid_style="-",
    
    text_color="#424242",
    title_color="#212121",
    label_size=9,
    title_size=12,
    
    axis_color="#BDBDBD",
    spine_width=0.5,
    
    watermark_color="#EEEEEE",
    watermark_alpha=0.7,
)


def get_theme(name: str = "dark") -> ChartTheme:
    """Get theme by name."""
    themes = {
        "dark": DarkTheme,
        "light": LightTheme,
    }
    return themes.get(name.lower(), DarkTheme)
