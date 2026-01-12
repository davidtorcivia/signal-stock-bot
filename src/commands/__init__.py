"""Commands package."""

from .base import BaseCommand, CommandContext, CommandResult
from .dispatcher import CommandDispatcher
from .stock_commands import (
    PriceCommand,
    QuoteCommand,
    FundamentalsCommand,
    MarketCommand,
    HelpCommand,
    StatusCommand,
    CryptoCommand,
    OptionCommand,
    ForexCommand,
    FuturesCommand,
    EconomyCommand,
    ProRequiredCommand,
    ChartCommand,
)
from .ta_commands import (
    TechnicalAnalysisCommand,
    TLDRCommand,
    RSICommand,
    SMACommand,
    MACDCommand,
    SupportResistanceCommand,
)
from .earnings_commands import (
    EarningsCommand,
    DividendCommand,
)
from .news_commands import (
    NewsCommand,
)
from .admin_commands import (
    MetricsCommand,
    CacheCommand,
    AdminCommand,
)
from .analytics_commands import (
    RatingCommand,
    InsiderCommand,
    ShortCommand,
    CorrelationCommand,
)
from .alert_commands import (
    AlertCommand,
)
from .watchlist_commands import (
    WatchCommand,
)

__all__ = [
    "BaseCommand",
    "CommandContext",
    "CommandResult",
    "CommandDispatcher",
    "PriceCommand",
    "QuoteCommand",
    "FundamentalsCommand",
    "MarketCommand",
    "HelpCommand",
    "StatusCommand",
    "CryptoCommand",
    "OptionCommand",
    "ForexCommand",
    "FuturesCommand",
    "EconomyCommand",
    "ProRequiredCommand",
    "ChartCommand",
    # TA commands
    "TechnicalAnalysisCommand",
    "TLDRCommand",
    "RSICommand",
    "SMACommand",
    "MACDCommand",
    "SupportResistanceCommand",
    # Earnings/Dividend
    "EarningsCommand",
    "DividendCommand",
    # News
    "NewsCommand",
    # Admin
    "MetricsCommand",
    "CacheCommand",
    "AdminCommand",
    # Analytics
    "RatingCommand",
    "InsiderCommand",
    "ShortCommand",
    "CorrelationCommand",
    # Alerts
    "AlertCommand",
    # Watchlist
    "WatchCommand",
]
