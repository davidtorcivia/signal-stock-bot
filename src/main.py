"""
Signal Stock Bot - Main entry point.

Usage:
    python -m src.main

Environment variables:
    SIGNAL_API_URL - URL of signal-cli-rest-api (default: http://localhost:8080)
    SIGNAL_PHONE_NUMBER - Bot's phone number (required)
    COMMAND_PREFIX - Command prefix (default: !)
    LOG_LEVEL - Logging level (default: INFO)
    
    ALPHAVANTAGE_API_KEY - Alpha Vantage API key (optional)
    POLYGON_API_KEY - Polygon API key (optional)
"""

import logging
import sys
import os

from .config import Config
from .providers import ProviderManager, YahooFinanceProvider, AlphaVantageProvider, MassiveProvider
from .commands import (
    CommandDispatcher,
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
    # TA commands
    TechnicalAnalysisCommand,
    TLDRCommand,
    RSICommand,
    SMACommand,
    MACDCommand,
    SupportResistanceCommand,
    # Earnings/Dividend
    EarningsCommand,
    DividendCommand,
    # News
    NewsCommand,
    # Admin
    MetricsCommand,
    CacheCommand,
)
from .signal import SignalHandler, SignalConfig
from .server import create_app


def setup_logging(level: str):
    """Configure logging for the application"""
    
    # Create logs directory if it doesn't exist
    os.makedirs("logs", exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("logs/bot.log"),
        ]
    )
    
    # Reduce noise from libraries
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def create_provider_manager(config: Config) -> ProviderManager:
    """Create and configure provider manager"""
    manager = ProviderManager()
    
    for provider_config in config.providers:
        if not provider_config.enabled:
            continue
        
        if provider_config.name == "yahoo":
            manager.add_provider(YahooFinanceProvider())
            
        elif provider_config.name == "alphavantage":
            if provider_config.api_key:
                manager.add_provider(AlphaVantageProvider(provider_config.api_key))
            else:
                logging.warning("Alpha Vantage configured but no API key provided")
        
        elif provider_config.name in ("massive", "polygon"):
            if provider_config.api_key:
                manager.add_provider(MassiveProvider(provider_config.api_key))
            else:
                logging.warning("Massive/Polygon configured but no API key provided")
    
    if not manager.providers:
        logging.warning("No providers configured! Adding Yahoo Finance as fallback.")
        manager.add_provider(YahooFinanceProvider())
    
    return manager


def create_dispatcher(provider_manager: ProviderManager, config: Config) -> CommandDispatcher:
    """Create and configure command dispatcher"""
    dispatcher = CommandDispatcher(prefix=config.command_prefix, bot_name=config.bot_name)
    
    # Create commands
    price_cmd = PriceCommand(provider_manager)
    quote_cmd = QuoteCommand(provider_manager)
    info_cmd = FundamentalsCommand(provider_manager)
    market_cmd = MarketCommand(provider_manager)
    status_cmd = StatusCommand(provider_manager)
    crypto_cmd = CryptoCommand(provider_manager)
    fx_cmd = ForexCommand(provider_manager)
    fut_cmd = FuturesCommand(provider_manager)
    
    # Register core commands
    dispatcher.register(price_cmd)
    dispatcher.register(quote_cmd)
    dispatcher.register(info_cmd)
    dispatcher.register(market_cmd)
    dispatcher.register(status_cmd)
    dispatcher.register(crypto_cmd)
    dispatcher.register(fx_cmd)
    dispatcher.register(fut_cmd)
    
    # Build command list for help
    help_commands = [price_cmd, quote_cmd, info_cmd, market_cmd, status_cmd, crypto_cmd, fx_cmd, fut_cmd]
    
    # Options and Economy commands require Massive Pro plan
    if config.massive_pro:
        opt_cmd = OptionCommand(provider_manager)
        eco_cmd = EconomyCommand(provider_manager)
        dispatcher.register(opt_cmd)
        dispatcher.register(eco_cmd)
        help_commands.extend([opt_cmd, eco_cmd])
    else:
        # Register stub commands that return helpful message
        opt_stub = ProRequiredCommand("option", ["opt", "o"], "Get option quote", "!opt TSLA230120C00150000")
        eco_stub = ProRequiredCommand("economy", ["eco", "macro"], "Get economic data", "!eco CPI")
        dispatcher.register(opt_stub)
        dispatcher.register(eco_stub)
    
    # Chart command - always available
    chart_cmd = ChartCommand(provider_manager, config.bot_name)
    dispatcher.register(chart_cmd)
    help_commands.append(chart_cmd)
    
    # Technical Analysis commands
    ta_cmd = TechnicalAnalysisCommand(provider_manager)
    tldr_cmd = TLDRCommand(provider_manager)
    rsi_cmd = RSICommand(provider_manager)
    sma_cmd = SMACommand(provider_manager)
    macd_cmd = MACDCommand(provider_manager)
    support_cmd = SupportResistanceCommand(provider_manager)
    for cmd in [ta_cmd, tldr_cmd, rsi_cmd, sma_cmd, macd_cmd, support_cmd]:
        dispatcher.register(cmd)
    help_commands.extend([ta_cmd, tldr_cmd, rsi_cmd, sma_cmd, macd_cmd, support_cmd])
    
    # Earnings and Dividend commands
    earnings_cmd = EarningsCommand(provider_manager)
    dividend_cmd = DividendCommand(provider_manager)
    dispatcher.register(earnings_cmd)
    dispatcher.register(dividend_cmd)
    help_commands.extend([earnings_cmd, dividend_cmd])
    
    # News command
    news_cmd = NewsCommand(provider_manager)
    dispatcher.register(news_cmd)
    help_commands.append(news_cmd)
    
    # Admin commands (available to everyone, can restrict via admin_numbers)
    metrics_cmd = MetricsCommand()
    cache_cmd = CacheCommand()
    dispatcher.register(metrics_cmd)
    dispatcher.register(cache_cmd)
    # Don't add to help_commands - these are admin-only
    
    # Help command needs list of all visible commands
    help_cmd = HelpCommand(help_commands, config.bot_name)
    dispatcher.register(help_cmd)
    
    return dispatcher


def main():
    """Main entry point"""
    
    # Load configuration
    config = Config.from_env()
    
    # Setup logging
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    
    logger.info("Starting Signal Stock Bot")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        sys.exit(1)
    
    # Set up providers
    provider_manager = create_provider_manager(config)
    logger.info(f"Configured {len(provider_manager.providers)} provider(s)")
    
    # Set up commands
    dispatcher = create_dispatcher(provider_manager, config)
    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")
    
    # Set up Signal handler
    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    logger.info(f"Signal handler configured for {config.signal_phone_number[-4:]}")
    
    # Create and run Flask app
    app = create_app(signal_handler)
    
    logger.info(f"Starting webhook server on {config.host}:{config.port}")
    app.run(
        host=config.host,
        port=config.port,
        debug=config.log_level.upper() == "DEBUG",
    )


def create_gunicorn_app():
    """
    Factory function for gunicorn to create the Flask app.
    
    Called by gunicorn with: gunicorn src.main:create_gunicorn_app
    """
    config = Config.from_env()
    setup_logging(config.log_level)
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Signal Stock Bot (gunicorn)")
    
    # Validate configuration
    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise RuntimeError(f"Configuration errors: {errors}")
    
    # Set up providers
    provider_manager = create_provider_manager(config)
    logger.info(f"Configured {len(provider_manager.providers)} provider(s)")
    
    # Set up commands
    dispatcher = create_dispatcher(provider_manager, config)
    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")
    
    # Set up Signal handler
    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    logger.info(f"Signal handler configured for {config.signal_phone_number[-4:]}")
    
    # Start message poller as fallback for webhooks
    # This actively polls signal-api since RECEIVE_WEBHOOK_URL is broken
    from .signal import SignalPoller
    
    poller = SignalPoller(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
        on_message=signal_handler.handle_webhook,
        poll_interval=1.0,  # Poll every second
    )
    poller.start()
    logger.info("Message poller started (fallback for webhooks)")
    
    # Create and return Flask app
    app = create_app(signal_handler)
    app.signal_poller = poller  # Keep reference to prevent GC
    return app



if __name__ == "__main__":
    main()
