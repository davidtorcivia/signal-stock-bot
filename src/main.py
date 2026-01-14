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
from pathlib import Path

from dotenv import load_dotenv

# Robustly find .env file
# Try current directory first, then project root (up one level from src/)
current_dir = Path(os.getcwd())
root_env = current_dir / ".env"
src_env = Path(__file__).parent.parent / ".env"

if root_env.exists():
    load_dotenv(root_env)
    print(f"Loaded configuration from {root_env}")
elif src_env.exists():
    load_dotenv(src_env)
    print(f"Loaded configuration from {src_env}")
else:
    print("WARNING: No .env file found! Configuration will rely on system environment variables.")

# Debug loaded variables
if os.getenv("FRED_API_KEY"):
    print("FRED_API_KEY: Found")
else:
    print("FRED_API_KEY: NOT FOUND")

if os.getenv("ADMIN_NUMBERS"):
    print(f"ADMIN_NUMBERS: Found ({len(os.getenv('ADMIN_NUMBERS').split(','))} numbers)")
else:
    print("ADMIN_NUMBERS: NOT FOUND")

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
    AdminCommand,
    # Analytics
    RatingCommand,
    InsiderCommand,
    ShortCommand,
    CorrelationCommand,
    # Alerts
    AlertCommand,
    # Watchlist
    WatchCommand,
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
    from .providers import FinnhubProvider, TwelveDataProvider, FredProvider
    
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
        
        elif provider_config.name == "finnhub":
            if provider_config.api_key:
                manager.add_provider(FinnhubProvider(provider_config.api_key))
            else:
                logging.warning("Finnhub configured but no API key provided")
        
        elif provider_config.name == "twelvedata":
            if provider_config.api_key:
                manager.add_provider(TwelveDataProvider(provider_config.api_key))
            else:
                logging.warning("Twelve Data configured but no API key provided")
        
        elif provider_config.name == "fred":
            if provider_config.api_key:
                manager.add_provider(FredProvider(provider_config.api_key))
            else:
                logging.warning("FRED configured but no API key provided")
    
    if not manager.providers:
        logging.warning("No providers configured! Adding Yahoo Finance as fallback.")
        manager.add_provider(YahooFinanceProvider())
    
    return manager


def create_dispatcher(provider_manager: ProviderManager, config: Config, watchlist_db=None, alerts_db=None, context_manager=None) -> CommandDispatcher:
    """Create and configure command dispatcher"""
    dispatcher = CommandDispatcher(
        prefix=config.command_prefix, 
        bot_name=config.bot_name,
        rate_limit=config.user_rate_limit,
        context_manager=context_manager,
    )
    
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
    
    # Options command requires Massive Pro plan (not available via free providers)
    if config.massive_pro:
        opt_cmd = OptionCommand(provider_manager)
        dispatcher.register(opt_cmd)
        help_commands.append(opt_cmd)
    else:
        # Register stub command that returns helpful message
        opt_stub = ProRequiredCommand("option", ["opt", "o"], "Get option quote", "!opt TSLA230120C00150000")
        dispatcher.register(opt_stub)
    
    # Economy command - now available to all via FRED (free) fallback
    eco_cmd = EconomyCommand(provider_manager)
    dispatcher.register(eco_cmd)
    help_commands.append(eco_cmd)
    
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
    
    # Watchlist command
    if watchlist_db:
        watch_cmd = WatchCommand(provider_manager, watchlist_db)
        dispatcher.register(watch_cmd)
        help_commands.append(watch_cmd)
    
    # Analytics commands
    rating_cmd = RatingCommand(provider_manager)
    insider_cmd = InsiderCommand(provider_manager)
    short_cmd = ShortCommand(provider_manager)
    corr_cmd = CorrelationCommand(provider_manager)
    dispatcher.register(rating_cmd)
    dispatcher.register(insider_cmd)
    dispatcher.register(short_cmd)
    dispatcher.register(corr_cmd)
    help_commands.extend([rating_cmd, insider_cmd, short_cmd, corr_cmd])
    
    # Alert command
    if alerts_db:
        alert_cmd = AlertCommand(provider_manager, alerts_db)
        dispatcher.register(alert_cmd)
        help_commands.append(alert_cmd)
    
    # Admin commands (restrict via admin_numbers if configured)
    admin_numbers = config.admin_numbers
    metrics_cmd = MetricsCommand(admin_numbers=admin_numbers)
    cache_cmd = CacheCommand(admin_numbers=admin_numbers)
    admin_cmd = AdminCommand(
        admin_numbers=admin_numbers,
        watchlist_db=watchlist_db,
    )
    dispatcher.register(metrics_cmd)
    dispatcher.register(cache_cmd)
    dispatcher.register(admin_cmd)
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
    
    # Set up database
    from .database import WatchlistDB, AlertsDB
    from .context import ContextManager
    
    watchlist_db = WatchlistDB(config.watchlist_db_path)
    alerts_db = AlertsDB(config.watchlist_db_path)
    context_manager = ContextManager(config.watchlist_db_path)
    
    logger.info(f"Database: {config.watchlist_db_path}")
    
    # Set up commands
    dispatcher = create_dispatcher(provider_manager, config, watchlist_db, alerts_db, context_manager)
    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")
    
    # Set up Signal handler
    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    logger.info(f"Signal handler configured for {config.signal_phone_number[-4:]}")
    
    # Start background alert worker
    if alerts_db:
        import threading
        import asyncio
        import time
        
        def run_alert_worker():
            logger.info("Starting background alert worker")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def check_alerts():
                while True:
                    try:
                        alerts = await alerts_db.get_all_active_alerts()
                        if not alerts:
                            await asyncio.sleep(60)
                            continue
                            
                        logger.debug(f"Checking {len(alerts)} active alerts...")
                        
                        # Group by symbol to batch requests
                        by_symbol = {}
                        for alert in alerts:
                            sym = alert['symbol']
                            if sym not in by_symbol:
                                by_symbol[sym] = []
                            by_symbol[sym].append(alert)
                        
                        # Check each symbol
                        for symbol, symbol_alerts in by_symbol.items():
                            try:
                                quote = await provider_manager.get_quote(symbol)
                                current_price = quote.price
                                
                                for alert in symbol_alerts:
                                    triggered = False
                                    condition = alert['condition']
                                    target = alert['target_value']
                                    
                                    if condition == 'above' and current_price > target:
                                        triggered = True
                                        msg = f"🔔 ALERT: {symbol} is above ${target:.2f} (Current: ${current_price:.2f})"
                                    elif condition == 'below' and current_price < target:
                                        triggered = True
                                        msg = f"🔔 ALERT: {symbol} is below ${target:.2f} (Current: ${current_price:.2f})"
                                    elif condition == 'change_pct' and abs(quote.change_percent) >= target:
                                        triggered = True
                                        msg = f"🔔 ALERT: {symbol} moved {quote.change_percent:+.2f}% (Target: {target}%)"
                                    
                                    if triggered:
                                        logger.info(f"Alert triggered: {alert['id']} for {symbol}")
                                        # Deactivate alert first
                                        await alerts_db.trigger_alert(alert['id'])
                                        
                                        # Send notification
                                        group_id = alert.get('group_id')
                                        recipient = alert['user_phone']
                                        
                                        if group_id:
                                            # Send to group. We still pass recipient (source) but it's ignored for group sends usually
                                            # or needed for some logic. send_message signature requires recipient.
                                            await signal_handler.send_message(recipient=recipient, message=msg, group_id=group_id)
                                        else:
                                            # DM - send directly to user
                                            await signal_handler.send_message(recipient=recipient, message=msg, group_id=None)
                                            
                            except Exception as e:
                                logger.error(f"Error checking {symbol}: {e}")
                                
                        await asyncio.sleep(60)
                        
                    except Exception as e:
                        logger.error(f"Alert worker error: {e}")
                        await asyncio.sleep(60)

            loop.run_until_complete(check_alerts())

        # Start thread
        t = threading.Thread(target=run_alert_worker, daemon=True)
        t.start()
    
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
    
    # Set up database
    from .database import WatchlistDB, AlertsDB
    from .context import ContextManager
    
    watchlist_db = WatchlistDB(config.watchlist_db_path)
    alerts_db = AlertsDB(config.watchlist_db_path)
    context_manager = ContextManager(config.watchlist_db_path)
    logger.info(f"Database: {config.watchlist_db_path}")
    
    # Set up commands
    dispatcher = create_dispatcher(provider_manager, config, watchlist_db, alerts_db, context_manager)
    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")
    
    # Set up Signal handler
    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    logger.info(f"Signal handler configured for {config.signal_phone_number[-4:]}")
    
    # Start background alert worker
    if alerts_db:
        import threading
        import asyncio
        
        def run_alert_worker():
            logger.info("Starting background alert worker")
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            async def check_alerts():
                while True:
                    try:
                        alerts = await alerts_db.get_all_active_alerts()
                        if not alerts:
                            await asyncio.sleep(60)
                            continue
                            
                        logger.debug(f"Checking {len(alerts)} active alerts...")
                        
                        # Group by symbol to batch requests
                        by_symbol = {}
                        for alert in alerts:
                            sym = alert['symbol']
                            if sym not in by_symbol:
                                by_symbol[sym] = []
                            by_symbol[sym].append(alert)
                        
                        # Check each symbol
                        for symbol, symbol_alerts in by_symbol.items():
                            try:
                                quote = await provider_manager.get_quote(symbol)
                                current_price = quote.price
                                
                                for alert in symbol_alerts:
                                    triggered = False
                                    condition = alert['condition']
                                    target = alert['target_value']
                                    
                                    if condition == 'above' and current_price > target:
                                        triggered = True
                                        msg = f">>> ALERT: {symbol} is above ${target:.2f} (Current: ${current_price:.2f})"
                                    elif condition == 'below' and current_price < target:
                                        triggered = True
                                        msg = f">>> ALERT: {symbol} is below ${target:.2f} (Current: ${current_price:.2f})"
                                    elif condition == 'change_pct' and abs(quote.change_percent) >= target:
                                        triggered = True
                                        msg = f">>> ALERT: {symbol} moved {quote.change_percent:+.2f}% (Target: {target}%)"
                                    
                                    if triggered:
                                        logger.info(f"Alert triggered: {alert['id']} for {symbol}")
                                        # Deactivate alert first
                                        await alerts_db.trigger_alert(alert['id'])
                                        
                                        # Send notification
                                        group_id = alert.get('group_id')
                                        recipient = alert['user_phone']
                                        
                                        if group_id:
                                            await signal_handler.send_message(recipient=recipient, message=msg, group_id=group_id)
                                        else:
                                            await signal_handler.send_message(recipient=recipient, message=msg, group_id=None)
                                            
                            except Exception as e:
                                logger.error(f"Error checking {symbol}: {e}")
                                
                        await asyncio.sleep(60)
                        
                    except Exception as e:
                        logger.error(f"Alert worker error: {e}")
                        await asyncio.sleep(60)

            loop.run_until_complete(check_alerts())

        # Start thread
        t = threading.Thread(target=run_alert_worker, daemon=True)
        t.start()
        logger.info("Background alert worker started")
    
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
