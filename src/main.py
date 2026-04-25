"""
Signal Stock Bot - Main entry point.

Usage:
    python -m src.main                        # dev: Flask dev server
    gunicorn src.main:create_gunicorn_app()   # prod

Environment variables: see .env.example
"""

import asyncio
import logging
import os
import sys
import threading
from logging.handlers import RotatingFileHandler
from pathlib import Path

from dotenv import load_dotenv

# Robustly find .env file
current_dir = Path(os.getcwd())
root_env = current_dir / ".env"
src_env = Path(__file__).parent.parent / ".env"

if root_env.exists():
    load_dotenv(root_env)
elif src_env.exists():
    load_dotenv(src_env)

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
    TechnicalAnalysisCommand,
    TLDRCommand,
    RSICommand,
    SMACommand,
    MACDCommand,
    SupportResistanceCommand,
    EarningsCommand,
    DividendCommand,
    NewsCommand,
    MetricsCommand,
    CacheCommand,
    AdminCommand,
    RatingCommand,
    InsiderCommand,
    ShortCommand,
    CorrelationCommand,
    AlertCommand,
    WatchCommand,
    AskCommand,
)
from .signal import SignalHandler, SignalConfig, SignalPoller
from .server import create_app


def setup_logging(level: str):
    """Configure logging for the application with rotation."""
    os.makedirs("logs", exist_ok=True)

    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    file_handler = RotatingFileHandler(
        "logs/bot.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
    )
    file_handler.setFormatter(formatter)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)

    root = logging.getLogger()
    root.setLevel(getattr(logging, level.upper(), logging.INFO))
    # Replace any handlers added by prior logging.basicConfig calls
    root.handlers.clear()
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)


def create_provider_manager(config: Config) -> ProviderManager:
    """Create and configure provider manager."""
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


def create_dispatcher(
    provider_manager: ProviderManager,
    config: Config,
    watchlist_db=None,
    alerts_db=None,
    context_manager=None,
    settings_store=None,
    ask_command=None,
    group_log=None,
    context_registry=None,
    llm_client=None,
) -> CommandDispatcher:
    """Create and configure command dispatcher."""
    dispatcher = CommandDispatcher(
        prefix=config.command_prefix,
        bot_name=config.bot_name,
        rate_limit=config.user_rate_limit,
        context_manager=context_manager,
        max_message_length=config.max_message_length,
        settings_store=settings_store,
        group_log=group_log,
        context_registry=context_registry,
        llm_client=llm_client,
        ask_command=ask_command,
    )

    price_cmd = PriceCommand(provider_manager)
    quote_cmd = QuoteCommand(provider_manager)
    info_cmd = FundamentalsCommand(provider_manager)
    market_cmd = MarketCommand(provider_manager)
    status_cmd = StatusCommand(provider_manager)
    crypto_cmd = CryptoCommand(provider_manager)
    fx_cmd = ForexCommand(provider_manager)
    fut_cmd = FuturesCommand(provider_manager)

    dispatcher.register(price_cmd)
    dispatcher.register(quote_cmd)
    dispatcher.register(info_cmd)
    dispatcher.register(market_cmd)
    dispatcher.register(status_cmd)
    dispatcher.register(crypto_cmd)
    dispatcher.register(fx_cmd)
    dispatcher.register(fut_cmd)

    help_commands = [price_cmd, quote_cmd, info_cmd, market_cmd, status_cmd, crypto_cmd, fx_cmd, fut_cmd]

    if config.massive_pro:
        opt_cmd = OptionCommand(provider_manager)
        dispatcher.register(opt_cmd)
        help_commands.append(opt_cmd)
    else:
        opt_stub = ProRequiredCommand("option", ["opt", "o"], "Get option quote", "!opt TSLA230120C00150000")
        dispatcher.register(opt_stub)

    eco_cmd = EconomyCommand(provider_manager, config.bot_name)
    dispatcher.register(eco_cmd)
    help_commands.append(eco_cmd)

    chart_cmd = ChartCommand(provider_manager, config.bot_name)
    dispatcher.register(chart_cmd)
    help_commands.append(chart_cmd)

    ta_cmd = TechnicalAnalysisCommand(provider_manager)
    tldr_cmd = TLDRCommand(provider_manager)
    rsi_cmd = RSICommand(provider_manager)
    sma_cmd = SMACommand(provider_manager)
    macd_cmd = MACDCommand(provider_manager)
    support_cmd = SupportResistanceCommand(provider_manager)
    for cmd in [ta_cmd, tldr_cmd, rsi_cmd, sma_cmd, macd_cmd, support_cmd]:
        dispatcher.register(cmd)
    help_commands.extend([ta_cmd, tldr_cmd, rsi_cmd, sma_cmd, macd_cmd, support_cmd])

    earnings_cmd = EarningsCommand(provider_manager)
    dividend_cmd = DividendCommand(provider_manager)
    dispatcher.register(earnings_cmd)
    dispatcher.register(dividend_cmd)
    help_commands.extend([earnings_cmd, dividend_cmd])

    news_cmd = NewsCommand(provider_manager)
    dispatcher.register(news_cmd)
    help_commands.append(news_cmd)

    if watchlist_db:
        watch_cmd = WatchCommand(provider_manager, watchlist_db)
        dispatcher.register(watch_cmd)
        help_commands.append(watch_cmd)

    rating_cmd = RatingCommand(provider_manager)
    insider_cmd = InsiderCommand(provider_manager)
    short_cmd = ShortCommand(provider_manager)
    corr_cmd = CorrelationCommand(provider_manager)
    dispatcher.register(rating_cmd)
    dispatcher.register(insider_cmd)
    dispatcher.register(short_cmd)
    dispatcher.register(corr_cmd)
    help_commands.extend([rating_cmd, insider_cmd, short_cmd, corr_cmd])

    if alerts_db:
        alert_cmd = AlertCommand(provider_manager, alerts_db)
        dispatcher.register(alert_cmd)
        help_commands.append(alert_cmd)

    if ask_command is not None:
        dispatcher.register(ask_command)
        help_commands.append(ask_command)

    admin_numbers = config.admin_numbers
    metrics_cmd = MetricsCommand(admin_numbers=admin_numbers)
    cache_cmd = CacheCommand(admin_numbers=admin_numbers)
    admin_cmd = AdminCommand(
        admin_numbers=admin_numbers,
        watchlist_db=watchlist_db,
        alerts_db=alerts_db,
    )
    dispatcher.register(metrics_cmd)
    dispatcher.register(cache_cmd)
    dispatcher.register(admin_cmd)

    help_cmd = HelpCommand(help_commands, config.bot_name)
    dispatcher.register(help_cmd)

    return dispatcher


async def _alert_worker(alerts_db, provider_manager, signal_handler):
    """Poll active alerts and dispatch notifications when conditions trigger."""
    logger = logging.getLogger(__name__)
    logger.info("Alert worker started")

    while True:
        try:
            alerts = await alerts_db.get_all_active_alerts()
            if not alerts:
                await asyncio.sleep(60)
                continue

            logger.debug(f"Checking {len(alerts)} active alerts")

            by_symbol: dict[str, list[dict]] = {}
            for alert in alerts:
                by_symbol.setdefault(alert["symbol"], []).append(alert)

            for symbol, symbol_alerts in by_symbol.items():
                try:
                    quote = await provider_manager.get_quote(symbol)
                except Exception as e:
                    logger.error(f"Alert worker: failed to quote {symbol}: {e}")
                    continue

                current_price = quote.price
                for alert in symbol_alerts:
                    condition = alert["condition"]
                    target = alert["target_value"]
                    triggered = False
                    msg = None

                    if condition == "above" and current_price > target:
                        triggered = True
                        msg = f">>> ALERT: {symbol} is above ${target:.2f} (Current: ${current_price:.2f})"
                    elif condition == "below" and current_price < target:
                        triggered = True
                        msg = f">>> ALERT: {symbol} is below ${target:.2f} (Current: ${current_price:.2f})"
                    elif condition == "change_pct" and abs(quote.change_percent) >= target:
                        triggered = True
                        msg = f">>> ALERT: {symbol} moved {quote.change_percent:+.2f}% (Target: {target}%)"

                    if not triggered:
                        continue

                    logger.info(f"Alert triggered: {alert['id']} for {symbol}")
                    await alerts_db.trigger_alert(alert["id"])

                    try:
                        await signal_handler.send_message(
                            recipient=alert["user_phone"],
                            message=msg,
                            group_id=alert.get("group_id"),
                        )
                    except Exception as e:
                        logger.error(f"Failed to send alert notification: {e}")

            await asyncio.sleep(60)

        except asyncio.CancelledError:
            logger.info("Alert worker cancelled")
            raise
        except Exception as e:
            logger.error(f"Alert worker error: {e}")
            await asyncio.sleep(60)


def _start_background_loop() -> asyncio.AbstractEventLoop:
    """Start a persistent asyncio loop in a daemon thread and return it."""
    loop = asyncio.new_event_loop()
    ready = threading.Event()

    def runner():
        asyncio.set_event_loop(loop)
        ready.set()
        loop.run_forever()

    t = threading.Thread(target=runner, name="bot-async-loop", daemon=True)
    t.start()
    ready.wait()
    return loop


def build_app(config: Config):
    """Wire up providers, databases, dispatcher, signal handler, workers, and Flask app."""
    logger = logging.getLogger(__name__)

    errors = config.validate()
    if errors:
        for error in errors:
            logger.error(f"Configuration error: {error}")
        raise RuntimeError(f"Configuration errors: {errors}")

    provider_manager = create_provider_manager(config)
    logger.info(f"Configured {len(provider_manager.providers)} provider(s)")

    from .database import WatchlistDB, AlertsDB
    from .context import ContextManager
    from .contexts import ContextRegistry
    from .settings_store import SettingsStore
    from .llm import LLMClient, ConversationHistory
    from .group_log import GroupMessageLog
    from .mcp_integration import MCPRegistry, MCPManager
    from .enrichment import TwitterExpander

    watchlist_db = WatchlistDB(config.watchlist_db_path)
    alerts_db = AlertsDB(config.watchlist_db_path)
    context_manager = ContextManager(config.watchlist_db_path)
    settings_store = SettingsStore(config.watchlist_db_path)
    context_registry = ContextRegistry(config.watchlist_db_path)
    logger.info(f"Database: {config.watchlist_db_path}")

    mcp_registry = MCPRegistry(config.watchlist_db_path)
    mcp_manager = MCPManager(mcp_registry)

    llm_client = LLMClient(settings_store)
    llm_history = ConversationHistory(
        config.watchlist_db_path,
        turns_per_user=int(settings_store.get("llm_history_turns") or 6),
        settings_store=settings_store,
    )
    twitter_expander = TwitterExpander()
    group_log = GroupMessageLog(
        config.watchlist_db_path,
        settings_store=settings_store,
        enricher=twitter_expander,
    )

    # ask_command is built first (without bot_tools) so the dispatcher knows
    # about it, then bot_tools is built once the dispatcher is available, and
    # attached back — bot tools need to introspect the dispatcher's commands.
    ask_command = AskCommand(
        llm_client,
        llm_history,
        group_log=group_log,
        mcp_manager=mcp_manager,
        enricher=twitter_expander,
    )

    dispatcher = create_dispatcher(
        provider_manager, config, watchlist_db, alerts_db, context_manager,
        settings_store=settings_store,
        ask_command=ask_command,
        group_log=group_log,
        context_registry=context_registry,
        llm_client=llm_client,
    )

    from .commands.tools import BotCommandTools
    ask_command.bot_tools = BotCommandTools(dispatcher)

    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")

    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    tail = config.signal_phone_number[-4:] if config.signal_phone_number else "????"
    logger.info(f"Signal handler configured for ...{tail}")

    # Single shared event loop for all async work
    loop = _start_background_loop()

    # Start any MCP servers marked enabled=1 in the registry.
    try:
        asyncio.run_coroutine_threadsafe(
            mcp_manager.start_all_enabled(), loop
        ).result(timeout=60)
    except Exception as e:
        logger.error(f"MCP auto-start error: {e}")

    # Background alert worker
    asyncio.run_coroutine_threadsafe(
        _alert_worker(alerts_db, provider_manager, signal_handler),
        loop,
    )
    logger.info("Background alert worker scheduled")

    # WebSocket message poller
    poller = SignalPoller(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
        on_message=signal_handler.handle_webhook,
        poll_interval=1.0,
        loop=loop,
    )
    poller.start()

    admin_phone = config.admin_numbers[0] if config.admin_numbers else ""

    app = create_app(
        signal_handler,
        loop=loop,
        webhook_secret=config.webhook_secret,
        flask_secret_key=config.flask_secret_key,
        session_cookie_secure=config.session_cookie_secure,
        admin_password_hash=config.admin_password_hash,
        settings_store=settings_store,
        admin_phone=admin_phone,
        provider_manager=provider_manager,
        mcp_registry=mcp_registry,
        mcp_manager=mcp_manager,
        context_registry=context_registry,
        dispatcher=dispatcher,
    )
    # Keep references so these aren't garbage-collected
    app.signal_poller = poller
    app.async_loop = loop
    return app


def main():
    """Local/dev entry point."""
    config = Config.from_env()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Signal Stock Bot (dev)")

    app = build_app(config)

    logger.info(f"Starting webhook server on {config.host}:{config.port}")
    app.run(
        host=config.host,
        port=config.port,
        debug=config.log_level.upper() == "DEBUG",
        use_reloader=False,
    )


def create_gunicorn_app():
    """Factory for gunicorn."""
    config = Config.from_env()
    setup_logging(config.log_level)
    logger = logging.getLogger(__name__)
    logger.info("Starting Signal Stock Bot (gunicorn)")
    return build_app(config)


if __name__ == "__main__":
    main()
