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
    TarotCommand,
    IChingCommand,
    PredictCommand,
    PredictionsCommand,
    ResolveCommand,
    LeaderboardCommand,
)
from .signal import SignalHandler, SignalConfig, SignalPoller
from .server import create_app
from .users import NameRegistry


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
    reactor=None,
    name_registry=None,
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
        reactor=reactor,
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

    tarot_cmd = TarotCommand(
        db_path=config.watchlist_db_path,
        llm_client=llm_client,
    )
    dispatcher.register(tarot_cmd)
    help_commands.append(tarot_cmd)

    # Kick off a background download of the RWS deck if the persistent
    # data/tarot/ volume is missing cards. First start on a fresh volume
    # takes ~90s; subsequent starts are an instant no-op. The !tarot
    # command returns a friendly "still preparing" message until done.
    from .commands.tarot_command import ensure_deck_ready_async
    ensure_deck_ready_async()

    iching_cmd = IChingCommand(
        db_path=config.watchlist_db_path,
        llm_client=llm_client,
    )
    dispatcher.register(iching_cmd)
    help_commands.append(iching_cmd)

    # Numerology: pure-Python birthdate/name calculation, no I/O. Lives in
    # the woo command cluster alongside tarot + iching.
    from .commands.numerology_command import NumerologyCommand
    numerology_cmd = NumerologyCommand()
    dispatcher.register(numerology_cmd)
    help_commands.append(numerology_cmd)

    # Predictions: log dated claims, follow up at the deadline, score them.
    # The store is also exposed on the dispatcher so the background
    # resolver worker (scheduled in build_app) can reuse the same instance.
    from .predictions import PredictionStore
    prediction_store = PredictionStore(config.watchlist_db_path)
    dispatcher.prediction_store = prediction_store
    predict_cmd = PredictCommand(
        prediction_store, llm_client=llm_client, name_registry=name_registry,
    )
    predictions_cmd = PredictionsCommand(prediction_store, name_registry=name_registry)
    resolve_cmd = ResolveCommand(
        prediction_store,
        name_registry=name_registry,
        admin_numbers=config.admin_numbers,
    )
    leaderboard_cmd = LeaderboardCommand(prediction_store, name_registry=name_registry)
    for cmd in (predict_cmd, predictions_cmd, resolve_cmd, leaderboard_cmd):
        dispatcher.register(cmd)
        help_commands.append(cmd)

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

    help_cmd = HelpCommand(help_commands, config.bot_name, llm_client=llm_client)
    dispatcher.register(help_cmd)

    return dispatcher


async def _metrics_flush_worker(interval_seconds: float = 10.0) -> None:
    """Drain the in-memory metric event queue to SQLite every N seconds.

    Buffered writes mean the dashboard sees data ~`interval_seconds`
    behind real time, which is fine for a graph that updates on page
    reload. The queue is bounded so even a stuck flusher can't OOM.
    """
    log = logging.getLogger(__name__)
    from .metrics_log import get_metrics_log
    metrics_log = get_metrics_log()
    log.info(f"Metrics flush worker started (interval={interval_seconds}s)")
    while True:
        try:
            written = await metrics_log.flush()
            if written:
                log.debug(f"Flushed {written} metric events")
        except asyncio.CancelledError:
            log.info("Metrics flush worker cancelled")
            try:
                await metrics_log.flush()  # final drain
            except Exception:
                pass
            raise
        except Exception as e:
            log.error(f"Metrics flush error: {e}")
        await asyncio.sleep(interval_seconds)


async def _metrics_prune_worker(
    interval_seconds: float = 24 * 3600,
    retention_seconds: float = 30 * 24 * 3600,
) -> None:
    """Drop metric events older than `retention_seconds` once a day.

    30d matches the largest dashboard window so older events would never
    be visible anyway. Capping retention keeps the table small and the
    windowed queries fast.
    """
    log = logging.getLogger(__name__)
    from .metrics_log import get_metrics_log
    metrics_log = get_metrics_log()
    log.info(
        f"Metrics prune worker started "
        f"(retention={int(retention_seconds / 86400)}d, "
        f"interval={int(interval_seconds / 3600)}h)"
    )
    while True:
        try:
            deleted = await metrics_log.prune(retention_seconds)
            if deleted:
                log.info(f"Pruned {deleted} stale metric events")
        except asyncio.CancelledError:
            log.info("Metrics prune worker cancelled")
            raise
        except Exception as e:
            log.error(f"Metrics prune error: {e}")
        await asyncio.sleep(interval_seconds)


async def _attachment_cleanup_worker(
    attachments_dir: str,
    retention_days: int = 14,
    sweep_interval_hours: int = 24,
):
    """Periodically delete attachments older than `retention_days`.

    signal-cli writes every inbound and outbound attachment to its own data
    directory and never garbage-collects them. With heavy use (charts,
    tarot spreads, link previews) the dir grows without bound. We prune
    files whose mtime is older than the retention window — old enough that
    no in-flight message could plausibly still be referencing them.

    Implementation notes:
      * Sweep runs once at startup, then every `sweep_interval_hours`.
      * Errors per-file (permission, race) are logged but don't stop the sweep.
      * Errors per-iteration are logged and the worker keeps running on the
        next interval — never crashes the loop.
    """
    log = logging.getLogger(__name__)
    log.info(
        f"Attachment cleanup worker started "
        f"(dir={attachments_dir}, retention={retention_days}d, "
        f"interval={sweep_interval_hours}h)"
    )
    interval_seconds = sweep_interval_hours * 3600

    while True:
        try:
            await asyncio.to_thread(
                _sweep_attachments_once, attachments_dir, retention_days, log
            )
        except asyncio.CancelledError:
            log.info("Attachment cleanup worker cancelled")
            raise
        except Exception as e:
            log.error(f"Attachment cleanup error: {e}")
        await asyncio.sleep(interval_seconds)


def _sweep_attachments_once(
    attachments_dir: str,
    retention_days: int,
    log: logging.Logger,
) -> None:
    """Single-shot sweep — runs in a thread because os.walk is blocking."""
    import time

    path = Path(attachments_dir)
    if not path.is_dir():
        log.debug(f"Attachment cleanup: {path} not present, skipping")
        return

    cutoff = time.time() - retention_days * 86400
    removed = 0
    bytes_freed = 0
    errors = 0

    for entry in path.iterdir():
        if not entry.is_file():
            continue
        try:
            stat = entry.stat()
        except OSError:
            errors += 1
            continue
        if stat.st_mtime >= cutoff:
            continue
        try:
            entry.unlink()
            removed += 1
            bytes_freed += stat.st_size
        except OSError as e:
            errors += 1
            log.debug(f"Could not remove {entry.name}: {e}")

    if removed:
        log.info(
            f"Attachment cleanup: removed {removed} files "
            f"({bytes_freed // 1024} KB), errors={errors}"
        )
    else:
        log.debug(
            f"Attachment cleanup: nothing to remove (errors={errors})"
        )


async def _warm_pyodide(mcp_manager) -> None:
    """Warm the Pyodide sandbox so the first writer-LLM call doesn't pay
    the bundle-load + wheel-download cost.

    Two-phase warmup, both fire-and-forget:
      1. Trigger Pyodide's WASM bundle load and import the always-needed
         scientific libs (numpy/pandas/scipy come pre-built in the
         Pyodide bundle so this is just a deserialize + import).
      2. Pre-fetch yfinance + statsmodels wheels into the persistent
         cache directory so subsequent cold-starts skip the download.

    Failures degrade gracefully — the first user call pays whatever
    cost was skipped. We don't block bot startup on Pyodide being ready.
    """
    # Brief grace so start_all_enabled finishes before we look at tools.
    await asyncio.sleep(2.0)

    # If pyodide isn't running (admin disabled it, or start failed),
    # skip warmup silently.
    has_pyodide = any(
        t.server_name == "pyodide" for t in mcp_manager.all_tools()
    )
    if not has_pyodide:
        logging.info("Pyodide warmup: server not running, skipping")
        return

    try:
        await mcp_manager.call_tool(
            "pyodide__pyodide_execute",
            {
                "code": (
                    "import numpy, pandas, scipy, json, datetime\n"
                    "'pyodide warmup ready'"
                ),
                "timeout": 25000,
            },
        )
        logging.info("Pyodide warmup: bundle loaded + scientific libs imported")
    except Exception as e:
        logging.warning(
            f"Pyodide warmup phase 1 (bundle load) failed: {e} — "
            f"first user call will pay the cold-start"
        )
        return

    try:
        await mcp_manager.call_tool(
            "pyodide__pyodide_install-packages",
            {"package": "yfinance statsmodels"},
        )
        logging.info("Pyodide warmup: yfinance + statsmodels wheels cached")
    except Exception as e:
        logging.warning(
            f"Pyodide warmup phase 2 (package install) failed: {e} — "
            f"writer LLM can install on demand via pyodide_install-packages"
        )


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
    from .memory import MemoryStore
    from .settings_store import SettingsStore
    from .llm import LLMClient, ConversationHistory, EmojiReactor, DeepThinkClient
    from .llm.summarizer import Summarizer
    from .group_log import GroupMessageLog
    from .mcp_integration import MCPRegistry, MCPManager
    from .enrichment import CompositeEnricher, RichLinkExpander, TwitterExpander

    watchlist_db = WatchlistDB(config.watchlist_db_path)
    alerts_db = AlertsDB(config.watchlist_db_path)
    context_manager = ContextManager(config.watchlist_db_path)
    settings_store = SettingsStore(config.watchlist_db_path)
    context_registry = ContextRegistry(config.watchlist_db_path)
    memory_store = MemoryStore(config.watchlist_db_path)
    logger.info(f"Database: {config.watchlist_db_path}")

    mcp_registry = MCPRegistry(config.watchlist_db_path)
    mcp_manager = MCPManager(mcp_registry)

    llm_client = LLMClient(settings_store)
    name_registry = NameRegistry(config.watchlist_db_path, bot_name=config.bot_name)
    llm_history = ConversationHistory(
        config.watchlist_db_path,
        turns_per_user=int(settings_store.get("llm_history_turns") or 6),
        settings_store=settings_store,
        name_registry=name_registry,
    )
    # Two-stage link enrichment: TwitterExpander handles x.com / twitter.com
    # via the fxtwitter API (high-quality, no HTML parsing); RichLinkExpander
    # handles everything else by fetching og:title / og:description meta tags.
    # Composite runs them in order, both append snippets after the original
    # text. Shared by ask, reactor, and group log.
    enricher = CompositeEnricher(TwitterExpander(), RichLinkExpander())
    group_log = GroupMessageLog(
        config.watchlist_db_path,
        settings_store=settings_store,
        enricher=enricher,
    )

    # Reactor needs the dispatcher's signal_handler, but signal_handler
    # itself depends on the dispatcher. Build the reactor with a None
    # handler now and attach it once signal_handler exists below — the
    # reactor only fires after dispatch starts, so the handler is set by
    # the time any maybe_react() task actually runs.
    reactor = EmojiReactor(
        settings_store=settings_store,
        llm_client=llm_client,
        signal_handler=None,
        group_log=group_log,
        enricher=enricher,
        name_registry=name_registry,
        memory_store=memory_store,
    )

    summarizer = Summarizer(
        llm_client=llm_client,
        history=llm_history,
        settings_store=settings_store,
    )

    # Deep-think client — points at a separately configured (smarter,
    # slower, more expensive) model. Off by default; admins flip it on
    # at /admin/llm. The writer LLM gets it as a tool call, gated per-
    # context via ContextPolicy.deep_think_enabled. The deep model gets
    # the same tool kit the writer has (bot_tools wired below after the
    # dispatcher exists; mcp_manager already exists at this point).
    deep_think_client = DeepThinkClient(
        settings_store,
        mcp_manager=mcp_manager,
    )

    # ask_command is built first (without bot_tools) so the dispatcher knows
    # about it, then bot_tools is built once the dispatcher is available, and
    # attached back — bot tools need to introspect the dispatcher's commands.
    ask_command = AskCommand(
        llm_client,
        llm_history,
        group_log=group_log,
        mcp_manager=mcp_manager,
        enricher=enricher,
        name_registry=name_registry,
        summarizer=summarizer,
        reactor=reactor,
        deep_think=deep_think_client,
        memory_store=memory_store,
    )

    dispatcher = create_dispatcher(
        provider_manager, config, watchlist_db, alerts_db, context_manager,
        settings_store=settings_store,
        ask_command=ask_command,
        group_log=group_log,
        context_registry=context_registry,
        llm_client=llm_client,
        reactor=reactor,
        name_registry=name_registry,
    )

    from .commands.tools import BotCommandTools
    bot_tools = BotCommandTools(dispatcher)
    ask_command.bot_tools = bot_tools
    # PredictionStore is constructed inside create_dispatcher (so the
    # background resolver can share it via dispatcher.prediction_store);
    # late-bind it onto ask_command here so the writer LLM gets the
    # `predict_self` tool exposed.
    ask_command.prediction_store = dispatcher.prediction_store
    # Same tool adapter for the deep model — late-binding is the same
    # circular-dep workaround as ask_command's.
    deep_think_client.bot_tools = bot_tools

    logger.info(f"Registered {len(dispatcher.get_commands())} command(s)")

    signal_config = SignalConfig(
        api_url=config.signal_api_url,
        phone_number=config.signal_phone_number,
    )
    signal_handler = SignalHandler(signal_config, dispatcher)
    reactor.signal = signal_handler  # late binding (see comment near reactor construction)
    # Same late-binding trick: ask_command needs the handler to drive the
    # typing indicator while its tool loop runs.
    ask_command.signal_handler = signal_handler
    # Dispatcher needs the handler to send spontaneous replies fired by the
    # reactor's should_respond tool — those run outside the normal request/
    # response path so dispatch_implicit_ask must send its own messages.
    dispatcher.signal_handler = signal_handler
    # Reactor delegates its should_respond decisions to the dispatcher's
    # implicit-ask path. This is what makes natural responses actually fire.
    reactor.implicit_response_handler = dispatcher.dispatch_implicit_ask

    # Auto-vote on inbound polls. Bot picks option(s) via the LLM using
    # recent group context. No policy gate — per the user's request, every
    # poll the bot sees gets a vote (skipping self-authored ones).
    from .signal.poll_voter import PollVoter
    signal_handler.poll_voter = PollVoter(
        llm_client=llm_client,
        signal_handler=signal_handler,
        group_log=group_log,
        name_registry=name_registry,
        bot_phone=config.signal_phone_number,
    )
    tail = config.signal_phone_number[-4:] if config.signal_phone_number else "????"
    logger.info(f"Signal handler configured for ...{tail}")

    # Single shared event loop for all async work
    loop = _start_background_loop()

    # Auto-register MCP servers we ship as defaults (e.g. pyodide). No-op
    # if they're already in the registry — admin edits survive deploys.
    try:
        asyncio.run_coroutine_threadsafe(
            mcp_registry.ensure_defaults(), loop
        ).result(timeout=10)
    except Exception as e:
        logger.error(f"MCP defaults registration error: {e}")

    # Start any MCP servers marked enabled=1 in the registry.
    try:
        asyncio.run_coroutine_threadsafe(
            mcp_manager.start_all_enabled(), loop
        ).result(timeout=60)
    except Exception as e:
        logger.error(f"MCP auto-start error: {e}")

    # Pre-warm the Pyodide sandbox so the writer LLM's first call doesn't
    # eat the bundle-load + wheel-download cost (~10-15s otherwise).
    # Fire-and-forget — failure just means the first user call pays the
    # cold-start, no impact on bot availability.
    asyncio.run_coroutine_threadsafe(_warm_pyodide(mcp_manager), loop)

    # Background alert worker
    asyncio.run_coroutine_threadsafe(
        _alert_worker(alerts_db, provider_manager, signal_handler),
        loop,
    )
    logger.info("Background alert worker scheduled")

    # Persistent metrics: flush every 10s, prune events older than the
    # max selectable dashboard window (30d) once a day. Without flushing,
    # queue events would accumulate in memory between samples; without
    # pruning the metric_events table grows forever.
    asyncio.run_coroutine_threadsafe(_metrics_flush_worker(), loop)
    asyncio.run_coroutine_threadsafe(_metrics_prune_worker(), loop)
    logger.info("Metrics flush + prune workers scheduled")

    # Periodic cleanup of signal-cli's attachment cache. The cache lives on
    # the persistent volume and grows without bound (signal-cli never
    # collects). Default: prune anything older than 14 days, sweep daily.
    # Override via env (SIGNAL_ATTACHMENT_RETENTION_DAYS).
    attachments_dir = str(
        Path(config.watchlist_db_path).parent / "signal-cli" / "attachments"
    )
    retention_days = int(os.getenv("SIGNAL_ATTACHMENT_RETENTION_DAYS", "14"))
    asyncio.run_coroutine_threadsafe(
        _attachment_cleanup_worker(attachments_dir, retention_days=retention_days),
        loop,
    )
    logger.info(
        f"Attachment cleanup scheduled (dir={attachments_dir}, "
        f"retention={retention_days}d)"
    )

    # Prediction resolver: every 15 minutes, judge any due predictions and
    # post the verdict. Structured (ticker+threshold) predictions get a
    # live quote check; free-form claims go to the LLM. Failures keep the
    # row pending for the next sweep.
    from .predictions_resolver import PredictionResolver
    prediction_resolver = PredictionResolver(
        store=dispatcher.prediction_store,
        provider_manager=provider_manager,
        signal_handler=signal_handler,
        llm_client=llm_client,
        # Give the resolver Sigil's full research toolkit so free-form
        # claims get verified against live data (price/news/web) instead
        # of judged from training memory. The synthetic policy inside the
        # resolver narrows this to read-only commands.
        bot_tools=bot_tools,
        mcp_manager=mcp_manager,
        bot_phone=config.signal_phone_number,
    )
    asyncio.run_coroutine_threadsafe(prediction_resolver.run_forever(), loop)
    logger.info("Prediction resolver scheduled")

    # Daily oracle worker — per-context, multi-oracle. Each enabled
    # oracle fires once per day at its configured time (sunrise/sunset
    # +/- offset, or fixed clock time). On first boot of this code,
    # heuristic prepopulation seeds default oracle rows for existing
    # group contexts (disabled by default — admin opts in via the UI).
    # Old global settings are honored as a one-time migration.
    from .daily_oracle import DailyOracleWorker
    from .contexts.oracles import OracleStore, prepopulate_for_existing_contexts

    oracle_store = OracleStore(config.watchlist_db_path)
    try:
        asyncio.run_coroutine_threadsafe(
            prepopulate_for_existing_contexts(
                oracle_store, context_registry, settings_store=settings_store,
            ),
            loop,
        ).result(timeout=15)
    except Exception as e:
        logger.error(f"Oracle prepopulation failed: {e}")

    tarot_cmd_for_oracle = dispatcher.commands.get("tarot")
    iching_cmd_for_oracle = dispatcher.commands.get("iching")
    if tarot_cmd_for_oracle is not None and iching_cmd_for_oracle is not None:
        daily_oracle = DailyOracleWorker(
            oracle_store=oracle_store,
            context_registry=context_registry,
            tarot_command=tarot_cmd_for_oracle,
            iching_command=iching_cmd_for_oracle,
            ask_command=ask_command,
            signal_handler=signal_handler,
            bot_phone=config.signal_phone_number,
        )
        asyncio.run_coroutine_threadsafe(daily_oracle.run_forever(), loop)
        logger.info("Daily oracle worker scheduled (multi-oracle)")
    else:
        logger.warning(
            "Daily oracle: tarot/iching commands not registered; worker not started"
        )

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
        name_registry=name_registry,
        deep_think_client=deep_think_client,
        memory_store=memory_store,
        prediction_store=dispatcher.prediction_store,
        prediction_resolver=prediction_resolver,
        oracle_store=oracle_store,
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
