"""
Conditional-order watcher worker.

Long-running async task that scans `portfolio_orders` for pending
triggers every 5 minutes during US equity regular hours and fires
crossings via the existing executor. Out of hours it short-circuits
quickly without quote fetches so we don't spam the price provider
overnight.

Notification path: when orders fill (or fail), the worker fires a
synthetic `!ask` invocation per affected chat so the bot speaks the
fill in its own voice — and may decide to comment, place a follow-up
order, or just acknowledge. Multiple resolutions in the same chat
within one tick are batched into a single !ask so we don't spam the
chat with parallel reactions. Falls back to a plain status line when
ask_command isn't wired (e.g., in tests or restricted deploys).

Design notes:
  - Fixed 300s tick. The user explicitly chose this cadence over per-
    second polling — it bounds quote-provider load to ~12 ticks/hour
    and keeps Yahoo from rate-limiting us.
  - Idempotent fills: PortfolioStore.mark_order_filled UPDATEs WHERE
    status IN (pending, in_flight), so a second tick that re-evaluates
    a just-filled order silently no-ops.
  - No leader election. Single-process bot — if we ever shard, this
    worker needs a row-level claim.
"""

from __future__ import annotations

import asyncio
import logging

from .commands.base import CommandContext
from .paper_portfolio import PortfolioStore, SOURCE_ORDER
from .paper_portfolio_executor import (
    PaperPortfolioExecutor,
    market_closed_reason,
)

logger = logging.getLogger(__name__)


class OrdersWorker:
    """Polls pending conditional orders and fires triggered ones."""

    # Tick cadence in seconds. Picked to balance freshness vs quote
    # provider load: 5 min is fast enough for typical stop/limit use
    # cases (intraday swings, not HFT) and slow enough that 50+
    # portfolios with diverse tickers don't burn through Yahoo's quota.
    POLL_INTERVAL_SECONDS = 300

    # How often we tick when the market is closed. Much slower — there
    # are no quotes to fetch and no orders to fire, but we still want
    # to expire stale orders periodically so the table stays clean.
    CLOSED_POLL_SECONDS = 1800

    def __init__(
        self,
        *,
        store: PortfolioStore,
        executor: PaperPortfolioExecutor,
        signal_pool=None,
        signal_handler=None,
        ask_command=None,
        context_registry=None,
        bot_phone: str = "",
        bot_registry=None,
    ):
        self.store = store
        self.executor = executor
        self.signal_pool = signal_pool
        self.signal = signal_handler or (
            signal_pool.default() if signal_pool is not None else None
        )
        # ask_command is optional: if wired, the worker fires a
        # synthetic !ask per affected chat so the bot voices the
        # resolution itself. If None, we fall back to a static
        # notification line.
        self.ask = ask_command
        self.contexts = context_registry
        self.bot_phone = bot_phone or (
            signal_pool.default_phone if signal_pool is not None else ""
        )
        self.bot_registry = bot_registry

    def _resolve_bot_for(self, policy):
        """Same per-context-pin → registry-default chain as the
        dispatcher uses; orders are group-only so we skip DM."""
        if self.bot_registry is None:
            return None
        pinned = getattr(policy, "default_bot_id", None)
        if pinned is not None:
            bot = self.bot_registry.get_sync(pinned)
            if bot and bot.enabled:
                return bot
        return self.bot_registry.default_for_kind_sync("group")

    async def run_forever(self) -> None:
        logger.info(
            f"Orders watcher started (poll {self.POLL_INTERVAL_SECONDS}s "
            f"open / {self.CLOSED_POLL_SECONDS}s closed)"
        )
        # Reconcile any orders left in the `in_flight` claim state from
        # a prior crash. In normal operation this is empty — every
        # claim is followed by a finalize within ~1s. A non-empty list
        # means we crashed mid-fire; the trade row in `trades` is the
        # truth, and admin should resolve manually. We log loudly here
        # rather than silently retrying because a re-fire would
        # double-debit cash.
        try:
            stuck = await self.store.list_in_flight_orders()
            if stuck:
                logger.error(
                    f"Orders watcher: {len(stuck)} order(s) stuck in "
                    f"in_flight on startup — likely from a prior crash. "
                    f"Trade rows in `trades` are authoritative. Resolve "
                    f"manually: ids={[o.id for o in stuck]}"
                )
        except Exception as e:
            logger.warning(f"Orders watcher: in_flight check failed: {e}")
        while True:
            try:
                sleep_for = await self.tick()
            except asyncio.CancelledError:
                logger.info("Orders watcher cancelled")
                raise
            except Exception as e:
                logger.exception(f"Orders watcher tick failed: {e}")
                sleep_for = self.POLL_INTERVAL_SECONDS
            await asyncio.sleep(sleep_for)

    async def tick(self) -> int:
        """Run one watcher pass. Returns the next sleep interval — the
        loop alternates fast (open) and slow (closed) cadences without
        the run_forever loop needing to know the market state itself.
        """
        closed = market_closed_reason()
        if closed is not None:
            # Off-hours: still expire stale orders so they don't pile up
            # over a long weekend, then sleep deeply. No quote fetches
            # for orders. We DO settle expired options here — it's the
            # natural moment to run it (right after close on expiry day)
            # and the settlement path needs an underlying-spot quote
            # per distinct ticker, which is cheap.
            try:
                n_expired = await self.store.expire_stale_orders()
                if n_expired:
                    logger.info(
                        f"Orders watcher (market closed: {closed}): "
                        f"expired {n_expired} stale order(s)"
                    )
            except Exception as e:
                logger.warning(
                    f"Orders watcher: expire pass failed: {e}"
                )
            await self._settle_expired_options()
            return self.CLOSED_POLL_SECONDS

        # Capture the pre-tick pending set so we can correlate post-tick
        # statuses (filled / failed / still pending) and notify the
        # originating chats. list_pending_orders is cheap (indexed); we
        # call it twice per tick on purpose so the diff is authoritative.
        try:
            before = await self.store.list_pending_orders()
        except Exception as e:
            logger.warning(f"Orders watcher: pre-list failed: {e}")
            before = []
        before_ids = {o.id: o for o in before}

        try:
            stats = await self.executor.try_fill_pending()
        except Exception as e:
            logger.exception(f"Orders watcher: try_fill_pending failed: {e}")
            return self.POLL_INTERVAL_SECONDS

        if stats.get("scanned"):
            logger.info(
                f"Orders watcher: scanned={stats['scanned']} "
                f"filled={stats['filled']} failed={stats['failed']} "
                f"expired={stats['expired']} "
                f"tickers_quoted={stats['tickers_quoted']}"
            )

        # Notify on resolved orders. We re-fetch the post-state for any
        # id that was pending before; the alternative (threading the
        # diff through the executor) couples the worker too tightly to
        # the executor's internal contract.
        if stats.get("filled") or stats.get("failed"):
            await self._notify_resolutions(list(before_ids.values()))

        # Options settlement runs every tick — idempotent and cheap when
        # nothing is due (a single SELECT). Putting it on the open path
        # in addition to closed catches positions whose 16:00 ET expiry
        # falls inside the next 5-min tick, no need to wait for the
        # market-closed cadence.
        await self._settle_expired_options()

        return self.POLL_INTERVAL_SECONDS

    async def _settle_expired_options(self) -> None:
        """Settle every expired options position across all contexts.
        Logged at info on activity; warn on errors. Wraps the executor
        call so the watcher loop can stay simple."""
        try:
            stats = await self.executor.settle_expired_options()
        except Exception as e:
            logger.exception(
                f"Orders watcher: settle_expired_options failed: {e}"
            )
            return
        if stats.get("settled") or stats.get("errors"):
            logger.info(
                f"Orders watcher: option settlement "
                f"checked={stats.get('checked', 0)} "
                f"due={stats.get('due', 0)} "
                f"settled={stats.get('settled', 0)} "
                f"errors={stats.get('errors', 0)}"
            )

    async def _notify_resolutions(self, prior_pending: list) -> None:
        """Notify chats about orders that resolved this tick.

        Resolution states we surface:
          - filled  → bot reacts in its own voice
          - failed  → bot reacts in its own voice
        Cancelled, expired, and in-flight are silent — those came from
        explicit actions, natural decay, or are still in progress.

        Resolutions are batched by context_key so multiple fills in the
        same chat in one tick produce ONE bot reply instead of N
        parallel ones (a market-crash that fires 5 stops shouldn't
        spam the chat with 5 separate posts).

        Two delivery modes:
          - bot voice: when ask_command is wired, fires a synthetic
            !ask per affected chat with the fills as context. The bot
            speaks naturally, may comment on the thesis, may place a
            follow-up trade.
          - static: when ask_command is None, sends a plain
            "Order #N filled" line per resolution.
        """
        if self.signal is None or not prior_pending:
            return

        prior_ids = [o.id for o in prior_pending]
        try:
            current_orders = await self.store.list_orders_by_ids(prior_ids)
        except Exception as e:
            logger.warning(
                f"Orders watcher: bulk fetch for notify failed: {e}"
            )
            return

        # Group resolved orders by context so each chat gets ONE reply
        # even when multiple orders fired together.
        by_context: dict[str, list] = {}
        for current in current_orders:
            if current.status not in ("filled", "failed"):
                continue
            by_context.setdefault(current.context_key, []).append(current)

        for ctx_key, orders in by_context.items():
            if self.ask is not None:
                await self._fire_bot_reaction(ctx_key, orders)
            else:
                # Fallback: static line per order.
                for o in orders:
                    msg = self._format_resolution(o)
                    await self._send_to_context(ctx_key, msg)

    async def _fire_bot_reaction(
        self, context_key: str, orders: list,
    ) -> None:
        """Synthesize an !ask invocation so the bot voices the fills
        itself. Mirrors the trading cron's pattern: build a
        CommandContext, run ask_command.execute, post the result.

        Skips DM contexts (we don't have a path from hashed phone back
        to a real number). Group contexts get the message routed to
        their group_id."""
        if not context_key.startswith("group:"):
            # DM-context portfolios can't be reached without a recipient
            # number; fall back to the static path so logs at least
            # capture the resolution.
            for o in orders:
                logger.info(
                    f"Orders watcher: DM context {context_key}, "
                    f"static log fallback: "
                    f"{self._format_resolution(o)!r}"
                )
            return

        group_id = context_key[len("group:"):]

        # Resolve the policy so the !ask invocation respects the same
        # gates the chat normally has (portfolio allowed, deep_think
        # enabled or not, etc). Without a policy we'd be calling !ask
        # in a chat that may have opted out of the feature entirely.
        policy = None
        if self.contexts is not None:
            try:
                policy = await self.contexts.get_by_key(group_id)
            except Exception as e:
                logger.warning(
                    f"Orders watcher: policy lookup for {context_key} "
                    f"failed: {e}"
                )
                return
        if policy is None or not policy.allows_command("portfolio"):
            logger.info(
                f"Orders watcher: {context_key} not opted into portfolio "
                f"(policy={policy!r}); falling back to static notify"
            )
            for o in orders:
                msg = self._format_resolution(o)
                await self._send_to_context(context_key, msg)
            return

        prompt = self._build_reaction_prompt(orders)
        order_bot = self._resolve_bot_for(policy)
        order_phone = (
            (order_bot.signal_phone if order_bot else None)
            or self.bot_phone
            or ""
        )
        ctx = CommandContext(
            sender=order_phone,
            group_id=group_id,
            raw_message=f"!ask {prompt}",
            command="ask",
            args=[prompt],
            policy=policy,
            bot=order_bot,
            # Tag any follow-up trades from this !ask as order-driven
            # so the trade ledger is honest about provenance.
            automation_source=SOURCE_ORDER,
        )
        try:
            result = await self.ask.execute(ctx)
        except Exception as e:
            logger.exception(
                f"Orders watcher: !ask reaction failed for {context_key}: {e}"
            )
            # Don't leave the chat silent — degrade to static notify.
            for o in orders:
                msg = self._format_resolution(o)
                await self._send_to_context(context_key, msg)
            return
        if not result or not result.success:
            logger.warning(
                f"Orders watcher: !ask returned unsuccessful for "
                f"{context_key}: {getattr(result, 'text', '(none)')!r}"
            )
            for o in orders:
                msg = self._format_resolution(o)
                await self._send_to_context(context_key, msg)
            return

        # Prepend a small header so chat readers can tell the bot's
        # reply was triggered by an order fire (the bot's own text may
        # not lead with that, especially after a few rounds).
        n = len(orders)
        plural = "s" if n != 1 else ""
        header = f"🔔 Order trigger{plural} fired"
        body = result.text or ""
        if not body.startswith("🔔"):
            body = f"{header}\n\n{body}"

        sender_handler = (
            self.signal_pool.for_bot(order_bot)
            if self.signal_pool is not None
            else self.signal
        )
        if sender_handler is None:
            logger.warning(
                f"Orders watcher: no signal handler for {context_key}; "
                f"skipping post"
            )
            return
        try:
            await sender_handler.send_message(
                recipient="",
                message=body,
                group_id=group_id,
                attachments=result.attachments,
                styled=getattr(result, "styled", False),
            )
        except Exception as e:
            logger.error(
                f"Orders watcher: send_message failed for {context_key}: {e}"
            )

    @staticmethod
    def _build_reaction_prompt(orders: list) -> str:
        """Compose the synthetic-!ask body that briefs the bot on
        which orders just resolved and asks for a short reaction."""
        lines = []
        for o in orders:
            if o.status == "filled":
                qty_part = ""
                if o.fill_qty and o.fill_qty > 0:
                    qty_part = f" {o.fill_qty:.4f}"
                price_part = (
                    f" @ ${o.fill_price:.2f}" if o.fill_price else ""
                )
                side_word = "Bought" if o.side == "buy" else "Sold"
                lines.append(
                    f"- Order #{o.id} FILLED: {side_word}{qty_part} "
                    f"{o.ticker}{price_part} (was a {o.kind}-{o.side} "
                    f"@ ${o.trigger_price:.2f}). "
                    f"Original thesis: {o.reason or '(none)'}"
                )
            else:  # failed
                lines.append(
                    f"- Order #{o.id} FAILED to fill ({o.kind}-{o.side} "
                    f"{o.ticker} @ ${o.trigger_price:.2f}). "
                    f"Executor said: {o.fill_note or '(no detail)'}. "
                    f"Original thesis: {o.reason or '(none)'}"
                )
        body = "\n".join(lines)
        return (
            f"One or more of your conditional orders just resolved while "
            f"you weren't looking. Here's what happened:\n\n"
            f"{body}\n\n"
            f"This is your moment to speak. Comment briefly in your own "
            f"voice — 1-3 sentences, conversational, the way you'd "
            f"naturally announce a fill to the chat. Lead with what "
            f"happened, optionally note whether the thesis worked.\n\n"
            f"Consider follow-ups:\n"
            f"  - Filled buy → place a protective stop-loss "
            f"    (`portfolio_place_order` sell+stop) below your "
            f"    invalidation, and/or a take-profit limit above your "
            f"    target.\n"
            f"  - Filled sell that closed a position → optional "
            f"    re-entry order if the thesis is intact at a "
            f"    different price.\n"
            f"  - Failed fill → decide whether to replace with a "
            f"    different size/trigger or let it go.\n\n"
            f"If a thesis genuinely worked or genuinely broke, "
            f"`portfolio_journal_append` a short reflection — that's "
            f"how next-week's-you learns from this fill. Skip the "
            f"journal entry for routine fills with no real lesson. "
            f"Don't pile on follow-up trades just because something "
            f"fired; only act when there's a real reason."
        )

    @staticmethod
    def _format_resolution(order) -> str:
        """One-line notification body. Mirrors the visual style of
        cron-trade headers (◈ / ▲ / ▼) so the chat reads it as a
        portfolio event, not a system error."""
        side_word = "Bought" if order.side == "buy" else "Sold"
        if order.status == "filled":
            qty_part = ""
            if order.fill_qty and order.fill_qty > 0:
                qty_part = f" {order.fill_qty:.4f}"
            price_part = (
                f" @ ${order.fill_price:.2f}"
                if order.fill_price else ""
            )
            return (
                f"◈ Order #{order.id} triggered — {side_word}"
                f"{qty_part} {order.ticker}{price_part}\n"
                f"  ({order.kind} @ ${order.trigger_price:.2f}: "
                f"{order.reason or 'no reason recorded'})"
            )
        # failed
        return (
            f"◇ Order #{order.id} triggered but didn't fill: "
            f"{order.fill_note or 'unknown reason'}\n"
            f"  ({order.kind}-{order.side} {order.ticker} @ "
            f"${order.trigger_price:.2f})"
        )

    async def _send_to_context(self, context_key: str, message: str) -> None:
        """Route a message to the originating chat. Group context keys
        carry a base64 group_id; DM keys carry a hashed phone we cannot
        send to directly (Signal needs the real number). For DMs we
        simply log — there's no current path from hashed phone back to
        a recipient, and DM-context portfolios are rare in practice."""
        if context_key.startswith("group:"):
            group_id = context_key[len("group:"):]
            sender_handler = self.signal
            if self.signal_pool is not None:
                # No per-context bot lookup here (this is the static
                # fallback path); route through the default bot's phone.
                fallback_bot = (
                    self.bot_registry.default_for_kind_sync("group")
                    if self.bot_registry is not None else None
                )
                sender_handler = self.signal_pool.for_bot(fallback_bot)
            if sender_handler is None:
                logger.warning(
                    f"Orders watcher: no signal handler for {context_key}; "
                    f"skipping notify"
                )
                return
            try:
                await sender_handler.send_message(
                    recipient="",
                    message=message,
                    group_id=group_id,
                )
            except Exception as e:
                logger.warning(
                    f"Orders watcher: notify failed for {context_key}: {e}"
                )
            return

        # DM contexts use a hashed phone — we can't recover the real
        # number to send to. Log so operators know the fill happened
        # even if the chat doesn't get a ping.
        logger.info(
            f"Orders watcher: skipping DM notify for {context_key} "
            f"(no reverse-lookup); message would have been: {message!r}"
        )


# Helper used by the main wiring code so callers don't need to import
# both classes plus the deps. Keeps wiring concise in main.py.
def build_orders_worker(
    *,
    store: PortfolioStore,
    executor: PaperPortfolioExecutor,
    signal_handler=None,
) -> OrdersWorker:
    return OrdersWorker(
        store=store, executor=executor, signal_handler=signal_handler,
    )


__all__ = ["OrdersWorker", "build_orders_worker"]
