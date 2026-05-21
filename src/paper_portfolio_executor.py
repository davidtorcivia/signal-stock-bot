"""
Paper-portfolio execution layer.

Wraps `PortfolioStore` with a quote-driven fill model:
  - Fetches the live quote from the existing `ProviderManager`.
  - Rejects trades outside US equity regular hours (9:30-16:00 ET, M-F).
  - Computes fractional share quantity from a dollar amount when given.
  - Marks open positions to market for status / PnL output.

Kept separate from `paper_portfolio.py` so the store stays
provider-agnostic and easy to unit-test without mocking quotes.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import math
import time
from typing import Optional, Union
from zoneinfo import ZoneInfo

from .options_symbols import (
    friendly_name as occ_friendly_name,
    intrinsic_value as occ_intrinsic_value,
    normalize_contract,
    parse_occ,
)
from .paper_portfolio import (
    KIND_LIMIT,
    KIND_STOP,
    Order,
    ORDER_PENDING,
    OptionPosition,
    PortfolioStore,
    SIDE_BUY,
    SIDE_SELL,
    SOURCE_ORDER,
    SOURCE_REACTIVE,
    SOURCE_SETTLEMENT,
    VALID_ORDER_KINDS,
    VALID_SOURCES,
)


def _finite(value: Optional[float]) -> bool:
    """True if value is a real finite number. Used at the executor
    boundary to reject NaN/Inf before they hit SQL — defence in depth
    over PortfolioStore's own check."""
    return value is not None and math.isfinite(value)

logger = logging.getLogger(__name__)


_ET = ZoneInfo("America/New_York")

# US equity regular hours. Pre/post are not allowed in V1 — the user
# asked for "no after hours" explicitly.
_MARKET_OPEN = dt.time(9, 30)
_MARKET_CLOSE = dt.time(16, 0)


def is_market_open(now: Optional[dt.datetime] = None) -> bool:
    """True iff `now` (default: real wall clock) lies within US equity
    regular hours on a weekday. Federal holidays aren't tracked — the
    quote provider effectively handles those by returning stale data,
    but we don't try to be more strict than weekday + clock-hours.
    """
    if now is None:
        now = dt.datetime.now(_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)
    if now.weekday() >= 5:  # 5 = Saturday, 6 = Sunday
        return False
    t = now.time()
    return _MARKET_OPEN <= t < _MARKET_CLOSE


# Bounds applied to LLM-supplied order inputs at the executor boundary.
# Defense-in-depth against prompt-injection or model-error inputs that
# would otherwise fill the DB with garbage or DoS the watcher.
_MAX_PENDING_ORDERS_PER_CONTEXT = 25
_MAX_TICKER_LEN = 16          # generous: real tickers are 1-5, ETFs use suffixes
_MAX_REASON_LEN = 500
_MAX_TRIGGER_PRICE = 1_000_000.0  # no equity priced > $1M/share is realistic


def _order_to_view(order: Order) -> dict:
    """Render an Order for status() consumers (LLM tool result, image
    renderer, admin UI). Same shape used everywhere downstream so the
    image and text paths can rely on identical keys."""
    return {
        "id": order.id,
        "ticker": order.ticker,
        "side": order.side,
        "kind": order.kind,
        "trigger_price": order.trigger_price,
        "trigger_direction": order.trigger_direction(),
        "qty": order.qty,
        "dollars": order.dollars,
        "close_position": order.close_position,
        "reason": order.reason,
        "created_at": order.created_at,
        "expires_at": order.expires_at,
    }


def market_closed_reason(now: Optional[dt.datetime] = None) -> Optional[str]:
    """Return None when market is open, else a short reason string used
    in user-facing rejection messages."""
    if now is None:
        now = dt.datetime.now(_ET)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=_ET)
    else:
        now = now.astimezone(_ET)
    if now.weekday() >= 5:
        return "market closed (weekend)"
    t = now.time()
    if t < _MARKET_OPEN:
        return f"market not open yet (opens 9:30 ET, now {t.strftime('%H:%M')} ET)"
    if t >= _MARKET_CLOSE:
        return f"market closed (after 16:00 ET, now {t.strftime('%H:%M')} ET)"
    return None


class PaperPortfolioExecutor:
    """Quote-driven executor over a PortfolioStore."""

    def __init__(self, store: PortfolioStore, provider_manager):
        self.store = store
        self.providers = provider_manager

    async def _fetch_price(self, ticker: str) -> tuple[Optional[float], Optional[str]]:
        """Returns (price, error). Exactly one is None."""
        try:
            quote = await self.providers.get_quote(ticker)
        except Exception as e:
            return None, f"couldn't quote {ticker}: {type(e).__name__}"
        price = getattr(quote, "price", None)
        if price is None or price <= 0:
            return None, f"no valid quote for {ticker}"
        return float(price), None

    async def _fetch_option_quote(self, contract: str):
        """Returns (option_quote, error). Exactly one is None. `contract`
        must already be canonical OCC — callers normalize friendly input
        upstream so the error path here is fresh-and-clear."""
        try:
            quote = await self.providers.get_option_quote(contract)
        except Exception as e:
            return None, f"couldn't quote {contract}: {type(e).__name__}"
        if quote is None:
            return None, f"no option quote for {contract}"
        return quote, None

    async def _fetch_option_premium(self, contract: str) -> tuple[Optional[float], Optional[str]]:
        """Returns (per-share premium, error). Zero/negative premiums
        are treated as a quote miss so the executor refuses to open
        positions on stale or empty-book contracts."""
        quote, err = await self._fetch_option_quote(contract)
        if err is not None or quote is None:
            return None, err
        premium = getattr(quote, "price", None)
        if premium is None or premium <= 0:
            return None, f"no valid premium for {contract}"
        return float(premium), None

    async def execute_buy(
        self,
        context_key: str,
        *,
        ticker: str,
        dollars: Optional[float] = None,
        qty: Optional[float] = None,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
        force_market_open: bool = False,
    ) -> dict:
        """Buy via live quote. Caller passes either `dollars` or `qty`.

        force_market_open=True bypasses the market-hours check; only
        admin-initiated trades from the dashboard should pass it.
        Cron and reactive paths use the default (False) so trades
        outside RTH bounce.

        Returns dict from `PortfolioStore.buy` plus `price` and
        `ticker` for echoing in the response, or
        `{"ok": False, "error": str}` on quote failure / closed market.
        """
        if source not in VALID_SOURCES:
            return {"ok": False, "error": f"invalid source: {source!r}"}
        ticker = (ticker or "").strip().upper()
        if not ticker:
            return {"ok": False, "error": "Ticker required."}
        if dollars is None and qty is None:
            return {"ok": False, "error": "Pass either dollars or qty."}
        if dollars is not None and qty is not None:
            return {"ok": False, "error": "Pass dollars OR qty, not both."}

        if not force_market_open:
            closed = market_closed_reason()
            if closed:
                return {"ok": False, "error": f"Trade rejected — {closed}."}

        price, err = await self._fetch_price(ticker)
        if err is not None or price is None:
            return {"ok": False, "error": err or "no quote"}

        if dollars is not None:
            if not _finite(dollars):
                return {"ok": False, "error": "Dollar amount must be a finite number."}
            if dollars <= 0:
                return {"ok": False, "error": "Dollar amount must be positive."}
            qty = dollars / price
        assert qty is not None
        if not _finite(qty):
            return {"ok": False, "error": "Quantity must be a finite number."}
        if qty <= 0:
            return {"ok": False, "error": "Quantity must be positive."}

        result = await self.store.buy(
            context_key, ticker, qty, price,
            reason=reason, source=source,
        )
        if result.get("ok"):
            result["price"] = price
            result["ticker"] = ticker
        return result

    async def execute_sell(
        self,
        context_key: str,
        *,
        ticker: str,
        qty: Union[float, str, None] = None,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
        force_market_open: bool = False,
    ) -> dict:
        """Sell via live quote. `qty="all"` (or None) closes the
        position. Numeric `qty` sells that many fractional shares."""
        if source not in VALID_SOURCES:
            return {"ok": False, "error": f"invalid source: {source!r}"}
        ticker = (ticker or "").strip().upper()
        if not ticker:
            return {"ok": False, "error": "Ticker required."}

        if not force_market_open:
            closed = market_closed_reason()
            if closed:
                return {"ok": False, "error": f"Trade rejected — {closed}."}

        # Resolve "all" against the stored qty before we hit the price
        # provider, so a quote failure on a position the bot doesn't
        # actually hold returns the more useful "no position" error.
        sell_qty: Optional[float]
        if qty is None or (isinstance(qty, str) and qty.strip().lower() == "all"):
            pos = await self.store.get_position(context_key, ticker)
            if pos is None:
                return {"ok": False, "error": f"No position in {ticker}."}
            sell_qty = pos.qty
        else:
            try:
                sell_qty = float(qty)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"Invalid qty: {qty!r}"}
            if not _finite(sell_qty):
                return {"ok": False, "error": "Quantity must be a finite number."}
            if sell_qty <= 0:
                return {"ok": False, "error": "Quantity must be positive."}

        price, err = await self._fetch_price(ticker)
        if err is not None or price is None:
            return {"ok": False, "error": err or "no quote"}

        result = await self.store.sell(
            context_key, ticker, sell_qty, price,
            reason=reason, source=source,
        )
        if result.get("ok"):
            result["price"] = price
            result["ticker"] = ticker
        return result

    async def status(self, context_key: str) -> dict:
        """Snapshot for !portfolio / portfolio_status tool. Includes
        cash, positions with mark-to-market, total PnL.

        Quote failures on individual positions don't block the snapshot
        — those positions show `mark_price=None` and contribute their
        cost basis to total value (a conservative fallback that doesn't
        invent gains). Logged at warn level.
        """
        portfolio = await self.store.ensure_portfolio(context_key)
        positions = await self.store.positions(context_key)
        options_pos = await self.store.options_positions(context_key)
        tip_total = await self.store.tip_total(context_key)
        realized_total = await self.store.realized_pnl_total(context_key)
        pending_orders = await self.store.list_orders_for_context(
            context_key, statuses={ORDER_PENDING},
        )

        # Fetch all marks concurrently — each await is a network round-
        # trip to the quote provider, and serial fetching turns
        # !portfolio into a many-second wait once Sigil is holding more
        # than a couple names. asyncio.gather preserves order so the
        # zip below pairs correctly with `positions`.
        quote_results = await asyncio.gather(
            *(self._fetch_price(pos.ticker) for pos in positions),
            return_exceptions=False,
        ) if positions else []

        position_views: list[dict] = []
        market_value = 0.0
        unrealized_total = 0.0
        for pos, (price, err) in zip(positions, quote_results):
            if err is not None or price is None:
                logger.warning(
                    f"status: quote miss for {pos.ticker} ({err}); "
                    f"showing cost basis as fallback"
                )
                cost = pos.qty * pos.avg_cost
                position_views.append({
                    "ticker": pos.ticker,
                    "qty": pos.qty,
                    "avg_cost": pos.avg_cost,
                    "mark_price": None,
                    "cost_basis": cost,
                    "market_value": cost,
                    "unrealized_pnl": 0.0,
                    "unrealized_pct": 0.0,
                })
                market_value += cost
                continue
            mv = pos.qty * price
            cost = pos.qty * pos.avg_cost
            unrealized = mv - cost
            position_views.append({
                "ticker": pos.ticker,
                "qty": pos.qty,
                "avg_cost": pos.avg_cost,
                "mark_price": price,
                "cost_basis": cost,
                "market_value": mv,
                "unrealized_pnl": unrealized,
                "unrealized_pct": (unrealized / cost) if cost > 0 else 0.0,
            })
            market_value += mv
            unrealized_total += unrealized

        # Mark-to-market for options positions. Each contract's value is
        # qty × multiplier × premium. Quote misses fall back to cost
        # basis (same shape as equity positions above) so the snapshot
        # never invents value, but also doesn't punish the user for a
        # transient quote failure.
        option_views: list[dict] = []
        option_market_value = 0.0
        if options_pos:
            option_quote_results = await asyncio.gather(
                *(self._fetch_option_premium(op.contract_symbol) for op in options_pos),
                return_exceptions=False,
            )
            for op, (premium, err) in zip(options_pos, option_quote_results):
                cost = op.qty * op.multiplier * op.avg_premium
                friendly = occ_friendly_name(op.contract_symbol)
                if err is not None or premium is None:
                    logger.warning(
                        f"status: option quote miss for {op.contract_symbol} "
                        f"({err}); showing cost basis as fallback"
                    )
                    option_views.append({
                        "contract": op.contract_symbol,
                        "friendly": friendly,
                        "underlying": op.underlying,
                        "option_type": op.option_type,
                        "strike": op.strike,
                        "expiration": op.expiration,
                        "qty": op.qty,
                        "multiplier": op.multiplier,
                        "avg_premium": op.avg_premium,
                        "mark_premium": None,
                        "cost_basis": cost,
                        "market_value": cost,
                        "unrealized_pnl": 0.0,
                        "unrealized_pct": 0.0,
                    })
                    option_market_value += cost
                    continue
                mv = op.qty * op.multiplier * premium
                unrealized = mv - cost
                option_views.append({
                    "contract": op.contract_symbol,
                    "friendly": friendly,
                    "underlying": op.underlying,
                    "option_type": op.option_type,
                    "strike": op.strike,
                    "expiration": op.expiration,
                    "qty": op.qty,
                    "multiplier": op.multiplier,
                    "avg_premium": op.avg_premium,
                    "mark_premium": premium,
                    "cost_basis": cost,
                    "market_value": mv,
                    "unrealized_pnl": unrealized,
                    "unrealized_pct": (unrealized / cost) if cost > 0 else 0.0,
                })
                option_market_value += mv
                unrealized_total += unrealized

        total_funded = portfolio.starting_balance + tip_total
        total_market_value = market_value + option_market_value
        equity = portfolio.cash + total_market_value
        total_pnl = equity - total_funded
        total_pnl_pct = (total_pnl / total_funded) if total_funded > 0 else 0.0

        return {
            "context_key": context_key,
            "label": portfolio.label,
            "cash": portfolio.cash,
            "starting_balance": portfolio.starting_balance,
            "tip_total": tip_total,
            "total_funded": total_funded,
            "market_value": total_market_value,
            "equity_market_value": market_value,
            "options_market_value": option_market_value,
            "equity": equity,
            "realized_pnl": realized_total,
            "unrealized_pnl": unrealized_total,
            "total_pnl": total_pnl,
            "total_pnl_pct": total_pnl_pct,
            "positions": position_views,
            "options_positions": option_views,
            "pending_orders": [_order_to_view(o) for o in pending_orders],
            "market_open": market_closed_reason() is None,
        }

    async def place_order(
        self,
        context_key: str,
        *,
        ticker: str,
        side: str,
        kind: str,
        trigger_price: float,
        qty: Optional[float] = None,
        dollars: Optional[float] = None,
        close_position: bool = False,
        reason: Optional[str] = None,
        expires_in_days: Optional[float] = 30.0,
    ) -> dict:
        """Register a conditional order. Validation happens here so the
        store sees only well-formed inputs.

        Returns dict with `ok=True` and `order_id` on success, or
        `ok=False` + `error` on rejection. Does NOT require market hours
        — orders persist across the closed window and the watcher only
        fires them during RTH.
        """
        ticker = (ticker or "").strip().upper()
        if not ticker:
            return {"ok": False, "error": "Ticker required."}
        if len(ticker) > _MAX_TICKER_LEN:
            return {
                "ok": False,
                "error": (
                    f"Ticker too long ({len(ticker)} chars > "
                    f"{_MAX_TICKER_LEN}). Real tickers are 1-5 chars; "
                    f"ETFs go up to ~10 with suffixes."
                ),
            }
        if side not in (SIDE_BUY, SIDE_SELL):
            return {"ok": False, "error": f"Invalid side {side!r}; must be 'buy' or 'sell'."}
        if kind not in VALID_ORDER_KINDS:
            return {"ok": False, "error": f"Invalid kind {kind!r}; must be 'stop' or 'limit'."}
        if not _finite(trigger_price) or trigger_price <= 0:
            return {"ok": False, "error": "trigger_price must be a positive finite number."}
        if trigger_price > _MAX_TRIGGER_PRICE:
            return {
                "ok": False,
                "error": (
                    f"trigger_price ${trigger_price:,.2f} exceeds the "
                    f"sanity ceiling ${_MAX_TRIGGER_PRICE:,.0f}. Did "
                    f"you mean a smaller number?"
                ),
            }
        n_set = sum(
            1 for x in (
                qty is not None, dollars is not None, bool(close_position),
            ) if x
        )
        if n_set != 1:
            return {
                "ok": False,
                "error": (
                    "Specify exactly one of qty, dollars, or close_position. "
                    "(close_position is sell-only; dollars is buy-only.)"
                ),
            }
        if dollars is not None:
            if side != SIDE_BUY:
                return {"ok": False, "error": "dollars is only valid on buy orders."}
            if not _finite(dollars) or dollars <= 0:
                return {"ok": False, "error": "dollars must be a positive finite number."}
        if qty is not None:
            if not _finite(qty) or qty <= 0:
                return {"ok": False, "error": "qty must be a positive finite number."}
        if close_position and side != SIDE_SELL:
            return {"ok": False, "error": "close_position is sell-only."}
        if not reason:
            return {"ok": False, "error": "reason required (one-sentence thesis)."}
        # Truncate over-long reasons rather than reject — the LLM
        # sometimes gets verbose, and dropping the order entirely
        # would be more frustrating than silently clipping the thesis.
        if len(reason) > _MAX_REASON_LEN:
            reason = reason[:_MAX_REASON_LEN]

        # Cap pending-order count per context. Without this, a runaway
        # LLM (or a deliberately-poisoned prompt) can flood the watcher
        # with hundreds of orders, blowing up quote-provider load.
        existing = await self.store.list_orders_for_context(
            context_key, statuses={ORDER_PENDING},
            limit=_MAX_PENDING_ORDERS_PER_CONTEXT + 1,
        )
        if len(existing) >= _MAX_PENDING_ORDERS_PER_CONTEXT:
            return {
                "ok": False,
                "error": (
                    f"Too many pending orders in this chat "
                    f"({len(existing)}/{_MAX_PENDING_ORDERS_PER_CONTEXT}). "
                    f"Cancel some via portfolio_cancel_order before "
                    f"placing more."
                ),
            }

        # Reject sell orders for tickers we don't currently hold. Without
        # this, the bot can sit on a stale sell-stop forever for a name
        # it never bought, which is dead state and surprises the chat.
        if side == SIDE_SELL:
            pos = await self.store.get_position(context_key, ticker)
            if pos is None:
                return {
                    "ok": False,
                    "error": (
                        f"No {ticker} position to sell against. Place a "
                        f"buy order first or buy at market."
                    ),
                }

        # Sanity-check the trigger against the current quote. If the
        # trigger has already crossed, the watcher will fire the order
        # on its next tick — flag it in the result so the LLM knows
        # what's about to happen instead of being surprised by an
        # immediate fill. Quote failures are non-fatal: register
        # anyway, watcher will retry quoting.
        warning: Optional[str] = None
        try:
            current_price, qerr = await self._fetch_price(ticker)
        except Exception as e:
            current_price, qerr = None, str(e)
        if current_price is not None:
            already_crossed = False
            if side == SIDE_BUY and kind == KIND_STOP and current_price >= trigger_price:
                already_crossed = True
            elif side == SIDE_BUY and kind == KIND_LIMIT and current_price <= trigger_price:
                already_crossed = True
            elif side == SIDE_SELL and kind == KIND_STOP and current_price <= trigger_price:
                already_crossed = True
            elif side == SIDE_SELL and kind == KIND_LIMIT and current_price >= trigger_price:
                already_crossed = True
            if already_crossed:
                warning = (
                    f"trigger ${trigger_price:.2f} is already crossed "
                    f"(current ${current_price:.2f}); order will fire on "
                    f"next watcher tick"
                )

        expires_at: Optional[float] = None
        if expires_in_days is not None and expires_in_days > 0:
            expires_at = time.time() + float(expires_in_days) * 86400.0

        try:
            order_id = await self.store.create_order(
                context_key,
                ticker=ticker, side=side, kind=kind,
                trigger_price=float(trigger_price),
                qty=qty, dollars=dollars, close_position=close_position,
                reason=reason, expires_at=expires_at,
            )
        except ValueError as e:
            return {"ok": False, "error": str(e)}
        except Exception as e:
            logger.exception(f"place_order: store insert failed: {e}")
            return {"ok": False, "error": f"store error: {type(e).__name__}"}

        return {
            "ok": True,
            "order_id": order_id,
            "ticker": ticker,
            "side": side,
            "kind": kind,
            "trigger_price": float(trigger_price),
            "current_price": current_price,
            "expires_at": expires_at,
            "warning": warning,
        }

    async def cancel_order(
        self,
        context_key: str,
        order_id: int,
    ) -> dict:
        """LLM-facing cancel. Scoped to the calling context — see
        PortfolioStore.cancel_order for the rationale."""
        try:
            status = await self.store.cancel_order(
                int(order_id), context_key=context_key,
            )
        except Exception as e:
            logger.exception(f"cancel_order failed: {e}")
            return {"ok": False, "error": f"store error: {type(e).__name__}"}
        if status == "ok":
            return {"ok": True, "order_id": int(order_id)}
        return {"ok": False, "error": status}

    async def try_fill_pending(self) -> dict:
        """Scan all pending orders, fire those whose triggers have
        crossed. One quote per ticker (regardless of how many orders
        reference it). Returns a stats dict for the worker to log.

        Caller is expected to have already gated on market hours and to
        run this on a multi-minute cadence — there is no debouncing
        inside this method.
        """
        # Expire stale orders up front so they don't waste a quote slot.
        n_expired = await self.store.expire_stale_orders()

        orders = await self.store.list_pending_orders()
        if not orders:
            return {
                "scanned": 0, "filled": 0, "failed": 0, "expired": n_expired,
                "tickers_quoted": 0,
            }

        unique_tickers = sorted({o.ticker for o in orders})
        # Parallel quote fetch — same shape as status()'s mark-to-market
        # call. Each quote is independent so a slow ticker doesn't
        # block the rest.
        quote_results = await asyncio.gather(
            *(self._fetch_price(t) for t in unique_tickers),
            return_exceptions=False,
        )
        prices: dict[str, Optional[float]] = {}
        for ticker, (price, err) in zip(unique_tickers, quote_results):
            if err is not None or price is None:
                logger.warning(
                    f"orders watcher: quote miss for {ticker} ({err}); "
                    f"orders on this ticker stay pending this tick"
                )
                prices[ticker] = None
            else:
                prices[ticker] = price

        n_filled = 0
        n_failed = 0
        for order in orders:
            price = prices.get(order.ticker)
            if price is None:
                continue
            if not order.should_fire(price):
                continue
            ok, note = await self._fire_order(order, price)
            if ok:
                n_filled += 1
            else:
                n_failed += 1
                logger.warning(
                    f"orders watcher: fill failed id={order.id} "
                    f"{order.ticker} {order.side}/{order.kind}: {note}"
                )

        return {
            "scanned": len(orders),
            "filled": n_filled,
            "failed": n_failed,
            "expired": n_expired,
            "tickers_quoted": len(unique_tickers),
        }

    async def _fire_order(
        self,
        order: Order,
        current_price: float,
    ) -> tuple[bool, str]:
        """Execute a triggered order via the existing buy/sell paths.
        Returns (ok, note) — note is a short string used for logs and
        for fill_note when the underlying execution rejects.

        Atomic claim → execute → finalize: at-most-once execution is
        guaranteed by claim_for_fill, which moves pending → in_flight
        in a single UPDATE conditional on status='pending'. If the
        claim returns False (race with another tick, cancel, or
        expiry), we abort without firing. If the process crashes
        between the claim and the finalize UPDATE in mark_order_*
        the order stays `in_flight` and is never re-fired — a stuck
        row is preferable to double-debiting cash."""
        # Atomic claim. Two ticks racing on the same order can both
        # call _fire_order; only one wins the UPDATE. The loser bails
        # quietly here.
        claimed = await self.store.claim_for_fill(order.id)
        if not claimed:
            return False, "claim lost (cancelled, already filled, or in-flight elsewhere)"

        # Use a "trigger" reason prefix so the chat can tell at a glance
        # that this fill came from an automatic order, not a fresh LLM
        # decision.
        reason = (
            f"[order #{order.id}: {order.kind}-{order.side} @ "
            f"${order.trigger_price:.2f}] {order.reason or ''}"
        ).strip()
        # Captured before the fill for close_position=True orders so
        # the order ledger records the actual sold qty (post-sell the
        # position is gone and we lose the number).
        close_qty: Optional[float] = None
        if order.side == SIDE_BUY:
            try:
                result = await self.execute_buy(
                    order.context_key,
                    ticker=order.ticker,
                    dollars=order.dollars,
                    qty=order.qty,
                    reason=reason,
                    source=SOURCE_ORDER,  # order watcher drove this fill
                )
            except Exception as e:
                await self.store.mark_order_failed(
                    order.id, note=f"execute_buy raised: {type(e).__name__}",
                )
                return False, f"execute_buy raised: {e}"
        else:
            # For close-all sells we capture the position size BEFORE
            # the fill so the order ledger records the actual qty sold,
            # not a 0 sentinel. Done outside execute_sell because the
            # sell itself zeroes the position.
            if order.close_position:
                pos = await self.store.get_position(
                    order.context_key, order.ticker,
                )
                close_qty = pos.qty if pos is not None else None
            try:
                # close_position=True → sell entire position; otherwise
                # sell the exact qty the order specified. This handles
                # the case where the bot trimmed/added to the position
                # between order placement and fill — close means close,
                # not "sell the qty I had at registration".
                qty_arg: Union[float, str, None]
                if order.close_position:
                    qty_arg = "all"
                else:
                    qty_arg = order.qty
                result = await self.execute_sell(
                    order.context_key,
                    ticker=order.ticker,
                    qty=qty_arg,
                    reason=reason,
                    source=SOURCE_ORDER,
                )
            except Exception as e:
                await self.store.mark_order_failed(
                    order.id, note=f"execute_sell raised: {type(e).__name__}",
                )
                return False, f"execute_sell raised: {e}"

        if not result.get("ok"):
            err_text = str(result.get("error") or "rejected")
            await self.store.mark_order_failed(order.id, note=err_text)
            return False, err_text

        # Fill price comes from the executor's actual fill, NOT the
        # trigger — slippage between trigger crossing and quote fetch
        # is real, and the trade record reflects what actually filled.
        fill_price = float(result.get("price") or current_price)
        # Reconstruct fill_qty from the order's original intent, since
        # buy/sell results return position state rather than fill
        # delta. For close_position=True we captured the pre-fill qty
        # above so the order ledger records what was actually sold,
        # not a sentinel zero.
        if order.qty is not None:
            fill_qty = float(order.qty)
        elif order.dollars is not None and fill_price > 0:
            fill_qty = float(order.dollars) / fill_price
        elif order.side == SIDE_SELL and order.close_position:
            # `close_qty` was captured above for sell-side close orders.
            # If the position vanished mid-fire (race with another
            # sell), fall back to 0 — the trade row in `trades` is the
            # source of truth, this is just a fill summary on the order.
            fill_qty = float(close_qty) if close_qty is not None else 0.0
        else:
            fill_qty = 0.0
        await self.store.mark_order_filled(
            order.id, fill_price=fill_price, fill_qty=fill_qty,
        )
        return True, f"filled {fill_qty:.4f} @ ${fill_price:.2f}"

    # ------------------------------------------------------------------
    # Options trading (long-only, single-leg, 100x multiplier)
    # ------------------------------------------------------------------

    async def options_chain(
        self,
        underlying: str,
        *,
        expiration: Optional[str] = None,
        limit: int = 100,
    ) -> dict:
        """List available contracts on `underlying`. Pure read — does
        not check market hours (chains are publishable any time)."""
        underlying = (underlying or "").strip().upper()
        if not underlying:
            return {"ok": False, "error": "underlying required"}
        try:
            quotes = await self.providers.get_options_chain(
                underlying, expiration=expiration, limit=limit,
            )
        except NotImplementedError as e:
            return {"ok": False, "error": f"chain not supported: {e}"}
        except Exception as e:
            logger.warning(f"options_chain({underlying}): {e}")
            return {"ok": False, "error": f"chain fetch failed: {type(e).__name__}"}
        rows = [
            {
                "contract": getattr(q, "symbol", ""),
                "underlying": getattr(q, "underlying", underlying),
                "option_type": getattr(q, "type", "unknown"),
                "strike": float(getattr(q, "strike", 0.0) or 0.0),
                "expiration": (
                    q.expiration.isoformat() if getattr(q, "expiration", None)
                    else None
                ),
                "premium": float(getattr(q, "price", 0.0) or 0.0),
                "volume": int(getattr(q, "volume", 0) or 0),
                "open_interest": int(getattr(q, "open_interest", 0) or 0),
                "iv": getattr(q, "implied_volatility", None),
                "delta": (getattr(q, "greeks", None) or {}).get("delta") if getattr(q, "greeks", None) else None,
            }
            for q in quotes
        ]
        return {"ok": True, "rows": rows, "count": len(rows)}

    async def execute_buy_option(
        self,
        context_key: str,
        *,
        contract: str,
        qty: int,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
        force_market_open: bool = False,
        multiplier: int = 100,
    ) -> dict:
        """Open or add to a long options position. `contract` may be a
        canonical OCC symbol or a friendly form (``"AAPL 175C
        2026-06-20"``) — both are normalized. Quantity is whole
        contracts (we don't trade partial options)."""
        if source not in VALID_SOURCES:
            return {"ok": False, "error": f"invalid source: {source!r}"}
        if not isinstance(qty, (int, float)) or qty != int(qty) or int(qty) <= 0:
            return {"ok": False, "error": "qty must be a positive whole number of contracts."}
        qty_int = int(qty)
        try:
            occ = normalize_contract(contract)
        except ValueError as e:
            return {"ok": False, "error": f"contract not parseable: {e}"}
        try:
            parts = parse_occ(occ)
        except ValueError as e:
            return {"ok": False, "error": f"contract not parseable: {e}"}

        if not force_market_open:
            closed = market_closed_reason()
            if closed:
                return {"ok": False, "error": f"Trade rejected — {closed}."}

        # Reject opening a position on an already-expired contract — the
        # provider will happily quote an expired symbol's last-known
        # close, but the bot would just immediately get settled out.
        exp_ts = dt.datetime.combine(
            parts.expiration, dt.time(16, 0), tzinfo=_ET,
        ).timestamp()
        if exp_ts < dt.datetime.now(_ET).timestamp():
            return {"ok": False, "error": f"contract {occ} is already expired"}

        premium, err = await self._fetch_option_premium(occ)
        if err is not None or premium is None:
            return {"ok": False, "error": err or "no option quote"}

        result = await self.store.buy_option(
            context_key,
            contract_symbol=occ,
            underlying=parts.root,
            option_type=parts.option_type,
            strike=parts.strike,
            expiration=exp_ts,
            qty=qty_int,
            premium=premium,
            multiplier=multiplier,
            reason=reason,
            source=source,
        )
        if result.get("ok"):
            result["contract"] = occ
            result["friendly"] = occ_friendly_name(occ)
            result["premium"] = premium
            result["multiplier"] = multiplier
            result["underlying"] = parts.root
            result["option_type"] = parts.option_type
            result["strike"] = parts.strike
            result["expiration"] = exp_ts
        return result

    async def execute_sell_option(
        self,
        context_key: str,
        *,
        contract: str,
        qty: Union[int, str, None] = None,
        reason: Optional[str] = None,
        source: str = SOURCE_REACTIVE,
        force_market_open: bool = False,
    ) -> dict:
        """Close (some of) a long options position. ``qty="all"`` (or
        None) closes the position; an integer closes that many
        contracts. V1 is long-only — no opening-short selling here."""
        if source not in VALID_SOURCES:
            return {"ok": False, "error": f"invalid source: {source!r}"}
        try:
            occ = normalize_contract(contract)
        except ValueError as e:
            return {"ok": False, "error": f"contract not parseable: {e}"}

        if not force_market_open:
            closed = market_closed_reason()
            if closed:
                return {"ok": False, "error": f"Trade rejected — {closed}."}

        sell_qty: Optional[int]
        if qty is None or (isinstance(qty, str) and qty.strip().lower() == "all"):
            pos = await self.store.get_option_position(context_key, occ)
            if pos is None:
                return {"ok": False, "error": f"No position in {occ}."}
            sell_qty = int(round(pos.qty))
        else:
            try:
                sell_qty = int(qty)
            except (TypeError, ValueError):
                return {"ok": False, "error": f"Invalid qty: {qty!r}"}
            if sell_qty <= 0:
                return {"ok": False, "error": "qty must be positive."}

        premium, err = await self._fetch_option_premium(occ)
        if err is not None or premium is None:
            return {"ok": False, "error": err or "no option quote"}

        result = await self.store.sell_option(
            context_key,
            contract_symbol=occ,
            qty=sell_qty,
            premium=premium,
            reason=reason,
            source=source,
        )
        if result.get("ok"):
            result["contract"] = occ
            result["friendly"] = occ_friendly_name(occ)
            result["premium"] = premium
        return result

    async def settle_expired_options(
        self,
        *,
        now_ts: Optional[float] = None,
    ) -> dict:
        """Cash-settle every expired options position across all
        contexts. ITM contracts close at intrinsic × multiplier × qty,
        OTM contracts close at $0 (expire worthless). Idempotent:
        once settled the row is deleted, so subsequent ticks no-op.

        Settlement uses the underlying's current quote. For positions
        that expired during a long weekend or while the bot was down,
        this is the next-available close price — the best approximation
        we have without intraday history.
        """
        positions = await self.store.list_options_positions_all()
        now = now_ts if now_ts is not None else time.time()
        # Filter to positions whose 16:00 ET expiration timestamp is in
        # the past. The store stamps expiration this way already
        # (execute_buy_option computes 16:00 ET of expiry day).
        due = [op for op in positions if op.expiration <= now]
        if not due:
            return {"checked": len(positions), "settled": 0, "errors": 0}

        # Group by underlying so we fetch each spot once even when
        # multiple contracts of the same name expire together.
        underlyings = sorted({op.underlying for op in due})
        spots = await asyncio.gather(
            *(self._fetch_price(u) for u in underlyings),
            return_exceptions=False,
        )
        spot_map: dict[str, Optional[float]] = {}
        for sym, (price, err) in zip(underlyings, spots):
            if err is not None or price is None:
                logger.warning(
                    f"settle: spot quote failed for {sym} ({err}); "
                    f"contracts on this underlying will stay open until "
                    f"the next sweep"
                )
                spot_map[sym] = None
            else:
                spot_map[sym] = price

        settled = 0
        errors = 0
        for op in due:
            spot = spot_map.get(op.underlying)
            if spot is None:
                errors += 1
                continue
            intrinsic = occ_intrinsic_value(op.option_type, op.strike, spot)
            reason = (
                f"auto-settled at expiration: {op.option_type.upper()} "
                f"{op.underlying} @ ${op.strike:.2f} vs spot ${spot:.2f} "
                f"→ intrinsic ${intrinsic:.4f}/sh"
            )
            try:
                result = await self.store.sell_option(
                    op.context_key,
                    contract_symbol=op.contract_symbol,
                    qty=op.qty,
                    premium=intrinsic,
                    reason=reason,
                    source=SOURCE_SETTLEMENT,
                    settlement_intrinsic=intrinsic,
                )
            except Exception as e:
                logger.exception(
                    f"settle: store.sell_option raised for "
                    f"{op.contract_symbol}: {e}"
                )
                errors += 1
                continue
            if not result.get("ok"):
                logger.warning(
                    f"settle: rejected for {op.contract_symbol}: "
                    f"{result.get('error')}"
                )
                errors += 1
                continue
            settled += 1
            logger.info(
                f"settle: {op.contract_symbol} (ctx={op.context_key}) "
                f"qty={op.qty:g} @ ${intrinsic:.4f}/sh "
                f"realized=${result.get('realized_pnl') or 0:.2f}"
            )
        return {
            "checked": len(positions),
            "due": len(due),
            "settled": settled,
            "errors": errors,
        }
