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
# Options-specific caps. Per-fill: 10k contracts at $1 premium = $1M
# cost — far past any reasonable paper-portfolio budget but cheap to
# enforce. Per-context: 200 distinct open positions covers an active
# strategy without letting a runaway LLM mint thousands of unique
# rows the settlement worker has to iterate every tick.
_MAX_OPTION_CONTRACTS_PER_FILL = 10_000
_MAX_OPTION_POSITIONS_PER_CONTEXT = 200
# Settlement poison-row policy. When the underlying's quote stays
# unavailable for more than this many seconds past expiration, the
# settle path force-closes the contract at $0 (expired worthless) so
# the row doesn't sit in options_positions forever, retried on every
# tick of the worker. Realized PnL records the full premium loss —
# worst-case for the user but the honest outcome when we can't
# determine intrinsic.
_SETTLEMENT_STALE_GRACE_SECONDS = 86400  # 24 hours


def _order_to_view(order: Order) -> dict:
    """Render an Order for status() consumers (LLM tool result, image
    renderer, admin UI). Same shape used everywhere downstream so the
    image and text paths can rely on identical keys."""
    return {
        "id": order.id,
        "ticker": order.ticker,
        "contract": order.contract_symbol,
        "is_option": bool(order.contract_symbol),
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
        """Returns (price, error). Exactly one is None.

        NaN/Inf are explicitly rejected — `price <= 0` is False for
        NaN, so without this check a poisoned quote (provider bug,
        delisted stub) would slip through into downstream math.
        """
        try:
            quote = await self.providers.get_quote(ticker)
        except Exception as e:
            return None, f"couldn't quote {ticker}: {type(e).__name__}"
        price = getattr(quote, "price", None)
        if price is None:
            return None, f"no valid quote for {ticker}"
        try:
            price_f = float(price)
        except (TypeError, ValueError):
            return None, f"non-numeric quote for {ticker}"
        if not math.isfinite(price_f) or price_f <= 0:
            return None, f"no valid quote for {ticker} (got {price_f!r})"
        return price_f, None

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

    async def _fetch_option_premiums_batched(
        self, options_pos,
    ) -> list[tuple[Optional[float], Optional[str]]]:
        """Mark-to-market premium for each position, grouped by
        underlying so multiple contracts on the same name share one
        chain fetch. Result order matches input order so the caller
        can zip directly against `options_pos`.

        Single-contract underlyings fall back to per-contract quotes
        (the chain endpoint returns up to 250 rows; querying it for
        one strike wastes bandwidth and the provider's quota).
        """
        from collections import defaultdict
        groups: dict[str, list[int]] = defaultdict(list)
        for i, op in enumerate(options_pos):
            groups[op.underlying].append(i)

        results: list[tuple[Optional[float], Optional[str]]] = [
            (None, "unfilled") for _ in options_pos
        ]

        # Schedule fetches: chain for any underlying with ≥2 holdings,
        # per-contract otherwise. asyncio.gather runs them all in
        # parallel.
        tasks: list = []
        task_meta: list[tuple[str, list[int]]] = []  # (kind, indices)
        for underlying, indices in groups.items():
            if len(indices) >= 2:
                tasks.append(self._safe_chain_lookup(underlying))
                task_meta.append(("chain", indices))
            else:
                idx = indices[0]
                tasks.append(self._fetch_option_premium(
                    options_pos[idx].contract_symbol,
                ))
                task_meta.append(("single", indices))

        responses = await asyncio.gather(*tasks, return_exceptions=False)

        for response, (kind, indices) in zip(responses, task_meta):
            if kind == "single":
                # Direct quote result is already (premium, err).
                results[indices[0]] = response
                continue
            # Chain result: {contract_symbol -> premium}. Missing
            # contracts get a fallback per-contract quote.
            chain_map, chain_err = response
            missing: list[int] = []
            for idx in indices:
                contract = options_pos[idx].contract_symbol
                if chain_err is not None:
                    missing.append(idx)
                    continue
                premium = chain_map.get(contract)
                if premium is None or premium <= 0:
                    missing.append(idx)
                else:
                    results[idx] = (float(premium), None)
            if missing:
                fallbacks = await asyncio.gather(*(
                    self._fetch_option_premium(
                        options_pos[idx].contract_symbol,
                    ) for idx in missing
                ), return_exceptions=False)
                for idx, premium_err in zip(missing, fallbacks):
                    results[idx] = premium_err

        return results

    async def _safe_chain_lookup(
        self, underlying: str,
    ) -> tuple[dict, Optional[str]]:
        """Fetch the options chain for an underlying and reduce to a
        {contract_symbol -> premium} map. Errors are swallowed and
        surfaced as the `error` half so the caller can fall back to
        per-contract quotes instead of failing the whole snapshot."""
        try:
            chain = await self.providers.get_options_chain(
                underlying, expiration=None, limit=250,
            )
        except Exception as e:
            logger.debug(
                f"status: chain fetch for {underlying} failed "
                f"({type(e).__name__}); will fall back to per-contract quotes"
            )
            return {}, str(e)
        out: dict[str, float] = {}
        for q in chain:
            symbol = getattr(q, "symbol", "")
            if symbol.startswith("O:"):
                symbol = symbol[2:]
            premium = getattr(q, "price", 0.0) or 0.0
            if symbol and premium > 0:
                out[symbol] = float(premium)
        return out, None

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

    async def status(
        self, context_key: str, *, label_hint: Optional[str] = None,
    ) -> dict:
        """Snapshot for !portfolio / portfolio_status tool. Includes
        cash, positions with mark-to-market, total PnL.

        `label_hint` (when set) is passed through to ensure_portfolio
        for first-time creation only — `INSERT OR IGNORE` means an
        existing portfolio's label isn't overwritten. Callers pass the
        bot's display_name so newly-seeded per-bot portfolios get
        named (otherwise the caption and admin view show empty labels
        forever).

        Quote failures on individual positions don't block the snapshot
        — those positions show `mark_price=None` and contribute their
        cost basis to total value (a conservative fallback that doesn't
        invent gains). Logged at warn level.
        """
        portfolio = await self.store.ensure_portfolio(
            context_key, label=label_hint,
        )
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
        #
        # Performance: when the portfolio holds multiple contracts on
        # the same underlying (typical: a long-call + protective put
        # on one name, or several strikes), prefer ONE chain fetch
        # per underlying over N per-contract quotes. Single-position
        # underlyings still use the direct quote — the chain endpoint
        # returns up to 250 rows, wasted bandwidth for a one-strike
        # case.
        option_views: list[dict] = []
        option_market_value = 0.0
        if options_pos:
            option_quote_results = await self._fetch_option_premiums_batched(options_pos)
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
        ticker: Optional[str] = None,
        contract: Optional[str] = None,
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

        Two modes — discriminated by which of `ticker` / `contract` is
        set:
          - Equity: pass `ticker`. trigger_price = equity share price.
          - Options: pass `contract` (OCC or friendly). trigger_price =
            per-share PREMIUM the order fires on. `dollars` is not
            supported for options; `qty` is whole contracts.

        Returns dict with `ok=True` and `order_id` on success, or
        `ok=False` + `error` on rejection. Does NOT require market
        hours — orders persist across the closed window and the watcher
        only fires them during RTH.
        """
        # Discriminate equity vs options. Caller passes one or the
        # other; `contract` takes precedence if both somehow arrive.
        contract_symbol: Optional[str] = None
        is_option = bool(contract and contract.strip())
        if is_option:
            try:
                contract_symbol = normalize_contract(contract or "")
            except ValueError as e:
                return {"ok": False, "error": f"contract not parseable: {e}"}
            try:
                parts = parse_occ(contract_symbol)
            except ValueError as e:
                return {"ok": False, "error": f"contract not parseable: {e}"}
            # ticker column stores the underlying — used as the
            # grouping/display key in admin and chat views.
            ticker = parts.root
        else:
            ticker = (ticker or "").strip().upper()
        if not ticker:
            return {"ok": False, "error": "Ticker (or contract) required."}
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
            if is_option:
                return {
                    "ok": False,
                    "error": (
                        "dollars is not supported for options orders. "
                        "Pass qty (number of contracts) or close_position."
                    ),
                }
            if side != SIDE_BUY:
                return {"ok": False, "error": "dollars is only valid on buy orders."}
            if not _finite(dollars) or dollars <= 0:
                return {"ok": False, "error": "dollars must be a positive finite number."}
        if qty is not None:
            if not _finite(qty) or qty <= 0:
                return {"ok": False, "error": "qty must be a positive finite number."}
            if is_option and qty != int(qty):
                return {
                    "ok": False,
                    "error": "qty must be a whole number of contracts.",
                }
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

        # Reject sell orders for positions we don't currently hold.
        # For equities check the share position; for options check the
        # specific contract position. Either way, sitting on a stale
        # sell-stop forever for something never opened is dead state.
        if side == SIDE_SELL:
            if is_option and contract_symbol is not None:
                opt_pos = await self.store.get_option_position(
                    context_key, contract_symbol,
                )
                if opt_pos is None:
                    return {
                        "ok": False,
                        "error": (
                            f"No {contract_symbol} options position to "
                            f"sell against. Buy contracts first."
                        ),
                    }
            else:
                pos = await self.store.get_position(context_key, ticker)
                if pos is None:
                    return {
                        "ok": False,
                        "error": (
                            f"No {ticker} position to sell against. Place a "
                            f"buy order first or buy at market."
                        ),
                    }

        # Sanity-check the trigger against the current quote. For
        # options the "quote" is the per-share premium; for equities
        # it's the share price. If the trigger has already crossed,
        # the watcher will fire the order on its next tick — flag it
        # in the result so the LLM knows what's about to happen instead
        # of being surprised by an immediate fill. Quote failures are
        # non-fatal: register anyway, watcher will retry quoting.
        warning: Optional[str] = None
        try:
            if is_option and contract_symbol is not None:
                current_price, qerr = await self._fetch_option_premium(contract_symbol)
            else:
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
                contract_symbol=contract_symbol,
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
            "contract": contract_symbol,
            "is_option": is_option,
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
        crossed.

        For equity orders the trigger compares against the underlying
        equity quote (one quote per ticker). For options orders the
        trigger compares against the per-share option premium (one
        quote per contract). Both kinds of quotes are fetched in
        parallel; each order references whichever applies via its
        `contract_symbol` field.
        """
        # Expire stale orders up front so they don't waste a quote slot.
        n_expired = await self.store.expire_stale_orders()

        orders = await self.store.list_pending_orders()
        if not orders:
            return {
                "scanned": 0, "filled": 0, "failed": 0, "expired": n_expired,
                "tickers_quoted": 0,
            }

        equity_orders = [o for o in orders if not o.contract_symbol]
        option_orders = [o for o in orders if o.contract_symbol]
        unique_tickers = sorted({o.ticker for o in equity_orders})
        unique_contracts = sorted({o.contract_symbol for o in option_orders if o.contract_symbol})

        # Parallel quote fetches: equity quotes for unique_tickers and
        # option premiums for unique_contracts. asyncio.gather preserves
        # order so we can split the result list back into the two maps.
        all_tasks = [
            *(self._fetch_price(t) for t in unique_tickers),
            *(self._fetch_option_premium(c) for c in unique_contracts),
        ]
        all_results = await asyncio.gather(*all_tasks, return_exceptions=False)
        eq_results = all_results[:len(unique_tickers)]
        opt_results = all_results[len(unique_tickers):]

        eq_prices: dict[str, Optional[float]] = {}
        for ticker, (price, err) in zip(unique_tickers, eq_results):
            if err is not None or price is None:
                logger.warning(
                    f"orders watcher: equity quote miss for {ticker} "
                    f"({err}); orders on this ticker stay pending this tick"
                )
                eq_prices[ticker] = None
            else:
                eq_prices[ticker] = price
        opt_prices: dict[str, Optional[float]] = {}
        for contract, (price, err) in zip(unique_contracts, opt_results):
            if err is not None or price is None:
                logger.warning(
                    f"orders watcher: option quote miss for {contract} "
                    f"({err}); orders on this contract stay pending this tick"
                )
                opt_prices[contract] = None
            else:
                opt_prices[contract] = price

        n_filled = 0
        n_failed = 0
        for order in orders:
            if order.contract_symbol:
                price = opt_prices.get(order.contract_symbol)
            else:
                price = eq_prices.get(order.ticker)
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
            "tickers_quoted": len(unique_tickers) + len(unique_contracts),
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
        is_option_order = bool(order.contract_symbol)
        # Captured before the fill for close_position=True orders so
        # the order ledger records the actual sold qty (post-sell the
        # position is gone and we lose the number).
        close_qty: Optional[float] = None
        if order.side == SIDE_BUY:
            try:
                if is_option_order:
                    # contract_symbol is guaranteed set when
                    # is_option_order is True — guarded above.
                    result = await self.execute_buy_option(
                        order.context_key,
                        contract=str(order.contract_symbol),
                        qty=int(order.qty or 0),
                        reason=reason,
                        source=SOURCE_ORDER,
                    )
                else:
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
                    order.id,
                    note=f"execute_buy raised: {type(e).__name__}",
                )
                return False, f"execute_buy raised: {e}"
        else:
            # For close-all sells we capture the position size BEFORE
            # the fill so the order ledger records the actual qty sold,
            # not a 0 sentinel. Done outside execute_sell because the
            # sell itself zeroes the position.
            if order.close_position:
                if is_option_order:
                    opt_pos = await self.store.get_option_position(
                        order.context_key, order.contract_symbol or "",
                    )
                    close_qty = opt_pos.qty if opt_pos is not None else None
                else:
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
                if is_option_order:
                    # Options sell qty is an integer of contracts.
                    option_qty_arg: Union[int, str, None]
                    if order.close_position:
                        option_qty_arg = "all"
                    elif order.qty is not None:
                        option_qty_arg = int(order.qty)
                    else:
                        option_qty_arg = None
                    result = await self.execute_sell_option(
                        order.context_key,
                        contract=str(order.contract_symbol),
                        qty=option_qty_arg,
                        reason=reason,
                        source=SOURCE_ORDER,
                    )
                else:
                    result = await self.execute_sell(
                        order.context_key,
                        ticker=order.ticker,
                        qty=qty_arg,
                        reason=reason,
                        source=SOURCE_ORDER,
                    )
            except Exception as e:
                await self.store.mark_order_failed(
                    order.id,
                    note=f"execute_sell raised: {type(e).__name__}",
                )
                return False, f"execute_sell raised: {e}"

        if not result.get("ok"):
            err_text = str(result.get("error") or "rejected")
            await self.store.mark_order_failed(order.id, note=err_text)
            return False, err_text

        # Fill price comes from the executor's actual fill, NOT the
        # trigger — slippage between trigger crossing and quote fetch
        # is real, and the trade record reflects what actually filled.
        # For options, the executor returns `premium`; for equities,
        # `price`.
        fill_price = float(
            result.get("price")
            or result.get("premium")
            or current_price
        )
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
                # Date-only ISO (YYYY-MM-DD) so the LLM can copy this
                # back into `expiration` on a subsequent chain call
                # without the parser rejecting a timestamp form.
                "expiration": (
                    q.expiration.date().isoformat() if getattr(q, "expiration", None)
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
        # Strict qty validation. Order matters: math.isfinite ruled out
        # before int() so NaN/Inf don't escape as uncaught exceptions
        # ("cannot convert float NaN to integer").
        if not isinstance(qty, (int, float)):
            return {"ok": False, "error": "qty must be a positive whole number of contracts."}
        if not math.isfinite(qty):
            return {"ok": False, "error": "qty must be a finite number."}
        if qty != int(qty) or int(qty) <= 0:
            return {"ok": False, "error": "qty must be a positive whole number of contracts."}
        if int(qty) > _MAX_OPTION_CONTRACTS_PER_FILL:
            return {
                "ok": False,
                "error": (
                    f"qty {int(qty)} exceeds sanity cap "
                    f"{_MAX_OPTION_CONTRACTS_PER_FILL}; split into smaller fills."
                ),
            }
        qty_int = int(qty)
        # Apply the same reason-length truncation that the equity
        # place_order path uses, so a verbose LLM can't fill the table
        # with multi-kilobyte reasons.
        if reason and len(reason) > _MAX_REASON_LEN:
            reason = reason[:_MAX_REASON_LEN]
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

        # Cap distinct open positions per context. Existing position on
        # this contract counts as 0 new — only the (qty>0, position-
        # didn't-exist) case is rejected. This stops a runaway loop
        # opening hundreds of one-strike positions; adding to one is
        # always allowed.
        existing = await self.store.get_option_position(context_key, occ)
        if existing is None:
            current = await self.store.options_positions(context_key)
            if len(current) >= _MAX_OPTION_POSITIONS_PER_CONTEXT:
                return {
                    "ok": False,
                    "error": (
                        f"Too many open option positions in this chat "
                        f"({len(current)}/{_MAX_OPTION_POSITIONS_PER_CONTEXT}). "
                        f"Close some before opening new contracts."
                    ),
                }

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
        if reason and len(reason) > _MAX_REASON_LEN:
            reason = reason[:_MAX_REASON_LEN]
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
            # Mirror the strict-int validation from execute_buy_option:
            # NaN/Inf are screened before int() conversion, and
            # fractional qty is rejected instead of silently truncated.
            if not isinstance(qty, (int, float)):
                return {"ok": False, "error": f"qty must be a whole number, got {qty!r}"}
            if not math.isfinite(qty):
                return {"ok": False, "error": "qty must be a finite number."}
            if qty != int(qty) or int(qty) <= 0:
                return {
                    "ok": False,
                    "error": "qty must be a positive whole number of contracts.",
                }
            sell_qty = int(qty)

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

        Returns a stats dict plus a ``settlements`` list of per-position
        outcomes (context_key, contract, side, intrinsic, realized_pnl,
        underlying, option_type, strike) so the caller can notify the
        originating chats. Failed-quote positions are NOT included in
        the settlements list (they stay open for the next tick).
        """
        # Push the expiration filter into SQL when available so we don't
        # scan every open option position across every context each tick.
        now = now_ts if now_ts is not None else time.time()
        if hasattr(self.store, "list_options_positions_due"):
            due = await self.store.list_options_positions_due(now)
            checked = len(due)  # approximation — we didn't fetch the rest
        else:
            positions = await self.store.list_options_positions_all()
            due = [op for op in positions if op.expiration <= now]
            checked = len(positions)
        if not due:
            return {"checked": checked, "settled": 0, "errors": 0, "settlements": []}

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
        settlements: list[dict] = []
        for op in due:
            spot = spot_map.get(op.underlying)
            # Poison-row mitigation: if the underlying has been
            # un-quotable for too long (delisted, ticker renamed,
            # persistent provider failure), force-settle at $0 so the
            # row doesn't sit in options_positions forever, retried on
            # every tick. We approximate "too long" with how stale the
            # expiration is relative to now — anything expired more
            # than 24h ago with no quote is treated as worthless.
            if spot is None:
                stale_seconds = now - op.expiration
                if stale_seconds > _SETTLEMENT_STALE_GRACE_SECONDS:
                    logger.warning(
                        f"settle: underlying {op.underlying} unquotable "
                        f"and contract expired {stale_seconds/3600:.1f}h "
                        f"ago — force-settling at $0 (poison-row mitigation)"
                    )
                    intrinsic = 0.0
                    reason = (
                        f"force-settled at $0: underlying {op.underlying} "
                        f"unquotable for {stale_seconds/3600:.1f}h after "
                        f"expiration. Treated as expired worthless."
                    )
                else:
                    errors += 1
                    continue
            else:
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
            settlements.append({
                "context_key": op.context_key,
                "contract": op.contract_symbol,
                "underlying": op.underlying,
                "option_type": op.option_type,
                "strike": op.strike,
                "expiration": op.expiration,
                "qty": op.qty,
                "intrinsic": intrinsic,
                "spot": spot,
                "realized_pnl": result.get("realized_pnl") or 0.0,
                "proceeds": result.get("proceeds") or 0.0,
                "force_settled": spot is None,
            })
            logger.info(
                f"settle: {op.contract_symbol} (ctx={op.context_key}) "
                f"qty={op.qty:g} @ ${intrinsic:.4f}/sh "
                f"realized=${result.get('realized_pnl') or 0:.2f}"
            )
        return {
            "checked": checked,
            "due": len(due),
            "settled": settled,
            "errors": errors,
            "settlements": settlements,
        }
