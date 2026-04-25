"""
Background resolver for due predictions.

Runs as a long-lived asyncio task: every `interval_seconds`, queries due
predictions, judges them, posts the verdict back to the originating chat.

Two judgment paths:
  1. **Structured** — `Prediction.is_structured` (ticker + threshold +
     direction) → fetch live quote, compare, post directly.
  2. **LLM** — for free-form claims, ask the LLM to call the tool with a
     verdict + short note. If the model returns "unclear", we mark the
     prediction `unclear` rather than `expired` so the leaderboard stays
     truthful (an honestly-undecidable claim isn't a wrong one).

Errors during a single prediction's resolution don't kill the worker —
the row stays `pending` and we'll retry on the next sweep.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Optional

from .predictions import (
    Prediction,
    PredictionStore,
    VERDICT_EMOJI,
    VERDICT_RIGHT,
    VERDICT_WRONG,
    VERDICT_UNCLEAR,
)

logger = logging.getLogger(__name__)


_LLM_VERDICT_PROMPT = """\
You are judging whether a prediction came true. Reply with ONLY a JSON \
object (no preamble, no explanation outside the JSON):

  {
    "verdict": "right" | "wrong" | "unclear",
    "note": "<one short sentence of reasoning>"
  }

Rules:
  - "right": the predicted outcome happened.
  - "wrong": the opposite happened, or the deadline passed without it.
  - "unclear": you genuinely cannot tell from public knowledge. Use this \
when the claim is unfalsifiable, the outcome is partially fulfilled, or \
recent events that would settle it are outside your knowledge. Don't use \
it as a cop-out — only when there's real ambiguity.
  - The note must be one sentence, plain prose, no hedging language."""


class PredictionResolver:
    def __init__(
        self,
        *,
        store: PredictionStore,
        provider_manager,
        signal_handler,
        llm_client=None,
        interval_seconds: int = 900,  # 15 min
    ):
        self.store = store
        self.providers = provider_manager
        self.signal = signal_handler
        self.llm = llm_client
        self.interval = interval_seconds

    async def run_forever(self) -> None:
        logger.info(
            f"Prediction resolver started (interval={self.interval}s)"
        )
        while True:
            try:
                await self.sweep_once()
            except asyncio.CancelledError:
                logger.info("Prediction resolver cancelled")
                raise
            except Exception as e:
                logger.error(f"Prediction resolver sweep error: {e}")
            await asyncio.sleep(self.interval)

    async def sweep_once(self) -> int:
        """Process all due predictions. Returns count handled.

        Strategy:
          1. Dedupe structured predictions by ticker; fetch each ticker's
             price once in parallel (5 AAPL predictions resolving on the
             same sweep share one quote fetch).
          2. Resolve all due predictions concurrently. Per-prediction
             errors don't affect the rest of the sweep.
          3. Post verdicts back to their groups sequentially — Signal
             rate-limits bursts and 20 verdict messages in a row would
             feel spammy anyway.
        """
        due = await self.store.list_due()
        if not due:
            return 0

        # Pre-fetch quotes for all unique tickers in a single gather. Use a
        # list (not a set) so the gather order matches the zip order — set
        # iteration order isn't guaranteed identical across two passes.
        seen: set[str] = set()
        tickers: list[str] = []
        for p in due:
            if p.is_structured and p.ticker and p.ticker not in seen:
                seen.add(p.ticker)
                tickers.append(p.ticker)
        quotes: dict[str, float] = {}
        if tickers:
            results = await asyncio.gather(
                *(self._safe_quote(t) for t in tickers),
                return_exceptions=False,
            )
            for ticker, price in zip(tickers, results):
                if price is not None:
                    quotes[ticker] = price

        # Resolve all due predictions in parallel
        outcomes = await asyncio.gather(
            *(self._resolve_one(pred, quotes) for pred in due),
            return_exceptions=True,
        )

        handled = 0
        for pred, outcome in zip(due, outcomes):
            if isinstance(outcome, Exception):
                logger.exception(
                    f"Resolver: error on prediction {pred.id}: {outcome}"
                )
                continue
            if outcome is None:
                continue
            verdict, note = outcome
            await self._post_resolution(pred, verdict, note)
            handled += 1

        if handled:
            logger.info(f"Resolver: handled {handled}/{len(due)} due predictions")
        return handled

    async def _safe_quote(self, ticker: str) -> Optional[float]:
        try:
            q = await self.providers.get_quote(ticker)
            return q.price
        except Exception as e:
            logger.warning(f"Resolver: quote failed for {ticker}: {e}")
            return None

    async def _resolve_one(
        self, pred: Prediction, quotes: dict[str, float]
    ) -> Optional[tuple[str, str]]:
        """Resolve and persist one prediction. Returns (verdict, note) on
        success, None when the row should stay pending for the next sweep.
        """
        if pred.is_structured:
            return await self._resolve_structured(pred, quotes)
        return await self._resolve_with_llm(pred)

    async def _resolve_structured(
        self, pred: Prediction, quotes: dict[str, float]
    ) -> Optional[tuple[str, str]]:
        price = quotes.get(pred.ticker or "")
        if price is None:
            return None  # quote fetch failed — retry next sweep

        threshold = pred.threshold or 0.0
        if pred.direction == "above":
            hit = price > threshold
        elif pred.direction == "below":
            hit = price < threshold
        else:
            return None
        verdict = VERDICT_RIGHT if hit else VERDICT_WRONG
        note = (
            f"{pred.ticker} at ${price:.2f}; threshold ${threshold:.2f} "
            f"({pred.direction})"
        )
        await self.store.resolve(pred.id, verdict=verdict, note=note)
        return verdict, note

    async def _resolve_with_llm(
        self, pred: Prediction
    ) -> Optional[tuple[str, str]]:
        if self.llm is None:
            return None
        try:
            if not self.llm.status().get("ready"):
                return None
        except Exception:
            return None

        user_content = (
            f"Prediction: \"{pred.claim}\"\n"
            f"Made by [{pred.user_label}]\n"
            f"Deadline (now passed): "
            f"{_iso(pred.deadline_utc)}\n\n"
            f"Judge it now."
        )
        try:
            msg = await self.llm.chat_messages(
                messages=[
                    {"role": "system", "content": _LLM_VERDICT_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                overrides={"max_tokens": 200, "temperature": 0.2},
                suppress_response_style=True,
                purpose="predict_resolve",
            )
        except Exception as e:
            logger.warning(f"Resolver: LLM judgement failed for #{pred.id}: {e}")
            return None

        verdict, note = _parse_verdict(msg.get("content") or "")
        if verdict is None:
            # Parsing failed — leave pending; we'll retry next sweep.
            logger.info(
                f"Resolver: couldn't parse verdict for #{pred.id} from "
                f"{msg.get('content')!r}"
            )
            return None
        await self.store.resolve(pred.id, verdict=verdict, note=note)
        return verdict, note

    async def _post_resolution(
        self, pred: Prediction, verdict: str, note: str
    ) -> None:
        record = await self.store.user_record(pred.user_hash, pred.context_key)
        record_str = ""
        if record["accuracy"] is not None and (record["right"] + record["wrong"]):
            judged = record["right"] + record["wrong"]
            record_str = (
                f"\n  {pred.user_label}'s record: {record['right']}/{judged} "
                f"({int(record['accuracy'] * 100)}%)"
            )
        emoji = VERDICT_EMOJI.get(verdict, "•")
        text = (
            f"⏰ Prediction #{pred.id} by {pred.user_label} is in:\n"
            f"  \"{pred.claim}\"\n\n"
            f"  {emoji} {verdict.upper()}{f' — {note}' if note else ''}"
            f"{record_str}"
        )
        try:
            if pred.group_id:
                await self.signal.send_message(
                    recipient="", message=text, group_id=pred.group_id,
                )
            else:
                # DM context — context_key is "dm:<hash>". We don't have
                # the phone here (only its hash), so DMs can't be
                # auto-posted. Log it and let !predictions show the
                # resolved status.
                logger.info(
                    f"Resolver: #{pred.id} resolved in DM context "
                    f"(no auto-post): {verdict} — {note}"
                )
        except Exception as e:
            logger.error(f"Resolver: post failed for #{pred.id}: {e}")


def _iso(ts: float) -> str:
    import datetime as dt
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )


def _parse_verdict(content: str) -> tuple[Optional[str], str]:
    """Pull the first JSON object from `content`, validate, return tuple.

    Returns (None, '') on any parse failure so the caller can keep the
    prediction pending instead of guessing.
    """
    if not content:
        return None, ""
    import re
    match = re.search(r"\{.*\}", content, re.DOTALL)
    if not match:
        return None, ""
    try:
        parsed = json.loads(match.group(0))
    except Exception:
        return None, ""
    if not isinstance(parsed, dict):
        return None, ""
    verdict = parsed.get("verdict")
    note = (parsed.get("note") or "").strip()
    if verdict in (VERDICT_RIGHT, VERDICT_WRONG, VERDICT_UNCLEAR):
        return verdict, note
    return None, ""
