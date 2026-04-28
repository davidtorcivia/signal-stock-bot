"""
Prediction commands.

  !predict <claim> by <when>     create
  !predictions [@user]           list open
  !resolve <id> right|wrong|unclear [reason]   manually mark
  !leaderboard                   group accuracy ranking

Parser strategy:
  1. Try a regex extraction for stock-shape predictions
     (`<TICKER> (above|below|over|under|>|<) $<num> by <date>`).
     Cheap, deterministic, covers the most common case.
  2. Fall back to LLM-based extraction for free-form claims. The LLM
     returns JSON with claim text, deadline, and (when applicable) a
     ticker / threshold / direction tuple for auto-resolution.
  3. If both fail, store the prediction unstructured but only if a
     deadline was parseable from the text — otherwise reject so the
     user has to add one.
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import re
from typing import Optional

import dateparser

from ..database import hash_phone
from ..predictions import (
    PredictionStore,
    STATUS_PENDING,
    VALID_VERDICTS,
    VERDICT_EMOJI,
)
from .base import BaseCommand, CommandContext, CommandResult

logger = logging.getLogger(__name__)


# Stock-shape predictions — cover the common form deterministically before
# we pay for an LLM call. Examples that match:
#   AAPL above $200 by June 1
#   $TSLA below 250 by 2026-08-01
#   NVDA > $500 by EOY
_STOCK_RE = re.compile(
    r"""
    \$?(?P<ticker>[A-Z]{1,5}(?:[-.][A-Z]{1,5})?)
    \s+(?P<dir>above|below|over|under|>=?|<=?)
    \s*\$?\s*(?P<thresh>\d+(?:\.\d+)?)
    \s+by\s+(?P<when>.+)
    $
    """,
    re.IGNORECASE | re.VERBOSE,
)

_DIRECTION_MAP = {
    "above": "above", "over": "above", ">": "above", ">=": "above",
    "below": "below", "under": "below", "<": "below", "<=": "below",
}


def _utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


_DATEPARSER_SETTINGS = {
    "PREFER_DATES_FROM": "future",   # "by June 1" in May → this June; in July → next June
    "RETURN_AS_TIMEZONE_AWARE": True,
    "TIMEZONE": "UTC",
    "TO_TIMEZONE": "UTC",
}

_WEEKDAYS = {
    "monday": 0, "mon": 0, "tuesday": 1, "tue": 1, "tues": 1,
    "wednesday": 2, "wed": 2, "thursday": 3, "thu": 3, "thurs": 3,
    "friday": 4, "fri": 4, "saturday": 5, "sat": 5, "sunday": 6, "sun": 6,
}

# "next Friday", "this Monday", "Friday" — dateparser handles weekday names
# unevenly across versions. We resolve them ourselves to the next future
# occurrence.
_WEEKDAY_RE = re.compile(
    r"^(?:(?P<qualifier>next|this|coming)\s+)?(?P<day>"
    + "|".join(sorted(_WEEKDAYS.keys(), key=len, reverse=True))
    + r")$",
    re.IGNORECASE,
)


def _resolve_weekday(text: str) -> Optional[dt.datetime]:
    m = _WEEKDAY_RE.match(text.strip())
    if not m:
        return None
    target_dow = _WEEKDAYS[m.group("day").lower()]
    now = _utc_now()
    days_ahead = (target_dow - now.weekday()) % 7
    if days_ahead == 0:
        days_ahead = 7  # "Friday" said on Friday means next Friday
    if (m.group("qualifier") or "").lower() == "next" and days_ahead < 7:
        # "next Friday" said on Sunday traditionally means the Friday of
        # the *following* week — bump if we'd otherwise resolve to this week.
        days_ahead += 7
    target = now + dt.timedelta(days=days_ahead)
    return target.replace(hour=21, minute=0, second=0, microsecond=0)


def _parse_deadline(text: str) -> Optional[float]:
    """Parse a deadline expression into a UTC unix timestamp.

    Uses `dateparser` which natively handles relative phrases ("tomorrow",
    "in 2 weeks", "next Friday") as well as ISO/named dates. EOY/EOM/EOW
    are handled explicitly because dateparser doesn't recognize them.
    Defaults missing time-of-day to 21:00 UTC ≈ post-NY-market-close so
    same-day "by today" checks fire after the close.
    """
    text = text.strip()
    if not text:
        return None

    now = _utc_now()
    upper = text.upper().replace(" ", "")
    if upper in ("EOY", "ENDOFYEAR", "ENDOFTHEYEAR"):
        return dt.datetime(now.year, 12, 31, 21, 0, tzinfo=dt.timezone.utc).timestamp()
    if upper in ("EOM", "ENDOFMONTH"):
        next_month = now.replace(day=28) + dt.timedelta(days=4)
        last = next_month - dt.timedelta(days=next_month.day)
        return last.replace(hour=21, minute=0, second=0, microsecond=0).timestamp()
    if upper in ("EOW", "ENDOFWEEK"):
        days_to_friday = (4 - now.weekday()) % 7 or 7
        target = now + dt.timedelta(days=days_to_friday)
        return target.replace(hour=21, minute=0, second=0, microsecond=0).timestamp()

    # Weekday phrases — dateparser is unreliable on these
    weekday_match = _resolve_weekday(text)
    if weekday_match is not None:
        return weekday_match.timestamp()

    try:
        parsed = dateparser.parse(text, settings=_DATEPARSER_SETTINGS)
    except Exception:
        return None
    if parsed is None or parsed <= now:
        return None
    # Ensure UTC even if dateparser elided tzinfo somehow
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=dt.timezone.utc)
    # If user gave a date with no explicit time, default to 21:00 UTC
    if parsed.hour == 0 and parsed.minute == 0 and parsed.second == 0:
        parsed = parsed.replace(hour=21)
    return parsed.timestamp()


def _parse_stock_shape(text: str) -> Optional[dict]:
    m = _STOCK_RE.match(text.strip())
    if not m:
        return None
    when_ts = _parse_deadline(m.group("when"))
    if when_ts is None:
        return None
    direction = _DIRECTION_MAP.get(m.group("dir").lower())
    if direction is None:
        return None
    return {
        "claim": text.strip(),
        "deadline_utc": when_ts,
        "ticker": m.group("ticker").upper(),
        "threshold": float(m.group("thresh")),
        "direction": direction,
    }


_DEADLINE_TRIGGERS = (
    "by ", " on ", " in ", " before ", "until ", "until: ",
)
_HAS_DATE_SHAPE = re.compile(
    r"""
    \b(\d{4}-\d{1,2}-\d{1,2}      # ISO 2026-06-01
    |\d{1,2}/\d{1,2}              # 6/1 or 6/1/2026
    |jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec  # named month prefix
    |today|tomorrow|tonight|next |this |coming        # relative
    |EOY|EOM|EOW                  # shortcut tokens
    |\d+\s*(?:day|week|month|year)s?\b)               # "in 2 weeks"
    """,
    re.IGNORECASE | re.VERBOSE,
)


def _has_any_deadline(text: str) -> bool:
    """Cheap pre-flight: does the text plausibly contain a deadline?

    Used to skip the LLM extraction round-trip when the user obviously
    forgot to specify one. Returns True on a soft signal — the actual
    parse may still fail, in which case we fall through to the LLM.
    """
    low = text.lower()
    if any(t in low for t in _DEADLINE_TRIGGERS):
        return True
    if _HAS_DATE_SHAPE.search(text):
        return True
    return False


_LLM_EXTRACT_PROMPT = """\
Extract a prediction's structured fields from the user's claim. Reply with \
ONLY a JSON object. No preamble, no explanation.

Fields:
  - claim: the predicted statement (string), with the deadline phrase removed
  - deadline: ISO 8601 UTC timestamp of the deadline (string), or null if \
the user gave none
  - ticker: stock ticker if the claim is about a stock price, else null
  - threshold: numeric price threshold if the claim is about a stock, else null
  - direction: "above" or "below" if the claim is about a stock, else null

Rules:
  - If you can't pin down a deadline, return deadline: null. Don't invent one.
  - Don't ticker-extract unless the claim is genuinely about a stock price.
    "Apple has a great quarter" is not a price claim — leave ticker null.
  - Always return well-formed JSON."""


async def _llm_extract(llm_client, raw: str) -> Optional[dict]:
    """LLM-based fallback parser. Returns None on any failure."""
    if llm_client is None:
        return None
    try:
        if not llm_client.status().get("ready"):
            return None
    except Exception:
        return None

    try:
        msg = await llm_client.chat_messages(
            messages=[
                {"role": "system", "content": _LLM_EXTRACT_PROMPT},
                {"role": "user", "content": raw},
            ],
            overrides={"max_tokens": 300, "temperature": 0.2},
            suppress_response_style=True,
            purpose="predict_extract",
        )
    except Exception as e:
        logger.debug(f"predict: LLM extract failed: {e}")
        return None

    content = (msg.get("content") or "").strip()
    # Strip code fences the LLM might wrap around the JSON
    fence = re.search(r"\{.*\}", content, re.DOTALL)
    if not fence:
        return None
    try:
        parsed = json.loads(fence.group(0))
    except Exception:
        return None
    if not isinstance(parsed, dict):
        return None

    deadline_str = parsed.get("deadline")
    deadline_utc = None
    if isinstance(deadline_str, str) and deadline_str:
        deadline_utc = _parse_deadline(deadline_str)
    if deadline_utc is None:
        return None  # without a deadline we can't auto-resolve; reject

    out = {
        "claim": (parsed.get("claim") or raw).strip(),
        "deadline_utc": deadline_utc,
        "ticker": None,
        "threshold": None,
        "direction": None,
    }
    ticker = parsed.get("ticker")
    threshold = parsed.get("threshold")
    direction = parsed.get("direction")
    if (
        isinstance(ticker, str) and ticker
        and isinstance(threshold, (int, float))
        and isinstance(direction, str)
        and direction.lower() in ("above", "below")
    ):
        out["ticker"] = ticker.upper()
        out["threshold"] = float(threshold)
        out["direction"] = direction.lower()
    return out


def _format_remaining(deadline_utc: float) -> str:
    delta = deadline_utc - _utc_now().timestamp()
    if delta <= 0:
        return "due"
    days = delta / 86400
    if days >= 1:
        return f"{int(round(days))}d"
    hours = delta / 3600
    if hours >= 1:
        return f"{int(round(hours))}h"
    return "<1h"


def _format_deadline(deadline_utc: float) -> str:
    return dt.datetime.fromtimestamp(deadline_utc, tz=dt.timezone.utc).strftime("%Y-%m-%d")


async def extract_prediction(text: str, llm_client=None) -> tuple[Optional[dict], Optional[str]]:
    """Public wrapper around the predict-parser pipeline.

    Returns `(parsed, error)` where exactly one is non-None: the parsed
    prediction dict (claim, deadline_utc, optional ticker/threshold/direction)
    on success, or a human-readable error string on failure. Used by both
    `PredictCommand.execute` (via inline calls) and the LLM-facing
    `predict_self` tool so the two paths stay in sync.
    """
    raw = (text or "").strip()
    if not raw:
        return None, (
            "Tell me what you're predicting and by when. "
            "Example: AAPL above $200 by June 1"
        )
    parsed = _parse_stock_shape(raw)
    if parsed is None and not _has_any_deadline(raw):
        return None, (
            "I couldn't find a deadline in that. Add `by <date>` "
            "(e.g. `by June 1`, `by 2026-09-01`, `by EOY`)."
        )
    if parsed is None:
        parsed = await _llm_extract(llm_client, raw)
    if parsed is None:
        return None, (
            "I couldn't pin down what you're predicting or when. Try "
            "`<claim> by <date>`."
        )
    return parsed, None


# LLM tool schema — exposed to the writer model in groups when a
# PredictionStore is wired and the per-context policy allows the !predict
# command. Intentionally distinct from `bot__predict` (which logs a
# prediction on behalf of the human asker) — this one authors as the bot
# itself, so Sigil's own forecasts land on the leaderboard separately.
PREDICT_SELF_TOOL = {
    "type": "function",
    "function": {
        "name": "predict_self",
        "description": (
            "Log YOUR OWN (the bot's) prediction. Use this when you want "
            "to put YOUR forecast on record — \"I think AAPL hits $250 "
            "by Friday\" — distinct from `bot__predict` which logs a "
            "prediction for the human who asked. Your predictions land "
            "on the leaderboard under your bot name, alongside humans, "
            "and auto-resolve at the deadline.\n\n"
            "Format the claim like a user would: \"<claim> by <date>\". "
            "Stock-shape claims auto-resolve via live price "
            "(\"AAPL above $200 by June 1\"); free-form claims get "
            "judged by the LLM at the deadline. Always include a "
            "concrete deadline — vague ones get rejected.\n\n"
            "When to use: you've made a substantive forecast in chat "
            "and want to commit to it, or a human dared you to put "
            "your money where your mouth is. Skip for hedged language "
            "(\"probably\", \"might\") — only stake actual claims."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "claim": {
                    "type": "string",
                    "description": (
                        "Full prediction text including the deadline, "
                        "exactly as you'd write it after !predict. "
                        "Examples: \"AAPL above $250 by next Friday\", "
                        "\"the Fed cuts 25bp at the next meeting by "
                        "2026-06-12\", \"Bitcoin under $80k by EOM\"."
                    ),
                },
            },
            "required": ["claim"],
        },
    },
}


class PredictCommand(BaseCommand):
    name = "predict"
    aliases = ["bet", "forecast"]
    description = "Log a prediction with a deadline; bot follows up at the date."
    usage = "!predict <claim> by <when>"
    help_explanation = (
        "Logs a dated prediction. Examples: `!predict AAPL above $200 by "
        "June 1`, `!predict Trump wins the election by 2026-11-04`. "
        "At the deadline I'll check or ask the group, then update your "
        "leaderboard score."
    )

    def __init__(self, store: PredictionStore, llm_client=None, name_registry=None):
        self.store = store
        self.llm = llm_client
        self.name_registry = name_registry

    def _label(self, phone: str) -> str:
        if self.name_registry is not None:
            return self.name_registry.label_for(phone)
        return f"...{(phone or '')[-4:]}"

    async def execute(self, ctx: CommandContext) -> CommandResult:
        raw = " ".join(ctx.args).strip()
        parsed, err = await extract_prediction(raw, llm_client=self.llm)
        if err is not None:
            return CommandResult.error(err)
        assert parsed is not None  # extract_prediction guarantees this when err is None

        try:
            pred_id = await self.store.create(
                user_hash=hash_phone(ctx.sender),
                user_label=self._label(ctx.sender),
                group_id=ctx.group_id,
                context_key=ctx.context_key(),
                claim=parsed["claim"],
                deadline_utc=parsed["deadline_utc"],
                ticker=parsed.get("ticker"),
                threshold=parsed.get("threshold"),
                direction=parsed.get("direction"),
            )
        except Exception as e:
            logger.exception(f"predict: store.create failed: {e}")
            return CommandResult.error("Couldn't save the prediction.")

        pretty_deadline = _format_deadline(parsed["deadline_utc"])
        kind_note = ""
        if parsed.get("ticker"):
            kind_note = (
                f"\n  Auto-resolves: {parsed['ticker']} {parsed['direction']} "
                f"${parsed['threshold']:g}"
            )
        return CommandResult.ok(
            f"✓ #{pred_id} logged: \"{parsed['claim']}\"\n"
            f"  Due {pretty_deadline}{kind_note}\n"
            f"  Anyone can `!resolve {pred_id} right|wrong|unclear` later."
        )


class PredictionsCommand(BaseCommand):
    name = "predictions"
    aliases = ["preds", "mybets"]
    description = "List open predictions for yourself or another user."
    usage = "!predictions [@user]"
    help_explanation = (
        "Shows pending predictions in this chat. With no argument, shows "
        "your own; with `@user` or a name, shows theirs. Resolved ones "
        "are not listed — see !leaderboard for tallies."
    )

    def __init__(self, store: PredictionStore, name_registry=None):
        self.store = store
        self.name_registry = name_registry

    async def execute(self, ctx: CommandContext) -> CommandResult:
        # For now: only the caller's own list. Cross-user lookup needs a
        # name→hash map, which the registry doesn't expose; punt on that.
        target_label = "Your"
        target_hash = hash_phone(ctx.sender)

        rows = await self.store.list_for_user(
            target_hash,
            context_key=ctx.context_key(),
            only_pending=True,
            limit=20,
        )
        if not rows:
            return CommandResult.ok(
                f"{target_label} open predictions: (none)\n"
                f"  Make one with `!predict <claim> by <date>`."
            )

        lines = [f"{target_label} open predictions:"]
        for p in rows:
            remaining = _format_remaining(p.deadline_utc)
            deadline_str = _format_deadline(p.deadline_utc)
            lines.append(
                f"  #{p.id} ({remaining}, due {deadline_str}) — {p.claim}"
            )
        return CommandResult.ok("\n".join(lines))


class ResolveCommand(BaseCommand):
    name = "resolve"
    aliases = ["verdict"]
    description = "Mark a prediction as right, wrong, or unclear."
    usage = "!resolve <id> right|wrong|unclear [reason]"
    help_explanation = (
        "Manually resolves a prediction. Anyone in the chat can resolve "
        "any prediction — group accountability. Use `unclear` when the "
        "outcome is genuinely ambiguous; it doesn't count against the "
        "predictor's record."
    )

    def __init__(self, store: PredictionStore, name_registry=None):
        self.store = store
        self.name_registry = name_registry

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if len(ctx.args) < 2:
            return CommandResult.error(
                "Usage: `!resolve <id> right|wrong|unclear [reason]`"
            )
        try:
            pred_id = int(ctx.args[0])
        except ValueError:
            return CommandResult.error(
                f"`{ctx.args[0]}` isn't a prediction id. Try `!predictions` to find one."
            )
        verdict = ctx.args[1].lower()
        if verdict not in VALID_VERDICTS:
            return CommandResult.error(
                f"Verdict must be one of: {', '.join(sorted(VALID_VERDICTS))}"
            )
        note = " ".join(ctx.args[2:]).strip() or None

        pred = await self.store.get(pred_id)
        if pred is None:
            return CommandResult.error(f"No prediction #{pred_id}.")
        if pred.context_key != ctx.context_key():
            return CommandResult.error(
                f"#{pred_id} belongs to a different chat — resolve it there."
            )
        if pred.status != STATUS_PENDING:
            return CommandResult.error(
                f"#{pred_id} is already {pred.status} ({pred.verdict or '—'})."
            )

        ok = await self.store.resolve(
            pred_id,
            verdict=verdict,
            note=note,
            resolver_user_hash=hash_phone(ctx.sender),
        )
        if not ok:
            return CommandResult.error(f"Couldn't resolve #{pred_id}.")

        record = await self.store.user_record(pred.user_hash, pred.context_key)
        record_str = ""
        if record["accuracy"] is not None:
            record_str = (
                f"\n  {pred.user_label}'s record: {record['right']}/"
                f"{record['right'] + record['wrong']} "
                f"({int(record['accuracy'] * 100)}%)"
            )
        emoji = VERDICT_EMOJI.get(verdict, "•")
        return CommandResult.ok(
            f"{emoji} #{pred_id} marked **{verdict}**"
            f"{f' — {note}' if note else ''}\n"
            f"  Claim: {pred.claim}{record_str}"
        )


class LeaderboardCommand(BaseCommand):
    name = "leaderboard"
    aliases = ["lb", "scores"]
    description = "Prediction accuracy ranking for this chat."
    usage = "!leaderboard"
    help_explanation = (
        "Ranks chat members by prediction accuracy. `unclear` and "
        "still-pending predictions don't affect the percentage."
    )

    def __init__(self, store: PredictionStore, name_registry=None):
        self.store = store
        self.name_registry = name_registry

    def _live_label(self, user_hash: str, fallback: str) -> str:
        """Prefer the registry's current name over the stored label.

        Stored labels freeze at prediction-creation time; if a user
        registers a name (or changes it) afterwards, the leaderboard
        should reflect that without having to resolve every old row.
        """
        if self.name_registry is not None:
            cached = self.name_registry._cache.get(user_hash)
            if cached:
                return cached
        return fallback or "(unknown)"

    async def execute(self, ctx: CommandContext) -> CommandResult:
        rows = await self.store.leaderboard(ctx.context_key(), limit=20)
        scoring = [r for r in rows if (r["right"] + r["wrong"]) > 0]
        if not scoring and not any(r["pending"] for r in rows):
            return CommandResult.ok(
                "No predictions yet. Try `!predict AAPL above $200 by June 1`."
            )

        lines = ["⚖ Prediction leaderboard"]
        for i, r in enumerate(scoring, 1):
            judged = r["right"] + r["wrong"]
            pct = int(r["accuracy"] * 100)
            label = self._live_label(r["user_hash"], r["label"])
            lines.append(
                f"  {i}. {label:<14} {r['right']}/{judged} ({pct}%)"
            )
        pending_total = sum(r["pending"] for r in rows)
        if pending_total:
            lines.append(f"\n  {pending_total} still pending.")
        return CommandResult.ok("\n".join(lines))
