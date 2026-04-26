"""
!iching — cast a hexagram and (optionally) get an LLM-narrated reading.

Modes:
  !iching                       three-coin cast (uniform-ish probabilities)
  !iching yarrow [question]     yarrow-stalk simulation (1/16, 5/16, 7/16, 3/16)
  !iching coins  [question]     three-coin (default; alias for the no-flag form)
  !iching daily                 hexagram of the day — cached per user × UTC date

The image is composed by ``iching_composer`` and attached as a base64 PNG.
A short reading is produced by the configured LLM when available; the
fallback is a static keyword-based read so the command still works with
the LLM disabled.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import hash_phone
from ..executor import run_blocking
from .base import BaseCommand, CommandContext, CommandResult
from .iching_composer import Cast, compose, fonts_ready
from .iching_data import Hexagram

logger = logging.getLogger(__name__)


# Cryptographically-strong RNG to match the woo aesthetic and to avoid any
# observable seed pattern in cast results.
_RNG = random.SystemRandom()


# ---------------------------------------------------------------------------
# Casting methods.
#
# Three-coin: each line, toss three coins. Tails=2, Heads=3 (or vice
# versa); sum gives a value 6..9. Distribution: 1/8, 3/8, 3/8, 1/8 →
# changing-line probability per line is 1/4 (so ~1.5 changing lines
# expected per cast).
#
# Yarrow stalks: the traditional method's distribution is 1/16, 7/16,
# 5/16, 3/16 over (6, 7, 8, 9). Changing-line probability per line is
# 1/16 + 3/16 = 1/4 — same expected count as coins, but old yang (9)
# is more common than old yin (6), and young yang (7) is more common
# than young yin (8). Subtle but meaningful for traditionalists.
# ---------------------------------------------------------------------------
def _coin_line() -> int:
    """One line by three-coin method. Heads=3, Tails=2; sum ∈ {6,7,8,9}."""
    return sum(_RNG.choice((2, 3)) for _ in range(3))


def _yarrow_line() -> int:
    """One line by yarrow-stalk distribution: 6=1/16, 7=5/16, 8=7/16, 9=3/16."""
    r = _RNG.randint(1, 16)
    if r <= 1:
        return 6
    if r <= 1 + 5:
        return 7
    if r <= 1 + 5 + 7:
        return 8
    return 9


def _cast_lines(method: str) -> tuple[int, int, int, int, int, int]:
    """Six lines, bottom-up (line 1 first)."""
    fn = _yarrow_line if method == "yarrow" else _coin_line
    return tuple(fn() for _ in range(6))  # type: ignore[return-value]


def _today_utc() -> str:
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


# ---------------------------------------------------------------------------
# Daily hexagram store. Mirrors the tarot daily cache (per user × UTC date,
# idempotent INSERT OR IGNORE for race safety). We store the cast's six
# line values verbatim so the rendered image is identical on every recall.
# ---------------------------------------------------------------------------
class IChingDB:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS iching_daily (
                        user_hash TEXT NOT NULL,
                        draw_date TEXT NOT NULL,
                        line_values TEXT NOT NULL,   -- "7,8,7,9,8,6" bottom-up
                        method TEXT NOT NULL,
                        drawn_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_hash, draw_date)
                    )
                    """
                )
                await db.commit()
            self._initialized = True

    async def get_or_cast(
        self, user_hash: str, method: str
    ) -> tuple[tuple[int, ...], str, bool]:
        """Return (line_values, method, was_already_drawn_today)."""
        await self._ensure_initialized()
        today = _today_utc()
        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT line_values, method FROM iching_daily "
                "WHERE user_hash = ? AND draw_date = ?",
                (user_hash, today),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                stored = tuple(int(x) for x in row[0].split(","))
                return stored, row[1], True

            line_values = _cast_lines(method)
            await db.execute(
                "INSERT OR IGNORE INTO iching_daily "
                "(user_hash, draw_date, line_values, method) VALUES (?, ?, ?, ?)",
                (user_hash, today, ",".join(str(v) for v in line_values), method),
            )
            await db.commit()

            # Re-read in case another writer beat us (IGNORE no-ops).
            async with db.execute(
                "SELECT line_values, method FROM iching_daily "
                "WHERE user_hash = ? AND draw_date = ?",
                (user_hash, today),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                stored = tuple(int(x) for x in row[0].split(","))
                return stored, row[1], False
            return line_values, method, False


# ---------------------------------------------------------------------------
# LLM glue. Mirrors the tarot prompt's brevity rules — Signal screens are
# small, and a wall-of-text I Ching reading is unreadable.
# ---------------------------------------------------------------------------
ICHING_SYSTEM_PROMPT = """\
You are reading the I Ching for a querent in a fast-moving group chat on
a phone screen. The image of the cast is already attached and shows the
hexagram(s); you do not need to redescribe what the picture looks like.

LENGTH IS A HARD RULE. ~120 words for a cast with no changing lines,
~180 words when there is a transformed hexagram. If you cannot say it in
that space, cut it.

  - No preamble, no "let's see what the I Ching says", no restating the
    cast list — the image already shows it. Open with the read itself.
  - Speak from the canonical tradition: Wilhelm/Baynes phrasing where
    natural; trigram pairings (Heaven over Earth = Standstill, etc.) are
    first language. Reach for line texts only for the changing lines,
    and only when they sharpen the read.
  - When there are changing lines: name the dynamic between primary and
    transformed hexagrams. The changing lines are the *moving* part —
    they are the active advice. Treat the transformed hexagram as the
    direction the situation is heading if the querent does (or fails to
    do) what the changing lines indicate.
  - Plainspoken, warm, specific. Avoid mystical filler ("the universe
    whispers"), avoid hedging. The book is direct; you should be too.
  - Never give medical, legal, or specific financial advice — the cast is
    a mirror for reflection, not a forecast.

If the querent asked a question, answer through the cast directly. If
they didn't, give a general read of the situation."""


# Generous to absorb thinking-model preamble; the prompt's hard length rule
# keeps the actual visible text short.
ICHING_MAX_TOKENS = {
    "no_changes": 800,
    "changes": 1400,
}


def _format_cast_for_llm(
    primary: Hexagram,
    line_values: tuple[int, ...],
    transformed: Optional[Hexagram],
    changing_indices: list[int],
    method: str,
) -> str:
    """Render the cast in a way the LLM can reason over."""
    out: list[str] = []
    out.append(f"Casting method: {method}")
    out.append("")
    out.append(
        f"Primary hexagram: #{primary.number} {primary.chinese} "
        f"({primary.pinyin}) — {primary.name}"
    )
    out.append(f"Trigrams: {primary.trigram_pair_label()}")
    out.append(f"Keywords: {primary.keywords}")
    out.append(f"Essence: {primary.meaning}")
    out.append("")
    # Lines bottom-up
    out.append("Lines (bottom-up):")
    for i, v in enumerate(line_values):
        kind = {6: "old yin (changing)", 7: "young yang", 8: "young yin", 9: "old yang (changing)"}[v]
        marker = "  *" if i in changing_indices else "   "
        out.append(f"  {marker} Line {i+1}: {v} — {kind}")
    if transformed is not None:
        out.append("")
        out.append(
            f"Transformed hexagram: #{transformed.number} {transformed.chinese} "
            f"({transformed.pinyin}) — {transformed.name}"
        )
        out.append(f"Trigrams: {transformed.trigram_pair_label()}")
        out.append(f"Keywords: {transformed.keywords}")
        out.append(f"Essence: {transformed.meaning}")
        out.append("")
        out.append(
            "Changing lines: "
            + ", ".join(str(i + 1) for i in changing_indices)
            + ". The advice lives in these lines and in the relationship "
            "between primary and transformed."
        )
    else:
        out.append("")
        out.append("No changing lines — the situation is stable in the primary hexagram.")
    return "\n".join(out)


def _static_reading(
    primary: Hexagram, transformed: Optional[Hexagram], indices: list[int]
) -> str:
    """Keyword-based fallback used when the LLM isn't available."""
    parts = [f"**{primary.pinyin} · {primary.name}**", primary.meaning]
    if transformed is not None:
        parts.append("")
        parts.append(
            f"Transforming through line(s) {', '.join(str(i+1) for i in indices)} → "
            f"**{transformed.pinyin} · {transformed.name}** — {transformed.meaning}"
        )
    return "\n".join(parts)


METHOD_ALIAS_TO_KIND: dict[str, str] = {
    "yarrow": "yarrow",
    "stalks": "yarrow",
    "stalk": "yarrow",
    "y": "yarrow",
    "coins": "coins",
    "coin": "coins",
    "c": "coins",
    "3coin": "coins",
}


class IChingCommand(BaseCommand):
    name = "iching"
    aliases = ["ic", "yi", "yijing"]
    description = "Cast a hexagram from the I Ching."
    usage = "!iching [yarrow|coins|daily] [question]"
    help_explanation = (
        "Casts a hexagram and sends an image of the result. Modes: "
        "`!iching` for a three-coin cast, `!iching yarrow your question` "
        "for the traditional yarrow-stalk distribution, `!iching daily` "
        "for a hexagram-of-the-day cached per UTC day. Changing lines "
        "produce a transformed hexagram alongside the primary."
    )

    def __init__(self, db_path: str, llm_client=None):
        self.db = IChingDB(db_path)
        self.llm = llm_client

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not fonts_ready():
            return CommandResult.error(
                "I Ching renderer can't find a CJK font on this host — "
                "the rebuild that installs `fonts-noto-cjk` may still be in "
                "flight. Try again in a moment."
            )

        method, daily, question = self._parse_args(ctx.args)
        already_drawn = False

        if daily:
            user_hash = hash_phone(ctx.sender)
            try:
                line_values_t, method, already_drawn = await self.db.get_or_cast(
                    user_hash, method
                )
                line_values = tuple(line_values_t)  # type: ignore[arg-type]
            except Exception as e:
                logger.exception(f"iching daily: db error: {e}")
                return CommandResult.error("Couldn't reach the daily-cast store.")
        else:
            line_values = _cast_lines(method)

        cast = Cast(line_values=line_values, question=question)  # type: ignore[arg-type]
        primary = cast.primary()
        transformed = cast.transformed()
        changing = cast.changing_indices()

        # Compose the image off the event loop — PIL is blocking.
        try:
            image_b64 = await run_blocking(compose, cast, timeout=15.0)
        except Exception as e:
            logger.exception(f"iching: image compose failed: {e}")
            return CommandResult.error("Couldn't render the hexagram.")

        reading_text = await self._build_reading(
            primary, transformed, changing, line_values, method, question
        )

        header = self._header_for(daily=daily, already_drawn=already_drawn)
        body = f"{header}\n\n{reading_text}"

        return CommandResult(
            text=body,
            success=True,
            attachments=[image_b64],
            styled=True,
        )

    @staticmethod
    def _parse_args(args: list[str]) -> tuple[str, bool, Optional[str]]:
        """Return (method, daily, question)."""
        if not args:
            return "coins", False, None
        first = args[0].lower()
        rest = args[1:]
        if first == "daily":
            # Optional method as second token (e.g. !iching daily yarrow)
            if rest and rest[0].lower() in METHOD_ALIAS_TO_KIND:
                return METHOD_ALIAS_TO_KIND[rest[0].lower()], True, " ".join(rest[1:]).strip() or None
            return "coins", True, " ".join(rest).strip() or None
        if first in METHOD_ALIAS_TO_KIND:
            return METHOD_ALIAS_TO_KIND[first], False, " ".join(rest).strip() or None
        # First token is the start of the question on a default-coins cast
        return "coins", False, " ".join(args).strip() or None

    @staticmethod
    def _header_for(*, daily: bool, already_drawn: bool) -> str:
        if daily and already_drawn:
            return "☯ Your hexagram of the day (cast earlier — same one all day):"
        if daily:
            return "☯ Your hexagram of the day:"
        return "☯ Your cast:"

    async def _build_reading(
        self,
        primary: Hexagram,
        transformed: Optional[Hexagram],
        changing_indices: list[int],
        line_values: tuple[int, ...],
        method: str,
        question: Optional[str],
    ) -> str:
        if self.llm is None:
            return _static_reading(primary, transformed, changing_indices)
        try:
            status = self.llm.status()
        except Exception:
            return _static_reading(primary, transformed, changing_indices)
        if not status.get("ready"):
            return _static_reading(primary, transformed, changing_indices)

        prompt_parts: list[str] = []
        if question:
            prompt_parts.append(f"Querent's question: {question}")
        else:
            prompt_parts.append("No specific question — give a general read.")
        prompt_parts.append("")
        prompt_parts.append(_format_cast_for_llm(
            primary, line_values, transformed, changing_indices, method,
        ))

        messages = [
            {"role": "system", "content": ICHING_SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(prompt_parts)},
        ]
        budget_key = "changes" if transformed is not None else "no_changes"
        overrides = {"max_tokens": ICHING_MAX_TOKENS[budget_key]}

        try:
            msg = await self.llm.chat_messages(
                messages,
                overrides=overrides,
                suppress_response_style=True,
                purpose="iching",
            )
            text = (msg.get("content") or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"iching: LLM reading failed, using fallback: {e}")
        return _static_reading(primary, transformed, changing_indices)
