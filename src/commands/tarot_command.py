"""
!tarot — draw cards from a Rider-Waite-Smith deck and (optionally) get an
LLM-narrated reading.

Modes:
  !tarot                       single random card
  !tarot 3 [question]          past / present / future
  !tarot celtic [question]     10-card Celtic Cross
  !tarot daily                 card of the day — cached per user × UTC date

The image is composed by ``tarot_composer`` and attached as a base64 PNG.
Reading text is produced by the configured LLM when available; otherwise
a static keyword-based fallback is used so the command is still useful
when the LLM is disabled.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import random
import threading
from pathlib import Path
from typing import Optional

import aiosqlite

from ..database import hash_phone
from ..executor import run_blocking
from .base import BaseCommand, CommandContext, CommandResult
from .tarot_composer import ASSETS_DIR, Draw, compose
from .tarot_data import DECK, DECK_BY_SLUG, Card

logger = logging.getLogger(__name__)


# Cryptographically-strong RNG for the woo aesthetic — avoids any chance
# of users observing patterns from a seeded PRNG.
_RNG = random.SystemRandom()

THREE_CARD_POSITIONS = ["Past", "Present", "Future"]
CELTIC_POSITIONS = [
    "Present",
    "Challenge",
    "Foundation",
    "Recent Past",
    "Crown",
    "Near Future",
    "Self",
    "Environment",
    "Hopes & Fears",
    "Outcome",
]


def _draw_n(n: int) -> list[Draw]:
    """Draw n distinct cards from a fresh shuffle, each independently reversed."""
    cards = _RNG.sample(DECK, n)
    return [Draw(card=c, reversed=_RNG.random() < 0.5) for c in cards]


def _today_utc() -> str:
    """ISO date in UTC — stable globally regardless of where the user is."""
    return dt.datetime.now(dt.timezone.utc).date().isoformat()


def _format_cards_for_llm(draws: list[Draw]) -> str:
    """Render the drawn cards in a way the LLM can reason over."""
    lines = []
    for i, d in enumerate(draws, 1):
        pos = f"Position {i} ({d.position})" if d.position else f"Card {i}"
        orient = "reversed" if d.reversed else "upright"
        kw = d.card.keywords_rev if d.reversed else d.card.keywords_up
        lines.append(
            f"{pos}: {d.card.name} — {orient}\n"
            f"  Essence: {d.card.meaning}\n"
            f"  Keywords ({orient}): {kw}"
        )
    return "\n\n".join(lines)


def _static_reading(draws: list[Draw], spread: str) -> str:
    """Keyword-based fallback when the LLM isn't available.

    The caller adds its own header — return only the per-card body.
    """
    pieces = []
    for d in draws:
        kw = d.card.keywords_rev if d.reversed else d.card.keywords_up
        orient = " (reversed)" if d.reversed else ""
        pos = f"  ◇ {d.position} — " if d.position else "  ◇ "
        pieces.append(f"{pos}{d.card.name}{orient}\n     {kw}\n     {d.card.meaning}")
    return "\n\n".join(pieces)


TAROT_SYSTEM_PROMPT = """\
You are a tarot reader delivering readings inside a group chat on a phone.
The chat is fast-moving and people are reading on small screens — your
single most important constraint is BREVITY. A reading that doesn't fit
on one screen has failed.

  - LENGTH IS A HARD RULE. Single card: 2–3 sentences (~40 words).
    Three-card: 4–6 sentences (~90 words). Celtic Cross: ~150 words MAX,
    delivered as 3–4 short paragraphs that group cards thematically rather
    than walking position by position. If you can't say it in that space,
    cut it.
  - No preamble, no "let's see what the cards say", no restating the card
    list — the image already shows it. Open with the read itself.
  - Treat each card's position and orientation as meaningful — a reversed
    card softens, blocks, or inverts the upright meaning depending on context.
  - Weave cards together as a single arc, not a list of definitions. For
    Celtic Cross especially: name the arc (what's the story across the ten
    cards?), then highlight only the 2–3 cards that carry the most weight.
  - Speak plainly and warmly. Avoid mystical filler ("the universe
    whispers"), avoid hedging. Poetic but specific.
  - Never make medical, legal, or financial claims; the cards are a mirror
    for reflection, not a forecast.

If the querent asked a question, answer through the cards directly. If they
didn't, give a general read."""


# Per-spread token budgets. Generous enough to absorb thinking/reasoning
# preamble from models that emit it before the final answer (deepseek-r1
# style), while the prompt's hard length rule keeps the actual visible
# reading short enough for a group chat.
TAROT_MAX_TOKENS = {
    "single": 600,
    "three": 900,
    "celtic": 1600,
}


class TarotDB:
    """SQLite-backed daily-card cache.

    One row per (user_hash, draw_date_utc). Reads return None when no draw
    exists for that user/day; writes are idempotent (INSERT OR IGNORE) so
    a race between two simultaneous !tarot daily calls doesn't double-draw.
    """

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
                await db.execute("""
                    CREATE TABLE IF NOT EXISTS tarot_daily (
                        user_hash TEXT NOT NULL,
                        draw_date TEXT NOT NULL,
                        card_slug TEXT NOT NULL,
                        reversed INTEGER NOT NULL,
                        drawn_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        PRIMARY KEY (user_hash, draw_date)
                    )
                """)
                await db.commit()
            self._initialized = True

    async def get_or_draw(self, user_hash: str) -> tuple[Card, bool, bool]:
        """Return (card, reversed, was_already_drawn_today).

        Idempotent: if the user has a row for today's UTC date, returns it.
        Otherwise draws once and stores the result. INSERT OR IGNORE handles
        the rare race where two queries land simultaneously.
        """
        await self._ensure_initialized()
        today = _today_utc()

        async with aiosqlite.connect(self.db_path) as db:
            async with db.execute(
                "SELECT card_slug, reversed FROM tarot_daily "
                "WHERE user_hash = ? AND draw_date = ?",
                (user_hash, today),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                card = DECK_BY_SLUG.get(row[0])
                if card is not None:
                    return card, bool(row[1]), True
                # Fall through if the slug got renamed under us — re-draw.

            card = _RNG.choice(DECK)
            reversed_ = _RNG.random() < 0.5
            await db.execute(
                "INSERT OR IGNORE INTO tarot_daily "
                "(user_hash, draw_date, card_slug, reversed) VALUES (?, ?, ?, ?)",
                (user_hash, today, card.slug, int(reversed_)),
            )
            await db.commit()

            # If another writer beat us, the IGNORE no-op'd. Re-read to
            # return the canonical row.
            async with db.execute(
                "SELECT card_slug, reversed FROM tarot_daily "
                "WHERE user_hash = ? AND draw_date = ?",
                (user_hash, today),
            ) as cur:
                row = await cur.fetchone()
            if row is not None:
                final = DECK_BY_SLUG.get(row[0]) or card
                return final, bool(row[1]), False
            # Shouldn't happen — fall back to what we tried to insert
            return card, reversed_, False


def deck_is_ready() -> bool:
    """True iff every card in DECK has a non-empty file at ASSETS_DIR."""
    if not ASSETS_DIR.is_dir():
        return False
    for card in DECK:
        p = ASSETS_DIR / f"{card.slug}.jpg"
        if not p.is_file() or p.stat().st_size < 1024:
            return False
    return True


_PREP_LOCK = threading.Lock()
_PREP_THREAD: Optional[threading.Thread] = None


def ensure_deck_ready_async() -> None:
    """Kick off a one-shot background download if the deck is incomplete.

    Idempotent: subsequent calls do nothing while the thread runs and exit
    immediately once the deck is fully populated. Designed to be safe to
    call from startup hooks without blocking event-loop-bound code.

    The download itself is run via the existing scripts/download_tarot.py
    so we don't duplicate retry/backoff/resize logic here.
    """
    global _PREP_THREAD

    if deck_is_ready():
        return

    with _PREP_LOCK:
        if _PREP_THREAD is not None and _PREP_THREAD.is_alive():
            return
        if deck_is_ready():
            return

        def _run():
            try:
                # Imported lazily so module import never depends on the
                # download script being present (e.g. in tests).
                import importlib.util
                script_path = (
                    Path(__file__).resolve().parent.parent.parent
                    / "scripts" / "download_tarot.py"
                )
                if not script_path.is_file():
                    logger.warning(f"tarot deck prep: script missing at {script_path}")
                    return
                spec = importlib.util.spec_from_file_location(
                    "download_tarot", script_path
                )
                if spec is None or spec.loader is None:
                    logger.warning("tarot deck prep: could not load download script")
                    return
                module = importlib.util.module_from_spec(spec)
                logger.info(
                    f"tarot deck prep: downloading missing cards to {ASSETS_DIR}"
                )
                spec.loader.exec_module(module)
                rc = module.main()
                if rc == 0:
                    logger.info("tarot deck prep: deck ready")
                else:
                    logger.warning(
                        "tarot deck prep: finished with errors — "
                        "some cards may be missing"
                    )
            except Exception as e:
                logger.exception(f"tarot deck prep: unexpected error: {e}")

        _PREP_THREAD = threading.Thread(
            target=_run, name="tarot-deck-prep", daemon=True
        )
        _PREP_THREAD.start()


# Word -> canonical spread name. The canonical names ("daily", "three",
# "celtic", "single") are also accepted directly. Lookup is a single dict
# get; previously had four parallel sets and four sequential `if x in ...`
# checks.
SPREAD_ALIAS_TO_KIND: dict[str, str] = {
    **{w: "daily" for w in ("daily", "today", "dotd", "day")},
    **{w: "three" for w in ("3", "three", "ppf")},
    **{w: "celtic" for w in ("celtic", "cross", "celtic-cross", "celticcross", "10")},
    **{w: "single" for w in ("1", "one", "single", "draw")},
}


class TarotCommand(BaseCommand):
    name = "tarot"
    aliases = ["cards", "card"]
    description = "Draw tarot cards (single, 3-card, Celtic Cross, or card of the day)."
    usage = "!tarot [3|celtic|daily] [question]"
    help_explanation = (
        "Draws from a Rider-Waite-Smith deck and sends an image of the spread. "
        "Modes: `!tarot` for one card, `!tarot 3 your question` for past/present/"
        "future, `!tarot celtic your question` for a 10-card Celtic Cross, "
        "`!tarot daily` for a card-of-the-day that's stable for the rest of "
        "the UTC day."
    )

    def __init__(self, db_path: str, llm_client=None):
        self.db = TarotDB(db_path)
        self.llm = llm_client

    async def execute(self, ctx: CommandContext) -> CommandResult:
        if not deck_is_ready():
            ensure_deck_ready_async()
            return CommandResult.error(
                "The deck is still being prepared (one-time download from "
                "Wikimedia, ~90s). Try again in a moment."
            )

        spread, question = self._parse_args(ctx.args)
        already_drawn = False

        if spread == "daily":
            user_hash = hash_phone(ctx.sender)
            try:
                card, reversed_, already_drawn = await self.db.get_or_draw(user_hash)
            except Exception as e:
                logger.exception(f"tarot daily: db error: {e}")
                return CommandResult.error("Couldn't reach the daily-card store.")
            draws = [Draw(card, reversed_, "Today")]
            spread_kind = "single"
        else:
            n_cards, positions, spread_kind = {
                "three": (3, THREE_CARD_POSITIONS, "three"),
                "celtic": (10, CELTIC_POSITIONS, "celtic"),
                "single": (1, [None], "single"),
            }[spread]
            draws = [
                Draw(d.card, d.reversed, p)
                for d, p in zip(_draw_n(n_cards), positions)
            ]

        # Compose the image off the event loop — PIL is blocking.
        try:
            image_b64 = await run_blocking(compose, draws, spread_kind, timeout=15.0)
        except Exception as e:
            logger.exception(f"tarot: image compose failed: {e}")
            return CommandResult.error("Couldn't render the spread.")

        reading_text = await self._build_reading(draws, spread_kind, question)

        header = self._header_for(
            spread_kind, daily=(spread == "daily"), already_drawn=already_drawn,
        )
        body = f"{header}\n\n{reading_text}"

        return CommandResult(
            text=body,
            success=True,
            attachments=[image_b64],
            styled=False,
        )

    @staticmethod
    def _parse_args(args: list[str]) -> tuple[str, Optional[str]]:
        """Return (spread_kind, question) from the raw arg list."""
        if not args:
            return "single", None
        first = args[0].lower()
        kind = SPREAD_ALIAS_TO_KIND.get(first)
        if kind is not None:
            return kind, " ".join(args[1:]).strip() or None
        # First word doesn't name a spread → treat the whole thing as a
        # question on a single-card draw.
        return "single", " ".join(args).strip() or None

    @staticmethod
    def _header_for(spread_kind: str, *, daily: bool, already_drawn: bool) -> str:
        if daily and already_drawn:
            return "✦ Your card of the day (drawn earlier — same card all day):"
        if daily:
            return "✦ Your card of the day:"
        return {
            "single": "✦ Your card:",
            "three": "✦ Past · Present · Future:",
            "celtic": "✦ Celtic Cross:",
        }.get(spread_kind, "✦ Your reading:")

    async def _build_reading(
        self,
        draws: list[Draw],
        spread_kind: str,
        question: Optional[str],
    ) -> str:
        if self.llm is None:
            return _static_reading(draws, spread_kind)
        try:
            status = self.llm.status()
        except Exception:
            return _static_reading(draws, spread_kind)
        if not status.get("ready"):
            return _static_reading(draws, spread_kind)

        spread_name = {
            "single": "single-card",
            "three": "three-card past/present/future",
            "celtic": "Celtic Cross",
        }.get(spread_kind, spread_kind)

        prompt_parts = [f"Spread: {spread_name}"]
        if question:
            prompt_parts.append(f"Querent's question: {question}")
        else:
            prompt_parts.append("No specific question — give a general read.")
        prompt_parts.append("\nCards drawn:\n" + _format_cards_for_llm(draws))

        # Use chat_messages (not chat) so we can override max_tokens — the
        # global default truncates Celtic Cross readings mid-sentence.
        messages = [
            {"role": "system", "content": TAROT_SYSTEM_PROMPT},
            {"role": "user", "content": "\n\n".join(prompt_parts)},
        ]
        overrides = {"max_tokens": TAROT_MAX_TOKENS.get(spread_kind, 1200)}

        try:
            msg = await self.llm.chat_messages(
                messages,
                overrides=overrides,
                suppress_response_style=True,
                purpose="tarot",
            )
            text = (msg.get("content") or "").strip()
            if text:
                return text
        except Exception as e:
            logger.warning(f"tarot: LLM reading failed, using fallback: {e}")
        return _static_reading(draws, spread_kind)
