"""
Numerology command.

Pythagorean-system numerology: birthdate → life path / personal year /
personal day, name → expression / soul urge / personality / birthday
number. Master numbers (11, 22, 33) are preserved; everything else
reduces to a single digit.

Usage:
  !numerology <birthdate>                  → date-derived numbers only
  !numerology <birthdate> <full name>      → date + name
  !numerology <full name>                  → name-only (no dates)
  !numerology                              → usage hint
"""

from __future__ import annotations

import datetime as dt
import logging
from typing import Optional

import dateparser

from .base import BaseCommand, CommandContext, CommandResult

logger = logging.getLogger(__name__)


# Pythagorean letter values. The Chaldean system is alternative; we pick
# Pythagorean because it's the most common in modern Western numerology
# and the LLM/users will recognize the outputs.
_LETTER_VALUE = {
    "A": 1, "J": 1, "S": 1,
    "B": 2, "K": 2, "T": 2,
    "C": 3, "L": 3, "U": 3,
    "D": 4, "M": 4, "V": 4,
    "E": 5, "N": 5, "W": 5,
    "F": 6, "O": 6, "X": 6,
    "G": 7, "P": 7, "Y": 7,
    "H": 8, "Q": 8, "Z": 8,
    "I": 9, "R": 9,
}

_VOWELS = frozenset("AEIOU")  # Y treated as consonant; the "sometimes vowel"
                              # rule has multiple competing conventions and
                              # consonant-by-default is the one most modern
                              # numerology references settle on.

_MASTER_NUMBERS = frozenset({11, 22, 33})

_NUMBER_MEANING = {
    1:  "the initiator — independence, leadership, fresh starts",
    2:  "the diplomat — partnership, sensitivity, balance",
    3:  "the communicator — expression, joy, creativity",
    4:  "the builder — structure, discipline, foundation",
    5:  "the seeker — change, freedom, motion",
    6:  "the nurturer — responsibility, harmony, service",
    7:  "the seer — analysis, mysticism, inner work",
    8:  "the achiever — power, abundance, mastery of the material",
    9:  "the humanitarian — completion, compassion, the wider field",
    11: "master of intuition — visionary current, heightened perception",
    22: "master builder — vision into form on the world stage",
    33: "master teacher — devotion, healing, sacrificial love",
}


def _reduce(n: int) -> int:
    """Reduce to a single digit, preserving 11/22/33."""
    if n < 0:
        n = abs(n)
    while n > 9:
        if n in _MASTER_NUMBERS:
            return n
        n = sum(int(d) for d in str(n))
    return n


def _digit_sum(s: str) -> int:
    return sum(int(c) for c in s if c.isdigit())


def life_path(birth: dt.date) -> int:
    """Sum every digit of the birthdate, then reduce."""
    return _reduce(_digit_sum(birth.strftime("%Y%m%d")))


def birthday_number(birth: dt.date) -> int:
    """Just the day-of-month, reduced. Day 11/22 are masters."""
    return _reduce(birth.day)


def personal_year(birth: dt.date, year: int) -> int:
    """Month + day + the year you're asking about, all digits summed."""
    return _reduce(_digit_sum(f"{birth.month}{birth.day}{year}"))


def personal_day(birth: dt.date, today: dt.date) -> int:
    """Month + day + today's full date — reading for a single calendar day."""
    return _reduce(
        _digit_sum(f"{birth.month}{birth.day}{today.strftime('%Y%m%d')}")
    )


def expression_number(name: str) -> int:
    """Sum every letter's value, reduce. The 'destiny' number."""
    return _reduce(sum(
        _LETTER_VALUE.get(c, 0) for c in name.upper() if c.isalpha()
    ))


def soul_urge_number(name: str) -> int:
    """Vowels-only — what drives them internally."""
    return _reduce(sum(
        _LETTER_VALUE[c]
        for c in name.upper()
        if c in _VOWELS
    ))


def personality_number(name: str) -> int:
    """Consonants-only — the outward-facing self."""
    return _reduce(sum(
        _LETTER_VALUE[c]
        for c in name.upper()
        if c.isalpha() and c not in _VOWELS
    ))


_DATEPARSER_SETTINGS = {
    "PREFER_DATES_FROM": "past",   # birthdates are in the past
    "DATE_ORDER": "YMD",           # bias toward ISO when ambiguous (1990-05-15)
}


def _try_parse_date(text: str) -> Optional[dt.date]:
    """Best-effort birthdate parse. Returns None if `text` doesn't look like
    a date — caller treats it as a name fragment instead."""
    if not text:
        return None
    parsed = dateparser.parse(text, settings=_DATEPARSER_SETTINGS)
    if parsed is None:
        return None
    # Sanity: birthdates are in the last ~120 years
    today = dt.date.today()
    candidate = parsed.date()
    if candidate > today:
        return None
    if (today - candidate).days > 120 * 365:
        return None
    return candidate


def _split_args(args: list[str]) -> tuple[Optional[dt.date], Optional[str]]:
    """Pull a birthdate off the front of `args` if possible, treat the
    rest as a name. Tries progressively-longer leading slices so phrases
    like 'May 15 1990 David Smith' parse correctly.
    """
    if not args:
        return None, None
    # Try slices [0:1], [0:2], [0:3], [0:4] — birthdate phrasing rarely
    # exceeds 4 tokens ("May 15, 1990" or "1990-05-15")
    for take in (4, 3, 2, 1):
        if take > len(args):
            continue
        head = " ".join(args[:take]).strip().rstrip(",")
        date = _try_parse_date(head)
        if date is not None:
            name_tokens = args[take:]
            name = " ".join(name_tokens).strip() or None
            return date, name
    # Nothing parsed as a date — treat everything as a name
    return None, " ".join(args).strip() or None


class NumerologyCommand(BaseCommand):
    name = "numerology"
    aliases = ["numbers", "num"]
    description = "Pythagorean numerology from a birthdate and/or name."
    usage = "!numerology <birthdate> [<full name>]"
    help_explanation = (
        "Computes life path, personal year/day, expression, soul urge, "
        "personality, and birthday number. Master numbers (11, 22, 33) "
        "are preserved. Examples: `!numerology 1990-05-15`, "
        "`!numerology May 15 1990 David Smith`, `!numerology David Smith`."
    )

    async def execute(self, ctx: CommandContext) -> CommandResult:
        birth, name = _split_args(ctx.args)
        if birth is None and not name:
            return CommandResult.error(
                "Give me a birthdate, a name, or both. "
                "Examples: `!numerology 1990-05-15`, "
                "`!numerology May 15 1990 David Smith`."
            )

        today = dt.date.today()
        lines: list[str] = []
        header_parts: list[str] = []
        if name:
            header_parts.append(name)
        if birth:
            header_parts.append(f"born {birth.isoformat()}")
        header = (
            f"✦ Numerology for {' · '.join(header_parts)}"
            if header_parts else "✦ Numerology"
        )
        lines.append(header)
        lines.append("")

        if birth is not None:
            lp = life_path(birth)
            bd = birthday_number(birth)
            py = personal_year(birth, today.year)
            pd = personal_day(birth, today)
            lines.append(f"  Life path: *{lp}* — {_NUMBER_MEANING[lp]}")
            lines.append(f"  Birthday:  *{bd}* — {_NUMBER_MEANING[bd]}")
            lines.append(
                f"  Personal year ({today.year}): *{py}* — "
                f"{_NUMBER_MEANING[py]}"
            )
            lines.append(
                f"  Personal day ({today.isoformat()}): *{pd}* — "
                f"{_NUMBER_MEANING[pd]}"
            )

        if name:
            if birth is not None:
                lines.append("")
            ex = expression_number(name)
            su = soul_urge_number(name)
            ps = personality_number(name)
            lines.append(f"  Expression:  *{ex}* — {_NUMBER_MEANING[ex]}")
            lines.append(f"  Soul urge:   *{su}* — {_NUMBER_MEANING[su]}")
            lines.append(f"  Personality: *{ps}* — {_NUMBER_MEANING[ps]}")

        return CommandResult(text="\n".join(lines), success=True, styled=True)
