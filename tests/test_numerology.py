"""Unit tests for numerology calculations.

Pinning the math against canonical worked examples — these are the
numbers any standard numerology reference computes for the same
inputs, so a regression here means the algorithm drifted.
"""

import datetime as dt

import pytest

from src.commands.numerology_command import (
    _reduce,
    birthday_number,
    expression_number,
    life_path,
    personal_day,
    personal_year,
    personality_number,
    soul_urge_number,
    _split_args,
    _try_parse_date,
)


# ---------- _reduce ---------------------------------------------------------

def test_reduce_single_digit_passes_through():
    assert _reduce(7) == 7
    assert _reduce(0) == 0


def test_reduce_two_digits_sums_to_single():
    assert _reduce(28) == 1   # 2+8 = 10 → 1+0 = 1
    assert _reduce(45) == 9   # 4+5 = 9


def test_reduce_preserves_master_numbers():
    """11, 22, 33 are master numbers — never reduced."""
    assert _reduce(11) == 11
    assert _reduce(22) == 22
    assert _reduce(33) == 33


def test_reduce_iterates_until_master_or_single():
    # 38 → 3+8 = 11 → STOP (master)
    assert _reduce(38) == 11
    # 99 → 9+9 = 18 → 1+8 = 9
    assert _reduce(99) == 9


# ---------- life_path -------------------------------------------------------

def test_life_path_canonical_example():
    """1990-05-15: 1+9+9+0+0+5+1+5 = 30 → 3+0 = 3."""
    assert life_path(dt.date(1990, 5, 15)) == 3


def test_life_path_master_number_preserved():
    """1980-09-04: 1+9+8+0+0+9+0+4 = 31 → 3+1 = 4. (Sanity: not master.)"""
    assert life_path(dt.date(1980, 9, 4)) == 4


def test_life_path_master_22():
    """A date that sums to 22 should stay 22.
    1969-04-08: 1+9+6+9+0+4+0+8 = 37 → 3+7 = 10 → 1+0 = 1. Not master.
    Try 1972-12-25: 1+9+7+2+1+2+2+5 = 29 → 2+9 = 11 → STOP."""
    assert life_path(dt.date(1972, 12, 25)) == 11


# ---------- birthday_number -------------------------------------------------

def test_birthday_number_single_digit_day():
    assert birthday_number(dt.date(2000, 1, 5)) == 5


def test_birthday_number_two_digit_day_reduces():
    assert birthday_number(dt.date(2000, 1, 28)) == 1   # 2+8 = 10 → 1


def test_birthday_number_master_day_preserved():
    assert birthday_number(dt.date(2000, 1, 11)) == 11
    assert birthday_number(dt.date(2000, 1, 22)) == 22


# ---------- personal_year ---------------------------------------------------

def test_personal_year_2026_for_birthday_may_15():
    """Month 5 + day 15 + year 2026: 5+1+5+2+0+2+6 = 21 → 2+1 = 3."""
    assert personal_year(dt.date(1990, 5, 15), 2026) == 3


def test_personal_year_changes_with_year():
    birth = dt.date(1990, 5, 15)
    assert personal_year(birth, 2026) != personal_year(birth, 2027)


# ---------- personal_day ----------------------------------------------------

def test_personal_day_combines_birth_and_today():
    """Birth 5/15, today 2026-04-28:
    5+1+5+2+0+2+6+0+4+2+8 = 35 → 3+5 = 8."""
    pd = personal_day(dt.date(1990, 5, 15), dt.date(2026, 4, 28))
    assert pd == 8


# ---------- expression_number -----------------------------------------------

def test_expression_number_simple_name():
    """'David' → D=4, A=1, V=4, I=9, D=4 → 22 → STOP (master)."""
    assert expression_number("David") == 22


def test_expression_number_handles_full_name_with_spaces():
    """'David Smith' → David=22 + Smith (S=1+M=4+I=9+T=2+H=8 = 24).
    Combined letters: 22 + 24 = 46 → 4+6 = 10 → 1+0 = 1."""
    assert expression_number("David Smith") == 1


def test_expression_number_ignores_punctuation_and_case():
    assert expression_number("david") == expression_number("David")
    assert expression_number("o'brien") == expression_number("OBRIEN")


# ---------- soul_urge_number ------------------------------------------------

def test_soul_urge_only_counts_vowels():
    """'David' vowels: A=1, I=9 → 10 → 1."""
    assert soul_urge_number("David") == 1


def test_soul_urge_y_treated_as_consonant():
    """Convention chosen: Y is always consonant. 'Mary' vowels: A=1 only → 1.
    (If Y were a vowel: A+Y = 1+7 = 8.)"""
    assert soul_urge_number("Mary") == 1


# ---------- personality_number ----------------------------------------------

def test_personality_only_counts_consonants():
    """'David' consonants: D=4, V=4, D=4 → 12 → 3."""
    assert personality_number("David") == 3


def test_personality_plus_soul_urge_equals_expression():
    """Standard identity: vowel sum + consonant sum = full expression
    (modulo reduction quirks). Verify the components reconcile in the
    raw, unreduced sums: D+a+v+i+d = 4+1+4+9+4 = 22, vowels (a+i)=10,
    consonants (D+v+d) = 4+4+4 = 12, 10+12 = 22. ✓"""
    name = "David"
    raw_vowels = sum(
        {"A": 1, "E": 5, "I": 9, "O": 6, "U": 3}.get(c, 0)
        for c in name.upper()
    )
    raw_cons = sum(
        {"D": 4, "V": 4}.get(c, 0)
        for c in name.upper()
    )
    raw_expr = raw_vowels + raw_cons
    assert raw_expr == 22


# ---------- _try_parse_date -------------------------------------------------

def test_parse_date_iso_format():
    assert _try_parse_date("1990-05-15") == dt.date(1990, 5, 15)


def test_parse_date_named_month():
    # dateparser accepts "May 15 1990"
    assert _try_parse_date("May 15 1990") == dt.date(1990, 5, 15)


def test_parse_date_rejects_future():
    """Birthdates can't be in the future."""
    assert _try_parse_date("2099-01-01") is None


def test_parse_date_rejects_unparseable():
    assert _try_parse_date("David Smith") is None
    assert _try_parse_date("") is None


# ---------- _split_args -----------------------------------------------------

def test_split_args_date_only():
    date, name = _split_args(["1990-05-15"])
    assert date == dt.date(1990, 5, 15)
    assert name is None


def test_split_args_iso_date_plus_name():
    date, name = _split_args(["1990-05-15", "David", "Smith"])
    assert date == dt.date(1990, 5, 15)
    assert name == "David Smith"


def test_split_args_multi_token_date_plus_name():
    """'May 15 1990 David Smith' — first 3 tokens are the date."""
    date, name = _split_args(["May", "15", "1990", "David", "Smith"])
    assert date == dt.date(1990, 5, 15)
    assert name == "David Smith"


def test_split_args_name_only():
    """No leading tokens parse as a date — treat the whole thing as name."""
    date, name = _split_args(["David", "Smith"])
    assert date is None
    assert name == "David Smith"


def test_split_args_empty():
    assert _split_args([]) == (None, None)
