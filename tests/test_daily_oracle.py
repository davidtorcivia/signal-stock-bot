"""Header-replacement tests for the daily oracle worker.

The schedule math moved to `tests/test_oracles.py` (now that it lives
in `src/contexts/oracles.py` for the per-context refactor). What's
left in `daily_oracle.py` is the worker loop and the header rewrite,
which has to play well with the existing tarot/iching command output
and slot the active bot's display_name into the framing.
"""

from src.daily_oracle import _replace_header


def test_replace_header_strips_default_tarot_single_header():
    body = "✦ Your card:\n\nThe Fool — beginnings, leap of faith"
    out = _replace_header(body, oracle_label="", bot_name="Sigil")
    assert out.startswith("🌅 Today's oracle from Sigil:")
    assert "✦ Your card:" not in out
    assert "The Fool" in out


def test_replace_header_uses_bot_name_param():
    """The header is parameterized so a non-Sigil bot (e.g. Artaud)
    posts oracles attributed to itself, not to the historic default."""
    body = "✦ Your card:\n\nThe Fool"
    out = _replace_header(body, oracle_label="", bot_name="Artaud")
    assert out.startswith("🌅 Today's oracle from Artaud:")
    assert "Sigil" not in out


def test_replace_header_with_label_appends_to_header():
    body = "✦ Your card:\n\nThe Star"
    out = _replace_header(body, oracle_label="sunrise tarot", bot_name="Sigil")
    assert out.startswith("🌅 Today's oracle from Sigil — sunrise tarot:")
    assert "The Star" in out


def test_replace_header_label_combines_with_arbitrary_bot():
    body = "✦ Your card:\n\nThe Star"
    out = _replace_header(body, oracle_label="sunrise tarot", bot_name="Artaud")
    assert out.startswith("🌅 Today's oracle from Artaud — sunrise tarot:")


def test_replace_header_strips_card_of_the_day_header():
    body = "✦ Your card of the day:\n\nThe Star — hope"
    out = _replace_header(body, oracle_label="", bot_name="Sigil")
    assert "card of the day" not in out
    assert "The Star" in out


def test_replace_header_handles_iching_header():
    body = "☷ Your hexagram:\n\n23: Splitting Apart"
    out = _replace_header(body, oracle_label="dawn cast", bot_name="Sigil")
    assert out.startswith("🌅 Today's oracle from Sigil — dawn cast:")
    assert "Splitting Apart" in out


def test_replace_header_prepends_when_no_known_header():
    """Unknown header? Prepend the oracle framing rather than swallow
    real content."""
    body = "Some surprise format from the future"
    out = _replace_header(body, oracle_label="", bot_name="Sigil")
    assert out.startswith("🌅 Today's oracle from Sigil:")
    assert "Some surprise format" in out


def test_replace_header_default_bot_name_is_neutral():
    """Default exists so the function is callable without a bot_name in
    edge cases, but it must not be 'Sigil' — a missing bot_name should
    read as obviously generic, not silently impersonate a real bot."""
    body = "✦ Your card:\n\nThe Fool"
    out = _replace_header(body, oracle_label="")
    assert "Sigil" not in out
    assert "Artaud" not in out
    assert out.startswith("🌅 Today's oracle from Bot:")
