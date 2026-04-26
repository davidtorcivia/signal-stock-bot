"""Tests for the prompt-formatting helpers introduced in the v2 prompt build.

Covers:
  - format_relative_age bucket boundaries
  - _wrap_xml empty/non-empty behaviour
  - ConversationHistory.load() timestamp + attribution prefixes
  - ConversationHistory.latest_turn_timestamp() empty + populated
"""

import time

import pytest

from src.commands.ask_command import _wrap_xml, STALENESS_THRESHOLD_SECONDS
from src.llm.history import ConversationHistory, format_relative_age


# ---------- format_relative_age ----------------------------------------------

def test_format_relative_age_just_now():
    assert format_relative_age(0) == "just now"
    assert format_relative_age(59) == "just now"


def test_format_relative_age_minutes():
    assert format_relative_age(60) == "1m ago"
    assert format_relative_age(599) == "9m ago"
    assert format_relative_age(3599) == "59m ago"


def test_format_relative_age_hours():
    assert format_relative_age(3600) == "1h ago"
    assert format_relative_age(7200) == "2h ago"
    assert format_relative_age(86399) == "23h ago"


def test_format_relative_age_days():
    assert format_relative_age(86400) == "1d ago"
    assert format_relative_age(3 * 86400) == "3d ago"
    assert format_relative_age(7 * 86400 - 1) == "6d ago"


def test_format_relative_age_weeks():
    assert format_relative_age(7 * 86400) == "1w ago"
    assert format_relative_age(30 * 86400) == "4w ago"


def test_format_relative_age_negative_clamps_to_just_now():
    # Clock skew between writer and reader shouldn't crash or produce
    # nonsense — treat any negative age as "fresh".
    assert format_relative_age(-5) == "just now"


# ---------- _wrap_xml --------------------------------------------------------

def test_wrap_xml_empty_returns_empty_string():
    assert _wrap_xml("foo", "") == ""
    assert _wrap_xml("foo", "   \n  ") == ""
    assert _wrap_xml("foo", None) == ""  # type: ignore[arg-type]


def test_wrap_xml_non_empty_wraps():
    assert _wrap_xml("group_context", "hi there") == "<group_context>\nhi there\n</group_context>"


def test_wrap_xml_strips_inner_whitespace():
    assert _wrap_xml("x", "\n  body  \n") == "<x>\nbody\n</x>"


# ---------- staleness threshold ----------------------------------------------

def test_staleness_threshold_is_hours_not_minutes():
    # If someone tunes this down to minutes, every chat triggers the stale
    # advisory and the model starts treating live conversations as cold.
    assert STALENESS_THRESHOLD_SECONDS >= 3600


# ---------- history.load + timestamps ----------------------------------------

@pytest.fixture
def history(tmp_path):
    db = tmp_path / "test.db"
    return ConversationHistory(db_path=str(db), turns_per_user=10)


async def _seed_turn(h, ctx, role, content, sender_tail=None, age_seconds=0):
    """Insert a turn and backdate created_at by `age_seconds`."""
    await h.append(ctx, role, content, sender_tail=sender_tail)
    if age_seconds:
        import aiosqlite
        async with aiosqlite.connect(h.db_path) as db:
            await db.execute(
                "UPDATE conversation_turns SET created_at = ? "
                "WHERE id = (SELECT MAX(id) FROM conversation_turns WHERE context_key = ?)",
                (time.time() - age_seconds, ctx),
            )
            await db.commit()


async def test_load_dm_no_attribution_no_timestamps_by_default(history):
    h = history
    await h.append("dm:bob", "user", "hello")
    await h.append("dm:bob", "assistant", "hi")
    turns = await h.load("dm:bob")
    assert turns == [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "hi"},
    ]


async def test_load_dm_timestamps_when_now_passed(history):
    h = history
    await _seed_turn(h, "dm:bob", "user", "old question", age_seconds=3600 * 2)
    await _seed_turn(h, "dm:bob", "assistant", "old answer", age_seconds=3600 * 2)
    now = time.time()
    turns = await h.load("dm:bob", now=now)
    assert turns[0] == {"role": "user", "content": "[2h ago] old question"}
    # Assistant turns are not timestamp-prefixed — they implicitly follow
    # the user turn they answered.
    assert turns[1] == {"role": "assistant", "content": "old answer"}


async def test_load_group_attribution_and_timestamps(history):
    h = history
    await _seed_turn(h, "group:42", "user", "q1", sender_tail="4137", age_seconds=120)
    await _seed_turn(h, "group:42", "assistant", "a1", age_seconds=120)
    now = time.time()
    turns = await h.load("group:42", attribute_senders=True, now=now)
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == "[...4137, 2m ago] q1"
    assert turns[1] == {"role": "assistant", "content": "a1"}


async def test_load_group_attribution_only_no_now(history):
    h = history
    await h.append("group:42", "user", "q1", sender_tail="4137")
    turns = await h.load("group:42", attribute_senders=True)
    # No now → no timestamp portion in the bracket.
    assert turns[0]["content"] == "[...4137] q1"


# ---------- history.latest_turn_timestamp ------------------------------------

async def test_latest_turn_timestamp_empty_returns_none(history):
    h = history
    assert await h.latest_turn_timestamp("nobody:here") is None


async def test_latest_turn_timestamp_returns_max(history):
    h = history
    await _seed_turn(h, "ctx:x", "user", "older", age_seconds=3600)
    await _seed_turn(h, "ctx:x", "user", "newer", age_seconds=60)
    ts = await h.latest_turn_timestamp("ctx:x")
    # Most recent turn is ~60s old, so timestamp should be near now-60.
    assert ts is not None
    assert time.time() - ts == pytest.approx(60, abs=5)
