"""Tests for the prompt-formatting helpers introduced in the v2 prompt build.

Covers:
  - format_history_timestamp absolute UTC rendering (cache-stable)
  - _wrap_xml empty/non-empty behaviour
  - ConversationHistory.load() timestamp + attribution prefixes
  - ConversationHistory.latest_turn_timestamp() empty + populated
"""

import re
import time
from datetime import datetime, timezone

import pytest

from src.commands.ask_command import (
    _strip_addressee_leak,
    _strip_meta_leak,
    _wrap_xml,
    STALENESS_THRESHOLD_SECONDS,
)
from src.llm.history import ConversationHistory, format_history_timestamp


# Matches the `YYYY-MM-DD HH:MM UTC` stamp the history renderer now emits.
_UTC_STAMP = r"\d{4}-\d{2}-\d{2} \d{2}:\d{2} UTC"


# ---------- format_history_timestamp -----------------------------------------

def test_format_history_timestamp_shape():
    # A known epoch second renders to its exact UTC wall-clock minute.
    # 1749234600 == 2025-06-06 18:30:00 UTC.
    assert format_history_timestamp(1749234600) == "2025-06-06 18:30 UTC"


def test_format_history_timestamp_truncates_to_minute():
    # Sub-minute seconds are dropped so the string is stable for a full
    # minute — adding 59s within the same minute must not change it.
    base = 1749234600  # ...18:30:00 UTC
    assert format_history_timestamp(base) == format_history_timestamp(base + 59)


def test_format_history_timestamp_is_absolute_not_relative():
    # The whole point of the change: the rendered stamp depends ONLY on the
    # turn's own created_at, never on "now". Rendering the same timestamp
    # twice (as if from two different requests) yields identical text, so the
    # prompt prefix stays cacheable.
    ts = time.time() - 3600
    assert format_history_timestamp(ts) == format_history_timestamp(ts)


def test_format_history_timestamp_matches_utc_clock():
    ts = time.time()
    expected = datetime.fromtimestamp(ts, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M UTC"
    )
    assert format_history_timestamp(ts) == expected


# ---------- _wrap_xml --------------------------------------------------------

def test_wrap_xml_empty_returns_empty_string():
    assert _wrap_xml("foo", "") == ""
    assert _wrap_xml("foo", "   \n  ") == ""
    assert _wrap_xml("foo", None) == ""  # type: ignore[arg-type]


def test_wrap_xml_non_empty_wraps():
    assert _wrap_xml("group_context", "hi there") == "<group_context>\nhi there\n</group_context>"


def test_wrap_xml_strips_inner_whitespace():
    assert _wrap_xml("x", "\n  body  \n") == "<x>\nbody\n</x>"


# ---------- _strip_addressee_leak --------------------------------------------

def test_strip_addressee_leak_basic():
    # The most common case: model copied "[to David] " into its reply.
    assert _strip_addressee_leak("[to David] hey, AAPL is at 150") == "hey, AAPL is at 150"


def test_strip_addressee_leak_with_timestamp():
    # Sometimes models echo the full bracket including the time portion.
    assert (
        _strip_addressee_leak("[to David, 2m ago] price check")
        == "price check"
    )


def test_strip_addressee_leak_with_tail_form():
    # Unregistered users come through as `...4137` — the leak form should
    # also strip cleanly when the addressee uses that fallback.
    assert _strip_addressee_leak("[to ...4137] sure") == "sure"


def test_strip_addressee_leak_leading_whitespace():
    assert _strip_addressee_leak("  [to David] yo") == "yo"


def test_strip_addressee_leak_double_label():
    # If the model somehow stacks two brackets, both come off.
    assert (
        _strip_addressee_leak("[to David, 2m ago] [to David] hi")
        == "hi"
    )


def test_strip_addressee_leak_passthrough():
    # Plain text is unchanged.
    assert _strip_addressee_leak("hello there") == "hello there"


def test_strip_addressee_leak_doesnt_eat_other_brackets():
    # Brackets that are NOT the addressee form must survive — bullets,
    # citations, code, etc.
    assert _strip_addressee_leak("[1] see ref") == "[1] see ref"
    assert _strip_addressee_leak("[note] foo") == "[note] foo"


def test_strip_addressee_leak_empty_input():
    assert _strip_addressee_leak("") == ""
    assert _strip_addressee_leak(None) is None  # type: ignore[arg-type]


# ---------- _strip_meta_leak (general directive-text leaks) ------------------

def test_strip_meta_leak_spontaneous_reply_opener():
    """The implicit-ask directive used to start with `Spontaneous reply:` —
    some models echoed that opener verbatim into their visible output."""
    leaked = (
        "Spontaneous reply: this message was NOT addressed to you directly\n"
        "Hey, AAPL just hit 250."
    )
    assert _strip_meta_leak(leaked) == "Hey, AAPL just hit 250."


def test_strip_meta_leak_spontaneous_reply_path_variant():
    # The post-fix directive uses "spontaneous-reply path" — also strip if
    # the model parrots that phrasing.
    leaked = "spontaneous-reply path: ok\nReal answer here."
    out = _strip_meta_leak(leaked)
    assert "spontaneous" not in out.lower()
    assert "Real answer here." in out


def test_strip_meta_leak_other_directive_labels():
    # Each of these is a known directive-block opener; none should ever
    # legitimately start a reply.
    for opener in (
        "Reflex note: emoji reactions are mine\nReal text",
        "Identity note: registered names available\nReal text",
        "Attribution rules: this is multi-speaker\nReal text",
    ):
        out = _strip_meta_leak(opener)
        assert out.strip().startswith("Real text"), f"didn't strip: {opener!r}"


def test_strip_meta_leak_stacked_leaks():
    """Sometimes the model stacks an addressee bracket AND a directive
    label. Both come off."""
    leaked = "[to David, 2m ago] Spontaneous reply: hey\nthe price is $150"
    out = _strip_meta_leak(leaked)
    assert "Spontaneous" not in out
    assert "[to" not in out
    assert "the price is $150" in out


def test_strip_meta_leak_mid_reply_quotation_preserved():
    # If the bot legitimately quotes one of these phrases mid-reply (e.g.
    # explaining what `[to ...]` is), that's not a leak — only leading
    # matches are stripped.
    answer = "Sure — `[to David]` means the addressee is David."
    assert _strip_meta_leak(answer) == answer


def test_strip_meta_leak_passthrough():
    assert _strip_meta_leak("hello there") == "hello there"
    assert _strip_meta_leak("") == ""


def test_strip_meta_leak_user_turn_speaker_bracket():
    """The actual leak observed in production: the model echoed the
    user-turn `[Name, time]` format from history playback into its
    own reply. Real example: `[J, just now] That tracks. ...`"""
    leaked = (
        "[J, just now] That tracks. The Pirate Party is the only "
        "platform that requires you to read the manifesto cover to cover."
    )
    out = _strip_meta_leak(leaked)
    assert out.startswith("That tracks.")
    assert "[J, just now]" not in out


def test_strip_meta_leak_user_turn_with_minutes_ago():
    assert (
        _strip_meta_leak("[David, 2m ago] hey there")
        == "hey there"
    )


def test_strip_meta_leak_user_turn_with_tail_form():
    assert (
        _strip_meta_leak("[...4137, 5h ago] sure")
        == "sure"
    )


def test_strip_meta_leak_user_turn_with_utc_stamp():
    """Current bracket form: the model echoes the absolute-UTC speaker label
    from history playback. Both the named and tail variants must strip."""
    assert (
        _strip_meta_leak("[David, 2026-06-06 18:30 UTC] hey there")
        == "hey there"
    )
    assert (
        _strip_meta_leak("[...4137, 2026-06-06 18:30 UTC] sure")
        == "sure"
    )


def test_strip_meta_leak_assistant_turn_with_utc_stamp():
    assert (
        _strip_meta_leak("[to David, 2026-06-06 18:30 UTC] price check")
        == "price check"
    )


def test_strip_meta_leak_user_turn_paraphrased_time():
    """The model sometimes paraphrases the time ('a moment ago',
    '5 minutes ago') instead of the canonical short form. Strip those too."""
    assert (
        _strip_meta_leak("[Sarah, a moment ago] yo")
        == "yo"
    )
    assert (
        _strip_meta_leak("[Sarah, 5 minutes ago] yo")
        == "yo"
    )


def test_strip_meta_leak_does_not_eat_legit_bracket_pairs():
    """Sanity: the comma+time constraint protects against accidentally
    stripping legitimate `[X, Y]` pairs that aren't time-shaped."""
    # A footnote-style citation should survive.
    assert _strip_meta_leak("[Ref, p.42] details") == "[Ref, p.42] details"
    # A title in brackets, no time-shape, also survives.
    assert _strip_meta_leak("[Q1, 2026] earnings") == "[Q1, 2026] earnings"


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


async def _seed_turn(h, ctx, role, content, sender_tail=None, user_hash=None, age_seconds=0):
    """Insert a turn, backdate created_at by `age_seconds`, return that
    created_at so callers can assert the exact UTC stamp the renderer emits."""
    await h.append(ctx, role, content, sender_tail=sender_tail, user_hash=user_hash)
    if not age_seconds:
        return None
    created = time.time() - age_seconds
    import aiosqlite
    async with aiosqlite.connect(h.db_path) as db:
        await db.execute(
            "UPDATE conversation_turns SET created_at = ? "
            "WHERE id = (SELECT MAX(id) FROM conversation_turns WHERE context_key = ?)",
            (created, ctx),
        )
        await db.commit()
    return created


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
    created = await _seed_turn(
        h, "dm:bob", "user", "old question", age_seconds=3600 * 2
    )
    await _seed_turn(h, "dm:bob", "assistant", "old answer", age_seconds=3600 * 2)
    now = time.time()
    turns = await h.load("dm:bob", now=now)
    stamp = format_history_timestamp(created)
    assert turns[0] == {"role": "user", "content": f"[{stamp}] old question"}
    # The stamp is absolute UTC, not "2h ago".
    assert re.fullmatch(rf"\[{_UTC_STAMP}\] old question", turns[0]["content"])
    # Assistant turns are not timestamp-prefixed — they implicitly follow
    # the user turn they answered.
    assert turns[1] == {"role": "assistant", "content": "old answer"}


async def test_load_group_attribution_and_timestamps(history):
    h = history
    created = await _seed_turn(
        h, "group:42", "user", "q1", sender_tail="4137", age_seconds=120
    )
    await _seed_turn(h, "group:42", "assistant", "a1", age_seconds=120)
    now = time.time()
    turns = await h.load("group:42", attribute_senders=True, now=now)
    stamp = format_history_timestamp(created)
    assert turns[0]["role"] == "user"
    assert turns[0]["content"] == f"[...4137, {stamp}] q1"
    assert turns[1] == {"role": "assistant", "content": "a1"}


async def test_load_group_attribution_only_no_now(history):
    h = history
    await h.append("group:42", "user", "q1", sender_tail="4137")
    turns = await h.load("group:42", attribute_senders=True)
    # No now → no timestamp portion in the bracket.
    assert turns[0]["content"] == "[...4137] q1"


async def test_load_group_assistant_tagged_with_addressee(history):
    """Assistant turns in groups get `[to <addressee>, <ago>]` so the model
    can pair its prior replies with the right asker across speakers."""
    h = history
    q_created = await _seed_turn(
        h, "group:42", "user", "q1", sender_tail="4137", age_seconds=180,
    )
    a_created = await _seed_turn(
        h, "group:42", "assistant", "a1", sender_tail="4137", age_seconds=180,
    )
    now = time.time()
    turns = await h.load("group:42", attribute_senders=True, now=now)
    assert turns[0]["content"] == f"[...4137, {format_history_timestamp(q_created)}] q1"
    assert turns[1]["content"] == f"[to ...4137, {format_history_timestamp(a_created)}] a1"


async def test_load_group_assistant_no_addressee_falls_back_to_bare(history):
    """When neither user_hash nor sender_tail is stored on the assistant
    row (legacy data), the load path falls back to bare content rather than
    inventing an addressee."""
    h = history
    q_created = await _seed_turn(
        h, "group:42", "user", "q1", sender_tail="4137", age_seconds=120,
    )
    await _seed_turn(h, "group:42", "assistant", "a1", age_seconds=120)
    now = time.time()
    turns = await h.load("group:42", attribute_senders=True, now=now)
    assert turns[0]["content"] == f"[...4137, {format_history_timestamp(q_created)}] q1"
    assert turns[1] == {"role": "assistant", "content": "a1"}


async def test_load_dm_assistant_unaffected_by_addressee_change(history):
    """Even with sender_tail set on the assistant row, DM playback (no
    attribute_senders) must keep assistant content bare — the addressee
    label is purely a multi-speaker affordance."""
    h = history
    q_created = await _seed_turn(
        h, "dm:bob", "user", "q1", sender_tail="4137", age_seconds=120
    )
    await _seed_turn(h, "dm:bob", "assistant", "a1", sender_tail="4137", age_seconds=120)
    now = time.time()
    turns = await h.load("dm:bob", now=now)
    assert turns[0] == {
        "role": "user", "content": f"[{format_history_timestamp(q_created)}] q1",
    }
    assert turns[1] == {"role": "assistant", "content": "a1"}


async def test_load_group_assistant_resolves_name_via_registry(history, tmp_path):
    """When a name is registered for the addressee user_hash, assistant
    turns render as `[to <Name>, <UTC stamp>]` instead of falling back to the
    `...tail` form."""
    from src.users import NameRegistry
    from src.database import hash_phone

    phone = "+15555554137"
    user_hash = hash_phone(phone)

    registry = NameRegistry(db_path=str(tmp_path / "names.db"), bot_name="Sigil")
    await registry.set_name("David", user_hash=user_hash)
    history.name_registry = registry

    q_created = await _seed_turn(
        history, "group:99", "user", "q1",
        sender_tail="4137", user_hash=user_hash, age_seconds=60,
    )
    a_created = await _seed_turn(
        history, "group:99", "assistant", "a1",
        sender_tail="4137", user_hash=user_hash, age_seconds=60,
    )
    now = time.time()
    turns = await history.load("group:99", attribute_senders=True, now=now)
    assert turns[0]["content"] == f"[David, {format_history_timestamp(q_created)}] q1"
    assert turns[1]["content"] == f"[to David, {format_history_timestamp(a_created)}] a1"


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
