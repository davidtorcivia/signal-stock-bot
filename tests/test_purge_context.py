"""
Tests for the purge-context flow:

  * `ContextRegistry.set_purge_floor` stamps the floor and round-trips.
  * `ConversationHistory.load`, `get_summary`, `latest_turn_timestamp`,
    and `turns_to_summarize` honor `floor_at` — pre-floor rows are
    invisible even if they're still in the table.
  * `GroupMessageLog.recent` filters ALL rows (bot AND user) older than
    the floor — a purge means the room starts fresh.
  * `ContextPolicy.storage_key` maps to the prefixed history key the
    purge action must clear.
  * `GroupMessageLog.clear_bot_rows` only deletes BOT_SENDER rows.
  * `EmojiReactor.clear_recent` drops the in-process deque.
"""

import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.contexts import ContextPolicy, ContextRegistry
from src.group_log import BOT_SENDER, GroupMessageLog
from src.llm.history import ConversationHistory


@pytest.fixture
def db_path():
    with tempfile.TemporaryDirectory() as d:
        yield str(Path(d) / "test.db")


# ── registry: set/round-trip the floor ────────────────────────────────


async def test_set_purge_floor_round_trip(db_path):
    reg = ContextRegistry(db_path=db_path)
    p = ContextPolicy(id=None, kind="group", key="grp-floor", label="Floor")
    new_id = await reg.upsert(p)

    fresh = await reg.get(new_id)
    assert fresh is not None and fresh.purge_floor_at is None

    floor = time.time()
    assert await reg.set_purge_floor(new_id, floor) is True

    reloaded = await reg.get(new_id)
    assert reloaded is not None
    assert reloaded.purge_floor_at == pytest.approx(floor, abs=0.01)


def test_policy_storage_key_matches_history_keying():
    """The purge action clears history by storage_key(); it must match the
    `group:`/`dm:` form CommandContext.context_key() writes under — passing
    the bare row key (group_id / phone) matched nothing, so purges never
    freed history (every purge logged turns/summary=0)."""
    from src.database import hash_phone

    grp = ContextPolicy(id=1, kind="group", key="ABC123==", label="G")
    assert grp.storage_key() == "group:ABC123=="

    dm = ContextPolicy(id=2, kind="dm", key="+15551234567", label="D")
    assert dm.storage_key() == f"dm:{hash_phone('+15551234567')}"

    # Sanity: it equals what a CommandContext would compute for the same chat.
    from src.commands.base import CommandContext
    gctx = CommandContext(
        sender="+1", group_id="ABC123==", raw_message="", command="", args=[],
    )
    assert gctx.context_key() == grp.storage_key()
    dctx = CommandContext(
        sender="+15551234567", group_id=None, raw_message="", command="",
        args=[],
    )
    assert dctx.context_key() == dm.storage_key()


async def test_upsert_preserves_existing_floor(db_path):
    """Editing the row via upsert (admin form) must not clear the floor."""
    reg = ContextRegistry(db_path=db_path)
    p = ContextPolicy(id=None, kind="group", key="grp-keep", label="Keep")
    new_id = await reg.upsert(p)
    await reg.set_purge_floor(new_id, 12345.6)

    edited = await reg.get(new_id)
    assert edited is not None
    edited.label = "renamed"
    edited.purge_floor_at = None  # admin form doesn't expose it
    await reg.upsert(edited)

    after = await reg.get(new_id)
    assert after is not None
    assert after.label == "renamed"
    assert after.purge_floor_at == pytest.approx(12345.6, abs=0.01)


# ── history reads gated by floor ──────────────────────────────────────


async def test_history_load_excludes_pre_floor_rows(db_path):
    h = ConversationHistory(db_path=db_path, turns_per_user=50)
    await h._ensure_initialized()

    # Backdate the first two rows by an hour so they predate the floor
    # but still survive the 7-day retention prune that append() runs.
    now = time.time()
    old_ts = now - 3600
    floor = now - 1800

    await h.append("ctx-a", "user", "OLD QUESTION")
    await h.append("ctx-a", "assistant", "OLD ANSWER")
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE conversation_turns SET created_at = ? WHERE context_key = ?",
            (old_ts, "ctx-a"),
        )
        await db.commit()

    await h.append("ctx-a", "user", "NEW QUESTION")
    await h.append("ctx-a", "assistant", "NEW ANSWER")

    # Without floor: everything visible.
    all_turns = await h.load("ctx-a")
    assert [t["content"] for t in all_turns] == [
        "OLD QUESTION", "OLD ANSWER", "NEW QUESTION", "NEW ANSWER",
    ]

    # With floor: only post-floor.
    post = await h.load("ctx-a", floor_at=floor)
    assert [t["content"] for t in post] == ["NEW QUESTION", "NEW ANSWER"]


async def test_history_get_summary_floored_out(db_path):
    """A summary updated_at before the floor reads as absent."""
    h = ConversationHistory(db_path=db_path)
    await h._ensure_initialized()
    await h.upsert_summary("ctx-s", "old summary", 7, 4)

    now = time.time()
    old_ts = now - 3600
    floor = now - 1800

    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE conversation_summaries SET updated_at = ? WHERE context_key = ?",
            (old_ts, "ctx-s"),
        )
        await db.commit()

    fetched = await h.get_summary("ctx-s")
    assert fetched is not None
    assert fetched["summary"] == "old summary"
    assert await h.get_summary("ctx-s", floor_at=floor) is None


async def test_history_latest_timestamp_respects_floor(db_path):
    h = ConversationHistory(db_path=db_path)
    await h._ensure_initialized()
    await h.append("ctx-t", "user", "old")
    now = time.time()
    old_ts = now - 3600
    floor = now - 1800
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE conversation_turns SET created_at = ? "
            "WHERE context_key = 'ctx-t'",
            (old_ts,),
        )
        await db.commit()

    assert await h.latest_turn_timestamp("ctx-t") == pytest.approx(old_ts, abs=0.01)
    assert await h.latest_turn_timestamp("ctx-t", floor_at=floor) is None


async def test_turns_to_summarize_respects_floor(db_path):
    h = ConversationHistory(db_path=db_path, turns_per_user=50)
    await h._ensure_initialized()
    for _ in range(6):
        await h.append("ctx-r", "user", "msg")
        await h.append("ctx-r", "assistant", "ans")

    now = time.time()
    old_ts = now - 3600
    floor = now - 1800
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        # Backdate the first 4 turns; later 8 stay current.
        await db.execute(
            "UPDATE conversation_turns SET created_at = ? "
            "WHERE context_key = 'ctx-r' AND id <= "
            "(SELECT MIN(id)+3 FROM conversation_turns WHERE context_key = 'ctx-r')",
            (old_ts,),
        )
        await db.commit()

    all_pending = await h.turns_to_summarize("ctx-r", 0, keep_recent=0)
    floored = await h.turns_to_summarize("ctx-r", 0, keep_recent=0, floor_at=floor)
    assert len(floored) == len(all_pending) - 4


# ── group_log: bot-row floor + user-row preservation ──────────────────


async def test_group_log_recent_floor_filters_all_rows(db_path):
    g = GroupMessageLog(db_path=db_path)
    await g.append("grp-1", "+15551234567", "user-old")
    await g.append_bot("grp-1", "bot-old")
    now = time.time()
    old_ts = now - 3600
    floor = now - 1800
    import aiosqlite
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            "UPDATE group_messages SET created_at = ? WHERE group_id = 'grp-1'",
            (old_ts,),
        )
        await db.commit()
    await g.append("grp-1", "+15551234567", "user-new")
    await g.append_bot("grp-1", "bot-new")

    msgs = await g.recent("grp-1", limit=50)
    assert {m["text"] for m in msgs} == {"user-old", "bot-old", "user-new", "bot-new"}

    # Floor filters EVERY pre-floor row — bot AND user — so a purged room
    # starts fresh and the bots can't replay the pre-purge conversation.
    msgs = await g.recent("grp-1", limit=50, floor_at=floor)
    assert {m["text"] for m in msgs} == {"user-new", "bot-new"}


async def test_clear_bot_rows_leaves_users_alone(db_path):
    g = GroupMessageLog(db_path=db_path)
    await g.append("grp-2", "+15559999999", "u1")
    await g.append_bot("grp-2", "b1")
    await g.append_bot("grp-2", "b2")
    await g.append("grp-2", "+15558888888", "u2")

    removed = await g.clear_bot_rows("grp-2")
    assert removed == 2

    remaining = await g.recent("grp-2", limit=50)
    senders = {m["sender"] for m in remaining}
    assert BOT_SENDER not in senders
    assert {m["text"] for m in remaining} == {"u1", "u2"}


# ── reactor: in-process clear ─────────────────────────────────────────


def test_reactor_clear_recent_drops_deque():
    """clear_recent returns the count it dropped and forgets the group."""
    # Import lazily to avoid pulling heavy deps when reactor isn't needed.
    from src.llm.reactor import EmojiReactor

    r = EmojiReactor(
        llm_client=MagicMock(),
        signal_handler=MagicMock(),
        settings_store=MagicMock(),
    )
    # Seed the in-process deque directly (the public path requires an LLM call).
    r._record_recent(
        group_id="grp-rx", sender_label="...4137",
        target_text="hello", emoji="👍",
    )
    r._record_recent(
        group_id="grp-rx", sender_label="...4137",
        target_text="world", emoji="🙂",
    )

    assert r.clear_recent("grp-rx") == 2
    assert r.recent_reactions("grp-rx") == []
    # Idempotent: a second clear is a no-op.
    assert r.clear_recent("grp-rx") == 0
