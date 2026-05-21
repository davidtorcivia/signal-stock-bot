"""Multi-bot-in-one-chat integration tests.

Validates that two bots sharing the same Signal group:
  - get separate portfolios (battle-bots scoping)
  - keep their memory isolated (each bot's own mental model)
  - see each other's prior turns as participants in group_log
  - get history filtered to their own assistant arc but shared user turns
  - register cross-bot alias matching at the addressing layer
"""

from __future__ import annotations

import pytest

from src.group_log import GroupMessageLog, BOT_SENDER
from src.llm.history import ConversationHistory
from src.memory import MemoryStore, KIND_FACT
from src.paper_portfolio import (
    PortfolioStore, portfolio_context_key, parse_portfolio_key,
)


CTX = "group:battle-arena"


# ---------- portfolio per-bot scoping --------------------------------------

@pytest.mark.asyncio
async def test_portfolio_per_bot_keys_isolate(tmp_path):
    """Two bots in the same group keep separate cash + positions."""
    store = PortfolioStore(db_path=str(tmp_path / "p.db"))
    sigil_key = portfolio_context_key(CTX, bot_id=1)
    artaud_key = portfolio_context_key(CTX, bot_id=2)
    assert sigil_key != artaud_key

    # Each bot's first interaction auto-seeds its own $1000 portfolio.
    sigil = await store.ensure_portfolio(sigil_key, label="Sigil")
    artaud = await store.ensure_portfolio(artaud_key, label="Artaud")
    assert sigil.cash == pytest.approx(1000.0)
    assert artaud.cash == pytest.approx(1000.0)

    # Buy a position only on Sigil's portfolio.
    await store.buy(sigil_key, "AAPL", qty=2, price=150.0)
    sigil_pos = await store.get_position(sigil_key, "AAPL")
    artaud_pos = await store.get_position(artaud_key, "AAPL")
    assert sigil_pos is not None
    assert sigil_pos.qty == pytest.approx(2)
    # Artaud's portfolio is untouched — that's the whole point.
    assert artaud_pos is None
    artaud_portfolio = await store.get_portfolio(artaud_key)
    assert artaud_portfolio.cash == pytest.approx(1000.0)


@pytest.mark.asyncio
async def test_portfolio_key_round_trips(tmp_path):
    """parse_portfolio_key is the inverse of portfolio_context_key."""
    scoped = portfolio_context_key("group:abc", bot_id=42)
    ck, bot_id = parse_portfolio_key(scoped)
    assert ck == "group:abc"
    assert bot_id == 42
    # Unscoped legacy key round-trips as (key, None).
    ck, bot_id = parse_portfolio_key("group:legacy")
    assert ck == "group:legacy"
    assert bot_id is None


@pytest.mark.asyncio
async def test_portfolio_migration_idempotent(tmp_path):
    """Legacy unscoped rows get migrated; a second migration is a no-op."""
    store = PortfolioStore(db_path=str(tmp_path / "p.db"))
    # Seed a legacy unscoped portfolio + position.
    await store.ensure_portfolio(CTX)
    await store.buy(CTX, "AAPL", qty=1, price=100.0)
    # Migrate to default bot (id=1).
    n1 = await store.migrate_to_bot_scope(default_bot_id=1)
    assert n1 > 0
    # Second call: nothing left to migrate.
    n2 = await store.migrate_to_bot_scope(default_bot_id=1)
    assert n2 == 0
    # The portfolio is now reachable via the scoped key.
    scoped = portfolio_context_key(CTX, bot_id=1)
    p = await store.get_portfolio(scoped)
    assert p is not None
    pos = await store.get_position(scoped, "AAPL")
    assert pos is not None and pos.qty == pytest.approx(1)


# ---------- memory per-bot isolation ---------------------------------------

@pytest.mark.asyncio
async def test_memory_isolated_by_bot_id(tmp_path):
    """Sigil's self-memory is invisible to Artaud (and vice-versa)."""
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    # Sigil stores a self-fact.
    sigil_mem = await store.add(
        context_id=1, subject_key="__self__",
        subject_label="the bot", kind=KIND_FACT,
        content="I prefer terse responses",
        bot_id=1,
    )
    # Artaud stores a different self-fact.
    artaud_mem = await store.add(
        context_id=1, subject_key="__self__",
        subject_label="the bot", kind=KIND_FACT,
        content="I lean theatrical and long-winded",
        bot_id=2,
    )
    assert sigil_mem != artaud_mem  # distinct rows (per-bot dedup scope)

    sigil_view = await store.list_for_subject(
        context_id=1, subject_key="__self__", bot_id=1,
    )
    artaud_view = await store.list_for_subject(
        context_id=1, subject_key="__self__", bot_id=2,
    )
    # Each bot sees its own self-memory and only that one.
    sigil_contents = {r["content"] for r in sigil_view}
    artaud_contents = {r["content"] for r in artaud_view}
    assert "I prefer terse responses" in sigil_contents
    assert "I lean theatrical and long-winded" not in sigil_contents
    assert "I lean theatrical and long-winded" in artaud_contents
    assert "I prefer terse responses" not in artaud_contents


@pytest.mark.asyncio
async def test_memory_legacy_null_rows_shared(tmp_path):
    """Legacy rows (pre-multi-bot) carry bot_id=NULL and remain visible
    to any bot's read — back-compat with single-bot installs."""
    store = MemoryStore(db_path=str(tmp_path / "m.db"))
    legacy = await store.add(
        context_id=1, subject_key="__context__",
        subject_label="the chat", kind=KIND_FACT,
        content="this chat formed in 2024",
        bot_id=None,
    )
    assert legacy is not None
    sigil_view = await store.list_for_subject(
        context_id=1, subject_key="__context__", bot_id=1,
    )
    artaud_view = await store.list_for_subject(
        context_id=1, subject_key="__context__", bot_id=2,
    )
    assert any("formed in 2024" in r["content"] for r in sigil_view)
    assert any("formed in 2024" in r["content"] for r in artaud_view)


# ---------- group_log bot attribution --------------------------------------

@pytest.mark.asyncio
async def test_group_log_records_writing_bot(tmp_path):
    """append_bot stores bot_id; recent() surfaces it for the renderer."""
    log = GroupMessageLog(db_path=str(tmp_path / "g.db"))
    await log.append("g1", "+15551111111", "hello bots")
    await log.append_bot("g1", "hi from sigil", bot_id=1)
    await log.append_bot("g1", "hi from artaud", bot_id=2)
    msgs = await log.recent("g1", limit=10)
    assert len(msgs) == 3
    user_msg = msgs[0]
    sigil_msg = msgs[1]
    artaud_msg = msgs[2]
    assert user_msg["sender"] == "+15551111111"
    assert user_msg["bot_id"] is None  # user turn, no bot_id
    assert sigil_msg["sender"] == BOT_SENDER
    assert sigil_msg["bot_id"] == 1
    assert artaud_msg["sender"] == BOT_SENDER
    assert artaud_msg["bot_id"] == 2


# ---------- conversation history per-bot filter -----------------------------

@pytest.mark.asyncio
async def test_history_filters_assistant_by_bot(tmp_path):
    """load(bot_id=N) returns all user turns + only bot N's assistant
    turns. Other bots' assistant turns surface via group_log instead."""
    h = ConversationHistory(
        db_path=str(tmp_path / "h.db"), turns_per_user=50,
    )
    # Sigil asked + answered.
    await h.append(CTX, "user", "first question", bot_id=1)
    await h.append(CTX, "assistant", "sigil's first reply", bot_id=1)
    # Artaud asked + answered.
    await h.append(CTX, "user", "second question", bot_id=2)
    await h.append(CTX, "assistant", "artaud's reply", bot_id=2)

    sigil_view = await h.load(CTX, bot_id=1)
    artaud_view = await h.load(CTX, bot_id=2)

    # Both bots see ALL user turns.
    sigil_user_msgs = [t["content"] for t in sigil_view if t["role"] == "user"]
    artaud_user_msgs = [t["content"] for t in artaud_view if t["role"] == "user"]
    assert "first question" in sigil_user_msgs
    assert "second question" in sigil_user_msgs
    assert "first question" in artaud_user_msgs
    assert "second question" in artaud_user_msgs

    # Each bot only sees ITS OWN assistant turns.
    sigil_replies = [t["content"] for t in sigil_view if t["role"] == "assistant"]
    artaud_replies = [t["content"] for t in artaud_view if t["role"] == "assistant"]
    assert "sigil's first reply" in sigil_replies
    assert "artaud's reply" not in sigil_replies
    assert "artaud's reply" in artaud_replies
    assert "sigil's first reply" not in artaud_replies


@pytest.mark.asyncio
async def test_history_legacy_null_bot_rows_visible(tmp_path):
    """Legacy assistant turns with bot_id=NULL replay as the calling
    bot's own (back-compat for pre-multi-bot installs)."""
    h = ConversationHistory(
        db_path=str(tmp_path / "h.db"), turns_per_user=50,
    )
    await h.append(CTX, "user", "old question")
    await h.append(CTX, "assistant", "old answer")  # bot_id=None

    sigil_view = await h.load(CTX, bot_id=1)
    sigil_replies = [t["content"] for t in sigil_view if t["role"] == "assistant"]
    assert "old answer" in sigil_replies
