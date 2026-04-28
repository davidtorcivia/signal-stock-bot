"""Tests for the bot-authored prediction path.

Covers:
  - extract_prediction's happy path + error returns (the public wrapper
    used by both !predict and the predict_self LLM tool)
  - PredictionStore accepting a bot-author row and surfacing it through
    the leaderboard with the stored bot label
"""

import time

import pytest

from src.commands.predict_command import extract_prediction
from src.database import hash_phone
from src.group_log import BOT_SENDER
from src.predictions import (
    PredictionStore,
    STATUS_PENDING,
    STATUS_RESOLVED,
    VERDICT_RIGHT,
    VERDICT_UNCLEAR,
    VERDICT_WRONG,
)


# ---------- extract_prediction -----------------------------------------------

@pytest.mark.asyncio
async def test_extract_prediction_stock_shape_parses_without_llm():
    parsed, err = await extract_prediction(
        "AAPL above $250 by 2099-12-31", llm_client=None,
    )
    assert err is None
    assert parsed is not None
    assert parsed["ticker"] == "AAPL"
    assert parsed["direction"] == "above"
    assert parsed["threshold"] == 250.0
    assert parsed["deadline_utc"] > time.time()


@pytest.mark.asyncio
async def test_extract_prediction_empty_input_errors():
    parsed, err = await extract_prediction("", llm_client=None)
    assert parsed is None
    assert err is not None
    assert "predicting" in err.lower()


@pytest.mark.asyncio
async def test_extract_prediction_no_deadline_errors_before_llm():
    # No "by ...", no date-shape, no relative phrase — bail without
    # calling the LLM (llm_client=None would otherwise let it fall through).
    parsed, err = await extract_prediction(
        "AAPL is going to the moon", llm_client=None,
    )
    assert parsed is None
    assert err is not None
    assert "deadline" in err.lower()


@pytest.mark.asyncio
async def test_extract_prediction_freeform_falls_through_when_llm_unavailable():
    # Has a deadline-trigger phrase, but no llm to extract structure → fail
    # gracefully with a useful message rather than crashing.
    parsed, err = await extract_prediction(
        "Bitcoin crashes by next month", llm_client=None,
    )
    assert parsed is None
    assert err is not None


# ---------- PredictionStore with a bot author --------------------------------

@pytest.fixture
def store(tmp_path):
    return PredictionStore(db_path=str(tmp_path / "predictions.db"))


@pytest.mark.asyncio
async def test_store_accepts_bot_author_and_appears_on_leaderboard(store):
    """The schema's user_hash is just TEXT; a bot-sentinel hash must
    round-trip cleanly and show up on the leaderboard with its stored
    label."""
    bot_hash = hash_phone(BOT_SENDER)

    # Bot logs one prediction in the chat.
    pred_id = await store.create(
        user_hash=bot_hash,
        user_label="Sigil",
        group_id="g1",
        context_key="group:g1",
        claim="AAPL above $250 by 2099-12-31",
        deadline_utc=time.time() + 86400 * 30,
        ticker="AAPL",
        threshold=250.0,
        direction="above",
    )
    assert pred_id > 0

    # Resolve it correct so it counts toward leaderboard.
    await store.resolve(pred_id, verdict=VERDICT_RIGHT, note="hit", resolver_user_hash=None)

    rows = await store.leaderboard("group:g1", limit=10)
    bot_rows = [r for r in rows if r["user_hash"] == bot_hash]
    assert len(bot_rows) == 1
    row = bot_rows[0]
    assert row["label"] == "Sigil"
    assert row["right"] == 1
    assert row["wrong"] == 0


@pytest.mark.asyncio
async def test_total_counts_aggregates_across_contexts(store):
    """Dashboard counter should sum across every context, including
    pending/resolved/expired and per-verdict tallies."""
    deadline = time.time() + 86400
    p1 = await store.create(
        user_hash="h1", user_label="Alice", group_id=None,
        context_key="dm:h1", claim="x", deadline_utc=deadline,
    )
    p2 = await store.create(
        user_hash="h2", user_label="Bob", group_id="g1",
        context_key="group:g1", claim="y", deadline_utc=deadline,
    )
    p3 = await store.create(
        user_hash="h3", user_label="Cara", group_id="g1",
        context_key="group:g1", claim="z", deadline_utc=deadline,
    )
    await store.resolve(p1, verdict=VERDICT_RIGHT, note="", resolver_user_hash=None)
    await store.expire(p2, note="resolver gave up")
    # p3 stays pending

    counts = await store.total_counts()
    assert counts["total"] == 3
    assert counts["pending"] == 1
    assert counts["resolved"] == 1
    assert counts["expired"] == 1
    assert counts["right"] == 1
    assert counts["wrong"] == 0
    assert counts["accuracy"] == 1.0


@pytest.mark.asyncio
async def test_total_counts_empty_store_has_no_accuracy(store):
    counts = await store.total_counts()
    assert counts["total"] == 0
    assert counts["accuracy"] is None


@pytest.mark.asyncio
async def test_contexts_with_predictions_groups_by_context_key(store):
    deadline = time.time() + 86400
    await store.create(
        user_hash="h1", user_label="Alice", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=deadline,
    )
    await store.create(
        user_hash="h2", user_label="Bob", group_id="g1",
        context_key="group:g1", claim="y", deadline_utc=deadline,
    )
    await store.create(
        user_hash="h3", user_label="Cara", group_id=None,
        context_key="dm:h3", claim="z", deadline_utc=deadline,
    )

    rows = await store.contexts_with_predictions()
    by_ck = {r["context_key"]: r for r in rows}
    assert by_ck["group:g1"]["total"] == 2
    assert by_ck["group:g1"]["pending"] == 2
    assert by_ck["dm:h3"]["total"] == 1


@pytest.mark.asyncio
async def test_upcoming_all_orders_by_deadline_ascending(store):
    now = time.time()
    far = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="far", deadline_utc=now + 86400 * 30,
    )
    near = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="near", deadline_utc=now + 60,
    )
    middle = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="middle", deadline_utc=now + 86400,
    )

    upcoming = await store.upcoming_all(limit=5)
    assert [p.id for p in upcoming] == [near, middle, far]


@pytest.mark.asyncio
async def test_upcoming_all_excludes_resolved_and_expired(store):
    now = time.time()
    pending_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="open", deadline_utc=now + 60,
    )
    resolved_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="done", deadline_utc=now + 30,
    )
    await store.resolve(resolved_id, verdict=VERDICT_RIGHT, note="", resolver_user_hash=None)

    upcoming = await store.upcoming_all(limit=10)
    assert [p.id for p in upcoming] == [pending_id]


@pytest.mark.asyncio
async def test_force_set_verdict_overwrites_resolved_row(store):
    """Admin override must work on rows that already have a verdict —
    that's the whole reason it exists, fixing wrong auto-resolutions."""
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="x", deadline_utc=time.time() + 60,
    )
    # Auto-resolver got it wrong
    await store.resolve(pred_id, verdict=VERDICT_WRONG, note="auto", resolver_user_hash=None)

    ok = await store.force_set_verdict(
        pred_id, verdict=VERDICT_RIGHT, note="admin override",
    )
    assert ok is True

    pred = await store.get(pred_id)
    assert pred is not None
    assert pred.status == STATUS_RESOLVED
    assert pred.verdict == VERDICT_RIGHT
    assert pred.resolution_note == "admin override"


@pytest.mark.asyncio
async def test_force_set_verdict_promotes_expired_to_resolved(store):
    """Expired rows don't count for the leaderboard. Admin override
    promotes them back to resolved with a real verdict so they do."""
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="x", deadline_utc=time.time() - 60,
    )
    await store.expire(pred_id, note="resolver gave up")

    ok = await store.force_set_verdict(pred_id, verdict=VERDICT_UNCLEAR, note="admin")
    assert ok is True
    pred = await store.get(pred_id)
    assert pred.status == STATUS_RESOLVED
    assert pred.verdict == VERDICT_UNCLEAR


@pytest.mark.asyncio
async def test_force_set_verdict_rejects_invalid_verdict(store):
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="x", deadline_utc=time.time() + 60,
    )
    with pytest.raises(ValueError):
        await store.force_set_verdict(pred_id, verdict="maybe", note="")


@pytest.mark.asyncio
async def test_revert_to_pending_clears_verdict(store):
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="x", deadline_utc=time.time() - 60,
    )
    await store.resolve(pred_id, verdict=VERDICT_RIGHT, note="auto", resolver_user_hash=None)

    ok = await store.revert_to_pending(pred_id)
    assert ok is True

    pred = await store.get(pred_id)
    assert pred.status == STATUS_PENDING
    assert pred.verdict is None
    assert pred.resolution_note is None
    assert pred.resolved_at is None


@pytest.mark.asyncio
async def test_revert_to_pending_makes_row_due_again(store):
    """After revert, list_due() picks up the row so the auto-resolver
    can re-judge it on the next sweep."""
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="x", deadline_utc=time.time() - 60,
    )
    await store.resolve(pred_id, verdict=VERDICT_WRONG, note="auto", resolver_user_hash=None)
    assert await store.list_due() == []  # resolved → not in due list

    await store.revert_to_pending(pred_id)
    due = await store.list_due()
    assert [p.id for p in due] == [pred_id]


@pytest.mark.asyncio
async def test_bot_and_human_predictions_are_separate_rows(store):
    """Bot author and human author must produce distinct leaderboard rows
    (different user_hash → no aggregation across them)."""
    bot_hash = hash_phone(BOT_SENDER)
    human_hash = hash_phone("+15555550100")

    deadline = time.time() + 86400
    bot_pred = await store.create(
        user_hash=bot_hash, user_label="Sigil", group_id="g1",
        context_key="group:g1", claim="bot claim", deadline_utc=deadline,
    )
    human_pred = await store.create(
        user_hash=human_hash, user_label="Alice", group_id="g1",
        context_key="group:g1", claim="alice claim", deadline_utc=deadline,
    )
    await store.resolve(bot_pred, verdict=VERDICT_RIGHT, note="ok", resolver_user_hash=None)
    await store.resolve(human_pred, verdict=VERDICT_RIGHT, note="ok", resolver_user_hash=None)

    rows = await store.leaderboard("group:g1", limit=10)
    hashes = {r["user_hash"] for r in rows}
    assert bot_hash in hashes
    assert human_hash in hashes
