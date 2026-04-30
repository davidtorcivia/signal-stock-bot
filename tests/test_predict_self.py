"""Tests for the bot-authored prediction path.

Covers:
  - extract_prediction's happy path + error returns (the public wrapper
    used by both !predict and the predict_self LLM tool)
  - PredictionStore accepting a bot-author row and surfacing it through
    the leaderboard with the stored bot label
"""

import datetime as dt
import time

import pytest

from src.commands.predict_command import (
    PREDICT_FOR_TOOL,
    PREDICT_SELF_TOOL,
    PREDICT_UPDATE_TOOL,
    _deadline_is_future_day,
    _has_explicit_future_year,
    _parse_deadline,
    extract_prediction,
)
from src.group_log import GroupMessageLog, BOT_SENDER as _BOT_SENDER_ALIAS
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


def test_deadline_is_future_day_rejects_same_et_day():
    """Same ET calendar day as now → False, regardless of how many hours
    away. Anchors the rule directly without depending on the parser."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    now_et = dt.datetime.now(et)
    # Pick noon-ET on today's date — same calendar day as now in ET.
    same_day_noon = dt.datetime(
        now_et.year, now_et.month, now_et.day, 12, 0, tzinfo=et,
    )
    assert _deadline_is_future_day(same_day_noon.timestamp()) is False
    # Last second of the ET day — still same day, still rejected.
    same_day_end = dt.datetime(
        now_et.year, now_et.month, now_et.day, 23, 59, 59, tzinfo=et,
    )
    assert _deadline_is_future_day(same_day_end.timestamp()) is False


def test_deadline_is_future_day_accepts_next_et_day():
    """Tomorrow-ET-00:01 is strictly later in ET — must pass."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    tomorrow = dt.datetime.now(et) + dt.timedelta(days=1)
    early_tomorrow = dt.datetime(
        tomorrow.year, tomorrow.month, tomorrow.day, 0, 1, tzinfo=et,
    )
    assert _deadline_is_future_day(early_tomorrow.timestamp()) is True


def test_deadline_is_future_day_handles_late_night_et_window():
    """The whole point of the rule: predicting tomorrow's close at 9pm
    tonight (ET) must work. Verifies a deadline crossing midnight ET
    is treated as a different calendar day."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    # Tomorrow 16:00 ET is always strictly after today's date in ET.
    tomorrow_close = dt.datetime.now(et).replace(
        hour=16, minute=0, second=0, microsecond=0,
    ) + dt.timedelta(days=1)
    assert _deadline_is_future_day(tomorrow_close.timestamp()) is True


@pytest.mark.asyncio
async def test_extract_prediction_accepts_next_day_deadline():
    """Tomorrow-ET is the earliest valid deadline. A 9pm-tonight prediction
    about tomorrow's close must still pass through extract_prediction."""
    from zoneinfo import ZoneInfo
    et = ZoneInfo("America/New_York")
    tomorrow_et = (dt.datetime.now(et) + dt.timedelta(days=1)).date()
    deadline_et = dt.datetime(
        tomorrow_et.year, tomorrow_et.month, tomorrow_et.day, 16, 0, tzinfo=et,
    )
    iso = deadline_et.astimezone(dt.timezone.utc).strftime("%Y-%m-%d %H:%M")
    parsed, err = await extract_prediction(
        f"AAPL above $1 by {iso} UTC", llm_client=None,
    )
    assert err is None, f"unexpected rejection: {err}"
    assert parsed is not None


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


# ---------- year-bump bug regression ----------------------------------------

def test_has_explicit_future_year_detects_4digit():
    assert _has_explicit_future_year("April 29 2027") is True
    assert _has_explicit_future_year("by 2027-05-15") is True


def test_has_explicit_future_year_detects_phrases():
    assert _has_explicit_future_year("April 29 next year") is True
    assert _has_explicit_future_year("in 2 years") is True
    assert _has_explicit_future_year("a year from now") is True


def test_has_explicit_future_year_misses_bare_dates():
    """Bare month-day strings have NO explicit year — that's the
    case the year-bump correction needs to catch."""
    assert _has_explicit_future_year("April 29") is False
    assert _has_explicit_future_year("by Friday") is False
    assert _has_explicit_future_year("by May 5") is False


def test_parse_deadline_does_not_jump_to_next_year_for_today():
    """The bug: dateparser's `PREFER_DATES_FROM: future` on the
    morning of April 29 (UTC) interpreted "April 29" as 2027 because
    today's 00:00 had passed. The correction backs the year off when
    no explicit year was given AND the result is way in the future."""
    today = dt.date.today()
    # Use a date string that names today's month + day with no year.
    text = today.strftime("%B %-d") if hasattr(dt.date, "strftime") else None
    if text is None:
        return
    ts = _parse_deadline(text)
    if ts is None:
        # Could legitimately be None if we're past 21:00 UTC today —
        # in that case there's no in-day window left.
        return
    parsed = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    # Must be within ~14 months — never bumped to next year.
    delta = parsed - dt.datetime.now(dt.timezone.utc)
    assert delta.days < 365, (
        f"deadline parsed as {parsed} ({delta.days} days out) — "
        f"year-bump bug hasn't been corrected"
    )


def test_parse_deadline_keeps_explicit_year():
    """If user explicitly says a future year, the correction must NOT
    undo it. 'April 29 2027' stays 2027."""
    ts = _parse_deadline("April 29 2027")
    if ts is None:
        # 2027 is ~1y out — should always parse
        raise AssertionError("'April 29 2027' should parse")
    parsed = dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    assert parsed.year == 2027


# ---------- predict_for: tool schema -----------------------------------------

def test_predict_for_and_predict_self_are_distinct_tools():
    """Each maps to a different intent and the LLM dispatches by name —
    if these collided we'd lose third-party attribution."""
    self_name = PREDICT_SELF_TOOL["function"]["name"]
    for_name = PREDICT_FOR_TOOL["function"]["name"]
    assert self_name == "predict_self"
    assert for_name == "predict_for"
    assert self_name != for_name


def test_predict_for_schema_requires_subject_and_claim():
    schema = PREDICT_FOR_TOOL["function"]["parameters"]
    required = set(schema["required"])
    assert required == {"subject", "claim"}


# ---------- predict_update: tool schema + store contract ---------------------

def test_predict_update_tool_requires_id_and_claim():
    schema = PREDICT_UPDATE_TOOL["function"]["parameters"]
    required = set(schema["required"])
    assert required == {"id", "claim"}


def test_predict_update_distinct_from_create_tools():
    names = {
        PREDICT_SELF_TOOL["function"]["name"],
        PREDICT_FOR_TOOL["function"]["name"],
        PREDICT_UPDATE_TOOL["function"]["name"],
    }
    assert names == {"predict_self", "predict_for", "predict_update"}


@pytest.mark.asyncio
async def test_update_pending_within_grace_window(store):
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="old", deadline_utc=time.time() + 86400,
        ticker="AAPL", threshold=200.0, direction="above",
    )
    new_deadline = time.time() + 7 * 86400
    status = await store.update_pending(
        pred_id,
        claim="new claim",
        deadline_utc=new_deadline,
        ticker="MSFT", threshold=400.0, direction="below",
    )
    assert status == "ok"
    pred = await store.get(pred_id)
    assert pred is not None
    assert pred.claim == "new claim"
    assert pred.deadline_utc == new_deadline
    assert pred.ticker == "MSFT"
    assert pred.threshold == 400.0
    assert pred.direction == "below"


@pytest.mark.asyncio
async def test_update_pending_rejects_stale(store):
    """Past the 15-minute grace window the row is locked. We backdate
    the prediction's created_at to simulate a row that's just out of
    range so the test isn't time-flaky."""
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="old", deadline_utc=time.time() + 86400,
    )
    # Backdate to 16 minutes ago — past the 15-min grace.
    import sqlite3
    with sqlite3.connect(store.db_path) as conn:
        conn.execute(
            "UPDATE predictions SET created_at = ? WHERE id = ?",
            (time.time() - 16 * 60, pred_id),
        )
        conn.commit()

    status = await store.update_pending(
        pred_id, claim="too late",
        deadline_utc=time.time() + 86400 * 14,
    )
    assert status == "stale"
    pred = await store.get(pred_id)
    assert pred.claim == "old"  # untouched


@pytest.mark.asyncio
async def test_update_pending_rejects_resolved(store):
    """Resolved predictions are locked even within grace — the verdict
    happened, can't rewrite history."""
    pred_id = await store.create(
        user_hash="h", user_label="X", group_id=None,
        context_key="dm:h", claim="old", deadline_utc=time.time() + 60,
    )
    await store.resolve(pred_id, verdict=VERDICT_RIGHT, note="auto", resolver_user_hash=None)

    status = await store.update_pending(
        pred_id, claim="new", deadline_utc=time.time() + 86400,
    )
    assert status == "not_pending"


# ---------- vote_to_resolve: consensus + admin path -------------------------

@pytest.mark.asyncio
async def test_vote_to_resolve_admin_resolves_solo(store):
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="admin-h", verdict=VERDICT_RIGHT,
        is_admin=True,
    )
    assert out["status"] == "resolved"
    assert out["by_admin"] is True
    pred = await store.get(pred_id)
    assert pred.status == "resolved"
    assert pred.verdict == VERDICT_RIGHT


@pytest.mark.asyncio
async def test_vote_to_resolve_single_voter_not_enough(store):
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_RIGHT,
    )
    assert out["status"] == "voted"
    assert out["agreeing_count"] == 1
    pred = await store.get(pred_id)
    assert pred.status == "pending"


@pytest.mark.asyncio
async def test_vote_to_resolve_two_agreeing_voters_resolves(store):
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_WRONG,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="b-h", verdict=VERDICT_WRONG,
    )
    assert out["status"] == "resolved"
    assert out["agreeing_count"] == 2
    pred = await store.get(pred_id)
    assert pred.status == "resolved"
    assert pred.verdict == VERDICT_WRONG


@pytest.mark.asyncio
async def test_vote_to_resolve_split_vote_does_not_resolve(store):
    """Disagreement keeps the prediction pending."""
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_RIGHT,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="b-h", verdict=VERDICT_WRONG,
    )
    assert out["status"] == "voted"
    pred = await store.get(pred_id)
    assert pred.status == "pending"


@pytest.mark.asyncio
async def test_vote_to_resolve_self_vote_rejected(store):
    """Predictor can't vote on their own prediction."""
    pred_id = await store.create(
        user_hash="me-h", user_label="Me", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="me-h", verdict=VERDICT_RIGHT,
    )
    assert out["status"] == "self_vote"
    pred = await store.get(pred_id)
    assert pred.status == "pending"


@pytest.mark.asyncio
async def test_vote_to_resolve_admin_can_self_vote(store):
    """An admin who happens to be the predictor still resolves solo —
    they have authority and the path is intentional."""
    pred_id = await store.create(
        user_hash="admin-h", user_label="A", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="admin-h", verdict=VERDICT_RIGHT,
        is_admin=True,
    )
    assert out["status"] == "resolved"


@pytest.mark.asyncio
async def test_vote_to_resolve_voter_changes_mind(store):
    """A voter can revise their verdict — the new verdict replaces
    the old, and vote count reflects only their current choice."""
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_RIGHT,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_WRONG,
    )
    # After replacement, only 1 voter total (with WRONG verdict) — still
    # below threshold.
    assert out["status"] == "voted"
    assert out["agreeing_count"] == 1


@pytest.mark.asyncio
async def test_vote_to_resolve_already_resolved_rejected(store):
    pred_id = await store.create(
        user_hash="predictor-h", user_label="P", group_id="g1",
        context_key="group:g1", claim="x", deadline_utc=time.time() + 60,
    )
    await store.resolve(
        pred_id, verdict=VERDICT_RIGHT, note="auto", resolver_user_hash=None,
    )
    out = await store.vote_to_resolve(
        pred_id, voter_user_hash="a-h", verdict=VERDICT_WRONG,
    )
    assert out["status"] == "not_pending"


@pytest.mark.asyncio
async def test_update_pending_returns_not_found_for_bad_id(store):
    status = await store.update_pending(
        99999, claim="x", deadline_utc=time.time() + 86400,
    )
    assert status == "not_found"


# ---------- group_log: tail lookup (used by predict_for) ---------------------

@pytest.fixture
def gl(tmp_path):
    return GroupMessageLog(db_path=str(tmp_path / "group.db"))


@pytest.mark.asyncio
async def test_find_sender_by_tail_returns_most_recent_match(gl):
    """A user can change phones (rare) but the most recent sender with
    a given tail is the one the LLM was just looking at in the chat."""
    await gl.append("g1", "+15551114810", "older message")
    await gl.append("g1", "+15552224810", "newer message")
    found = await gl.find_recent_sender_by_tail("g1", "4810")
    assert found == "+15552224810"


@pytest.mark.asyncio
async def test_find_sender_by_tail_skips_bot_sender(gl):
    """The bot's own posts share the BOT_SENDER sentinel — never use
    that as a predict_for target. Otherwise the bot could end up
    predicting on its own behalf via this code path."""
    await gl.append("g1", "+15551114810", "human message")
    await gl.append_bot("g1", "bot reply")
    # Hypothetical sentinel-tail collision: the bot's tail is "_bot__"
    # so it would never match a 4-digit numeric tail in practice, but
    # the lookup excludes BOT_SENDER explicitly regardless. Test the
    # exclusion by querying the sentinel's own tail substring.
    found = await gl.find_recent_sender_by_tail(
        "g1", _BOT_SENDER_ALIAS[-4:],
    )
    assert found is None


@pytest.mark.asyncio
async def test_find_sender_by_tail_scoped_to_group(gl):
    await gl.append("g1", "+15551114810", "in g1")
    await gl.append("g2", "+15552224810", "in g2")
    assert await gl.find_recent_sender_by_tail("g1", "4810") == "+15551114810"
    assert await gl.find_recent_sender_by_tail("g2", "4810") == "+15552224810"


@pytest.mark.asyncio
async def test_find_sender_by_tail_no_match_returns_none(gl):
    await gl.append("g1", "+15551112222", "msg")
    assert await gl.find_recent_sender_by_tail("g1", "4810") is None


@pytest.mark.asyncio
async def test_find_sender_by_tail_handles_empty_inputs(gl):
    assert await gl.find_recent_sender_by_tail("", "4810") is None
    assert await gl.find_recent_sender_by_tail("g1", "") is None


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
