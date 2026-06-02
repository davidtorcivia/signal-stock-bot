"""Tests for per-context oracle storage, schedule math, and prepopulation.

The interesting properties to pin:
  - `next_fire_time` returns today vs. tomorrow correctly across the
    boundary, respecting weekdays_only.
  - `default_oracles_for_label` matches finance/woo keywords without
    eating unrelated labels.
  - `OracleStore` round-trips every field cleanly and validation
    rejects bad inputs.
  - `prepopulate_for_existing_contexts` is idempotent and honors the
    legacy global setting.
"""

import datetime as dt
from zoneinfo import ZoneInfo

import pytest

from src.contexts.oracles import (
    ContextOracle,
    OracleStore,
    SCHEDULE_KINDS,
    KINDS,
    default_oracles_for_label,
    fire_time_for,
    next_fire_time,
    prepopulate_for_existing_contexts,
)


_NY = ZoneInfo("America/New_York")
_UTC = dt.timezone.utc


# ---------- ContextOracle.validate -----------------------------------------

def test_validate_rejects_bad_kind():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="palantir",
        schedule_kind="sunrise",
    )
    errs = o.validate()
    assert any("kind" in e for e in errs)


def test_validate_rejects_bad_schedule():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="midnight",
    )
    errs = o.validate()
    assert any("schedule_kind" in e for e in errs)


def test_validate_clock_requires_clock_time():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="clock", clock_time=None,
    )
    assert any("clock_time" in e for e in o.validate())


def test_validate_clock_accepts_valid_time():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="clock", clock_time="09:25",
    )
    assert o.validate() == []


def test_validate_clock_rejects_out_of_range():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="clock", clock_time="25:99",
    )
    assert o.validate()


def test_validate_freeform_requires_prompt():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="freeform",
        schedule_kind="sunrise", prompt=None,
    )
    assert any("prompt" in e for e in o.validate())


def test_validate_rejects_unknown_timezone():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise", timezone="Mars/Olympus_Mons",
    )
    assert any("timezone" in e for e in o.validate())


def test_validate_rejects_out_of_range_days():
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise", days_of_week=frozenset({0, 7}),
    )
    assert any("days_of_week" in e for e in o.validate())


# ---------- effective_days / days_label -------------------------------------

def test_effective_days_falls_back_to_weekdays_only():
    o = ContextOracle(id=None, context_id=1, enabled=True, kind="tarot",
                      schedule_kind="sunrise", weekdays_only=True)
    assert o.effective_days() == frozenset({0, 1, 2, 3, 4})


def test_effective_days_none_means_every_day():
    o = ContextOracle(id=None, context_id=1, enabled=True, kind="tarot",
                      schedule_kind="sunrise")
    assert o.effective_days() is None


def test_empty_days_of_week_normalizes_to_none():
    """An explicit empty set must collapse to None (every day), not fall through
    to the weekdays_only branch."""
    o = ContextOracle(id=None, context_id=1, enabled=True, kind="tarot",
                      schedule_kind="sunrise", weekdays_only=True,
                      days_of_week=frozenset())
    assert o.days_of_week is None
    assert o.effective_days() == frozenset({0, 1, 2, 3, 4})  # back to the flag


def test_days_of_week_wins_over_weekdays_only():
    o = ContextOracle(id=None, context_id=1, enabled=True, kind="tarot",
                      schedule_kind="sunrise", weekdays_only=True,
                      days_of_week=frozenset({6, 0, 1, 2, 3}))
    assert o.effective_days() == frozenset({6, 0, 1, 2, 3})


def test_days_label_presets_and_list():
    def lbl(days=None, wd=False):
        return ContextOracle(id=None, context_id=1, enabled=True, kind="tarot",
                             schedule_kind="sunrise", weekdays_only=wd,
                             days_of_week=days).days_label()
    assert lbl() == "every day"
    assert lbl(frozenset(range(7))) == "every day"
    assert lbl(wd=True) == "Mon-Fri"
    assert lbl(frozenset({6, 0, 1, 2, 3})) == "Sun-Thu"
    assert lbl(frozenset({5, 6})) == "weekends"
    assert lbl(frozenset({0, 2, 4})) == "Mon, Wed, Fri"


# ---------- fire_time_for + next_fire_time ----------------------------------

def _sunrise_oracle(offset_minutes=0, weekdays_only=False) -> ContextOracle:
    return ContextOracle(
        id=1, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise", offset_minutes=offset_minutes,
        weekdays_only=weekdays_only,
    )


def _clock_oracle(clock_time="09:25", weekdays_only=False, tz="America/New_York") -> ContextOracle:
    return ContextOracle(
        id=1, context_id=1, enabled=True, kind="market_open",
        schedule_kind="clock", clock_time=clock_time,
        timezone=tz, weekdays_only=weekdays_only,
    )


def test_fire_time_clock_returns_local_tz():
    o = _clock_oracle(clock_time="09:25", tz="America/New_York")
    fire = fire_time_for(o, dt.date(2026, 4, 28))
    assert fire.hour == 9 and fire.minute == 25
    assert fire.tzinfo == ZoneInfo("America/New_York")


def test_fire_time_sunrise_offset_applies():
    """A +30min offset shifts the firing instant 30 minutes after
    sunrise."""
    base = fire_time_for(_sunrise_oracle(0), dt.date(2026, 4, 28))
    plus30 = fire_time_for(_sunrise_oracle(30), dt.date(2026, 4, 28))
    assert (plus30 - base).total_seconds() == 30 * 60


def test_next_fire_time_today_when_before_window():
    """Clock=09:25 at 03:00am same day → today's 09:25."""
    o = _clock_oracle(clock_time="09:25")
    now = dt.datetime(2026, 4, 28, 3, 0, tzinfo=_NY).astimezone(_UTC)
    target = next_fire_time(o, now)
    target_local = target.astimezone(_NY)
    assert target_local.date() == dt.date(2026, 4, 28)
    assert target_local.hour == 9 and target_local.minute == 25


def test_next_fire_time_tomorrow_when_after_window():
    """Clock=09:25 at 18:00 same day → tomorrow's 09:25."""
    o = _clock_oracle(clock_time="09:25")
    now = dt.datetime(2026, 4, 28, 18, 0, tzinfo=_NY).astimezone(_UTC)
    target = next_fire_time(o, now).astimezone(_NY)
    assert target.date() == dt.date(2026, 4, 29)


def test_next_fire_time_skips_weekends_when_weekdays_only():
    """Friday 18:00 with weekdays_only → Monday's window, not Saturday."""
    # Friday May 1 2026 at 6pm
    o = _clock_oracle(clock_time="09:25", weekdays_only=True)
    fri_evening = dt.datetime(2026, 5, 1, 18, 0, tzinfo=_NY).astimezone(_UTC)
    target = next_fire_time(o, fri_evening).astimezone(_NY)
    # Mon May 4 = next weekday
    assert target.weekday() < 5
    assert target.date() >= dt.date(2026, 5, 4)


def test_next_fire_time_days_of_week_sun_thu_skips_friday_evening():
    """Sun-Thu schedule on Friday 18:00 → Sunday's window (skips Fri leftover
    and Saturday)."""
    o = ContextOracle(
        id=1, context_id=1, enabled=True, kind="wsb_digest",
        schedule_kind="clock", clock_time="20:00", timezone="America/New_York",
        days_of_week=frozenset({6, 0, 1, 2, 3}),
    )
    fri_evening = dt.datetime(2026, 5, 1, 21, 0, tzinfo=_NY).astimezone(_UTC)
    target = next_fire_time(o, fri_evening).astimezone(_NY)
    assert target.date() == dt.date(2026, 5, 3)  # Sunday
    assert target.weekday() == 6


def test_next_fire_time_single_day_jumps_a_week():
    """A Sunday-only oracle on Monday → the coming Sunday, within the window."""
    o = ContextOracle(
        id=1, context_id=1, enabled=True, kind="tarot",
        schedule_kind="clock", clock_time="09:25", timezone="America/New_York",
        days_of_week=frozenset({6}),
    )
    monday = dt.datetime(2026, 5, 4, 10, 0, tzinfo=_NY).astimezone(_UTC)
    target = next_fire_time(o, monday).astimezone(_NY)
    assert target.date() == dt.date(2026, 5, 10)  # next Sunday
    assert target.weekday() == 6


def test_next_fire_time_returns_utc():
    """Always tz-aware UTC for clean delta math at the worker."""
    o = _clock_oracle()
    now = dt.datetime.now(_UTC)
    assert next_fire_time(o, now).tzinfo == _UTC


# ---------- default_oracles_for_label ---------------------------------------

def test_default_oracles_finance_label_yields_market_recaps_and_wsb():
    out = default_oracles_for_label("Money Chat", context_id=42)
    by_kind = {o.kind: o for o in out}
    assert sorted(by_kind) == ["market_close", "market_open", "wsb_digest"]
    # market recaps run weekdays; wsb defaults to Sun-Thu (no Fri/Sat read)
    assert by_kind["market_open"].weekdays_only
    assert by_kind["market_close"].weekdays_only
    assert by_kind["wsb_digest"].effective_days() == frozenset({6, 0, 1, 2, 3})
    assert all(not o.enabled for o in out)  # admin opts in


def test_default_oracles_woo_label_yields_sunrise_tarot():
    out = default_oracles_for_label("woo chat", context_id=42)
    assert len(out) == 1
    assert out[0].kind == "tarot"
    assert out[0].schedule_kind == "sunrise"
    assert out[0].offset_minutes == 1
    assert not out[0].enabled


def test_default_oracles_unrelated_label_yields_nothing():
    """Random labels don't trip a heuristic — admin starts clean."""
    assert default_oracles_for_label("dinner planning", context_id=1) == []
    assert default_oracles_for_label("", context_id=1) == []


def test_default_oracles_does_not_match_substring_false_positives():
    """'stockholm' shouldn't match the 'stock' keyword. We use specific
    multi-word keys ('stocks', 'stock chat') to avoid this."""
    out = default_oracles_for_label("stockholm trip", context_id=1)
    # If any false-positive occurs, the test makes it visible — adjust
    # keyword list. Currently this returns empty.
    assert out == []


# ---------- OracleStore (round-trip) ----------------------------------------

@pytest.fixture
def store(tmp_path):
    return OracleStore(db_path=str(tmp_path / "oracles.db"))


@pytest.mark.asyncio
async def test_store_round_trip_preserves_every_field(store):
    o = ContextOracle(
        id=None, context_id=7, enabled=True, kind="freeform",
        schedule_kind="clock", offset_minutes=15, clock_time="06:30",
        timezone="Europe/Berlin", weekdays_only=True,
        prompt="What's the daily theme?", label="berlin morning",
    )
    oracle_id = await store.upsert(o)
    fetched = await store.get(oracle_id)
    assert fetched is not None
    assert fetched.context_id == 7
    assert fetched.enabled is True
    assert fetched.kind == "freeform"
    assert fetched.schedule_kind == "clock"
    assert fetched.offset_minutes == 15
    assert fetched.clock_time == "06:30"
    assert fetched.timezone == "Europe/Berlin"
    assert fetched.weekdays_only is True
    assert fetched.prompt == "What's the daily theme?"
    assert fetched.label == "berlin morning"


@pytest.mark.asyncio
async def test_store_migrates_legacy_schema_without_days_column(tmp_path):
    """A DB created before the picker (no days_of_week column) must gain the
    column on first use, read legacy rows via the weekdays_only fallback, and
    accept new days_of_week writes."""
    import aiosqlite

    db_path = str(tmp_path / "legacy.db")
    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """CREATE TABLE context_oracles (
                   id INTEGER PRIMARY KEY AUTOINCREMENT,
                   context_id INTEGER NOT NULL, enabled INTEGER NOT NULL DEFAULT 0,
                   kind TEXT NOT NULL, schedule_kind TEXT NOT NULL,
                   offset_minutes INTEGER NOT NULL DEFAULT 0, clock_time TEXT,
                   timezone TEXT NOT NULL DEFAULT 'America/New_York',
                   weekdays_only INTEGER NOT NULL DEFAULT 0, prompt TEXT,
                   label TEXT NOT NULL DEFAULT '', last_fired_at REAL,
                   created_at REAL NOT NULL, updated_at REAL NOT NULL )"""
        )
        await db.execute(
            "INSERT INTO context_oracles (context_id, enabled, kind, schedule_kind,"
            " clock_time, weekdays_only, created_at, updated_at) "
            "VALUES (5, 1, 'market_open', 'clock', '09:25', 1, 0, 0)"
        )
        await db.commit()

    store = OracleStore(db_path=db_path)
    rows = await store.list_for_context(5)  # triggers the ALTER migration
    assert len(rows) == 1 and rows[0].days_of_week is None
    assert rows[0].effective_days() == frozenset({0, 1, 2, 3, 4})  # legacy fallback
    # the migrated table now accepts a days_of_week write
    rows[0].days_of_week = frozenset({6, 0, 1, 2, 3})
    await store.upsert(rows[0])
    assert (await store.get(rows[0].id)).days_of_week == frozenset({6, 0, 1, 2, 3})


@pytest.mark.asyncio
async def test_store_round_trips_days_of_week(store):
    o = ContextOracle(
        id=None, context_id=3, enabled=True, kind="wsb_digest",
        schedule_kind="clock", clock_time="20:00",
        days_of_week=frozenset({6, 0, 1, 2, 3}),
    )
    oid = await store.upsert(o)
    fetched = await store.get(oid)
    assert fetched.days_of_week == frozenset({6, 0, 1, 2, 3})
    assert fetched.effective_days() == frozenset({6, 0, 1, 2, 3})
    # updating to a different set persists
    fetched.days_of_week = frozenset({0, 2, 4})
    await store.upsert(fetched)
    again = await store.get(oid)
    assert again.days_of_week == frozenset({0, 2, 4})
    # clearing to None (every day) round-trips as NULL
    again.days_of_week = None
    await store.upsert(again)
    cleared = await store.get(oid)
    assert cleared.days_of_week is None


@pytest.mark.asyncio
async def test_list_enabled_filters_disabled(store):
    on = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise",
    )
    off = ContextOracle(
        id=None, context_id=1, enabled=False, kind="iching",
        schedule_kind="sunrise",
    )
    await store.upsert(on)
    await store.upsert(off)
    enabled = await store.list_enabled()
    assert {o.kind for o in enabled} == {"tarot"}


@pytest.mark.asyncio
async def test_list_for_context_scopes_by_context(store):
    a = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise",
    )
    b = ContextOracle(
        id=None, context_id=2, enabled=True, kind="iching",
        schedule_kind="sunrise",
    )
    await store.upsert(a)
    await store.upsert(b)
    only_one = await store.list_for_context(1)
    assert len(only_one) == 1
    assert only_one[0].kind == "tarot"


@pytest.mark.asyncio
async def test_upsert_validates(store):
    bad = ContextOracle(
        id=None, context_id=1, enabled=True, kind="palantir",
        schedule_kind="sunrise",
    )
    with pytest.raises(ValueError):
        await store.upsert(bad)


@pytest.mark.asyncio
async def test_mark_fired_updates_last_fired_at(store):
    o = ContextOracle(
        id=None, context_id=1, enabled=True, kind="tarot",
        schedule_kind="sunrise",
    )
    oracle_id = await store.upsert(o)
    await store.mark_fired(oracle_id, 1234567890.5)
    fetched = await store.get(oracle_id)
    assert fetched.last_fired_at == 1234567890.5


@pytest.mark.asyncio
async def test_delete_for_context_removes_all_rows(store):
    for kind in ("tarot", "iching"):
        await store.upsert(ContextOracle(
            id=None, context_id=99, enabled=True, kind=kind,
            schedule_kind="sunrise",
        ))
    n = await store.delete_for_context(99)
    assert n == 2
    assert await store.list_for_context(99) == []


# ---------- prepopulation ---------------------------------------------------

class _FakePolicy:
    def __init__(self, id, kind, key, label):
        self.id = id
        self.kind = kind
        self.key = key
        self.label = label


class _FakeRegistry:
    def __init__(self, policies):
        self._policies = policies

    async def list(self):
        return list(self._policies)


class _FakeSettingsStore:
    def __init__(self, **kv):
        self._kv = kv

    def get(self, key, default=None):
        return self._kv.get(key, default)


@pytest.mark.asyncio
async def test_prepopulate_seeds_only_group_contexts(store):
    """DM and default rows shouldn't get oracles auto-created."""
    registry = _FakeRegistry([
        _FakePolicy(1, "default", "default:group", "Default (groups)"),
        _FakePolicy(2, "dm", "abc", "Some DM"),
        _FakePolicy(3, "group", "money-group-id", "Money Chat"),
        _FakePolicy(4, "group", "woo-group-id", "Woo Chat"),
    ])
    n = await prepopulate_for_existing_contexts(store, registry)
    # Money chat → 3 oracles (market open/close + wsb digest), woo chat → 1
    assert n == 4
    assert len(await store.list_for_context(1)) == 0
    assert len(await store.list_for_context(2)) == 0
    assert len(await store.list_for_context(3)) == 3
    assert len(await store.list_for_context(4)) == 1


@pytest.mark.asyncio
async def test_prepopulate_is_idempotent(store):
    """Running twice doesn't double-populate — contexts with any
    existing oracle are skipped."""
    registry = _FakeRegistry([
        _FakePolicy(1, "group", "woo-id", "woo chat"),
    ])
    first = await prepopulate_for_existing_contexts(store, registry)
    second = await prepopulate_for_existing_contexts(store, registry)
    assert first == 1
    assert second == 0


@pytest.mark.asyncio
async def test_prepopulate_carries_over_legacy_global_setting(store):
    """If old daily_oracle_enabled was true and the group_id matches,
    the prepopulated tarot oracle for that context lands enabled."""
    registry = _FakeRegistry([
        _FakePolicy(1, "group", "woo-key", "woo chat"),
        _FakePolicy(2, "group", "other-key", "Random Chat"),
    ])
    settings = _FakeSettingsStore(
        daily_oracle_enabled=True,
        daily_oracle_group_id="woo-key",
    )
    await prepopulate_for_existing_contexts(store, registry, settings_store=settings)
    woo_oracles = await store.list_for_context(1)
    other_oracles = await store.list_for_context(2)
    # woo got a tarot, enabled=True by carry-over
    assert any(o.kind == "tarot" and o.enabled for o in woo_oracles)
    # other chat (no woo keywords) gets nothing
    assert other_oracles == []


@pytest.mark.asyncio
async def test_prepopulate_without_legacy_settings_leaves_disabled(store):
    """Without the carry-over signal, defaults stay disabled — admin
    reviews before posts fire."""
    registry = _FakeRegistry([
        _FakePolicy(1, "group", "woo-key", "woo chat"),
    ])
    await prepopulate_for_existing_contexts(store, registry, settings_store=None)
    oracles = await store.list_for_context(1)
    assert all(not o.enabled for o in oracles)
