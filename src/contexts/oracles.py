"""
Per-context daily oracle configuration.

Each context (group) can have zero or more oracles. An oracle fires
once per day at a configured time and posts something to the chat:
either a tarot/iching draw (existing commands) or a Sigil-narrated
recap (market_open / market_close / freeform — runs through
ask_command's tool loop so Sigil can fetch live data first).

Schedule kinds:
  - sunrise: NYC-anchored civil sunrise + offset_minutes (signed)
  - sunset:  NYC-anchored civil sunset + offset_minutes (signed)
  - clock:   fixed HH:MM in the oracle's timezone

Idempotency: `last_fired_at` is bumped after each post so a bot
restart that lands inside the firing window doesn't double-post.

Schema lives in the same SQLite database as everything else; the
table is upserted by `OracleStore._ensure_initialized()` on first
use, no separate migration step required.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import aiosqlite

from ..database import db_session
from astral import LocationInfo
from astral.sun import sun

logger = logging.getLogger(__name__)


KINDS = (
    "tarot",
    "iching",
    "market_open",
    "market_close",
    "freeform",
)
SCHEDULE_KINDS = ("sunrise", "sunset", "clock")

# NYC anchors all sunrise/sunset calculations. The chat-side timezone
# is configurable per oracle (used only for `clock` schedules) — the
# astronomical events themselves are observer-dependent and we pick
# one canonical observer for consistency.
_LOCATION = LocationInfo(
    name="New York",
    region="USA",
    timezone="America/New_York",
    latitude=40.7128,
    longitude=-74.0060,
)


KIND_DESCRIPTIONS = {
    "tarot": (
        "Single-card tarot draw with the rendered spread image — the "
        "same output as `!tarot` but headlined as the day's oracle "
        "from Sigil."
    ),
    "iching": (
        "Three-coin I Ching cast with the hexagram image — same as "
        "`!iching`, headlined as the morning's reading."
    ),
    "market_open": (
        "Pre-market check. Sigil pulls major-index futures, key "
        "events for the day, and writes a short opener. Best run a "
        "few minutes before US cash open (09:25 ET) on weekdays."
    ),
    "market_close": (
        "Close-of-day recap. Sigil pulls the day's index moves, "
        "leaders/laggards, and any notable news. Best run shortly "
        "after the 4pm ET close on weekdays."
    ),
    "freeform": (
        "Sigil generates a post from a custom prompt you supply. "
        "Use for theme-specific morning notes (e.g. \"Today's "
        "moon phase + a one-line ritual suggestion\")."
    ),
}

SCHEDULE_DESCRIPTIONS = {
    "sunrise": (
        "Fires relative to NYC civil sunrise. Offset is in minutes "
        "(positive = after sunrise, negative = before)."
    ),
    "sunset": (
        "Fires relative to NYC civil sunset. Offset is in minutes "
        "(positive = after sunset, negative = before)."
    ),
    "clock": (
        "Fires at a fixed HH:MM clock time in the oracle's "
        "timezone (default America/New_York)."
    ),
}


@dataclass
class ContextOracle:
    id: Optional[int]
    context_id: int
    enabled: bool
    kind: str                            # one of KINDS
    schedule_kind: str                   # one of SCHEDULE_KINDS
    offset_minutes: int = 0              # for sunrise/sunset
    clock_time: Optional[str] = None     # "HH:MM" for clock
    timezone: str = "America/New_York"   # tz for clock + display
    weekdays_only: bool = False
    prompt: Optional[str] = None         # for freeform
    label: str = ""
    last_fired_at: Optional[float] = None

    def validate(self) -> list[str]:
        errs: list[str] = []
        if self.kind not in KINDS:
            errs.append(f"kind must be one of {KINDS}")
        if self.schedule_kind not in SCHEDULE_KINDS:
            errs.append(f"schedule_kind must be one of {SCHEDULE_KINDS}")
        if self.schedule_kind == "clock":
            if not self._valid_clock_time(self.clock_time):
                errs.append("clock_time must be HH:MM (24h) when schedule is clock")
        if self.kind == "freeform" and not (self.prompt or "").strip():
            errs.append("prompt is required when kind is freeform")
        try:
            ZoneInfo(self.timezone)
        except Exception:
            errs.append(f"timezone {self.timezone!r} is not a valid IANA tz")
        return errs

    @staticmethod
    def _valid_clock_time(s: Optional[str]) -> bool:
        if not s:
            return False
        try:
            hh, mm = s.split(":")
            return 0 <= int(hh) <= 23 and 0 <= int(mm) <= 59
        except (ValueError, AttributeError):
            return False

    def describe(self) -> str:
        """Human-readable schedule string for admin UI / logs."""
        if self.schedule_kind == "clock":
            return f"{self.clock_time} {self.timezone}"
        sign = "+" if self.offset_minutes >= 0 else ""
        return f"{self.schedule_kind}{sign}{self.offset_minutes}m (NYC)"


# ---------- schedule math ----------------------------------------------------

def _astro_event_for(date: dt.date, kind: str) -> dt.datetime:
    """Sunrise or sunset on `date` in NYC, returned as a tz-aware
    datetime in NYC local time."""
    s = sun(_LOCATION.observer, date=date)
    event = s["sunrise"] if kind == "sunrise" else s["sunset"]
    return event.astimezone(_LOCATION.tzinfo)


def _is_weekday(dt_local: dt.datetime) -> bool:
    """Mon=0..Sun=6 — exclude Sat/Sun for weekdays_only oracles. US
    holidays aren't filtered (would need a calendar dependency); the
    market_recap oracles will just post a quiet day on holidays."""
    return dt_local.weekday() < 5


def fire_time_for(oracle: ContextOracle, date: dt.date) -> dt.datetime:
    """Compute the firing instant for `oracle` on `date`. Returns a
    tz-aware datetime in the oracle's local timezone (or NYC for
    sunrise/sunset, since those are observer-anchored)."""
    if oracle.schedule_kind in ("sunrise", "sunset"):
        base = _astro_event_for(date, oracle.schedule_kind)
        return base + dt.timedelta(minutes=oracle.offset_minutes)
    # clock schedule
    tz = ZoneInfo(oracle.timezone)
    hh, mm = oracle.clock_time.split(":")  # validated upstream
    return dt.datetime(
        date.year, date.month, date.day, int(hh), int(mm), tzinfo=tz
    )


def next_fire_time(
    oracle: ContextOracle, now: Optional[dt.datetime] = None
) -> dt.datetime:
    """Soonest firing instant strictly after `now`, respecting
    weekdays_only. Always returns a tz-aware UTC datetime so the
    caller can take naive `(target - now)` deltas without surprise."""
    # Anchor "today" in the oracle's local frame so a clock=00:30 oracle
    # in Tokyo doesn't compute against UTC midnight and miss the day.
    local_tz = (
        ZoneInfo(oracle.timezone)
        if oracle.schedule_kind == "clock"
        else _LOCATION.tzinfo
    )
    if now is None:
        now = dt.datetime.now(dt.timezone.utc)
    now_local = now.astimezone(local_tz)

    # Look at today and the next 8 days (covers weekdays_only with two
    # weekend skips). Stop at the first valid future fire time.
    for delta in range(0, 9):
        candidate_date = (now_local + dt.timedelta(days=delta)).date()
        fire = fire_time_for(oracle, candidate_date)
        if fire <= now_local:
            continue
        if oracle.weekdays_only and not _is_weekday(fire):
            continue
        return fire.astimezone(dt.timezone.utc)

    # Should be unreachable — 8-day window covers every weekly schedule
    raise RuntimeError(
        f"Could not find next fire time for oracle #{oracle.id} within 8d"
    )


# ---------- store ------------------------------------------------------------

class OracleStore:
    """SQLite-backed CRUD for per-context oracles."""

    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self._initialized = False
        self._init_lock = asyncio.Lock()

    async def _ensure_initialized(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            async with aiosqlite.connect(self.db_path) as db:
                from ..database import apply_db_pragmas
                await apply_db_pragmas(db)
                await db.execute(
                    """
                    CREATE TABLE IF NOT EXISTS context_oracles (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        context_id INTEGER NOT NULL,
                        enabled INTEGER NOT NULL DEFAULT 0,
                        kind TEXT NOT NULL,
                        schedule_kind TEXT NOT NULL,
                        offset_minutes INTEGER NOT NULL DEFAULT 0,
                        clock_time TEXT,
                        timezone TEXT NOT NULL DEFAULT 'America/New_York',
                        weekdays_only INTEGER NOT NULL DEFAULT 0,
                        prompt TEXT,
                        label TEXT NOT NULL DEFAULT '',
                        last_fired_at REAL,
                        created_at REAL NOT NULL,
                        updated_at REAL NOT NULL
                    )
                    """
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_oracle_context "
                    "ON context_oracles(context_id)"
                )
                await db.execute(
                    "CREATE INDEX IF NOT EXISTS idx_oracle_enabled "
                    "ON context_oracles(enabled, context_id)"
                )
                await db.commit()
            self._initialized = True

    _COLS = (
        "id, context_id, enabled, kind, schedule_kind, offset_minutes, "
        "clock_time, timezone, weekdays_only, prompt, label, last_fired_at"
    )

    @staticmethod
    def _row_to_oracle(row) -> ContextOracle:
        return ContextOracle(
            id=row[0],
            context_id=row[1],
            enabled=bool(row[2]),
            kind=row[3],
            schedule_kind=row[4],
            offset_minutes=row[5],
            clock_time=row[6],
            timezone=row[7],
            weekdays_only=bool(row[8]),
            prompt=row[9],
            label=row[10] or "",
            last_fired_at=row[11],
        )

    async def list_for_context(self, context_id: int) -> list[ContextOracle]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._COLS} FROM context_oracles "
                f"WHERE context_id = ? ORDER BY id",
                (context_id,),
            )
            rows = await cursor.fetchall()
        return [self._row_to_oracle(r) for r in rows]

    async def list_enabled(self) -> list[ContextOracle]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._COLS} FROM context_oracles "
                f"WHERE enabled = 1 ORDER BY id"
            )
            rows = await cursor.fetchall()
        return [self._row_to_oracle(r) for r in rows]

    async def get(self, oracle_id: int) -> Optional[ContextOracle]:
        async with db_session(self) as db:
            cursor = await db.execute(
                f"SELECT {self._COLS} FROM context_oracles WHERE id = ?",
                (oracle_id,),
            )
            row = await cursor.fetchone()
        return self._row_to_oracle(row) if row else None

    async def upsert(self, oracle: ContextOracle) -> int:
        errs = oracle.validate()
        if errs:
            raise ValueError("; ".join(errs))
        await self._ensure_initialized()
        now = time.time()
        async with aiosqlite.connect(self.db_path) as db:
            if oracle.id is None:
                cursor = await db.execute(
                    """INSERT INTO context_oracles
                       (context_id, enabled, kind, schedule_kind,
                        offset_minutes, clock_time, timezone,
                        weekdays_only, prompt, label, last_fired_at,
                        created_at, updated_at)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        oracle.context_id,
                        1 if oracle.enabled else 0,
                        oracle.kind,
                        oracle.schedule_kind,
                        oracle.offset_minutes,
                        oracle.clock_time,
                        oracle.timezone,
                        1 if oracle.weekdays_only else 0,
                        oracle.prompt,
                        oracle.label,
                        oracle.last_fired_at,
                        now,
                        now,
                    ),
                )
                await db.commit()
                return cursor.lastrowid or 0
            await db.execute(
                """UPDATE context_oracles SET
                       enabled = ?, kind = ?, schedule_kind = ?,
                       offset_minutes = ?, clock_time = ?, timezone = ?,
                       weekdays_only = ?, prompt = ?, label = ?,
                       updated_at = ?
                   WHERE id = ?""",
                (
                    1 if oracle.enabled else 0,
                    oracle.kind,
                    oracle.schedule_kind,
                    oracle.offset_minutes,
                    oracle.clock_time,
                    oracle.timezone,
                    1 if oracle.weekdays_only else 0,
                    oracle.prompt,
                    oracle.label,
                    now,
                    oracle.id,
                ),
            )
            await db.commit()
            return oracle.id

    async def delete(self, oracle_id: int) -> bool:
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM context_oracles WHERE id = ?", (oracle_id,)
            )
            await db.commit()
            return cursor.rowcount > 0

    async def delete_for_context(self, context_id: int) -> int:
        """Cleanup hook for ContextRegistry.delete — drops orphan rows."""
        async with db_session(self) as db:
            cursor = await db.execute(
                "DELETE FROM context_oracles WHERE context_id = ?",
                (context_id,),
            )
            await db.commit()
            return cursor.rowcount

    async def mark_fired(self, oracle_id: int, fired_at: float) -> None:
        async with db_session(self) as db:
            await db.execute(
                "UPDATE context_oracles SET last_fired_at = ?, updated_at = ? "
                "WHERE id = ?",
                (fired_at, time.time(), oracle_id),
            )
            await db.commit()


# ---------- heuristic prepopulation ------------------------------------------

# Keyword → suggested oracles. Lowercased substring match against the
# context label. Keep these specific enough that an unrelated label
# doesn't accidentally match (e.g. "stockholm" wouldn't trigger the
# stock-market default).
_FINANCE_KEYWORDS = (
    "money", "stocks", "stock chat", "finance", "market", "trader",
    "trading", "invest", "wsb", "wallstreet",
)
_WOO_KEYWORDS = (
    "woo", "astro", "tarot", "magic", "witch", "spirit", "occult",
    "esoteric", "divin", "ritual", "mystic", "moon",
)


async def prepopulate_for_existing_contexts(
    oracle_store: "OracleStore",
    context_registry,
    settings_store=None,
) -> int:
    """One-time migration: walk every group context, run the heuristic,
    insert defaults for contexts that have no oracles yet.

    Idempotent — a context with any existing oracle rows is skipped, so
    running this on every boot doesn't double-populate. Default rows
    land disabled; admin opts in per-row via the UI.

    Carry-over: if the legacy global `daily_oracle_enabled` was on AND
    `daily_oracle_group_id` matches one of the prepopulated contexts,
    the corresponding tarot oracle is enabled so service continues
    without admin intervention. Old settings are NOT deleted (they
    remain harmlessly in the kv store) — admins can clear them via
    /admin/settings if they want.

    Returns: number of oracles inserted.
    """
    contexts = await context_registry.list()
    legacy_enabled = False
    legacy_group_id = ""
    if settings_store is not None:
        try:
            legacy_enabled = bool(settings_store.get("daily_oracle_enabled", False))
            legacy_group_id = (settings_store.get("daily_oracle_group_id") or "").strip()
        except Exception:
            pass

    inserted = 0
    for ctx in contexts:
        if ctx.kind != "group":
            continue
        if ctx.id is None:
            continue
        existing = await oracle_store.list_for_context(ctx.id)
        if existing:
            continue
        defaults = default_oracles_for_label(ctx.label, ctx.id)
        if not defaults:
            continue
        # Carry over the legacy global setting if it points at this
        # context AND we're inserting a tarot default.
        carry_target_match = (
            legacy_enabled and ctx.key == legacy_group_id
        )
        for oracle in defaults:
            if carry_target_match and oracle.kind == "tarot":
                oracle.enabled = True
            try:
                await oracle_store.upsert(oracle)
                inserted += 1
            except Exception as e:
                logger.error(
                    f"Failed to insert default oracle for context #{ctx.id} "
                    f"({ctx.label!r}): {e}"
                )
    if inserted:
        logger.info(
            f"Oracle prepopulation: inserted {inserted} default oracle row(s)"
        )
    return inserted


def default_oracles_for_label(label: str, context_id: int) -> list[ContextOracle]:
    """Suggested oracle rows based on a context's label keywords.

    Returns instances ready to be saved (enabled=False — admin opts in
    after reviewing). Empty list when nothing matches; admin still
    gets a clean "no oracles configured" UI to add their own.
    """
    low = (label or "").lower()
    out: list[ContextOracle] = []

    if any(k in low for k in _FINANCE_KEYWORDS):
        out.append(ContextOracle(
            id=None, context_id=context_id, enabled=False,
            kind="market_open", schedule_kind="clock",
            clock_time="09:25", timezone="America/New_York",
            weekdays_only=True,
            label="morning market check",
        ))
        out.append(ContextOracle(
            id=None, context_id=context_id, enabled=False,
            kind="market_close", schedule_kind="clock",
            clock_time="16:05", timezone="America/New_York",
            weekdays_only=True,
            label="close-of-day recap",
        ))

    if any(k in low for k in _WOO_KEYWORDS):
        out.append(ContextOracle(
            id=None, context_id=context_id, enabled=False,
            kind="tarot", schedule_kind="sunrise", offset_minutes=1,
            timezone="America/New_York", weekdays_only=False,
            label="sunrise tarot",
        ))

    return out
