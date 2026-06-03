"""Header-replacement tests for the daily oracle worker.

The schedule math moved to `tests/test_oracles.py` (now that it lives
in `src/contexts/oracles.py` for the per-context refactor). What's
left in `daily_oracle.py` is the worker loop and the header rewrite,
which has to play well with the existing tarot/iching command output
and slot the active bot's display_name into the framing.
"""

import datetime as dt

import pytest

from src.contexts.oracles import ContextOracle
from src.daily_oracle import DailyOracleWorker, _replace_header


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


# --- Stamp gating: mark_fired must only fire on a successful post ----------

class _StubPolicy:
    id = 1
    key = "group-key"
    kind = "group"
    default_bot_id = None


class _StubContexts:
    async def get(self, _ctx_id):
        return _StubPolicy()


class _StubStore:
    def __init__(self):
        self.marked: list[tuple[int, float]] = []

    async def mark_fired(self, oracle_id, fired_at):
        self.marked.append((oracle_id, fired_at))


class _StubResult:
    def __init__(self, success: bool, text: str = "", styled: bool = False):
        self.success = success
        self.text = text
        self.attachments = None
        self.styled = styled


class _StubAsk:
    def __init__(self, result):
        self.result = result
        self.calls = 0

    async def execute(self, _ctx):
        self.calls += 1
        return self.result


class _StubSender:
    def __init__(self):
        self.sends: list[dict] = []

    async def send_message(self, **kwargs):
        self.sends.append(kwargs)


def _make_worker(*, ask_result, sender=None):
    ask = _StubAsk(ask_result)
    sender = sender if sender is not None else _StubSender()
    worker = DailyOracleWorker(
        oracle_store=_StubStore(),
        context_registry=_StubContexts(),
        tarot_command=None,
        iching_command=None,
        ask_command=ask,
        signal_handler=sender,
        bot_phone="+15550001111",
    )
    return worker, ask, sender


def _make_oracle():
    return ContextOracle(
        id=42,
        context_id=1,
        enabled=True,
        kind="market_close",
        schedule_kind="clock",
        clock_time="16:05",
    )


@pytest.mark.asyncio
async def test_fire_oracle_skips_mark_fired_on_llm_failure():
    """The exact bug the DeepSeek 520 outage hit on 2026-05-27: a transient
    provider error returns a failed result from ask_command, but the old
    code still stamped last_fired_at — burning the slot for the day. The
    fix must leave the stamp untouched so the next worker tick retries."""
    worker, ask, sender = _make_worker(
        ask_result=_StubResult(success=False, text="◇ LLM HTTP 520: ...")
    )
    await worker._fire_oracle(_make_oracle())
    assert ask.calls == 1
    assert sender.sends == []
    assert worker.store.marked == []  # slot left open for retry


@pytest.mark.asyncio
async def test_fire_oracle_marks_fired_on_successful_post():
    worker, ask, sender = _make_worker(
        ask_result=_StubResult(success=True, text="Markets closed mixed today.")
    )
    await worker._fire_oracle(_make_oracle())
    assert ask.calls == 1
    assert len(sender.sends) == 1
    assert len(worker.store.marked) == 1
    assert worker.store.marked[0][0] == 42


@pytest.mark.asyncio
async def test_fire_oracle_skips_mark_fired_when_no_signal_handler():
    """No handler available is also a no-post — same retry semantics."""
    worker, _, _ = _make_worker(
        ask_result=_StubResult(success=True, text="body"),
        sender=None,
    )
    worker.signal = None  # simulate handler-not-wired
    worker.signal_pool = None
    await worker._fire_oracle(_make_oracle())
    assert worker.store.marked == []


def _evening_et_oracle():
    """A 20:00 America/New_York clock oracle — its fire instant lands at
    00:00 UTC (EDT) / 01:00 UTC (EST), right on the UTC date boundary."""
    return ContextOracle(
        id=4,
        context_id=1,
        enabled=True,
        kind="wsb_digest",
        schedule_kind="clock",
        clock_time="20:00",
        timezone="America/New_York",
    )


def test_already_fired_today_handles_utc_date_boundary():
    """Regression: the WSB digest (20:00 ET == 00:00 UTC) double-fired on
    2026-06-02 because _already_fired_today keyed off `now`'s UTC date. Once
    the oracle fired the clock rolled into the next UTC day, so the date-keyed
    lookup pointed at *tomorrow's* 20:00 ET instant (~24h away) and reported
    'not fired today' — re-firing every tick inside the 600s grace window."""
    oracle = _evening_et_oracle()
    # Fired at 20:00:04 ET on 2026-06-02 == 00:00:04 UTC 2026-06-03, marked a
    # few minutes later once the ~4-minute deep-think run finished.
    fired = dt.datetime(2026, 6, 3, 0, 4, 35, tzinfo=dt.timezone.utc)
    oracle.last_fired_at = fired.timestamp()

    # A tick a few minutes later (the second crawl actually started 00:09:42).
    now = dt.datetime(2026, 6, 3, 0, 9, 40, tzinfo=dt.timezone.utc)
    assert DailyOracleWorker._already_fired_today(oracle, now) is True

    # ...and right after the (bug-era) second post, still fired-for-today.
    now2 = dt.datetime(2026, 6, 3, 0, 14, 15, tzinfo=dt.timezone.utc)
    assert DailyOracleWorker._already_fired_today(oracle, now2) is True


def test_already_fired_today_allows_next_days_fire():
    """The fix must not over-suppress: the morning after a fire, the oracle is
    fireable again for tonight's occurrence."""
    oracle = _evening_et_oracle()
    fired = dt.datetime(2026, 6, 3, 0, 4, 35, tzinfo=dt.timezone.utc)
    oracle.last_fired_at = fired.timestamp()

    # ~13:00 UTC 2026-06-03 == 09:00 ET, well before tonight's 20:00 ET fire.
    morning = dt.datetime(2026, 6, 3, 13, 0, tzinfo=dt.timezone.utc)
    assert DailyOracleWorker._already_fired_today(oracle, morning) is False


def test_already_fired_today_never_fired():
    oracle = _evening_et_oracle()
    oracle.last_fired_at = None
    now = dt.datetime(2026, 6, 3, 0, 9, 40, tzinfo=dt.timezone.utc)
    assert DailyOracleWorker._already_fired_today(oracle, now) is False
