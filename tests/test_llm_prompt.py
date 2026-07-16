"""Tests for the prompt-formatting helpers introduced in the v2 prompt build.

Covers:
  - format_history_timestamp absolute UTC rendering (cache-stable)
  - _wrap_xml empty/non-empty behaviour
  - ConversationHistory.load() timestamp + attribution prefixes
  - ConversationHistory.latest_turn_timestamp() empty + populated
"""

import asyncio
import re
import time
from datetime import datetime, timezone
from types import SimpleNamespace

import pytest

from src.commands.ask_command import (
    AskCommand,
    _strip_addressee_leak,
    _strip_meta_leak,
    _wrap_xml,
    STALENESS_THRESHOLD_SECONDS,
)
from src.commands.base import CommandContext
from src.bots.models import Bot
from src.llm.client import DEFAULT_SYSTEM_PROMPT
from src.llm.history import ConversationHistory, format_history_timestamp
from src.group_log import GroupMessageLog


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


# ---------- identity authority ----------------------------------------------

def _identity_ctx(*, name="Sigil", persona=""):
    return SimpleNamespace(
        bot=SimpleNamespace(
            display_name=name,
            aliases=[],
            persona=persona,
        )
    )


def test_identity_block_does_not_contradict_custom_writer_prompt():
    """The resolved writer prompt, not a stale registry display name, owns
    persona identity. Regression for `You are Artaud` followed by
    `<your_identity>Your name is Sigil` in the same system message."""
    ask = AskCommand.__new__(AskCommand)
    block = ask._build_identity_block(
        _identity_ctx(name="Sigil"),
        authoritative_prompt="You are Artaud and you engage with brevity.",
    )
    assert "Your name is Sigil" not in block
    assert "Chat routing handles for this bot: Sigil" in block
    assert "identity and voice come from the primary system prompt" in block


def test_identity_block_names_bot_with_generic_builtin_prompt():
    """The built-in prompt has no persona name, so the registry display
    name remains the useful and authoritative fallback."""
    ask = AskCommand.__new__(AskCommand)
    block = ask._build_identity_block(
        _identity_ctx(name="Sigil"),
        authoritative_prompt=DEFAULT_SYSTEM_PROMPT,
    )
    assert block.startswith("Your name is Sigil")


def test_identity_block_names_bot_with_non_persona_custom_prompt():
    """Custom behavioral instructions do not automatically erase the
    registry identity when they declare no competing character."""
    ask = AskCommand.__new__(AskCommand)
    block = ask._build_identity_block(
        _identity_ctx(name="Sigil"),
        authoritative_prompt="Keep answers concise and cite your sources.",
    )
    assert block.startswith("Your name is Sigil")


# ---------- volatile reaction state / prompt-cache locality -----------------

class _PromptStore:
    def get(self, key, default=None):
        if key == "reactor_enabled":
            return True
        return default

    def get_int(self, key, default, *, min_value=None):
        value = default
        if min_value is not None:
            value = max(min_value, value)
        return value


class _CapturingLLM:
    def __init__(self):
        self.store = _PromptStore()
        self.calls = []

    def _resolve_system_prompt(self, override, suffix):
        base = override or "You are Artaud and you engage with brevity."
        return f"{base}\n\n{suffix}" if suffix else base

    async def chat_messages(self, messages, tools=None, **kwargs):
        # The harness mutates its local list after this call, so snapshot the
        # strings needed for cache-locality assertions now.
        self.calls.append([dict(m) for m in messages])
        return {"role": "assistant", "content": "done"}


class _NoopHistory:
    def __init__(self):
        self.lock = asyncio.Lock()
        self.append_count = 0
        self.summary_read_count = 0

    async def load(self, *args, **kwargs):
        return []

    async def get_summary(self, *args, **kwargs):
        self.summary_read_count += 1
        return None

    async def latest_turn_timestamp(self, *args, **kwargs):
        return None

    def lock_for(self, context_key):
        return self.lock

    async def append(self, *args, **kwargs):
        self.append_count += 1
        return None


class _StaleSummaryHistory(_NoopHistory):
    async def get_summary(self, *args, **kwargs):
        self.summary_read_count += 1
        return {"summary": "SECRET OLD SUMMARY"}


class _OneShotPolicy:
    history_turns_override = 0
    purge_floor_at = None
    system_prompt = None
    reactor_enabled = True
    id = 1
    kind = "dm"

    def allows_deep_think(self):
        return False

    def allows_command(self, command):
        return False

    def allows_mcp(self, server):
        return False


class _SummaryPolicy(_OneShotPolicy):
    history_turns_override = 2


class _MutableReactor:
    def __init__(self):
        self.target = "first message"

    def recent_reactions(self, group_id, limit=5, bot_id=None):
        return [{"emoji": "🔥", "sender": "David", "target": self.target}]

    def is_enabled(self, bot=None, policy=None):
        return True


@pytest.mark.asyncio
async def test_recent_reactions_change_only_tail_user_message():
    """Changing reflex state must not change the static system prefix.

    This is the cache regression: the provider can keep its large prompt
    prefix cached while the recent-reaction log changes in the final user
    turn next to group context.
    """
    llm = _CapturingLLM()
    reactor = _MutableReactor()
    ask = AskCommand(llm, _NoopHistory(), reactor=reactor)
    bot = Bot(id=1, slug="sigil", display_name="Sigil")
    ctx = CommandContext(
        sender="+15551234123",
        group_id="group-1",
        raw_message="!ask why the reaction?",
        command="ask",
        args=["why", "the", "reaction?"],
        bot=bot,
    )

    first = await ask.execute(ctx)
    reactor.target = "second message"
    second = await ask.execute(ctx)

    assert first.success and second.success
    first_system = llm.calls[0][0]["content"]
    second_system = llm.calls[1][0]["content"]
    first_user = llm.calls[0][-1]["content"]
    second_user = llm.calls[1][-1]["content"]

    assert first_system == second_system
    # The static reflex instructions may name the tag, but no volatile log
    # body may be embedded in the system message.
    assert "<recent_reactions>\nRecent emoji reactions" not in first_system
    assert "do not have an explicit memory" not in first_system
    assert "short explicit log" in first_system
    assert "Your name is Sigil" not in first_system
    assert first_system.startswith("You are Artaud")
    assert "<recent_reactions>" in first_user
    assert "first message" in first_user
    assert "second message" in second_user


@pytest.mark.asyncio
async def test_zero_history_mode_skips_summary_and_persistence():
    """A one-shot context cannot inherit an old rolling summary or write
    turns that are immediately pruned."""
    llm = _CapturingLLM()
    history = _StaleSummaryHistory()
    ask = AskCommand(llm, history)
    bot = Bot(id=1, slug="sigil", display_name="Sigil")
    ctx = CommandContext(
        sender="+15551234123",
        group_id=None,
        raw_message="!ask clean slate",
        command="ask",
        args=["clean", "slate"],
        policy=_OneShotPolicy(),
        bot=bot,
    )

    result = await ask.execute(ctx)

    assert result.success
    assert history.summary_read_count == 0
    assert history.append_count == 0
    assert "SECRET OLD SUMMARY" not in llm.calls[0][0]["content"]


@pytest.mark.asyncio
async def test_conversation_summary_is_volatile_user_tail_not_system_prefix():
    llm = _CapturingLLM()
    history = _StaleSummaryHistory()
    ask = AskCommand(llm, history)
    ctx = CommandContext(
        sender="+15551234123", group_id=None, raw_message="!ask continue",
        command="ask", args=["continue"], policy=_SummaryPolicy(),
        bot=Bot(id=1, slug="sigil", display_name="Sigil"),
    )

    result = await ask.execute(ctx)

    assert result.success
    assert history.summary_read_count == 1
    assert "SECRET OLD SUMMARY" not in llm.calls[0][0]["content"]
    assert "<conversation_memory>" in llm.calls[0][-1]["content"]
    assert "SECRET OLD SUMMARY" in llm.calls[0][-1]["content"]


@pytest.mark.asyncio
async def test_zero_turn_history_override_loads_no_prior_turns(history):
    await history.append("ctx-zero", "user", "old question")
    await history.append("ctx-zero", "assistant", "old answer")
    assert await history.load("ctx-zero", turns_per_user=0) == []


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


async def test_load_group_can_expose_stable_turn_ids_and_source_pointer(history):
    await history.append(
        "group:42", "user", "what does that mean?",
        sender_tail="4137", source_message_ts=1784220300000,
    )
    turns = await history.load(
        "group:42", attribute_senders=True, now=time.time(),
        include_turn_ids=True, include_internal=True,
    )
    assert re.match(
        rf"\[turn h\d+; \.\.\.4137, {_UTC_STAMP}\] what does that mean\?",
        turns[0]["content"],
    )
    assert re.fullmatch(r"h\d+", turns[0]["_turn_id"])
    assert turns[0]["_source_message_ts"] == 1784220300000
    assert turns[0]["_raw_content"] == "what does that mean?"


@pytest.mark.asyncio
async def test_group_context_dedupes_history_and_points_followup(tmp_path):
    class _GroupPromptStore(_PromptStore):
        def get_int(self, key, default, *, min_value=None):
            value = 20 if key == "group_context_messages" else default
            return max(min_value, value) if min_value is not None else value

    llm = _CapturingLLM()
    llm.store = _GroupPromptStore()
    history = ConversationHistory(
        db_path=str(tmp_path / "history.db"), turns_per_user=10,
    )
    group_log = GroupMessageLog(db_path=str(tmp_path / "group.db"))
    await history.append(
        "group:group-1", "user", "how do you feel about Asher and Israel?",
        sender_tail="4123", source_message_ts=1001,
    )
    await history.append(
        "group:group-1", "assistant", "Asher was a gift; Israel remains.",
        sender_tail="4123",
    )
    await group_log.append(
        "group-1", "+15551234123",
        "how do you feel about Asher and Israel?", message_ts=1001,
    )
    # Dispatcher logs the current inbound row before AskCommand runs; the
    # group-context reader intentionally excludes this last row.
    await group_log.append(
        "group-1", "+15551234123", "what does that mean?", message_ts=1002,
    )

    ask = AskCommand(llm, history, group_log=group_log)
    bot = Bot(
        id=2, slug="artaud", display_name="Artaud",
        signal_phone="+15550000002",
    )
    ctx = CommandContext(
        sender="+15551234123", group_id="group-1",
        raw_message="!ask what does that mean?", command="ask",
        args=["what", "does", "that", "mean?"], bot=bot,
        message_timestamp=1002,
    )

    result = await ask.execute(ctx)
    assert result.success
    prompt = llm.calls[0]
    tail = prompt[-1]["content"]
    assistant_history = prompt[-2]["content"]
    turn_match = re.match(r"\[turn (h\d+); to ", assistant_history)
    assert turn_match
    assert f'follows="{turn_match.group(1)}"' in tail
    assert f'parent="{turn_match.group(1)}"' in tail
    # The user question remains in role history only, not a second time in
    # group_context, and private assembly keys never reach the provider.
    assert "<group_context>" not in tail
    assert all(not any(k.startswith("_") for k in m) for m in prompt)

    stored = await history.load(
        "group:group-1", turns_per_user=10, include_internal=True,
        include_turn_ids=True,
    )
    current_user = next(
        turn for turn in stored
        if turn.get("_raw_content") == "what does that mean?"
    )
    current_answer = stored[stored.index(current_user) + 1]
    assert current_user["_parent_turn_ref"] == turn_match.group(1)
    assert current_answer["_parent_turn_ref"] == current_user["_turn_id"]


@pytest.mark.asyncio
async def test_signal_quote_points_to_exact_group_turn(tmp_path):
    class _GroupPromptStore(_PromptStore):
        def get_int(self, key, default, *, min_value=None):
            value = 20 if key == "group_context_messages" else default
            return max(min_value, value) if min_value is not None else value

    llm = _CapturingLLM()
    llm.store = _GroupPromptStore()
    history = ConversationHistory(
        db_path=str(tmp_path / "history.db"), turns_per_user=10,
    )
    group_log = GroupMessageLog(db_path=str(tmp_path / "group.db"))
    await group_log.append(
        "group-1", "+15559994137", "That sounds intense.", message_ts=2001,
    )
    await group_log.append(
        "group-1", "+15551234123", "\ufffc", message_ts=2002,
    )
    ask = AskCommand(llm, history, group_log=group_log)
    ctx = CommandContext(
        sender="+15551234123", group_id="group-1",
        raw_message="!ask \ufffc", command="ask", args=["\ufffc"],
        bot=Bot(id=2, slug="artaud", display_name="Artaud"),
        quote_text="That sounds intense.", quote_author="+15559994137",
        quote_timestamp=2001, message_timestamp=2002,
    )

    assert (await ask.execute(ctx)).success
    tail = llm.calls[0][-1]["content"]
    group_turn = re.search(r"\[turn (g\d+);", tail)
    assert group_turn
    assert f'<replying_to turn="{group_turn.group(1)}"' in tail
    assert "[unrendered Signal object]" in tail


def test_strip_meta_leak_turn_pointer_form():
    assert _strip_meta_leak(
        "[turn h17; to David, 2026-07-16 16:45 UTC] actual answer"
    ) == "actual answer"


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
