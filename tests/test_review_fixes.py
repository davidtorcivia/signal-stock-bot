"""Regression tests for the 2026-06-10 review fixes.

Covers:
  * conversation_summaries legacy-PK rebuild (the live bug: every
    summarizer write failed with "ON CONFLICT clause does not match any
    PRIMARY KEY or UNIQUE constraint" because ALTER TABLE ADD COLUMN
    can't extend a primary key)
  * leaked pseudo-tool-call markup stripped from writer output
  * attachment MIME sniffing (no more hardcoded image/png)
  * per-context append lock exists and serializes
"""

import asyncio
import base64
import sqlite3
import time

import pytest

from src.commands.ask_command import _strip_tool_call_leak
from src.llm.history import ConversationHistory
from src.signal.handler import SignalHandler


# ---------------------------------------------------------------------------
# conversation_summaries migration
# ---------------------------------------------------------------------------


def _make_legacy_summaries_db(path):
    """Reproduce the production schema: single-column PK + bolted-on
    bot_id column (what the old ALTER-only migration produced)."""
    conn = sqlite3.connect(path)
    conn.execute(
        """
        CREATE TABLE conversation_summaries (
            context_key TEXT PRIMARY KEY,
            summary TEXT NOT NULL,
            summary_through_id INTEGER NOT NULL DEFAULT 0,
            turns_summarized INTEGER NOT NULL DEFAULT 0,
            updated_at REAL NOT NULL
        )
        """
    )
    conn.execute(
        "ALTER TABLE conversation_summaries "
        "ADD COLUMN bot_id INTEGER NOT NULL DEFAULT 0"
    )
    conn.execute(
        "INSERT INTO conversation_summaries "
        "(context_key, summary, summary_through_id, turns_summarized, "
        " updated_at, bot_id) VALUES (?, ?, ?, ?, ?, 0)",
        ("group:legacy", "old recap", 7, 3, time.time()),
    )
    conn.commit()
    conn.close()


@pytest.mark.asyncio
async def test_summaries_legacy_pk_rebuilt_and_upsert_works(tmp_path):
    db_path = tmp_path / "wl.db"
    _make_legacy_summaries_db(db_path)

    hist = ConversationHistory(db_path=str(db_path))
    await hist._ensure_initialized()

    # Composite PK present after migration.
    conn = sqlite3.connect(db_path)
    pk_cols = {
        r[1] for r in conn.execute("PRAGMA table_info(conversation_summaries)")
        if r[5]
    }
    conn.close()
    assert pk_cols == {"context_key", "bot_id"}

    # Legacy row survived as the bot_id=0 sentinel.
    legacy = await hist.get_summary("group:legacy")
    assert legacy is not None and legacy["summary"] == "old recap"

    # The exact write that failed in production: per-bot upserts.
    await hist.upsert_summary("group:woo", "sigil recap", 10, 4, bot_id=1)
    await hist.upsert_summary("group:woo", "artaud recap", 12, 5, bot_id=2)
    # Update path of the upsert (the ON CONFLICT branch).
    await hist.upsert_summary("group:woo", "sigil recap v2", 20, 8, bot_id=1)

    s1 = await hist.get_summary("group:woo", bot_id=1)
    s2 = await hist.get_summary("group:woo", bot_id=2)
    assert s1["summary"] == "sigil recap v2"
    assert s2["summary"] == "artaud recap"


@pytest.mark.asyncio
async def test_summaries_fresh_install_unaffected(tmp_path):
    hist = ConversationHistory(db_path=str(tmp_path / "fresh.db"))
    await hist.upsert_summary("group:x", "recap", 1, 1, bot_id=3)
    s = await hist.get_summary("group:x", bot_id=3)
    assert s["summary"] == "recap"


# ---------------------------------------------------------------------------
# tool-call markup leak stripping
# ---------------------------------------------------------------------------


def test_strip_tool_call_leak_function_markup():
    # The exact shape observed 2026-06-09 in bot.log.
    leaked = (
        "I'll ignore that. Nothing to add.\n\n"
        "<function=brave-search__brave_web_search>\n"
        " <parameter =query >something</parameter>"
    )
    assert _strip_tool_call_leak(leaked) == "I'll ignore that. Nothing to add."


def test_strip_tool_call_leak_deepseek_markers():
    leaked = "Sure.\n<|tool▁calls▁begin|>{...}"
    assert _strip_tool_call_leak(leaked) == "Sure."


def test_strip_tool_call_leak_closing_style_function_markup():
    # The exact shape observed 2026-06-12 from Artaud's local writer:
    # a closing-style `</function=...>` marker plus a bare parameter
    # block. The original pattern required `<function`, no slash.
    leaked = (
        "What does that say about how seriously I should take it?\n\n"
        "</function=remember>\n"
        "<parameter=fact>\n"
        "Taylor addressed Sigil as Mr. X in a group chat.\n"
        "</parameter>"
    )
    assert _strip_tool_call_leak(leaked) == (
        "What does that say about how seriously I should take it?"
    )


def test_strip_tool_call_leak_bare_parameter_block():
    leaked = "Fine.\n<parameter=fact>\nsomething\n</parameter>"
    assert _strip_tool_call_leak(leaked) == "Fine."


def test_strip_tool_call_leak_leaves_normal_text():
    for text in (
        "AAPL is up 2% today.",
        "use a < b in the comparison",
        "the <em>markup</em> stays",
        "",
    ):
        assert _strip_tool_call_leak(text) == text


def test_strip_tool_call_leak_whole_output_empties():
    assert _strip_tool_call_leak("<function=foo>\n<parameter=x>1") == ""


# ---------------------------------------------------------------------------
# attachment MIME sniffing
# ---------------------------------------------------------------------------


def _b64(head: bytes) -> str:
    return base64.b64encode(head + b"\x00" * 64).decode()


def test_attachment_mime_sniffing():
    cases = [
        (b"\x89PNG\r\n\x1a\n", "image/png"),
        (b"\xff\xd8\xff\xe0", "image/jpeg"),
        (b"GIF89a", "image/gif"),
        (b"RIFF\x00\x00\x00\x00WEBP", "image/webp"),
        (b"%PDF-1.7", "application/pdf"),
    ]
    for head, mime in cases:
        uri = SignalHandler._attachment_data_uri(_b64(head))
        assert uri.startswith(f"data:{mime};filename="), (head, uri)


def test_attachment_mime_unknown_falls_back_to_png():
    uri = SignalHandler._attachment_data_uri(_b64(b"garbage!"))
    assert uri.startswith("data:image/png;")
    # Not valid base64 at all — still must not raise.
    uri = SignalHandler._attachment_data_uri("@@@not-base64@@@")
    assert uri.startswith("data:image/png;")


# ---------------------------------------------------------------------------
# per-context history lock
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_lock_for_is_per_context_and_bounded():
    hist = ConversationHistory(db_path=":memory:")
    a = hist.lock_for("group:a")
    assert hist.lock_for("group:a") is a          # stable per context
    assert hist.lock_for("group:b") is not a      # distinct per context
    assert isinstance(a, asyncio.Lock)

    # Growth past the cap sweeps idle locks instead of growing forever.
    for i in range(600):
        hist.lock_for(f"ctx:{i}")
    assert len(hist._ctx_locks) <= 600
