"""Tests for the per-portfolio markdown journal.

Covers:
  - First-write file initialization (header + first entry).
  - Append-only chronological storage.
  - Read returns last N entries by `## ts` heading.
  - Path stability across calls (same context_key → same hash).
  - Concurrent writes don't interleave / corrupt.
  - Per-entry truncation at the size cap.
  - File rotation when total size cap is exceeded.
"""

from __future__ import annotations

import asyncio
import datetime as dt

import pytest

from src.portfolio_journal import PortfolioJournal


@pytest.fixture
def journal(tmp_path):
    return PortfolioJournal(tmp_path / "notes", bot_name="Sigil")


CTX = "group:abc123"


# ---------- basic append + read --------------------------------------------

@pytest.mark.asyncio
async def test_first_append_creates_file_with_header(journal, tmp_path):
    res = await journal.append(CTX, "First note: bought AAPL.")
    assert res["ok"] is True
    assert res["file_size"] > 0

    path = journal._path_for(CTX)
    assert path.exists()
    content = path.read_text(encoding="utf-8")
    # Header on first line.
    assert content.startswith("# Sigil's portfolio journal")
    # Entry heading + body present.
    assert "## " in content
    assert "First note: bought AAPL." in content


@pytest.mark.asyncio
async def test_read_recent_returns_last_n_entries(journal):
    for i in range(5):
        await journal.append(CTX, f"Entry {i}")
    res = await journal.read_recent(CTX, limit=3)
    assert res["ok"] is True
    assert res["total_entries"] == 5
    assert len(res["entries"]) == 3
    bodies = [e["body"] for e in res["entries"]]
    # Last 3 in order.
    assert bodies == ["Entry 2", "Entry 3", "Entry 4"]


@pytest.mark.asyncio
async def test_read_recent_empty_journal_returns_empty_list(journal):
    res = await journal.read_recent(CTX, limit=10)
    assert res["ok"] is True
    assert res["entries"] == []
    assert res["total_entries"] == 0
    assert res["file_exists"] is False


@pytest.mark.asyncio
async def test_read_recent_clamps_limit(journal):
    for i in range(3):
        await journal.append(CTX, f"e{i}")
    too_high = await journal.read_recent(CTX, limit=9999)
    assert len(too_high["entries"]) == 3
    too_low = await journal.read_recent(CTX, limit=0)
    assert len(too_low["entries"]) == 1  # clamped to 1


# ---------- path stability + isolation -------------------------------------

@pytest.mark.asyncio
async def test_path_for_is_stable(journal):
    p1 = journal._path_for(CTX)
    p2 = journal._path_for(CTX)
    assert p1 == p2


@pytest.mark.asyncio
async def test_different_contexts_get_different_files(journal):
    p_a = journal._path_for("group:a")
    p_b = journal._path_for("group:b")
    assert p_a != p_b


@pytest.mark.asyncio
async def test_unfriendly_context_keys_sanitized(journal):
    """context_key with slashes / equals signs / unicode shouldn't
    leak into the filesystem path."""
    weird = "group:t9UuaZITXKy0Kq6Quhqbn9jWZvTTMfITS+s1sphMzTg=/../etc/passwd"
    path = journal._path_for(weird)
    # Must be inside base_dir, no traversal.
    assert journal.base_dir in path.parents
    # File name is the hash + .md, not the literal weird string.
    assert path.suffix == ".md"
    assert "/" not in path.name and ".." not in path.name


# ---------- empty / blank inputs --------------------------------------------

@pytest.mark.asyncio
async def test_empty_entry_rejected(journal):
    res = await journal.append(CTX, "")
    assert res["ok"] is False
    res = await journal.append(CTX, "   \n\n  ")
    assert res["ok"] is False


# ---------- length caps -----------------------------------------------------

@pytest.mark.asyncio
async def test_oversize_entry_silently_truncated(journal):
    """Entries over the per-entry cap are trimmed with an ellipsis
    rather than rejected."""
    huge = "x" * 10_000
    res = await journal.append(CTX, huge)
    assert res["ok"] is True
    read = await journal.read_recent(CTX, limit=1)
    body = read["entries"][0]["body"]
    assert len(body) <= 4100  # cap (4000) + small marker
    assert "[truncated]" in body


# ---------- concurrent writes -----------------------------------------------

@pytest.mark.asyncio
async def test_concurrent_appends_do_not_interleave(journal):
    """Two coroutines appending in parallel: every entry must land
    fully (no torn writes), and the entry count must equal the call
    count."""
    n = 30
    await asyncio.gather(
        *(journal.append(CTX, f"concurrent-{i}") for i in range(n)),
    )
    res = await journal.read_recent(CTX, limit=50)
    assert res["total_entries"] == n
    bodies = {e["body"] for e in res["entries"]}
    expected = {f"concurrent-{i}" for i in range(n)}
    assert bodies == expected


# ---------- timestamp ordering ----------------------------------------------

@pytest.mark.asyncio
async def test_timestamps_are_utc_iso_like(journal):
    """The writer uses a fixed UTC strftime so entries are sortable
    lexically."""
    fixed = dt.datetime(2026, 4, 30, 14, 35, tzinfo=dt.timezone.utc)
    res = await journal.append(CTX, "x", now=fixed)
    assert res["ts"] == "2026-04-30 14:35 UTC"


# ---------- file rotation ---------------------------------------------------

@pytest.mark.asyncio
async def test_size_cap_rotates_to_archive(monkeypatch, journal):
    """When the file would exceed the size cap, the journal renames
    the existing file to an archive and starts a fresh one. We use a
    cap large enough that the first three entries fit comfortably,
    then the fourth (with extra padding) deliberately crosses it."""
    # 2KB cap fits ~3 small entries; the deliberately-large 4th will
    # tip it over.
    monkeypatch.setattr("src.portfolio_journal._MAX_FILE_CHARS", 2_000)
    for i in range(3):
        await journal.append(CTX, f"entry-{i}")
    path = journal._path_for(CTX)
    pre_rotate_size = path.stat().st_size
    assert pre_rotate_size > 0
    pre_rotate_content = path.read_text(encoding="utf-8")
    assert "entry-0" in pre_rotate_content
    assert "entry-1" in pre_rotate_content
    assert "entry-2" in pre_rotate_content

    # Big entry that pushes us past the cap.
    big = "x" * 1_900
    await journal.append(CTX, big)

    archives = list(journal.base_dir.glob(f"{path.stem}.archive-*.md"))
    assert len(archives) == 1
    # Fresh file: new header + just the big entry, no old entries.
    fresh_content = path.read_text(encoding="utf-8")
    assert fresh_content.startswith("# Sigil's portfolio journal")
    assert "Previous entries archived" in fresh_content
    assert "entry-0" not in fresh_content
    assert "entry-1" not in fresh_content
    # The big entry made it into the fresh file.
    assert "x" * 100 in fresh_content
    # The archive holds the originals.
    archive_content = archives[0].read_text(encoding="utf-8")
    assert "entry-0" in archive_content
    assert "entry-1" in archive_content
    assert "entry-2" in archive_content
