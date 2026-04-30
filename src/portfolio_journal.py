"""
Per-portfolio markdown trading journal.

Each chat with a portfolio gets one markdown file under
``data/portfolio_notes/<hash>.md``. The bot uses two LLM tools to
read and append: a free-form notebook for tracking thesis evolution,
acknowledging losses, planning follow-up trades, and remembering what
worked. Distinct from the structured `memory` store (subject + kind +
content) — the journal is the bot's narrative space.

Format:

    # Sigil's portfolio journal — group:abc...

    ## 2026-04-30 14:35 UTC

    Bought 5 AAPL @ $200. Breakout above 200 on volume; thesis is
    that tech rotation continues this week. Set a stop at 180 — that
    was the prior consolidation high.

    ## 2026-04-30 16:20 UTC

    Stop fired, took the $100. Lesson: didn't account for CPI print
    weakness. Bid too aggressive on entry; should have waited for
    confirmation candle. Adjusting playbook.

Each entry begins with a ``## <timestamp>`` H2 heading. The reader
parses on those markers, returning the last N entries. Append-only
in v1 — no edit / delete tools, so a misjudged entry stays in the
record (which is the point). Admins can edit the file by hand if
needed.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import hashlib
import logging
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Timestamp format used as the H2 marker. UTC keeps things stable
# across server timezone changes.
_TS_FORMAT = "%Y-%m-%d %H:%M UTC"

# Heading regex used by the reader to split entries. Anchored to line
# start; `## ` followed by anything = entry boundary. Markdown's other
# uses of `## ` (mid-line, escaped) are not a concern here since we
# control the writes.
_HEADING_RE = re.compile(r"^## ", re.MULTILINE)

# Hard caps. Defense-in-depth against a runaway LLM dumping megabytes
# of "reflection" into one file.
_MAX_ENTRY_CHARS = 4000          # one journal entry
_MAX_FILE_CHARS = 1_000_000      # ~1MB per portfolio's journal
_DEFAULT_READ_LIMIT = 10
_MAX_READ_LIMIT = 50


class PortfolioJournal:
    """Per-context markdown notebook.

    Files live under ``<base_dir>/<hash16>.md`` where hash16 is a
    sha256-truncated digest of the context_key. Stable, filesystem-
    safe (no slashes / equals signs from base64), and uniformly
    short. The bot doesn't see the path; it just calls
    append(context_key, ...) / read_recent(context_key, ...).

    Locking: per-path asyncio.Lock prevents interleaved writes. SQLite
    connections aren't an option here — files don't have transactions
    — so we take the lock around the read-modify-write of the size cap.
    """

    def __init__(self, base_dir: str | Path, *, bot_name: str = "Sigil"):
        self.base_dir = Path(base_dir)
        self.bot_name = bot_name
        # Per-path locks held in a dict. Purged lazily; for a few
        # hundred portfolios the dict is fine to keep in memory.
        self._locks: dict[Path, asyncio.Lock] = {}
        # Lock for the lock dict itself — preventing two coroutines
        # both creating a fresh asyncio.Lock for the same path and
        # racing on which one wins.
        self._dict_lock = asyncio.Lock()

    def _path_for(self, context_key: str) -> Path:
        """Stable filesystem path for a context_key. Hash truncation
        is short enough to read in `ls` output but wide enough to
        avoid collisions (16 hex = 64 bits)."""
        digest = hashlib.sha256(context_key.encode("utf-8")).hexdigest()[:16]
        return self.base_dir / f"{digest}.md"

    async def _lock_for(self, path: Path) -> asyncio.Lock:
        async with self._dict_lock:
            lock = self._locks.get(path)
            if lock is None:
                lock = asyncio.Lock()
                self._locks[path] = lock
            return lock

    async def append(
        self,
        context_key: str,
        entry: str,
        *,
        now: Optional[dt.datetime] = None,
    ) -> dict:
        """Append a timestamped entry. Returns a small dict the LLM
        tool dispatcher can echo back ({ok, entry_count, file_size}).
        Quietly truncates over-long entries rather than rejecting —
        the journal should never feel like it's punishing the bot for
        getting verbose."""
        entry = (entry or "").strip()
        if not entry:
            return {"ok": False, "error": "empty entry"}
        if len(entry) > _MAX_ENTRY_CHARS:
            # Trim with an ellipsis so the read-back is honest about
            # what was preserved.
            entry = entry[:_MAX_ENTRY_CHARS - 12].rstrip() + "\n…[truncated]"

        path = self._path_for(context_key)
        ts = (now or dt.datetime.now(dt.timezone.utc)).strftime(_TS_FORMAT)
        block = f"\n\n## {ts}\n\n{entry}\n"

        lock = await self._lock_for(path)
        async with lock:
            self.base_dir.mkdir(parents=True, exist_ok=True)
            # Bootstrap the file with a header on first write so a
            # human opening it knows what it is.
            if not path.exists():
                header = (
                    f"# {self.bot_name}'s portfolio journal — "
                    f"{context_key}\n\n"
                    f"Append-only chronological notebook. The bot owns "
                    f"this file; humans should treat it as read-only "
                    f"unless cleaning up obvious garbage.\n"
                )
                path.write_text(header, encoding="utf-8")

            # Size cap: if the file is over the cap, archive the old
            # content out of the way and start fresh. We don't try to
            # be clever about preserving recent entries — the bot can
            # always re-summarize from past trades + memory.
            current_size = path.stat().st_size
            if current_size + len(block) > _MAX_FILE_CHARS:
                archive = path.with_name(
                    f"{path.stem}.archive-{int(dt.datetime.now().timestamp())}.md"
                )
                try:
                    path.rename(archive)
                    logger.info(
                        f"PortfolioJournal: archived {path.name} "
                        f"({current_size} bytes) → {archive.name} "
                        f"and starting fresh"
                    )
                except Exception as e:
                    # If archive failed (e.g., perm), still let the
                    # append happen but log loudly.
                    logger.warning(
                        f"PortfolioJournal: archive failed for "
                        f"{path}: {e}"
                    )
                # Recreate the header on the fresh file.
                header = (
                    f"# {self.bot_name}'s portfolio journal — "
                    f"{context_key}\n\n"
                    f"(Previous entries archived; see "
                    f"{archive.name} alongside this file.)\n"
                )
                path.write_text(header, encoding="utf-8")

            # Append the new entry. Open in append mode so two writers
            # can't truncate each other (the asyncio Lock above
            # prevents interleave within this process, and append-mode
            # writes are atomic at the OS level for short writes).
            with path.open("a", encoding="utf-8") as f:
                f.write(block)

            new_size = path.stat().st_size

        return {
            "ok": True,
            "ts": ts,
            "file_size": new_size,
            "path": str(path),
        }

    async def read_recent(
        self,
        context_key: str,
        *,
        limit: int = _DEFAULT_READ_LIMIT,
    ) -> dict:
        """Return the last N entries (by `## ts` heading) plus a count
        of total entries. limit is clamped to [1, _MAX_READ_LIMIT]."""
        limit = max(1, min(_MAX_READ_LIMIT, int(limit)))
        path = self._path_for(context_key)
        if not path.exists():
            return {
                "ok": True,
                "entries": [],
                "total_entries": 0,
                "file_exists": False,
            }

        # No write lock needed for read — append-only writes mean we
        # can never see torn state mid-entry; worst case we miss the
        # absolute newest entry by a few ms, which is fine.
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as e:
            logger.warning(f"PortfolioJournal: read {path} failed: {e}")
            return {
                "ok": False,
                "error": f"could not read journal: {type(e).__name__}",
            }

        # Split on `^## ` markers. The first chunk before the first
        # `## ` is the file header (intro line + blank lines) — drop it.
        chunks = _HEADING_RE.split(content)
        if len(chunks) <= 1:
            # No entries yet, only the header.
            return {
                "ok": True,
                "entries": [],
                "total_entries": 0,
                "file_exists": True,
            }
        # chunks[0] = pre-first-heading content (file header)
        # chunks[1:] = each entry without its leading `## `
        entry_chunks = [c.strip() for c in chunks[1:] if c.strip()]
        total = len(entry_chunks)
        recent = entry_chunks[-limit:]

        # Each chunk's first line is the timestamp; the rest is body.
        entries = []
        for c in recent:
            lines = c.split("\n", 1)
            ts = lines[0].strip()
            body = lines[1].strip() if len(lines) > 1 else ""
            entries.append({"ts": ts, "body": body})

        return {
            "ok": True,
            "entries": entries,
            "total_entries": total,
            "file_exists": True,
            "showing": len(entries),
        }


__all__ = ["PortfolioJournal"]
