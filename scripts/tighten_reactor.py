#!/usr/bin/env python3
"""One-shot: retune the emoji reactor from "fires on half of everything" to
"fires on roughly one message in ten".

Measured starting point (from metric_events, lifetime): 497 LLM evaluations
produced 234 reactions — a 47% react rate — with only 5 cooldown skips,
because every cheap gate was configured effectively off (sender/group
cooldowns at 5s against code defaults of 30/10, min_length blank,
temperature 1.0). The custom system prompts made it worse than the built-in
default by adding two firing triggers ("interesting link, source, or piece
of information"; "a good point that was made") and carving links out of the
"short/transactional → don't react" brake.

This script fixes the configuration half. The mechanical brakes (rolling
budget, no-repeat window, score threshold) ship in the code alongside it.

Two things worth knowing before running:

  * Per-context `reactor_prompt` OVERRIDES the global `reactor_system_prompt`
    entirely, so rewriting only the global would leave every real chat on
    its old permissive prompt. Five contexts carry the permissive text and
    are rewritten here, each keeping its own custom closing line.

  * Context 300 ("Slop & Awe") has a deliberate "do not react, only reply"
    prompt. It is left untouched — it is already maximally discerning.

Idempotent: re-running rewrites the same values. Prints a before/after diff
and takes a database backup first.
"""

from __future__ import annotations

import shutil
import sqlite3
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.settings_store import SettingsStore  # noqa: E402

DB = Path("data/watchlist.db")

# ── The rewritten prompt ───────────────────────────────────────────────────
# Changes from the version it replaces:
#   - drops "Is an interesting link, source, or piece of information" and
#     "Is a good point that was made" as firing triggers
#   - drops the "(does not apply to links)" carve-out from the main brake,
#     and states positively that sharing is not itself an event
#   - drops "Bias toward standard reactions ... when they fit", which was
#     pushing toward a house style
#   - replaces the 🤔 blacklist with a categorical rule about empty
#     reactions. The old prompt banned 🤔 by name and the model simply
#     migrated the hedge to 😬, which became 18% of all reactions (43/234).
#     Naming another emoji would just move it again.
#   - adds the one-in-ten budget framing and the score rubric
PROMPT_BODY = """\
You decide whether to react to messages in a Signal group chat with a single \
emoji. You see EVERY message in this chat. React to roughly one in ten.

React when the message:
  - Expresses strong sentiment (excitement, frustration, win, loss, awe, grief)
  - Lands a notable moment, milestone, or punchline
  - Asks for acknowledgement (a "good morning", a confession, a check-in)
  - Is a question that can be honestly answered with a single emoji

Do NOT react when:
  - The message is short, transactional, or expects a written reply
  - It is about logistics, scheduling, or routine updates
  - The bot is already answering it
  - Someone shared a link and you have nothing specific to say about its \
actual content. Sharing is not itself an event worth marking.
  - The only emoji that fits would just signal "I registered this." A \
reaction has to carry something a reader could not already guess. If it says \
nothing beyond acknowledgement, stay silent. This rule is about the emoji's \
emptiness, not about any particular emoji — reaching for a different vague \
one does not satisfy it.

Match the emoji to the actual content, not its category. A tweet about a \
housing crash is not a shrug — it is 🏚 or 😬 or 💀 depending on tone. A link \
to a recipe is 🍳 or 😋. A question about a paradox is not a hedge — answer \
with the choice you would make (📦 for one box, 💰💰 for both, etc).

Before reacting, ask whether the next message might deserve it more. If this \
one is not clearly among the best you will see this hour, skip it. Reacting \
to consecutive messages, or repeating an emoji you have used recently, reads \
as automatic rather than considered. Silence is correct here far more often \
than a hum.

Be clever, be insightful, communicate a lot with your choice.

Call the emoji_react tool with a SINGLE emoji that fits, plus an honest \
`score` for how much this message warranted one: 1-3 = you are reaching, \
4-6 = mild, 7-8 = clearly wants a nod, 9-10 = the room would notice if \
nobody reacted. Most messages are 1-5. Otherwise, do not call the tool."""


# Per-context closing lines to preserve, keyed by context id. These are the
# admin's own additions on top of the shared permissive base; the rewrite
# replaces the base and keeps the intent.
CONTEXT_TAILS = {
    3: "Try and restrain yourself and only react when you have a really strong conviction.",
    4: "Try and restrain yourself and only react when you have a really strong conviction.",
    19: "",
    189: "Try and restrain yourself and only react when you have a really strong conviction.",
    232: (
        "Try and restrain yourself and only react when you have a really "
        "strong conviction. Sparing and occasional reactions are much more "
        "impactful."
    ),
}

# Contexts deliberately excluded: 300 ("Slop & Awe") is configured to never
# react at all, only to trigger replies. Nothing to tighten.
SKIP_CONTEXTS = {300}


GLOBAL_SETTINGS = {
    # Cheap gates, restored from "effectively off" to values that bite.
    # These now suppress only the emoji tool, not the whole LLM call, so
    # raising them no longer throttles the natural-response path.
    "reactor_sender_cooldown": 300,
    "reactor_group_cooldown": 120,
    "reactor_min_length": 40,
    # 1.0 was sampling the react/don't-react decision itself. Keep the
    # decision near-deterministic; the emoji choice stays varied because
    # the no-repeat window forces it to.
    "reactor_temperature": 0.2,
    # Post-LLM brakes (new). Hourly is the burst cap, daily the volume cap.
    "reactor_hourly_budget": 3,
    "reactor_daily_budget": 12,
    "reactor_repeat_window": 3,
    # Log-only: record the score distribution for a week, then set this
    # from the dashboard histogram rather than guessing.
    "reactor_min_score": 0,
    "reactor_system_prompt": PROMPT_BODY,
}


def main() -> int:
    if not DB.exists():
        print(f"error: {DB} not found — run from the repo root", file=sys.stderr)
        return 1

    backup = DB.with_suffix(f".db.bak-reactor-tighten-{int(time.time())}")
    shutil.copy2(DB, backup)
    print(f"backup: {backup}\n")

    store = SettingsStore(str(DB))

    print("── global settings ──")
    for key, value in GLOBAL_SETTINGS.items():
        before = store.get(key)
        if key.endswith("_prompt"):
            print(f"  {key}: {len(str(before or ''))}c → {len(value)}c")
        else:
            print(f"  {key}: {before!r} → {value!r}")
        store.set(key, value)

    print("\n── per-context reactor prompts ──")
    conn = sqlite3.connect(DB)
    try:
        rows = dict(
            conn.execute(
                "SELECT id, label FROM contexts "
                "WHERE COALESCE(reactor_prompt,'') <> ''"
            ).fetchall()
        )
        for cid, tail in CONTEXT_TAILS.items():
            if cid in SKIP_CONTEXTS:
                continue
            label = rows.get(cid)
            if label is None:
                print(f"  [{cid}] not found or already blank — skipped")
                continue
            new = PROMPT_BODY + (f"\n\n{tail}" if tail else "")
            conn.execute(
                "UPDATE contexts SET reactor_prompt = ? WHERE id = ?",
                (new, cid),
            )
            print(f"  [{cid}] {label}: rewritten ({len(new)}c)")
        for cid in sorted(SKIP_CONTEXTS & rows.keys()):
            print(f"  [{cid}] {rows[cid]}: left alone (never-react prompt)")
        conn.commit()
    finally:
        conn.close()

    print("\nDone. Restart the bot to pick up the new settings.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
