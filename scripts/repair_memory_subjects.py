#!/usr/bin/env python3
"""One-shot repair of `context_memories.subject_key` fragmentation.

Two Signal accounts report the same contact differently (one phones, one
UUIDs), so every regular ended up with two `user_names` rows — same name,
two hashes — and memories scattered across both, plus a pile of
`freetext:<name>` rows minted before the resolver stopped doing that.

    python scripts/repair_memory_subjects.py data/watchlist.db [--apply]

Without --apply it runs the whole thing in a transaction and rolls back,
printing exactly what would change.
"""

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import sqlite3  # noqa: E402

from src.memory import _is_near_duplicate  # noqa: E402

PHONE_TAIL_RE = re.compile(r"^\[?\.{0,3}\d{4}\]?$")


def slugify(name: str) -> str:
    return (name or "").strip().lower().replace(" ", "-")


def main() -> int:
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    db_path = sys.argv[1]
    apply = "--apply" in sys.argv[2:]
    conn = sqlite3.connect(db_path)
    conn.execute("BEGIN")

    print(f"=== repair_memory_subjects on {db_path} "
          f"({'APPLY' if apply else 'DRY RUN'}) ===\n")

    # (a) name -> oldest registered hash; hash -> canonical hash.
    canonical_of_name: dict[str, tuple[str, str]] = {}  # slug -> (hash, name)
    canonical_of_hash: dict[str, str] = {}
    rows = conn.execute(
        "SELECT user_hash, name FROM user_names ORDER BY updated_at"
    ).fetchall()
    name_of_hash = dict(rows)
    for user_hash, name in rows:
        slug = slugify(name)
        canonical_of_name.setdefault(slug, (user_hash, name))
        canonical_of_hash[user_hash] = canonical_of_name[slug][0]
    print(f"(a) {len(rows)} registered names -> "
          f"{len(canonical_of_name)} people")
    for slug, (h, name) in sorted(canonical_of_name.items()):
        dupes = [u for u in canonical_of_hash if canonical_of_hash[u] == h]
        if len(dupes) > 1:
            print(f"    {name}: {len(dupes)} hashes -> {h[:8]}")

    bot_slugs = set()
    for slug, display in conn.execute("SELECT slug, display_name FROM bots"):
        bot_slugs.update({slugify(slug), slugify(display)})
    bot_slugs.discard("")
    print(f"    bot names: {sorted(bot_slugs)}\n")

    # (b) sibling hashes -> canonical hash.
    moved = 0
    for user_hash, canon in canonical_of_hash.items():
        if user_hash == canon:
            continue
        label = next(
            (name for _h, name in canonical_of_name.values() if _h == canon),
            None,
        )
        cur = conn.execute(
            "UPDATE context_memories SET subject_key = ?, "
            "subject_label = COALESCE(?, subject_label) WHERE subject_key = ?",
            (canon, label, user_hash),
        )
        if cur.rowcount:
            print(f"(b) {user_hash[:8]} -> {canon[:8]}: {cur.rowcount} rows")
        moved += cur.rowcount
    print(f"(b) {moved} rows remapped onto canonical hashes\n")

    # (c) freetext:<registered name> -> that person; bot names -> __self__.
    freetext = conn.execute(
        "SELECT DISTINCT subject_key, subject_label FROM context_memories "
        "WHERE subject_key LIKE 'freetext:%'"
    ).fetchall()
    named = 0
    for key, label in freetext:
        slug = key.split(":", 1)[1]
        if slug in bot_slugs:
            new_key, new_label = "__self__", "the bot"
        elif slug in canonical_of_name:
            new_key, new_label = canonical_of_name[slug]
        else:
            continue
        cur = conn.execute(
            "UPDATE context_memories SET subject_key = ?, subject_label = ? "
            "WHERE subject_key = ?",
            (new_key, new_label, key),
        )
        print(f"(c) {key} -> {new_key[:8]} ({new_label}): {cur.rowcount} rows")
        named += cur.rowcount
    print(f"(c) {named} freetext rows attached to a real subject\n")

    # (d) phone-tail placeholders -> the speaker who triggered them, or gone.
    tails = conn.execute(
        "SELECT id, subject_key, subject_label, source_user_hash, content "
        "FROM context_memories WHERE subject_key LIKE 'freetext:%'"
    ).fetchall()
    tail_fixed = tail_dropped = 0
    for mem_id, key, label, source_hash, content in tails:
        if not PHONE_TAIL_RE.match((label or "").strip()):
            continue
        canon = canonical_of_hash.get(source_hash or "")
        if canon:
            conn.execute(
                "UPDATE context_memories SET subject_key = ?, "
                "subject_label = ? WHERE id = ?",
                (canon, name_of_hash[canon], mem_id),
            )
            print(f"(d) #{mem_id} {key} -> {canon[:8]}")
            tail_fixed += 1
        else:
            conn.execute("DELETE FROM context_memories WHERE id = ?", (mem_id,))
            print(f"(d) deleted #{mem_id} ({key}, unregistered source "
                  f"{(source_hash or '')[:8] or 'none'}): {content}")
            tail_dropped += 1
    print(f"(d) {tail_fixed} placeholder rows re-attached, "
          f"{tail_dropped} deleted\n")

    # (e) dedup within (context_id, subject_key, kind, bot_id) — the
    #     remapping above merged rows that add()'s dedup never compared.
    groups: dict[tuple, list[tuple[int, str]]] = {}
    for mem_id, ctx, key, kind, bot_id, content in conn.execute(
        "SELECT id, context_id, subject_key, kind, bot_id, content "
        "FROM context_memories ORDER BY id"
    ):
        groups.setdefault((ctx, key, kind, bot_id), []).append((mem_id, content))
    deduped = 0
    for group in groups.values():
        kept: list[tuple[int, str]] = []
        for mem_id, content in group:
            if any(
                k_content == content or _is_near_duplicate(k_content, content)
                for _, k_content in kept
            ):
                conn.execute(
                    "DELETE FROM context_memories WHERE id = ?", (mem_id,)
                )
                print(f"(e) deleted duplicate #{mem_id}: {content}")
                deduped += 1
            else:
                kept.append((mem_id, content))
    print(f"(e) {deduped} duplicate rows deleted\n")

    # (f) flatten the vestigial scoring columns.
    cur = conn.execute(
        "UPDATE context_memories SET confidence = 1.0, corroborations = 1, "
        "distinct_speakers = '[]'"
    )
    print(f"(f) {cur.rowcount} rows normalized (confidence/corroborations)\n")

    # (g) resulting distribution.
    print("(g) subject_key distribution after repair:")
    for key, label, n in conn.execute(
        "SELECT subject_key, subject_label, COUNT(*) FROM context_memories "
        "GROUP BY 1, 2 ORDER BY 3 DESC"
    ):
        print(f"    {n:>4}  {key[:16]:<18} {label}")

    if apply:
        conn.commit()
        print("\nCOMMITTED.")
    else:
        conn.rollback()
        print("\nrolled back (dry run) — re-run with --apply to keep this.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
