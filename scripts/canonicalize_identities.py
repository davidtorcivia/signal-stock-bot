#!/usr/bin/env python3
"""One-shot rekey of every identity column from phone numbers onto UUIDs.

signal-cli reports the same contact as a phone number on one account's view
and as a UUID on another's, so with two linked accounts every person ended
up with two identities: two `user_names` rows, two DM contexts, two piles of
memories. `sourceUuid` is always present on inbound envelopes, so ingress
now keys off the UUID (src/signal/handler.py `sender_id`) — this script
drags the existing rows over to match.

    python scripts/canonicalize_identities.py data/watchlist.db \\
        [--signal-api http://127.0.0.1:8093 | --identities-json pairs.json] \\
        [--apply]

Without --apply everything runs in one transaction and rolls back, printing
exactly what would change. Idempotent: a second run is a no-op.
"""

import argparse
import hashlib
import json
import re
import sqlite3
import urllib.request

# Same as src.database.hash_phone — inlined so this script stays stdlib-only.
def sha(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


HEX64 = re.compile(r"^[0-9a-f]{64}$")

# (a) columns holding a RAW sender. Deliberately NOT auto-discovered: a
# generic "looks like a phone" scan would also rewrite `bots.signal_phone`,
# which is an account number, not a contact.
RAW_COLUMNS = [("group_messages", "sender"), ("alerts", "user_phone")]

# Columns the task enumerated; the generic scan below must cover all of
# them or we print a loud warning (schema drifted).
EXPECTED_COLUMNS = [
    ("tips", "tipper_user_hash"), ("alerts", "user_hash"),
    ("conversation_turns", "user_hash"), ("iching_daily", "user_hash"),
    ("tarot_daily", "user_hash"), ("watchlists", "user_hash"),
    ("predictions", "user_hash"), ("predictions", "resolver_user_hash"),
    ("resolution_votes", "voter_user_hash"),
    ("conversation_context", "user_hash"),
    ("context_memories", "subject_key"),
    ("context_memories", "source_user_hash"),
    ("conversation_turns", "context_key"),
    ("conversation_summaries", "context_key"),
    ("llm_tool_operations", "source_key"),
]


def mask(v: str) -> str:
    return f"+1XXXXXX{v[-4:]}" if v.startswith("+") else v


def get_json(url: str):
    with urllib.request.urlopen(url, timeout=30) as r:
        return json.load(r)


def load_identities(args) -> list[tuple[str, str]]:
    if args.identities_json:
        with open(args.identities_json) as f:
            raw = json.load(f)
    else:
        raw = []
        for account in get_json(f"{args.signal_api}/v1/accounts"):
            raw.extend(get_json(f"{args.signal_api}/v1/identities/{account}"))
    pairs = {}
    for e in raw:
        number = (e.get("number") or "").strip()
        uuid = (e.get("uuid") or "").strip()
        if number and uuid:
            pairs[number] = uuid
    return sorted(pairs.items())


def text_columns(conn) -> list[tuple[str, str]]:
    out = []
    for (table,) in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' "
        "AND name NOT LIKE 'sqlite_%'"
    ).fetchall():
        for row in conn.execute(f"PRAGMA table_info({table})"):
            if "INT" not in (row[2] or "").upper() and "REAL" != (row[2] or ""):
                out.append((table, row[1]))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("db_path")
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--identities-json")
    ap.add_argument("--signal-api")
    args = ap.parse_args()
    if not (args.identities_json or args.signal_api):
        ap.error("need --identities-json or --signal-api")

    pairs = load_identities(args)
    conn = sqlite3.connect(args.db_path)
    conn.execute("BEGIN")
    print(f"=== canonicalize_identities on {args.db_path} "
          f"({'APPLY' if args.apply else 'DRY RUN'}) ===")
    print(f"identities: {len(pairs)} (number, uuid) pairs\n")

    phone2uuid = dict(pairs)
    # hash(phone) -> hash(uuid), plus the "dm:<hash>" storage-key form.
    remap: dict[str, str] = {}
    for phone, uuid in pairs:
        remap[sha(phone)] = sha(uuid)
        remap[f"dm:{sha(phone)}"] = f"dm:{sha(uuid)}"
    canonical = {sha(u) for u in phone2uuid.values()}
    canonical |= {f"dm:{h}" for h in canonical}

    # (a) raw phone -> uuid.
    for table, col in RAW_COLUMNS:
        n = 0
        for phone, uuid in pairs:
            n += conn.execute(
                f'UPDATE "{table}" SET "{col}" = ? WHERE "{col}" = ?', (uuid, phone)
            ).rowcount
        print(f"(a) {table}.{col}: {n} rows phone -> uuid")

    # (a2) contexts.key for DMs holds the RAW sender. UNIQUE(key), and a
    # merge would have to fold two contexts' settings together — refuse and
    # report instead. The storage-key rewrites below still run for those
    # people: ingress now resolves by UUID, so history has to land there
    # regardless of which contexts row survives.
    ctx_rows = {
        k: (i, lbl) for i, k, lbl in conn.execute(
            "SELECT id, key, label FROM contexts WHERE kind = 'dm'"
        )
    }
    ctx_moved = 0
    for phone, uuid in pairs:
        if phone not in ctx_rows:
            continue
        if uuid in ctx_rows:
            print(f"(a2) CONFLICT manual merge needed: "
                  f"#{ctx_rows[phone][0]} {mask(phone)} "
                  f"({ctx_rows[phone][1]!r}) vs #{ctx_rows[uuid][0]} {uuid} "
                  f"({ctx_rows[uuid][1]!r}) — left alone")
            continue
        conn.execute("UPDATE contexts SET key = ? WHERE id = ?",
                     (uuid, ctx_rows[phone][0]))
        print(f"(a2) contexts #{ctx_rows[phone][0]} {mask(phone)} -> {uuid}")
        ctx_moved += 1
    print(f"(a2) {ctx_moved} dm contexts rekeyed\n")

    # (b) user_names first: the generic pass would keep the uuid row's
    # (possibly stale) name and leave nothing to merge.
    names = {h: (n, u) for h, n, u in
             conn.execute("SELECT user_hash, name, updated_at FROM user_names")}
    merged = renamed = 0
    for phone, uuid in pairs:
        hp, hu = sha(phone), sha(uuid)
        if hp not in names:
            continue
        if hu in names:
            newest = max(names[hp], names[hu], key=lambda r: r[1] or 0)
            conn.execute(
                "UPDATE user_names SET name = ?, updated_at = ? "
                "WHERE user_hash = ?", (newest[0], newest[1], hu))
            conn.execute("DELETE FROM user_names WHERE user_hash = ?", (hp,))
            print(f"(b) merged {names[hp][0]!r} {hp[:8]} -> {hu[:8]} "
                  f"(kept {newest[0]!r})")
            merged += 1
        else:
            conn.execute("UPDATE user_names SET user_hash = ? "
                         "WHERE user_hash = ?", (hu, hp))
            print(f"(b) {names[hp][0]!r} {hp[:8]} -> {hu[:8]}")
            renamed += 1
    print(f"(b) user_names: {renamed} rekeyed, {merged} merged\n")

    # (c) every other column holding a user hash or a "dm:<hash>" storage
    # key, discovered from the schema rather than a hand-list. Exact-value
    # membership only, so a column that happens to hold some other sha256
    # can't be hit. UPDATE OR IGNORE leaves rows that would violate a
    # UNIQUE/PK (watchlists(user_hash,symbol), the daily draws, ...)
    # untouched; those leftovers are by definition duplicates of a row that
    # already exists under the uuid hash, so they get dropped.
    keys = list(remap)
    placeholders = ",".join("?" * len(keys))
    scanned, total_up, total_del = [], 0, 0
    for table, col in text_columns(conn):
        if (table, col) in (("user_names", "user_hash"), ("contexts", "key")):
            continue
        present = [v for (v,) in conn.execute(
            f'SELECT DISTINCT "{col}" FROM "{table}" WHERE "{col}" IN ({placeholders})',
            keys)]
        scanned.append((table, col))
        if not present:
            continue
        up = dropped = 0
        for old in present:
            up += conn.execute(
                f'UPDATE OR IGNORE "{table}" SET "{col}" = ? WHERE "{col}" = ?',
                (remap[old], old)).rowcount
            dropped += conn.execute(
                f'DELETE FROM "{table}" WHERE "{col}" = ?', (old,)).rowcount
        total_up += up
        total_del += dropped
        print(f"(c) {table}.{col}: {up} rekeyed, {dropped} collided+dropped")
    print(f"(c) {total_up} rows rekeyed, {total_del} duplicates dropped")
    missing = [c for c in EXPECTED_COLUMNS
               if c not in scanned and c != ("user_names", "user_hash")]
    if missing:
        print(f"(c) WARNING: expected columns not scanned: {missing}")
    print()

    # (d) the rekey merged memories that add()'s dedup never compared.
    seen: dict[tuple, set] = {}
    deduped = 0
    for mem_id, ctx, key, kind, bot_id, content in conn.execute(
        "SELECT id, context_id, subject_key, kind, bot_id, content "
        "FROM context_memories ORDER BY id"
    ).fetchall():
        bucket = seen.setdefault((ctx, key, kind, bot_id), set())
        if content in bucket:
            conn.execute("DELETE FROM context_memories WHERE id = ?", (mem_id,))
            print(f"(d) deleted duplicate #{mem_id}: {content[:70]}")
            deduped += 1
        else:
            bucket.add(content)
    print(f"(d) {deduped} duplicate memories deleted\n")

    # (e) what's left that still looks like an identity we have no pair for.
    print("(e) orphans (identity-shaped values with no known pair):")
    orphans = 0
    known = set(text_columns(conn))
    for table, col in EXPECTED_COLUMNS + RAW_COLUMNS + [
        ("contexts", "key"), ("user_names", "user_hash")
    ]:
        if (table, col) not in known:
            continue
        rows = []
        for v, n in conn.execute(
            f'SELECT "{col}", COUNT(*) FROM "{table}" '
            f'WHERE "{col}" IS NOT NULL GROUP BY 1'
        ):
            v = str(v)
            body = v[3:] if v.startswith("dm:") else v
            if v in canonical or body in canonical:
                continue
            if v.startswith("+") or HEX64.match(body):
                rows.append((v, n))
        if rows:
            n = sum(n for _v, n in rows)
            orphans += n
            print(f"    {table}.{col}: {n} rows / {len(rows)} distinct "
                  f"({', '.join(mask(v)[:12] for v, _ in rows[:4])}"
                  f"{'...' if len(rows) > 4 else ''})")
    print(f"(e) {orphans} orphan rows total\n")

    if args.apply:
        conn.commit()
        print("COMMITTED.")
    else:
        conn.rollback()
        print("rolled back (dry run) — re-run with --apply to keep this.")
    conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
