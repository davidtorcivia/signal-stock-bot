#!/usr/bin/env python3
"""
Manual preview of the WSB digest — crawls a live Redlib instance and prints the
compiled digest (ticker tally + the text block that gets handed to deep-think).
No Signal, no LLM, no DB. Use it to eyeball / tune the crawl and ticker filter.

    python -m scripts.wsb_digest_preview
    python -m scripts.wsb_digest_preview --base-url https://gaegis.disinfo.zone \
        --subreddit wallstreetbets --top-n 15

Run from the repo root.
"""

import argparse
import asyncio
import sys

from src.wsb.digest import compile_wsb_digest, render_digest_text
from src.wsb.redlib import DEFAULT_BASE_URL, DEFAULT_SUBREDDIT, RedlibSource


async def _run(args) -> int:
    source = RedlibSource(base_url=args.base_url, subreddit=args.subreddit)
    try:
        digest = await compile_wsb_digest(
            source,
            top_n=args.top_n,
            max_posts=args.max_posts,
            max_discussion_posts=args.discussion,
            max_parent_threads=args.parents,
        )
    finally:
        await source.close()

    if digest.is_empty:
        print(f"!! empty digest — is {args.base_url} reachable and serving "
              f"r/{args.subreddit}?", file=sys.stderr)
        return 1

    print("=" * 72)
    print(f" r/{digest.subreddit} — {digest.date}  (source: {digest.source_label})")
    print(f" {digest.posts_scanned} posts, {digest.comments_scanned} comments, "
          f"{len(digest.megathreads)} megathread(s)")
    print("=" * 72)
    print("\nTOP TICKERS:")
    for i, t in enumerate(digest.tickers, 1):
        print(f" {i:2}. {t.symbol:6} {t.mentions:>3} mentions  {t.lean:8}"
              f"  bull{t.bull}/bear{t.bear}  ${t.cashtags} cashtags  w={t.weight}")
    print("\n" + "-" * 72)
    print("DIGEST TEXT (fed to deep-think as context):")
    print("-" * 72)
    print(render_digest_text(digest))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Preview the WSB digest from a Redlib instance.")
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--subreddit", default=DEFAULT_SUBREDDIT)
    ap.add_argument("--top-n", type=int, default=15)
    ap.add_argument("--max-posts", type=int, default=25)
    ap.add_argument("--discussion", type=int, default=6,
                    help="top non-megathread discussion posts to also pull comments from")
    ap.add_argument("--parents", type=int, default=40,
                    help="top parent comments to keep per thread (for the LLM context)")
    args = ap.parse_args()
    return asyncio.run(_run(args))


if __name__ == "__main__":
    raise SystemExit(main())
