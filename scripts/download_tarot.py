#!/usr/bin/env python3
"""
One-shot fetch of the Rider-Waite-Smith tarot deck from Wikimedia Commons.

Reads the deck definition in src/commands/tarot_data.py, downloads each
card's image from the Wikimedia ``Special:FilePath`` redirect (which handles
the MD5-prefixed thumbnail-host routing for us), and saves them under
``assets/tarot/<slug>.jpg`` at original resolution.

Idempotent: skips files that already exist non-empty. Run during the
Docker build so the bot ships with all 78 images baked in.

The RWS deck is in the public domain (Pamela Colman Smith / Arthur Edward
Waite, 1909) — fine to redistribute.
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from PIL import Image

# Add the repo root so we can import the deck module without installing.
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.commands.tarot_data import DECK  # noqa: E402

OUT_DIR = REPO_ROOT / "data" / "tarot"
WIKI_BASE = "https://commons.wikimedia.org/wiki/Special:FilePath/"
USER_AGENT = "signal-stock-bot/1.0 (+https://github.com/davidtorcivia/signal-stock-bot) tarot-fetch"
# Wikimedia rate-limits anonymous bot traffic; 1s pacing kept us under the
# threshold across multiple test runs. Raise if 429s reappear.
THROTTLE_SECONDS = 1.0
RETRIES = 4               # on top of the initial attempt
BACKOFF_BASE_SECONDS = 4  # sleep = base * 2**attempt

# Original Wikimedia scans run ~1100x1900 px / ~800KB each. That's overkill
# for phone display and bloats the Docker image. Resize to 500px wide
# (preserving aspect) and re-encode at JPEG q85 — still crisp on retina
# screens, drops the deck from ~65MB to ~5MB.
TARGET_WIDTH = 500
JPEG_QUALITY = 85


def fetch(url: str) -> bytes:
    """GET with bounded retries and exponential backoff on 429 / transient errors."""
    last_exc: Exception | None = None
    for attempt in range(RETRIES + 1):
        try:
            req = Request(url, headers={"User-Agent": USER_AGENT})
            with urlopen(req, timeout=30) as resp:
                return resp.read()
        except HTTPError as e:
            last_exc = e
            # Only back off on rate-limit / server-side; everything else
            # (404, 403) is permanent and should fail fast.
            if e.code not in (429, 500, 502, 503, 504):
                raise
        except URLError as e:
            last_exc = e
        if attempt < RETRIES:
            sleep_for = BACKOFF_BASE_SECONDS * (2 ** attempt)
            print(
                f"    retry {attempt + 1}/{RETRIES} in {sleep_for}s ({last_exc})",
                file=sys.stderr,
            )
            time.sleep(sleep_for)
    assert last_exc is not None
    raise last_exc


def resize_jpeg(data: bytes, target_width: int) -> bytes:
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    if img.width > target_width:
        ratio = target_width / img.width
        new_size = (target_width, int(img.height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=JPEG_QUALITY, optimize=True)
    return buf.getvalue()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    skipped = downloaded = failed = 0

    for card in DECK:
        out = OUT_DIR / f"{card.slug}.jpg"
        if out.exists() and out.stat().st_size > 1024:
            skipped += 1
            continue

        url = WIKI_BASE + card.wiki_filename
        try:
            raw = fetch(url)
            if len(raw) < 1024:
                raise RuntimeError(f"suspiciously small payload: {len(raw)} bytes")
            data = resize_jpeg(raw, TARGET_WIDTH)
            out.write_bytes(data)
            downloaded += 1
            print(f"  ✓ {card.slug}  ({len(data) // 1024} KB, was {len(raw) // 1024} KB)")
            time.sleep(THROTTLE_SECONDS)
        except Exception as e:
            failed += 1
            print(f"  ✗ {card.slug}  ({card.wiki_filename}): {e}", file=sys.stderr)

    print(
        f"\nTarot fetch: {downloaded} downloaded, {skipped} already present, "
        f"{failed} failed. Output: {OUT_DIR}"
    )
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
