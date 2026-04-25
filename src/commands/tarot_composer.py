"""
Compose tarot spreads into a single image attachment.

Each draw is a list of (Card, reversed: bool, position_label: str | None).
The composer loads card images from ``assets/tarot/{slug}.jpg``, rotates
reversed cards 180°, and arranges them on a canvas appropriate to the
spread shape:

  - single: one card, centered
  - three: three cards in a row (e.g. past / present / future)
  - celtic: traditional Celtic Cross (cross + staff)

Returns base64-encoded PNG bytes — same shape the chart command uses.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFont

from .tarot_data import Card

logger = logging.getLogger(__name__)


# Canonical asset dir: data/tarot/{slug}.jpg.
# `data/` is the persistent-volume mount in docker-compose, so the deck
# downloads once on first startup and survives rebuilds. The path is
# resolved from this file's location so it works both inside the container
# (/app/data/tarot/) and during local development (<repo>/data/tarot/).
ASSETS_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "tarot"

CARD_W = 240          # rendered card width in the spread image
CARD_H = 402          # 240 * (838/500), preserving the source 500x838 ratio
CARD_GAP = 24         # spacing between cards in row layouts
LABEL_GAP = 28        # space below the card for the position label
BG = (16, 16, 22)     # near-black, slightly cool — reads well in Signal
FG = (235, 230, 215)  # warm off-white for labels


@dataclass(frozen=True)
class Draw:
    """A single drawn card within a spread."""
    card: Card
    reversed: bool
    position: Optional[str] = None  # e.g. "Past", "Crowning", "Outcome"


def _load_card(card: Card, reversed: bool) -> Image.Image:
    path = ASSETS_DIR / f"{card.slug}.jpg"
    if not path.exists():
        raise FileNotFoundError(f"Tarot asset missing: {path}")
    img = Image.open(path).convert("RGB")
    img = img.resize((CARD_W, CARD_H), Image.Resampling.LANCZOS)
    if reversed:
        img = img.rotate(180)
    return img


def _font(size: int) -> ImageFont.ImageFont:
    """Best available system font; falls back to PIL's bundled default."""
    candidates = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
    ]
    for p in candidates:
        if Path(p).exists():
            try:
                return ImageFont.truetype(p, size)
            except Exception:
                continue
    return ImageFont.load_default()


def _draw_label(canvas: ImageDraw.ImageDraw, text: str, cx: int, top: int,
                font: ImageFont.ImageFont, color=FG,
                background: Optional[tuple[int, int, int]] = None) -> None:
    """Render `text` centered at column `cx`, with its top at `top`.

    When `background` is set, a filled rounded rectangle is drawn behind
    the text first — used for labels that sit on top of card artwork
    (e.g. the rotated crossing card in a Celtic Cross), where the busy
    art behind the label kills contrast.
    """
    bbox = canvas.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    x = cx - w // 2
    if background is not None:
        pad_x = 8
        pad_y = 4
        canvas.rounded_rectangle(
            (x - pad_x, top - pad_y, x + w + pad_x, top + h + pad_y),
            radius=6,
            fill=background,
        )
    canvas.text((x, top), text, fill=color, font=font)


def _label_for(d: Draw) -> str:
    suffix = " (rev)" if d.reversed else ""
    if d.position:
        return f"{d.position}: {d.card.name}{suffix}"
    return f"{d.card.name}{suffix}"


def compose_single(draw: Draw) -> Image.Image:
    pad = 32
    label_h = 40
    canvas = Image.new("RGB",
                       (CARD_W + pad * 2, CARD_H + pad * 2 + label_h),
                       BG)
    canvas.paste(_load_card(draw.card, draw.reversed), (pad, pad))
    d = ImageDraw.Draw(canvas)
    _draw_label(d, _label_for(draw),
                cx=pad + CARD_W // 2, top=pad + CARD_H + 6,
                font=_font(20))
    return canvas


def compose_row(draws: list[Draw]) -> Image.Image:
    pad = 32
    label_h = 50
    n = len(draws)
    width = pad * 2 + n * CARD_W + (n - 1) * CARD_GAP
    height = pad * 2 + CARD_H + label_h
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    font = _font(20)
    for i, draw in enumerate(draws):
        x = pad + i * (CARD_W + CARD_GAP)
        canvas.paste(_load_card(draw.card, draw.reversed), (x, pad))
        _draw_label(d, _label_for(draw),
                    cx=x + CARD_W // 2, top=pad + CARD_H + 8,
                    font=font)
    return canvas


# Celtic Cross layout. The traditional positions are:
#   1. Significator / Present (center)
#   2. Crossing / Challenge   (rotated 90° on top of #1)
#   3. Foundation             (below #1)
#   4. Recent Past            (left of #1)
#   5. Crown / Possible       (above #1)
#   6. Near Future            (right of #1)
#   7. Self                   (staff, bottom)
#   8. Environment            (staff, second)
#   9. Hopes & Fears          (staff, third)
#  10. Outcome                (staff, top)
def compose_celtic(draws: list[Draw]) -> Image.Image:
    assert len(draws) == 10, "Celtic Cross requires 10 cards"

    # Layout constants — tuned so the cross + staff fit comfortably with
    # room for labels under each card.
    pad = 40
    inter = 28          # vertical/horizontal spacing between cross arms
    staff_gap = 36      # horizontal gap between cross and staff
    label_h = 36

    # Cross block: 3 columns × 3 rows of card slots (column 0, 1, 2;
    # row 0, 1, 2). Card 1 sits in (col=1, row=1); card 2 is the
    # 90°-rotated overlay; cards 3/4/5/6 are at S/W/N/E of card 1.
    col_w = CARD_W
    row_h = CARD_H + label_h
    cross_w = col_w * 3 + inter * 2
    cross_h = row_h * 3 + inter * 2

    # Staff block: 4 cards stacked vertically.
    staff_w = col_w
    staff_h = (CARD_H + label_h) * 4 + inter * 3

    width = pad * 2 + cross_w + staff_gap + staff_w
    height = pad * 2 + max(cross_h, staff_h)
    canvas = Image.new("RGB", (width, height), BG)
    d = ImageDraw.Draw(canvas)
    font = _font(18)

    # Cross origin (top-left of the cross block)
    cx0 = pad
    cy0 = pad + (height - pad * 2 - cross_h) // 2

    def cross_slot(col: int, row: int) -> tuple[int, int]:
        x = cx0 + col * (col_w + inter)
        y = cy0 + row * (row_h + inter)
        return x, y

    def place(draw: Draw, x: int, y: int, *, rotate90: bool = False) -> None:
        img = _load_card(draw.card, draw.reversed)
        if rotate90:
            img = img.rotate(90, expand=True)
            # When rotated, the image is now CARD_H wide x CARD_W tall.
            # Center it on the slot of the cross center card.
            slot_cx = x + col_w // 2
            slot_cy = y + CARD_H // 2
            paste_x = slot_cx - img.width // 2
            paste_y = slot_cy - img.height // 2
            canvas.paste(img, (paste_x, paste_y))
            # Label sits below the rotated card but lands on top of card #1's
            # artwork — paint a panel behind it so it's still readable
            # against the busy art.
            _draw_label(d, _label_for(draw),
                        cx=slot_cx, top=paste_y + img.height + 6, font=font,
                        background=BG)
        else:
            canvas.paste(img, (x, y))
            _draw_label(d, _label_for(draw),
                        cx=x + col_w // 2, top=y + CARD_H + 6, font=font)

    # 1: center (col=1, row=1)
    x1, y1 = cross_slot(1, 1)
    place(draws[0], x1, y1)
    # 2: crossing — rotated 90°, on top of #1
    place(draws[1], x1, y1, rotate90=True)
    # 3: below #1
    x3, y3 = cross_slot(1, 2)
    place(draws[2], x3, y3)
    # 4: left of #1
    x4, y4 = cross_slot(0, 1)
    place(draws[3], x4, y4)
    # 5: above #1
    x5, y5 = cross_slot(1, 0)
    place(draws[4], x5, y5)
    # 6: right of #1
    x6, y6 = cross_slot(2, 1)
    place(draws[5], x6, y6)

    # Staff origin
    sx0 = cx0 + cross_w + staff_gap
    sy0 = pad + (height - pad * 2 - staff_h) // 2

    for i in range(4):
        # Staff bottom-up: 7 at bottom, 10 at top — so iterate reverse.
        draw = draws[6 + i]
        sy = sy0 + (3 - i) * (CARD_H + label_h + inter)
        place(draw, sx0, sy)

    return canvas


def to_png_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def compose(draws: list[Draw], spread: str) -> str:
    """Compose a spread to a base64-encoded PNG.

    spread ∈ {"single", "three", "celtic"}.
    """
    if spread == "single":
        if len(draws) != 1:
            raise ValueError("single spread takes exactly 1 card")
        img = compose_single(draws[0])
    elif spread == "three":
        if len(draws) != 3:
            raise ValueError("three-card spread takes exactly 3 cards")
        img = compose_row(draws)
    elif spread == "celtic":
        if len(draws) != 10:
            raise ValueError("celtic spread takes exactly 10 cards")
        img = compose_celtic(draws)
    else:
        raise ValueError(f"unknown spread: {spread!r}")
    return to_png_b64(img)
