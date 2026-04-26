"""
Render an I Ching cast to a single PNG attachment.

The cast is fully procedural — no card images on disk. We draw on a
parchment-tinted canvas with a deep ink palette, the hexagram name in
brush-weight CJK type, the six lines as solid/broken bars (with X/○
markers next to changing lines), and — when there are changing lines — a
side-by-side primary→transformed render with a connecting arrow.

Returns a base64-encoded PNG.
"""

from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

from .iching_data import Hexagram, HEXAGRAM_BY_LINES

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Palette — warm ink-on-parchment. The accent red is reserved for
# changing-line markers and the "→" between primary and transformed.
# ---------------------------------------------------------------------------
PARCHMENT_BASE = (244, 232, 208)   # warm cream
PARCHMENT_EDGE = (220, 200, 168)   # darker cream at the edges
INK            = (28, 22, 18)      # near-black warm
INK_SOFT       = (90, 70, 56)      # subtitle text
ACCENT         = (154, 42, 38)     # cinnabar red
HAIR           = (160, 130, 100)   # muted parchment-line color


# ---------------------------------------------------------------------------
# Font discovery. Debian's fonts-noto-cjk + fonts-dejavu-core are installed
# in the Dockerfile. We probe a list of likely paths so the module also
# works on a host with different font layouts (or in tests where the
# packages aren't installed at all).
# ---------------------------------------------------------------------------
_CJK_FONT_CANDIDATES = [
    # Serif first — gives a calligraphic / woodblock feel that fits the
    # parchment aesthetic better than the blockier sans variants.
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSerifCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Bold.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
]
_SERIF_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSerif.ttf",
    "/usr/share/fonts/opentype/noto/NotoSerif-Bold.ttf",
]
_SANS_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
]


def _first_existing(paths: list[str]) -> Optional[str]:
    for p in paths:
        if Path(p).is_file():
            return p
    return None


_CJK_PATH = _first_existing(_CJK_FONT_CANDIDATES)
_SERIF_PATH = _first_existing(_SERIF_CANDIDATES)
_SANS_PATH = _first_existing(_SANS_CANDIDATES)

_FONT_CACHE: dict[tuple[str, int], ImageFont.FreeTypeFont] = {}


def _font(path: Optional[str], size: int) -> ImageFont.ImageFont:
    if path is None:
        return ImageFont.load_default()
    key = (path, size)
    cached = _FONT_CACHE.get(key)
    if cached is not None:
        return cached
    try:
        f = ImageFont.truetype(path, size)
    except Exception as e:
        logger.warning(f"failed to load font {path}: {e}")
        return ImageFont.load_default()
    _FONT_CACHE[key] = f
    return f


def cjk(size: int) -> ImageFont.ImageFont:
    return _font(_CJK_PATH, size)


def serif(size: int) -> ImageFont.ImageFont:
    return _font(_SERIF_PATH, size)


def sans(size: int) -> ImageFont.ImageFont:
    return _font(_SANS_PATH, size)


# ---------------------------------------------------------------------------
# Cast result — what comes out of the casting code in iching_command.py.
# Each line value is one of {6, 7, 8, 9}:
#   6 = old yin   (yin that is changing → becomes yang)
#   7 = young yang (stable)
#   8 = young yin  (stable)
#   9 = old yang  (yang that is changing → becomes yin)
# Stored bottom-up, exactly like the static `lines` field on Hexagram.
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Cast:
    line_values: tuple[int, int, int, int, int, int]   # bottom..top, each in {6,7,8,9}
    question: Optional[str] = None

    def primary_bits(self) -> str:
        """Bottom-up bit string for the *current* state."""
        return "".join("1" if v in (7, 9) else "0" for v in self.line_values)

    def transformed_bits(self) -> str:
        """Bottom-up bit string after every changing line flips."""
        return "".join(
            ("0" if v == 9 else "1" if v == 6 else "1" if v == 7 else "0")
            for v in self.line_values
        )

    def changing_indices(self) -> list[int]:
        """Zero-indexed positions of changing lines (0=bottom, 5=top)."""
        return [i for i, v in enumerate(self.line_values) if v in (6, 9)]

    def primary(self) -> Hexagram:
        return HEXAGRAM_BY_LINES[self.primary_bits()]

    def transformed(self) -> Optional[Hexagram]:
        idx = self.changing_indices()
        if not idx:
            return None
        return HEXAGRAM_BY_LINES[self.transformed_bits()]


# ---------------------------------------------------------------------------
# Canvas constants. The "panel" is the rectangle that holds one hexagram
# (lines + name + label). For a no-changes cast we render a single panel
# centered on the canvas. With changes, we render two panels side-by-side
# with a small arrow rail between them.
# ---------------------------------------------------------------------------
PANEL_W = 460
PANEL_H = 600           # tightened — was 760, then 660; 600 lands the bottom border just under the keywords
PANEL_PAD = 36          # outer canvas padding around the panel(s)
PANEL_GAP = 56          # space between primary and transformed panels

LINE_WIDTH    = 280     # full bar width for a yang line
LINE_HEIGHT   = 18      # bar thickness
LINE_GAP_V    = 18      # vertical space between successive lines
YIN_GAP       = 36      # gap in the middle of a yin line
LINE_BLOCK_H  = 6 * LINE_HEIGHT + 5 * LINE_GAP_V

# Seal size for the upper-right hexagram number stamp.
SEAL_SIZE = 56
SEAL_INSET = 18         # distance from panel top-right corner


def _make_parchment(width: int, height: int) -> Image.Image:
    """Background: a soft radial vignette from cream to a darker cream.

    Built via PIL's bundled radial gradient (256x256 single-channel) used as
    a blend mask between two solid-color images — orders of magnitude faster
    than a per-pixel Python loop, and the result is indistinguishable.
    """
    base = Image.new("RGB", (width, height), PARCHMENT_BASE)
    edge = Image.new("RGB", (width, height), PARCHMENT_EDGE)
    # radial_gradient("L") is a 256×256 single-channel ramp, white in the
    # center and black at the edges. Resize to canvas, then *invert* for use
    # as the edge layer's alpha (so the edge color shows where the gradient
    # is darkest, i.e. at the canvas borders).
    mask = Image.radial_gradient("L").resize((width, height), Image.Resampling.BILINEAR)
    # Soften the falloff slightly so the vignette is gentle, not a halo.
    mask = mask.point(lambda v: int(min(255, v * 0.95)))
    composed = Image.composite(edge, base, mask)
    # A single light blur knocks down any banding from the resize.
    return composed.filter(ImageFilter.GaussianBlur(radius=0.6))


# Cache rendered backgrounds per size — they're expensive to build (per-pixel
# vignette) and identical for every cast at a given canvas size.
_PARCHMENT_CACHE: dict[tuple[int, int], Image.Image] = {}


def _parchment(width: int, height: int) -> Image.Image:
    key = (width, height)
    cached = _PARCHMENT_CACHE.get(key)
    if cached is not None:
        return cached.copy()
    img = _make_parchment(width, height)
    _PARCHMENT_CACHE[key] = img
    return img.copy()


def _draw_line_bar(
    d: ImageDraw.ImageDraw,
    *,
    cx: int,
    cy: int,
    yang: bool,
    changing: bool,
) -> None:
    """Draw one hexagram line centered at (cx, cy).

    - Yang: single solid bar of width LINE_WIDTH.
    - Yin:  two short bars with YIN_GAP between them.
    - changing=True: an accent-coloured marker drawn to the right
      (○ for yang→yin, × for yin→yang) plus the bar in the same accent
      so the eye lands on it.
    """
    color = ACCENT if changing else INK
    half_w = LINE_WIDTH // 2
    half_h = LINE_HEIGHT // 2

    if yang:
        d.rounded_rectangle(
            (cx - half_w, cy - half_h, cx + half_w, cy + half_h),
            radius=4,
            fill=color,
        )
    else:
        seg_w = (LINE_WIDTH - YIN_GAP) // 2
        d.rounded_rectangle(
            (cx - half_w, cy - half_h, cx - half_w + seg_w, cy + half_h),
            radius=4,
            fill=color,
        )
        d.rounded_rectangle(
            (cx + half_w - seg_w, cy - half_h, cx + half_w, cy + half_h),
            radius=4,
            fill=color,
        )

    if changing:
        marker_x = cx + half_w + 28
        marker_r = LINE_HEIGHT
        if yang:
            # yang → yin: hollow circle
            d.ellipse(
                (marker_x - marker_r, cy - marker_r,
                 marker_x + marker_r, cy + marker_r),
                outline=ACCENT,
                width=3,
            )
        else:
            # yin → yang: cross
            d.line(
                (marker_x - marker_r, cy - marker_r,
                 marker_x + marker_r, cy + marker_r),
                fill=ACCENT, width=3,
            )
            d.line(
                (marker_x - marker_r, cy + marker_r,
                 marker_x + marker_r, cy - marker_r),
                fill=ACCENT, width=3,
            )


def _draw_text_centered(
    d: ImageDraw.ImageDraw,
    text: str,
    cx: int,
    top: int,
    font: ImageFont.ImageFont,
    color=INK,
) -> tuple[int, int]:
    """Center-align `text` on column `cx`, with its top at `top`. Returns (w, h)."""
    bbox = d.textbbox((0, 0), text, font=font)
    w = bbox[2] - bbox[0]
    h = bbox[3] - bbox[1]
    # textbbox includes the empty space above ascenders; shift up slightly so
    # `top` matches the visual top of the first glyph.
    d.text((cx - w // 2 - bbox[0], top - bbox[1]), text, fill=color, font=font)
    return w, h


def _hairline(d: ImageDraw.ImageDraw, x0: int, y: int, x1: int, color=HAIR) -> None:
    d.line((x0, y, x1, y), fill=color, width=1)


# ---------------------------------------------------------------------------
# Mini trigram glyph drawn procedurally. We can't trust DejaVu Sans / Noto
# CJK to ship the eight Unicode trigram codepoints (U+2630..U+2637) in any
# distinguishable form — DejaVu in particular renders all eight as a
# fallback "three lines" placeholder, so every trigram label looks like
# every other. Drawing them ourselves takes about 8 lines, matches the
# aesthetic of the main hexagram, and decouples us from font coverage.
# ---------------------------------------------------------------------------
MINI_BAR_W = 26
MINI_BAR_H = 3
MINI_BAR_GAP = 4         # vertical space between the 3 stacked lines
MINI_YIN_GAP = 6         # horizontal break in a yin line
MINI_TRIGRAM_W = MINI_BAR_W
MINI_TRIGRAM_H = MINI_BAR_H * 3 + MINI_BAR_GAP * 2


def _draw_mini_trigram(
    d: ImageDraw.ImageDraw,
    *,
    x: int,
    top: int,
    bits: str,
    color=INK_SOFT,
) -> None:
    """Draw a 3-line trigram with its top-left at (x, top).

    `bits` is bottom-up (matches the rest of the codebase). The trigram
    is drawn TOP-DOWN visually, so iterate the bit string in reverse.
    """
    for visual_row in range(3):
        line_index = 2 - visual_row
        yang = bits[line_index] == "1"
        cy = top + visual_row * (MINI_BAR_H + MINI_BAR_GAP)
        if yang:
            d.rectangle((x, cy, x + MINI_BAR_W - 1, cy + MINI_BAR_H - 1), fill=color)
        else:
            seg_w = (MINI_BAR_W - MINI_YIN_GAP) // 2
            d.rectangle((x, cy, x + seg_w - 1, cy + MINI_BAR_H - 1), fill=color)
            d.rectangle(
                (x + MINI_BAR_W - seg_w, cy,
                 x + MINI_BAR_W - 1, cy + MINI_BAR_H - 1),
                fill=color,
            )


def _autofit_font(
    d: ImageDraw.ImageDraw,
    text: str,
    *,
    font_path: Optional[str],
    max_width: int,
    sizes: tuple[int, ...],
) -> ImageFont.ImageFont:
    """Pick the largest size from `sizes` whose rendered width fits in `max_width`.

    Falls back to the smallest size in `sizes` if nothing fits — better to
    risk a tiny overflow at the smallest size than to crash, and the smaller
    size will overflow much less than the larger ones.
    """
    for size in sizes:
        f = _font(font_path, size)
        if d.textbbox((0, 0), text, font=f)[2] <= max_width:
            return f
    return _font(font_path, sizes[-1])


_CHINESE_DIGITS = "〇一二三四五六七八九"


def _chinese_number(n: int) -> str:
    """Convert 1..99 to Chinese numerals.

    Used on the seal stamp; the I Ching only needs 1..64.
    """
    if n < 10:
        return _CHINESE_DIGITS[n]
    tens, ones = divmod(n, 10)
    if tens == 1:
        return "十" + (_CHINESE_DIGITS[ones] if ones else "")
    if ones == 0:
        return _CHINESE_DIGITS[tens] + "十"
    return _CHINESE_DIGITS[tens] + "十" + _CHINESE_DIGITS[ones]


def _draw_seal(
    canvas: Image.Image,
    *,
    x: int,
    y: int,
    number: int,
    size: int = SEAL_SIZE,
) -> None:
    """Cinnabar 'name seal' carrying the hexagram number in Chinese.

    Standard East-Asian print/manuscript marker: a red square with white
    characters carved into it. We draw it with a slightly textured fill
    (a touch of cream noise punched out of the red) so it reads as an
    inked stamp rather than a flat rectangle.
    """
    # 1. Solid red square with rounded corners
    seal = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    sd = ImageDraw.Draw(seal)
    sd.rounded_rectangle((0, 0, size - 1, size - 1), radius=4, fill=ACCENT + (255,))

    # 2. Knock a few darker speckles into the seal so it looks like a worn
    #    chop print rather than a vector rectangle.
    import random as _r
    rng = _r.Random((number * 7919) ^ 0xBADC0FFEE)
    for _ in range(size * size // 14):
        sx = rng.randrange(2, size - 2)
        sy = rng.randrange(2, size - 2)
        sd.point((sx, sy), fill=(120, 32, 30, 255))

    # 3. The number in Chinese, in cream, centered. Two-character numbers
    #    (e.g. 二十七 = 27) need a smaller font than single-character ones.
    label = _chinese_number(number)
    if len(label) == 1:
        font_size = int(size * 0.62)
    elif len(label) == 2:
        font_size = int(size * 0.42)
    else:
        font_size = int(size * 0.32)
    f = cjk(font_size)
    bbox = sd.textbbox((0, 0), label, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    sd.text(
        (size // 2 - tw // 2 - bbox[0], size // 2 - th // 2 - bbox[1] - 1),
        label,
        fill=PARCHMENT_BASE + (255,),
        font=f,
    )

    canvas.paste(seal, (x, y), seal)


def _render_panel(
    canvas: Image.Image,
    panel_x: int,
    panel_y: int,
    *,
    hexagram: Hexagram,
    line_values: Optional[tuple[int, ...]] = None,
    show_changes: bool = True,
    title_label: Optional[str] = None,
) -> None:
    """Render one hexagram into the canvas at (panel_x, panel_y).

    line_values is bottom-up. When given, changing lines (6 or 9) are
    rendered with an accent marker — but only if `show_changes` is true,
    so the transformed-side panel can re-use the same machinery while
    drawing the *result* lines as plain (non-marked) bars.
    """
    d = ImageDraw.Draw(canvas)

    # 1. Optional small label above the title (e.g. "Primary" / "Transforms to")
    cursor_y = panel_y + 8
    cx = panel_x + PANEL_W // 2
    if title_label:
        f = serif(18)
        _, h = _draw_text_centered(d, title_label.upper(), cx, cursor_y, f, color=INK_SOFT)
        cursor_y += h + 14

    # 2. Hexagram number rendered as a cinnabar seal in the top-right —
    #    classic East-Asian print element (chop / 印章).
    _draw_seal(
        canvas,
        x=panel_x + PANEL_W - SEAL_SIZE - SEAL_INSET,
        y=panel_y + SEAL_INSET,
        number=hexagram.number,
    )

    # 3. Chinese name — the visual centerpiece. Two characters need a smaller
    #    font than one to keep the rendered width similar.
    chinese = hexagram.chinese
    cjk_size = 168 if len(chinese) == 1 else 132
    cf = cjk(cjk_size)
    _draw_text_centered(d, chinese, cx, cursor_y, cf, color=INK)
    chinese_bbox = d.textbbox((0, 0), chinese, font=cf)
    cursor_y += (chinese_bbox[3] - chinese_bbox[1]) + 18

    # 4. Pinyin · English title — auto-fit so long names don't overflow
    title = f"{hexagram.pinyin} · {hexagram.name}"
    title_max_w = PANEL_W - 32
    title_font = _autofit_font(
        d, title,
        font_path=_SERIF_PATH,
        max_width=title_max_w,
        sizes=(26, 24, 22, 20, 18),
    )
    _, h = _draw_text_centered(d, title, cx, cursor_y, title_font, color=INK)
    cursor_y += h + 8

    # 5. Trigram pair: procedurally-drawn glyph + name + separator + glyph + name.
    #    Order is upper · lower (matches "Mountain over Thunder" Wilhelm
    #    convention). Layout is computed centered on the panel.
    pair_font = serif(18)
    upper = hexagram.upper_trigram
    lower = hexagram.lower_trigram
    glyph_text_gap = 8           # space between glyph and its name
    sep_text = "  ·  "
    sep_w = d.textbbox((0, 0), sep_text, font=pair_font)[2]
    upper_name_w = d.textbbox((0, 0), upper[2], font=pair_font)[2]
    lower_name_w = d.textbbox((0, 0), lower[2], font=pair_font)[2]
    total_w = (
        MINI_TRIGRAM_W + glyph_text_gap + upper_name_w
        + sep_w
        + MINI_TRIGRAM_W + glyph_text_gap + lower_name_w
    )
    name_top_offset = (MINI_TRIGRAM_H - 14) // 2  # nudge text to align with glyph
    glyph_top = cursor_y + 4
    text_top = glyph_top - name_top_offset

    pen_x = cx - total_w // 2
    _draw_mini_trigram(d, x=pen_x, top=glyph_top, bits=hexagram.upper_bits, color=INK_SOFT)
    pen_x += MINI_TRIGRAM_W + glyph_text_gap
    d.text((pen_x, text_top), upper[2], fill=INK_SOFT, font=pair_font)
    pen_x += upper_name_w
    d.text((pen_x, text_top), sep_text, fill=INK_SOFT, font=pair_font)
    pen_x += sep_w
    _draw_mini_trigram(d, x=pen_x, top=glyph_top, bits=hexagram.lower_bits, color=INK_SOFT)
    pen_x += MINI_TRIGRAM_W + glyph_text_gap
    d.text((pen_x, text_top), lower[2], fill=INK_SOFT, font=pair_font)
    cursor_y = glyph_top + MINI_TRIGRAM_H + 24

    # 6. Hairline above the lines
    _hairline(d, panel_x + 64, cursor_y, panel_x + PANEL_W - 64)
    cursor_y += 28

    # 7. The six lines themselves — drawn TOP-DOWN visually (line 6 at the
    #    top of the column, line 1 at the bottom), so we iterate the
    #    bottom-up bit string in reverse.
    lines_top = cursor_y
    bits = hexagram.lines
    for visual_row in range(6):
        # visual_row 0 = topmost stroke (i.e. line 6, position 5 in bottom-up)
        line_index = 5 - visual_row
        bit = bits[line_index]
        yang = bit == "1"
        changing = False
        if show_changes and line_values is not None:
            v = line_values[line_index]
            changing = v in (6, 9)
        cy = lines_top + visual_row * (LINE_HEIGHT + LINE_GAP_V) + LINE_HEIGHT // 2
        _draw_line_bar(d, cx=cx, cy=cy, yang=yang, changing=changing)
    cursor_y = lines_top + LINE_BLOCK_H + 28

    # 8. Hairline below the lines
    _hairline(d, panel_x + 64, cursor_y, panel_x + PANEL_W - 64)
    cursor_y += 24

    # 9. Keywords (italic-ish look via the serif at smaller size)
    kw_font = serif(18)
    kw = hexagram.keywords
    # Wrap manually if it overflows the panel — usually doesn't.
    max_w = PANEL_W - 64
    if d.textbbox((0, 0), kw, font=kw_font)[2] > max_w:
        # Crude wrap: split on commas.
        parts = [p.strip() for p in kw.split(",")]
        lines: list[str] = []
        cur = ""
        for p in parts:
            cand = (cur + ", " + p).strip(", ") if cur else p
            if d.textbbox((0, 0), cand, font=kw_font)[2] <= max_w:
                cur = cand
            else:
                if cur:
                    lines.append(cur)
                cur = p
        if cur:
            lines.append(cur)
        for line in lines:
            _, h = _draw_text_centered(d, line, cx, cursor_y, kw_font, color=INK_SOFT)
            cursor_y += h + 4
    else:
        _draw_text_centered(d, kw, cx, cursor_y, kw_font, color=INK_SOFT)


def _draw_transformation_indicator(
    canvas: Image.Image,
    *,
    cx: int,
    cy: int,
) -> None:
    """The character 變 (biàn — 'change / transformation') in cinnabar,
    set between the primary and transformed hexagram panels.

    Replaces what used to be a plain arrow. The character carries the
    same semantic load with a much stronger visual identity, and ties
    the layout back to the East-Asian printed-text aesthetic.
    """
    d = ImageDraw.Draw(canvas)
    # The character itself
    f = cjk(58)
    text = "變"
    bbox = d.textbbox((0, 0), text, font=f)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    d.text(
        (cx - tw // 2 - bbox[0], cy - th // 2 - bbox[1]),
        text,
        fill=ACCENT,
        font=f,
    )
    # Two thin hairlines flanking it — emphasizes that it is the bridge
    # between the two panels rather than a free-floating glyph.
    rule_w = 28
    rule_gap = 14
    d.line(
        (cx - tw // 2 - rule_gap - rule_w, cy,
         cx - tw // 2 - rule_gap, cy),
        fill=ACCENT, width=2,
    )
    d.line(
        (cx + tw // 2 + rule_gap, cy,
         cx + tw // 2 + rule_gap + rule_w, cy),
        fill=ACCENT, width=2,
    )


def _draw_changing_summary(
    d: ImageDraw.ImageDraw,
    *,
    cx: int,
    y: int,
    indices: list[int],
) -> None:
    """Below the panels: 'Lines changing: 1, 4' or 'No changing lines'."""
    f = serif(20)
    if not indices:
        text = "No changing lines"
    else:
        # Convert 0-indexed bottom-up to 1-indexed I Ching numbering.
        nums = ", ".join(str(i + 1) for i in indices)
        text = f"Changing lines: {nums}"
    _draw_text_centered(d, text, cx, y, f, color=INK_SOFT)


def compose(cast: Cast) -> str:
    """Compose a Cast into a base64 PNG.

    Single-panel layout when nothing changes; otherwise side-by-side
    primary→transformed with the character 變 (change) between them.
    Canvas height is sized to the panel content + (when relevant) a thin
    changing-lines caption — no fixed dead space at the bottom.
    """
    primary = cast.primary()
    transformed = cast.transformed()
    has_changes = transformed is not None

    if has_changes:
        canvas_w = PANEL_PAD * 2 + PANEL_W * 2 + PANEL_GAP
        # Vertical room for a "Changing lines: 1, 4" caption sitting
        # outside the panel frame. Only present when there is a caption
        # to show — otherwise the absence of red bars carries the meaning.
        caption_h = 56
    else:
        canvas_w = PANEL_PAD * 2 + PANEL_W
        caption_h = 0
    canvas_h = PANEL_PAD * 2 + PANEL_H + caption_h

    canvas = _parchment(canvas_w, canvas_h)
    d = ImageDraw.Draw(canvas)

    # Outer thin border — frames the artifact.
    d.rectangle(
        (PANEL_PAD - 12, PANEL_PAD - 12,
         canvas_w - PANEL_PAD + 12,
         PANEL_PAD - 12 + PANEL_H + 24),
        outline=HAIR, width=1,
    )

    if has_changes:
        _render_panel(
            canvas,
            PANEL_PAD,
            PANEL_PAD,
            hexagram=primary,
            line_values=cast.line_values,
            show_changes=True,
            title_label="Primary",
        )
        _render_panel(
            canvas,
            PANEL_PAD + PANEL_W + PANEL_GAP,
            PANEL_PAD,
            hexagram=transformed,
            line_values=None,
            show_changes=False,
            title_label="Becomes",
        )
        # 變 (biàn — change) sits between the panels, vertically centered
        # on the hexagram line block.
        indicator_cy = PANEL_PAD + 360
        _draw_transformation_indicator(
            canvas,
            cx=canvas_w // 2,
            cy=indicator_cy,
        )
        _draw_changing_summary(
            d,
            cx=canvas_w // 2,
            y=PANEL_PAD + PANEL_H + 28,   # below the panel frame, with breathing room
            indices=cast.changing_indices(),
        )
    else:
        _render_panel(
            canvas,
            PANEL_PAD,
            PANEL_PAD,
            hexagram=primary,
            line_values=cast.line_values,
            show_changes=False,
        )
        # No bottom caption — the empty parchment IS the statement that
        # nothing is changing.

    buf = io.BytesIO()
    canvas.save(buf, format="PNG", optimize=True)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def fonts_ready() -> bool:
    """True iff at least the CJK font is available — without it we can't
    render the Chinese name and the panel is incomplete."""
    return _CJK_PATH is not None
