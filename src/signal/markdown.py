"""
Convert standard markdown into the styled-text syntax that
bbernhard/signal-cli-rest-api accepts when sent with text_mode="styled".

The REST API's parser (utils/textstyleparser.go) is the authority:
  **bold**     → BOLD
  *italic*     → ITALIC
  ~strike~     → STRIKETHROUGH
  `mono`       → MONOSPACE
  ||spoiler||  → SPOILER

Critically, single-underscore `_italic_` is NOT recognised by this parser
— it gets sent verbatim. So GFM-style underscore italics need rewriting
to single asterisks, and double-asterisk bolds must be PRESERVED (any
collapse to a single asterisk produces italic, not bold).

Order matters — bold (`**…**`, `__…__`) is converted via placeholders
*before* italic, so we don't accidentally split a bold marker into two
italic ones.
"""

import re

# Sentinels — chosen to never appear in normal text and to be regex-safe.
_BOLD_OPEN = "\x00B<\x00"
_BOLD_CLOSE = "\x00B>\x00"
_ITALIC_OPEN = "\x00I<\x00"
_ITALIC_CLOSE = "\x00I>\x00"


_FENCE_RE = re.compile(r"```[a-zA-Z0-9_+-]*\s*\n?(.*?)```", re.DOTALL)
_FENCE_TILDE_RE = re.compile(r"~~~[a-zA-Z0-9_+-]*\s*\n?(.*?)~~~", re.DOTALL)
_INLINE_CODE_RE = re.compile(r"`([^`\n]+)`")

_HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*#*\s*$", re.MULTILINE)
_BULLET_RE = re.compile(r"^(\s*)[-*+]\s+", re.MULTILINE)
_BLOCKQUOTE_RE = re.compile(r"^\s{0,3}>\s?", re.MULTILINE)
_HR_RE = re.compile(r"^\s{0,3}([-*_])\s*\1\s*\1[\s\1]*$", re.MULTILINE)

# Image first (so the trailing ) doesn't get eaten by link), then link.
_IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)\s]+)(?:\s+\"[^\"]*\")?\)")

# Bold: **text** or __text__ — non-greedy, no newlines inside.
_BOLD_STAR_RE = re.compile(r"\*\*(?!\s)([^*\n]+?)(?<!\s)\*\*")
_BOLD_UND_RE = re.compile(r"__(?!\s)([^_\n]+?)(?<!\s)__")

# Italic: *text* or _text_ (single). Negative lookarounds on the opening
# delimiter ensure we don't match part of a still-untouched bold marker.
_ITALIC_STAR_RE = re.compile(r"(?<![*\w])\*(?!\s)([^*\n]+?)(?<!\s)\*(?![*\w])")
_ITALIC_UND_RE = re.compile(r"(?<![_\w])_(?!\s)([^_\n]+?)(?<!\s)_(?![_\w])")

_STRIKE_RE = re.compile(r"~~(?!\s)([^~\n]+?)(?<!\s)~~")


def to_signal_styled(text: str) -> str:
    """Rewrite GFM-flavored markdown into Signal styled-text syntax.

    The output should be sent with the signal-cli text_mode=\"styled\"
    parameter. Inline-code segments are protected from rewriting.
    """
    if not text:
        return text

    # ── Step 1: protect code segments so their contents aren't rewritten ──
    #
    # Each protected segment is replaced by a unique placeholder; we splice
    # them back at the end. This avoids mangling code/asterisks inside.
    code_segments: list[str] = []

    def _stash_code(match: re.Match) -> str:
        idx = len(code_segments)
        code_segments.append(match.group(1))
        return f"\x00CODE{idx}\x00"

    text = _FENCE_RE.sub(_stash_code, text)
    text = _FENCE_TILDE_RE.sub(_stash_code, text)

    # Inline code stays as `…` (Signal supports it natively) but we still
    # protect its contents from emphasis rewrites.
    inline_segments: list[str] = []

    def _stash_inline(match: re.Match) -> str:
        idx = len(inline_segments)
        inline_segments.append(match.group(1))
        return f"\x00ICODE{idx}\x00"

    text = _INLINE_CODE_RE.sub(_stash_inline, text)

    # ── Step 2: block-level rewrites ──
    text = _HR_RE.sub("──────────", text)
    text = _HEADING_RE.sub(rf"{_BOLD_OPEN}\1{_BOLD_CLOSE}", text)
    text = _BLOCKQUOTE_RE.sub("", text)
    text = _BULLET_RE.sub(r"\1• ", text)

    # ── Step 3: links + images ──
    # Images: keep alt text; drop URL (Signal can't render).
    text = _IMAGE_RE.sub(lambda m: f"[image: {m.group(1)}]" if m.group(1) else "[image]", text)
    # Links: "text (url)" so users still see the URL.
    text = _LINK_RE.sub(r"\1 (\2)", text)

    # ── Step 4: bold first (with placeholders), then italic, then strike ──
    text = _BOLD_STAR_RE.sub(rf"{_BOLD_OPEN}\1{_BOLD_CLOSE}", text)
    text = _BOLD_UND_RE.sub(rf"{_BOLD_OPEN}\1{_BOLD_CLOSE}", text)
    text = _ITALIC_STAR_RE.sub(rf"{_ITALIC_OPEN}\1{_ITALIC_CLOSE}", text)
    text = _ITALIC_UND_RE.sub(rf"{_ITALIC_OPEN}\1{_ITALIC_CLOSE}", text)
    text = _STRIKE_RE.sub(r"~\1~", text)

    # ── Step 5: swap placeholders to Signal syntax ──
    # Bold uses **double**, italic uses *single*. The asterisk count is the
    # ONLY thing the REST API parser distinguishes between bold and italic.
    text = text.replace(_BOLD_OPEN, "**").replace(_BOLD_CLOSE, "**")
    text = text.replace(_ITALIC_OPEN, "*").replace(_ITALIC_CLOSE, "*")

    # ── Step 6: restore protected code segments ──
    for idx, content in enumerate(inline_segments):
        text = text.replace(f"\x00ICODE{idx}\x00", f"`{content}`")

    for idx, content in enumerate(code_segments):
        # Keep block code on its own lines, framed by backticks for clarity.
        snippet = content.rstrip("\n")
        text = text.replace(f"\x00CODE{idx}\x00", f"`{snippet}`" if "\n" not in snippet else snippet)

    return text
