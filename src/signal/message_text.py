"""Normalize structured Signal message bodies for text-only consumers.

Signal can place U+FFFC (OBJECT REPLACEMENT CHARACTER) in ``message`` as a
pointer to structured content carried elsewhere in the dataMessage.  Passing
that invisible placeholder to an LLM loses the useful part of the message and
encourages it to invent context.  This module turns it into a compact, honest
text descriptor while preserving ordinary Unicode emoji verbatim.
"""

from __future__ import annotations

import re
from typing import Optional


OBJECT_REPLACEMENT_CHARACTER = "\ufffc"
GENERIC_SIGNAL_OBJECT = "[unrendered Signal object]"


def _strip_structured_mention_tokens(
    value: str, data_message: Optional[dict],
) -> str:
    """Remove Signal's bracketed account identifiers for real mentions.

    Some signal-cli payloads render a tapped mention as ``[+15551234567]``
    in the text while also supplying the same account in ``mentions``.  The
    structured field already drives bot routing, so retaining the bracketed
    phone in model input is redundant, confusing (it looks like a second
    speaker), and leaks a full phone number into transcripts.  Only tokens
    corroborated by structured mention metadata are removed; a user typing a
    bracketed number normally is left untouched.
    """
    if not isinstance(data_message, dict):
        return value
    for mention in data_message.get("mentions") or []:
        if not isinstance(mention, dict):
            continue
        identifiers = {
            str(mention.get(key) or "").strip()
            for key in ("number", "uuid", "aci", "author")
        }
        for identifier in identifiers - {""}:
            token = re.escape(f"[{identifier}]")
            # Consume adjacent horizontal whitespace, but preserve newlines.
            value = re.sub(
                rf"(?<!\S){token}(?:[ \t]+)?", "", value,
            )
    return value


def _clean_label(value: object, *, limit: int = 80) -> str:
    text = " ".join(str(value or "").split())
    return text[:limit]


def _attachment_label(attachment: dict) -> str:
    mime = _clean_label(attachment.get("contentType"), limit=60).lower()
    filename = _clean_label(
        attachment.get("filename") or attachment.get("fileName"), limit=80,
    )
    if mime.startswith("image/"):
        kind = "image attachment"
    elif mime.startswith("audio/"):
        kind = "audio attachment"
    elif mime.startswith("video/"):
        kind = "video attachment"
    else:
        kind = "file attachment"
    return f"[{kind}: {filename}]" if filename else f"[{kind}]"


def _structured_descriptors(data_message: Optional[dict]) -> list[str]:
    if not isinstance(data_message, dict):
        return []

    descriptors: list[str] = []

    sticker = data_message.get("sticker")
    if isinstance(sticker, dict):
        sticker_data = sticker.get("data")
        nested_emoji = (
            sticker_data.get("emoji")
            if isinstance(sticker_data, dict)
            else None
        )
        emoji = _clean_label(
            sticker.get("emoji") or nested_emoji,
            limit=16,
        )
        descriptors.append(f"[sticker: {emoji}]" if emoji else "[sticker]")

    for mention in data_message.get("mentions") or []:
        if not isinstance(mention, dict):
            continue
        name = _clean_label(
            mention.get("name") or mention.get("displayName"), limit=40,
        )
        number = _clean_label(mention.get("number"), limit=40)
        if name:
            descriptors.append(f"[@mention: {name}]")
        elif number:
            descriptors.append(f"[@mention: ...{number[-4:]}]")
        else:
            descriptors.append("[@mention]")

    for attachment in data_message.get("attachments") or []:
        if isinstance(attachment, dict):
            descriptors.append(_attachment_label(attachment))

    if data_message.get("contacts"):
        descriptors.append("[shared contact]")
    return descriptors


def normalize_signal_text(
    text: object,
    data_message: Optional[dict] = None,
) -> str:
    """Replace U+FFFC placeholders with structured Signal descriptors.

    Real emoji (including 🚂) are ordinary Unicode code points and are left
    alone.  When metadata is incomplete, the fallback names the missing
    object explicitly instead of assigning it an invented meaning.
    """
    value = "" if text is None else str(text)
    value = _strip_structured_mention_tokens(value, data_message)
    if OBJECT_REPLACEMENT_CHARACTER not in value:
        return value

    descriptors = _structured_descriptors(data_message)
    replacements = iter(descriptors)

    def _next_replacement() -> str:
        return next(replacements, GENERIC_SIGNAL_OBJECT)

    return "".join(
        _next_replacement() if ch == OBJECT_REPLACEMENT_CHARACTER else ch
        for ch in value
    )
