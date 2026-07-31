"""
Tests for inbound-image vision plumbing.

Covers:
  * Bot.vision_enabled round-trip through BotRegistry
  * _read_inbound_image_attachments filtering (mime, size, missing files)
  * Multimodal payload construction is well-formed when vision is active
"""

import base64
import tempfile
from pathlib import Path

import pytest

from src.bots import Bot, BotRegistry
from src.signal.handler import (
    VISION_ALLOWED_MIMES,
    VISION_MAX_PER_IMAGE_BYTES,
    _read_inbound_image_attachments,
)
from src.signal.message_text import normalize_signal_text


@pytest.fixture
async def registry():
    with tempfile.TemporaryDirectory() as d:
        reg = BotRegistry(db_path=str(Path(d) / "bots.db"))
        yield reg


@pytest.fixture
def attachments_tmp(monkeypatch):
    with tempfile.TemporaryDirectory() as d:
        monkeypatch.setenv("SIGNAL_ATTACHMENTS_DIR", d)
        yield Path(d)


# ── Bot vision_enabled ─────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_vision_default_off(registry):
    b = Bot(id=None, slug="vis-default", display_name="V", aliases=["v"])
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.vision_enabled is False


@pytest.mark.asyncio
async def test_vision_persists_through_upsert(registry):
    b = Bot(
        id=None, slug="vis-on", display_name="VOn", aliases=["v"],
        vision_enabled=True,
    )
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.vision_enabled is True

    reloaded.vision_enabled = False
    await registry.upsert(reloaded)
    again = await registry.get(bid)
    assert again is not None
    assert again.vision_enabled is False


# ── _read_inbound_image_attachments ────────────────────────────────────────

def test_signal_object_placeholder_uses_sticker_emoji_metadata():
    assert normalize_signal_text(
        "\ufffc", {"sticker": {"emoji": "🚂"}},
    ) == "[sticker: 🚂]"


def test_signal_object_placeholder_has_honest_generic_fallback():
    assert normalize_signal_text("look \ufffc") == (
        "look [unrendered Signal object]"
    )


def test_real_train_emoji_is_not_rewritten():
    assert normalize_signal_text("model train 🚂") == "model train 🚂"


def test_structured_mention_phone_token_is_removed():
    text = "[+16467699190] last thing, what car do you drive?"
    data_message = {"mentions": [{"number": "+16467699190"}]}
    assert normalize_signal_text(text, data_message) == (
        "last thing, what car do you drive?"
    )


def test_unstructured_bracketed_phone_is_preserved():
    text = "Call [+16467699190] tomorrow"
    assert normalize_signal_text(text, {"mentions": []}) == text

def test_no_attachments_returns_empty(attachments_tmp):
    assert _read_inbound_image_attachments({}) == []
    assert _read_inbound_image_attachments({"attachments": []}) == []
    assert _read_inbound_image_attachments({"attachments": None}) == []


def test_skips_non_image_mime(attachments_tmp):
    (attachments_tmp / "doc.pdf").write_bytes(b"%PDF-")
    data_msg = {
        "attachments": [
            {"id": "doc.pdf", "contentType": "application/pdf"},
        ]
    }
    assert _read_inbound_image_attachments(data_msg) == []


def test_reads_jpeg(attachments_tmp):
    raw = b"\xff\xd8\xff\xe0fake-jpeg-bytes"
    (attachments_tmp / "img.jpg").write_bytes(raw)
    data_msg = {
        "attachments": [
            {"id": "img.jpg", "contentType": "image/jpeg", "filename": "shot.jpg"},
        ]
    }
    out = _read_inbound_image_attachments(data_msg)
    assert len(out) == 1
    assert out[0]["mime"] == "image/jpeg"
    assert out[0]["filename"] == "shot.jpg"
    assert base64.b64decode(out[0]["data_b64"]) == raw


def test_id_glob_fallback(attachments_tmp):
    """Some signal-cli configs store the file as `<id>.<ext>` even though
    the dataMessage.id field is just the stem. Confirm we still find it."""
    raw = b"\x89PNG\r\n\x1a\n"
    (attachments_tmp / "abc123.png").write_bytes(raw)
    data_msg = {
        "attachments": [
            {"id": "abc123", "contentType": "image/png"},
        ]
    }
    out = _read_inbound_image_attachments(data_msg)
    assert len(out) == 1
    assert out[0]["mime"] == "image/png"


def test_skips_missing_file(attachments_tmp):
    data_msg = {
        "attachments": [
            {"id": "ghost.jpg", "contentType": "image/jpeg"},
        ]
    }
    assert _read_inbound_image_attachments(data_msg) == []


def test_size_cap(attachments_tmp, monkeypatch):
    """Anything above VISION_MAX_PER_IMAGE_BYTES is dropped silently."""
    monkeypatch.setattr("src.signal.handler.VISION_MAX_PER_IMAGE_BYTES", 100)
    (attachments_tmp / "big.jpg").write_bytes(b"x" * 200)
    (attachments_tmp / "small.jpg").write_bytes(b"y" * 50)
    data_msg = {
        "attachments": [
            {"id": "big.jpg", "contentType": "image/jpeg"},
            {"id": "small.jpg", "contentType": "image/jpeg"},
        ]
    }
    out = _read_inbound_image_attachments(data_msg)
    assert len(out) == 1
    assert out[0]["filename"] == "small.jpg"


def test_max_images_cap(attachments_tmp):
    """The 5th image should be dropped silently."""
    for i in range(6):
        (attachments_tmp / f"img{i}.jpg").write_bytes(b"x")
    data_msg = {
        "attachments": [
            {"id": f"img{i}.jpg", "contentType": "image/jpeg"}
            for i in range(6)
        ]
    }
    out = _read_inbound_image_attachments(data_msg, max_images=4)
    assert len(out) == 4
    assert [o["filename"] for o in out] == [f"img{i}.jpg" for i in range(4)]


def test_media_bypasses_research_and_tool_bot_modes():
    """Regression: when vision OR audio is active on a turn AND the bot is
    in research/tool_bot mode, ask_command must short-circuit the handoff
    so the writer sees the media directly. The sibling model can't see
    images or hear clips (text-only context) and would mislead the writer
    with "I can't see/hear that" notes the writer then parrots. Media
    turns must route writer-direct; subsequent text-only turns keep the
    handoff.

    This test verifies the source-level fork — running the full ask path
    needs the whole LLM stack mocked, which is out of scope. Both
    booleans must appear in both guards: wiring vision and forgetting
    audio (or vice versa) is the easy mistake here.
    """
    import src.commands.ask_command as ask_mod
    src_text = open(ask_mod.__file__, encoding="utf-8").read()
    assert "Multimodal short-circuit" in src_text, (
        "Multimodal short-circuit comment missing — bypass logic may "
        "have been removed. Media turns will be incorrectly routed "
        "through deep_think notes."
    )
    assert "if use_research_mode and (vision_active or audio_active):" in src_text
    assert "if use_tool_bot_mode and (vision_active or audio_active):" in src_text
    assert "use_research_mode = False" in src_text
    assert "use_tool_bot_mode = False" in src_text


def test_allowed_mimes_constant():
    """Regression guard against accidentally adding application/pdf etc."""
    assert VISION_ALLOWED_MIMES == frozenset({
        "image/jpeg", "image/png", "image/webp", "image/gif",
    })
    assert VISION_MAX_PER_IMAGE_BYTES > 0
