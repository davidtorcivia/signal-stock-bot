"""Tests for inbound-audio (voice note) plumbing.

Covers:
  * Bot.audio_enabled / audio_part_style round-trip through BotRegistry
  * _read_inbound_audio_attachments filtering + real ffmpeg transcode
  * Multimodal part construction in both wire dialects
  * The caption-less-voice-note path: a dataMessage with `message: null`
    must survive the empty-text drop and reach dispatch with a descriptor
  * Transcript redaction of the raw base64 an `input_audio` part carries
"""

from __future__ import annotations

import base64
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.bots import Bot, BotRegistry
from src.bots.models import (
    AUDIO_PART_STYLE_AUDIO_URL,
    AUDIO_PART_STYLE_INPUT_AUDIO,
)
from src.commands.ask_command import _audio_parts, _image_parts
from src.llm.transcript import snapshot_payload
from src.signal.audio import (
    AUDIO_ALLOWED_MIMES,
    AUDIO_MAX_SOURCE_BYTES,
    AUDIO_OUTPUT_FORMAT,
    AUDIO_OUTPUT_MIME,
    _parse_duration,
    describe_clips,
    ffmpeg_available,
    format_duration,
)
from src.signal.handler import (
    _read_inbound_audio_attachments,
    has_audio_attachments,
)
from src.signal.pool import SignalHandlerPool

requires_ffmpeg = pytest.mark.skipif(
    not ffmpeg_available(), reason="ffmpeg not installed",
)


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


def _write_test_audio(path: Path, *, seconds: float = 1.0, codec: str = "aac"):
    """Synthesize a real clip with ffmpeg so the transcode path is
    exercised against actual bytes rather than a stub. `aac` reproduces
    what Signal sends for a voice note."""
    ext = {"aac": "m4a", "libmp3lame": "mp3", "pcm_s16le": "wav"}[codec]
    subprocess.run(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
         "-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}",
         "-c:a", codec, str(path.with_suffix(f".{ext}"))],
        check=True, capture_output=True,
    )
    return path.with_suffix(f".{ext}")


def _voice_note_message(att_id: str, *, mime: str = "audio/aac") -> dict:
    return {
        "attachments": [{
            "contentType": mime,
            "id": att_id,
            "filename": None,
            "voiceNote": True,
        }],
    }


# ── Bot flags ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_audio_default_off(registry):
    b = Bot(id=None, slug="aud-default", display_name="A", aliases=["a"])
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.audio_enabled is False
    assert reloaded.audio_part_style == AUDIO_PART_STYLE_INPUT_AUDIO


@pytest.mark.asyncio
async def test_audio_flags_persist_through_upsert(registry):
    b = Bot(
        id=None, slug="aud-on", display_name="AOn", aliases=["a"],
        audio_enabled=True, audio_part_style=AUDIO_PART_STYLE_AUDIO_URL,
    )
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.audio_enabled is True
    assert reloaded.audio_part_style == AUDIO_PART_STYLE_AUDIO_URL


@pytest.mark.asyncio
async def test_unknown_part_style_normalizes_to_default(registry):
    """A hand-edited row must not put a part shape no server understands
    in front of the payload builder."""
    b = Bot(
        id=None, slug="aud-bogus", display_name="AB", aliases=["ab"],
        audio_enabled=True, audio_part_style="wav_url_lol",
    )
    bid = await registry.upsert(b)
    reloaded = await registry.get(bid)
    assert reloaded is not None
    assert reloaded.audio_part_style == AUDIO_PART_STYLE_INPUT_AUDIO


# ── Attachment detection ───────────────────────────────────────────────────

def test_has_audio_attachments_is_metadata_only():
    """Runs before the empty-text drop, so it must not touch the disk."""
    assert has_audio_attachments(_voice_note_message("abc")) is True
    assert has_audio_attachments({"attachments": []}) is False
    assert has_audio_attachments({}) is False


def test_has_audio_attachments_ignores_images_and_idless_entries():
    assert has_audio_attachments({
        "attachments": [{"contentType": "image/jpeg", "id": "x"}],
    }) is False
    assert has_audio_attachments({
        "attachments": [{"contentType": "audio/aac", "id": ""}],
    }) is False


def test_allowed_mimes_cover_signal_voice_notes():
    """Signal records voice notes as AAC in an m4a container; both the
    codec mime and the container mime appear in the wild."""
    for mime in ("audio/aac", "audio/mp4", "audio/m4a", "audio/x-m4a"):
        assert mime in AUDIO_ALLOWED_MIMES
    # Guard against someone widening this to video or arbitrary files.
    assert not any(m.startswith(("video/", "application/"))
                   for m in AUDIO_ALLOWED_MIMES)


# ── Extraction + transcode ─────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_no_attachments_returns_empty(attachments_tmp):
    assert await _read_inbound_audio_attachments({}) == []


@pytest.mark.asyncio
async def test_skips_non_audio_mime(attachments_tmp):
    (attachments_tmp / "img1").write_bytes(b"\xff\xd8\xff")
    data_message = {"attachments": [
        {"contentType": "image/jpeg", "id": "img1"},
    ]}
    assert await _read_inbound_audio_attachments(data_message) == []


@pytest.mark.asyncio
async def test_skips_missing_file(attachments_tmp):
    """signal-cli write race: the envelope arrives before the bytes."""
    assert await _read_inbound_audio_attachments(
        _voice_note_message("not-on-disk")
    ) == []


@pytest.mark.asyncio
async def test_size_cap(attachments_tmp, monkeypatch):
    monkeypatch.setattr("src.signal.handler.AUDIO_MAX_SOURCE_BYTES", 10)
    (attachments_tmp / "big").write_bytes(b"x" * 64)
    assert await _read_inbound_audio_attachments(
        _voice_note_message("big")
    ) == []


@pytest.mark.asyncio
async def test_undecodable_input_is_dropped_not_raised(attachments_tmp):
    """A file with an audio mime that isn't actually audio must degrade
    the turn to text-only, never fail the message."""
    (attachments_tmp / "junk").write_bytes(b"this is not audio at all")
    assert await _read_inbound_audio_attachments(
        _voice_note_message("junk")
    ) == []


@requires_ffmpeg
@pytest.mark.asyncio
async def test_transcodes_aac_voice_note_to_mp3(attachments_tmp):
    """The whole point: Signal sends AAC, which servers can't decode."""
    src = _write_test_audio(attachments_tmp / "vn1", seconds=1.0, codec="aac")
    src.rename(attachments_tmp / "vn1")

    clips = await _read_inbound_audio_attachments(_voice_note_message("vn1"))
    assert len(clips) == 1
    clip = clips[0]
    assert clip["mime"] == AUDIO_OUTPUT_MIME
    assert clip["format"] == AUDIO_OUTPUT_FORMAT
    assert clip["voice_note"] is True
    assert clip["duration_sec"] == pytest.approx(1.0, abs=0.3)
    # Decodes as real base64 and starts with an mp3 frame sync (or the
    # ID3 tag ffmpeg writes ahead of it).
    raw = base64.b64decode(clip["data_b64"])
    assert raw[:3] == b"ID3" or raw[0] == 0xFF


@requires_ffmpeg
@pytest.mark.asyncio
async def test_id_glob_fallback(attachments_tmp):
    """Some signal-cli configs keep the extension on the stored file."""
    _write_test_audio(attachments_tmp / "vn2", seconds=0.5, codec="aac")
    clips = await _read_inbound_audio_attachments(_voice_note_message("vn2"))
    assert len(clips) == 1


@requires_ffmpeg
@pytest.mark.asyncio
async def test_max_clips_cap(attachments_tmp):
    for i in range(4):
        src = _write_test_audio(
            attachments_tmp / f"c{i}", seconds=0.3, codec="aac",
        )
        src.rename(attachments_tmp / f"c{i}")
    data_message = {"attachments": [
        {"contentType": "audio/aac", "id": f"c{i}", "voiceNote": True}
        for i in range(4)
    ]}
    clips = await _read_inbound_audio_attachments(data_message, max_clips=2)
    assert len(clips) == 2


@requires_ffmpeg
@pytest.mark.asyncio
async def test_shared_audio_file_is_not_a_voice_note(attachments_tmp):
    """`voiceNote` is absent when the user attaches a music file — only
    the descriptor wording changes, the clip is still sent."""
    src = _write_test_audio(attachments_tmp / "song", seconds=0.5,
                            codec="libmp3lame")
    src.rename(attachments_tmp / "song")
    data_message = {"attachments": [
        {"contentType": "audio/mpeg", "id": "song", "filename": "song.mp3"},
    ]}
    clips = await _read_inbound_audio_attachments(data_message)
    assert len(clips) == 1
    assert clips[0]["voice_note"] is False
    assert describe_clips(clips).startswith("[audio,")


# ── Descriptors ────────────────────────────────────────────────────────────

def test_format_duration():
    assert format_duration(0) == "0:00"
    assert format_duration(23.4) == "0:23"
    assert format_duration(247) == "4:07"
    assert format_duration(None) == ""


def test_parse_duration_from_ffmpeg_banner():
    banner = (
        "Input #0, mov,mp4,m4a, from 'x.m4a':\n"
        "  Duration: 00:01:23.45, start: 0.000000, bitrate: 32 kb/s\n"
    )
    assert _parse_duration(banner) == pytest.approx(83.45)
    assert _parse_duration("no duration here") is None


def test_describe_clips():
    assert describe_clips([]) == ""
    assert describe_clips(
        [{"voice_note": True, "duration_sec": 23}]
    ) == "[voice note, 0:23]"
    assert describe_clips(
        [{"voice_note": True, "duration_sec": None}]
    ) == "[voice note]"
    assert describe_clips([
        {"voice_note": True, "duration_sec": 5},
        {"voice_note": False, "duration_sec": 61},
    ]) == "[voice note, 0:05] [audio, 1:01]"


# ── Payload parts ──────────────────────────────────────────────────────────

def _clip(b64: str = "QUJD") -> dict:
    return {
        "data_b64": b64, "mime": AUDIO_OUTPUT_MIME,
        "format": AUDIO_OUTPUT_FORMAT, "voice_note": True,
    }


def test_input_audio_part_shape():
    parts = _audio_parts([_clip()], style=AUDIO_PART_STYLE_INPUT_AUDIO)
    assert parts == [{
        "type": "input_audio",
        "input_audio": {"data": "QUJD", "format": "mp3"},
    }]


def test_audio_url_part_shape():
    parts = _audio_parts([_clip()], style=AUDIO_PART_STYLE_AUDIO_URL)
    assert parts == [{
        "type": "audio_url",
        "audio_url": {"url": "data:audio/mpeg;base64,QUJD"},
    }]


def test_unknown_style_falls_back_to_input_audio():
    parts = _audio_parts([_clip()], style="something-else")
    assert parts[0]["type"] == "input_audio"


def test_empty_and_malformed_clips_are_skipped():
    clips = [_clip(""), "not a dict", {"no_data": 1}, _clip()]
    parts = _audio_parts(clips, style=AUDIO_PART_STYLE_INPUT_AUDIO)
    assert len(parts) == 1


def test_image_parts_shape_unchanged():
    """The shared helper must keep emitting exactly what the vision path
    emitted before it was factored out."""
    parts = _image_parts([{"mime": "image/png", "data_b64": "QUJD"}])
    assert parts == [{
        "type": "image_url",
        "image_url": {"url": "data:image/png;base64,QUJD"},
    }]
    assert _image_parts([{"data_b64": ""}]) == []


# ── Transcript redaction ───────────────────────────────────────────────────

def test_transcript_redacts_input_audio_bytes():
    """`input_audio` hides its base64 under the generic key "data" — a
    transcript export must not inline a megabyte of it."""
    payload = {"messages": [{"role": "user", "content": [
        {"type": "input_audio",
         "input_audio": {"data": "A" * 5000, "format": "mp3"}},
    ]}]}
    snap = snapshot_payload(payload)
    stored = snap["messages"][0]["content"][0]["input_audio"]["data"]
    assert "inline binary omitted" in stored
    assert "AAAA" not in stored


def test_transcript_keeps_short_data_values():
    """"data" is a common key — a short legitimate value passes through."""
    snap = snapshot_payload({"arguments": {"data": "AAPL"}})
    assert snap["arguments"]["data"] == "AAPL"


def test_transcript_redacts_audio_data_uri():
    snap = snapshot_payload({"url": f"data:audio/mpeg;base64,{'A' * 5000}"})
    assert "inline binary omitted" in snap["url"]


# ── Caption-less voice note reaches dispatch ───────────────────────────────

@dataclass
class _FakeBotRegistry:
    bots: list = field(default_factory=list)

    def list_sync(self) -> list:
        return list(self.bots)

    def get_sync(self, bot_id: int):
        return next((b for b in self.bots if b.id == bot_id), None)

    def default_for_kind_sync(self, kind: str):
        flag = "default_for_dm" if kind == "dm" else "default_for_group"
        for b in self.bots:
            if b.enabled and getattr(b, flag):
                return b
        return self.bots[0] if self.bots else None


def _audio_bot(audio_enabled: bool = True) -> Bot:
    return Bot(
        id=1, slug="artaud", display_name="Artaud", aliases=["artaud"],
        default_for_dm=True, default_for_group=True,
        audio_enabled=audio_enabled,
    )


def _wire_handler(bots: list[Bot]):
    dispatcher = MagicMock()
    dispatcher.bot_registry = _FakeBotRegistry(bots)
    pool = SignalHandlerPool(
        default_api_url="http://signal-api:8080",
        default_phone="+15550000001",
        dispatcher=dispatcher,
        bot_registry=dispatcher.bot_registry,
    )
    pool.build()
    handler = pool.default()
    dispatch_mock = AsyncMock(return_value=None)
    dispatcher.dispatch = dispatch_mock
    dispatcher._resolve_bot = lambda g, policy=None, addressed_bot=None: bots[0]
    dispatcher.context_registry = None
    dispatcher.signal_pool = pool
    return dispatch_mock, handler


def _voice_note_envelope(att_id: str, *, message=None) -> dict:
    return {
        "envelope": {
            "source": "+15551112222",
            "sourceUuid": "uuid-1111-2222",
            "timestamp": 1234567890,
            "dataMessage": {
                "message": message,
                "timestamp": 1234567890,
                "attachments": [{
                    "contentType": "audio/aac",
                    "id": att_id,
                    "voiceNote": True,
                }],
            },
        }
    }


@requires_ffmpeg
@pytest.mark.asyncio
async def test_captionless_voice_note_reaches_dispatch(attachments_tmp):
    """Regression for the empty-text drop: a voice note sent with no
    caption has `message: null`, which used to be discarded outright as
    a blank message. It must dispatch with the clip attached and a
    duration-bearing descriptor standing in for the missing text."""
    src = _write_test_audio(attachments_tmp / "vn", seconds=1.0, codec="aac")
    src.rename(attachments_tmp / "vn")

    dispatch_mock, handler = _wire_handler([_audio_bot()])
    await handler.handle_webhook(_voice_note_envelope("vn"))

    assert dispatch_mock.call_count == 1
    kwargs = dispatch_mock.call_args.kwargs
    assert len(kwargs["inbound_audio"]) == 1
    assert kwargs["inbound_audio"][0]["format"] == AUDIO_OUTPUT_FORMAT
    assert kwargs["message"].startswith("[voice note,")


@requires_ffmpeg
@pytest.mark.asyncio
async def test_unaddressed_voice_note_extracts_when_default_bot_is_deaf(
    attachments_tmp,
):
    """The seam between the two gates. An unaddressed clip resolves to
    the per-context default bot — picked from chat settings, unrelated
    to who can hear. Gating extraction on that bot would leave
    `inbound_audio` empty whenever the default is the deaf one, silently
    disabling the reroute, the duration descriptor, and the reactor's
    prompt note in exactly the mixed-roster case they exist for.
    """
    src = _write_test_audio(attachments_tmp / "vn", seconds=1.0, codec="aac")
    src.rename(attachments_tmp / "vn")

    deaf = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"],
               default_for_dm=True, default_for_group=True)
    hears = Bot(id=2, slug="artaud", display_name="Artaud",
                aliases=["artaud"], audio_enabled=True)
    dispatch_mock, handler = _wire_handler([deaf, hears])

    await handler.handle_webhook(_voice_note_envelope("vn"))

    kwargs = dispatch_mock.call_args.kwargs
    assert len(kwargs["inbound_audio"]) == 1, (
        "clip must be extracted even though the resolved bot is deaf — "
        "the reroute downstream depends on inbound_audio being populated"
    )
    # Descriptor upgraded with the real duration, not the bare
    # pre-transcode placeholder.
    assert kwargs["message"].startswith("[voice note, ")


@requires_ffmpeg
@pytest.mark.asyncio
async def test_captioned_voice_note_keeps_caption_and_gains_descriptor(
    attachments_tmp,
):
    """With a caption, the descriptor trails the user's own words — the
    text-only reactor and history need to know audio came with it."""
    src = _write_test_audio(attachments_tmp / "vn", seconds=1.0, codec="aac")
    src.rename(attachments_tmp / "vn")

    dispatch_mock, handler = _wire_handler([_audio_bot()])
    await handler.handle_webhook(
        _voice_note_envelope("vn", message="listen to this")
    )

    kwargs = dispatch_mock.call_args.kwargs
    assert kwargs["message"].startswith("listen to this [voice note,")


@pytest.mark.asyncio
async def test_captionless_voice_note_dropped_when_nobody_can_listen(
    attachments_tmp,
):
    """The empty-text exception is scoped to installs that can actually
    use it. With no audio-enabled bot, a caption-less voice note keeps
    the old behavior — dropped — rather than surfacing a bare
    `[voice note]` placeholder the reactor might answer blindly."""
    (attachments_tmp / "vn").write_bytes(b"unused")

    dispatch_mock, handler = _wire_handler([_audio_bot(audio_enabled=False)])
    await handler.handle_webhook(_voice_note_envelope("vn"))

    assert dispatch_mock.call_count == 0


@requires_ffmpeg
@pytest.mark.asyncio
async def test_captioned_voice_note_dispatches_without_audio_bot(
    attachments_tmp,
):
    """A voice note WITH a caption was always a valid message and still
    is — it just arrives without the clip attached."""
    src = _write_test_audio(attachments_tmp / "vn", seconds=0.5, codec="aac")
    src.rename(attachments_tmp / "vn")

    dispatch_mock, handler = _wire_handler([_audio_bot(audio_enabled=False)])
    await handler.handle_webhook(
        _voice_note_envelope("vn", message="thoughts?")
    )

    assert dispatch_mock.call_count == 1
    kwargs = dispatch_mock.call_args.kwargs
    assert kwargs["inbound_audio"] == []
    assert kwargs["message"] == "thoughts?"


def test_source_cap_is_sane():
    assert 0 < AUDIO_MAX_SOURCE_BYTES <= 100 * 1024 * 1024


# ── Reactor routing for unaddressed voice notes ────────────────────────────

def _reactor_dispatcher(bots):
    """A real CommandDispatcher with a stub reactor, no commands, and no
    stores — enough to reach the reactor hand-off inside dispatch()."""
    from src.commands.dispatcher import CommandDispatcher

    reactor = MagicMock()
    reactor.maybe_react = AsyncMock(return_value=None)
    d = CommandDispatcher(
        enable_inline_symbols=False,
        reactor=reactor,
        bot_registry=_FakeBotRegistry(bots),
    )
    return d


@pytest.mark.asyncio
async def test_voice_note_routes_to_a_bot_that_can_hear():
    """The per-context default is resolved from chat settings, not from
    capability. In a mixed group an unaddressed voice note must not be
    handed to the deaf bot — it would answer the bare descriptor."""
    deaf = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"],
               default_for_group=True)
    hears = Bot(id=2, slug="artaud", display_name="Artaud",
                aliases=["artaud"], audio_enabled=True)
    d = _reactor_dispatcher([deaf, hears])

    clips = [_clip()]
    await _run_reactor_handoff(d, inbound_audio=clips)

    kwargs = d.reactor.maybe_react.call_args.kwargs
    assert kwargs["bot"].slug == "artaud"
    assert [b.slug for b in kwargs["candidate_bots"]] == ["artaud"]
    assert kwargs["inbound_audio"] == clips


@pytest.mark.asyncio
async def test_text_message_routing_is_unchanged():
    deaf = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"],
               default_for_group=True)
    hears = Bot(id=2, slug="artaud", display_name="Artaud",
                aliases=["artaud"], audio_enabled=True)
    d = _reactor_dispatcher([deaf, hears])

    await _run_reactor_handoff(d, inbound_audio=[])

    kwargs = d.reactor.maybe_react.call_args.kwargs
    assert kwargs["bot"].slug == "sigil"
    assert [b.slug for b in kwargs["candidate_bots"]] == ["sigil", "artaud"]


@pytest.mark.asyncio
async def test_voice_note_routing_noop_when_nobody_hears():
    deaf = Bot(id=1, slug="sigil", display_name="Sigil", aliases=["sigil"],
               default_for_group=True)
    d = _reactor_dispatcher([deaf])

    await _run_reactor_handoff(d, inbound_audio=[_clip()])

    kwargs = d.reactor.maybe_react.call_args.kwargs
    assert kwargs["bot"].slug == "sigil"


async def _run_reactor_handoff(dispatcher, *, inbound_audio):
    """Drive dispatch() far enough to reach the reactor hand-off.

    The message is a plain descriptor with no command prefix and no
    ticker, so it falls through command parsing and the NLP path without
    needing any of those subsystems wired.
    """
    import asyncio

    await dispatcher.dispatch(
        sender="+15551112222",
        message="[voice note, 0:05]",
        group_id="grp1",
        target_timestamp=1234567890,
        inbound_audio=inbound_audio,
    )
    # The reactor runs as a fire-and-forget task; let it start.
    await asyncio.sleep(0)
