"""Inbound-audio normalization for multimodal bots.

Signal records voice notes as AAC (in an MP4/m4a container). Essentially
no inference server decodes that: vLLM/llama.cpp route audio through
librosa → libsndfile, which handles wav/flac/ogg/mp3 and nothing else,
and OpenAI's own `input_audio` part accepts only wav and mp3. So every
clip is transcoded to mono 16 kHz mp3 before it goes anywhere near a
model.

mp3 rather than wav: a 30-second voice note is ~1.3 MB of base64 as wav
versus ~160 KB as mp3, and every byte is paid for on every writer round
that replays the turn. 16 kHz mono is what speech-capable models
downsample to internally anyway, so the fidelity loss is nil for voice.

ffmpeg is invoked on the file path, never on stdin — MP4 demuxing needs
a seekable input, and a caption-less voice note piped to `pipe:0` fails
with "moov atom not found" whenever the container wasn't written
faststart. signal-cli has already put the bytes on disk, so we just
point at them.
"""

from __future__ import annotations

import asyncio
import base64
import logging
import re
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Audio-payload guardrails, deliberately tighter than the vision caps.
# Base64 audio is billed per second of speech and a voice note has no
# natural upper bound the way a photo does — someone can hold the mic
# button for ten minutes. 2 clips covers "voice note plus a follow-up
# voice note in the same send"; 5 minutes is well past the length at
# which a caller would reasonably expect a chat bot to answer.
AUDIO_MAX_CLIPS = 2
AUDIO_MAX_DURATION_SEC = 300
# Cap on the SOURCE file. Compressed speech runs ~8 KB/s, so 25 MB is a
# generous ceiling that still refuses someone attaching a wav album.
AUDIO_MAX_SOURCE_BYTES = 25 * 1024 * 1024

AUDIO_ALLOWED_MIMES = frozenset({
    # Signal voice notes (Android and iOS both land here)
    "audio/aac", "audio/mp4", "audio/m4a", "audio/x-m4a",
    # Shared audio files
    "audio/mpeg", "audio/mp3", "audio/ogg", "audio/opus",
    "audio/webm", "audio/wav", "audio/x-wav", "audio/vnd.wave",
    "audio/flac", "audio/x-flac",
})

# Transcode output. Mirrors what a speech encoder wants at the front of
# its pipeline; anything richer is discarded by the model anyway.
AUDIO_OUTPUT_FORMAT = "mp3"
AUDIO_OUTPUT_MIME = "audio/mpeg"
_SAMPLE_RATE = "16000"
_BITRATE = "32k"

# A 5-minute clip transcodes in well under a second on any machine that
# can host this bot; 45s is a hang guard, not a budget.
_TRANSCODE_TIMEOUT_SEC = 45

_DURATION_RE = re.compile(r"Duration:\s*(\d+):(\d{2}):(\d{2}(?:\.\d+)?)")


@dataclass
class AudioClip:
    """A transcoded inbound clip, ready for an LLM multimodal part."""
    data_b64: str
    mime: str = AUDIO_OUTPUT_MIME
    fmt: str = AUDIO_OUTPUT_FORMAT
    filename: str = ""
    duration_sec: Optional[float] = None
    voice_note: bool = False

    def as_dict(self) -> dict:
        return {
            "mime": self.mime,
            "format": self.fmt,
            "data_b64": self.data_b64,
            "filename": self.filename,
            "duration_sec": self.duration_sec,
            "voice_note": self.voice_note,
        }


def ffmpeg_available() -> bool:
    """Whether an ffmpeg binary is on PATH.

    Not cached: the container's PATH is fixed, but a False result is the
    difference between "audio silently never works" and a log line an
    admin can act on, and `shutil.which` is a handful of stat calls.
    """
    return shutil.which("ffmpeg") is not None


def _parse_duration(stderr: str) -> Optional[float]:
    """Pull the source duration out of ffmpeg's banner.

    Free — ffmpeg prints `Duration: 00:00:23.45` for every input — where
    a separate ffprobe call would cost a second process spawn per clip.
    """
    m = _DURATION_RE.search(stderr or "")
    if not m:
        return None
    try:
        hours, minutes, seconds = m.groups()
        return int(hours) * 3600 + int(minutes) * 60 + float(seconds)
    except (TypeError, ValueError):
        return None


def format_duration(seconds: Optional[float]) -> str:
    """`0:23` / `4:07`. Empty string when the duration is unknown."""
    if seconds is None or seconds < 0:
        return ""
    total = int(round(seconds))
    return f"{total // 60}:{total % 60:02d}"


async def transcode_to_mp3(
    path: Path,
    *,
    max_duration_sec: int = AUDIO_MAX_DURATION_SEC,
) -> Optional[AudioClip]:
    """Transcode any inbound audio file to mono 16 kHz mp3.

    Returns None (with a log line) on every failure mode — missing
    ffmpeg, unreadable/undecodable input, over-length clip, timeout. A
    dropped clip degrades the turn to text-only, which is always
    preferable to failing the user's message.

    `-t` bounds the OUTPUT rather than rejecting a long input outright:
    a rambling five-minute voice note still gets answered, just on its
    first five minutes. The duration reported back is the source's, so
    the descriptor shown to the chat stays honest about what was sent.
    """
    if not ffmpeg_available():
        logger.warning(
            "Inbound audio: ffmpeg not on PATH — clip dropped. Install "
            "ffmpeg in the image to enable audio input."
        )
        return None

    cmd = [
        "ffmpeg", "-hide_banner", "-nostdin",
        "-i", str(path),
        "-vn",                      # drop cover art / video streams
        "-t", str(max_duration_sec),
        "-ac", "1",
        "-ar", _SAMPLE_RATE,
        "-b:a", _BITRATE,
        "-f", AUDIO_OUTPUT_FORMAT,
        "pipe:1",
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
    except OSError as e:
        logger.warning(f"Inbound audio: ffmpeg spawn failed: {e}")
        return None

    try:
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=_TRANSCODE_TIMEOUT_SEC,
        )
    except asyncio.TimeoutError:
        logger.warning(
            f"Inbound audio: ffmpeg timed out after "
            f"{_TRANSCODE_TIMEOUT_SEC}s on {path.name}; killing"
        )
        try:
            proc.kill()
            await proc.wait()
        except (ProcessLookupError, OSError):
            pass
        return None

    err_text = (stderr or b"").decode("utf-8", errors="replace")
    if proc.returncode != 0 or not stdout:
        tail = " ".join(err_text.split())[-300:]
        logger.warning(
            f"Inbound audio: ffmpeg failed on {path.name} "
            f"(rc={proc.returncode}, out={len(stdout or b'')}b): {tail}"
        )
        return None

    duration = _parse_duration(err_text)
    if duration is not None and duration > max_duration_sec:
        logger.info(
            f"Inbound audio: {path.name} is {format_duration(duration)}; "
            f"sending only the first {max_duration_sec}s"
        )

    return AudioClip(
        data_b64=base64.b64encode(stdout).decode("ascii"),
        filename=path.name,
        duration_sec=duration,
    )


def describe_clips(clips: list[dict]) -> str:
    """A short text marker naming the audio that rides along with a turn.

    This is the ONLY record of the clip that survives past the current
    round: the bytes are never persisted to conversation history (a
    handful of replayed voice notes would dwarf every other token in
    the prompt), so the descriptor is what a later turn — and the
    text-only reactor, tool-bot, and sibling models — sees in its place.

    Also stands in as the message text for a caption-less voice note,
    which would otherwise be an empty string that every downstream
    guard drops on the floor.
    """
    if not clips:
        return ""
    parts: list[str] = []
    for clip in clips:
        if not isinstance(clip, dict):
            continue
        label = "voice note" if clip.get("voice_note") else "audio"
        stamp = format_duration(clip.get("duration_sec"))
        parts.append(f"[{label}, {stamp}]" if stamp else f"[{label}]")
    return " ".join(parts)
