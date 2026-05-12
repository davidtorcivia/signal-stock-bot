"""Image-history replay: persist inbound images across turns and re-inflate
the last N user turns' images into multimodal payload so follow-ups can
refer back to the picture.

Covers:
  * conversation_turns gains an image_refs column (migration is idempotent)
  * ConversationHistory.append serializes refs to JSON
  * ConversationHistory.load returns refs alongside content
  * AskCommand._inflate_image_history keeps refs on the last N user
    turns and drops them on older turns
  * Bot.vision_enabled gates replay — text-only bots never see images
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.bots.models import Bot
from src.commands.ask_command import AskCommand
from src.llm.history import ConversationHistory


@pytest.fixture
async def history():
    with tempfile.TemporaryDirectory() as d:
        h = ConversationHistory(db_path=str(Path(d) / "h.db"))
        yield h


def _image_ref(b64: str = "AAAA", mime: str = "image/png") -> dict:
    return {"mime": mime, "data_b64": b64, "filename": "test.png"}


# ── ConversationHistory ────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_append_and_load_roundtrip_image_refs(history):
    refs = [_image_ref("aGVsbG8="), _image_ref("d29ybGQ=", mime="image/jpeg")]
    await history.append("ctx", "user", "what's this?", image_refs=refs)
    await history.append("ctx", "assistant", "looks like a chart")
    turns = await history.load("ctx")
    assert turns[0]["role"] == "user"
    assert turns[0]["image_refs"] == refs
    # Assistant rows don't carry image_refs.
    assert "image_refs" not in turns[1]


@pytest.mark.asyncio
async def test_load_omits_image_refs_when_column_null(history):
    await history.append("ctx", "user", "text only")
    turns = await history.load("ctx")
    assert "image_refs" not in turns[0]


@pytest.mark.asyncio
async def test_image_refs_survives_load_with_attribution(history):
    """Bracket-prefix attribution shouldn't disturb the image_refs field."""
    refs = [_image_ref()]
    await history.append(
        "ctx", "user", "look", sender_tail="1234", image_refs=refs,
    )
    turns = await history.load("ctx", attribute_senders=True, now=1_700_000_000.0)
    # Attribution prepended a [..tail, just now] bracket onto content
    assert turns[0]["content"].startswith("[")
    # Image refs still attached
    assert turns[0]["image_refs"] == refs


# ── AskCommand._inflate_image_history ──────────────────────────────────────


def _user(content: str, image_refs=None) -> dict:
    t: dict = {"role": "user", "content": content}
    if image_refs is not None:
        t["image_refs"] = image_refs
    return t


def _assistant(content: str) -> dict:
    return {"role": "assistant", "content": content}


def _bot(vision_enabled: bool) -> Bot:
    return Bot(
        id=1,
        slug="seer",
        display_name="Seer",
        vision_enabled=vision_enabled,
    )


class _Ctx:
    def __init__(self, bot):
        self.bot = bot


def _inflate(prior, ctx):
    """Invoke the inflate helper without constructing a full AskCommand."""
    AskCommand._inflate_image_history(AskCommand.__new__(AskCommand), prior, ctx)
    return prior


class TestInflateImageHistory:
    def test_inflates_recent_user_turn_into_multimodal(self):
        prior = [_user("what's this?", image_refs=[_image_ref("aGk=")])]
        out = _inflate(prior, _Ctx(_bot(vision_enabled=True)))
        content = out[0]["content"]
        assert isinstance(content, list)
        assert content[0] == {"type": "text", "text": "what's this?"}
        assert content[1]["type"] == "image_url"
        assert "data:image/png;base64,aGk=" in content[1]["image_url"]["url"]
        # image_refs key removed — content is now OpenAI-shaped.
        assert "image_refs" not in out[0]

    def test_drops_images_when_bot_has_vision_disabled(self):
        prior = [_user("what's this?", image_refs=[_image_ref()])]
        out = _inflate(prior, _Ctx(_bot(vision_enabled=False)))
        # Content stays a string; image_refs scrubbed.
        assert out[0]["content"] == "what's this?"
        assert "image_refs" not in out[0]

    def test_keeps_images_on_last_N_user_turns_only(self):
        # Build 7 user/assistant rounds, all with images. Only the last
        # 5 user turns should retain refs.
        prior = []
        for i in range(7):
            prior.append(_user(f"q{i}", image_refs=[_image_ref(f"img{i}")]))
            prior.append(_assistant(f"a{i}"))

        out = _inflate(prior, _Ctx(_bot(vision_enabled=True)))

        # Older user turns (q0, q1) → no image, plain text.
        for old_idx in (0, 2):  # positions of q0, q1
            assert isinstance(out[old_idx]["content"], str)
            assert "image_refs" not in out[old_idx]

        # Last 5 user turns (q2..q6) → multimodal content.
        for fresh_idx in (4, 6, 8, 10, 12):  # positions of q2..q6
            content = out[fresh_idx]["content"]
            assert isinstance(content, list), f"expected list at {fresh_idx}"
            assert content[0]["type"] == "text"
            assert content[1]["type"] == "image_url"

    def test_user_turn_without_refs_passes_through(self):
        prior = [_user("hi"), _assistant("hello")]
        out = _inflate(prior, _Ctx(_bot(vision_enabled=True)))
        assert out[0]["content"] == "hi"
        assert isinstance(out[0]["content"], str)
        assert out[1]["content"] == "hello"

    def test_malformed_image_ref_entry_skipped(self):
        # data_b64 missing → that image is dropped, but the turn still
        # works (becomes pure text).
        prior = [_user("look", image_refs=[{"mime": "image/png"}])]
        out = _inflate(prior, _Ctx(_bot(vision_enabled=True)))
        # Only the text part survives — len(parts) == 1, so we keep
        # content as the original string rather than wrapping it.
        assert out[0]["content"] == "look"

    def test_no_bot_on_ctx_treated_as_no_vision(self):
        prior = [_user("hi", image_refs=[_image_ref()])]
        out = _inflate(prior, _Ctx(bot=None))
        assert out[0]["content"] == "hi"
        assert "image_refs" not in out[0]

    def test_handles_empty_prior(self):
        out = _inflate([], _Ctx(_bot(vision_enabled=True)))
        assert out == []
