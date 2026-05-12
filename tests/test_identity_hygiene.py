"""
Tests for identity-leak hygiene: tool-content scrubbing and chat-history
bot-label routing.

Background: prior bot turns in the group_log are stored under a single
BOT_SENDER sentinel, so the replay path can't tell which bot wrote a row.
The replay must label them with the CURRENT bot's display_name so a
multi-bot setup (Sigil + Artaud) doesn't have Artaud's response history
show up labeled "Sigil" and corrupt its in-context identity.

Separately, the fetch MCP echoes our outbound User-Agent verbatim in
robots.txt error payloads. Without scrubbing, the bot's persona name
leaks back into the writer's context window on every blocked fetch and
the writer flips identity via in-context learning.
"""

from src.commands.ask_command import _scrub_tool_content


# ── _scrub_tool_content ────────────────────────────────────────────────────

def test_scrubs_useragent_xml_tag():
    text = (
        "ERROR: The site's robots.txt blocks autonomous fetching, "
        "<useragent>Sigil signal-stock-bot (https://github.com/x/y)</useragent>"
    )
    out = _scrub_tool_content(text)
    assert "Sigil" not in out
    assert "github.com" not in out
    assert "(useragent redacted)" in out


def test_scrubs_user_agent_header_line():
    text = (
        "Request failed: 403\n"
        "User-Agent: ArtaudBot/1.0 (curl-compat)\n"
        "Reason: blocked"
    )
    out = _scrub_tool_content(text)
    assert "ArtaudBot" not in out
    assert "User-Agent: (redacted)" in out
    assert "Request failed: 403" in out
    assert "Reason: blocked" in out


def test_case_insensitive_useragent_tag():
    text = "<UserAgent>SecretName/1.0</UserAgent>"
    out = _scrub_tool_content(text)
    assert "SecretName" not in out


def test_passthrough_when_no_match():
    """Legitimate tool output unchanged."""
    text = "AAPL closed at 185.92, +1.27%"
    assert _scrub_tool_content(text) == text


def test_handles_empty_and_non_string():
    assert _scrub_tool_content("") == ""
    # Non-string inputs (None, dicts) pass through untouched.
    assert _scrub_tool_content(None) is None  # type: ignore[arg-type]
    assert _scrub_tool_content(123) == 123  # type: ignore[arg-type]


def test_preserves_surrounding_text():
    text = (
        "Fetched ok.\n"
        "<useragent>SomeBot</useragent>\n"
        "Body: hello world"
    )
    out = _scrub_tool_content(text)
    assert "Fetched ok." in out
    assert "Body: hello world" in out
    assert "SomeBot" not in out


# ── _active_bot_display ────────────────────────────────────────────────────

def test_active_bot_display_prefers_ctx_bot():
    """The label used for BOT_SENDER replay rows should be the current
    bot's display_name — not the seed-startup name_registry.bot_name."""
    from src.commands.ask_command import AskCommand

    class FakeNameRegistry:
        bot_name = "Sigil"

    class FakeBot:
        display_name = "Artaud"

    class FakeCtx:
        bot = FakeBot()

    # Construct the cheapest AskCommand instance that has name_registry.
    # We can't instantiate AskCommand fully (lots of deps); test the
    # method via a thin object that provides the same attributes.
    class Probe:
        name_registry = FakeNameRegistry()
        _active_bot_display = AskCommand._active_bot_display

    assert Probe()._active_bot_display(FakeCtx()) == "Artaud"


def test_active_bot_display_falls_back_to_registry():
    """No ctx.bot → use the registry's seed name."""
    from src.commands.ask_command import AskCommand

    class FakeNameRegistry:
        bot_name = "Sigil"

    class FakeCtx:
        bot = None

    class Probe:
        name_registry = FakeNameRegistry()
        _active_bot_display = AskCommand._active_bot_display

    assert Probe()._active_bot_display(FakeCtx()) == "Sigil"


def test_active_bot_display_handles_missing_everything():
    """Defensive: even with no registry and no ctx.bot, return *something*."""
    from src.commands.ask_command import AskCommand

    class Probe:
        name_registry = None
        _active_bot_display = AskCommand._active_bot_display

    assert Probe()._active_bot_display(None) == "Bot"
