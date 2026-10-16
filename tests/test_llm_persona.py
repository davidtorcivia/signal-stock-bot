"""
Tests for the writer system-prompt resolution chain:

  1. bot_llm_settings(bot_id, role, 'system_prompt')  — explicit per-bot
  2. bots.persona  via persona_provider callable        — bot identity
  3. admin_settings.llm_system_prompt                  — global fallback
  4. DEFAULT_SYSTEM_PROMPT                             — built-in

Regression: a non-default bot whose identity lives only in `bots.persona`
must NOT inherit the global llm_system_prompt (which leaks the default
bot's identity across other bots).
"""

from src.llm.client import LLMClient, DEFAULT_SYSTEM_PROMPT


class FakeStore:
    """Minimal SettingsStore stand-in for prompt-resolution tests."""

    def __init__(self, global_settings=None, bot_settings=None):
        self._global = dict(global_settings or {})
        # bot_settings: {(bot_id, role, key): value}
        self._bot = dict(bot_settings or {})

    def get(self, key, default=None):
        return self._global.get(key, default)

    def get_bot(self, bot_id, role, key, default=None):
        return self._bot.get((bot_id, role, key), default)

    # Stubbed accessors LLMClient._config uses for non-prompt keys.
    def get_int(self, key, default=0, min_value=None):
        v = self._global.get(key, default)
        try:
            v = int(v) if v not in (None, "") else default
        except (TypeError, ValueError):
            v = default
        if min_value is not None and v < min_value:
            v = min_value
        return v


def _client(store, *, bot_id=None, persona_provider=None):
    return LLMClient(store, bot_id=bot_id, persona_provider=persona_provider)


def test_explicit_bot_override_wins():
    store = FakeStore(
        global_settings={"llm_system_prompt": "GLOBAL"},
        bot_settings={(2, "writer", "system_prompt"): "EXPLICIT"},
    )
    persona_provider = lambda bid: "PERSONA"  # noqa: E731
    cfg = _client(store, bot_id=2, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == "EXPLICIT"


def test_persona_used_when_no_explicit_override():
    """Bug fix: bot with a persona but no bot_llm_settings.system_prompt
    must use its persona, not fall through to the global."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL_SIGIL"})
    persona_provider = lambda bid: "I am Artaud."  # noqa: E731
    cfg = _client(store, bot_id=2, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == "I am Artaud."


def test_global_used_when_no_persona():
    """Sigil's path: empty persona → global prompt applies."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL_SIGIL"})
    persona_provider = lambda bid: ""  # noqa: E731
    cfg = _client(store, bot_id=1, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == "GLOBAL_SIGIL"


def test_persona_none_falls_through():
    """When the provider returns None (registry miss), global wins."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL"})
    persona_provider = lambda bid: None  # noqa: E731
    cfg = _client(store, bot_id=2, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == "GLOBAL"


def test_default_used_when_global_unset_and_no_persona():
    store = FakeStore()  # no global, no per-bot
    persona_provider = lambda bid: None  # noqa: E731
    cfg = _client(store, bot_id=2, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == DEFAULT_SYSTEM_PROMPT


def test_no_bot_id_no_persona_lookup():
    """Legacy single-bot callers (bot_id=None) skip persona entirely."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL"})
    persona_provider = lambda bid: "SHOULD_NOT_BE_USED"  # noqa: E731
    cfg = _client(store, bot_id=None, persona_provider=persona_provider)._config()
    assert cfg["system_prompt"] == "GLOBAL"


def test_persona_provider_exception_swallowed():
    """A broken registry must not crash the LLM path — fall through silently."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL"})

    def boom(_bid):
        raise RuntimeError("registry exploded")

    cfg = _client(store, bot_id=2, persona_provider=boom)._config()
    assert cfg["system_prompt"] == "GLOBAL"


def test_persona_provider_optional_when_none():
    """No persona_provider wired → behavior matches pre-fix (global wins)."""
    store = FakeStore(global_settings={"llm_system_prompt": "GLOBAL"})
    cfg = _client(store, bot_id=2, persona_provider=None)._config()
    assert cfg["system_prompt"] == "GLOBAL"


def test_context_prompt_only_applies_to_pinned_bot():
    """Woo's on First's "You are Sigil…" prompt made a summoned Artaud
    introduce itself as Sigil."""
    from types import SimpleNamespace as NS
    from src.commands.ask_command import _context_prompt_for

    policy = NS(system_prompt="You are Sigil.", default_bot_id=1)
    assert _context_prompt_for(NS(policy=policy, bot=NS(id=1))) == "You are Sigil."
    assert _context_prompt_for(NS(policy=policy, bot=NS(id=2))) is None
    unpinned = NS(system_prompt="Be terse.", default_bot_id=None)
    assert _context_prompt_for(NS(policy=unpinned, bot=NS(id=2))) == "Be terse."
    assert _context_prompt_for(NS(policy=NS(system_prompt=None, default_bot_id=1), bot=NS(id=1))) is None


def test_per_bot_context_prompt_wins():
    from types import SimpleNamespace as NS
    from src.commands.ask_command import _context_prompt_for

    policy = NS(system_prompt="You are Sigil.", default_bot_id=1,
                bot_prompts={2: "You are Artaud, here."})
    assert _context_prompt_for(NS(policy=policy, bot=NS(id=2))) == "You are Artaud, here."
    assert _context_prompt_for(NS(policy=policy, bot=NS(id=1))) == "You are Sigil."


def test_bot_prompts_survive_registry_round_trip(tmp_path):
    import asyncio
    from src.contexts.registry import ContextRegistry
    from src.contexts.policy import ContextPolicy

    async def go():
        reg = ContextRegistry(str(tmp_path / "c.db"))
        pid = await reg.upsert(ContextPolicy(id=None, kind="group", key="g1",
                                             bot_prompts={2: "Artaud here"}))
        got = await reg.get(pid)
        assert got.bot_prompts == {2: "Artaud here"}
    asyncio.run(go())


def test_dm_small_talk_reaches_llm_without_name():
    """'She works for the CIA' in a DM was dropped: only messages that
    looked like stock queries (or named the bot) reached the LLM."""
    import asyncio
    from src.commands.dispatcher import CommandDispatcher
    from src.commands.base import CommandResult
    from src.contexts.policy import ContextPolicy

    seen = []

    class _Ask:
        async def execute(self, ctx):
            seen.append(ctx.args)
            return CommandResult.ok("ok")

    d = CommandDispatcher(ask_command=_Ask())
    dm = ContextPolicy(id=None, kind="default", key="default:dm", llm_intent=True)
    asyncio.run(d.dispatch("+1555", "She works for the CIA", policy=dm))
    assert seen == [["She", "works", "for", "the", "CIA"]]
    # Groups still need the bot addressed.
    asyncio.run(d.dispatch("+1555", "She works for the CIA", group_id="g", policy=dm))
    assert len(seen) == 1
