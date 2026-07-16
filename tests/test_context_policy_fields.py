"""
Round-trip tests for the new ContextPolicy fields:
  * transcript_logging_enabled
  * history_turns_override
"""

import tempfile
from pathlib import Path

import pytest

from src.contexts import ContextPolicy, ContextRegistry
from src.contexts.policy import MODE_ALLOW_ALL, MODE_ALLOW_LIST, PERMISSIVE


@pytest.fixture
async def registry():
    with tempfile.TemporaryDirectory() as d:
        reg = ContextRegistry(db_path=str(Path(d) / "ctx.db"))
        yield reg


@pytest.mark.asyncio
async def test_defaults_for_new_policy(registry):
    """New policies carry no MCP schemas until an admin opts servers in."""
    p = ContextPolicy(id=None, kind="group", key="grp-test", label="Test")
    assert p.transcript_logging_enabled is False
    assert p.history_turns_override is None
    assert p.mcp_mode == MODE_ALLOW_LIST
    assert p.mcp_servers == []
    assert p.allows_mcp("large-server") is False


def test_registry_failure_fallback_remains_permissive():
    """The emergency fallback keeps legacy availability semantics even
    though ordinary contexts now default to an empty MCP allow-list."""
    assert PERMISSIVE.mcp_mode == MODE_ALLOW_ALL
    assert PERMISSIVE.allows_mcp("any-running-server") is True


@pytest.mark.asyncio
async def test_seeded_defaults_use_empty_mcp_allow_list(registry):
    await registry._ensure_initialized()
    policies = {p.key: p for p in await registry.list()}
    for key in ("default:group", "default:dm"):
        assert policies[key].mcp_mode == MODE_ALLOW_LIST
        assert policies[key].mcp_servers == []


@pytest.mark.asyncio
async def test_persist_and_reload_new_fields(registry):
    p = ContextPolicy(
        id=None,
        kind="group",
        key="grp-persist",
        label="Persist test",
        transcript_logging_enabled=True,
        history_turns_override=12,
    )
    new_id = await registry.upsert(p)
    reloaded = await registry.get(new_id)
    assert reloaded is not None
    assert reloaded.transcript_logging_enabled is True
    assert reloaded.history_turns_override == 12


@pytest.mark.asyncio
async def test_zero_is_distinct_from_none(registry):
    """Override=0 (explicit no-history) must survive a write/read cycle as 0,
    not flatten to None."""
    p = ContextPolicy(
        id=None,
        kind="dm",
        key="+15551234567",
        label="One-shot DM",
        history_turns_override=0,
    )
    new_id = await registry.upsert(p)
    reloaded = await registry.get(new_id)
    assert reloaded is not None
    assert reloaded.history_turns_override == 0


@pytest.mark.asyncio
async def test_update_preserves_fields(registry):
    p = ContextPolicy(
        id=None, kind="group", key="grp-upd", label="Upd",
        transcript_logging_enabled=True, history_turns_override=8,
    )
    pid = await registry.upsert(p)
    p2 = await registry.get(pid)
    assert p2 is not None
    # Edit label only, leave logging on.
    p2.label = "Renamed"
    await registry.upsert(p2)
    p3 = await registry.get(pid)
    assert p3 is not None
    assert p3.label == "Renamed"
    assert p3.transcript_logging_enabled is True
    assert p3.history_turns_override == 8
