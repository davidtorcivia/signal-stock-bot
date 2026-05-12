"""
Round-trip tests for the new ContextPolicy fields:
  * transcript_logging_enabled
  * history_turns_override
"""

import tempfile
from pathlib import Path

import pytest

from src.contexts import ContextPolicy, ContextRegistry


@pytest.fixture
async def registry():
    with tempfile.TemporaryDirectory() as d:
        reg = ContextRegistry(db_path=str(Path(d) / "ctx.db"))
        yield reg


@pytest.mark.asyncio
async def test_defaults_for_new_policy(registry):
    """Newly created policies default to logging-off + override-none."""
    p = ContextPolicy(id=None, kind="group", key="grp-test", label="Test")
    assert p.transcript_logging_enabled is False
    assert p.history_turns_override is None


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
