"""Memory tools reach the deep_think / tool_bot loops, not just the writer.

Covers the gating (`memory_tool_schemas`), the shared dispatcher
(`dispatch_memory_tool`) round-tripping remember → recall, and both
tool-loop clients exposing the schemas once a store + policy are set.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pytest

from src.commands.base import CommandContext
from src.contexts.policy import ContextPolicy
from src.llm.deep_think import DeepThinkClient
from src.llm.tool_bot import ToolBotClient
from src.memory import (
    MemoryStore,
    SubjectResolver,
    dispatch_memory_tool,
    memory_tool_schemas,
)


def _names(schemas) -> set:
    return {(s.get("function") or {}).get("name") for s in (schemas or [])}


@dataclass
class _FakeRegistry:
    _cache: dict

    def label_for(self, phone: Optional[str]) -> str:
        return "David"


def _policy(**kw) -> ContextPolicy:
    return ContextPolicy(key="group:g1", id=7, kind="group", **kw)


def test_schemas_gate_on_policy_and_writes():
    store = MemoryStore(db_path="unused.db")
    # Real context row, writes on → all three.
    assert _names(memory_tool_schemas(store, _policy())) == {
        "recall", "remember", "forget",
    }
    # Writes off → recall only.
    assert _names(
        memory_tool_schemas(store, _policy(memory_writes_enabled=False))
    ) == {"recall"}
    # Default row / no store / no policy → nothing.
    assert memory_tool_schemas(store, ContextPolicy(key="default:group", id=None, kind="default")) == []
    assert memory_tool_schemas(None, _policy()) == []
    assert memory_tool_schemas(store, None) == []


async def test_dispatch_remember_then_recall(tmp_path: Path):
    store = MemoryStore(db_path=str(tmp_path / "mem.db"))
    resolver = SubjectResolver(_FakeRegistry(_cache={}))
    ctx = CommandContext(
        sender="+15550001111", group_id="g1", raw_message="", command="ask",
        args=[], policy=_policy(),
    )

    saved = await dispatch_memory_tool(
        store=store, resolver=resolver, name="remember",
        args={
            "subject": "speaker",
            "kind": "fact",
            "content": "runs a hedge fund out of Brooklyn",
        },
        caller_ctx=ctx, bot_id=3,
    )
    assert saved.startswith("saved memory #")

    found = await dispatch_memory_tool(
        store=store, resolver=resolver, name="recall",
        args={"subject": "speaker"}, caller_ctx=ctx, bot_id=3,
    )
    assert "hedge fund out of Brooklyn" in found


@pytest.mark.parametrize("cls", [DeepThinkClient, ToolBotClient])
def test_tool_loop_clients_expose_memory_tools(cls):
    client = cls(settings_store=object())
    assert "remember" not in _names(client._collect_tools(policy=_policy()))
    client.memory_store = MemoryStore(db_path="unused.db")
    client.subject_resolver = SubjectResolver(_FakeRegistry(_cache={}))
    assert _names(client._collect_tools(policy=_policy())) >= {
        "recall", "remember", "forget",
    }


async def test_duplicate_registration_collapses_to_oldest_hash(tmp_path: Path):
    """One person, two Signal identities (phone-era + UUID-era hash).

    Both must resolve to the first-registered hash, and the preamble must
    surface memories stored under it when the newer identity speaks.
    """
    from src.database import hash_phone
    from src.memory import build_preamble
    from src.users import NameRegistry

    db = str(tmp_path / "names.db")
    registry = NameRegistry(db_path=db)
    old_phone, new_phone = "+15550001111", "uuid-abc-123"
    old_hash, new_hash = hash_phone(old_phone), hash_phone(new_phone)
    await registry.set_name("David", phone=old_phone)
    await registry.set_name("David", phone=new_phone)

    resolver = SubjectResolver(registry)
    assert resolver.resolve("speaker", sender_phone=new_phone)[0] == old_hash
    assert resolver.resolve("David")[0] == old_hash

    store = MemoryStore(db_path=db)
    await store.add(
        context_id=7, subject_key=old_hash, subject_label="David",
        kind="fact", content="runs a hedge fund out of Brooklyn", bot_id=3,
    )
    preamble = await build_preamble(
        memory_store=store, context_id=7, sender_phone=new_phone,
        sender_user_hash=new_hash, sender_label="David",
        name_registry=registry, bot_id=3,
    )
    assert "hedge fund out of Brooklyn" in preamble
