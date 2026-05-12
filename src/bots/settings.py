"""
Per-bot settings resolution.

`resolve_setting` is the single source of truth for "what value should
the LLM use for this bot/role/key right now?" It composes three
layers in this order, returning the first hit:

  1. bot-scoped override     (bot_llm_settings(bot_id, role, key))
  2. global setting(s)       (admin_settings[g] for g in global_keys)
  3. caller-supplied default

The list of global keys is what lets deep_think fall back to the
writer's settings when its own aren't configured: pass
`global_keys=['deep_think_temperature','llm_temperature']` and the
resolver tries each in order. For the writer it's just
`['llm_temperature']`. The order is significant — first hit wins.

Typed wrappers (`resolve_int` / `resolve_float` / `resolve_bool`)
match `SettingsStore`'s typed accessors so the call site stays the
same shape it always was: `resolve_int(store, bot_id, role, 'max_tokens',
global_keys=['llm_max_tokens'], default=1000)`.

When `bot_id` is None, the bot-scoped lookup is skipped entirely —
behavior reduces to the legacy "global keys only" chain. This is
the back-compat path for callers that aren't bot-aware yet.
"""

from __future__ import annotations

from typing import Any, Optional


def resolve_setting(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: Any = None,
) -> Any:
    if bot_id is not None:
        v = store.get_bot(bot_id, role, key, default=None)
        if v is not None:
            return v
    for gk in global_keys:
        v = store.get(gk)
        if v is not None:
            return v
    return default


def _coerce_int(raw: Any, default: int, min_value: Optional[int] = None) -> int:
    try:
        v = int(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        v = default
    if min_value is not None and v < min_value:
        v = min_value
    return v


def _coerce_float(raw: Any, default: float) -> float:
    try:
        return float(raw) if raw not in (None, "") else default
    except (TypeError, ValueError):
        return default


def _coerce_bool(raw: Any, default: bool) -> bool:
    """Coerce a stored setting to bool.

    Strings need their own truthiness rule — `bool("0")` and `bool("false")`
    are both True in Python, which would silently invert any admin-entered
    "0" / "false" / "no" in the per-bot form. Numbers and proper bools use
    the standard truthiness.
    """
    if raw is None:
        return default
    if isinstance(raw, str):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    return bool(raw)


def resolve_int(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: int,
    min_value: Optional[int] = None,
) -> int:
    raw = resolve_setting(store, bot_id, role, key, global_keys=global_keys, default=None)
    return _coerce_int(raw, default, min_value)


def resolve_float(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: float,
) -> float:
    raw = resolve_setting(store, bot_id, role, key, global_keys=global_keys, default=None)
    return _coerce_float(raw, default)


def resolve_bool(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: bool = False,
) -> bool:
    raw = resolve_setting(store, bot_id, role, key, global_keys=global_keys, default=None)
    return _coerce_bool(raw, default)


def resolve_str(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: str = "",
) -> str:
    raw = resolve_setting(store, bot_id, role, key, global_keys=global_keys, default=None)
    return str(raw) if raw is not None else default


def resolve_stripped(
    store,
    bot_id: Optional[int],
    role: str,
    key: str,
    *,
    global_keys: list[str],
    default: str = "",
) -> str:
    return resolve_str(
        store, bot_id, role, key, global_keys=global_keys, default=default
    ).strip()
