"""
Factory for the admin Blueprint.

Mounts auth + dashboard + settings under /admin/*.
"""

import asyncio
import json
import logging
from datetime import timedelta
from typing import Optional

from flask import Blueprint, Flask, Response, abort, flash, redirect, render_template, request, session, url_for

from ..bots import Bot, BotRegistry
from ..bots.models import DEEP_THINK_MODES
from ..cache import get_cache_manager, get_metrics
from ..contexts import ContextPolicy, ContextRegistry
from .events import get_bus
from ..contexts.policy import MODE_ALLOW_ALL, MODE_ALLOW_LIST, MODE_DENY_LIST, MODES
from ..mcp_integration import MCPManager, MCPRegistry, MCPServerConfig
from ..mcp_integration.models import TRANSPORTS
from ..settings_store import ALLOWED_KEYS, LIVE_KEYS, RESTART_KEYS, SettingsStore
from .auth import (
    SESSION_LIFETIME_SECONDS,
    admin_required,
    csrf_token,
    register_auth_routes,
    verify_csrf,
)

logger = logging.getLogger(__name__)


def create_admin_blueprint(
    *,
    app: Flask,
    password_hash: bytes,
    settings_store: SettingsStore,
    signal_handler,
    loop: asyncio.AbstractEventLoop,
    admin_phone: str,
    provider_manager=None,
    mcp_registry: Optional[MCPRegistry] = None,
    mcp_manager: Optional[MCPManager] = None,
    context_registry: Optional[ContextRegistry] = None,
    dispatcher=None,
    name_registry=None,
    deep_think_client=None,
    memory_store=None,
    prediction_store=None,
    prediction_resolver=None,
    oracle_store=None,
    portfolio_store=None,
    portfolio_executor=None,
    bot_registry: Optional[BotRegistry] = None,
) -> Blueprint:
    """Build the admin blueprint and register it on `app`."""
    bp = Blueprint(
        "admin",
        __name__,
        url_prefix="/admin",
        template_folder="templates",
        static_folder="static",
    )

    app.permanent_session_lifetime = timedelta(seconds=SESSION_LIFETIME_SECONDS)

    register_auth_routes(
        bp,
        password_hash=password_hash,
        signal_handler=signal_handler,
        loop=loop,
        admin_phone=admin_phone,
    )

    _register_dashboard_routes(
        bp,
        settings_store=settings_store,
        provider_manager=provider_manager,
        mcp_registry=mcp_registry,
        mcp_manager=mcp_manager,
        context_registry=context_registry,
        db_path=str(settings_store.db_path) if settings_store else None,
        loop=loop,
        deep_think_client=deep_think_client,
    )
    _register_llm_routes(bp, settings_store=settings_store)

    if bot_registry is not None:
        _register_bots_routes(
            bp,
            registry=bot_registry,
            settings_store=settings_store,
            context_registry=context_registry,
            loop=loop,
        )

    if mcp_registry is not None and mcp_manager is not None:
        _register_mcp_routes(
            bp,
            registry=mcp_registry,
            manager=mcp_manager,
            loop=loop,
        )

    if context_registry is not None:
        _register_context_routes(
            bp,
            registry=context_registry,
            mcp_registry=mcp_registry,
            dispatcher=dispatcher,
            loop=loop,
            memory_store=memory_store,
            name_registry=name_registry,
            oracle_store=oracle_store,
            bot_registry=bot_registry,
        )

    if name_registry is not None:
        _register_users_routes(bp, registry=name_registry, loop=loop)

    if prediction_store is not None:
        _register_predictions_routes(
            bp,
            prediction_store=prediction_store,
            prediction_resolver=prediction_resolver,
            context_registry=context_registry,
            name_registry=name_registry,
            loop=loop,
        )

    if portfolio_store is not None and portfolio_executor is not None:
        _register_portfolio_routes(
            bp,
            portfolio_store=portfolio_store,
            portfolio_executor=portfolio_executor,
            context_registry=context_registry,
            loop=loop,
        )

    _register_live_routes(bp, name_registry=name_registry, loop=loop)

    # Make csrf_token available to every rendered template in this blueprint.
    @bp.context_processor
    def inject_csrf():
        return {"csrf_token": csrf_token}

    # Expose the set of registered admin endpoints so base.html's nav can
    # gate links on whether the route actually exists. Without this,
    # disabling MCP / portfolios / predictions / etc. via missing
    # dependencies would 500 every page that extends base.html.
    @app.context_processor
    def inject_endpoint_set():
        return {
            "admin_endpoints": {
                rule.endpoint for rule in app.url_map.iter_rules()
                if rule.endpoint.startswith("admin.")
            }
        }

    app.register_blueprint(bp)
    return bp


_DB_TABLES_FOR_DASH = (
    "watchlists",
    "alerts",
    "conversation_turns",
    "group_messages",
    "contexts",
    "mcp_servers",
    "admin_settings",
)


def _collect_db_stats(db_path: str) -> dict:
    """Synchronous SQLite probe for the dashboard. Light queries only."""
    import os
    import sqlite3

    out = {"path": db_path, "size_bytes": 0, "size_human": "0 B", "counts": {}}
    try:
        size = os.path.getsize(db_path)
        out["size_bytes"] = size
        out["size_human"] = _human_bytes(size)
    except OSError:
        pass

    try:
        with sqlite3.connect(db_path) as conn:
            for table in _DB_TABLES_FOR_DASH:
                try:
                    cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
                    row = cur.fetchone()
                    out["counts"][table] = row[0] if row else 0
                except sqlite3.OperationalError:
                    out["counts"][table] = None  # table absent
    except sqlite3.Error as e:
        logger.debug(f"DB stats probe failed: {e}")
    return out


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024:
            return f"{n:.0f} {unit}" if unit == "B" else f"{n:.1f} {unit}"
        n /= 1024  # type: ignore
    return f"{n:.1f} PB"


def _register_dashboard_routes(
    bp: Blueprint,
    *,
    settings_store: SettingsStore,
    provider_manager,
    mcp_registry: Optional[MCPRegistry] = None,
    mcp_manager: Optional[MCPManager] = None,
    context_registry: Optional[ContextRegistry] = None,
    db_path: Optional[str] = None,
    loop: Optional[asyncio.AbstractEventLoop] = None,
    deep_think_client=None,
) -> None:
    @bp.route("/", methods=["GET"])
    @admin_required
    def dashboard():
        from ..metrics_log import get_metrics_log, parse_window, WINDOWS

        # Window selector — `?window=24h|7d|30d`. Default 24h. Unknown
        # values fall back to default rather than 4xx so a stale link
        # from an old admin session doesn't 400.
        window_key, window_seconds = parse_window(request.args.get("window"))

        metrics = get_metrics().get_all_stats()
        provider_status = provider_manager.get_status() if provider_manager else {}

        # Pull windowed stats from the persistent log. Replaces the
        # since-boot LLM/deep-think/reactor counters in `metrics` so the
        # template gets a 24h/7d/30d view that survives restarts.
        windowed = None
        if loop is not None:
            try:
                windowed = _run_on_loop(loop, get_metrics_log().query_window(window_seconds))
                # Splice the windowed values into the same keys the
                # existing template reads (llm/deep_think/reactor) — keeps
                # the template diff minimal and the cumulative-since-boot
                # numbers still available under the *_lifetime keys.
                metrics["llm_lifetime"] = metrics.get("llm")
                metrics["deep_think_lifetime"] = metrics.get("deep_think")
                metrics["reactor_lifetime"] = metrics.get("reactor")
                metrics["llm"] = windowed["llm"]
                metrics["deep_think"] = windowed["deep_think"]
                metrics["reactor"] = windowed["reactor"]
                metrics["window_request_count"] = windowed["request_count"]
            except Exception as e:
                logger.error(f"Dashboard windowed metrics failed: {e}")

        metrics["window_key"] = window_key
        metrics["window_options"] = list(WINDOWS.keys())

        # MCP — server list + per-session status
        mcp_view = []
        if mcp_registry is not None and mcp_manager is not None and loop is not None:
            try:
                servers = _run_on_loop(loop, mcp_registry.list())
                status_map = mcp_manager.status()
                for s in servers:
                    st = status_map.get(s.id) or {}
                    mcp_view.append({
                        "name": s.name,
                        "transport": s.transport,
                        "enabled": s.enabled,
                        "running": bool(st.get("running")),
                        "tool_count": st.get("tool_count", 0),
                        "last_error": st.get("last_error"),
                    })
            except Exception as e:
                logger.error(f"Dashboard MCP collect failed: {e}")

        # Contexts — count by kind
        context_view = {"total": 0, "group": 0, "dm": 0, "default": 0,
                        "with_intent": 0, "with_reactor": 0, "with_prompt": 0}
        if context_registry is not None and loop is not None:
            try:
                rows = _run_on_loop(loop, context_registry.list())
                for c in rows:
                    context_view["total"] += 1
                    if c.kind in context_view:
                        context_view[c.kind] += 1
                    if c.llm_intent:
                        context_view["with_intent"] += 1
                    if c.reactor_enabled:
                        context_view["with_reactor"] += 1
                    if c.system_prompt:
                        context_view["with_prompt"] += 1
            except Exception as e:
                logger.error(f"Dashboard contexts collect failed: {e}")

        # Storage — DB size + table counts
        storage_view = _collect_db_stats(db_path) if db_path else {}

        # Reactor stats include emoji breakdown — also expose top-1 for header
        reactor_view = dict(metrics.get("reactor") or {})
        reactor_view["top_emoji"] = (
            reactor_view.get("top_emojis", [[None, 0]])[0][0]
            if reactor_view.get("top_emojis") else None
        )

        # Deep-think stats: status snapshot + today's usage + recent calls
        # log so the dashboard surfaces "is it on, who's hitting it, what
        # have they asked, how slow/cheap was it" without leaving the page.
        deep_think_view: dict = dict(metrics.get("deep_think") or {})
        if deep_think_client is not None:
            try:
                deep_think_view["status"] = deep_think_client.status()
                deep_think_view["usage_today"] = deep_think_client.usage_today()
                deep_think_view["recent_calls"] = deep_think_client.recent_calls(limit=10)
            except Exception as e:
                logger.error(f"Dashboard deep_think collect failed: {e}")
                deep_think_view["status"] = {"enabled": False, "ready": False}
                deep_think_view["usage_today"] = {"total_calls": 0, "per_user": {}, "per_group": {}}
                deep_think_view["recent_calls"] = []
        else:
            deep_think_view["status"] = {"enabled": False, "ready": False}
            deep_think_view["usage_today"] = {"total_calls": 0, "per_user": {}, "per_group": {}}
            deep_think_view["recent_calls"] = []

        return render_template(
            "dashboard.html",
            metrics=metrics,
            provider_status=provider_status,
            mcp_view=mcp_view,
            context_view=context_view,
            storage_view=storage_view,
            reactor_view=reactor_view,
            deep_think_view=deep_think_view,
        )

    @bp.route("/settings", methods=["GET", "POST"])
    @admin_required
    def settings_page():
        saved = False
        error = None

        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    _apply_settings_form(settings_store, request.form)
                    saved = True
                except ValueError as e:
                    error = str(e)

        values = _collect_setting_values(settings_store)
        return render_template(
            "settings.html",
            values=values,
            live_keys=sorted(LIVE_KEYS),
            restart_keys=sorted(RESTART_KEYS),
            saved=saved,
            error=error,
        )

    @bp.route("/cache/clear", methods=["POST"])
    @admin_required
    def cache_clear():
        if verify_csrf():
            get_cache_manager().clear_all()
        return redirect(url_for("admin.dashboard"))


# Keys exposed on /admin/llm. Ordering here controls display order.
LLM_KEYS = [
    "llm_enabled",
    "llm_base_url",
    "llm_model",
    "llm_api_key",
    "llm_temperature",
    "llm_max_tokens",
    "llm_timeout_seconds",
    "llm_history_turns",
    "llm_retention_days",
    "llm_max_tool_rounds",
    "group_context_messages",
    "ask_command_name",
    "llm_augment_commands",
    "llm_augment_prompt",
    "llm_system_prompt",
    "llm_response_style",
    "llm_extra_body",
    # OpenRouter provider routing — pin which upstream provider serves
    # the model so latency stays predictable across requests.
    "llm_provider_order",
    "llm_provider_only",
    "llm_provider_sort",
    # Emoji reactor (cheap secondary path)
    "reactor_enabled",
    "reactor_model",
    "reactor_max_tokens",
    "reactor_temperature",
    "reactor_extra_body",
    "reactor_min_length",
    "reactor_sender_cooldown",
    "reactor_group_cooldown",
    "reactor_context_messages",
    "reactor_system_prompt",
    # Natural response: piggybacks on the reactor model. Global kill switch +
    # tunables; per-context opt-in lives on ContextPolicy.natural_response.
    "natural_response_enabled",
    "natural_response_cooldown",
    "natural_response_extra_prompt",
    # Deep think (separate model for hard sub-problems)
    "deep_think_enabled",
    "deep_think_base_url",
    "deep_think_api_key",
    "deep_think_model",
    "deep_think_temperature",
    "deep_think_max_tokens",
    "deep_think_timeout_seconds",
    "deep_think_extra_body",
    "deep_think_system_prompt",
    "deep_think_context_max_chars",
    "deep_think_max_tool_rounds",
    "deep_think_caps_enabled",
    "deep_think_user_daily_cap",
    "deep_think_group_daily_cap",
]

# Imported lazily inside the route to avoid a circular import.
def _default_response_style():
    from ..llm.client import DEFAULT_RESPONSE_STYLE
    return DEFAULT_RESPONSE_STYLE


LLM_DEFAULTS = {
    "llm_enabled": False,
    "llm_base_url": "https://api.openai.com/v1",
    "llm_model": "gpt-4o-mini",
    "llm_temperature": 0.7,
    "llm_max_tokens": 1000,
    "llm_timeout_seconds": 30,
    "llm_history_turns": 6,
    "llm_retention_days": 7,
    "llm_max_tool_rounds": 25,
    "group_context_messages": 0,
    "ask_command_name": "ask",
    "llm_augment_commands": "",
    "llm_augment_prompt": "You are looking at the output of a stock-bot command. Add a brief (1-2 sentence) plain-language interpretation. No headers, no lists.",
    "llm_system_prompt": "",
    "llm_response_style": "",   # empty => use the built-in DEFAULT_RESPONSE_STYLE
    "llm_extra_body": "",
    # Reactor defaults
    "reactor_enabled": False,
    "reactor_model": "",                   # empty => same as main llm_model
    "reactor_max_tokens": 50,
    "reactor_temperature": 0.3,
    "reactor_extra_body": "",
    "reactor_min_length": 0,
    "reactor_sender_cooldown": 30,
    "reactor_group_cooldown": 10,
    "reactor_context_messages": 5,
    "reactor_system_prompt": "",           # empty => use the built-in DEFAULT_REACTOR_PROMPT
    # Natural response (piggybacks on reactor model)
    "natural_response_enabled": False,
    "natural_response_cooldown": 300,      # per-group seconds between spontaneous replies
    "natural_response_extra_prompt": "",   # appended to reactor prompt; empty => built-in default
    # Deep think — empty string fields fall back to llm_* equivalents at runtime,
    # so admins can override only the model + max_tokens to point at a smarter
    # endpoint and inherit everything else.
    "deep_think_enabled": False,
    "deep_think_base_url": "",
    "deep_think_api_key": "",
    "deep_think_model": "",
    "deep_think_temperature": 0.7,
    "deep_think_max_tokens": 8000,
    "deep_think_timeout_seconds": 120,
    "deep_think_extra_body": "",
    "deep_think_system_prompt": "",        # empty => use built-in CAREFUL_REASONER prompt
    "deep_think_context_max_chars": 8000,  # writer-supplied context budget cap
    "deep_think_max_tool_rounds": 15,      # cap on tool-loop rounds inside one think() call
    "deep_think_caps_enabled": False,      # counters always track; enforcement gated by this
    "deep_think_user_daily_cap": 10,
    "deep_think_group_daily_cap": 50,
}


def _register_llm_routes(bp: Blueprint, *, settings_store: SettingsStore) -> None:
    @bp.route("/llm", methods=["GET", "POST"])
    @admin_required
    def llm_page():
        saved = False
        error = None

        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    _apply_llm_form(settings_store, request.form)
                    saved = True
                except ValueError as e:
                    error = str(e)

        values = {}
        for key in LLM_KEYS:
            values[key] = settings_store.get(key, LLM_DEFAULTS.get(key))

        # Don't round-trip the API key to the browser — show a "configured" flag.
        api_key_set = bool(values.get("llm_api_key"))
        values["llm_api_key"] = ""
        deep_think_api_key_set = bool(values.get("deep_think_api_key"))
        values["deep_think_api_key"] = ""

        return render_template(
            "llm.html",
            values=values,
            api_key_set=api_key_set,
            deep_think_api_key_set=deep_think_api_key_set,
            saved=saved,
            error=error,
        )


def _apply_llm_form(store: SettingsStore, form) -> None:
    """Persist LLM form values with schema-aware coercion."""
    import json

    bool_keys = {
        "llm_enabled",
        "llm_provider_only",
        "reactor_enabled",
        "natural_response_enabled",
        "deep_think_enabled",
        "deep_think_caps_enabled",
    }
    int_keys = {
        "llm_max_tokens",
        "llm_timeout_seconds",
        "llm_history_turns",
        "llm_retention_days",
        "llm_max_tool_rounds",
        "group_context_messages",
        "reactor_max_tokens",
        "reactor_min_length",
        "reactor_sender_cooldown",
        "reactor_group_cooldown",
        "reactor_context_messages",
        "natural_response_cooldown",
        "deep_think_max_tokens",
        "deep_think_timeout_seconds",
        "deep_think_context_max_chars",
        "deep_think_max_tool_rounds",
        "deep_think_user_daily_cap",
        "deep_think_group_daily_cap",
    }
    float_keys = {"llm_temperature", "reactor_temperature", "deep_think_temperature"}

    for key in LLM_KEYS:
        if key in ("llm_api_key", "deep_think_api_key"):
            # Only overwrite when the user typed a new value; empty = keep existing.
            # Same masked-input pattern for both API keys.
            submitted = form.get(key, "").strip()
            if submitted:
                store.set(key, submitted)
            continue

        raw = form.get(key, "").strip()

        if key in bool_keys:
            store.set(key, raw.lower() in ("1", "true", "yes", "on"))
            continue

        if key in int_keys:
            if not raw:
                continue
            try:
                store.set(key, int(raw))
            except ValueError:
                raise ValueError(f"{key} must be an integer") from None
            continue

        if key in float_keys:
            if not raw:
                continue
            try:
                store.set(key, float(raw))
            except ValueError:
                raise ValueError(f"{key} must be a number") from None
            continue

        if key in ("llm_extra_body", "reactor_extra_body"):
            if raw:
                # Validate JSON up-front; reject anything that's not an object.
                try:
                    parsed = json.loads(raw)
                except json.JSONDecodeError as e:
                    raise ValueError(f"{key} is not valid JSON: {e.msg}") from None
                if not isinstance(parsed, dict):
                    raise ValueError(f"{key} must be a JSON object (e.g. {{\"key\": \"value\"}})")
            store.set(key, raw)
            continue

        # Strings (base URL, model, command name, system prompt)
        store.set(key, raw)


# ---------------------------------------------------------------------------
# Bots routes — identity CRUD + per-bot, per-role LLM overrides
# ---------------------------------------------------------------------------

# Per-role override fields exposed on the bot edit form. Empty submission
# = "remove the bot-scoped row, fall back to the global llm_*/reactor_*
# /deep_think_* setting." That's why we DELETE on blank rather than
# storing an empty string — it preserves the bot -> global -> default
# resolver behavior.
_BOT_LLM_FIELDS = {
    "writer": [
        ("base_url", "str"),
        ("api_key", "secret"),
        ("model", "str"),
        ("temperature", "float"),
        ("max_tokens", "int"),
        ("timeout_seconds", "int"),
        ("system_prompt", "str"),
        ("extra_body", "json"),
    ],
    "reactor": [
        ("model", "str"),
        ("temperature", "float"),
        ("max_tokens", "int"),
        ("system_prompt", "str"),
        ("extra_body", "json"),
    ],
    "deep_think": [
        ("base_url", "str"),
        ("api_key", "secret"),
        ("model", "str"),
        ("temperature", "float"),
        ("max_tokens", "int"),
        ("timeout_seconds", "int"),
        ("system_prompt", "str"),
        ("extra_body", "json"),
    ],
}


def _form_to_bot(form, *, existing: Optional[Bot]) -> Bot:
    """Read identity fields from the form into a Bot. Aliases come in
    as comma- or newline-separated text; deduplicated and trimmed."""
    raw_aliases = form.get("aliases", "") or ""
    aliases: list[str] = []
    seen: set[str] = set()
    for chunk in raw_aliases.replace(",", "\n").split("\n"):
        a = chunk.strip()
        if not a:
            continue
        low = a.lower()
        if low in seen:
            continue
        seen.add(low)
        aliases.append(a)

    slug = (form.get("slug", "") or "").strip().lower()
    display_name = (form.get("display_name", "") or "").strip()
    if not slug:
        raise ValueError("slug is required (lowercase, stable; not user-facing)")
    if not display_name:
        raise ValueError("display_name is required")
    # Default the alias list to [display_name] if nothing was entered —
    # without an alias, the bot can't be named-triggered from chat.
    if not aliases:
        aliases = [display_name]

    deep_think_mode = (form.get("deep_think_mode", "") or "replace").strip()
    if deep_think_mode not in DEEP_THINK_MODES:
        raise ValueError(
            f"deep_think_mode must be one of {DEEP_THINK_MODES}"
        )

    return Bot(
        id=existing.id if existing else None,
        slug=slug,
        display_name=display_name,
        aliases=aliases,
        persona=(form.get("persona", "") or "").strip() or None,
        signal_phone=(form.get("signal_phone", "") or "").strip() or None,
        signal_api_url=(form.get("signal_api_url", "") or "").strip() or None,
        enabled=(form.get("enabled", "") or "").lower() in ("1", "true", "yes", "on"),
        default_for_dm=(form.get("default_for_dm", "") or "").lower() in ("1", "true", "yes", "on"),
        default_for_group=(form.get("default_for_group", "") or "").lower() in ("1", "true", "yes", "on"),
        deep_think_mode=deep_think_mode,
        deep_think_handoff_prompt=(form.get("deep_think_handoff_prompt", "") or "").strip() or None,
    )


def _apply_bot_llm_form(
    store: SettingsStore, bot_id: int, form
) -> None:
    """Persist per-bot, per-role overrides. Each input is named
    `<role>__<key>` to avoid colliding with the identity fields. Empty
    values DELETE the corresponding bot-scoped row so the resolver
    falls back to the global setting on next read."""
    import json as _json

    for role, fields in _BOT_LLM_FIELDS.items():
        for key, kind in fields:
            field = f"{role}__{key}"
            raw = form.get(field, "").strip()
            if kind == "secret":
                # Masked input: only overwrite when the operator typed
                # something. An empty submission keeps the existing
                # bot-scoped value. To clear an api_key, the operator
                # uses the explicit "clear" button which posts
                # `<role>__<key>__clear=1`.
                if form.get(f"{field}__clear", "").strip() in ("1", "true", "yes", "on"):
                    store.delete_bot(bot_id, role, key)
                elif raw:
                    store.set_bot(bot_id, role, key, raw)
                continue
            if not raw:
                # Blank string for any non-secret field clears the
                # bot-scoped row → fall back to global on next read.
                store.delete_bot(bot_id, role, key)
                continue
            if kind == "int":
                try:
                    store.set_bot(bot_id, role, key, int(raw))
                except ValueError:
                    raise ValueError(
                        f"{field} must be an integer (got {raw!r})"
                    ) from None
                continue
            if kind == "float":
                try:
                    store.set_bot(bot_id, role, key, float(raw))
                except ValueError:
                    raise ValueError(
                        f"{field} must be a number (got {raw!r})"
                    ) from None
                continue
            if kind == "json":
                try:
                    parsed = _json.loads(raw)
                except _json.JSONDecodeError as e:
                    raise ValueError(
                        f"{field} is not valid JSON: {e.msg}"
                    ) from None
                if not isinstance(parsed, dict):
                    raise ValueError(
                        f"{field} must be a JSON object"
                    )
                store.set_bot(bot_id, role, key, raw)
                continue
            store.set_bot(bot_id, role, key, raw)


def _bot_llm_values(
    store: SettingsStore, bot_id: Optional[int]
) -> dict:
    """Build the {role: {key: value}} dict the form pre-populates from.
    api_keys are intentionally blanked so they aren't echoed to the
    browser; a parallel `<role>__api_key__set` flag indicates whether
    one is currently configured.

    Top-level keys are `fields` and `secrets_set` (NOT `values` — that
    collides with `dict.values` in Jinja and resolves to the bound
    method instead of the data)."""
    out: dict[str, dict] = {}
    flags: dict[str, dict] = {}
    if bot_id is None:
        # New-bot form: every override field starts empty.
        for role, fields in _BOT_LLM_FIELDS.items():
            out[role] = {key: "" for key, _ in fields}
            flags[role] = {key: False for key, kind in fields if kind == "secret"}
        return {"fields": out, "secrets_set": flags}
    bot_settings = store.list_bot(bot_id)
    for role, fields in _BOT_LLM_FIELDS.items():
        slot = bot_settings.get(role, {})
        out[role] = {}
        flags[role] = {}
        for key, kind in fields:
            v = slot.get(key, "")
            if kind == "secret":
                flags[role][key] = bool(v)
                out[role][key] = ""
            else:
                # Coerce typed values back to their string form for the
                # form input. Float "0.7" round-trips fine; True/False
                # for booleans isn't used in the override form.
                out[role][key] = "" if v in (None, "") else str(v)
    return {"fields": out, "secrets_set": flags}


def _register_bots_routes(
    bp: Blueprint,
    *,
    registry: BotRegistry,
    settings_store: SettingsStore,
    context_registry: Optional[ContextRegistry],
    loop: asyncio.AbstractEventLoop,
) -> None:
    @bp.route("/bots", methods=["GET"])
    @admin_required
    def bots_list():
        bots = _run_on_loop(loop, registry.list())
        # Map context counts per bot for the list view ("answers in N
        # contexts"). Cheap query; no N+1 because contexts is a small
        # table.
        context_counts: dict[int, int] = {}
        if context_registry is not None:
            try:
                rows = _run_on_loop(loop, context_registry.list())
                for r in rows:
                    bid = getattr(r, "default_bot_id", None)
                    if bid is not None:
                        context_counts[bid] = context_counts.get(bid, 0) + 1
            except Exception as e:
                logger.warning(f"bots_list context count failed: {e}")
        return render_template(
            "bots_list.html",
            bots=bots,
            context_counts=context_counts,
        )

    @bp.route("/bots/new", methods=["GET", "POST"])
    @admin_required
    def bots_new():
        error = None
        # Pre-populate aliases default to display_name so the form
        # doesn't render with no alias (which would make the bot
        # un-triggerable). User can edit before submit.
        defaults = {
            "slug": "",
            "display_name": "",
            "aliases": "",
            "persona": "",
            "signal_phone": "",
            "signal_api_url": "",
            "enabled": True,
            "default_for_dm": False,
            "default_for_group": False,
            "deep_think_mode": "replace",
            "deep_think_handoff_prompt": "",
        }
        bot_settings = _bot_llm_values(settings_store, bot_id=None)
        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    new_bot = _form_to_bot(request.form, existing=None)
                    new_id = _run_on_loop(loop, registry.upsert(new_bot))
                    _apply_bot_llm_form(settings_store, new_id, request.form)
                    return redirect(url_for("admin.bots_edit", bot_id=new_id))
                except ValueError as e:
                    error = str(e)
            defaults = {k: request.form.get(k, defaults[k]) for k in defaults}
        return render_template(
            "bots_form.html",
            is_new=True,
            bot=None,
            values=defaults,
            roles=list(_BOT_LLM_FIELDS.keys()),
            role_fields=_BOT_LLM_FIELDS,
            bot_settings=bot_settings,
            deep_think_modes=DEEP_THINK_MODES,
            error=error,
        )

    @bp.route("/bots/<int:bot_id>", methods=["GET", "POST"])
    @admin_required
    def bots_edit(bot_id: int):
        error = None
        bot = _run_on_loop(loop, registry.get(bot_id))
        if not bot:
            abort(404)

        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    updated = _form_to_bot(request.form, existing=bot)
                    _run_on_loop(loop, registry.upsert(updated))
                    _apply_bot_llm_form(settings_store, bot_id, request.form)
                    return redirect(url_for("admin.bots_edit", bot_id=bot_id))
                except ValueError as e:
                    error = str(e)

        bot_settings = _bot_llm_values(settings_store, bot_id=bot_id)
        values = {
            "slug": bot.slug,
            "display_name": bot.display_name,
            "aliases": "\n".join(bot.aliases),
            "persona": bot.persona or "",
            "signal_phone": bot.signal_phone or "",
            "signal_api_url": bot.signal_api_url or "",
            "enabled": bot.enabled,
            "default_for_dm": bot.default_for_dm,
            "default_for_group": bot.default_for_group,
            "deep_think_mode": bot.deep_think_mode,
            "deep_think_handoff_prompt": bot.deep_think_handoff_prompt or "",
        }
        return render_template(
            "bots_form.html",
            is_new=False,
            bot=bot,
            values=values,
            roles=list(_BOT_LLM_FIELDS.keys()),
            role_fields=_BOT_LLM_FIELDS,
            bot_settings=bot_settings,
            deep_think_modes=DEEP_THINK_MODES,
            error=error,
        )

    @bp.route("/bots/<int:bot_id>/delete", methods=["POST"])
    @admin_required
    def bots_delete(bot_id: int):
        if verify_csrf():
            try:
                ok = _run_on_loop(loop, registry.delete(bot_id))
                if not ok:
                    flash("Cannot delete: bot is the default sigil row, "
                          "or one or more contexts still reference it. "
                          "Reassign those contexts first.")
                else:
                    # Wipe per-bot overrides too — the bot_id is gone
                    # so the rows would orphan otherwise.
                    settings_store.delete_bot(bot_id)
            except Exception as e:
                logger.error(f"bots_delete failed: {e}")
                flash(f"Delete failed: {e}")
        return redirect(url_for("admin.bots_list"))


# ---------------------------------------------------------------------------
# MCP routes
# ---------------------------------------------------------------------------


def _run_on_loop(loop: asyncio.AbstractEventLoop, coro, timeout: float = 45.0):
    """Submit a coroutine to the bot's shared loop and block on its result."""
    return asyncio.run_coroutine_threadsafe(coro, loop).result(timeout=timeout)


def _parse_kv_textarea(raw: str) -> dict[str, str]:
    """Parse KEY=value lines from a textarea into a dict. Empty lines ignored."""
    out: dict[str, str] = {}
    for line in (raw or "").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            raise ValueError(f"Invalid KEY=value line: {line!r}")
        k, _, v = line.partition("=")
        k = k.strip()
        if not k:
            raise ValueError(f"Empty key in line: {line!r}")
        out[k] = v
    return out


def _parse_args(raw: str) -> list[str]:
    """Parse stdio args — one per non-empty line (handles spaces in args)."""
    return [ln.strip() for ln in (raw or "").splitlines() if ln.strip()]


def _form_to_config(form, existing_id: Optional[int] = None) -> MCPServerConfig:
    name = (form.get("name") or "").strip()
    transport = (form.get("transport") or "").strip()
    enabled = (form.get("enabled") or "").strip().lower() in ("1", "true", "on", "yes")
    command = (form.get("command") or "").strip() or None
    url = (form.get("url") or "").strip() or None
    args = _parse_args(form.get("args") or "")
    env = _parse_kv_textarea(form.get("env") or "")
    headers = _parse_kv_textarea(form.get("headers") or "")
    cfg = MCPServerConfig(
        id=existing_id,
        name=name,
        transport=transport,
        enabled=enabled,
        command=command,
        args=args,
        env=env,
        url=url,
        headers=headers,
    )
    errs = cfg.validate()
    if errs:
        raise ValueError("; ".join(errs))
    return cfg


def _register_mcp_routes(
    bp: Blueprint,
    *,
    registry: MCPRegistry,
    manager: MCPManager,
    loop: asyncio.AbstractEventLoop,
) -> None:
    @bp.route("/mcp", methods=["GET"])
    @admin_required
    def mcp_list():
        servers = _run_on_loop(loop, registry.list())
        status = manager.status()
        return render_template("mcp_list.html", servers=servers, status=status)

    @bp.route("/mcp/new", methods=["GET", "POST"])
    @admin_required
    def mcp_new():
        error = None
        values = {
            "name": "",
            "transport": "stdio",
            "enabled": "false",
            "command": "",
            "args": "",
            "env": "",
            "url": "",
            "headers": "",
        }
        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    cfg = _form_to_config(request.form)
                    _run_on_loop(loop, registry.upsert(cfg))
                    return redirect(url_for("admin.mcp_list"))
                except ValueError as e:
                    error = str(e)
                except Exception as e:
                    error = f"Save failed: {e}"
            values = {k: request.form.get(k, "") for k in values}
        return render_template(
            "mcp_form.html",
            is_new=True,
            values=values,
            transports=TRANSPORTS,
            error=error,
            server=None,
            tools=[],
            status=None,
        )

    @bp.route("/mcp/<int:server_id>", methods=["GET", "POST"])
    @admin_required
    def mcp_edit(server_id: int):
        error = None
        server = _run_on_loop(loop, registry.get(server_id))
        if not server:
            abort(404)

        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    cfg = _form_to_config(request.form, existing_id=server_id)
                    _run_on_loop(loop, registry.upsert(cfg))
                    # If it's running, hot-reload so new config takes effect.
                    if manager.get_session(server_id) is not None:
                        try:
                            _run_on_loop(loop, manager.restart_server(server_id))
                        except Exception as e:
                            error = f"Saved, but restart failed: {e}"
                    if not error:
                        return redirect(url_for("admin.mcp_edit", server_id=server_id))
                except ValueError as e:
                    error = str(e)
                except Exception as e:
                    error = f"Save failed: {e}"
            # Fall through with submitted values
            values = {k: request.form.get(k, "") for k in (
                "name", "transport", "enabled", "command", "args", "env", "url", "headers"
            )}
        else:
            values = {
                "name": server.name,
                "transport": server.transport,
                "enabled": "true" if server.enabled else "false",
                "command": server.command or "",
                "args": "\n".join(server.args or []),
                "env": "\n".join(f"{k}={v}" for k, v in (server.env or {}).items()),
                "url": server.url or "",
                "headers": "\n".join(f"{k}={v}" for k, v in (server.headers or {}).items()),
            }

        session_obj = manager.get_session(server_id)
        tools = session_obj.tools if session_obj else []
        status_entry = manager.status().get(server_id)
        return render_template(
            "mcp_form.html",
            is_new=False,
            values=values,
            transports=TRANSPORTS,
            error=error,
            server=server,
            tools=tools,
            status=status_entry,
        )

    @bp.route("/mcp/<int:server_id>/start", methods=["POST"])
    @admin_required
    def mcp_start(server_id: int):
        if verify_csrf():
            try:
                _run_on_loop(loop, manager.start_server(server_id))
            except Exception as e:
                logger.error(f"MCP start failed: {e}")
        return redirect(url_for("admin.mcp_edit", server_id=server_id))

    @bp.route("/mcp/<int:server_id>/stop", methods=["POST"])
    @admin_required
    def mcp_stop(server_id: int):
        if verify_csrf():
            try:
                _run_on_loop(loop, manager.stop_server(server_id))
            except Exception as e:
                logger.error(f"MCP stop failed: {e}")
        return redirect(url_for("admin.mcp_edit", server_id=server_id))

    @bp.route("/mcp/<int:server_id>/delete", methods=["POST"])
    @admin_required
    def mcp_delete(server_id: int):
        if verify_csrf():
            try:
                _run_on_loop(loop, manager.stop_server(server_id))
            except Exception:
                pass
            try:
                _run_on_loop(loop, registry.delete(server_id))
            except Exception as e:
                logger.error(f"MCP delete failed: {e}")
        return redirect(url_for("admin.mcp_list"))


def _collect_setting_values(store: SettingsStore) -> dict[str, object]:
    """Current value for every admin-editable key (None when unset)."""
    stored = store.all()
    return {key: stored.get(key) for key in sorted(ALLOWED_KEYS)}


def _apply_settings_form(store: SettingsStore, form) -> None:
    """Persist form values, coercing types per known schema."""
    for key in ALLOWED_KEYS:
        if key not in form:
            continue

        raw = form.get(key, "").strip()

        if key == "admin_numbers":
            if raw:
                numbers = [n.strip() for n in raw.split(",") if n.strip()]
                store.set(key, numbers)
            else:
                store.set(key, [])
            continue

        if key == "MASSIVE_PRO":
            store.set(key, raw.lower() in ("1", "true", "yes", "on"))
            continue

        if key in ("user_rate_limit", "max_message_length"):
            if not raw:
                continue
            try:
                store.set(key, int(raw))
            except ValueError:
                raise ValueError(f"{key} must be an integer") from None
            continue

        # Default: string
        store.set(key, raw)


# ---------------------------------------------------------------------------
# Context routes (per-chat command / MCP / prompt policies)
# ---------------------------------------------------------------------------


def _register_context_routes(
    bp: Blueprint,
    *,
    registry: ContextRegistry,
    mcp_registry: Optional[MCPRegistry],
    dispatcher,
    loop: asyncio.AbstractEventLoop,
    memory_store=None,
    name_registry=None,
    oracle_store=None,
    bot_registry: Optional[BotRegistry] = None,
) -> None:
    """Context CRUD + (when memory_store is wired) memory CRUD nested under
    each context. Memory routes are mounted unconditionally so the template
    link at /admin/contexts/<id> never builds against a missing route."""
    @bp.route("/contexts", methods=["GET"])
    @admin_required
    def context_list():
        rows = _run_on_loop(loop, registry.list())
        return render_template("context_list.html", contexts=rows)

    def _bots_for_template():
        """Return the list of enabled bots for the default_bot_id select.
        Empty when bot_registry isn't wired so the select still renders
        with the 'inherit' option (the only one)."""
        if bot_registry is None:
            return []
        try:
            bots = bot_registry.list_sync()
            return [b for b in bots if b.enabled]
        except Exception as e:
            logger.warning(f"context page bot list failed: {e}")
            return []

    @bp.route("/contexts/new", methods=["GET", "POST"])
    @admin_required
    def context_new():
        error = None
        values = {
            "kind": "group",
            "key": "",
            "label": "",
            "command_mode": MODE_ALLOW_ALL,
            "commands": [],
            "mcp_mode": MODE_ALLOW_ALL,
            "mcp_servers": [],
            "system_prompt": "",
            "llm_intent": False,
            "reactor_enabled": True,
            "reactor_prompt": "",
            "natural_response": False,
            "deep_think_enabled": True,
            "memory_writes_enabled": True,
            "reactor_memory_writes": False,
            "default_bot_id": "",
        }
        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    policy = _form_to_policy(request.form, existing_id=None)
                    _run_on_loop(loop, registry.upsert(policy))
                    return redirect(url_for("admin.context_list"))
                except ValueError as e:
                    error = str(e)
            values = _form_to_values(request.form)

        all_commands, all_servers = _available_commands_and_servers(dispatcher, mcp_registry, loop)
        return render_template(
            "context_edit.html",
            is_new=True,
            policy=None,
            values=values,
            modes=MODES,
            all_commands=all_commands,
            all_servers=all_servers,
            all_bots=_bots_for_template(),
            error=error,
        )

    @bp.route("/contexts/<int:context_id>", methods=["GET", "POST"])
    @admin_required
    def context_edit(context_id: int):
        error = None
        policy = _run_on_loop(loop, registry.get(context_id))
        if not policy:
            abort(404)

        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload the page."
            else:
                try:
                    updated = _form_to_policy(request.form, existing_id=context_id, base=policy)
                    _run_on_loop(loop, registry.upsert(updated))
                    return redirect(url_for("admin.context_edit", context_id=context_id))
                except ValueError as e:
                    error = str(e)

        values = {
            "kind": policy.kind,
            "key": policy.key,
            "label": policy.label,
            "command_mode": policy.command_mode,
            "commands": policy.commands,
            "mcp_mode": policy.mcp_mode,
            "mcp_servers": policy.mcp_servers,
            "system_prompt": policy.system_prompt or "",
            "llm_intent": policy.llm_intent,
            "reactor_enabled": policy.reactor_enabled,
            "reactor_prompt": policy.reactor_prompt or "",
            "natural_response": policy.natural_response,
            "deep_think_enabled": policy.deep_think_enabled,
            "memory_writes_enabled": policy.memory_writes_enabled,
            "reactor_memory_writes": policy.reactor_memory_writes,
            "default_bot_id": (
                "" if policy.default_bot_id is None
                else str(policy.default_bot_id)
            ),
        }
        if request.method == "POST":
            values = _form_to_values(request.form)
            values["kind"] = policy.kind  # kind is immutable after creation
            values["key"] = policy.key

        all_commands, all_servers = _available_commands_and_servers(dispatcher, mcp_registry, loop)

        # Per-context oracles. Only group contexts can have them; for
        # default/dm rows we still pass an empty list so the template
        # always has the variable defined.
        oracles_view: list[dict] = []
        if oracle_store is not None and policy.kind == "group":
            try:
                rows = _run_on_loop(loop, oracle_store.list_for_context(context_id))
                oracles_view = [_oracle_to_view(o) for o in rows]
            except Exception as e:
                logger.error(f"Oracle list failed for ctx #{context_id}: {e}")

        return render_template(
            "context_edit.html",
            is_new=False,
            policy=policy,
            values=values,
            modes=MODES,
            all_commands=all_commands,
            all_servers=all_servers,
            all_bots=_bots_for_template(),
            oracles=oracles_view,
            oracle_kinds=_ORACLE_KIND_OPTIONS,
            oracle_schedule_kinds=_ORACLE_SCHEDULE_OPTIONS,
            oracle_kind_descriptions=_ORACLE_KIND_DESCRIPTIONS,
            oracle_schedule_descriptions=_ORACLE_SCHEDULE_DESCRIPTIONS,
            error=error,
        )

    @bp.route("/contexts/<int:context_id>/delete", methods=["POST"])
    @admin_required
    def context_delete(context_id: int):
        if verify_csrf():
            try:
                # Cascade memories before the row goes — orphans would
                # otherwise sit unaddressable in the DB if the context
                # is later re-created with a new auto-increment id.
                if memory_store is not None:
                    try:
                        n = _run_on_loop(
                            loop, memory_store.delete_for_context(context_id)
                        )
                        if n:
                            logger.info(
                                f"Context delete: removed {n} memor"
                                f"{'y' if n == 1 else 'ies'} for "
                                f"context {context_id}"
                            )
                    except Exception as e:
                        logger.error(f"Memory cascade-delete failed: {e}")
                _run_on_loop(loop, registry.delete(context_id))
            except Exception as e:
                logger.error(f"Context delete failed: {e}")
        return redirect(url_for("admin.context_list"))

    if oracle_store is not None:
        _register_oracle_routes(
            bp,
            registry=registry,
            oracle_store=oracle_store,
            loop=loop,
        )

    _register_memory_routes(
        bp,
        registry=registry,
        memory_store=memory_store,
        loop=loop,
        name_registry=name_registry,
    )


_ORACLE_KIND_OPTIONS = ("tarot", "iching", "market_open", "market_close", "freeform")
_ORACLE_SCHEDULE_OPTIONS = ("sunrise", "sunset", "clock")


def _oracle_kind_descriptions() -> dict:
    from ..contexts.oracles import KIND_DESCRIPTIONS
    return KIND_DESCRIPTIONS


def _oracle_schedule_descriptions() -> dict:
    from ..contexts.oracles import SCHEDULE_DESCRIPTIONS
    return SCHEDULE_DESCRIPTIONS


# Lazy-loaded so the admin blueprint doesn't import astral at module
# load time when oracles aren't enabled. The two dicts above are
# string→string and cheap to import; doing it eagerly is fine.
_ORACLE_KIND_DESCRIPTIONS = _oracle_kind_descriptions()
_ORACLE_SCHEDULE_DESCRIPTIONS = _oracle_schedule_descriptions()


def _oracle_to_view(o) -> dict:
    """Flatten a ContextOracle into a template-friendly dict that
    includes a human-readable schedule string."""
    return {
        "id": o.id,
        "context_id": o.context_id,
        "enabled": o.enabled,
        "kind": o.kind,
        "schedule_kind": o.schedule_kind,
        "offset_minutes": o.offset_minutes,
        "clock_time": o.clock_time or "",
        "timezone": o.timezone,
        "weekdays_only": o.weekdays_only,
        "prompt": o.prompt or "",
        "label": o.label or "",
        "description": o.describe(),
    }


def _form_to_oracle(form, *, context_id: int, oracle_id: Optional[int] = None):
    from ..contexts.oracles import ContextOracle

    def _int(name: str, default: int = 0) -> int:
        try:
            return int((form.get(name) or "").strip() or default)
        except (TypeError, ValueError):
            return default

    def _bool(name: str) -> bool:
        return (form.get(name) or "").lower() in ("1", "true", "yes", "on")

    return ContextOracle(
        id=oracle_id,
        context_id=context_id,
        enabled=_bool("enabled"),
        kind=(form.get("kind") or "tarot").strip(),
        schedule_kind=(form.get("schedule_kind") or "sunrise").strip(),
        offset_minutes=_int("offset_minutes", 0),
        clock_time=(form.get("clock_time") or "").strip() or None,
        timezone=(form.get("timezone") or "America/New_York").strip(),
        weekdays_only=_bool("weekdays_only"),
        prompt=(form.get("prompt") or "").strip() or None,
        label=(form.get("label") or "").strip(),
    )


def _register_oracle_routes(
    bp: Blueprint,
    *,
    registry: ContextRegistry,
    oracle_store,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Per-context oracle CRUD nested under /admin/contexts/<id>/oracles.

    Add/edit/delete actions all redirect back to the context-edit
    page so admin sees the updated list immediately. Validation
    errors flash; the form lives on the context-edit page itself.
    """

    def _ctx_or_404(context_id: int):
        policy = _run_on_loop(loop, registry.get(context_id))
        if not policy:
            abort(404)
        if policy.kind != "group":
            abort(400)
        return policy

    @bp.route("/contexts/<int:context_id>/oracles/new", methods=["POST"])
    @admin_required
    def oracle_new(context_id: int):
        if not verify_csrf():
            abort(400)
        _ctx_or_404(context_id)
        try:
            oracle = _form_to_oracle(request.form, context_id=context_id)
            _run_on_loop(loop, oracle_store.upsert(oracle))
            flash("Oracle added.", "ok")
        except ValueError as e:
            flash(f"Oracle add failed: {e}", "error")
        except Exception as e:
            logger.exception(f"Oracle new failed: {e}")
            flash(f"Oracle add failed: {e}", "error")
        return redirect(url_for("admin.context_edit", context_id=context_id))

    @bp.route(
        "/contexts/<int:context_id>/oracles/<int:oracle_id>/edit",
        methods=["POST"],
    )
    @admin_required
    def oracle_edit(context_id: int, oracle_id: int):
        if not verify_csrf():
            abort(400)
        _ctx_or_404(context_id)
        existing = _run_on_loop(loop, oracle_store.get(oracle_id))
        if not existing or existing.context_id != context_id:
            abort(404)
        try:
            oracle = _form_to_oracle(
                request.form, context_id=context_id, oracle_id=oracle_id,
            )
            _run_on_loop(loop, oracle_store.upsert(oracle))
            flash(f"Oracle #{oracle_id} updated.", "ok")
        except ValueError as e:
            flash(f"Oracle update failed: {e}", "error")
        except Exception as e:
            logger.exception(f"Oracle edit failed: {e}")
            flash(f"Oracle update failed: {e}", "error")
        return redirect(url_for("admin.context_edit", context_id=context_id))

    @bp.route(
        "/contexts/<int:context_id>/oracles/<int:oracle_id>/delete",
        methods=["POST"],
    )
    @admin_required
    def oracle_delete(context_id: int, oracle_id: int):
        if not verify_csrf():
            abort(400)
        _ctx_or_404(context_id)
        existing = _run_on_loop(loop, oracle_store.get(oracle_id))
        if not existing or existing.context_id != context_id:
            abort(404)
        try:
            _run_on_loop(loop, oracle_store.delete(oracle_id))
            flash(f"Oracle #{oracle_id} deleted.", "ok")
        except Exception as e:
            logger.exception(f"Oracle delete failed: {e}")
            flash(f"Oracle delete failed: {e}", "error")
        return redirect(url_for("admin.context_edit", context_id=context_id))


def _register_memory_routes(
    bp: Blueprint,
    *,
    registry: ContextRegistry,
    memory_store,
    loop: asyncio.AbstractEventLoop,
    name_registry=None,
) -> None:
    """Routes for browsing and editing per-context memory rows.

    Always mounted (even when `memory_store` is None) so the link from the
    context-edit page builds. When the store is missing, every route 404s
    cleanly instead of `url_for` raising BuildError mid-render.
    """
    from ..memory import KINDS, freetext_subject_key, is_user_hash

    def _store_or_404():
        if memory_store is None:
            abort(404)

    def _named_users() -> list[dict]:
        """Return [{user_hash, name}] sorted by name, or [] if no registry."""
        if name_registry is None:
            return []
        try:
            rows = _run_on_loop(loop, name_registry.list_all())
        except Exception as e:
            logger.debug(f"named user list failed: {e}")
            return []
        return [
            {"user_hash": r["user_hash"], "name": r["name"]}
            for r in rows
            if r.get("user_hash") and r.get("name")
        ]

    @bp.route("/contexts/<int:context_id>/memories", methods=["GET"])
    @admin_required
    def context_memories(context_id: int):
        _store_or_404()
        policy = _run_on_loop(loop, registry.get(context_id))
        if not policy:
            abort(404)
        rows = _run_on_loop(loop, memory_store.list_for_context(context_id))
        # Group by subject for readable rendering.
        grouped: dict = {}
        for r in rows:
            key = r["subject_key"]
            grouped.setdefault(key, {
                "subject_key": key,
                "subject_label": r["subject_label"] or key,
                "rows": [],
            })["rows"].append(r)
        # Refresh group display labels for user-hash subjects from the live
        # registry so renames show through immediately, mirroring the
        # preamble's behavior.
        users = _named_users()
        users_by_hash = {u["user_hash"]: u["name"] for u in users}
        for g in grouped.values():
            live = users_by_hash.get(g["subject_key"])
            if live:
                g["subject_label"] = live
        groups_all = sorted(
            grouped.values(),
            key=lambda g: (g["subject_label"] or "").lower(),
        )

        # Paginate at the subject level — each "page" shows N subjects with
        # all their memories, which keeps related rows together. Within a
        # single subject most chats have a small number of memories, so
        # there's no inner-pagination yet.
        try:
            per_page = max(1, min(int(request.args.get("per_page") or 10), 50))
        except (TypeError, ValueError):
            per_page = 10
        try:
            page = max(1, int(request.args.get("page") or 1))
        except (TypeError, ValueError):
            page = 1
        total_subjects = len(groups_all)
        total_pages = max(1, (total_subjects + per_page - 1) // per_page)
        if page > total_pages:
            page = total_pages
        start = (page - 1) * per_page
        groups = groups_all[start:start + per_page]

        return render_template(
            "context_memories.html",
            policy=policy,
            groups=groups,
            kinds=list(KINDS),
            total=len(rows),
            total_subjects=total_subjects,
            named_users=users,
            page=page,
            per_page=per_page,
            total_pages=total_pages,
        )

    @bp.route("/contexts/<int:context_id>/memories/new", methods=["POST"])
    @admin_required
    def memory_create(context_id: int):
        _store_or_404()
        if not verify_csrf():
            flash("Session expired, please reload the page.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        policy = _run_on_loop(loop, registry.get(context_id))
        if not policy:
            abort(404)
        # The picker fills `subject` directly with a user_hash or
        # __context__; the free-text fallback is `subject_custom`. Either
        # one wins (picker first), so the form has both visible without
        # client-side JS to swap them.
        subject_hint = (request.form.get("subject") or "").strip()
        if not subject_hint:
            subject_hint = (request.form.get("subject_custom") or "").strip()
        kind = (request.form.get("kind") or "").strip().lower()
        content = (request.form.get("content") or "").strip()
        if not subject_hint or kind not in KINDS or not content:
            flash(
                "Memory not saved — subject, kind, and content are all required.",
                "error",
            )
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        # If the picker fed in a known user_hash with no explicit label,
        # fill the label from NameRegistry so it renders nicely in the UI
        # without the admin having to retype the name.
        label_in = request.form.get("subject_label") or ""
        if not label_in.strip() and is_user_hash(subject_hint):
            for u in _named_users():
                if u["user_hash"] == subject_hint:
                    label_in = u["name"]
                    break
        subject_key, subject_label = _admin_resolve_subject(
            subject_hint, label_in,
            is_user_hash, freetext_subject_key,
        )
        if not subject_key:
            flash("Memory not saved — subject is empty after normalization.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        try:
            mem_id = _run_on_loop(
                loop,
                memory_store.add(
                    context_id=context_id,
                    subject_key=subject_key,
                    subject_label=subject_label,
                    kind=kind,
                    content=content,
                    source="admin",
                ),
            )
            if mem_id:
                flash(f"Saved memory #{mem_id} about {subject_label}.", "ok")
            else:
                flash("Memory not saved — input rejected.", "error")
        except Exception as e:
            logger.error(f"Memory create failed: {e}")
            flash(f"Memory create failed: {e}", "error")
        return redirect(
            _memories_redirect_url(context_id, request.form.get("page"))
        )

    @bp.route(
        "/contexts/<int:context_id>/memories/<int:memory_id>/update",
        methods=["POST"],
    )
    @admin_required
    def memory_update(context_id: int, memory_id: int):
        _store_or_404()
        if not verify_csrf():
            flash("Session expired, please reload the page.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        # Cross-context guard: refuse updates that don't belong to the URL's
        # context_id. Stops admins (or a stale form posting to a different
        # URL) from mutating a memory in a different chat by id.
        existing = _run_on_loop(loop, memory_store.get(memory_id))
        if not existing or existing.get("context_id") != context_id:
            flash(f"No memory #{memory_id} in this chat.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        content = (request.form.get("content") or "").strip()
        confidence_raw = request.form.get("confidence")
        subject_hint = (request.form.get("subject") or "").strip()
        if not subject_hint:
            subject_hint = (request.form.get("subject_custom") or "").strip()
        subject_label_in = (request.form.get("subject_label") or "").strip()
        # Auto-fill label from the registry if the picker handed us a hash
        # with no explicit label override.
        if not subject_label_in and is_user_hash(subject_hint):
            for u in _named_users():
                if u["user_hash"] == subject_hint:
                    subject_label_in = u["name"]
                    break
        confidence: Optional[float] = None
        if confidence_raw is not None and confidence_raw != "":
            try:
                confidence = float(confidence_raw)
            except ValueError:
                confidence = None
        new_key: Optional[str] = None
        new_label: Optional[str] = None
        if subject_hint:
            new_key, resolved_label = _admin_resolve_subject(
                subject_hint, subject_label_in,
                is_user_hash, freetext_subject_key,
            )
            if not new_key:
                flash("Subject ignored — empty after normalization.", "error")
                new_key = None
            else:
                new_label = resolved_label
        elif subject_label_in:
            new_label = subject_label_in
        if not (content or confidence is not None or new_key or new_label):
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        try:
            ok = _run_on_loop(
                loop,
                memory_store.update(
                    memory_id,
                    content=content or None,
                    confidence=confidence,
                    subject_key=new_key,
                    subject_label=new_label,
                ),
            )
            if ok:
                flash(f"Memory #{memory_id} updated.", "ok")
            else:
                flash(f"No changes saved for #{memory_id}.", "error")
        except Exception as e:
            logger.error(f"Memory update failed: {e}")
            flash(f"Memory update failed: {e}", "error")
        return redirect(
            _memories_redirect_url(context_id, request.form.get("page"))
        )

    @bp.route(
        "/contexts/<int:context_id>/memories/<int:memory_id>/delete",
        methods=["POST"],
    )
    @admin_required
    def memory_delete(context_id: int, memory_id: int):
        _store_or_404()
        if not verify_csrf():
            flash("Session expired, please reload the page.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        # Same cross-context guard as update — never delete a memory whose
        # context_id doesn't match the URL.
        existing = _run_on_loop(loop, memory_store.get(memory_id))
        if not existing or existing.get("context_id") != context_id:
            flash(f"No memory #{memory_id} in this chat.", "error")
            return redirect(
                url_for("admin.context_memories", context_id=context_id)
            )
        try:
            ok = _run_on_loop(loop, memory_store.delete(memory_id))
            flash(
                f"Memory #{memory_id} deleted." if ok
                else f"Memory #{memory_id} could not be deleted.",
                "ok" if ok else "error",
            )
        except Exception as e:
            logger.error(f"Memory delete failed: {e}")
            flash(f"Memory delete failed: {e}", "error")
        return redirect(
            _memories_redirect_url(context_id, request.form.get("page"))
        )


def _memories_redirect_url(context_id: int, page_raw) -> str:
    """Build the memories-page redirect URL, preserving `page` when valid.

    Forms include a hidden `page` field so save/delete bounces the admin
    back to the same page they were on instead of page 1. Defaults to
    no `page=` (page 1) when the field is missing or unparseable.
    """
    try:
        page = int(page_raw or 0)
    except (TypeError, ValueError):
        page = 0
    if page > 1:
        return url_for("admin.context_memories", context_id=context_id, page=page)
    return url_for("admin.context_memories", context_id=context_id)


def _admin_resolve_subject(
    subject_hint: str,
    subject_label_in,
    is_user_hash_fn,
    freetext_subject_key_fn,
) -> tuple[str, str]:
    """Translate the admin form's subject input into (key, label).

    Hex hash → user (label from form, falls back to short-hash); literal
    "this chat"/"context"/"the room" → SUBJECT_CONTEXT; else free-text
    slug. Names are NOT auto-resolved against NameRegistry here on
    purpose — admins should paste a hash for cross-context user identity
    or use free-text for chat-local subjects, which avoids name-collision
    footguns when two registered users share a first name.
    """
    s = (subject_hint or "").strip()
    sub_low = s.lower()
    if sub_low in ("this chat", "context", "the room"):
        from ..memory import SUBJECT_CONTEXT
        return SUBJECT_CONTEXT, "this chat"
    if sub_low in (
        "the bot", "yourself", "myself", "you", "self bot", "bot",
        "the bot itself", "__self__",
    ):
        from ..memory import SUBJECT_SELF
        return SUBJECT_SELF, "the bot"
    if is_user_hash_fn(s):
        label = (subject_label_in or "").strip() or f"...{s[:6]}"
        return s, label
    return freetext_subject_key_fn(s), s


# Virtual command names that aren't real BaseCommand classes but are
# nonetheless gated through ContextPolicy (e.g. easter eggs in the
# dispatcher). Surfaced in the admin UI so they can be allow/deny-listed.
_VIRTUAL_COMMANDS = ["corn"]


def _available_commands_and_servers(dispatcher, mcp_registry, loop):
    commands: list[str] = []
    if dispatcher is not None:
        # Unique canonical names, not aliases
        seen = set()
        for cmd in dispatcher.get_commands():
            if cmd.name not in seen:
                seen.add(cmd.name)
                commands.append(cmd.name)
        for vname in _VIRTUAL_COMMANDS:
            if vname not in seen:
                commands.append(vname)
                seen.add(vname)
        commands.sort()

    servers: list[str] = []
    if mcp_registry is not None:
        try:
            cfgs = _run_on_loop(loop, mcp_registry.list())
            servers = sorted(c.name for c in cfgs)
        except Exception as e:
            logger.error(f"MCP list for context UI failed: {e}")

    return commands, servers


def _form_to_values(form) -> dict:
    return {
        "kind": form.get("kind", "group"),
        "key": form.get("key", "").strip(),
        "label": form.get("label", "").strip(),
        "command_mode": form.get("command_mode", MODE_ALLOW_ALL),
        "commands": form.getlist("commands"),
        "mcp_mode": form.get("mcp_mode", MODE_ALLOW_ALL),
        "mcp_servers": form.getlist("mcp_servers"),
        "system_prompt": form.get("system_prompt", ""),
        "llm_intent": form.get("llm_intent", "") == "on",
        "reactor_enabled": form.get("reactor_enabled", "") == "on",
        "reactor_prompt": form.get("reactor_prompt", ""),
        "natural_response": form.get("natural_response", "") == "on",
        "deep_think_enabled": form.get("deep_think_enabled", "") == "on",
        "memory_writes_enabled": form.get("memory_writes_enabled", "") == "on",
        "reactor_memory_writes": form.get("reactor_memory_writes", "") == "on",
        "default_bot_id": form.get("default_bot_id", "").strip(),
    }


def _form_to_policy(form, existing_id: Optional[int], base: Optional[ContextPolicy] = None) -> ContextPolicy:
    kind = (form.get("kind") or "group").strip()
    if base is not None:
        kind = base.kind  # immutable
    key = (form.get("key") or "").strip()
    if base is not None:
        key = base.key  # immutable
    label = (form.get("label") or "").strip()
    command_mode = (form.get("command_mode") or MODE_ALLOW_ALL).strip()
    mcp_mode = (form.get("mcp_mode") or MODE_ALLOW_ALL).strip()
    commands = form.getlist("commands")
    mcp_servers = form.getlist("mcp_servers")
    system_prompt = (form.get("system_prompt") or "").strip() or None
    llm_intent = form.get("llm_intent", "") == "on"
    reactor_enabled = form.get("reactor_enabled", "") == "on"
    reactor_prompt = (form.get("reactor_prompt") or "").strip() or None
    natural_response = form.get("natural_response", "") == "on"
    deep_think_enabled = form.get("deep_think_enabled", "") == "on"
    memory_writes_enabled = form.get("memory_writes_enabled", "") == "on"
    reactor_memory_writes = form.get("reactor_memory_writes", "") == "on"
    raw_default_bot_id = (form.get("default_bot_id") or "").strip()
    default_bot_id: Optional[int]
    if raw_default_bot_id == "":
        # Empty submission == "fall back to registry's default-for-kind".
        default_bot_id = None
    else:
        try:
            default_bot_id = int(raw_default_bot_id)
        except ValueError:
            raise ValueError("default_bot_id must be an integer or blank") from None

    if kind not in ("group", "dm", "default"):
        raise ValueError("kind must be group, dm, or default")
    if not key:
        raise ValueError("key is required (group id, phone number, or default:*)")
    if command_mode not in MODES:
        raise ValueError(f"command_mode must be one of {MODES}")
    if mcp_mode not in MODES:
        raise ValueError(f"mcp_mode must be one of {MODES}")

    return ContextPolicy(
        id=existing_id,
        kind=kind,
        key=key,
        label=label,
        command_mode=command_mode,
        commands=commands,
        mcp_mode=mcp_mode,
        mcp_servers=mcp_servers,
        system_prompt=system_prompt,
        llm_intent=llm_intent,
        reactor_enabled=reactor_enabled,
        reactor_prompt=reactor_prompt,
        natural_response=natural_response,
        deep_think_enabled=deep_think_enabled,
        memory_writes_enabled=memory_writes_enabled,
        reactor_memory_writes=reactor_memory_writes,
        default_bot_id=default_bot_id,
    )


def _register_users_routes(
    bp: Blueprint,
    *,
    registry,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Routes for managing display names attached to senders."""

    @bp.route("/users", methods=["GET"])
    @admin_required
    def users_page():
        named = _run_on_loop(loop, registry.list_all())
        seen = _run_on_loop(loop, registry.list_seen(limit=100))
        # Show seen-but-unnamed first (the actionable list); named entries
        # below for editing/deletion.
        unnamed = [s for s in seen if not s.get("name")]
        return render_template(
            "users.html", named=named, unnamed=unnamed, seen=seen,
        )

    @bp.route("/users/save", methods=["POST"])
    @admin_required
    def users_save():
        if not verify_csrf():
            return redirect(url_for("admin.users_page"))
        user_hash = (request.form.get("user_hash") or "").strip()
        name = (request.form.get("name") or "").strip()
        if user_hash:
            try:
                _run_on_loop(loop, registry.set_name(name, user_hash=user_hash))
            except Exception as e:
                logger.error(f"Save user name failed: {e}")
        return redirect(url_for("admin.users_page"))

    @bp.route("/users/delete", methods=["POST"])
    @admin_required
    def users_delete():
        if not verify_csrf():
            return redirect(url_for("admin.users_page"))
        user_hash = (request.form.get("user_hash") or "").strip()
        if user_hash:
            try:
                _run_on_loop(loop, registry.delete(user_hash=user_hash))
            except Exception as e:
                logger.error(f"Delete user name failed: {e}")
        return redirect(url_for("admin.users_page"))


def _register_predictions_routes(
    bp: Blueprint,
    *,
    prediction_store,
    prediction_resolver,
    context_registry: Optional[ContextRegistry],
    name_registry,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Prediction-game admin view: aggregate stats, per-context leaderboards,
    upcoming deadlines, and a flat recent feed. Read-only — manual resolution
    still happens via !resolve in chat.
    """
    import time as _time

    @bp.route("/predictions", methods=["GET"])
    @admin_required
    def predictions_page():
        store = prediction_store
        if store is None:
            return render_template("predictions.html", available=False,
                                   counts={}, leaderboards=[],
                                   upcoming=[], recent=[])

        # Resolve context_key → friendly label once. The registry stores
        # `kind`+`key` separately (no prefix); predictions use
        # `f"{kind}:{key}"`. Stitch them so the template can show
        # human-readable chat names instead of opaque hashes.
        ctx_labels: dict[str, str] = {}
        if context_registry is not None:
            try:
                policies = _run_on_loop(loop, context_registry.list())
                for p in policies:
                    if p.kind in ("group", "dm") and p.key:
                        ctx_labels[f"{p.kind}:{p.key}"] = p.label or "(unlabeled)"
                    elif p.kind == "default":
                        ctx_labels[f"default:{p.key or ''}"] = (
                            p.label or "(default)"
                        )
            except Exception as e:
                logger.error(f"Predictions: context label resolution failed: {e}")

        def label_for_ctx(ck: str) -> str:
            return ctx_labels.get(ck) or ck

        def label_for_user(user_hash: str, fallback: Optional[str]) -> str:
            if name_registry is not None:
                cached = name_registry._cache.get(user_hash)
                if cached:
                    return cached
                if user_hash == _bot_user_hash():
                    return name_registry.bot_name
            return fallback or "(unknown)"

        try:
            counts = _run_on_loop(loop, store.total_counts())
        except Exception as e:
            logger.error(f"Predictions: counts failed: {e}")
            counts = {}

        try:
            ctx_summaries = _run_on_loop(loop, store.contexts_with_predictions())
        except Exception as e:
            logger.error(f"Predictions: context list failed: {e}")
            ctx_summaries = []

        # Per-context leaderboards. Skip contexts that have no resolved
        # predictions yet — a leaderboard with everyone at 0/0 is noise.
        leaderboards = []
        for s in ctx_summaries:
            ck = s["context_key"]
            try:
                rows = _run_on_loop(loop, store.leaderboard(ck, limit=20))
            except Exception as e:
                logger.error(f"Predictions: leaderboard for {ck} failed: {e}")
                continue
            scoring = [r for r in rows if (r["right"] + r["wrong"]) > 0]
            pending = sum(r["pending"] for r in rows)
            entries = []
            for r in scoring:
                judged = r["right"] + r["wrong"]
                entries.append({
                    "label": label_for_user(r["user_hash"], r["label"]),
                    "right": r["right"],
                    "wrong": r["wrong"],
                    "judged": judged,
                    "accuracy": r["accuracy"],
                    "pending": r["pending"],
                })
            if entries or pending:
                leaderboards.append({
                    "context_label": label_for_ctx(ck),
                    "context_key": ck,
                    "total": s["total"],
                    "pending_total": pending,
                    "entries": entries,
                })

        try:
            upcoming_raw = _run_on_loop(loop, store.upcoming_all(limit=20))
        except Exception as e:
            logger.error(f"Predictions: upcoming failed: {e}")
            upcoming_raw = []

        try:
            recent_raw = _run_on_loop(loop, store.recent_all(limit=30))
        except Exception as e:
            logger.error(f"Predictions: recent failed: {e}")
            recent_raw = []

        def render_pred(p) -> dict:
            now = _time.time()
            delta = p.deadline_utc - now
            if p.status == "pending":
                if delta <= 0:
                    deadline_rel = "due"
                elif delta < 3600:
                    deadline_rel = "<1h"
                elif delta < 86400:
                    deadline_rel = f"{int(delta // 3600)}h"
                else:
                    deadline_rel = f"{int(delta // 86400)}d"
            else:
                deadline_rel = ""
            return {
                "id": p.id,
                "user_label": label_for_user(p.user_hash, p.user_label),
                "claim": p.claim,
                "context_label": label_for_ctx(p.context_key),
                "deadline_iso": _time.strftime(
                    "%Y-%m-%d", _time.gmtime(p.deadline_utc)
                ),
                "deadline_rel": deadline_rel,
                "status": p.status,
                "verdict": p.verdict or "",
                "resolution_note": p.resolution_note or "",
                "is_structured": p.is_structured,
                "ticker": p.ticker,
                "threshold": p.threshold,
                "direction": p.direction,
            }

        upcoming = [render_pred(p) for p in upcoming_raw]
        recent = [render_pred(p) for p in recent_raw]

        return render_template(
            "predictions.html",
            available=True,
            counts=counts,
            leaderboards=leaderboards,
            upcoming=upcoming,
            recent=recent,
        )

    def _bot_user_hash() -> str:
        # Lazy import — avoids pulling group_log at module load time.
        from ..database import hash_phone
        from ..group_log import BOT_SENDER
        return hash_phone(BOT_SENDER)

    @bp.route("/predictions/<int:pred_id>/resolve", methods=["POST"])
    @admin_required
    def predictions_resolve_now(pred_id: int):
        """Trigger Sigil to settle one prediction immediately (no waiting
        for the next sweep). Falls back to structured-quote when the
        prediction is stock-shape; otherwise hits the LLM tool loop.
        """
        if not verify_csrf():
            abort(400)
        if prediction_store is None or prediction_resolver is None:
            flash("Prediction resolver not configured.", "error")
            return redirect(url_for("admin.predictions_page"))

        try:
            pred = _run_on_loop(loop, prediction_store.get(pred_id))
            if pred is None:
                flash(f"No prediction #{pred_id}.", "error")
                return redirect(url_for("admin.predictions_page"))
            if pred.status != "pending":
                flash(
                    f"#{pred_id} is already {pred.status}. Use Override "
                    f"to change the verdict.",
                    "error",
                )
                return redirect(url_for("admin.predictions_page"))

            # Pre-fetch quote for structured predictions so the resolver
            # has the same shape it sees in the cron sweep.
            quotes = {}
            if pred.is_structured and pred.ticker:
                price = _run_on_loop(
                    loop, prediction_resolver._safe_quote(pred.ticker)
                )
                if price is not None:
                    quotes[pred.ticker] = price

            outcome = _run_on_loop(
                loop, prediction_resolver._resolve_one(pred, quotes),
                timeout=120.0,
            )
            if outcome is None:
                flash(
                    f"Sigil couldn't settle #{pred_id} right now "
                    f"(quote miss or LLM unavailable). Will retry on the "
                    f"next sweep.",
                    "error",
                )
            else:
                verdict, note = outcome
                # Mirror the cron behavior: post the verdict back to the
                # originating chat so users see it in real time, not just
                # in the dashboard.
                try:
                    _run_on_loop(
                        loop,
                        prediction_resolver._post_resolution(pred, verdict, note),
                    )
                except Exception as e:
                    logger.warning(f"Resolve-now post failed for #{pred_id}: {e}")
                flash(
                    f"#{pred_id} resolved {verdict.upper()}"
                    f"{' — ' + note if note else ''}.",
                    "ok",
                )
        except Exception as e:
            logger.exception(f"Resolve-now failed for #{pred_id}: {e}")
            flash(f"Failed to resolve #{pred_id}: {e}", "error")
        return redirect(url_for("admin.predictions_page"))

    @bp.route("/predictions/<int:pred_id>/override", methods=["POST"])
    @admin_required
    def predictions_override(pred_id: int):
        """Set verdict + note manually. Works on any status — the admin's
        whole point of being here is to fix wrong auto-resolutions, so
        unconditional update is the right contract."""
        if not verify_csrf():
            abort(400)
        if prediction_store is None:
            flash("Prediction store not configured.", "error")
            return redirect(url_for("admin.predictions_page"))

        verdict = (request.form.get("verdict") or "").strip().lower()
        note = (request.form.get("note") or "").strip() or None
        if verdict not in ("right", "wrong", "unclear"):
            flash("Verdict must be right, wrong, or unclear.", "error")
            return redirect(url_for("admin.predictions_page"))

        try:
            ok = _run_on_loop(
                loop,
                prediction_store.force_set_verdict(
                    pred_id, verdict=verdict, note=note,
                    resolver_user_hash=None,
                ),
            )
            if ok:
                flash(f"#{pred_id} overridden to {verdict.upper()}.", "ok")
            else:
                flash(f"No prediction #{pred_id}.", "error")
        except Exception as e:
            logger.exception(f"Override failed for #{pred_id}: {e}")
            flash(f"Failed to override #{pred_id}: {e}", "error")
        return redirect(url_for("admin.predictions_page"))

    @bp.route("/predictions/<int:pred_id>/revert", methods=["POST"])
    @admin_required
    def predictions_revert(pred_id: int):
        """Send a resolved/expired prediction back to pending. Useful when
        the admin wants the auto-resolver to take another pass — e.g.
        the LLM was wrong but Sigil now has more recent tools/data."""
        if not verify_csrf():
            abort(400)
        if prediction_store is None:
            flash("Prediction store not configured.", "error")
            return redirect(url_for("admin.predictions_page"))

        try:
            ok = _run_on_loop(loop, prediction_store.revert_to_pending(pred_id))
            if ok:
                flash(f"#{pred_id} reverted to pending.", "ok")
            else:
                flash(f"No prediction #{pred_id}.", "error")
        except Exception as e:
            logger.exception(f"Revert failed for #{pred_id}: {e}")
            flash(f"Failed to revert #{pred_id}: {e}", "error")
        return redirect(url_for("admin.predictions_page"))


def _register_live_routes(
    bp: Blueprint,
    *,
    name_registry,
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Live event stream + viewer page.

    The viewer is a thin HTML page that opens an SSE connection to the
    /admin/live/stream endpoint, where AdminEventBus replays recent
    history then streams new events as they happen. Sender resolution
    via NameRegistry happens server-side so the browser never sees raw
    phone tails it can't interpret.
    """

    @bp.route("/live", methods=["GET"])
    @admin_required
    def live_page():
        return render_template("live.html")

    @bp.route("/live/stream", methods=["GET"])
    @admin_required
    def live_stream():
        bus = get_bus()
        q = bus.subscribe()
        # Build a synchronous label resolver that warms the registry once.
        def label(tail: str | None, _phone=None) -> str:
            if name_registry is None or not tail:
                return f"...{tail}" if tail else "..."
            try:
                return name_registry.display_name_sync(tail=tail)
            except Exception:
                return f"...{tail}"

        def gen():
            # Initial comment forces some HTTP/SSE clients to flush headers.
            yield ": connected\n\n"
            try:
                while True:
                    try:
                        ev = q.get(timeout=15.0)
                    except Exception:
                        # Heartbeat — keeps the connection alive through
                        # proxies that drop idle streams (15s undershoots
                        # most defaults).
                        yield ": heartbeat\n\n"
                        continue
                    enriched = dict(ev)
                    if "sender_tail" in enriched:
                        enriched["sender_label"] = label(enriched.get("sender_tail"))
                    if "recipient_tail" in enriched:
                        enriched["recipient_label"] = label(enriched.get("recipient_tail"))
                    yield f"data: {json.dumps(enriched, default=str)}\n\n"
            except GeneratorExit:
                pass
            finally:
                bus.unsubscribe(q)

        # Return as an SSE stream. Disable Flask's response caching and
        # any intermediate buffering so events ship immediately.
        return Response(
            gen(),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
                "Connection": "keep-alive",
            },
        )


def _register_portfolio_routes(
    bp: Blueprint,
    *,
    portfolio_store,
    portfolio_executor,
    context_registry: Optional[ContextRegistry],
    loop: asyncio.AbstractEventLoop,
) -> None:
    """Paper-portfolio admin view: per-context status, recent trades,
    recent tips, plus a destructive 'reset' button.

    Read-mostly: trade execution still happens through Sigil's tools or
    the cron worker, not from this UI.
    """
    import time as _time

    @bp.route("/portfolios", methods=["GET"])
    @admin_required
    def portfolios_page():
        if portfolio_store is None or portfolio_executor is None:
            return render_template(
                "portfolios.html", available=False, portfolios=[],
            )

        ctx_labels: dict[str, str] = {}
        if context_registry is not None:
            try:
                policies = _run_on_loop(loop, context_registry.list())
                for p in policies:
                    if p.kind in ("group", "dm") and p.key:
                        ctx_labels[f"{p.kind}:{p.key}"] = p.label or "(unlabeled)"
            except Exception as e:
                logger.error(f"Portfolios: context labels failed: {e}")

        try:
            keys = _run_on_loop(loop, portfolio_store.list_portfolio_keys())
        except Exception as e:
            logger.error(f"Portfolios: list keys failed: {e}")
            keys = []

        rendered = []
        for ck in keys:
            try:
                snap = _run_on_loop(loop, portfolio_executor.status(ck))
                trades = _run_on_loop(
                    loop, portfolio_store.list_trades(ck, limit=15),
                )
                tips = _run_on_loop(
                    loop, portfolio_store.list_tips(ck, limit=15),
                )
                # Recent orders across all statuses (not just pending)
                # so admin can see fill / cancel / expired history. The
                # `pending_orders` field on the snap is already filtered
                # to pending; this list complements it for the
                # activity-log section of the template.
                from src.paper_portfolio import VALID_ORDER_STATUSES
                recent_orders = _run_on_loop(
                    loop,
                    portfolio_store.list_orders_for_context(
                        ck, statuses=set(VALID_ORDER_STATUSES), limit=20,
                    ),
                )
            except Exception as e:
                logger.error(f"Portfolios: snapshot for {ck} failed: {e}")
                continue

            row = dict(snap)
            row["context_key"] = ck
            row["label"] = (
                snap.get("label")
                or ctx_labels.get(ck)
                or ck
            )
            row["trades"] = [
                {
                    "ts_str": _time.strftime(
                        "%Y-%m-%d %H:%M", _time.localtime(t.ts),
                    ),
                    "side": t.side,
                    "qty": t.qty,
                    "ticker": t.ticker,
                    "price": t.price,
                    "source": t.source,
                    "realized_pnl": t.realized_pnl,
                    "reason": t.reason,
                }
                for t in trades
            ]
            row["tips"] = [
                {
                    "ts_str": _time.strftime(
                        "%Y-%m-%d %H:%M", _time.localtime(tip.ts),
                    ),
                    "tipper_label": tip.tipper_label,
                    "tipper_user_hash": tip.tipper_user_hash,
                    "amount": tip.amount,
                    "note": tip.note,
                }
                for tip in tips
            ]
            # Order activity log — show every status (not just pending)
            # so admins can audit fills, cancels, expiries, and
            # failures. Sorted newest-first to match the trade log.
            row["recent_orders"] = [
                {
                    "id": o.id,
                    "ticker": o.ticker,
                    "side": o.side,
                    "kind": o.kind,
                    "trigger_price": o.trigger_price,
                    "trigger_direction": (
                        "above"
                        if (
                            (o.side == "buy" and o.kind == "stop")
                            or (o.side == "sell" and o.kind == "limit")
                        ) else "below"
                    ),
                    "qty": o.qty,
                    "dollars": o.dollars,
                    "close_position": bool(o.close_position),
                    "reason": o.reason,
                    "status": o.status,
                    "created_str": _time.strftime(
                        "%Y-%m-%d %H:%M", _time.localtime(o.created_at),
                    ),
                    "filled_str": (
                        _time.strftime(
                            "%Y-%m-%d %H:%M", _time.localtime(o.filled_at),
                        ) if o.filled_at else None
                    ),
                    "fill_price": o.fill_price,
                    "fill_qty": o.fill_qty,
                    "fill_note": o.fill_note,
                }
                for o in recent_orders
            ]
            rendered.append(row)

        return render_template(
            "portfolios.html",
            available=True,
            portfolios=rendered,
        )

    @bp.route("/portfolios/<path:context_key>/reset", methods=["POST"])
    @admin_required
    def portfolios_reset(context_key: str):
        if not verify_csrf():
            abort(400)
        if portfolio_store is None:
            flash("Portfolio store not configured.", "error")
            return redirect(url_for("admin.portfolios_page"))
        try:
            ok = _run_on_loop(loop, portfolio_store.reset(context_key))
        except Exception as e:
            logger.exception(f"Portfolios: reset {context_key} failed: {e}")
            flash(f"Reset failed: {type(e).__name__}", "error")
            return redirect(url_for("admin.portfolios_page"))
        if ok:
            flash(f"Reset {context_key} to $1000 seed.", "ok")
        else:
            flash(f"No portfolio for {context_key}.", "error")
        return redirect(url_for("admin.portfolios_page"))

    @bp.route(
        "/portfolios/<path:context_key>/orders/<int:order_id>/cancel",
        methods=["POST"],
    )
    @admin_required
    def portfolios_cancel_order(context_key: str, order_id: int):
        """Admin override-cancel of a pending order. Scoped by
        context_key (matching the LLM-facing cancel) so the URL has to
        be valid both ways."""
        if not verify_csrf():
            abort(400)
        if portfolio_store is None:
            flash("Portfolio store not configured.", "error")
            return redirect(url_for("admin.portfolios_page"))
        try:
            status = _run_on_loop(
                loop,
                portfolio_store.cancel_order(
                    order_id, context_key=context_key,
                ),
            )
        except Exception as e:
            logger.exception(
                f"Portfolios: cancel_order {order_id} failed: {e}"
            )
            flash(f"Cancel failed: {type(e).__name__}", "error")
            return redirect(url_for("admin.portfolios_page"))
        if status == "ok":
            flash(f"Cancelled order #{order_id}.", "ok")
        elif status == "not_found":
            flash(f"Order #{order_id} doesn't exist.", "error")
        elif status == "wrong_context":
            flash(
                f"Order #{order_id} doesn't belong to this portfolio.",
                "error",
            )
        elif status == "not_pending":
            flash(
                f"Order #{order_id} is already filled / cancelled / "
                f"expired — nothing to cancel.",
                "error",
            )
        else:
            flash(f"Cancel returned unexpected status: {status!r}", "error")
        return redirect(url_for("admin.portfolios_page"))
