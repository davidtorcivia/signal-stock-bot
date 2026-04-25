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
    )
    _register_llm_routes(bp, settings_store=settings_store)

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
        )

    if name_registry is not None:
        _register_users_routes(bp, registry=name_registry, loop=loop)

    _register_live_routes(bp, name_registry=name_registry, loop=loop)

    # Make csrf_token available to every rendered template in this blueprint.
    @bp.context_processor
    def inject_csrf():
        return {"csrf_token": csrf_token}

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
) -> None:
    @bp.route("/", methods=["GET"])
    @admin_required
    def dashboard():
        metrics = get_metrics().get_all_stats()
        provider_status = provider_manager.get_status() if provider_manager else {}

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

        return render_template(
            "dashboard.html",
            metrics=metrics,
            provider_status=provider_status,
            mcp_view=mcp_view,
            context_view=context_view,
            storage_view=storage_view,
            reactor_view=reactor_view,
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

        return render_template(
            "llm.html",
            values=values,
            api_key_set=api_key_set,
            saved=saved,
            error=error,
        )


def _apply_llm_form(store: SettingsStore, form) -> None:
    """Persist LLM form values with schema-aware coercion."""
    import json

    bool_keys = {"llm_enabled", "reactor_enabled"}
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
    }
    float_keys = {"llm_temperature", "reactor_temperature"}

    for key in LLM_KEYS:
        if key == "llm_api_key":
            # Only overwrite when the user typed a new value; empty = keep existing
            submitted = form.get("llm_api_key", "").strip()
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
) -> None:
    @bp.route("/contexts", methods=["GET"])
    @admin_required
    def context_list():
        rows = _run_on_loop(loop, registry.list())
        return render_template("context_list.html", contexts=rows)

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
        }
        if request.method == "POST":
            values = _form_to_values(request.form)
            values["kind"] = policy.kind  # kind is immutable after creation
            values["key"] = policy.key

        all_commands, all_servers = _available_commands_and_servers(dispatcher, mcp_registry, loop)
        return render_template(
            "context_edit.html",
            is_new=False,
            policy=policy,
            values=values,
            modes=MODES,
            all_commands=all_commands,
            all_servers=all_servers,
            error=error,
        )

    @bp.route("/contexts/<int:context_id>/delete", methods=["POST"])
    @admin_required
    def context_delete(context_id: int):
        if verify_csrf():
            try:
                _run_on_loop(loop, registry.delete(context_id))
            except Exception as e:
                logger.error(f"Context delete failed: {e}")
        return redirect(url_for("admin.context_list"))


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
