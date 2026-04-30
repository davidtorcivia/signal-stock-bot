"""
Flask webhook server for receiving Signal messages.

The async handler runs on a persistent event loop owned by the caller.
Flask request threads submit coroutines via run_coroutine_threadsafe so
aiohttp sessions and other loop-bound resources survive across requests.
"""

import asyncio
import hmac
import logging
from typing import Optional

from flask import Flask, request, jsonify

from .signal.handler import SignalHandler

logger = logging.getLogger(__name__)


def create_app(
    signal_handler: SignalHandler,
    loop: asyncio.AbstractEventLoop,
    webhook_secret: str = "",
    handler_timeout: float = 60.0,
    *,
    flask_secret_key: Optional[str] = None,
    session_cookie_secure: bool = False,
    admin_password_hash: str = "",
    settings_store=None,
    admin_phone: str = "",
    provider_manager=None,
    mcp_registry=None,
    mcp_manager=None,
    context_registry=None,
    dispatcher=None,
    name_registry=None,
    deep_think_client=None,
    memory_store=None,
    prediction_store=None,
    prediction_resolver=None,
    oracle_store=None,
    portfolio_store=None,
    portfolio_executor=None,
) -> Flask:
    """
    Create Flask application with webhook endpoint and optional admin UI.

    If `admin_password_hash`, `flask_secret_key`, `settings_store`, and
    `admin_phone` are all provided, the /admin/* routes are mounted.
    """
    app = Flask(__name__)
    app.signal_handler = signal_handler

    if flask_secret_key:
        app.secret_key = flask_secret_key
        app.config.update(
            SESSION_COOKIE_HTTPONLY=True,
            SESSION_COOKIE_SAMESITE="Lax",
            SESSION_COOKIE_SECURE=session_cookie_secure,
        )

    @app.route("/webhook", methods=["POST"])
    def webhook():
        if webhook_secret:
            provided = request.headers.get("X-Webhook-Secret", "")
            if not hmac.compare_digest(provided, webhook_secret):
                logger.warning("Webhook rejected: invalid or missing secret")
                return jsonify({"status": "error", "message": "unauthorized"}), 401

        data = request.get_json(silent=True)
        if not data:
            logger.warning("Received empty or invalid webhook payload")
            return jsonify({"status": "error", "message": "Empty payload"}), 400

        logger.debug(f"Received webhook: {data}")

        future = asyncio.run_coroutine_threadsafe(
            signal_handler.handle_webhook(data), loop
        )
        try:
            future.result(timeout=handler_timeout)
        except Exception as e:
            logger.exception(f"Error handling webhook: {e}")
            return jsonify({"status": "error", "message": str(e)}), 500

        return jsonify({"status": "ok"})

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy"})

    @app.route("/", methods=["GET"])
    def index():
        return jsonify({"status": "running"})

    # Mount /admin/* if fully configured. Otherwise, admin routes are off.
    admin_enabled = bool(
        admin_password_hash
        and flask_secret_key
        and settings_store is not None
        and admin_phone
    )
    if admin_enabled:
        from .admin import create_admin_blueprint

        create_admin_blueprint(
            app=app,
            password_hash=admin_password_hash.encode(),
            settings_store=settings_store,
            signal_handler=signal_handler,
            loop=loop,
            admin_phone=admin_phone,
            provider_manager=provider_manager,
            mcp_registry=mcp_registry,
            mcp_manager=mcp_manager,
            context_registry=context_registry,
            dispatcher=dispatcher,
            name_registry=name_registry,
            deep_think_client=deep_think_client,
            memory_store=memory_store,
            prediction_store=prediction_store,
            prediction_resolver=prediction_resolver,
            oracle_store=oracle_store,
            portfolio_store=portfolio_store,
            portfolio_executor=portfolio_executor,
        )
        logger.info("Admin UI mounted at /admin")
    else:
        logger.info("Admin UI disabled (ADMIN_PASSWORD_HASH or FLASK_SECRET_KEY not set)")

    return app
