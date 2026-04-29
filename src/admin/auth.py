"""
Session / password / Signal-delivered 2FA plumbing for the admin UI.

Flow:
  POST /admin/login  (password)
    → validates bcrypt hash
    → generates 6-digit code, stores in session, DMs it to admin phone
  POST /admin/2fa    (code)
    → verifies code, sets session["authenticated"] = True
  POST /admin/logout → clears session
"""

import asyncio
import hashlib
import hmac
import logging
import secrets
import time
from collections import defaultdict
from functools import wraps
from typing import Callable

import bcrypt
from flask import (
    Blueprint,
    current_app,
    redirect,
    render_template,
    request,
    session,
    url_for,
)

logger = logging.getLogger(__name__)


LOGIN_RATE_LIMIT = 5       # attempts
LOGIN_RATE_WINDOW = 900    # 15 min
TWOFA_MAX_ATTEMPTS = 5
TWOFA_TTL_SECONDS = 300    # 5 min
SESSION_LIFETIME_SECONDS = 12 * 3600


class LoginRateLimiter:
    """Per-IP login attempt counter."""

    def __init__(self, limit: int = LOGIN_RATE_LIMIT, window: int = LOGIN_RATE_WINDOW):
        self.limit = limit
        self.window = window
        self._attempts: dict[str, list[float]] = defaultdict(list)

    def check(self, ip: str) -> tuple[bool, int]:
        now = time.time()
        cutoff = now - self.window
        self._attempts[ip] = [t for t in self._attempts[ip] if t > cutoff]
        if len(self._attempts[ip]) >= self.limit:
            oldest = self._attempts[ip][0]
            return False, int(oldest + self.window - now) + 1
        return True, 0

    def record(self, ip: str) -> None:
        self._attempts[ip].append(time.time())

    def reset(self, ip: str) -> None:
        self._attempts.pop(ip, None)


def _generate_code() -> str:
    return f"{secrets.randbelow(1_000_000):06d}"


def _hash_code(code: str) -> str:
    """Hash the 2FA code before stashing in the session.

    Flask sessions are signed (HMAC-protected) but not encrypted — the
    payload is base64-decodable by anyone holding the cookie. If a session
    cookie leaks (network capture on plain-HTTP LAN access, browser
    extension, log scraping), an attacker who reads the cookie can extract
    the still-active 6-digit code and bypass 2FA. Hashing keeps the cookie
    from carrying the secret; only the verifier (which has the user's
    submitted code) can recompute the hash and compare.
    """
    return hashlib.sha256(code.encode()).hexdigest()


def _client_ip() -> str:
    # request.remote_addr is proxy-aware when ProxyFix is used.
    return request.remote_addr or "unknown"


def csrf_token() -> str:
    """Return the per-session CSRF token, creating it if missing."""
    token = session.get("csrf_token")
    if not token:
        token = secrets.token_urlsafe(32)
        session["csrf_token"] = token
    return token


def verify_csrf() -> bool:
    sent = request.form.get("csrf_token", "")
    expected = session.get("csrf_token", "")
    return bool(sent) and bool(expected) and secrets.compare_digest(sent, expected)


def admin_required(fn: Callable) -> Callable:
    """Redirect unauthenticated users to the login page."""

    @wraps(fn)
    def wrapped(*args, **kwargs):
        if not session.get("authenticated"):
            return redirect(url_for("admin.login"))
        return fn(*args, **kwargs)

    return wrapped


def register_auth_routes(
    bp: Blueprint,
    *,
    password_hash: bytes,
    signal_handler,
    loop: asyncio.AbstractEventLoop,
    admin_phone: str,
) -> LoginRateLimiter:
    rate_limiter = LoginRateLimiter()

    @bp.route("/login", methods=["GET", "POST"])
    def login():
        if session.get("authenticated"):
            return redirect(url_for("admin.dashboard"))

        error = None
        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload and try again."
            else:
                ip = _client_ip()
                allowed, retry_after = rate_limiter.check(ip)
                if not allowed:
                    error = f"Too many attempts. Try again in {retry_after}s."
                else:
                    pw = request.form.get("password", "").encode()
                    if pw and bcrypt.checkpw(pw, password_hash):
                        code = _generate_code()
                        session["pending_2fa_hash"] = _hash_code(code)
                        session["pending_2fa_expires"] = time.time() + TWOFA_TTL_SECONDS
                        session["pending_2fa_attempts"] = 0
                        session.pop("authenticated", None)
                        _dispatch_2fa_code(code, signal_handler, loop, admin_phone)
                        return redirect(url_for("admin.twofa"))
                    rate_limiter.record(ip)
                    logger.warning(f"Admin login failed from {ip}")
                    error = "Incorrect password."

        return render_template("login.html", error=error, csrf_token=csrf_token())

    @bp.route("/2fa", methods=["GET", "POST"])
    def twofa():
        if session.get("authenticated"):
            return redirect(url_for("admin.dashboard"))
        if not session.get("pending_2fa_hash"):
            return redirect(url_for("admin.login"))

        error = None
        if request.method == "POST":
            if not verify_csrf():
                error = "Session expired, please reload and try again."
            elif time.time() > session.get("pending_2fa_expires", 0):
                _clear_pending()
                return redirect(url_for("admin.login"))
            elif session.get("pending_2fa_attempts", 0) >= TWOFA_MAX_ATTEMPTS:
                # Charge the IP rate limiter so an attacker who has the
                # password can't just keep re-logging in to get a fresh
                # set of 2FA attempts. Every burnt 2FA cycle costs an IP
                # login slot, capping brute-force throughput at
                # LOGIN_RATE_LIMIT bursts per LOGIN_RATE_WINDOW.
                rate_limiter.record(_client_ip())
                _clear_pending()
                error = "Too many attempts. Start over."
            else:
                submitted = request.form.get("code", "").strip()
                expected_hash = session.get("pending_2fa_hash", "")
                submitted_hash = _hash_code(submitted) if submitted else ""
                if submitted and expected_hash and hmac.compare_digest(
                    submitted_hash, expected_hash
                ):
                    _clear_pending()
                    session["authenticated"] = True
                    session["login_time"] = time.time()
                    rate_limiter.reset(_client_ip())
                    logger.info(f"Admin session started from {_client_ip()}")
                    return redirect(url_for("admin.dashboard"))
                session["pending_2fa_attempts"] = session.get("pending_2fa_attempts", 0) + 1
                error = "Incorrect code."

        return render_template("twofa.html", error=error, csrf_token=csrf_token())

    @bp.route("/logout", methods=["POST"])
    def logout():
        if verify_csrf():
            session.clear()
        return redirect(url_for("admin.login"))

    return rate_limiter


def _clear_pending() -> None:
    for key in (
        "pending_2fa_code",       # legacy key — clean up old sessions
        "pending_2fa_hash",
        "pending_2fa_expires",
        "pending_2fa_attempts",
    ):
        session.pop(key, None)


def _dispatch_2fa_code(
    code: str,
    signal_handler,
    loop: asyncio.AbstractEventLoop,
    admin_phone: str,
) -> None:
    """Fire-and-wait DM of the 2FA code via the Signal handler."""
    msg = f"Admin login code: {code}\nValid for 5 minutes."
    try:
        future = asyncio.run_coroutine_threadsafe(
            signal_handler.send_message(recipient=admin_phone, message=msg),
            loop,
        )
        future.result(timeout=15)
    except Exception as e:
        logger.error(f"Failed to DM 2FA code to admin: {e}")
