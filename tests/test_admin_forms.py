"""Admin forms round-trip: submitting a page exactly as rendered must not
change any stored setting. Catches fields dropped, renamed or mis-encoded
by template rewrites (a missing text field is saved as "", a toggle that
posts the wrong value flips a flag)."""

from html.parser import HTMLParser

import pytest
from flask import Flask
from werkzeug.datastructures import MultiDict

from src.admin.blueprint import create_admin_blueprint
from src.settings_store import SettingsStore


class _FormFields(HTMLParser):
    """Collect what a browser would submit for the page's main form."""

    def __init__(self):
        super().__init__()
        self.pairs: list[tuple[str, str]] = []
        self._in_form = False
        self._done = False
        self._select = None
        self._textarea = None

    def handle_starttag(self, tag, attrs):
        a = dict(attrs)
        # The page's own form posts back to itself (no action); the sidebar's
        # logout form has one.
        if (tag == "form" and not self._done and (a.get("method") or "").lower() == "post"
                and not a.get("action")):
            self._in_form = True
        if not self._in_form:
            return
        name = a.get("name")
        if tag == "input" and name:
            kind = (a.get("type") or "text").lower()
            if kind in ("checkbox", "radio"):
                if "checked" in a:
                    self.pairs.append((name, a.get("value", "on")))
            elif kind not in ("submit", "button"):
                self.pairs.append((name, a.get("value", "")))
        elif tag == "select" and name:
            self._select = [name, None, None]  # name, first value, selected value
        elif tag == "option" and self._select is not None:
            value = a.get("value", "")
            if self._select[1] is None:
                self._select[1] = value
            if "selected" in a:
                self._select[2] = value
        elif tag == "textarea" and name:
            self._textarea = [name, ""]

    def handle_endtag(self, tag):
        if tag == "select" and self._select is not None:
            name, first, chosen = self._select
            self.pairs.append((name, chosen if chosen is not None else (first or "")))
            self._select = None
        elif tag == "textarea" and self._textarea is not None:
            self.pairs.append(tuple(self._textarea))
            self._textarea = None
        elif tag == "form" and self._in_form:
            self._in_form = False
            self._done = True

    def handle_data(self, data):
        if self._textarea is not None:
            self._textarea[1] += data


@pytest.fixture
def admin(tmp_path):
    store = SettingsStore(str(tmp_path / "s.db"))
    app = Flask(__name__)
    app.secret_key = "t"
    create_admin_blueprint(app=app, password_hash=b"x", settings_store=store,
                           signal_handler=None, loop=None, admin_phone="+1")
    client = app.test_client()
    with client.session_transaction() as s:
        s["authenticated"] = True
    return store, client


def _resubmit(client, path):
    page = client.get(path)
    assert page.status_code == 200
    parser = _FormFields()
    parser.feed(page.get_data(as_text=True))
    resp = client.post(path, data=MultiDict(parser.pairs))
    assert resp.status_code == 200
    body = resp.get_data(as_text=True)
    assert 'class="error' not in body.split("<main")[-1], body[:2000]
    return body


SEEDED = {
    "llm_enabled": True,
    "llm_base_url": "https://openrouter.ai/api/v1",
    "llm_model": "z-ai/glm-5.3-flash",
    "llm_api_key": "sk-secret-llm",
    "llm_temperature": 1.0,
    "llm_max_tokens": 16000,
    "llm_system_prompt": "You are Sigil.",
    "llm_extra_body": '{"thinking": {"type": "enabled"}}',
    "llm_provider_order": "relace, streamlake/fp8",
    "llm_provider_only": True,
    "llm_provider_sort": "latency",
    "reactor_enabled": True,
    "reactor_model": "deepseek/deepseek-v4.1-flash",
    "natural_response_enabled": False,
    "deep_think_enabled": True,
    "deep_think_api_key": "sk-secret-dt",
    "deep_think_model": "deepseek-flash",
    "deep_think_max_tool_rounds": 60,
    "deep_think_caps_enabled": False,
    "jev_enabled": False,
    "wsb_subreddit": "wallstreetbets",
    "wsb_indexable": False,
    "MASSIVE_PRO": False,
    "FRED_API_KEY": "fred-secret",
    "admin_numbers": ["+15551234567", "9bd86151-d907-4ac4-b8ce-9822947f8ade"],
    "bot_name": "Sigil",
}


@pytest.mark.parametrize("path", ["/admin/llm", "/admin/settings"])
def test_resubmitting_a_page_changes_nothing(admin, path):
    store, client = admin
    for key, value in SEEDED.items():
        store.set(key, value)
    before = dict(store.all())
    body = _resubmit(client, path)
    after = dict(store.all())
    # Unset keys may be written with the default the page displays (as they
    # always were); anything already stored must come back unchanged.
    changed = {k: (v, after.get(k)) for k, v in before.items() if after.get(k) != v}
    assert not changed
    for secret in ("sk-secret-llm", "sk-secret-dt", "fred-secret"):
        assert secret not in body


def test_llm_toggles_turn_off_and_on(admin):
    """A switched-off toggle must save False, including JEV, whose keys are
    skipped when absent from the form."""
    store, client = admin
    for key in ("llm_provider_only", "reactor_enabled", "jev_enabled"):
        store.set(key, True)
    page = client.get("/admin/llm")
    parser = _FormFields()
    parser.feed(page.get_data(as_text=True))
    off = {"llm_provider_only", "reactor_enabled", "jev_enabled"}
    pairs = [(k, v) for k, v in parser.pairs if not (k in off and v == "true")]
    client.post("/admin/llm", data=MultiDict(pairs))
    assert all(store.get(k) is False for k in off)


def test_blank_wsb_settings_fall_back_to_defaults(tmp_path):
    """The settings page saves blank text fields as ""; the digest must read
    that as "use the default", not crawl an empty URL."""
    from src.wsb import service

    store = SettingsStore(str(tmp_path / "w.db"))
    for key in ("wsb_redlib_base_url", "wsb_subreddit", "wsb_public_base_url", "wsb_user_agent"):
        store.set(key, "")
    svc = service.WSBDigestService.__new__(service.WSBDigestService)
    svc.settings = store
    cfg = svc._cfg()
    assert cfg["base_url"] == service.DEFAULT_BASE_URL
    assert cfg["subreddit"] == service.DEFAULT_SUBREDDIT
    assert cfg["public_base_url"] == service.DEFAULT_PUBLIC_BASE_URL
    assert cfg["user_agent"] == service.DEFAULT_USER_AGENT
    assert cfg["indexable"] is True
