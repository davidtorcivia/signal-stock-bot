"""
Tests for the wsb_digest oracle fire path in DailyOracleWorker._fire_wsb.
Stubs the wsb_service, deep-think factory, bot registry, and sender so no
network/LLM/Signal is touched. Verifies the bool/post contract.
"""

import pytest

from src.daily_oracle import DailyOracleWorker
from src.contexts.oracles import ContextOracle
from src.wsb.service import WSBPostPayload


class _Bot:
    id = 1
    display_name = "Sigil"
    signal_phone = "+15550001111"
    enabled = True


class _Registry:
    def get_sync(self, _id):
        return _Bot()

    def default_for_kind_sync(self, _kind):
        return _Bot()


class _DTFactory:
    def __init__(self):
        self.requested = []

    def get_deep_think(self, bot_id):
        self.requested.append(bot_id)
        return object()  # opaque client; wsb_service stub ignores it


class _Sender:
    def __init__(self):
        self.sent = []

    async def send_message(self, recipient, message, group_id=None,
                           attachments=None, styled=False):
        self.sent.append({"message": message, "group_id": group_id})


class _Pool:
    def __init__(self, sender):
        self._s = sender
        self.default_phone = ""

    def for_bot(self, _bot):
        return self._s

    def default(self):
        return self._s


class _WSBService:
    def __init__(self, payload):
        self.payload = payload
        self.posted = []
        self.run_kwargs = None

    async def run(self, *, deep_think_client, caller_ctx, bot_name):
        self.run_kwargs = {"bot_name": bot_name, "caller_ctx": caller_ctx}
        return self.payload

    async def mark_posted(self, date, ok):
        self.posted.append((date, ok))


class _Policy:
    kind = "group"
    key = "grp"
    default_bot_id = None


class _Contexts:
    def __init__(self, policy):
        self._p = policy

    async def get(self, _cid):
        return self._p


def _worker(wsb_service, *, factory=None, sender=None, contexts=None):
    sender = sender or _Sender()
    return DailyOracleWorker(
        oracle_store=None,
        context_registry=contexts,
        tarot_command=None,
        iching_command=None,
        ask_command=None,
        signal_pool=_Pool(sender),
        bot_name="Sigil",
        bot_registry=_Registry(),
        llm_factory=factory or _DTFactory(),
        wsb_service=wsb_service,
    ), sender


def _oracle():
    return ContextOracle(id=7, context_id=1, enabled=True, kind="wsb_digest",
                         schedule_kind="clock", clock_time="20:00", label="")


@pytest.mark.asyncio
async def test_fire_wsb_posts_teaser_and_link():
    payload = WSBPostPayload(
        date="2026-06-02", headline="SPCE eats the tape",
        teaser="One halted name owns the thread; Sigil is unimpressed.",
        page_url="https://sigil.disinfo.zone/wsb/2026-06-02.html", body_md="...")
    svc = _WSBService(payload)
    worker, sender = _worker(svc)

    ok = await worker._fire_wsb(_oracle(), _Policy(), "groupid_abcd1234")
    assert ok is True
    assert len(sender.sent) == 1
    msg = sender.sent[0]["message"]
    assert "📊 WSB daily" in msg
    assert payload.teaser in msg
    assert payload.page_url in msg
    assert svc.posted == [("2026-06-02", True)]
    assert svc.run_kwargs["bot_name"] == "Sigil"


@pytest.mark.asyncio
async def test_fire_wsb_no_payload_returns_false_no_post():
    svc = _WSBService(None)        # service produced nothing (crawl/LLM down)
    worker, sender = _worker(svc)
    ok = await worker._fire_wsb(_oracle(), _Policy(), "groupid_abcd1234")
    assert ok is False
    assert sender.sent == []
    assert svc.posted == []        # slot not marked -> not burned


@pytest.mark.asyncio
async def test_fire_wsb_unwired_service_returns_false():
    worker, sender = _worker(None)
    ok = await worker._fire_wsb(_oracle(), _Policy(), "groupid_abcd1234")
    assert ok is False
    assert sender.sent == []


@pytest.mark.asyncio
async def test_fire_wsb_no_factory_returns_false():
    payload = WSBPostPayload(date="2026-06-02", headline="h", teaser="t",
                             page_url="u", body_md="b")
    worker, sender = _worker(_WSBService(payload), factory=None)
    # llm_factory=None -> _deep_think_for returns None -> skip
    worker.llm_factory = None
    ok = await worker._fire_wsb(_oracle(), _Policy(), "groupid_abcd1234")
    assert ok is False
    assert sender.sent == []


# ---- run_wsb_now (admin "Run now" button) -----------------------------------

@pytest.mark.asyncio
async def test_run_wsb_now_posts_when_requested():
    payload = WSBPostPayload(
        date="2026-06-02", headline="SPCE eats the tape", teaser="One name owns it.",
        page_url="https://sigil.disinfo.zone/wsb/2026-06-02.html", body_md="...")
    svc = _WSBService(payload)
    worker, sender = _worker(svc, contexts=_Contexts(_Policy()))
    res = await worker.run_wsb_now(99, post_to_chat=True)
    assert res["ok"] is True and res["posted"] is True
    assert res["page_url"] == payload.page_url
    assert len(sender.sent) == 1
    assert svc.posted == [("2026-06-02", True)]


@pytest.mark.asyncio
async def test_run_wsb_now_preview_publishes_but_no_post():
    payload = WSBPostPayload(date="2026-06-02", headline="h", teaser="t",
                             page_url="u", body_md="b")
    svc = _WSBService(payload)
    worker, sender = _worker(svc, contexts=_Contexts(_Policy()))
    res = await worker.run_wsb_now(99, post_to_chat=False)
    assert res["ok"] is True and res["posted"] is False
    assert svc.run_kwargs is not None     # the pipeline (page publish) DID run
    assert sender.sent == []              # but nothing posted to chat
    assert svc.posted == []


@pytest.mark.asyncio
async def test_run_wsb_now_rejects_non_group():
    svc = _WSBService(WSBPostPayload(date="d", headline="h", teaser="t",
                                     page_url="u", body_md="b"))
    worker, sender = _worker(svc, contexts=_Contexts(None))  # context not found
    res = await worker.run_wsb_now(99, post_to_chat=True)
    assert res["ok"] is False
    assert sender.sent == []
