"""Signal webhook in, signal-cli REST send out.

Runs the real SignalHandlerPool, CommandDispatcher and BotRegistry (sqlite).
Only signal-cli-rest-api is faked, by a local HTTP server that records every
/v2/send, so the assertions are on which number actually sent what.
"""

from dataclasses import replace
from types import SimpleNamespace

import pytest
from aiohttp import web

from src.bots.models import Bot
from src.bots.registry import BotRegistry
from src.commands.base import BaseCommand, CommandResult
from src.commands.dispatcher import CommandDispatcher
from src.signal.pool import SignalHandlerPool

SIGIL, ARTAUD = "+15550000001", "+15550000002"
USER = "+15559990000"


class WhoCommand(BaseCommand):
    name = "who"
    description = "Say which bot answered."
    usage = "!who"

    async def execute(self, ctx):
        return CommandResult.ok(f"this is {ctx.bot.slug}")


@pytest.fixture
async def signal_api():
    sent = []

    async def send(request):
        sent.append(await request.json())
        return web.json_response({"timestamp": "1"}, status=201)

    async def anything_else(request):
        return web.json_response([])

    app = web.Application()
    app.router.add_post("/v2/send", send)
    app.router.add_route("*", "/{tail:.*}", anything_else)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    await site.start()
    port = site._server.sockets[0].getsockname()[1]
    yield SimpleNamespace(url=f"http://127.0.0.1:{port}", sent=sent)
    await runner.cleanup()


@pytest.fixture
async def bot(tmp_path, signal_api):
    registry = BotRegistry(str(tmp_path / "bots.db"))
    registry.warm_sync()
    sigil = registry.get_by_slug_sync("sigil")
    await registry.upsert(replace(
        sigil, signal_phone=SIGIL, default_for_dm=True, default_for_group=True,
    ))
    await registry.upsert(Bot(
        id=None, slug="artaud", display_name="Artaud", aliases=["artaud"],
        signal_phone=ARTAUD,
    ))
    dispatcher = CommandDispatcher(bot_registry=registry)
    dispatcher.register(WhoCommand())
    pool = SignalHandlerPool(
        default_api_url=signal_api.url, default_phone=SIGIL,
        dispatcher=dispatcher, bot_registry=registry,
    )
    dispatcher.signal_pool = pool
    pool.build()

    async def receive(phone, text, group=None, ts=[1_000]):
        """Deliver one inbound message to the handler polling `phone`."""
        ts[0] += 1
        data = {"message": text, "timestamp": ts[0]}
        if group:
            data["groupInfo"] = {"groupId": group}
        await pool.for_phone(phone).handle_webhook({"envelope": {
            "source": USER, "sourceNumber": USER, "sourceUuid": "user-uuid",
            "timestamp": ts[0], "dataMessage": data,
        }})

    async def disable(slug):
        await registry.upsert(replace(registry.get_by_slug_sync(slug), enabled=False))

    def sends():
        return [(s["number"], s["message"]) for s in signal_api.sent]

    yield SimpleNamespace(receive=receive, disable=disable, sends=sends)
    for phone in (SIGIL, ARTAUD):
        await pool.for_phone(phone).close()


async def test_dm_to_artaud_is_answered_by_artaud_from_his_number(bot):
    await bot.receive(ARTAUD, "!who")
    assert bot.sends() == [(ARTAUD, "this is artaud")]


async def test_dm_small_talk_gets_no_canned_help_reply(bot):
    await bot.receive(ARTAUD, "lol thanks")
    assert bot.sends() == []


async def test_disabled_bot_number_goes_silent_without_restart(bot):
    await bot.disable("artaud")
    await bot.receive(ARTAUD, "!who")
    await bot.receive(ARTAUD, "!who", group="Z3JvdXA=")
    assert bot.sends() == []
    await bot.receive(SIGIL, "!who")
    assert bot.sends() == [(SIGIL, "this is sigil")]
