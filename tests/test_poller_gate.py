"""Early-connected pollers hold messages until the bot is wired."""

import asyncio

from src.signal.poller import SignalPoller


async def test_messages_wait_for_ready():
    ready, seen = asyncio.Event(), []

    async def on_message(data):
        seen.append(data["envelope"]["dataMessage"]["message"])
    p = SignalPoller("http://x", "+1", on_message, ready=ready)
    task = asyncio.create_task(p._handle_message({"envelope": {"dataMessage": {"message": "hi"}}}))
    await asyncio.sleep(0.01)
    assert seen == []
    ready.set()
    await task
    assert seen == ["hi"]
