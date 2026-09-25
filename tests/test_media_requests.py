"""Movie/TV request chat: JEV screen → LLM extract → lookup/pick → add,
the ⏳ → ✅ poll (with release-date skipping and per-poll dedupe), the
already-on-server shame line, season-aware TV progress, and the dispatcher
hook that keeps the reactor off these messages."""

import asyncio
import json
import time
from types import SimpleNamespace

import aiohttp
import pytest

import src.media_requests as media_requests
from src.media_requests import (
    AVAILABLE, NOT_FOUND, WAITING, Arr, MediaRequests, movie_release, tv_progress,
)

DAY = 86400


@pytest.fixture(autouse=True)
def no_episode_wait(monkeypatch):
    monkeypatch.setattr(media_requests, "EPISODE_WAIT_SECONDS", 0)

NOW = time.time()


def iso(ts):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ts))


class FakeArr(Arr):
    def __init__(self, kind, library=None, catalog=None, episodes=None):
        super().__init__(kind, "http://arr", "key")
        self.library = library or {}    # id -> record
        self.catalog = catalog or []    # lookup results
        self.eps = episodes or {}       # series id -> episodes
        self.added, self.searched, self.gets = [], [], 0

    async def lookup(self, term):
        await asyncio.sleep(0)  # let concurrent requests interleave
        return self.catalog

    async def get(self, item_id):
        self.gets += 1
        if item_id not in self.library:
            raise aiohttp.ClientResponseError(None, (), status=404)
        return self.library[item_id]

    async def episodes(self, series_id):
        return self.eps.get(series_id, [])

    async def episode_files(self, series_id):
        return [{"seasonNumber": e["seasonNumber"], "dateAdded": iso(NOW - 400 * DAY)}
                for e in self.eps.get(series_id, []) if e.get("hasFile")]

    async def add(self, found, seasons):
        await asyncio.sleep(0)
        item = {**found, "id": 100 + len(self.added), "hasFile": False}
        found["id"] = item["id"]  # later lookups see it in the library
        self.added.append((item, seasons))
        self.library[item["id"]] = item
        return item

    async def monitor_and_search(self, item, seasons):
        self.searched.append((item["id"], seasons))


class FakeLLM:
    def __init__(self, reply):
        self.reply, self.calls = reply, 0

    def status(self):
        return {"ready": True}

    async def chat_messages(self, **kw):
        self.calls += 1
        if kw.get("purpose") == "media_reply":
            return {"content": ""}  # no paraphrase: the template goes out
        return {"content": json.dumps(self.reply)}


class FakeJev:
    def __init__(self, **answers):
        self.answers, self.asked = answers, []

    async def choose(self, *, state, questions, purpose):
        self.asked.append(purpose)
        return {k: self.answers[k] for k in questions if k in self.answers}


class FakeHandler:
    config = SimpleNamespace(phone_number="+1555")

    def __init__(self):
        self.reactions, self.messages, self.removed = [], [], []
        self.fail_send = False

    async def send_reaction(self, **kw):
        (self.removed if kw.get("remove") else self.reactions).append(kw["emoji"])
        return True

    async def send_message(self, **kw):
        if self.fail_send:
            raise RuntimeError("signal send failed")
        self.messages.append(kw["message"])


class Store(dict):
    def get(self, key, default=None):
        return super().get(key, default)


def make(tmp_path, reply, radarr=None, sonarr=None, jev=None):
    handler = FakeHandler()
    pool = SimpleNamespace(for_phone=lambda p: handler, default=lambda: handler)
    mr = MediaRequests(
        settings_store=Store(media_requests_enabled=True, media_request_groups=["g1"]),
        llm=FakeLLM(reply), jev=jev or FakeJev(intent="request"), signal_pool=pool,
        state_path=str(tmp_path / "state.json"),
    )
    mr.arr = {"movie": radarr, "tv": sonarr}.get
    return mr, handler


def req(title, type="movie", year=None, seasons=None):
    return {"requests": [{"title": title, "type": type, "year": year, "seasons": seasons}]}


async def test_request_waits_then_flips_to_check(tmp_path):
    radarr = FakeArr("movie", catalog=[{"title": "Dune", "year": 2021, "status": "released"}])
    mr, h = make(tmp_path, req("Dune", year=2021), radarr)

    await mr.handle(h, "uuid-a", "can you add dune", "g1", 42)
    assert h.reactions == [WAITING]
    assert radarr.added and h.messages == []  # emoji only
    assert json.loads((tmp_path / "state.json").read_text())[0]["ts"] == 42

    await mr.check_pending()
    assert h.reactions == [WAITING]  # still downloading

    radarr.library[100]["hasFile"] = True
    await mr.check_pending()
    assert h.reactions == [WAITING, AVAILABLE]
    assert mr.pending == []


async def test_unreleased_movie_warns_and_skips_checks(tmp_path):
    out = NOW + 90 * DAY
    radarr = FakeArr("movie", catalog=[{"title": "Doomsday", "year": 2026, "status": "announced",
                                        "inCinemas": iso(NOW + 30 * DAY), "digitalRelease": iso(out)}])
    mr, h = make(tmp_path, req("Doomsday"), radarr)
    await mr.handle(h, "uuid-a", "doomsday pls", "g1", 1)
    assert h.messages[0].startswith("Heads up on Doomsday (2026): it isn't out on streaming or disc until")

    await mr.check_pending()          # first poll learns the release date
    await mr.check_pending()          # ...so the second doesn't fetch again
    assert radarr.gets == 1
    assert abs(mr.pending[0]["items"][0]["next_check"] - out) < 5


async def test_poll_fetches_each_record_once(tmp_path):
    radarr = FakeArr("movie", library={7: {"id": 7, "hasFile": False, "status": "released"}})
    mr, h = make(tmp_path, {}, radarr)
    item = {"kind": "movie", "id": 7, "title": "X", "next_check": 0}
    mr.pending = [{"phone": "+1555", "group_id": "g1", "author": a, "ts": 1, "items": [dict(item)]}
                  for a in ("a", "b", "c")]
    await mr.check_pending()
    assert radarr.gets == 1


async def test_already_on_server_says_since_when(tmp_path):
    radarr = FakeArr(
        "movie",
        library={7: {"id": 7, "title": "Heat", "year": 1995, "hasFile": True,
                     "movieFile": {"dateAdded": iso(NOW - 800 * DAY)}}},
        catalog=[{"id": 7, "title": "Heat", "year": 1995}],
    )
    mr, h = make(tmp_path, req("Heat"), radarr)
    await mr.handle(h, "uuid-a", "heat please", "g1", 1)
    assert h.reactions == [AVAILABLE]
    assert "Heat (1995) has been on the server since" in h.messages[0]
    assert "(2 years ago)" in h.messages[0]
    assert not radarr.added and mr.pending == []


async def test_jev_screens_out_chitchat_without_llm(tmp_path):
    mr, h = make(tmp_path, req("Heat"), FakeArr("movie"), jev=FakeJev(intent="other"))
    await mr.handle(h, "uuid-a", "thanks!", "g1", 3)
    assert mr.llm.calls == 0 and h.reactions == [] and h.messages == []


async def test_jev_picks_ambiguous_match_or_none(tmp_path):
    catalog = [{"title": "Dune", "year": 1984}, {"title": "Dune", "year": 2021}]
    radarr = FakeArr("movie", catalog=catalog)
    jev = FakeJev(intent="request", match="1")
    mr, h = make(tmp_path, req("Dune"), radarr, jev=jev)
    await mr.handle(h, "uuid-a", "the new dune", "g1", 1)
    assert radarr.added[0][0]["year"] == 2021 and "media_match" in jev.asked

    jev.answers["match"] = "none"
    await mr.handle(h, "uuid-a", "dune the musical", "g1", 2)
    assert h.reactions[-1] == NOT_FOUND


def _ep(season, days_from_now, has_file):
    return {"seasonNumber": season, "airDateUtc": iso(NOW + days_from_now * DAY), "hasFile": has_file}


def test_tv_progress_is_season_aware():
    eps = [_ep(1, -900, True), _ep(1, -890, True), _ep(2, -400, True), _ep(2, -390, False),
           _ep(3, 30, False), _ep(0, -1000, False)]
    # Season 1 alone is complete.
    assert tv_progress(eps, [1], NOW)[0] is True
    # Whole show: season 2 is missing an episode; unaired 3 and specials don't count.
    done, _, note, targets = tv_progress(eps, None, NOW)
    assert (done, targets, note) == (False, [1, 2], "")
    # Asking for the unaired season: not done, waits until its premiere.
    done, next_check, note, _ = tv_progress(eps, [3], NOW)
    assert not done and "season 3 doesn't start airing until" in note
    assert abs(next_check - (NOW + 30 * DAY)) < 5
    # A season that doesn't exist yet has no date.
    assert "season 9 has no air date yet" in tv_progress(eps, [9], NOW)[2]


def test_movie_release_notes():
    assert movie_release({"status": "released"}, NOW)[0] is True
    for m in ({"status": "announced", "inCinemas": iso(NOW + DAY)},
              {"status": "inCinemas", "inCinemas": iso(NOW - DAY)}, {"status": "announced"}):
        assert "no streaming or disc release date yet" in movie_release(m, NOW)[2]
    home = {"status": "inCinemas", "inCinemas": iso(NOW - DAY), "physicalRelease": iso(NOW + 9 * DAY)}
    assert "isn't out on streaming or disc until" in movie_release(home, NOW)[2]


async def test_season_request_on_existing_show(tmp_path):
    eps = [_ep(1, -900, True), _ep(2, -400, False)]
    sonarr = FakeArr(
        "tv",
        library={5: {"id": 5, "title": "Bear", "monitored": True,
                     "seasons": [{"seasonNumber": 1, "monitored": True},
                                 {"seasonNumber": 2, "monitored": False}]}},
        catalog=[{"id": 5, "title": "Bear", "year": 2022}],
        episodes={5: eps},
    )
    mr, h = make(tmp_path, req("Bear", "tv", seasons=[1]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "bear s1", "g1", 1)
    assert h.reactions == [AVAILABLE] and "Bear (2022) season 1 has been on the server" in h.messages[0]

    mr.llm = FakeLLM(req("Bear", "tv", seasons=[2]))
    await mr.handle(h, "uuid-a", "bear s2", "g1", 2)
    assert h.reactions[-1] == WAITING and sonarr.searched == [(5, [2])]

    eps[1]["hasFile"] = True
    await mr.check_pending()
    assert h.reactions[-1] == AVAILABLE and mr.pending == []


async def test_dispatcher_hands_off_before_reactor():
    from src.commands.dispatcher import CommandDispatcher
    calls = []

    class MR:
        def handles(self, g): return g == "g1"
        async def handle(self, *a): calls.append(a)

    d = CommandDispatcher.__new__(CommandDispatcher)
    d.media_requests, d.signal_pool, d.signal_handler = MR(), None, "H"
    d.prefix, d.context_registry, d.max_message_length = "!", None, 4000
    d.reactor = SimpleNamespace(maybe_react=lambda **kw: calls.append("reactor"))
    d._refresh_live_settings = lambda: None
    d.group_log, d.settings_store = None, None
    allowed = [True]
    d._rate_limiter = SimpleNamespace(check=lambda s: (allowed[0], 30))

    assert await d.dispatch(sender="u", message="dune pls", group_id="g1", target_timestamp=5) is None
    await asyncio.sleep(0)
    assert calls == [("H", "u", "dune pls", "g1", 5)]

    # Rate-limited senders don't reach the request handler.
    allowed[0] = False
    result = await d.dispatch(sender="u", message="more", group_id="g1", target_timestamp=6)
    await asyncio.sleep(0)
    assert result is not None and "Slow down" in result.text and len(calls) == 1


async def test_replies_are_paraphrased_but_keep_facts(tmp_path):
    radarr = FakeArr(
        "movie",
        library={7: {"id": 7, "title": "Heat", "year": 1995, "hasFile": True,
                     "movieFile": {"dateAdded": "2024-06-08T12:00:00Z"}}},
        catalog=[{"id": 7, "title": "Heat", "year": 1995}],
    )
    mr, h = make(tmp_path, req("Heat"), radarr)
    replies = iter([req("Heat"), "lol Heat has been sitting there since Jun 8, 2024, c'mon"])

    async def chat(**kw):
        r = next(replies)
        return {"content": r if isinstance(r, str) else json.dumps(r)}
    mr.llm.chat_messages = chat
    await mr.handle(h, "uuid-a", "heat", "g1", 1)
    assert h.messages == ["lol Heat has been sitting there since Jun 8, 2024, c'mon"]

    # A paraphrase that loses the date falls back to the template.
    replies = iter([req("Heat"), "Heat's already here, check first"])
    await mr.handle(h, "uuid-a", "heat", "g1", 2)
    assert h.messages[-1].startswith("Heat (1995) has been on the server since Jun 8, 2024")


async def test_failed_reply_still_tracks_request(tmp_path):
    radarr = FakeArr("movie", catalog=[{"title": "Soon", "year": 2027, "status": "announced"}])
    mr, h = make(tmp_path, req("Soon"), radarr)
    h.fail_send = True  # the heads-up reply blows up
    await mr.handle(h, "uuid-a", "soon pls", "g1", 1)
    assert h.reactions == [WAITING] and len(mr.pending) == 1


async def test_deleted_titles_drop_the_hourglass_not_check(tmp_path):
    radarr = FakeArr("movie")
    sonarr = FakeArr("tv")  # episodes() answers [] for unknown ids, like Sonarr
    mr, h = make(tmp_path, {}, radarr, sonarr)
    mr.pending = [
        {"phone": "+1555", "group_id": "g1", "author": "a", "ts": 1,
         "items": [{"kind": "movie", "id": 7, "title": "M", "next_check": 0}]},
        {"phone": "+1555", "group_id": "g1", "author": "b", "ts": 2,
         "items": [{"kind": "tv", "id": 9, "seasons": None, "title": "S", "next_check": 0}]},
    ]
    await mr.check_pending()
    assert h.reactions == [] and h.removed == [WAITING, WAITING] and mr.pending == []


async def test_missing_season_of_ended_show_is_not_found(tmp_path):
    sonarr = FakeArr("tv", catalog=[{"title": "The Office", "year": 2005, "status": "ended",
                                     "seasons": [{"seasonNumber": n} for n in range(10)]}])
    mr, h = make(tmp_path, req("The Office", "tv", seasons=[10]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "office s10", "g1", 1)
    assert h.reactions == [NOT_FOUND] and not sonarr.added


async def test_new_show_with_unannounced_season_warns(tmp_path):
    sonarr = FakeArr("tv", catalog=[{"title": "Severance", "year": 2022, "status": "continuing",
                                     "firstAired": iso(NOW - 900 * DAY),
                                     "seasons": [{"seasonNumber": n} for n in range(3)]}])
    mr, h = make(tmp_path, req("Severance", "tv", seasons=[3]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "severance s3", "g1", 1)
    assert h.reactions == [WAITING] and "season 3 has no air date yet" in h.messages[0]


async def test_simultaneous_requests_add_once(tmp_path):
    radarr = FakeArr("movie", catalog=[{"title": "Dune", "year": 2021, "status": "released"}])
    mr, h = make(tmp_path, req("Dune", year=2021), radarr)
    await asyncio.gather(mr.handle(h, "a", "dune", "g1", 1), mr.handle(h, "b", "dune", "g1", 2))
    assert len(radarr.added) == 1 and h.reactions == [WAITING, WAITING] and len(mr.pending) == 2


async def test_ended_show_drops_seasons_that_dont_exist(tmp_path):
    sonarr = FakeArr("tv", catalog=[{"title": "The Office", "year": 2005, "status": "ended",
                                     "seasons": [{"seasonNumber": n} for n in range(10)]}])
    mr, h = make(tmp_path, req("The Office", "tv", seasons=[9, 10]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "office s9-10", "g1", 1)
    assert sonarr.added[0][1] == [9] and mr.pending[0]["items"][0]["seasons"] == [9]


async def test_new_show_waits_for_episode_air_dates(tmp_path):
    sonarr = FakeArr("tv", catalog=[{"title": "TLOU", "year": 2023, "status": "continuing",
                                     "firstAired": iso(NOW - 900 * DAY),
                                     "seasons": [{"seasonNumber": n} for n in range(4)]}])
    calls = []

    async def episodes(series_id):  # empty until Sonarr's refresh lands
        calls.append(series_id)
        return [] if len(calls) < 3 else [_ep(1, -900, True), _ep(3, 40, False)]
    sonarr.episodes = episodes
    mr, h = make(tmp_path, req("TLOU", "tv", seasons=[3]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "tlou s3", "g1", 1)
    assert len(calls) == 3 and "season 3 doesn't start airing until" in h.messages[0]


async def test_future_season_on_monitored_show_turns_on_new_seasons(tmp_path):
    sonarr = FakeArr(
        "tv",
        library={5: {"id": 5, "title": "Show", "monitored": True, "monitorNewItems": "none",
                     "seasons": [{"seasonNumber": 1, "monitored": True}]}},
        catalog=[{"id": 5, "title": "Show", "year": 2024, "status": "continuing"}],
        episodes={5: [_ep(1, -100, True)]},
    )
    mr, h = make(tmp_path, req("Show", "tv", seasons=[3]), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "show s3", "g1", 1)
    assert sonarr.searched == [(5, [3])] and h.reactions == [WAITING]


async def test_episode_read_failure_after_add_still_tracks(tmp_path):
    sonarr = FakeArr("tv", catalog=[{"title": "New", "year": 2026, "status": "continuing",
                                     "seasons": [{"seasonNumber": 1}]}])

    async def episodes(series_id):
        raise aiohttp.ClientConnectionError("sonarr went away")
    sonarr.episodes = episodes
    mr, h = make(tmp_path, req("New", "tv"), sonarr=sonarr)
    await mr.handle(h, "uuid-a", "new show", "g1", 1)
    assert h.reactions == [WAITING] and len(mr.pending) == 1
