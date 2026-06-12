"""Tests for the !kalshi prediction-market command."""

import pytest

from src.commands.base import CommandContext
from src.commands.kalshi_command import (
    KalshiCommand,
    _match_events,
    _match_series,
    _price_cents,
    _volume_24h,
)


def _no_series(monkeypatch, cmd):
    """Stub the series catalog empty so unit tests never touch the API."""
    async def empty():
        return []
    monkeypatch.setattr(cmd, "_series_catalog", empty)


def _ctx(args):
    return CommandContext(
        sender="+15551234567",
        group_id=None,
        raw_message="!kalshi " + " ".join(args),
        command="kalshi",
        args=args,
    )


def _event(title, markets, ticker="EVT-26"):
    return {"event_ticker": ticker, "title": title, "markets": markets}


def _market(**kw):
    m = {"ticker": "EVT-26-T1", "yes_sub_title": "Yes outcome"}
    m.update(kw)
    return m


class TestPriceParsing:
    def test_dollars_fixed_point_preferred(self):
        assert _price_cents(_market(last_price_dollars="0.63", last_price=99)) == 63

    def test_legacy_cents_fallback(self):
        assert _price_cents(_market(last_price=41)) == 41

    def test_untraded_uses_book_midpoint(self):
        m = _market(last_price=0, yes_bid_dollars="0.30", yes_ask_dollars="0.36")
        assert _price_cents(m) == 33

    def test_no_data(self):
        assert _price_cents(_market()) is None

    def test_volume_fp_string(self):
        assert _volume_24h(_market(volume_24h_fp="1500")) == 1500


class TestMatching:
    EVENTS = [
        _event("Government Shutdown in 2026?", [_market(volume_24h=10)]),
        _event("Fed rate cut by July?", [_market(volume_24h=5000)]),
        _event("Fed rate cut by December?", [_market(volume_24h=200)]),
        _event("Heat wave in Chicago", [_market(volume_24h=99999)]),
    ]

    def test_all_token_matches_rank_first(self):
        got = _match_events(self.EVENTS, "fed rate cut", 5)
        assert [e["title"] for e in got] == [
            "Fed rate cut by July?", "Fed rate cut by December?",
        ]

    def test_volume_breaks_ties(self):
        got = _match_events(self.EVENTS, "fed", 5)
        assert got[0]["title"] == "Fed rate cut by July?"

    def test_substring_tokens(self):
        got = _match_events(self.EVENTS, "shutdown", 5)
        assert len(got) == 1 and "Shutdown" in got[0]["title"]

    def test_no_match(self):
        assert _match_events(self.EVENTS, "zebra futures", 5) == []

    def test_limit(self):
        assert len(_match_events(self.EVENTS, "fed", 1)) == 1


class TestSeriesMatching:
    CATALOG = [
        {"ticker": "KXBTCD", "title": "Bitcoin price today", "category": "Crypto", "tags": ["BTC"]},
        {"ticker": "KXBTCMAXW", "title": "How high will Bitcoin get this week?", "category": "Crypto", "tags": ["BTC"]},
        {"ticker": "KXHIGHNY", "title": "Highest temperature in NYC", "category": "Climate", "tags": []},
    ]

    def test_all_tokens_required(self):
        assert _match_series(self.CATALOG, "bitcoin temperature", 5) == []

    def test_tag_matches(self):
        got = _match_series(self.CATALOG, "btc", 5)
        assert len(got) == 2

    def test_shorter_title_ranks_first(self):
        got = _match_series(self.CATALOG, "bitcoin", 5)
        assert got[0]["ticker"] == "KXBTCD"


class TestSearchMerge:
    @pytest.mark.asyncio
    async def test_series_events_lead_and_dedupe(self, monkeypatch):
        cmd = KalshiCommand()

        async def fake_catalog():
            return [{"ticker": "KXBTCD", "title": "Bitcoin price today",
                     "category": "Crypto", "tags": ["BTC"]}]

        async def fake_index():
            # Same event also present in the index — must not duplicate.
            return [
                _event("Bitcoin price today at 5pm?", [_market(last_price=50)], ticker="KXBTCD-25JUN13"),
                _event("Will a bitcoin ETF launch?", [_market(last_price=20)], ticker="KXBTCETF-26"),
            ]

        async def fake_series_events(_ticker):
            return [_event("Bitcoin price today at 5pm?",
                           [_market(last_price_dollars="0.55")], ticker="KXBTCD-25JUN13")]

        monkeypatch.setattr(cmd, "_series_catalog", fake_catalog)
        monkeypatch.setattr(cmd, "_event_index", fake_index)
        monkeypatch.setattr(cmd, "_events_for_series", fake_series_events)

        got, exact = await cmd._search("bitcoin", 5)
        tickers = [e["event_ticker"] for e in got]
        assert tickers == ["KXBTCD-25JUN13", "KXBTCETF-26"]
        assert exact
        # The live series fetch (55¢), not the stale index copy, won.
        assert _price_cents(got[0]["markets"][0]) == 55

    @pytest.mark.asyncio
    async def test_partial_only_is_marked_inexact(self, monkeypatch):
        cmd = KalshiCommand()
        _no_series(monkeypatch, cmd)

        async def fake_index():
            return [_event("Government spending cuts", [_market(last_price=10)])]
        monkeypatch.setattr(cmd, "_event_index", fake_index)

        got, exact = await cmd._search("government shutdown", 5)
        assert got and not exact

        result = await cmd.execute(_ctx(["government", "shutdown"]))
        assert "no exact match" in result.text


class TestExecute:
    @pytest.mark.asyncio
    async def test_no_args_is_error(self):
        cmd = KalshiCommand()
        result = await cmd.execute(_ctx([]))
        assert not result.success

    @pytest.mark.asyncio
    async def test_search_formats_matches(self, monkeypatch):
        cmd = KalshiCommand()
        _no_series(monkeypatch, cmd)

        async def fake_index():
            return [_event(
                "Fed rate cut by July?",
                [_market(last_price_dollars="0.27", volume_24h=1234)],
            )]
        monkeypatch.setattr(cmd, "_event_index", fake_index)

        result = await cmd.execute(_ctx(["fed", "rate", "cut"]))
        assert result.success
        assert "Fed rate cut by July?" in result.text
        assert "YES 27¢" in result.text

    @pytest.mark.asyncio
    async def test_ticker_path_then_search_fallback(self, monkeypatch):
        """An unknown all-caps ticker should fall back to text search."""
        cmd = KalshiCommand()
        _no_series(monkeypatch, cmd)

        async def fake_lookup(_ticker):
            return None

        async def fake_index():
            return [_event("CPI2026 special", [_market(last_price=50)])]
        monkeypatch.setattr(cmd, "_lookup_ticker", fake_lookup)
        monkeypatch.setattr(cmd, "_event_index", fake_index)

        result = await cmd.execute(_ctx(["CPI2026"]))
        assert result.success
        assert "CPI2026 special" in result.text

    @pytest.mark.asyncio
    async def test_count_flag_clamped(self, monkeypatch):
        cmd = KalshiCommand()
        _no_series(monkeypatch, cmd)
        events = [
            _event(f"Fed market {i}", [_market(volume_24h=i)], ticker=f"E{i}")
            for i in range(12)
        ]

        async def fake_index():
            return events
        monkeypatch.setattr(cmd, "_event_index", fake_index)

        result = await cmd.execute(_ctx(["fed", "-n", "99"]))
        assert result.text.count("◆") == 10  # clamped to max 10

    @pytest.mark.asyncio
    async def test_api_failure_is_clean_error(self, monkeypatch):
        cmd = KalshiCommand()
        _no_series(monkeypatch, cmd)

        async def boom():
            raise RuntimeError("connection reset")
        monkeypatch.setattr(cmd, "_event_index", boom)

        result = await cmd.execute(_ctx(["fed"]))
        assert not result.success
        assert "Kalshi lookup failed" in result.text

    @pytest.mark.asyncio
    async def test_help_flag(self):
        cmd = KalshiCommand()
        result = await cmd.execute(_ctx(["-help"]))
        assert result.success
        assert "implied probability" in result.text.lower() or "63" in result.text
