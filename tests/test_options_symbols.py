"""Tests for OCC contract symbol helpers.

Cover the round-trip path: friendly form → OCC → parse → friendly form
preserves underlying, expiration, strike, and option type. The parser
is intentionally lenient (the LLM's friendly emissions vary) so the
tests pin down the supported shapes.
"""

from __future__ import annotations

import datetime as dt

import pytest

from src.options_symbols import (
    build_occ,
    friendly_name,
    intrinsic_value,
    is_occ,
    normalize_contract,
    parse_occ,
)


def test_build_occ_basic():
    occ = build_occ("AAPL", dt.date(2026, 6, 20), "call", 175.0)
    assert occ == "AAPL260620C00175000"


def test_build_occ_fractional_strike():
    occ = build_occ("SPY", dt.date(2025, 12, 19), "put", 482.50)
    assert occ == "SPY251219P00482500"


def test_parse_occ_round_trip():
    occ = "AAPL260620C00175000"
    parts = parse_occ(occ)
    assert parts.root == "AAPL"
    assert parts.expiration == dt.date(2026, 6, 20)
    assert parts.option_type == "call"
    assert parts.strike == pytest.approx(175.0)
    assert build_occ(parts.root, parts.expiration, parts.option_type, parts.strike) == occ


def test_parse_occ_strips_polygon_prefix():
    parts = parse_occ("O:TSLA260116P00150000")
    assert parts.root == "TSLA"
    assert parts.option_type == "put"
    assert parts.strike == pytest.approx(150.0)


def test_is_occ_positive_and_negative():
    assert is_occ("AAPL260620C00175000")
    assert is_occ("O:AAPL260620C00175000")
    assert not is_occ("AAPL 175C 2026-06-20")
    assert not is_occ("")
    assert not is_occ("not even close")


@pytest.mark.parametrize("friendly", [
    "AAPL 175C 2026-06-20",
    "AAPL 175c 2026-06-20",
    "AAPL 2026-06-20 175 call",
    "AAPL call 175 2026-06-20",
    "AAPL 06/20/2026 175c",
])
def test_normalize_contract_friendly_forms(friendly):
    assert normalize_contract(friendly) == "AAPL260620C00175000"


def test_normalize_contract_already_canonical():
    assert normalize_contract("AAPL260620C00175000") == "AAPL260620C00175000"


def test_normalize_contract_strips_polygon_prefix():
    assert normalize_contract("O:AAPL260620C00175000") == "AAPL260620C00175000"


def test_normalize_contract_rejects_garbage():
    with pytest.raises(ValueError):
        normalize_contract("definitely not a contract")
    with pytest.raises(ValueError):
        normalize_contract("AAPL 175")  # missing date + type
    with pytest.raises(ValueError):
        normalize_contract("")


def test_friendly_name_round_trip():
    occ = "AAPL260620C00175000"
    name = friendly_name(occ)
    assert "AAPL" in name
    assert "175" in name
    assert "2026-06-20" in name
    assert name.endswith("2026-06-20")


def test_friendly_name_drops_trailing_decimals():
    # Whole-dollar strike should render as $175C, not $175.00C.
    assert "$175C" in friendly_name("AAPL260620C00175000")
    # Fractional strike keeps the cents.
    assert "$482.5" in friendly_name("SPY251219P00482500")


def test_intrinsic_value_call_itm_and_otm():
    assert intrinsic_value("call", 100, 110) == pytest.approx(10.0)
    assert intrinsic_value("call", 100, 90) == pytest.approx(0.0)
    assert intrinsic_value("call", 100, 100) == pytest.approx(0.0)


def test_intrinsic_value_put_itm_and_otm():
    assert intrinsic_value("put", 100, 90) == pytest.approx(10.0)
    assert intrinsic_value("put", 100, 110) == pytest.approx(0.0)


def test_intrinsic_value_unknown_type_raises():
    with pytest.raises(ValueError):
        intrinsic_value("straddle", 100, 100)


def test_build_occ_rejects_zero_strike():
    with pytest.raises(ValueError):
        build_occ("AAPL", dt.date(2026, 6, 20), "call", 0.0)


def test_build_occ_rejects_unknown_type():
    with pytest.raises(ValueError):
        build_occ("AAPL", dt.date(2026, 6, 20), "straddle", 100.0)
