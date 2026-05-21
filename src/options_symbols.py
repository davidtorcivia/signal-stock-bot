"""
Options contract symbol helpers (OCC standard).

The OCC symbol is the canonical identifier used across the portfolio
system, the LLM tools, and the Polygon (Massive) chain endpoint. Format:

    ROOT YYMMDD C/P STRIKE(×1000, 8 digits zero-padded)

Example: ``AAPL250620C00175000`` → AAPL call, expiring 2025-06-20,
strike $175.00.

The LLM (and humans) often think in friendly terms like
``"AAPL 175C 2026-06-20"`` or ``"AAPL 2026-06-20 175 call"``. The
``normalize_contract`` entrypoint accepts those forms and rewrites to
OCC so the rest of the system stays canonical.
"""

from __future__ import annotations

import datetime as dt
import re
from dataclasses import dataclass
from typing import Optional


_OCC_RE = re.compile(
    r"^(?P<root>[A-Z]{1,6})"
    r"(?P<yy>\d{2})(?P<mm>\d{2})(?P<dd>\d{2})"
    r"(?P<cp>[CP])"
    r"(?P<strike>\d{8})$"
)


@dataclass(frozen=True)
class ContractParts:
    root: str           # underlying ticker (uppercase)
    expiration: dt.date # expiration date (calendar)
    option_type: str    # 'call' or 'put'
    strike: float       # dollar strike


def build_occ(
    root: str,
    expiration: dt.date,
    option_type: str,
    strike: float,
) -> str:
    """Construct the canonical OCC symbol. Strike is rounded to the
    nearest cent (8-digit strike-×-1000 has 3-decimal precision)."""
    root = root.strip().upper()
    if not root or not root.isalpha() or len(root) > 6:
        raise ValueError(f"invalid root {root!r}; need 1-6 alpha chars")
    ot = option_type.strip().lower()
    if ot in ("c", "call"):
        cp = "C"
    elif ot in ("p", "put"):
        cp = "P"
    else:
        raise ValueError(f"invalid option_type {option_type!r}; need call/put")
    if strike <= 0:
        raise ValueError(f"strike must be positive, got {strike!r}")
    strike_int = int(round(strike * 1000))
    if strike_int >= 10**8:
        raise ValueError(f"strike {strike!r} too large for OCC encoding")
    return (
        f"{root}{expiration.strftime('%y%m%d')}{cp}"
        f"{strike_int:08d}"
    )


def parse_occ(symbol: str) -> ContractParts:
    """Parse a canonical OCC symbol. Accepts an optional ``O:`` prefix
    (Polygon convention) so callers can pass either form."""
    s = symbol.strip().upper()
    if s.startswith("O:"):
        s = s[2:]
    m = _OCC_RE.match(s)
    if not m:
        raise ValueError(f"not an OCC contract symbol: {symbol!r}")
    yy = int(m.group("yy"))
    # OCC uses 2-digit years. Standard rollover convention: 00-49 → 20xx,
    # 50-99 → 19xx. We never trade pre-2000 contracts in practice, so
    # the rollover is purely defensive.
    year = 2000 + yy if yy < 70 else 1900 + yy
    expiration = dt.date(year, int(m.group("mm")), int(m.group("dd")))
    cp = m.group("cp")
    strike = int(m.group("strike")) / 1000.0
    return ContractParts(
        root=m.group("root"),
        expiration=expiration,
        option_type="call" if cp == "C" else "put",
        strike=strike,
    )


def is_occ(symbol: str) -> bool:
    """Best-effort check: does `symbol` look like a canonical OCC
    contract? Used to short-circuit normalization."""
    s = symbol.strip().upper()
    if s.startswith("O:"):
        s = s[2:]
    return bool(_OCC_RE.match(s))


# Accept friendly forms the LLM is likely to emit:
#   AAPL 175C 2026-06-20
#   AAPL 2026-06-20 175 call
#   AAPL 175 call 2026-06-20
#   AAPL 06/20/2026 175c
#
# Strategy: pull out the underlying, an ISO or US-style date, a strike
# number, and a call/put marker, then reassemble via build_occ. Anything
# we can't parse falls through with a clear error.
_FRIENDLY_DATE_PATTERNS = (
    "%Y-%m-%d",
    "%Y/%m/%d",
    "%m/%d/%Y",
    "%m-%d-%Y",
    "%m/%d/%y",
)


def _parse_friendly_date(token: str) -> Optional[dt.date]:
    for fmt in _FRIENDLY_DATE_PATTERNS:
        try:
            return dt.datetime.strptime(token, fmt).date()
        except ValueError:
            continue
    return None


_STRIKE_WITH_LETTER_RE = re.compile(r"^(\d+(?:\.\d+)?)([cCpP])$")
_FLOAT_RE = re.compile(r"^\d+(?:\.\d+)?$")


def normalize_contract(symbol_or_friendly: str) -> str:
    """Accept either a canonical OCC symbol or a friendly form and
    return the canonical OCC symbol. Friendly forms are recognized
    loosely so the LLM doesn't have to be pedantic. On parse failure,
    raises ValueError with a message the bot can read back to the
    chat.
    """
    if not symbol_or_friendly or not symbol_or_friendly.strip():
        raise ValueError("empty contract symbol")
    raw = symbol_or_friendly.strip()
    if is_occ(raw):
        # Drop any O: prefix and re-canonicalize uppercase. Round-tripping
        # through parse + build also catches malformed roots and dates.
        parts = parse_occ(raw)
        return build_occ(parts.root, parts.expiration, parts.option_type, parts.strike)

    tokens = [t for t in re.split(r"[\s,]+", raw) if t]
    if len(tokens) < 3:
        raise ValueError(
            f"can't parse contract {symbol_or_friendly!r}; expected "
            f"e.g. 'AAPL 175C 2026-06-20' or full OCC symbol"
        )

    root: Optional[str] = None
    expiration: Optional[dt.date] = None
    strike: Optional[float] = None
    opt_type: Optional[str] = None

    for tok in tokens:
        upper = tok.upper()
        # Underlying — must come before strike/date but parsers tolerate
        # any order. Match the first alpha-only token.
        if root is None and upper.isalpha() and len(upper) <= 6:
            if upper in ("CALL", "PUT", "C", "P"):
                opt_type = "call" if upper.startswith("C") else "put"
                continue
            root = upper
            continue
        # Combined strike+type like "175C" or "175.5p"
        m = _STRIKE_WITH_LETTER_RE.match(tok)
        if m and strike is None:
            strike = float(m.group(1))
            opt_type = "call" if m.group(2).upper() == "C" else "put"
            continue
        # Bare strike number
        if strike is None and _FLOAT_RE.match(tok):
            strike = float(tok)
            continue
        # Date
        if expiration is None:
            parsed = _parse_friendly_date(tok)
            if parsed is not None:
                expiration = parsed
                continue
        # Standalone option type word
        if opt_type is None and upper in ("CALL", "PUT", "C", "P"):
            opt_type = "call" if upper.startswith("C") else "put"
            continue
        # Anything else: ignore (lets descriptive words like "weekly"
        # slip through without breaking parse).

    missing = [
        n for n, v in (
            ("root", root), ("expiration", expiration),
            ("strike", strike), ("option_type", opt_type),
        ) if v is None
    ]
    if missing:
        raise ValueError(
            f"can't parse contract {symbol_or_friendly!r}; missing "
            f"{', '.join(missing)}. Expected e.g. 'AAPL 175C 2026-06-20'."
        )
    # mypy/typing assert — all four are non-None after the missing check
    assert root is not None and expiration is not None
    assert strike is not None and opt_type is not None
    return build_occ(root, expiration, opt_type, strike)


def friendly_name(occ_symbol: str) -> str:
    """Render an OCC symbol as a chat-friendly string for display.
    Example: ``AAPL250620C00175000`` → ``"AAPL $175C 2025-06-20"``."""
    parts = parse_occ(occ_symbol)
    tag = "C" if parts.option_type == "call" else "P"
    # Drop trailing .00 from whole-dollar strikes for readability.
    if abs(parts.strike - round(parts.strike)) < 1e-9:
        strike_part = f"${int(round(parts.strike))}"
    else:
        strike_part = f"${parts.strike:.2f}".rstrip("0").rstrip(".")
    return f"{parts.root} {strike_part}{tag} {parts.expiration.isoformat()}"


def intrinsic_value(
    option_type: str, strike: float, underlying_price: float,
) -> float:
    """Per-share intrinsic value at expiration / for settlement. For
    paper trading we model ITM contracts as cash-settled at this value
    × 100 × qty; OTM contracts settle at $0 (expire worthless)."""
    if option_type == "call":
        return max(0.0, underlying_price - strike)
    if option_type == "put":
        return max(0.0, strike - underlying_price)
    raise ValueError(f"unknown option_type {option_type!r}")


__all__ = [
    "ContractParts",
    "build_occ",
    "parse_occ",
    "is_occ",
    "normalize_contract",
    "friendly_name",
    "intrinsic_value",
]
