"""Tests for the OpenRouter provider-routing helper.

The helper composes the `provider` field that gets attached to the
chat-completion payload when the admin has pinned upstream providers.
Behavior matters because the cost of getting it wrong is silent: a
malformed `provider` block makes OpenRouter ignore it and silently
route to whatever it would have anyway.
"""

from src.llm.client import build_provider_routing


def test_empty_inputs_return_none():
    assert build_provider_routing(order=None, only=False, sort=None) is None
    assert build_provider_routing(order="", only=False, sort="") is None
    assert build_provider_routing(order="   ", only=True, sort="") is None


def test_order_alone_produces_order_field_no_fallback_flag():
    """`only=False` is the default — fallback should NOT be set so
    OpenRouter's default behavior (try listed first, then fall back) is
    preserved. Only flip allow_fallbacks when the user explicitly asks."""
    out = build_provider_routing(order="Cerebras, Groq", only=False, sort=None)
    assert out == {"order": ["Cerebras", "Groq"]}


def test_order_with_only_disables_fallbacks():
    out = build_provider_routing(order="Cerebras, Groq", only=True, sort=None)
    assert out == {
        "order": ["Cerebras", "Groq"],
        "allow_fallbacks": False,
    }


def test_only_without_order_doesnt_set_fallback_flag():
    """`only=True` is meaningless without an order — it would fail every
    request. Skip the flag so the request just uses default routing."""
    out = build_provider_routing(order="", only=True, sort=None)
    assert out is None


def test_order_accepts_whitespace_and_comma_separators():
    # Users will paste "Cerebras Groq" or "Cerebras,Groq" — both should work.
    out = build_provider_routing(order="Cerebras Groq Together", only=False, sort=None)
    assert out == {"order": ["Cerebras", "Groq", "Together"]}


def test_sort_normalizes_case_and_validates():
    assert build_provider_routing(order=None, only=False, sort="THROUGHPUT") == {"sort": "throughput"}
    assert build_provider_routing(order=None, only=False, sort="latency") == {"sort": "latency"}
    assert build_provider_routing(order=None, only=False, sort="price") == {"sort": "price"}


def test_sort_unknown_value_silently_dropped():
    """Unknown sort values must not be passed through — OpenRouter would
    400 on them, which would be a bad failure mode for a typo in the
    admin UI."""
    out = build_provider_routing(order=None, only=False, sort="fastest")
    assert out is None


def test_combined_order_and_sort():
    out = build_provider_routing(
        order="Cerebras, Groq", only=True, sort="throughput",
    )
    assert out == {
        "order": ["Cerebras", "Groq"],
        "allow_fallbacks": False,
        "sort": "throughput",
    }
