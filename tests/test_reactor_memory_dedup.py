"""Tests for the reactor's cross-kind memory dedup.

`MemoryStore.add()` already corroborates within (context_id, subject_key,
kind) using a strict Jaccard ≥ 0.85 check. The reactor sneaks duplicates
past that two ways:

  1. Same fact written under different `kind`s ("fact" vs. "preference"
     vs. "identity") — different kind means a different WHERE clause in
     the corroboration query, so the new row lands as a parallel one.
  2. Rephrasings whose Jaccard falls below 0.85 — short content where
     one different word drags overlap dramatically.

`find_similar_for_subject` + `_is_similar_for_dedup` close those holes
with a looser bidirectional-overlap check across all kinds. These tests
pin the contract on both pieces.
"""

import pytest

from src.memory import (
    MemoryStore,
    _is_similar_for_dedup,
    _is_near_duplicate,
)


# ---------- _is_similar_for_dedup --------------------------------------------

def test_similar_dedup_catches_rephrasing_strict_check_misses():
    """The whole point of the looser threshold: catch wording variations
    that the strict 0.85 same-kind check lets through."""
    a = "loves astrology"
    b = "is into astrology"
    # Strict check: tokens {loves, astrology} vs {into, astrology},
    # Jaccard 1/3 = 0.33, below 0.85 → not a duplicate by strict rules.
    assert _is_near_duplicate(a, b) is False
    # Looser check: forward overlap is 1/2 = 0.5, hits the threshold.
    assert _is_similar_for_dedup(a, b) is True


def test_similar_dedup_identical_content():
    assert _is_similar_for_dedup("Sun in Sag", "Sun in Sag") is True


def test_similar_dedup_disjoint_content():
    """Sanity: completely unrelated facts are NOT deduped."""
    assert _is_similar_for_dedup("works at Google", "lives in Brooklyn") is False


def test_similar_dedup_subset_of_existing():
    """If the new content's tokens are entirely contained in an existing
    memory's tokens, the new write is redundant — backward overlap = 1.0."""
    new = "loves astrology"
    existing = "loves astrology and tarot deeply"
    assert _is_similar_for_dedup(new, existing) is True


def test_similar_dedup_empty_inputs():
    assert _is_similar_for_dedup("", "anything") is False
    assert _is_similar_for_dedup("anything", "") is False
    assert _is_similar_for_dedup("", "") is False


def test_similar_dedup_threshold_is_tunable():
    """Caller can tighten or loosen the threshold per-call."""
    a, b = "loves astrology", "is into astrology"
    assert _is_similar_for_dedup(a, b, threshold=0.4) is True
    assert _is_similar_for_dedup(a, b, threshold=0.9) is False


# ---------- find_similar_for_subject (store integration) ---------------------

@pytest.fixture
def store(tmp_path):
    return MemoryStore(db_path=str(tmp_path / "memories.db"))


@pytest.mark.asyncio
async def test_find_similar_returns_none_when_no_matches(store):
    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:david",
        content="loves astrology",
    )
    assert found is None


@pytest.mark.asyncio
async def test_find_similar_catches_cross_kind_duplicate(store):
    """Reactor wrote the fact as `preference`; later wants to write the
    same content as `fact`. The strict same-kind path won't merge those
    (different WHERE), but find_similar_for_subject crosses kinds."""
    await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="preference",
        content="loves astrology",
    )

    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:david",
        content="loves astrology",
    )
    assert found is not None
    assert found["kind"] == "preference"
    assert found["content"] == "loves astrology"


@pytest.mark.asyncio
async def test_find_similar_catches_rephrasing(store):
    await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="preference",
        content="is into astrology",
    )

    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:david",
        content="loves astrology",
    )
    assert found is not None
    assert "astrology" in found["content"]


@pytest.mark.asyncio
async def test_find_similar_scoped_to_subject(store):
    """Same content under a different subject is NOT a duplicate — two
    people can both 'love astrology' independently."""
    await store.add(
        context_id=1, subject_key="user:alice",
        subject_label="Alice", kind="preference",
        content="loves astrology",
    )

    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:bob",
        content="loves astrology",
    )
    assert found is None


@pytest.mark.asyncio
async def test_find_similar_scoped_to_context(store):
    """Memories live per-chat; a fact from one context shouldn't suppress
    writes in another."""
    await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="preference",
        content="loves astrology",
    )

    found = await store.find_similar_for_subject(
        context_id=2, subject_key="user:david",
        content="loves astrology",
    )
    assert found is None


@pytest.mark.asyncio
async def test_find_similar_ignores_unrelated_facts_for_same_subject(store):
    """A subject can have multiple distinct facts — only near-duplicates
    of the new content count as "already stored"."""
    await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="fact",
        content="lives in Brooklyn",
    )
    await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="identity",
        content="senior software engineer",
    )

    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:david",
        content="loves astrology",
    )
    assert found is None


@pytest.mark.asyncio
async def test_find_similar_returns_first_match_ordered_by_recency(store):
    """When multiple stored memories would match, the most recently
    updated one is returned — that's the "current" fact for the subject."""
    older = await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="preference",
        content="loves astrology",
    )
    newer = await store.add(
        context_id=1, subject_key="user:david",
        subject_label="David", kind="fact",
        content="really into astrology",
    )

    found = await store.find_similar_for_subject(
        context_id=1, subject_key="user:david",
        content="astrology fan",
    )
    assert found is not None
    # ORDER BY updated_at DESC — newer match wins.
    assert found["id"] == newer
    # And: older still exists, just wasn't returned first.
    assert older != newer
