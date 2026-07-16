"""
Rolling per-context conversation summary.

When a chat accumulates more turns than the rolling history window holds,
verbatim turns get pruned and the LLM loses long-term continuity. The
Summarizer compresses the older portion of each conversation into cited,
structured JSON that lives separately from turn-by-turn history and is
rendered into the volatile user tail on subsequent calls.

Design:
  * Triggered by `maybe_summarize(context_key)` — fire-and-forget from
    the ask path after each persist. The function itself decides whether
    to do work based on how many turns have arrived since the last
    summary update.
  * Per-context lock so two concurrent triggers can't run the LLM twice
    on the same context at the same time.
  * Uses the cheaper "reactor_model" override when available, falling
    back to the main LLM model. Thinking is force-disabled for this call
    (summary is short, no benefit from chain-of-thought on top).
  * Idempotent: storing a summary records the highest turn id folded in
    (`summary_through_id`); the next run only feeds the LLM the new
    material plus the prior summary.
"""

import asyncio
import json
import logging
import re
from typing import Optional

from .history import format_history_timestamp

logger = logging.getLogger(__name__)

DEFAULT_KEEP_RECENT = 12          # turns kept verbatim — same as history window
DEFAULT_MIN_NEW_TURNS = 10        # don't bother summarizing < this many new turns
DEFAULT_MAX_SUMMARY_CHARS = 2500  # hard ceiling on the stored JSON summary
DEFAULT_MAX_TOKENS = 800          # output budget for the summarizer LLM call

SUMMARY_SECTIONS = ("facts", "decisions", "open_questions", "topics")


def _empty_summary() -> dict:
    return {"version": 1, **{section: [] for section in SUMMARY_SECTIONS}}


def parse_structured_summary(raw: str, *, allow_legacy: bool = True) -> Optional[dict]:
    """Validate summary JSON; optionally promote old prose without losing it."""
    text = (raw or "").strip()
    if not text:
        return _empty_summary()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except (TypeError, ValueError):
        if not allow_legacy:
            return None
        value = _empty_summary()
        value["facts"] = [{
            "text": text,
            "source_turn_ids": [],
            "last_confirmed": None,
        }]
        return value
    if not isinstance(value, dict):
        return None
    clean = _empty_summary()
    for section in SUMMARY_SECTIONS:
        rows = value.get(section) or []
        if not isinstance(rows, list):
            continue
        for row in rows:
            if isinstance(row, str):
                row = {"text": row}
            if not isinstance(row, dict):
                continue
            item_text = str(row.get("text") or "").strip()
            if not item_text:
                continue
            sources = row.get("source_turn_ids") or []
            if not isinstance(sources, list):
                sources = []
            clean[section].append({
                "text": item_text[:600],
                "source_turn_ids": [
                    str(source) for source in sources
                    if re.fullmatch(r"h\d+", str(source))
                ][:8],
                "last_confirmed": (
                    str(row.get("last_confirmed"))[:40]
                    if row.get("last_confirmed") else None
                ),
            })
    return clean


def serialize_structured_summary(value: dict, max_chars: int) -> str:
    """Serialize valid JSON under the hard limit without slicing syntax."""
    clean = parse_structured_summary(json.dumps(value), allow_legacy=False)
    if clean is None:
        raise ValueError("invalid structured summary")
    max_chars = max(256, int(max_chars))
    encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    # Remove lowest-value tail items until the complete JSON fits.  Open
    # questions and decisions survive longest because losing them breaks
    # continuity more visibly than dropping a background topic.
    removal_order = ("topics", "facts", "decisions", "open_questions")
    while len(encoded) > max_chars:
        removed = False
        for section in removal_order:
            if clean[section]:
                clean[section].pop()
                removed = True
                break
        if not removed:
            break
        encoded = json.dumps(clean, ensure_ascii=False, separators=(",", ":"))
    return encoded


def render_summary_for_prompt(raw: str) -> str:
    """Turn stored JSON into concise, evidence-linked model context."""
    value = parse_structured_summary(raw, allow_legacy=True)
    if not value:
        return ""
    labels = {
        "facts": "Durable facts",
        "decisions": "Decisions",
        "open_questions": "Open questions",
        "topics": "Ongoing topics",
    }
    parts = []
    for section in SUMMARY_SECTIONS:
        rows = value.get(section) or []
        if not rows:
            continue
        lines = []
        for row in rows:
            sources = row.get("source_turn_ids") or []
            source_note = f" [sources: {', '.join(sources)}]" if sources else ""
            lines.append(f"- {row['text']}{source_note}")
        parts.append(f"{labels[section]}:\n" + "\n".join(lines))
    return "\n\n".join(parts)

SUMMARIZER_PROMPT = """\
You maintain a long-term memory of a Signal chat conversation.

You will receive (a) the current rolling summary, if any, and (b) a batch
of new conversation turns since the last summary update. Produce an
UPDATED structured summary that folds the new material into the existing one.

Rules:
- Return one JSON object with exactly these array keys: facts, decisions,
  open_questions, topics. Each item must have: text, source_turn_ids (an
  array of h-number turn IDs), and last_confirmed (the supplied UTC timestamp
  or null). Do not emit markdown fences.
- Keep the serialized JSON under {max_chars} characters. Aggressively trim less-relevant
  context as the conversation evolves.
- Capture: who is in the conversation, what they care about, key facts
  they've shared (birth charts, watchlists, opinions, preferences),
  ongoing topics, unresolved threads, decisions or commitments.
- Drop: small talk, transient acknowledgements, command output details
  that aren't asked about again.
- Refer to people by their bracket label as it appears in the input
  (e.g. "David said…" not "user 4137 said…").
- If the new turns contradict an earlier fact, prefer the newer one and
  drop the older entry.
- Never invent source IDs. Preserve the source IDs of retained prior entries.
- Output ONLY the JSON object. No preamble or commentary."""


class Summarizer:
    def __init__(self, llm_client, history, settings_store):
        self.llm = llm_client
        self.history = history
        self.store = settings_store
        self._locks: dict[str, asyncio.Lock] = {}

    def _config(self) -> dict:
        store = self.store
        def _int(key, default):
            try:
                v = store.get(key)
                return int(v) if v is not None else default
            except (TypeError, ValueError):
                return default
        return {
            "enabled": bool(store.get("summary_enabled", True)),
            "keep_recent": _int("summary_keep_recent", DEFAULT_KEEP_RECENT),
            "min_new_turns": _int("summary_min_new_turns", DEFAULT_MIN_NEW_TURNS),
            "max_chars": _int("summary_max_chars", DEFAULT_MAX_SUMMARY_CHARS),
            "max_tokens": _int("summary_max_tokens", DEFAULT_MAX_TOKENS),
            "model": (store.get("summary_model") or "").strip() or None,
        }

    def _lock_for(self, context_key: str) -> asyncio.Lock:
        lock = self._locks.get(context_key)
        if lock is None:
            lock = asyncio.Lock()
            self._locks[context_key] = lock
        return lock

    @staticmethod
    def _format_turn(turn: dict) -> str:
        role = turn.get("role") or "?"
        content = (turn.get("content") or "").strip()
        if not content:
            return ""
        if role == "user":
            tail = turn.get("sender_tail")
            label = f"...{tail}" if tail else "user"
            stamp = format_history_timestamp(turn.get("created_at"))
            return f"[turn h{turn.get('id')}; {label}; {stamp}] {content}"
        if role == "assistant":
            stamp = format_history_timestamp(turn.get("created_at"))
            return f"[turn h{turn.get('id')}; bot; {stamp}] {content}"
        if role == "tool":
            # Tool results are too noisy and too long for the summary
            # context. Skip them — the LLM's own assistant turn that
            # follows will reference whatever was useful.
            return ""
        return f"[{role}] {content}"

    async def maybe_summarize(
        self,
        context_key: str,
        floor_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> bool:
        """Fold pending turns into the rolling summary if thresholds met.

        Returns True if a new summary was written. Errors are logged and
        swallowed — the summary is best-effort and must never break the
        ask path.

        `floor_at` is the context's purge floor. When set, an existing
        summary written before the floor is treated as absent (so the
        new summary builds from post-floor turns only) and turns older
        than the floor are excluded from the input batch.

        `bot_id` (when set) scopes both the summary row and the source
        turns to one bot — multi-bot groups thus get one summary per
        bot, so bot A's compressed memory never leaks into bot B's
        prompt.
        """
        cfg = self._config()
        if not cfg["enabled"]:
            return False
        if not context_key:
            return False

        # Per-bot lock key so concurrent !ask invocations by different
        # bots in the same chat each get to summarize independently.
        lock_key = f"{context_key}::{bot_id or 0}"
        lock = self._lock_for(lock_key)
        if lock.locked():
            # Another summarization is already running for this (context,bot); skip.
            return False

        async with lock:
            try:
                return await self._do_summarize(context_key, cfg, floor_at, bot_id)
            except Exception as e:
                logger.warning(f"Summarizer failed for {context_key[:24]}: {e}")
                return False

    async def _do_summarize(
        self,
        context_key: str,
        cfg: dict,
        floor_at: Optional[float] = None,
        bot_id: Optional[int] = None,
    ) -> bool:
        existing = await self.history.get_summary(
            context_key, floor_at=floor_at, bot_id=bot_id,
        )
        through_id = existing["summary_through_id"] if existing else 0
        prior_summary = existing["summary"] if existing else ""
        if prior_summary:
            prior_value = parse_structured_summary(prior_summary, allow_legacy=True)
            prior_summary = serialize_structured_summary(
                prior_value or _empty_summary(), cfg["max_chars"],
            )

        pending = await self.history.turns_to_summarize(
            context_key,
            summary_through_id=through_id,
            keep_recent=cfg["keep_recent"],
            floor_at=floor_at,
            bot_id=bot_id,
        )
        if len(pending) < cfg["min_new_turns"]:
            return False

        formatted = [self._format_turn(t) for t in pending]
        formatted = [s for s in formatted if s]
        if not formatted:
            return False

        new_max_id = max(t["id"] for t in pending)

        user_payload_parts = []
        if prior_summary:
            user_payload_parts.append(f"CURRENT SUMMARY:\n{prior_summary}")
        else:
            user_payload_parts.append("CURRENT SUMMARY: (none — this is the first batch)")
        user_payload_parts.append(
            "NEW TURNS:\n" + "\n".join(formatted)
        )
        user_payload = "\n\n".join(user_payload_parts)

        system_prompt = SUMMARIZER_PROMPT.format(max_chars=cfg["max_chars"])

        overrides: dict = {
            "max_tokens": cfg["max_tokens"],
            "temperature": 0.3,
            # Force-disable thinking for this call regardless of the global
            # extra_body — the summary is short and doesn't benefit from CoT.
            "extra_body": json.dumps({"thinking": {"type": "disabled"}}),
        }
        if cfg["model"]:
            overrides["model"] = cfg["model"]

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_payload},
        ]

        try:
            assistant_msg = await self.llm.chat_messages(
                messages,
                overrides=overrides,
                suppress_response_style=True,
                purpose="summary",
            )
        except Exception as e:
            logger.warning(f"Summarizer LLM call failed for {context_key[:24]}: {e}")
            return False

        raw_summary = (assistant_msg.get("content") or "").strip()
        if not raw_summary:
            logger.info(f"Summarizer returned empty content for {context_key[:24]}")
            return False
        parsed_summary = parse_structured_summary(raw_summary, allow_legacy=False)
        if parsed_summary is None:
            logger.warning(
                "Summarizer returned invalid structured JSON for %s",
                context_key[:24],
            )
            return False
        new_summary = serialize_structured_summary(
            parsed_summary, cfg["max_chars"],
        )

        prior_count = existing["turns_summarized"] if existing else 0
        await self.history.upsert_summary(
            context_key=context_key,
            summary=new_summary,
            summary_through_id=new_max_id,
            turns_summarized=prior_count + len(pending),
            bot_id=bot_id,
        )
        logger.info(
            f"Summarizer: updated {context_key[:24]} (+{len(pending)} turns, "
            f"{len(new_summary)} chars)"
        )
        return True
