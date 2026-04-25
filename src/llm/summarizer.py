"""
Rolling per-context conversation summary.

When a chat accumulates more turns than the rolling history window holds,
verbatim turns get pruned and the LLM loses long-term continuity. The
Summarizer compresses the older portion of each conversation into a
paragraph that lives separately from the turn-by-turn history and is
injected into the system prompt on subsequent calls.

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

logger = logging.getLogger(__name__)

DEFAULT_KEEP_RECENT = 12          # turns kept verbatim — same as history window
DEFAULT_MIN_NEW_TURNS = 10        # don't bother summarizing < this many new turns
DEFAULT_MAX_SUMMARY_CHARS = 1500  # hard ceiling on the stored summary
DEFAULT_MAX_TOKENS = 600          # output budget for the summarizer LLM call

SUMMARIZER_PROMPT = """\
You maintain a long-term memory of a Signal chat conversation.

You will receive (a) the current rolling summary, if any, and (b) a batch
of new conversation turns since the last summary update. Produce an
UPDATED rolling summary that folds the new material into the existing one.

Rules:
- Keep it under {max_chars} characters. Aggressively trim less-relevant
  context as the conversation evolves.
- Capture: who is in the conversation, what they care about, key facts
  they've shared (birth charts, watchlists, opinions, preferences),
  ongoing topics, unresolved threads, decisions or commitments.
- Drop: small talk, transient acknowledgements, command output details
  that aren't asked about again.
- Write in plain prose paragraphs. No headers, no bullet lists.
- Refer to people by their bracket label as it appears in the input
  (e.g. "David said…" not "user 4137 said…").
- If the new turns contradict an earlier fact, prefer the newer one and
  drop the older entry.
- Output ONLY the updated summary text. No preamble, no meta commentary,
  no "Updated summary:" prefix."""


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
            return f"[{label}] {content}"
        if role == "assistant":
            return f"[bot] {content}"
        if role == "tool":
            # Tool results are too noisy and too long for the summary
            # context. Skip them — the LLM's own assistant turn that
            # follows will reference whatever was useful.
            return ""
        return f"[{role}] {content}"

    async def maybe_summarize(self, context_key: str) -> bool:
        """Fold pending turns into the rolling summary if thresholds met.

        Returns True if a new summary was written. Errors are logged and
        swallowed — the summary is best-effort and must never break the
        ask path.
        """
        cfg = self._config()
        if not cfg["enabled"]:
            return False
        if not context_key:
            return False

        lock = self._lock_for(context_key)
        if lock.locked():
            # Another summarization is already running for this context; skip.
            return False

        async with lock:
            try:
                return await self._do_summarize(context_key, cfg)
            except Exception as e:
                logger.warning(f"Summarizer failed for {context_key[:24]}: {e}")
                return False

    async def _do_summarize(self, context_key: str, cfg: dict) -> bool:
        existing = await self.history.get_summary(context_key)
        through_id = existing["summary_through_id"] if existing else 0
        prior_summary = existing["summary"] if existing else ""

        pending = await self.history.turns_to_summarize(
            context_key,
            summary_through_id=through_id,
            keep_recent=cfg["keep_recent"],
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

        new_summary = (assistant_msg.get("content") or "").strip()
        if not new_summary:
            logger.info(f"Summarizer returned empty content for {context_key[:24]}")
            return False
        if len(new_summary) > cfg["max_chars"]:
            new_summary = new_summary[: cfg["max_chars"]].rstrip() + "…"

        prior_count = existing["turns_summarized"] if existing else 0
        await self.history.upsert_summary(
            context_key=context_key,
            summary=new_summary,
            summary_through_id=new_max_id,
            turns_summarized=prior_count + len(pending),
        )
        logger.info(
            f"Summarizer: updated {context_key[:24]} (+{len(pending)} turns, "
            f"{len(new_summary)} chars)"
        )
        return True
