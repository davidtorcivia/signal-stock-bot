"""
PollVoter — when a Signal poll arrives, ask the LLM to pick option(s) and
cast the vote.

Polls land in `dataMessage.pollCreate = {question, allowMultiple, options[]}`
with `message = null`. The normal command pipeline can't reach them, so the
SignalHandler routes them here. We:

  1. Skip self-authored polls (don't vote on our own polls)
  2. Build a small LLM prompt: question + options + recent group context
     (so polls about chat regulars — "David's identity" — can use what the
     bot knows about them)
  3. Parse a JSON array of option indices from the response
  4. Cast the vote via `SignalHandler.send_poll_vote`

The LLM call is best-effort: failures are logged and the poll is skipped.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Optional

from ..admin.events import get_bus

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are voting in a Signal group-chat poll. You will be shown the poll \
question, the options indexed from 0, and recent group-chat context to \
inform your vote.

Pick the option(s) you'd actually vote for. Be opinionated — a vote that \
just picks the first option or splits everything is the worst kind of \
vote. If the poll is about a person you've seen in the group context, \
use what you know about them. If the poll is genuinely a tossup, pick \
the most defensible single option rather than waffling.

Reply with ONLY a JSON array of option indices (integers). For \
single-select polls return exactly one index, e.g. [2]. For multi-select \
polls return one or more, e.g. [0, 3]. No other text, no preamble, no \
explanation. Just the JSON array."""


_JSON_ARRAY_RE = re.compile(r"\[\s*(?:-?\d+\s*(?:,\s*-?\d+\s*)*)?\]")


class PollVoter:
    def __init__(
        self,
        *,
        llm_client,
        signal_handler,
        group_log=None,
        name_registry=None,
        bot_phone: str = "",
    ):
        self.llm = llm_client
        self.signal = signal_handler
        self.group_log = group_log
        self.name_registry = name_registry
        self.bot_phone = bot_phone

    def _label(self, phone: Optional[str]) -> str:
        if self.name_registry is None:
            return f"...{(phone or '')[-4:] or '????'}"
        return self.name_registry.label_for(phone)

    async def _build_context_block(self, group_id: str, ctx_count: int = 12) -> str:
        if not self.group_log or ctx_count <= 0:
            return ""
        try:
            msgs = await self.group_log.recent(group_id, limit=ctx_count)
        except Exception as e:
            logger.debug(f"PollVoter: group log read failed: {e}")
            return ""
        if not msgs:
            return ""
        lines = [
            "Recent group chat (oldest first; senders by name when known):"
        ]
        for m in msgs:
            label = self._label(m.get("sender"))
            text = (m.get("text") or "").replace("\n", " ").strip()
            if text:
                lines.append(f"  [{label}] {text}")
        return "\n".join(lines)

    @staticmethod
    def _parse_indices(content: str, n_options: int) -> list[int]:
        """Extract a list of valid option indices from the LLM reply.

        Tolerant of preamble: pulls the first JSON-array-of-ints in the
        text and discards anything outside [0, n_options). Returns [] when
        nothing usable is found.
        """
        if not content:
            return []
        match = _JSON_ARRAY_RE.search(content)
        if not match:
            return []
        try:
            raw = json.loads(match.group(0))
        except Exception:
            return []
        if not isinstance(raw, list):
            return []
        out: list[int] = []
        seen: set[int] = set()
        for v in raw:
            if not isinstance(v, int):
                continue
            if 0 <= v < n_options and v not in seen:
                out.append(v)
                seen.add(v)
        return out

    async def handle_poll(self, envelope: dict, data_message: dict) -> None:
        poll = data_message.get("pollCreate") or {}
        question = (poll.get("question") or "").strip()
        options = poll.get("options") or []
        allow_multiple = bool(poll.get("allowMultiple"))
        if not question or not options:
            return

        author_phone = envelope.get("sourceNumber") or envelope.get("source") or ""
        author_uuid = envelope.get("sourceUuid") or ""
        # Don't vote on our own polls.
        if author_phone and author_phone == self.bot_phone:
            logger.info("PollVoter: skipping self-authored poll")
            return

        group_info = data_message.get("groupInfo") or {}
        group_id = group_info.get("groupId") or ""
        poll_ts = data_message.get("timestamp") or envelope.get("timestamp") or 0
        if not group_id or not poll_ts:
            logger.info("PollVoter: missing group or timestamp, skipping")
            return

        # LLM readiness gate. Without an LLM we have no policy for picking,
        # and "always vote option 0" is worse than not voting.
        try:
            ready = self.llm.status().get("ready")
        except Exception:
            ready = False
        if not ready:
            logger.info("PollVoter: LLM not ready, skipping poll vote")
            return

        # Build prompt
        opts_text = "\n".join(f"  {i}: {o}" for i, o in enumerate(options))
        ctx_block = await self._build_context_block(group_id)
        select_kind = (
            "This is a multi-select poll — picking more than one is allowed."
            if allow_multiple
            else "This is a single-select poll — pick exactly one."
        )

        author_label = self._label(author_phone)
        user_parts = []
        if ctx_block:
            user_parts.append(ctx_block)
        user_parts.append(
            f"Poll posted by [{author_label}]:\n"
            f"Question: {question}\n"
            f"Options:\n{opts_text}\n\n{select_kind}"
        )
        user_content = "\n\n".join(user_parts)

        get_bus().publish(
            "reactor",
            decision="poll_seen",
            sender_tail=(author_phone or "")[-4:],
            group_id=group_id,
            text=f"poll: {question[:120]}",
        )

        # Thinking-mode models burn a lot of tokens reasoning before they
        # produce the JSON answer. 200 was way too low — the model would
        # truncate mid-thought and never emit indices. Bumped to 2000;
        # the actual answer is a tiny JSON array, so the cap exists only
        # to bound runaway reasoning.
        try:
            msg = await self.llm.chat_messages(
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": user_content},
                ],
                overrides={"max_tokens": 2000, "temperature": 0.7},
                suppress_response_style=True,
                purpose="poll_vote",
            )
        except Exception as e:
            logger.warning(f"PollVoter: LLM call failed: {e}")
            get_bus().publish(
                "reactor",
                decision="poll_error",
                sender_tail=(author_phone or "")[-4:],
                group_id=group_id,
                text=str(e)[:200],
            )
            return

        # Some providers (DeepSeek, OpenRouter aggregator) put the model's
        # answer in `reasoning_content` / `reasoning` and leave `content`
        # empty when thinking mode is on. Search all three fields — the
        # parser is tolerant of preamble, so giving it the reasoning text
        # works fine. Without this fallback, every poll would be skipped
        # whenever the main LLM has thinking enabled.
        candidates = [
            (msg.get("content") or "").strip(),
            (msg.get("reasoning_content") or "").strip(),
            (msg.get("reasoning") or "").strip(),
        ]
        indices: list[int] = []
        used_field = ""
        for field_text in candidates:
            if not field_text:
                continue
            indices = self._parse_indices(field_text, len(options))
            if indices:
                used_field = field_text
                break
        if not indices:
            logger.info(
                f"PollVoter: no valid indices parsed from any field "
                f"(content={candidates[0]!r}, reasoning={candidates[1] or candidates[2]!r}); "
                f"skipping vote"
            )
            get_bus().publish(
                "reactor",
                decision="poll_skip",
                sender_tail=(author_phone or "")[-4:],
                group_id=group_id,
                text="LLM returned no parsable indices",
            )
            return
        logger.debug(
            f"PollVoter: indices {indices} parsed from "
            f"{used_field[:120]!r}"
        )
        if not allow_multiple:
            indices = indices[:1]

        # Prefer voting via UUID when we have it — phone numbers can change,
        # UUIDs are stable. signal-cli accepts either as poll_author.
        author_id = author_uuid or author_phone
        ok = await self.signal.send_poll_vote(
            poll_author=author_id,
            poll_timestamp=poll_ts,
            selected_answers=indices,
            group_id=group_id,
        )
        chosen = ", ".join(f"{i}:{options[i]}" for i in indices)
        if ok:
            logger.info(
                f"PollVoter: voted on \"{question}\" → {chosen} "
                f"(by [{author_label}] in group {group_id[:12]})"
            )
            get_bus().publish(
                "reactor",
                decision="poll_voted",
                sender_tail=(author_phone or "")[-4:],
                group_id=group_id,
                text=f"voted {chosen}",
            )
        else:
            logger.warning(
                f"PollVoter: vote send failed for poll {poll_ts}"
            )
            get_bus().publish(
                "reactor",
                decision="poll_error",
                sender_tail=(author_phone or "")[-4:],
                group_id=group_id,
                text="signal-cli send_poll_vote failed",
            )
