"""
EmojiReactor — fire-and-forget LLM-decided message reactions.

For every inbound group message, the dispatcher kicks off a background
maybe_react() task. The reactor:

  1. Skips by cheap rules (cooldowns, min length, per-context disable)
  2. Calls the configured "reactor" LLM (a cheap/fast variant of the
     main model, with admin-supplied extra_body to disable thinking) and
     offers it exactly one tool: emoji_react(emoji)
  3. If the LLM calls the tool → POSTs a Signal reaction to the message
  4. If it doesn't call the tool → no reaction; user never sees anything

When natural-response is enabled (global + per-context), the same LLM call
also gets a `should_respond` tool. If invoked, the reactor hands off to
`implicit_response_handler` (typically the dispatcher) which runs !ask
spontaneously and sends the result.

All errors are logged and swallowed. The reactor must never affect the
command-handling path or surface diagnostics to users.
"""

import asyncio
import json
import logging
import time
from collections import deque
from typing import Awaitable, Callable, Optional

from ..admin.events import get_bus
from ..cache import get_metrics
from ..group_log import BOT_SENDER
from ..memory import (
    NOTE_MEMORY_TOOL,
    REACTOR_ALLOWED_KINDS,
    REACTOR_DEFAULT_CONFIDENCE,
    SOURCE_REACTOR,
    SubjectResolver,
)


# How many recent reactions to retain per group for the writing LLM to
# reference when users ask "why did you react with X?". Small, in-memory,
# wipes on restart — matches the natural conversational half-life of
# "what just happened in chat?" questions.
RECENT_REACTIONS_PER_GROUP = 20
RECENT_TARGET_SNIPPET_LEN = 120

logger = logging.getLogger(__name__)


REACT_TOOL = {
    "type": "function",
    "function": {
        "name": "emoji_react",
        "description": (
            "React to the user's message with a single emoji. "
            "Only call this when the message clearly warrants a reaction. "
            "If no reaction is appropriate, do not call any tool."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "emoji": {
                    "type": "string",
                    "description": "A single Unicode emoji.",
                }
            },
            "required": ["emoji"],
        },
    },
}


SHOULD_RESPOND_TOOL = {
    "type": "function",
    "function": {
        "name": "should_respond",
        "description": (
            "Trigger a full text reply to this message — used when the user "
            "is asking an open-ended question the bot can usefully answer, "
            "or is clearly continuing a conversation with the bot without "
            "explicitly addressing it. Do NOT call for banter, logistics, "
            "or messages aimed at a specific other person. Mutually "
            "exclusive with emoji_react: pick one."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": (
                        "One short sentence explaining why a real reply is "
                        "warranted (passed to the writer model as a hint)."
                    ),
                }
            },
            "required": ["reason"],
        },
    },
}


# Appended to the reactor system prompt only when the should_respond tool is
# actually exposed (per-context flag on, global flag on, cooldown clear).
NATURAL_RESPONSE_GUIDANCE = """\

You ALSO have a should_respond(reason) tool. Call it instead of emoji_react when:
- Someone asked an open-ended question to the group that you can usefully answer
- A user is clearly continuing a conversation with you (responding to your earlier
  reply) without addressing you by name, @mention, or quote-reply

Do NOT call should_respond when:
- The message is banter, chatter, logistics, or scheduling
- The message is addressed to a specific other person (not you)
- A simple emoji reaction fits better than a written reply
- Anyone in the chat is already mid-thread on the topic and the bot would interrupt

If you call should_respond, do not also call emoji_react. The writer model will
look at the full context and may still decide to stay silent — your job is just
to flag messages that plausibly warrant a reply."""


DEFAULT_REACTOR_PROMPT = """\
You decide whether to react to messages in a Signal group chat with a single emoji.

React when the message clearly carries:
- Strong sentiment (excitement, frustration, joy, grief, anger, awe)
- A notable moment, milestone, or punchline worth acknowledging
- A small social ritual that wants a nod ("good morning", a confession,
  a check-in)
- A question that can be honestly answered with a single emoji
- A specific emotion you can name and match — surprise, disgust, love,
  relief, schadenfreude, etc.

Do NOT react when:
- The message is short, transactional, or expects a written reply
- It's logistics, scheduling, or a routine update
- The bot is already answering it
- The only thing that fits is a generic "thinking" emoji. Reacting with
  🤔 to a shared link or tweet is a tell that you didn't have a real
  reaction — skip it.

Match the emoji to the actual content, not its category. A tweet about a
housing crash isn't 🤔, it's 🏚 or 😬 or 💀 depending on tone. A link to a
recipe isn't 🤔, it's 🍳 or 😋. If nothing specific fits, don't react —
silence is correct here far more often than a vague hum.

Call the emoji_react tool with a SINGLE emoji that fits. Otherwise, don't
call any tool."""


class EmojiReactor:
    def __init__(
        self,
        settings_store,
        llm_client,
        signal_handler,
        group_log=None,
        enricher=None,
        name_registry=None,
        memory_store=None,
    ):
        self.store = settings_store
        self.llm = llm_client
        self.signal = signal_handler
        self.group_log = group_log
        # Optional async callable: text -> expanded text. Used to inline tweet
        # content from x.com / twitter.com URLs so the reactor can decide on
        # the actual content rather than just an opaque link.
        self.enricher = enricher
        self.name_registry = name_registry
        # Optional MemoryStore — when set AND the per-context
        # reactor_memory_writes flag is on, the reactor's LLM call also
        # gets a note_memory tool so it can passively learn from messages
        # the main bot never sees.
        self.memory_store = memory_store
        self._subject_resolver: Optional[SubjectResolver] = None
        if name_registry is not None:
            self._subject_resolver = SubjectResolver(name_registry)
        self._sender_last: dict[str, float] = {}
        self._group_last: dict[str, float] = {}
        # Rolling per-group log of (timestamp, sender_label, target_snippet,
        # emoji) so the writing LLM can reference what it reacted to and why
        # when users ask. In-memory only; survives until process restart.
        self._recent: dict[str, deque] = {}
        # Per-group cooldown for natural-response (spontaneous text replies).
        # Kept separate from emoji cooldowns because writes are louder than
        # reactions and want a longer minimum gap.
        self._implicit_response_last: dict[str, float] = {}
        # Late-bound async handler invoked when the LLM calls should_respond.
        # Signature: (sender, message, group_id, target_timestamp, policy,
        # reason) -> Awaitable[None]. Wired in main.py to dispatcher.
        self.implicit_response_handler: Optional[
            Callable[..., Awaitable[None]]
        ] = None

    def _sender_label(self, phone: str) -> str:
        if phone == BOT_SENDER:
            return (
                self.name_registry.bot_name
                if self.name_registry is not None
                else "Bot"
            )
        if self.name_registry is None:
            return f"...{(phone or '')[-4:] or '????'}"
        return self.name_registry.label_for(phone)

    def _config(self) -> dict:
        store = self.store
        return {
            "enabled": bool(store.get("reactor_enabled", False)),
            "model": (store.get("reactor_model") or "").strip() or None,
            "max_tokens": int(store.get("reactor_max_tokens") or 50),
            "temperature": float(store.get("reactor_temperature") or 0.3),
            "extra_body": store.get("reactor_extra_body") or "",
            "system_prompt": store.get("reactor_system_prompt") or DEFAULT_REACTOR_PROMPT,
            "min_length": int(store.get("reactor_min_length") or 0),
            "sender_cooldown": int(store.get("reactor_sender_cooldown") or 30),
            "group_cooldown": int(store.get("reactor_group_cooldown") or 10),
            "context_messages": int(store.get("reactor_context_messages") or 5),
            "natural_response_enabled": bool(
                store.get("natural_response_enabled", False)
            ),
            "natural_response_cooldown": int(
                store.get("natural_response_cooldown") or 300
            ),
            "natural_response_extra_prompt": (
                store.get("natural_response_extra_prompt") or ""
            ),
        }

    def _natural_response_active(self, group_id: str, cfg: dict, policy) -> bool:
        """Decide whether to expose should_respond for this evaluation.

        Three gates: global flag, per-context flag, and the per-group
        cooldown since the last spontaneous reply. Mentions/quotes route
        through the dispatcher's normal path and don't touch this cooldown.
        """
        if not cfg["natural_response_enabled"]:
            return False
        if policy is None or not getattr(policy, "natural_response", False):
            return False
        if self.implicit_response_handler is None:
            return False
        last = self._implicit_response_last.get(group_id, 0.0)
        if time.time() - last < cfg["natural_response_cooldown"]:
            return False
        return True

    def _memory_writes_active(self, policy) -> bool:
        """Decide whether to expose `note_memory` for this evaluation.

        Two gates: a wired MemoryStore + subject resolver, and the
        per-context `reactor_memory_writes` flag on a real (non-default)
        policy row. Default rows are excluded so writes don't pool
        across unregistered chats.
        """
        if self.memory_store is None or self._subject_resolver is None:
            return False
        if policy is None:
            return False
        if getattr(policy, "id", None) is None:
            return False
        if getattr(policy, "kind", None) == "default":
            return False
        return bool(getattr(policy, "reactor_memory_writes", False))

    async def _persist_note_memory(
        self,
        *,
        policy,
        sender: str,
        target_timestamp: Optional[int],
        args: dict,
    ) -> None:
        """Write a single reactor-sourced memory. Errors are logged + swallowed."""
        store = self.memory_store
        resolver = self._subject_resolver
        if store is None or resolver is None or policy is None:
            return
        try:
            subject_hint = (args.get("subject") or "").strip()
            kind = (args.get("kind") or "").strip().lower()
            content = (args.get("content") or "").strip()
            if not subject_hint or not content:
                return
            if kind not in REACTOR_ALLOWED_KINDS:
                logger.debug(
                    f"Reactor note_memory: skipping disallowed kind {kind!r}"
                )
                return
            key, label = resolver.resolve(
                subject_hint, sender_phone=sender
            )
            if not key:
                return

            # Cross-kind pre-write skip: if a near-duplicate memory is
            # already stored for this subject under ANY kind, don't write
            # a new row. The strict same-kind corroboration in
            # MemoryStore.add() (Jaccard ≥ 0.85) catches close rephrasings
            # within the SAME kind, but the reactor commonly writes the
            # same fact under different kinds ("fact: loves astrology"
            # vs. "preference: is into astrology") and those don't share
            # the WHERE clause there, so they slip through as duplicates.
            # Looser threshold here is the right knob for a passive,
            # low-confidence learning loop.
            try:
                existing = await store.find_similar_for_subject(
                    context_id=policy.id,
                    subject_key=key,
                    content=content,
                )
            except Exception as e:
                logger.debug(f"Reactor: dedup lookup failed: {e}")
                existing = None
            if existing is not None:
                logger.info(
                    f"Reactor: skipping duplicate memory for {label!r}: "
                    f"\"{content[:60]}\" already covered by "
                    f"#{existing['id']} [{existing['kind']}]: "
                    f"\"{(existing.get('content') or '')[:60]}\""
                )
                return

            from ..database import hash_phone
            sender_hash = hash_phone(sender) if sender else ""
            # target_timestamp is Signal's millisecond timestamp; normalize
            # to seconds so it lines up with conversation_turns.created_at
            # for cross-table audit lookups.
            msg_at = (
                float(target_timestamp) / 1000.0
                if target_timestamp else None
            )
            mem_id = await store.add(
                context_id=policy.id,
                subject_key=key,
                subject_label=label,
                kind=kind,
                content=content,
                confidence=REACTOR_DEFAULT_CONFIDENCE,
                source=SOURCE_REACTOR,
                source_user_hash=sender_hash,
                source_message_at=msg_at,
            )
            if mem_id is not None:
                logger.info(
                    f"Reactor: noted memory #{mem_id} "
                    f"[{kind}] about {label!r}: {content[:80]!r}"
                )
        except Exception as e:
            logger.debug(f"Reactor note_memory persist failed: {e}")

    def mark_implicit_response(self, group_id: str) -> None:
        """Record that a spontaneous reply just fired (or is about to).

        Called early — before ask_command runs — so concurrent reactor
        evaluations see the cooldown advanced and don't double-fire.
        """
        self._implicit_response_last[group_id] = time.time()

    def _within_cooldown(self, sender: str, group_id: str, cfg: dict) -> bool:
        now = time.time()
        if now - self._sender_last.get(sender, 0) < cfg["sender_cooldown"]:
            return True
        if now - self._group_last.get(group_id, 0) < cfg["group_cooldown"]:
            return True
        return False

    def _record_cooldowns(self, sender: str, group_id: str) -> None:
        now = time.time()
        self._sender_last[sender] = now
        self._group_last[group_id] = now

    def _record_recent(
        self, *, group_id: str, sender_label: str, target_text: str, emoji: str
    ) -> None:
        snippet = (target_text or "").replace("\n", " ").strip()
        if len(snippet) > RECENT_TARGET_SNIPPET_LEN:
            snippet = snippet[: RECENT_TARGET_SNIPPET_LEN - 1].rstrip() + "…"
        log = self._recent.get(group_id)
        if log is None:
            log = deque(maxlen=RECENT_REACTIONS_PER_GROUP)
            self._recent[group_id] = log
        log.append((time.time(), sender_label, snippet, emoji))

    def recent_reactions(self, group_id: str, limit: int = 5) -> list[dict]:
        """Return the most recent reactions in `group_id`, newest-first.

        Used by the writing LLM so it can answer "why did you react with X?"
        without confabulating. Empty list when nothing is logged for the
        group (including: feature off, restart-fresh, or no qualifying
        messages yet).
        """
        log = self._recent.get(group_id)
        if not log:
            return []
        items = list(log)[-max(1, limit):]
        items.reverse()
        return [
            {"ts": ts, "sender": sender, "target": target, "emoji": emoji}
            for ts, sender, target, emoji in items
        ]

    async def _build_user_content(
        self, sender: str, message: str, group_id: str, ctx_count: int
    ) -> str:
        sender_label = self._sender_label(sender)
        ctx_lines: list[str] = []
        if self.group_log is not None and ctx_count > 0:
            try:
                msgs = await self.group_log.recent(
                    group_id, limit=ctx_count, exclude_last=1
                )
                for m in msgs:
                    label = self._sender_label(m["sender"])
                    text = (m["text"] or "").replace("\n", " ").strip()
                    if text:
                        ctx_lines.append(f"[{label}] {text}")
            except Exception as e:
                logger.debug(f"Reactor: failed to load group context: {e}")

        # Make it explicit that lines beginning with `→` are link content
        # that the bot inlined as context — NOT a bot response. Otherwise
        # the "do not react when the bot is already answering" rule from
        # most reactor prompts kicks in incorrectly on tweet/URL messages.
        format_note = (
            "(Format note: lines starting with `→` are link content the bot "
            "inlined for context — they're part of the user's message, not "
            "a bot reply.)"
        )
        if ctx_lines:
            return (
                "Recent group chat (oldest first):\n"
                + "\n".join(ctx_lines)
                + f"\n\n{format_note}\n\nNew message to evaluate:\n"
                + f"[{sender_label}] {message}"
            )
        return (
            f"{format_note}\n\nNew message to evaluate:\n"
            f"[{sender_label}] {message}"
        )

    async def maybe_react(
        self,
        *,
        sender: str,
        message: str,
        group_id: Optional[str],
        target_timestamp: Optional[int],
        policy=None,
        bot_will_reply: bool = False,
    ) -> None:
        """Background task. Logs and swallows every error.

        `bot_will_reply` is set by the dispatcher when it can already tell the
        message will produce a reply (mention or prefixed command). The
        reactor still emoji-evaluates, but suppresses the should_respond tool
        so we don't fire a duplicate spontaneous reply on top of the explicit
        one.
        """
        metrics = get_metrics()
        try:
            # Groups only (per design); DMs explicitly excluded for now.
            if not group_id or not target_timestamp or not message:
                return

            cfg = self._config()
            if not cfg["enabled"]:
                metrics.record_reactor_skip("disabled")
                return

            if policy is not None and not getattr(policy, "reactor_enabled", True):
                metrics.record_reactor_skip("disabled")
                return

            text = message.strip()
            if cfg["min_length"] and len(text) < cfg["min_length"]:
                metrics.record_reactor_skip("short")
                return

            if self._within_cooldown(sender, group_id, cfg):
                metrics.record_reactor_skip("cooldown")
                return

            # Inline-expand tweet/X URLs so the reactor sees actual content
            # rather than an opaque link. Failures are non-fatal — fall back
            # to the raw text.
            if self.enricher is not None:
                try:
                    expanded = await self.enricher.expand(text)
                    if expanded:
                        text = expanded
                except Exception as e:
                    logger.debug(f"Reactor: link enrichment failed: {e}")

            metrics.record_reactor_evaluation()

            # Per-context prompt override wins over the global reactor prompt
            system_prompt = cfg["system_prompt"]
            if policy is not None:
                ctx_prompt = getattr(policy, "reactor_prompt", None)
                if ctx_prompt:
                    system_prompt = ctx_prompt

            # Natural-response gating: when active, expose the second tool and
            # append guidance describing when to use it. Off by default; both
            # the global flag and per-context flag must be on. Suppressed when
            # the dispatcher already knows it will reply (mention/command).
            tools = [REACT_TOOL]
            offer_should_respond = (
                not bot_will_reply
                and self._natural_response_active(group_id, cfg, policy)
            )
            if offer_should_respond:
                tools.append(SHOULD_RESPOND_TOOL)
                extra = cfg["natural_response_extra_prompt"].strip()
                system_prompt = (
                    f"{system_prompt}\n{extra or NATURAL_RESPONSE_GUIDANCE}"
                )
            # Passive memory writes: the reactor already pays for an LLM
            # call on every qualifying message, so memory extraction is
            # essentially free. Gated per-context (off by default — opt-in
            # in /admin/contexts) AND only when the policy belongs to a
            # real, non-default row (so writes don't bleed across the
            # default group/dm rows).
            offer_note_memory = self._memory_writes_active(policy)
            if offer_note_memory:
                tools.append(NOTE_MEMORY_TOOL)

            user_content = await self._build_user_content(
                sender, text, group_id, cfg["context_messages"]
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_content},
            ]

            overrides: dict = {
                "max_tokens": cfg["max_tokens"],
                "temperature": cfg["temperature"],
            }
            if cfg["model"]:
                overrides["model"] = cfg["model"]
            if cfg["extra_body"]:
                overrides["extra_body"] = cfg["extra_body"]

            sender_tail = (sender or "")[-4:] or "????"
            preview = text.replace("\n", " ")[:60]
            logger.info(
                f"Reactor: evaluating ...{sender_tail} ({len(text)}c"
                f"{', +respond' if offer_should_respond else ''}): {preview!r}"
            )

            try:
                assistant_msg = await self.llm.chat_messages(
                    messages,
                    tools=tools,
                    overrides=overrides,
                    suppress_response_style=True,
                    purpose="reactor",
                )
            except Exception as e:
                metrics.record_reactor_error()
                logger.warning(f"Reactor LLM call failed for ...{sender_tail}: {e}")
                return

            tool_calls = assistant_msg.get("tool_calls") or []
            if not tool_calls:
                metrics.record_reactor_skip("no_tool")
                # Capture text from wherever the model put it. Different
                # providers stash reasoning in different fields:
                #   - OpenAI/Anthropic style → "content"
                #   - DeepSeek → "reasoning_content" alongside "content"
                #   - OpenRouter aggregator → sometimes "reasoning"
                # Pull from each known location so we surface SOMETHING.
                refusal = (assistant_msg.get("content") or "").strip()
                reasoning = (
                    assistant_msg.get("reasoning")
                    or assistant_msg.get("reasoning_content")
                    or ""
                ).strip()
                # Show both fields when both present, with reasoning fully
                # untruncated — when the model's thinking pivots to "no" we
                # need to see the pivot, not just the lead-up.
                if refusal and reasoning:
                    logger.info(
                        f"Reactor: declined ...{sender_tail} — "
                        f"reasoning: {reasoning.replace(chr(10), ' ')!r} "
                        f"| content: {refusal.replace(chr(10), ' ')[:300]!r}"
                    )
                    shown_preview = (refusal or reasoning).replace("\n", " ")[:300]
                elif reasoning:
                    logger.info(
                        f"Reactor: declined ...{sender_tail} — "
                        f"reasoning: {reasoning.replace(chr(10), ' ')!r}"
                    )
                    shown_preview = reasoning.replace("\n", " ")[:300]
                elif refusal:
                    logger.info(
                        f"Reactor: declined ...{sender_tail} — "
                        f"content: {refusal.replace(chr(10), ' ')!r}"
                    )
                    shown_preview = refusal.replace("\n", " ")[:300]
                else:
                    shown_preview = ""
                if not shown_preview:
                    # Truly nothing — log the full assistant_msg keys so we
                    # know if a future provider invents a new field.
                    logger.info(
                        f"Reactor: declined ...{sender_tail} (no text); "
                        f"msg keys={sorted(assistant_msg.keys())}"
                    )
                get_bus().publish(
                    "reactor",
                    decision="decline",
                    sender_tail=sender_tail,
                    group_id=group_id,
                    text=shown_preview or None,
                )
                return

            # should_respond wins over emoji_react when both are present —
            # a real reply already conveys whatever a reaction would.
            # note_memory is collected separately because it's not mutually
            # exclusive with either: the reactor can react and note in the
            # same call, or just note silently.
            respond_reason: Optional[str] = None
            emoji_pick: Optional[str] = None
            memory_notes: list[dict] = []
            for call in tool_calls:
                fn = call.get("function") or {}
                fname = fn.get("name")
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = (
                        json.loads(raw_args)
                        if isinstance(raw_args, str)
                        else raw_args
                    )
                except Exception:
                    continue
                if not isinstance(args, dict):
                    continue
                if fname == "should_respond" and respond_reason is None:
                    respond_reason = (args.get("reason") or "").strip() or "(no reason given)"
                elif fname == "emoji_react" and emoji_pick is None:
                    emoji_pick = (args.get("emoji") or "").strip()
                elif fname == "note_memory":
                    memory_notes.append(args)

            # Persist any memory notes regardless of which reactor outcome
            # wins below — passive learning is independent of the public
            # reaction. Only fires when the per-context flag was on at
            # offer-time.
            if memory_notes and offer_note_memory:
                for note in memory_notes:
                    await self._persist_note_memory(
                        policy=policy,
                        sender=sender,
                        target_timestamp=target_timestamp,
                        args=note,
                    )

            if respond_reason and offer_should_respond:
                # Mark cooldown first so concurrent reactor calls in the same
                # group see it advanced and don't pile on. Cooldown stands
                # even if the writer model bails — one decision per window.
                self.mark_implicit_response(group_id)
                metrics.record_reactor_response()
                logger.info(
                    f"Reactor: triggered should_respond on ...{sender_tail} "
                    f"— {respond_reason!r}"
                )
                get_bus().publish(
                    "reactor",
                    decision="respond",
                    sender_tail=sender_tail,
                    group_id=group_id,
                    text=respond_reason[:300],
                )
                handler = self.implicit_response_handler
                if handler is not None:
                    try:
                        await handler(
                            sender=sender,
                            message=message,
                            group_id=group_id,
                            policy=policy,
                            reason=respond_reason,
                        )
                    except Exception as e:
                        logger.exception(
                            f"Implicit response handler failed: {e}"
                        )
                return

            if emoji_pick:
                self._record_cooldowns(sender, group_id)
                ok = await self.signal.send_reaction(
                    recipient=sender,
                    target_author=sender,
                    target_timestamp=int(target_timestamp),
                    emoji=emoji_pick,
                    group_id=group_id,
                )
                if ok:
                    metrics.record_reactor_reaction(emoji_pick)
                    logger.info(
                        f"Reactor: {emoji_pick} on ...{(sender or '')[-4:]} "
                        f"({len(text)}-char msg)"
                    )
                    self._record_recent(
                        group_id=group_id,
                        sender_label=self._sender_label(sender),
                        target_text=text,
                        emoji=emoji_pick,
                    )
                    get_bus().publish(
                        "reactor",
                        decision="react",
                        emoji=emoji_pick,
                        sender_tail=sender_tail,
                        group_id=group_id,
                    )
                else:
                    metrics.record_reactor_error()
                return

        except asyncio.CancelledError:
            raise
        except Exception as e:
            metrics.record_reactor_error()
            logger.error(f"Reactor unexpected error: {e}")
