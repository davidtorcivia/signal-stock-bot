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

All errors are logged and swallowed. The reactor must never affect the
command-handling path or surface diagnostics to users.
"""

import asyncio
import json
import logging
import time
from typing import Optional

from ..cache import get_metrics

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


DEFAULT_REACTOR_PROMPT = """\
You decide whether to react to messages in a Signal group chat with a single emoji.

React when the message:
- Expresses strong sentiment (excitement, frustration, win, loss)
- Shares a notable moment, milestone, or punchline
- Asks for acknowledgement (a "good morning", a confession, a check-in)
- Is an interesting link, source, or piece of information
- Is a good point that was made
- Is a question that can be answered with an emoji react

Do NOT react when:
- The message is short, transactional, or asks a question expecting a real reply
- It's about logistics, scheduling, or routine updates
- It's already being answered by the bot

Call the emoji_react tool with a SINGLE emoji that fits. Otherwise, don't call any tool."""


class EmojiReactor:
    def __init__(
        self,
        settings_store,
        llm_client,
        signal_handler,
        group_log=None,
        enricher=None,
    ):
        self.store = settings_store
        self.llm = llm_client
        self.signal = signal_handler
        self.group_log = group_log
        # Optional async callable: text -> expanded text. Used to inline tweet
        # content from x.com / twitter.com URLs so the reactor can decide on
        # the actual content rather than just an opaque link.
        self.enricher = enricher
        self._sender_last: dict[str, float] = {}
        self._group_last: dict[str, float] = {}

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
        }

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

    async def _build_user_content(
        self, sender: str, message: str, group_id: str, ctx_count: int
    ) -> str:
        sender_tail = (sender or "")[-4:] or "????"
        ctx_lines: list[str] = []
        if self.group_log is not None and ctx_count > 0:
            try:
                msgs = await self.group_log.recent(
                    group_id, limit=ctx_count, exclude_last=1
                )
                for m in msgs:
                    tail = (m["sender"] or "")[-4:] or "????"
                    text = (m["text"] or "").replace("\n", " ").strip()
                    if text:
                        ctx_lines.append(f"[...{tail}] {text}")
            except Exception as e:
                logger.debug(f"Reactor: failed to load group context: {e}")

        if ctx_lines:
            return (
                "Recent group chat (oldest first):\n"
                + "\n".join(ctx_lines)
                + "\n\nNew message to evaluate:\n"
                + f"[...{sender_tail}] {message}"
            )
        return f"New message to evaluate:\n[...{sender_tail}] {message}"

    async def maybe_react(
        self,
        *,
        sender: str,
        message: str,
        group_id: Optional[str],
        target_timestamp: Optional[int],
        policy=None,
    ) -> None:
        """Background task. Logs and swallows every error."""
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
                f"Reactor: evaluating ...{sender_tail} ({len(text)}c): {preview!r}"
            )

            try:
                assistant_msg = await self.llm.chat_messages(
                    messages,
                    tools=[REACT_TOOL],
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
                logger.info(f"Reactor: declined ...{sender_tail}")
                return

            # First call wins; subsequent ones ignored
            for call in tool_calls:
                fn = call.get("function") or {}
                if fn.get("name") != "emoji_react":
                    continue
                raw_args = fn.get("arguments") or "{}"
                try:
                    args = json.loads(raw_args) if isinstance(raw_args, str) else raw_args
                    emoji = (args.get("emoji") or "").strip() if isinstance(args, dict) else ""
                except Exception:
                    continue
                if not emoji:
                    continue

                self._record_cooldowns(sender, group_id)
                ok = await self.signal.send_reaction(
                    recipient=sender,
                    target_author=sender,
                    target_timestamp=int(target_timestamp),
                    emoji=emoji,
                    group_id=group_id,
                )
                if ok:
                    metrics.record_reactor_reaction(emoji)
                    logger.info(
                        f"Reactor: {emoji} on ...{(sender or '')[-4:]} "
                        f"({len(text)}-char msg)"
                    )
                else:
                    metrics.record_reactor_error()
                return  # exactly one reaction per inbound message

        except asyncio.CancelledError:
            raise
        except Exception as e:
            metrics.record_reactor_error()
            logger.error(f"Reactor unexpected error: {e}")
