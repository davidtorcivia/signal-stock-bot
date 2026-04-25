"""
Adapter that exposes the bot's commands as OpenAI-format tools.

Each command registered in the dispatcher becomes a tool named
`bot__{command_name}` with a generic `{args: [string, ...]}` schema, so the
LLM can invoke any command it's allowed to use.

Policy enforcement:
  * Tool list is filtered by the caller's ContextPolicy (commands the
    context blocks are not advertised to the LLM in the first place).
  * Tool execution re-checks policy at call time (defence in depth) — if a
    policy change mid-conversation removes a command, the call fails.
"""

import logging
import shlex
from typing import Any, Optional

from .base import BaseCommand, CommandContext, CommandResult

logger = logging.getLogger(__name__)


# Commands that call the LLM themselves — would recurse if the LLM invoked
# them as a tool. Always excluded from the advertised tool list.
_RECURSIVE_COMMANDS = {"ask"}


class BotCommandTools:
    """Exposes bot commands to the LLM tool-call interface."""

    NAMESPACE = "bot"

    def __init__(self, dispatcher):
        self.dispatcher = dispatcher

    def list_tools(self, policy=None) -> list[dict]:
        """Return OpenAI-format tool schemas for every command the caller
        (via `policy`) is allowed to run."""
        out: list[dict] = []
        for cmd in self.dispatcher.get_commands():
            if cmd.name in _RECURSIVE_COMMANDS:
                continue
            if policy is not None and not policy.allows_command(cmd.name):
                continue
            out.append(self._tool_schema(cmd))
        return out

    @staticmethod
    def _tool_schema(cmd: BaseCommand) -> dict:
        desc = (cmd.description or cmd.name).strip()
        usage = (cmd.usage or "").strip()
        if usage:
            desc = f"{desc}\nUsage example: {usage}"
        return {
            "type": "function",
            "function": {
                "name": f"{BotCommandTools.NAMESPACE}__{cmd.name}",
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "args": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Positional arguments for the command, exactly as a "
                                "user would type them after the command word. "
                                "Example for !price: [\"AAPL\"]. Example for "
                                "!chart with flags: [\"AAPL\", \"1m\", \"-c\"]."
                            ),
                        },
                    },
                    "required": ["args"],
                },
            },
        }

    async def call(
        self,
        qualified_name: str,
        arguments: dict,
        caller_ctx: CommandContext,
    ) -> CommandResult:
        """Execute a bot-tool call initiated by the LLM.

        `caller_ctx` carries the sender/group/policy of the original !ask
        invocation — bot-tool calls are evaluated as if they were issued by
        the same user in the same chat.
        """
        if not qualified_name.startswith(self.NAMESPACE + "__"):
            raise ValueError(f"Not a bot tool: {qualified_name!r}")
        cmd_name = qualified_name[len(self.NAMESPACE) + 2 :]

        if cmd_name in _RECURSIVE_COMMANDS:
            return CommandResult.error(f"!{cmd_name} can't be called as a tool.")

        if caller_ctx.policy is not None and not caller_ctx.policy.allows_command(cmd_name):
            return CommandResult.error(
                f"Policy blocks !{cmd_name} in this context."
            )

        args = _coerce_args(arguments)
        raw_message = f"{self.dispatcher.prefix}{cmd_name} {' '.join(args)}".rstrip()

        logger.info(f"LLM invoked bot tool: !{cmd_name} args={args}")

        # Route through the dispatcher's normal execution path so policy,
        # augmentation, and audit logging still apply.
        return await self.dispatcher._execute_command(
            cmd_name,
            args,
            caller_ctx.sender,
            raw_message,
            caller_ctx.group_id,
            caller_ctx.policy,
        )


def _coerce_args(arguments: Any) -> list[str]:
    """Accept arguments from the LLM in a few common shapes."""
    if arguments is None:
        return []
    if isinstance(arguments, dict):
        raw = arguments.get("args", [])
    else:
        raw = arguments

    if isinstance(raw, list):
        return [str(x) for x in raw]
    if isinstance(raw, str):
        # LLM sometimes sends a single shell-style string instead of a list.
        try:
            return shlex.split(raw)
        except ValueError:
            return [raw]
    return [str(raw)]
