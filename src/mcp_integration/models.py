"""Dataclasses for MCP server config and discovered tools."""

from dataclasses import dataclass, field
from typing import Any, Optional


TRANSPORTS = ("stdio", "sse", "http")


@dataclass
class MCPServerConfig:
    """Persisted MCP server configuration."""
    id: Optional[int]
    name: str
    transport: str                     # 'stdio' | 'sse' | 'http'
    enabled: bool
    # stdio fields
    command: Optional[str] = None      # binary / entry point, e.g. "npx"
    args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    # network fields (sse, http)
    url: Optional[str] = None
    headers: dict[str, str] = field(default_factory=dict)

    def validate(self) -> list[str]:
        errs: list[str] = []
        if not self.name.strip():
            errs.append("name is required")
        if self.transport not in TRANSPORTS:
            errs.append(f"transport must be one of {TRANSPORTS}")
        if self.transport == "stdio":
            if not (self.command or "").strip():
                errs.append("command is required for stdio transport")
        else:
            if not (self.url or "").strip():
                errs.append(f"url is required for {self.transport} transport")
        return errs


@dataclass
class MCPTool:
    """A tool exposed by an MCP server."""
    server_name: str
    name: str                          # bare tool name from server
    description: str
    input_schema: dict[str, Any]

    @property
    def qualified_name(self) -> str:
        """Namespaced `{server}__{tool}` name used in LLM tool calls."""
        return f"{self.server_name}__{self.name}"

    def to_openai_tool(self) -> dict[str, Any]:
        """Translate to OpenAI function-calling schema."""
        return {
            "type": "function",
            "function": {
                "name": self.qualified_name,
                "description": self.description or self.name,
                "parameters": self.input_schema or {"type": "object", "properties": {}},
            },
        }
