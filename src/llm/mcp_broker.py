"""Stable, compact broker schemas for MCP discovery and invocation."""

from __future__ import annotations

import json
import re
from typing import Any, Optional


MCP_DISCOVER_NAME = "mcp__discover"
MCP_INVOKE_NAME = "mcp__invoke"

MCP_DISCOVER_TOOL = {
    "type": "function",
    "function": {
        "name": MCP_DISCOVER_NAME,
        "description": (
            "Search the allowed MCP tool catalog. Returns exact qualified tool "
            "names, descriptions, and JSON argument schemas for a small set of "
            "matches. Use this before mcp__invoke when external data or an MCP "
            "capability may help."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Capability or task to search for, such as web search or stock quote.",
                },
                "server": {
                    "type": "string",
                    "description": "Optional exact MCP server name to narrow the catalog.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 10,
                    "description": "Maximum matching schemas to return. Defaults to 5.",
                },
            },
        },
    },
}

MCP_INVOKE_TOOL = {
    "type": "function",
    "function": {
        "name": MCP_INVOKE_NAME,
        "description": (
            "Invoke one allowed MCP tool by the exact qualified name returned "
            "by mcp__discover. Pass an arguments object matching the discovered "
            "JSON schema."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Exact qualified tool name, for example brave-search__brave_web_search.",
                },
                "arguments": {
                    "type": "object",
                    "description": "Arguments matching the schema returned by mcp__discover.",
                    "additionalProperties": True,
                },
            },
            "required": ["name", "arguments"],
        },
    },
}

MCP_BROKER_TOOLS = (MCP_DISCOVER_TOOL, MCP_INVOKE_TOOL)


def allowed_mcp_tools(mcp_manager, policy=None) -> list:
    if mcp_manager is None:
        return []
    tools = list(mcp_manager.all_tools())
    if policy is not None:
        tools = [tool for tool in tools if policy.allows_mcp(tool.server_name)]
    return sorted(tools, key=lambda tool: tool.qualified_name)


def broker_should_be_exposed(mcp_manager, policy=None) -> bool:
    """Keep broker presence stable across MCP process restarts.

    An empty allow-list is an explicit "no MCP" policy.  Other modes expose
    the fixed broker pair even when an allowed server is temporarily down, so
    session health cannot rewrite the provider's cached tool prefix.
    """

    if mcp_manager is None:
        return False
    if policy is None:
        return True
    mode = str(getattr(policy, "mcp_mode", "allow_all") or "allow_all")
    if mode == "allow_list":
        return bool(getattr(policy, "mcp_servers", None) or [])
    return True


def _terms(query: str) -> list[str]:
    return [part for part in re.split(r"[^a-z0-9]+", query.lower()) if part]


def discover_mcp_tools(
    mcp_manager,
    policy,
    *,
    query: str = "",
    server: str = "",
    limit: int = 5,
) -> str:
    tools = allowed_mcp_tools(mcp_manager, policy)
    server = str(server or "").strip()
    if server:
        tools = [tool for tool in tools if tool.server_name == server]
    terms = _terms(str(query or ""))

    ranked: list[tuple[int, str, Any]] = []
    for tool in tools:
        name = tool.qualified_name.lower()
        description = (tool.description or "").lower()
        haystack = f"{name} {description}"
        if terms and not any(term in haystack for term in terms):
            continue
        score = sum(5 for term in terms if term in name)
        score += sum(1 for term in terms if term in description)
        ranked.append((-score, tool.qualified_name, tool))
    ranked.sort(key=lambda row: (row[0], row[1]))

    try:
        cap = max(1, min(10, int(limit)))
    except (TypeError, ValueError):
        cap = 5
    selected = [row[2] for row in ranked[:cap]]
    result = {
        "query": str(query or ""),
        "server": server or None,
        "allowed_tools": len(tools),
        "returned": len(selected),
        "matches": [
            {
                "name": tool.qualified_name,
                "server": tool.server_name,
                "description": tool.description or tool.name,
                "parameters": tool.input_schema or {
                    "type": "object",
                    "properties": {},
                },
            }
            for tool in selected
        ],
    }
    if not selected:
        result["available_servers"] = sorted({tool.server_name for tool in tools})
        result["hint"] = "Try broader capability words or omit server."
    return json.dumps(result, ensure_ascii=False, separators=(",", ":"))


async def invoke_mcp_tool(
    mcp_manager,
    policy,
    *,
    name: str,
    arguments: Optional[dict],
) -> str:
    qualified = str(name or "").strip()
    if not qualified:
        return "ERROR: mcp__invoke requires a tool name from mcp__discover"
    if not isinstance(arguments, dict):
        return "ERROR: mcp__invoke arguments must be a JSON object"
    allowed = {tool.qualified_name for tool in allowed_mcp_tools(mcp_manager, policy)}
    if qualified not in allowed:
        return (
            f"ERROR: MCP tool {qualified!r} is unavailable or blocked in this "
            "chat. Call mcp__discover again."
        )
    return await mcp_manager.call_tool(qualified, arguments)
