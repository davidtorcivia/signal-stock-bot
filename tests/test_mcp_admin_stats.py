"""Context-admin MCP payload accounting."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.admin.blueprint import _mcp_tool_stats
from src.mcp_integration import MCPTool


class _Manager:
    def __init__(self, tools):
        self._tools = tools

    def all_tools(self):
        return self._tools


def _tool(server, name, description=""):
    return MCPTool(
        server_name=server,
        name=name,
        description=description,
        input_schema={
            "type": "object",
            "properties": {"query": {"type": "string"}},
        },
    )


def test_admin_stats_match_exact_openai_schema_encoding():
    tools = [
        _tool("web", "search", "Search the web"),
        _tool("web", "fetch", "Fetch one URL"),
        _tool("finance", "quote", "Get a quote"),
    ]
    stats = _mcp_tool_stats(_Manager(tools))

    expected_web_chars = sum(
        len(json.dumps(
            tool.to_openai_tool(),
            ensure_ascii=False,
            separators=(",", ":"),
        ))
        for tool in tools
        if tool.server_name == "web"
    )
    assert stats["web"] == {"tools": 2, "chars": expected_web_chars}
    assert stats["finance"]["tools"] == 1


def test_admin_stats_degrade_cleanly_without_manager():
    assert _mcp_tool_stats(None) == {}


def test_context_editor_template_compiles():
    templates = Path(__file__).parents[1] / "src" / "admin" / "templates"
    env = Environment(loader=FileSystemLoader(str(templates)))
    env.get_template("context_edit.html")
