"""Context-admin MCP payload and catalog accounting."""

import json
from pathlib import Path

from jinja2 import Environment, FileSystemLoader

from src.admin.blueprint import _mcp_tool_stats, _tool_payload_stats
from src.contexts.policy import ContextPolicy, MODE_ALLOW_LIST
from src.llm.mcp_broker import MCP_BROKER_TOOLS
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


def test_total_payload_stats_include_native_and_compact_broker():
    native = {
        "type": "function",
        "function": {
            "name": "bot__price",
            "description": "Get price",
            "parameters": {"type": "object", "properties": {}},
        },
    }

    class _Ask:
        def _collect_tools(self, policy=None, bot=None):
            return [native, *MCP_BROKER_TOOLS]

    policy = ContextPolicy(
        id=1, kind="group", key="g",
        mcp_mode=MODE_ALLOW_LIST, mcp_servers=["web"],
    )
    stats = _tool_payload_stats(_Ask(), policy)
    assert stats["native_tools"] == 1
    assert stats["broker_tools"] == 2
    assert stats["total_tools"] == 3
    assert stats["total_chars"] == stats["native_chars"] + stats["broker_chars"]
    assert stats["broker_available_chars"] == stats["broker_chars"]
