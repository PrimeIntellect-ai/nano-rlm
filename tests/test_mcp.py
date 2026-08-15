"""Tests for MCP-tools-as-skills (``rlm.mcp``).

Covers config parsing, JSON-schema → signature rendering, generation of importable skill
modules, and a live stdio transport round-trip. The streamable-HTTP path is exercised
end-to-end by the general-agent-v1 eval.
"""

from __future__ import annotations

import importlib
import inspect
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from mcp.types import Tool
import pytest

from rlm import mcp

SCHEMA = {
    "type": "object",
    "properties": {
        "day": {"type": "string"},
        "count": {"type": "integer"},
        "weird-name": {"type": "string"},
    },
    "required": ["day"],
}


def test_load_mcp_servers(monkeypatch):
    monkeypatch.delenv(mcp.MCP_CONFIG_ENV, raising=False)
    assert mcp.load_mcp_servers() == {}

    monkeypatch.setenv(
        mcp.MCP_CONFIG_ENV,
        '{"mcpServers": {"tools": {"url": "http://h/mcp"}, "web": {"url": "http://h/web", "headers": {"Authorization": "secret"}}, "local": {"command": "/bin/server", "args": ["--stdio"], "env": {"TOKEN": "secret"}}}}',
    )
    servers = mcp.load_mcp_servers()
    assert servers == {
        "tools": "http://h/mcp",
        "web": {
            "url": "http://h/web",
            "headers": {"Authorization": "secret"},
        },
        "local": {
            "command": "/bin/server",
            "args": ["--stdio"],
            "env": {"TOKEN": "secret"},
        },
    }
    assert json.loads(mcp.dump_mcp_servers(servers)) == {
        "mcpServers": {
            "tools": {"url": "http://h/mcp"},
            "web": {
                "url": "http://h/web",
                "headers": {"Authorization": "secret"},
            },
            "local": {
                "command": "/bin/server",
                "args": ["--stdio"],
                "env": {"TOKEN": "secret"},
            },
        }
    }


async def test_stdio_transport(monkeypatch, tmp_path):
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    caller_cwd = tmp_path / "caller"
    caller_cwd.mkdir()
    server = {
        "command": sys.executable,
        "args": [str(Path(__file__).parent / "fixtures" / "mcp_stdio.py")],
        "env": {"TEST_PREFIX": "stdio"},
    }
    found = await mcp.discover_tools({"local": server}, str(server_cwd))
    skills_dir = tmp_path / "skills"
    mcp.write_skill_modules(found, skills_dir, str(server_cwd))
    monkeypatch.setenv(mcp.MCP_CONFIG_ENV, mcp.dump_mcp_servers({"local": server}))
    monkeypatch.chdir(caller_cwd)

    assert list(found) == ["local_echo"]
    sys.path.insert(0, str(skills_dir))
    try:
        skill = importlib.import_module("local_echo")
        assert await skill.run(text="hello") == f"stdio:True:{server_cwd}:hello"
    finally:
        sys.path.remove(str(skills_dir))
        sys.modules.pop("local_echo", None)


def test_skill_name():
    assert mcp._skill_name("tools", "add_event") == "tools_add_event"
    assert mcp._skill_name("web", "search.run") == "web_search_run"
    assert mcp._skill_name("", "2fa") == "_2fa"


async def test_discover_tools_rejects_normalized_name_collision(monkeypatch):
    class FakeSession:
        async def initialize(self):
            pass

        async def list_tools(self):
            return SimpleNamespace(
                tools=[Tool(name="search", description="", inputSchema={})]
            )

    @asynccontextmanager
    async def fake_client_session(server, cwd=None):
        yield FakeSession()

    monkeypatch.setattr(mcp, "_client_session", fake_client_session)

    with pytest.raises(ValueError, match="MCP tool name collision.*a_b_search"):
        await mcp.discover_tools({"a-b": "one", "a_b": "two"})


def test_build_signature():
    params = mcp.build_signature(SCHEMA).parameters
    # non-identifier property is skipped; required comes before optional.
    assert list(params) == ["day", "count"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert params["day"].default is inspect.Parameter.empty
    assert params["day"].annotation is str
    assert params["count"].default is None
    assert params["count"].annotation is int


def test_write_skill_modules(tmp_path):
    tool = Tool(
        name="add_event", description="Add an event.\ndetails", inputSchema=SCHEMA
    )
    names = mcp.write_skill_modules({"tools_add_event": ("tools", tool)}, tmp_path)
    assert names == ["tools_add_event"]
    # the directory is the source of truth — the modules are readable back from it.
    assert mcp.list_skill_modules(tmp_path) == ["tools_add_event"]

    sys.path.insert(0, str(tmp_path))
    try:
        module = importlib.import_module("tools_add_event")
        assert inspect.iscoroutinefunction(module.run)
        assert str(inspect.signature(module.run)) == "(*, day: str, count: int = None)"
        assert module.run.__doc__ == "Add an event.\ndetails"
    finally:
        sys.path.remove(str(tmp_path))
        sys.modules.pop("tools_add_event", None)
