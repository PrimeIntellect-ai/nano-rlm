"""Tests for supervisor-owned MCP tools exposed as IPython skills."""

from __future__ import annotations

import asyncio
import inspect
import json
import os
import sys
from pathlib import Path

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result
from rlm import mcp
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.session import Session

SCHEMA = {
    "type": "object",
    "properties": {
        "day": {"type": "string"},
        "count": {"type": "integer"},
        "weird-name": {"type": "string"},
    },
    "required": ["day"],
}


def _config() -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url="http://interceptor", api_key="secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(),
    )


def _stdio_server(prefix: str = "stdio") -> dict:
    return {
        "command": sys.executable,
        "args": [str(Path(__file__).parent / "fixtures" / "mcp_stdio.py")],
        "env": {"TEST_PREFIX": prefix},
    }


def _blocking_stdio_server() -> dict:
    return {
        "command": sys.executable,
        "args": [str(Path(__file__).parent / "fixtures" / "mcp_blocking_stdio.py")],
        "env": {},
    }


def _process_exists(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def test_build_signature():
    params = mcp.build_signature(SCHEMA).parameters
    assert list(params) == ["day", "count"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert params["day"].default is inspect.Parameter.empty
    assert params["day"].annotation is str
    assert params["count"].default is None
    assert params["count"].annotation is int


async def test_real_kernel_uses_mcp_without_transport_secrets(tmp_path):
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    session = Session(tmp_path / "session")
    servers = {"local": _stdio_server("stdio-secret")}
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": """
import inspect, os, subprocess
print('local_echo' in globals())
matches = (await rlm.tools.search('echo'))['matches']
tool = await rlm.tools.get(matches[0]['id'])
print(await tool(text='hello'))
print(inspect.signature(tool))
print(tool.schema['properties']['text']['type'])
print(await tool('discovered'))
print('Returns text (str)' in tool.__doc__)
print('RLM_MCP_CONFIG' not in os.environ)
print('RLM_MCP_CONFIG=' not in subprocess.check_output(['env'], text=True))
"""
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
        mcp_servers=servers,
        cwd=str(server_cwd),
    )

    result = await engine.run("use the tool")

    assert result.answer == "done"
    assert tool_result(client).strip().splitlines() == [
        "False",
        f"stdio-secret:True:{server_cwd}:hello",
        "(text: str) -> str",
        "string",
        f"stdio-secret:True:{server_cwd}:discovered",
        "True",
        "True",
        "True",
    ]
    assert not (session.dir / "local_echo.py").exists()
    prompt = client.calls[0]["messages"][0]["content"]
    assert "local_echo" not in prompt
    assert "stdio-secret" not in prompt
    assert str(Path(__file__).parent / "fixtures" / "mcp_stdio.py") not in prompt
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["python_total"] == 2
    assert meta["programmatic_tool_call_stats"]["by_tool_python"] == {"local_echo": 2}


def _catalog_server(**env) -> dict:
    return {
        "command": sys.executable,
        "args": [str(Path(__file__).parent / "fixtures" / "mcp_catalog_stdio.py")],
        "env": env,
    }


async def test_large_paginated_catalog_discovery_in_real_kernel(tmp_path):
    session = Session(tmp_path / "session")
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": """
import inspect
print(await rlm.tools.servers())
print('catalog_calendarEvents' in globals())
matches = (await rlm.tools.search('calendar events', server='catalog'))['matches']
print(matches)
tool = await rlm.tools.get(matches[0]['id'])
print(inspect.signature(tool))
print(tool.schema['properties']['filter']['properties']['status']['enum'])
result = await tool(filter={'status': 'confirmed'})
print(result)
print(type(result).__name__, json.loads(result)['filter']['status'])
page = await rlm.tools.search('operation', limit=3, server='catalog')
print(len(page['matches']))
ids = [match['id'] for match in page['matches']]
while page['next_cursor'] is not None:
    page = await rlm.tools.search('operation', limit=20, server='catalog', cursor=page['next_cursor'])
    ids.extend(match['id'] for match in page['matches'])
print(len(ids), len(set(ids)), page['next_cursor'])
print(ids == sorted(ids))
print(await rlm.tools.search('no_matching_word'))
"""
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
        mcp_servers={"catalog": _catalog_server()},
        cwd=str(tmp_path),
    )
    result = await engine.run("find confirmed calendar events")
    assert result.answer == "done"
    assert tool_result(client).strip().splitlines() == [
        "[{'name': 'catalog', 'tool_count': 3000, "
        "'description': 'Find calendar events and catalog operations.'}]",
        "False",
        "[{'id': 'catalog_calendarEvents', 'server': 'catalog', "
        "'description': 'Find calendar events by confirmation status.'}]",
        "(filter: dict) -> str",
        "['confirmed']",
        '{"filter": {"status": "confirmed"}}',
        "str confirmed",
        "3",
        "2999 2999 None",
        "True",
        "{'matches': [], 'next_cursor': None}",
    ]
    assert not list(session.dir.glob("catalog_*.py"))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "rlm.tools.search" in prompt
    assert '"tool_count": 3000' in prompt
    assert "Find calendar events and catalog operations." in prompt
    assert "catalog_calendarEvents" not in prompt
    assert "catalog_operation_" not in prompt
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["by_tool_python"] == {
        "catalog_calendarEvents": 1
    }


async def test_discovery_rejects_invalid_or_oversized_catalogs(monkeypatch):
    registry = mcp.MCPRegistry({"catalog": _catalog_server(REPEAT_CURSOR="1")})
    with pytest.raises(RuntimeError, match="discovery failed"):
        await asyncio.wait_for(registry.discover(), timeout=10)
    with monkeypatch.context() as bounds:
        bounds.setattr(mcp, "MAX_MCP_DISCOVERY_PAGES", 3)
        registry = mcp.MCPRegistry(
            {"catalog": _catalog_server(ENDLESS_EMPTY_PAGES="1")}
        )
        with pytest.raises(ValueError, match="tool listing exceeded 3 pages"):
            await asyncio.wait_for(registry.discover(), timeout=10)
        assert registry._tools is None
    with monkeypatch.context() as bounds:
        bounds.setattr(mcp, "MAX_MCP_CATALOG_BYTES", 100)
        registry = mcp.MCPRegistry(
            {"catalog": _catalog_server(ENDLESS_EMPTY_PAGES="1", CURSOR_BYTES="60")}
        )
        with pytest.raises(ValueError, match="pagination metadata exceeds"):
            await asyncio.wait_for(registry.discover(), timeout=10)
        assert registry._tools is None
    monkeypatch.setattr(mcp, "MAX_MCP_CATALOG_BYTES", 1000)
    registry = mcp.MCPRegistry({"catalog": _catalog_server()})
    with pytest.raises(ValueError, match="catalog metadata exceeds"):
        await registry.discover()
    monkeypatch.setattr(mcp, "MAX_MCP_DESCRIPTOR_BYTES", 100)
    registry = mcp.MCPRegistry({"catalog": _catalog_server()})
    with pytest.raises(ValueError, match="public tool metadata exceeds"):
        await registry.discover()


async def test_search_cursor_is_bound_to_catalog_query_and_server(monkeypatch):
    converted = []
    descriptor_type = mcp.MCPToolDescriptor

    def convert(**fields):
        converted.append(fields["name"])
        return descriptor_type(**fields)

    monkeypatch.setattr(mcp, "MCPToolDescriptor", convert)
    registry = mcp.MCPRegistry({"catalog": _catalog_server()})
    await registry.discover()
    first = registry.search("operation", limit=3)
    cursor = first["next_cursor"]
    assert cursor is not None
    second = registry.search("operation", limit=2, cursor=cursor)
    assert len(second["matches"]) == 2
    assert not {match["id"] for match in first["matches"]} & {
        match["id"] for match in second["matches"]
    }
    assert second == registry.search("operation", limit=2, cursor=cursor)
    for query, server, invalid_cursor in [
        ("calendar events", None, cursor),
        ("operation", "catalog", cursor),
        ("operation", None, "invalid"),
        ("operation", None, "3:tampered"),
        ("operation", None, "3:\N{SNOWMAN}"),
    ]:
        with pytest.raises(ValueError, match="invalid MCP search cursor"):
            registry.search(query, server=server, cursor=invalid_cursor)
    other_catalog = mcp.MCPRegistry({"catalog": _catalog_server()})
    await other_catalog.discover()
    with pytest.raises(ValueError, match="invalid MCP search cursor"):
        other_catalog.search("operation", cursor=cursor)
    assert converted == []
    descriptor = registry.get("catalog_calendarEvents")
    assert registry.get("catalog_calendarEvents") is descriptor
    assert converted == ["catalog_calendarEvents"]
    assert descriptor.input_schema["properties"]["filter"]["properties"]["status"][
        "enum"
    ] == ["confirmed"]


async def test_cancelled_kernel_mcp_call_stops_server_and_session_reuses(tmp_path):
    marker = tmp_path / "started"
    session = Session(tmp_path / "session")
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": f"tool = await rlm.tools.get('local_wait'); await tool(marker={str(marker)!r})"
                        },
                    )
                ]
            ),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "tool = await rlm.tools.get('local_echo'); print(await tool(text='reused'))"
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
        mcp_servers={"local": _blocking_stdio_server()},
        cwd=str(tmp_path),
    )

    pending = asyncio.create_task(engine.prompt("wait"))
    for _ in range(100):
        if marker.exists():
            break
        await asyncio.sleep(0.05)
    assert marker.exists()
    server_pid = int(marker.read_text())
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=10)
    for _ in range(100):
        if not _process_exists(server_pid):
            break
        await asyncio.sleep(0.02)

    result = await engine.prompt("retry")
    await engine.aclose()

    assert _process_exists(server_pid) is False
    assert result.answer == "done"
    tool_messages = [
        message for message in client.calls[-1]["messages"] if message["role"] == "tool"
    ]
    assert tool_messages[-1]["content"].strip() == "reused"
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["python_total"] == 2
