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
print(await local_echo(text='hello'))
print(inspect.signature(local_echo))
matches = await rlm.tools.search('echo')
tool = await rlm.tools.get(matches[0]['id'])
print(tool.schema['properties']['text']['type'])
print(await tool('discovered'))
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
        f"stdio-secret:True:{server_cwd}:hello",
        "(text: str)",
        "string",
        f"stdio-secret:True:{server_cwd}:discovered",
        "True",
        "True",
    ]
    source = (session.dir / "local_echo.py").read_text()
    assert "stdio-secret" not in source
    assert str(Path(__file__).parent / "fixtures" / "mcp_stdio.py") not in source
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
matches = await rlm.tools.search('calendar events', server='catalog')
print(matches)
tool = await rlm.tools.get(matches[0]['id'])
print(inspect.signature(tool))
print(tool.schema['properties']['filter']['properties']['status']['enum'])
print(await tool(filter={'status': 'confirmed'}))
print(len(await rlm.tools.search('operation', limit=3)))
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
        "[{'name': 'catalog', 'tool_count': 2000}]",
        "False",
        "[{'id': 'catalog_calendarEvents', 'server': 'catalog', "
        "'description': 'Find calendar events by confirmation status.'}]",
        "(filter: dict)",
        "['confirmed']",
        '{"filter": {"status": "confirmed"}}',
        "3",
        "[]",
    ]
    assert not list(session.dir.glob("catalog_*.py"))
    prompt = client.calls[0]["messages"][0]["content"]
    assert "rlm.tools.search" in prompt
    assert '"tool_count": 2000' in prompt
    assert "catalog_calendarEvents" not in prompt
    assert "catalog_operation_" not in prompt
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["by_tool_python"] == {
        "catalog_calendarEvents": 1
    }


async def test_discovery_rejects_repeated_pagination_cursor():
    registry = mcp.MCPRegistry({"catalog": _catalog_server(REPEAT_CURSOR="1")})
    with pytest.raises(RuntimeError, match="discovery failed"):
        await asyncio.wait_for(registry.discover(), timeout=10)


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
                        {"code": "print(await local_echo(text='reused'))"},
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
