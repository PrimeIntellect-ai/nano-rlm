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


def test_load_mcp_servers(monkeypatch):
    monkeypatch.delenv(mcp.MCP_CONFIG_ENV, raising=False)
    assert mcp.load_mcp_servers() == {}

    monkeypatch.setenv(
        mcp.MCP_CONFIG_ENV,
        '{"mcpServers": {"tools": {"url": "http://h/mcp"}, "web": {"url": "http://h/web", "headers": {"Authorization": "secret"}}, "local": {"command": "/bin/server", "args": ["--stdio"], "env": {"TOKEN": "secret"}}}}',
    )
    servers = mcp.load_mcp_servers()
    assert servers == {
        "tools": mcp.MCPHTTPServer(url="http://h/mcp"),
        "web": mcp.MCPHTTPServer(
            url="http://h/web",
            headers={"Authorization": "secret"},
        ),
        "local": mcp.MCPStdioServer(
            command="/bin/server",
            args=["--stdio"],
            env={"TOKEN": "secret"},
        ),
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

    monkeypatch.setenv(
        mcp.MCP_CONFIG_ENV,
        '{"mcpServers":{"bad":{"url":123,"command":"also-bad"}}}',
    )
    with pytest.raises(ValueError):
        mcp.load_mcp_servers()


def test_build_signature():
    params = mcp.build_signature(SCHEMA).parameters
    assert list(params) == ["day", "count"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert params["day"].default is inspect.Parameter.empty
    assert params["day"].annotation is str
    assert params["count"].default is None
    assert params["count"].annotation is int


async def test_real_kernel_uses_mcp_without_transport_secrets(monkeypatch, tmp_path):
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    session = Session(tmp_path / "session")
    servers = {"local": _stdio_server("stdio-secret")}
    monkeypatch.setenv(mcp.MCP_CONFIG_ENV, mcp.dump_mcp_servers(servers))
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
        cwd=str(server_cwd),
    )

    result = await engine.run("use the tool")

    assert result.answer == "done"
    assert tool_result(client).strip().splitlines() == [
        f"stdio-secret:True:{server_cwd}:hello",
        "(*, text: str)",
        "True",
        "True",
    ]
    source = (session.dir / "local_echo.py").read_text()
    assert "stdio-secret" not in source
    assert str(Path(__file__).parent / "fixtures" / "mcp_stdio.py") not in source
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["python_total"] == 1
    assert meta["programmatic_tool_call_stats"]["by_tool_python"] == {"local_echo": 1}


async def test_cancelled_kernel_mcp_call_stops_server_and_session_reuses(tmp_path):
    marker = tmp_path / "started"
    session = Session(tmp_path / "session")
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {"code": f"await local_wait(marker={str(marker)!r})"},
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
