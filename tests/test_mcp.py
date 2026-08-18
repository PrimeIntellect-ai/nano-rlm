"""Tests for supervisor-owned MCP tools exposed as IPython skills."""

from __future__ import annotations

import asyncio
import importlib
import inspect
import json
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from mcp.types import Tool

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result
from rlm import broker, mcp
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor

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
        provider=ProviderConfig("http://interceptor", "secret"),
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


async def test_registry_keeps_transport_private_and_calls_stdio(tmp_path):
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    registry = mcp.MCPRegistry(
        {"local": _stdio_server("stdio-secret")}, str(server_cwd)
    )
    descriptors = await registry.discover()

    assert len(descriptors) == 1
    descriptor = descriptors[0]
    assert descriptor.name == "local_echo"
    assert descriptor.input_schema["required"] == ["text"]
    assert await registry.call(descriptor.capability, {"text": "hello"}) == (
        f"stdio-secret:True:{server_cwd}:hello"
    )
    with pytest.raises(PermissionError, match="unknown MCP tool capability"):
        registry.descriptor("invalid")

    skills_dir = tmp_path / "skills"
    assert registry.write_skill_modules(skills_dir) == ["local_echo"]
    source = (skills_dir / "local_echo.py").read_text()
    for private_value in (
        "stdio-secret",
        str(server_cwd),
        sys.executable,
        "mcp_stdio.py",
        "TEST_PREFIX",
    ):
        assert private_value not in source

    sys.path.insert(0, str(skills_dir))
    try:
        skill = importlib.import_module("local_echo")
        assert inspect.iscoroutinefunction(skill.run)
        assert str(inspect.signature(skill.run)) == "(*, text: str)"
        assert skill.__rlm_brokered__ is True
    finally:
        sys.path.remove(str(skills_dir))
        sys.modules.pop("local_echo", None)


async def test_registry_rejects_normalized_and_existing_name_collisions(
    monkeypatch, tmp_path
):
    tool = Tool(name="echo", description="Echo.", inputSchema=SCHEMA)

    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=[tool])

    @asynccontextmanager
    async def fake_client_session(server, cwd=None):
        yield FakeSession()

    monkeypatch.setattr(mcp, "_client_session", fake_client_session)
    duplicate = mcp.MCPRegistry({"a-b": "one", "a_b": "two"})
    with pytest.raises(ValueError, match="duplicate normalized MCP tool name"):
        await duplicate.discover()

    registry = mcp.MCPRegistry({"local": "one"})
    await registry.discover()
    with pytest.raises(ValueError, match="conflict with existing skills"):
        registry.write_skill_modules(tmp_path, {"local_echo"})


async def test_registry_redacts_discovery_errors_and_bounds_metadata(monkeypatch):
    sentinel = "https://secret.example/mcp?token=do-not-leak"

    @asynccontextmanager
    async def failing_client_session(server, cwd=None):
        raise RuntimeError(sentinel)
        yield

    monkeypatch.setattr(mcp, "_client_session", failing_client_session)
    with pytest.raises(
        RuntimeError, match="MCP server 'public' discovery failed"
    ) as exc:
        await mcp.MCPRegistry({"public": "ignored"}).discover()
    assert sentinel not in str(exc.value)

    tools = [
        Tool(name="one", description="One.", inputSchema=SCHEMA),
        Tool(name="two", description="Two.", inputSchema=SCHEMA),
    ]

    class FakeSession:
        async def initialize(self):
            return None

        async def list_tools(self):
            return SimpleNamespace(tools=tools)

    @asynccontextmanager
    async def fake_client_session(server, cwd=None):
        yield FakeSession()

    monkeypatch.setattr(mcp, "_client_session", fake_client_session)
    monkeypatch.setattr(mcp, "MAX_MCP_TOOLS", 1)
    with pytest.raises(ValueError, match="tool count exceeds 1"):
        await mcp.MCPRegistry({"public": "ignored"}).discover()


def test_build_signature():
    params = mcp.build_signature(SCHEMA).parameters
    assert list(params) == ["day", "count"]
    assert all(p.kind is inspect.Parameter.KEYWORD_ONLY for p in params.values())
    assert params["day"].default is inspect.Parameter.empty
    assert params["day"].annotation is str
    assert params["count"].default is None
    assert params["count"].annotation is int


async def test_broker_round_trip_records_trusted_metric(tmp_path):
    server_cwd = tmp_path / "server"
    server_cwd.mkdir()
    session = Session(tmp_path / "session")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(),
        cwd=str(server_cwd),
        mcp_servers={"local": _stdio_server()},
    )
    await supervisor.start()
    skills_dir = tmp_path / "skills"
    supervisor.write_brokered_skill_modules(skills_dir)
    sys.path.insert(0, str(skills_dir))
    skill = importlib.import_module("local_echo")
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    broker.configure(endpoint)
    broker.set_scope(scope)
    try:
        results = await asyncio.gather(
            skill.run(text="one"),
            skill.run(text="two"),
        )
        with pytest.raises(RuntimeError, match="unknown brokered skill capability"):
            await broker.call_skill("invalid", {})
        broker.configure(broker.BrokerEndpoint(endpoint.socket_path, "invalid"))
        with pytest.raises(RuntimeError, match="invalid broker capability"):
            await skill.run(text="forbidden")
        broker.configure(endpoint)
        await supervisor.close_scope(scope)
        with pytest.raises(RuntimeError, match="invalid broker capability"):
            await skill.run(text="stale")
    finally:
        broker.set_scope(None)
        broker.configure(None)
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()
        sys.path.remove(str(skills_dir))
        sys.modules.pop("local_echo", None)

    assert results == [
        f"stdio:True:{server_cwd}:one",
        f"stdio:True:{server_cwd}:two",
    ]
    stats, child_stats = supervisor.programmatic_tool_call_stats(supervisor.root_id)
    assert stats.python_total == 2
    assert stats.by_tool_python == {"local_echo": 2}
    assert child_stats.python_total == 0


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
