"""Native ACP transport and persistent engine lifecycle."""

from __future__ import annotations

import asyncio
import json
import os
import sys
import threading
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, RequestError, spawn_agent_process, text_block
from acp.schema import EnvVariable, HttpHeader, HttpMcpServer, McpServerStdio
import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.acp import RLMACPAgent
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult, TokenUsage


class _Client:
    def __init__(self) -> None:
        self.updates: list[tuple[str, Any]] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        self.updates.append((session_id, update))


class _Engine:
    instances: list[_Engine] = []

    def __init__(
        self,
        *,
        cwd: str,
        session,
        mcp_servers: dict[str, Any],
    ) -> None:
        self.cwd = cwd
        self.session = session
        self.mcp_servers = mcp_servers
        self.prompts: list[str] = []
        self.prompt_started = asyncio.Event()
        self.closed = False
        self.stop_reason = "done"
        self.instances.append(self)

    async def prompt(self, prompt: str) -> RLMResult:
        self.prompts.append(prompt)
        self.prompt_started.set()
        if prompt == "wait":
            await asyncio.Future()
        if prompt == "fail":
            raise RuntimeError("transient failure")
        return RLMResult(
            answer=f"reply:{prompt}",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
            turns=len(self.prompts),
        )

    def close(self) -> None:
        self.closed = True

    async def aclose(self) -> None:
        self.close()


async def test_engine_prompt_preserves_conversation(session):
    client = DummyClient(
        [DummyMessage(content="first"), DummyMessage(content="second")]
    )
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    try:
        first = await engine.prompt("one")
        second = await engine.prompt("two")
    finally:
        engine.close()

    assert first.answer == "first"
    assert second.answer == "second"
    assert first.turns == 1
    assert second.turns == 1
    assert first.usage == TokenUsage(prompt_tokens=1, completion_tokens=1)
    assert second.usage == TokenUsage(prompt_tokens=1, completion_tokens=1)
    assert engine._total_usage == TokenUsage(prompt_tokens=2, completion_tokens=2)
    assert client.calls[1]["messages"][-4:] == [
        {"role": "user", "content": "one"},
        {"role": "assistant", "content": "first"},
        {"role": "user", "content": "two"},
        {"role": "assistant", "content": "second"},
    ]
    meta = json.loads((Path(session.dir) / "meta.json").read_text())
    assert meta["turns"] == 2
    assert meta["answer_preview"] == "second"


async def test_engine_prompt_preserves_ipython_kernel(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "x = 41"})]),
            DummyMessage(content="stored"),
            DummyMessage(
                tool_calls=[DummyToolCall("ipython", {"code": "print(x + 1)"})]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    try:
        await engine.prompt("remember a value")
        result = await engine.prompt("use that value")
    finally:
        engine.close()

    tool_messages = [
        message for message in client.calls[-1]["messages"] if message["role"] == "tool"
    ]
    assert result.answer == "done"
    assert tool_messages[-1]["content"].strip() == "42"


async def test_engine_cancelled_prompt_can_be_retried(session):
    client = DummyClient([DummyMessage(content="continued")])
    create = client.create
    prompt_started = asyncio.Event()

    async def block_first_prompt(**kwargs):
        if not prompt_started.is_set():
            client.calls.append(kwargs)
            prompt_started.set()
            await asyncio.Future()
        return await create(**kwargs)

    client.create = block_first_prompt
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    pending = asyncio.create_task(engine.prompt("cancel me"))
    await prompt_started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    try:
        result = await engine.prompt("continue")
    finally:
        engine.close()

    assert result.answer == "continued"
    assert result.turns == 1
    assert client.calls[-1]["messages"][-2:] == [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "continued"},
    ]


async def test_engine_failed_prompt_can_be_retried(session):
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("boom", {})]),
            DummyMessage(content="continued"),
        ]
    )
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    with pytest.raises(RuntimeError, match="boom"):
        await engine.prompt("fail")

    try:
        result = await engine.prompt("continue")
    finally:
        engine.close()

    assert result.answer == "continued"
    assert result.turns == 1
    meta = json.loads((Path(session.dir) / "meta.json").read_text())
    assert meta["usage"] == {"prompt_tokens": 2, "completion_tokens": 2}
    log = [
        json.loads(line)
        for line in (Path(session.dir) / "messages.jsonl").read_text().splitlines()
    ]
    assert [entry["type"] for entry in log] == [
        "assistant",
        "prompt_rollback",
        "assistant",
        "done",
    ]
    assert log[1]["attempted_turns"] == 1
    assert log[1]["reason"] == "error"
    assert client.calls[-1]["messages"][-2:] == [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "continued"},
    ]


async def test_engine_cancel_masks_tool_cleanup_error(monkeypatch, session):
    started = threading.Event()
    interrupted = threading.Event()

    class FailingTool:
        def execute(self, args, context):
            started.set()
            assert interrupted.wait(timeout=5)
            raise RuntimeError("interrupted tool failed")

    class FakeREPL:
        def __init__(self):
            self.finished = False
            self.stopped = False

        def interrupt(self):
            interrupted.set()

        def finish_interrupt(self):
            self.finished = True

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr("rlm.engine.get_builtin_tool", lambda name: FailingTool())
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("failing", {})]),
            DummyMessage(content="continued"),
        ]
    )
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]
    repl = FakeREPL()
    engine._started = True
    engine._messages = [{"role": "system", "content": "system"}]
    engine._repl = repl  # type: ignore[assignment]

    pending = asyncio.create_task(engine.prompt("cancel"))
    for _ in range(100):
        if started.is_set():
            break
        await asyncio.sleep(0.01)
    assert started.is_set()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending

    try:
        result = await engine.prompt("continue")
    finally:
        engine.close()

    assert result.answer == "continued"
    assert repl.finished is True
    assert repl.stopped is True
    assert client.calls[-1]["messages"][-2:] == [
        {"role": "user", "content": "continue"},
        {"role": "assistant", "content": "continued"},
    ]


async def test_engine_cancelled_tool_recovers_kernel(session, tmp_path):
    started = tmp_path / "tool-started"
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": (
                                "from pathlib import Path; import time; "
                                f"kept = 41; Path({str(started)!r}).touch(); "
                                "time.sleep(30)"
                            )
                        },
                    )
                ]
            ),
            DummyMessage(
                tool_calls=[DummyToolCall("ipython", {"code": "print(kept + 1)"})]
            ),
            DummyMessage(content="continued"),
        ]
    )
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    pending = asyncio.create_task(engine.prompt("cancel the tool"))
    for _ in range(100):
        if started.exists():
            break
        await asyncio.sleep(0.05)
    assert started.exists()

    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=10)

    try:
        result = await engine.prompt("continue")
    finally:
        engine.close()

    tool_messages = [
        message for message in client.calls[-1]["messages"] if message["role"] == "tool"
    ]
    assert result.answer == "continued"
    assert tool_messages[-1]["content"].strip() == "42"


async def test_engine_failed_start_cleans_kernel_before_retry(
    monkeypatch, session, tmp_path
):
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    repls = []

    class FakeREPL:
        def __init__(self, **kwargs):
            self.started = False
            self.stopped = False
            repls.append(self)

        def start(self):
            self.started = True

        def shutdown(self):
            self.stopped = True

    monkeypatch.setattr("rlm.engine.IPythonREPL", FakeREPL)
    system_prompt = tmp_path / "system.txt"
    client = DummyClient([DummyMessage(content="continued")])
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        system_prompt_path=str(system_prompt),
    )

    with pytest.raises(FileNotFoundError):
        await engine.prompt("first")
    assert repls[0].started is True
    assert repls[0].stopped is True
    assert engine._repl is None

    system_prompt.write_text("system")
    result = await engine.prompt("retry")
    await engine.aclose()

    assert result.answer == "continued"
    assert len(repls) == 2
    assert repls[1].stopped is True


async def test_acp_failed_session_creation_closes_session(monkeypatch, tmp_path):
    session = Session(tmp_path / "session")

    class FailingEngine:
        def __init__(self, **kwargs):
            raise RuntimeError("engine init failed")

    monkeypatch.setattr("rlm.acp.Session", lambda: session)
    monkeypatch.setattr("rlm.acp.RLMEngine", FailingEngine)
    agent = RLMACPAgent()

    with pytest.raises(RuntimeError, match="engine init failed"):
        await agent.new_session(str(tmp_path))

    assert session._msg_file.closed is True
    assert agent._sessions == {}


async def test_acp_session_reuses_engine(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    client = _Client()
    agent = RLMACPAgent()
    agent.on_connect(client)  # type: ignore[arg-type]

    initialized = await agent.initialize(PROTOCOL_VERSION)
    assert initialized.agent_capabilities.mcp_capabilities.http is True
    assert initialized.agent_capabilities.load_session is False

    created = await agent.new_session(
        str(tmp_path),
        mcp_servers=[
            HttpMcpServer(
                type="http",
                name="tools",
                url="http://127.0.0.1:8000/mcp",
                headers=[HttpHeader(name="Authorization", value="Bearer task")],
            ),
            McpServerStdio(
                name="local",
                command="/usr/bin/tool-server",
                args=["--stdio"],
                env=[EnvVariable(name="TOKEN", value="task-secret")],
            ),
        ],
    )
    first = await agent.prompt(created.session_id, [text_block("one")])
    second = await agent.prompt(created.session_id, [text_block("two")])

    engine = _Engine.instances[0]
    assert engine.prompts == ["one", "two"]
    assert engine.mcp_servers == {
        "tools": {
            "url": "http://127.0.0.1:8000/mcp",
            "headers": {"Authorization": "Bearer task"},
        },
        "local": {
            "command": "/usr/bin/tool-server",
            "args": ["--stdio"],
            "env": {"TOKEN": "task-secret"},
        },
    }
    assert [update.content.text for _, update in client.updates] == [
        "reply:one",
        "reply:two",
    ]
    assert first.usage.total_tokens == 5
    assert second.stop_reason == "end_turn"

    await agent.close_session(created.session_id)
    assert engine.closed is True


async def test_acp_cancel_keeps_session_reusable(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await agent.new_session(str(tmp_path))
    engine = _Engine.instances[0]

    pending = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("wait")])
    )
    await engine.prompt_started.wait()
    await agent.cancel(created.session_id)

    assert (await pending).stop_reason == "cancelled"
    assert engine.closed is False
    resumed = await agent.prompt(created.session_id, [text_block("after")])
    assert resumed.stop_reason == "end_turn"
    assert engine.prompts == ["wait", "after"]

    await agent.close_session(created.session_id)


async def test_acp_failed_prompt_keeps_session_reusable(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await agent.new_session(str(tmp_path))
    engine = _Engine.instances[0]

    with pytest.raises(RuntimeError, match="transient failure"):
        await agent.prompt(created.session_id, [text_block("fail")])

    resumed = await agent.prompt(created.session_id, [text_block("after")])
    assert resumed.stop_reason == "end_turn"
    assert engine.prompts == ["fail", "after"]
    assert engine.closed is False

    await agent.close_session(created.session_id)


async def test_acp_close_rejects_queued_prompt(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await agent.new_session(str(tmp_path))
    engine = _Engine.instances[0]

    running = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("wait")])
    )
    await engine.prompt_started.wait()
    queued = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("after")])
    )
    await asyncio.sleep(0)

    await agent.close_session(created.session_id)

    assert (await running).stop_reason == "cancelled"
    with pytest.raises(RequestError):
        await queued
    assert engine.prompts == ["wait"]
    assert engine.closed is True


async def test_acp_stdio_lifecycle(tmp_path):
    executable = str(Path(sys.executable).parent / "rlm")
    env = {**os.environ, "RLM_HOME": str(tmp_path / "rlm")}

    async with spawn_agent_process(_Client(), executable, "--acp", env=env) as (
        connection,
        _process,
    ):
        initialized = await connection.initialize(PROTOCOL_VERSION)
        created = await connection.new_session(cwd=str(tmp_path), mcp_servers=[])
        await connection.close_session(created.session_id)

    assert initialized.agent_info.name == "rlm"
