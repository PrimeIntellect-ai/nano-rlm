"""Native ACP transport and persistent engine lifecycle."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

from acp import PROTOCOL_VERSION, spawn_agent_process, text_block
from acp.schema import HttpHeader, HttpMcpServer

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.acp import RLMACPAgent
from rlm.engine import RLMEngine
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
        self.closed = False
        self.stop_reason = "done"
        self.instances.append(self)

    async def prompt(self, prompt: str) -> RLMResult:
        self.prompts.append(prompt)
        return RLMResult(
            answer=f"reply:{prompt}",
            usage=TokenUsage(prompt_tokens=3, completion_tokens=2),
            turns=len(self.prompts),
        )

    def close(self) -> None:
        self.closed = True


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
            )
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
        }
    }
    assert [update.content.text for _, update in client.updates] == [
        "reply:one",
        "reply:two",
    ]
    assert first.usage.total_tokens == 5
    assert second.stop_reason == "end_turn"

    await agent.close_session(created.session_id)
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
