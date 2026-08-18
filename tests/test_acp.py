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
from rlm.acp import (
    CONTRACT_METADATA,
    CONTRACT_METADATA_KEY,
    RUNTIME_METADATA_KEY,
    SESSION_METADATA_KEY,
    RLMACPAgent,
)
from rlm.engine import RLMEngine
from rlm.config import ExecutionPolicy, InvocationContext, ProviderConfig, RuntimeConfig
from rlm.mcp import MCPHTTPServer, MCPStdioServer
from rlm.session import Session
from rlm.types import RLMResult, TokenUsage


def _contract_metadata() -> dict[str, Any]:
    return CONTRACT_METADATA.copy()


def _runtime_metadata(**overrides: Any) -> dict[str, Any]:
    payload = {
        "lineage_session_id": "test-session",
        "model": "test-model",
        "provider": {
            "base_url": "http://interceptor",
            "api_key": "test-secret",
            "headers": {},
            "max_retries": 2,
        },
        "policy": {
            "max_depth": 0,
            "exec_timeout": 300,
            "max_output": -1,
            "max_tokens": None,
            "summarize_at_tokens": None,
            "max_compactions": None,
            "max_concurrent_subagents": 4,
            "max_subagent_calls": 64,
            "max_tool_output_chars": None,
            "allow_git": False,
        },
        "system_prompt_path": None,
        "append_to_system_prompt": None,
        "skills": [],
        "kernel_env": {},
        "search_api_key": None,
    }
    payload.update(overrides)
    return {RUNTIME_METADATA_KEY: payload}


async def _initialize(agent: RLMACPAgent):
    return await agent.initialize(PROTOCOL_VERSION, **_contract_metadata())


async def _new_session(agent: RLMACPAgent, cwd: str, **kwargs: Any):
    await _initialize(agent)
    return await agent.new_session(cwd, **kwargs, **_runtime_metadata())


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
        runtime_config=None,
        lineage_session_id: str | None = None,
    ) -> None:
        self.cwd = cwd
        self.session = session
        self.mcp_servers = mcp_servers
        self.runtime_config = runtime_config
        self.lineage_session_id = lineage_session_id
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

    def execution_snapshot(self) -> dict[str, Any]:
        return {
            "session_id": self.lineage_session_id or self.session.dir.name,
            "root_invocation_id": "root",
            "model": "test-model",
            "turns": len(self.prompts),
            "usage": {
                "prompt_tokens": len(self.prompts) * 3,
                "completion_tokens": len(self.prompts) * 2,
                "total_tokens": len(self.prompts) * 5,
            },
            "metrics": {},
            "programmatic_tool_call_stats": {},
            "supervisor": {"subagent_calls": 0, "active_subagent_calls": 0},
            "limits": {},
        }


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
    first_headers = client.calls[0]["extra_headers"]
    second_headers = client.calls[1]["extra_headers"]
    assert first_headers["X-RLM-Session-ID"] == session.dir.name
    assert first_headers["X-RLM-Invocation-ID"] == second_headers["X-RLM-Invocation-ID"]
    assert first_headers["X-RLM-Segment-ID"] != second_headers["X-RLM-Segment-ID"]
    assert "X-RLM-Parent-Call-ID" not in first_headers
    assert second_headers["X-RLM-Parent-Call-ID"] == first_headers["X-RLM-Call-ID"]
    meta = json.loads((Path(session.dir) / "meta.json").read_text())
    assert meta["turns"] == 2
    assert meta["answer_preview"] == "second"


def test_execution_snapshot_after_finalize_is_numeric_and_credential_free(session):
    config = RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(
            base_url="http://interceptor",
            api_key="provider-secret",
            headers={"X-Task": "header-secret"},
        ),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(max_depth=1),
        kernel_env=(("TASK_TOKEN", "kernel-secret"),),
        search_api_key="search-secret",
    )
    (session.dir / "programmatic_tool_calls.jsonl").write_text(
        '{"tool":"demo","source":"python"}\n'
    )
    (session.dir / "sub-child").mkdir()
    engine = RLMEngine(
        client=DummyClient([]),  # type: ignore[arg-type]
        session=session,
        runtime_config=config,
    )
    engine._has_result = True
    engine._last_answer = "answer-secret"
    engine.close()

    snapshot = engine.execution_snapshot()

    assert snapshot["programmatic_tool_call_stats"]["by_tool_python"] == {"demo": 1}
    assert snapshot["metrics"]["sub_rlm_num_calls"] == 1
    assert snapshot["metrics"]["has_sub_rlm"] == 1
    assert all(
        isinstance(value, (int, float)) and not isinstance(value, bool)
        for value in snapshot["metrics"].values()
    )
    serialized = json.dumps(snapshot)
    for secret in (
        "provider-secret",
        "header-secret",
        "kernel-secret",
        "search-secret",
        "answer-secret",
    ):
        assert secret not in serialized


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
    failed_headers = client.calls[0]["extra_headers"]
    resumed_headers = client.calls[1]["extra_headers"]
    assert failed_headers["X-RLM-Call-ID"] != resumed_headers["X-RLM-Call-ID"]
    assert failed_headers["X-RLM-Segment-ID"] != resumed_headers["X-RLM-Segment-ID"]
    assert "X-RLM-Parent-Call-ID" not in resumed_headers


async def test_model_call_lineage_survives_retries_and_compaction(monkeypatch, session):
    monkeypatch.setattr("rlm.client._RETRY_DELAYS", (0,))
    client = DummyClient(
        [
            DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "print(1)"})]),
            DummyMessage(content="summary"),
            DummyMessage(content="done"),
        ]
    )
    create = client.create
    attempts = []

    async def flaky_first_call(**kwargs):
        attempts.append(kwargs)
        if len(attempts) == 1:
            raise ConnectionResetError("retry")
        return await create(**kwargs)

    client.create = flaky_first_call
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        summarize_at_tokens=1,
    )

    try:
        result = await engine.prompt("compact")
    finally:
        engine.close()

    assert result.answer == "done"
    assert attempts[0]["extra_headers"] == attempts[1]["extra_headers"]
    turn, compaction, resumed = [call["extra_headers"] for call in client.calls]
    assert {header["X-RLM-Segment-ID"] for header in (turn, compaction, resumed)} == {
        turn["X-RLM-Segment-ID"]
    }
    assert [
        turn["X-RLM-Call-Kind"],
        compaction["X-RLM-Call-Kind"],
        resumed["X-RLM-Call-Kind"],
    ] == ["turn", "compaction", "turn"]
    assert compaction["X-RLM-Parent-Call-ID"] == turn["X-RLM-Call-ID"]
    assert resumed["X-RLM-Parent-Call-ID"] == compaction["X-RLM-Call-ID"]


async def test_latest_cancelled_prompt_does_not_finalize_prior_result(session):
    client = DummyClient([DummyMessage(content="first")])
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]
    await engine.prompt("one")

    prompt_started = asyncio.Event()

    async def block_prompt(**kwargs):
        prompt_started.set()
        await asyncio.Future()

    client.create = block_prompt
    pending = asyncio.create_task(engine.prompt("two"))
    await prompt_started.wait()
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    engine.close()

    meta = json.loads((Path(session.dir) / "meta.json").read_text())
    assert meta["status"] == "running"
    assert "answer_preview" not in meta


async def test_depth_limit_is_a_completed_result(monkeypatch, session):
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "0")
    client = DummyClient([])
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]

    result = await engine.run("too deep")

    meta = json.loads((Path(session.dir) / "meta.json").read_text())
    assert result.answer == "[depth limit 0 reached, cannot start]"
    assert meta["status"] == "done"
    assert meta["metrics"]["stop_reason"] == "depth_limit"


async def test_compaction_counts_seed_prompt(session):
    client = DummyClient([DummyMessage(content="summary")])
    engine = RLMEngine(client=client, session=session)  # type: ignore[arg-type]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original prompt"},
        {"role": "assistant", "content": "work"},
    ]

    try:
        await engine._compact_branch(
            messages, turn=0, active_tools=[], segment_id="test-segment"
        )
    finally:
        engine.close()

    assert engine._metrics.num_compactions == 1
    assert engine._metrics.compaction_chars_dropped_mean == len("original promptwork")


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

    failed_headers = client.calls[0]["extra_headers"]
    resumed_headers = client.calls[1]["extra_headers"]
    assert failed_headers["X-RLM-Call-ID"] != resumed_headers["X-RLM-Call-ID"]
    assert "X-RLM-Parent-Call-ID" not in resumed_headers


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
    assert engine._metrics._ipython_call_count == 2


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
    await _initialize(agent)

    with pytest.raises(RuntimeError, match="engine init failed"):
        await agent.new_session(str(tmp_path), **_runtime_metadata())

    assert session._msg_file.closed is True
    assert agent._sessions == {}


@pytest.mark.parametrize(
    "metadata",
    [
        {},
        {CONTRACT_METADATA_KEY: False},
        {CONTRACT_METADATA_KEY: "v1"},
    ],
)
async def test_acp_rejects_missing_or_unsupported_contract(tmp_path, metadata):
    agent = RLMACPAgent()

    with pytest.raises(RequestError):
        await agent.initialize(PROTOCOL_VERSION, **metadata)
    with pytest.raises(RequestError):
        await agent.new_session(str(tmp_path), **_runtime_metadata())

    assert agent._sessions == {}


async def test_acp_requires_complete_runtime_metadata(tmp_path):
    agent = RLMACPAgent()
    await _initialize(agent)

    with pytest.raises(RequestError):
        await agent.new_session(str(tmp_path))
    with pytest.raises(RequestError):
        await agent.new_session(
            str(tmp_path),
            **{RUNTIME_METADATA_KEY: {"lineage_session_id": "old-client"}},
        )

    assert agent._sessions == {}


async def test_acp_session_reuses_engine(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    client = _Client()
    agent = RLMACPAgent()
    agent.on_connect(client)  # type: ignore[arg-type]

    initialized = await _initialize(agent)
    assert initialized.agent_capabilities.mcp_capabilities.http is True
    assert initialized.agent_capabilities.load_session is False
    assert initialized.agent_capabilities.field_meta == CONTRACT_METADATA
    assert initialized.field_meta is None

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
        **_runtime_metadata(),
    )
    first = await agent.prompt(created.session_id, [text_block("one")])
    second = await agent.prompt(created.session_id, [text_block("two")])

    engine = _Engine.instances[0]
    assert engine.prompts == ["one", "two"]
    assert engine.mcp_servers == {
        "tools": MCPHTTPServer(
            url="http://127.0.0.1:8000/mcp",
            headers={"Authorization": "Bearer task"},
        ),
        "local": MCPStdioServer(
            command="/usr/bin/tool-server",
            args=["--stdio"],
            env={"TOKEN": "task-secret"},
        ),
    }
    assert [update.content.text for _, update in client.updates] == [
        "reply:one",
        "reply:two",
    ]
    assert first.usage.total_tokens == 5
    assert second.stop_reason == "end_turn"
    created_snapshot = created.field_meta[SESSION_METADATA_KEY]
    first_snapshot = first.field_meta[SESSION_METADATA_KEY]
    second_snapshot = second.field_meta[SESSION_METADATA_KEY]
    assert created_snapshot["status"] == "created"
    assert created_snapshot["sequence"] == 0
    assert created_snapshot["final"] is False
    assert first_snapshot["turns"] == 1
    assert first_snapshot["status"] == "idle"
    assert first_snapshot["sequence"] == 1
    assert first_snapshot["last_stop_reason"] == "done"
    assert second_snapshot["sequence"] == 2
    assert created.model_dump(mode="json", by_alias=True)["_meta"] == created.field_meta

    closed = await agent.close_session(created.session_id)
    closed_snapshot = closed.field_meta[SESSION_METADATA_KEY]
    assert closed_snapshot["status"] == "closed"
    assert closed_snapshot["sequence"] == 3
    assert closed_snapshot["final"] is True
    assert engine.closed is True


async def test_acp_runtime_metadata_keeps_credentials_out_of_responses(
    monkeypatch, tmp_path
):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setenv("RLM_API_KEY", "ambient-provider-secret")
    monkeypatch.setenv("RLM_MODEL", "ambient-model")
    monkeypatch.setenv("RLM_MAX_DEPTH", "9")
    monkeypatch.setenv("RLM_SKILLS", "search")
    monkeypatch.setenv("SERPER_API_KEY", "ambient-search-secret")
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    metadata = _runtime_metadata(
        lineage_session_id="trace-123",
        model="session-model",
        provider={
            "base_url": "http://interceptor",
            "api_key": "session-provider-secret",
            "headers": {"X-Task": "header-secret"},
            "max_retries": 2,
        },
        policy={
            "max_depth": 2,
            "exec_timeout": 30,
            "max_output": 1000,
            "max_tokens": 500,
            "summarize_at_tokens": 300,
            "max_compactions": 2,
            "max_concurrent_subagents": 4,
            "max_subagent_calls": 8,
            "max_tool_output_chars": 2000,
            "allow_git": True,
        },
        append_to_system_prompt="session instructions",
        skills=["edit"],
        kernel_env={"TASK_VISIBLE": "task-secret"},
        search_api_key="search-secret",
    )

    await _initialize(agent)
    created = await agent.new_session(str(tmp_path), **metadata)
    engine = _Engine.instances[0]

    assert engine.lineage_session_id == "trace-123"
    assert engine.runtime_config.provider.base_url == "http://interceptor"
    assert engine.runtime_config.provider.api_key == "session-provider-secret"
    assert engine.runtime_config.provider.headers == {"X-Task": "header-secret"}
    assert engine.runtime_config.model == "session-model"
    assert engine.runtime_config.policy.max_depth == 2
    assert engine.runtime_config.policy.exec_timeout == 30
    assert engine.runtime_config.policy.max_tokens == 500
    assert engine.runtime_config.policy.max_tool_output_chars == 2000
    assert engine.runtime_config.policy.allow_git is True
    assert engine.runtime_config.append_to_system_prompt == "session instructions"
    assert engine.runtime_config.skills == ("edit",)
    assert engine.runtime_config.kernel_env == (("TASK_VISIBLE", "task-secret"),)
    assert engine.runtime_config.search_api_key == "search-secret"
    response_json = created.model_dump_json(by_alias=True)
    for secret in (
        "session-provider-secret",
        "header-secret",
        "task-secret",
        "search-secret",
        "ambient-provider-secret",
    ):
        assert secret not in response_json
    assert created.field_meta[SESSION_METADATA_KEY]["session_id"] == "trace-123"
    await agent.close_session(created.session_id)


@pytest.mark.parametrize(
    "payload",
    [
        {"lineage_session_id": "bad id"},
        {"provider": {}},
        {
            "provider": {
                "api_key": "secret",
                "headers": {"x-rlm-call-id": "forged"},
            }
        },
        {"kernel_env": {"RLM_API_KEY": "forbidden"}},
        {"unknown": "value"},
    ],
)
async def test_acp_rejects_invalid_runtime_metadata(monkeypatch, tmp_path, payload):
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    agent = RLMACPAgent()
    await _initialize(agent)

    with pytest.raises(RequestError):
        await agent.new_session(str(tmp_path), **_runtime_metadata(**payload))

    assert agent._sessions == {}


async def test_acp_cancel_keeps_session_reusable(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))
    engine = _Engine.instances[0]

    pending = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("wait")])
    )
    await engine.prompt_started.wait()
    await agent.cancel(created.session_id)

    cancelled = await pending
    assert cancelled.stop_reason == "cancelled"
    assert cancelled.field_meta[SESSION_METADATA_KEY]["status"] == "idle"
    assert cancelled.field_meta[SESSION_METADATA_KEY]["last_stop_reason"] == "cancelled"
    assert engine.closed is False
    resumed = await agent.prompt(created.session_id, [text_block("after")])
    assert resumed.stop_reason == "end_turn"
    assert engine.prompts == ["wait", "after"]

    await agent.close_session(created.session_id)


async def test_acp_transport_cancellation_propagates(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))
    engine = _Engine.instances[0]

    pending = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("wait")])
    )
    await engine.prompt_started.wait()
    pending.cancel()

    with pytest.raises(asyncio.CancelledError):
        await pending
    resumed = await agent.prompt(created.session_id, [text_block("after")])
    assert resumed.stop_reason == "end_turn"
    await agent.close_session(created.session_id)


async def test_acp_close_cancels_blocked_answer_delivery(monkeypatch, tmp_path):
    class BlockingClient(_Client):
        def __init__(self):
            super().__init__()
            self.started = asyncio.Event()

        async def session_update(
            self, session_id: str, update: Any, **kwargs: Any
        ) -> None:
            self.started.set()
            await asyncio.Future()

    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    client = BlockingClient()
    agent = RLMACPAgent()
    agent.on_connect(client)  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))

    pending = asyncio.create_task(agent.prompt(created.session_id, [text_block("one")]))
    await client.started.wait()
    await agent.cancel(created.session_id)
    await asyncio.sleep(0)
    assert pending.done() is False
    closed = await asyncio.wait_for(agent.close_session(created.session_id), timeout=1)

    assert (await pending).stop_reason == "cancelled"
    assert closed.field_meta[SESSION_METADATA_KEY]["status"] == "closed"
    assert _Engine.instances[0].closed is True


async def test_acp_cancelled_close_remains_owned_by_shutdown(monkeypatch, tmp_path):
    class SlowCancelEngine(_Engine):
        release = asyncio.Event()

        async def prompt(self, prompt: str) -> RLMResult:
            self.prompts.append(prompt)
            self.prompt_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                await self.release.wait()
                raise

    SlowCancelEngine.instances.clear()
    SlowCancelEngine.release = asyncio.Event()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", SlowCancelEngine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))
    engine = SlowCancelEngine.instances[0]
    pending = asyncio.create_task(
        agent.prompt(created.session_id, [text_block("wait")])
    )
    await engine.prompt_started.wait()
    closing = asyncio.create_task(agent.close_session(created.session_id))
    await asyncio.sleep(0)
    closing.cancel()
    with pytest.raises(asyncio.CancelledError):
        await closing

    SlowCancelEngine.release.set()
    assert (await pending).stop_reason == "cancelled"
    await agent.shutdown()
    assert engine.closed is True
    assert agent._sessions == {}


async def test_acp_failed_prompt_keeps_session_reusable(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))
    engine = _Engine.instances[0]

    with pytest.raises(RuntimeError, match="transient failure"):
        await agent.prompt(created.session_id, [text_block("fail")])

    resumed = await agent.prompt(created.session_id, [text_block("after")])
    assert resumed.stop_reason == "end_turn"
    assert engine.prompts == ["fail", "after"]
    assert engine.closed is False

    closed = await agent.close_session(created.session_id)
    assert closed.field_meta[SESSION_METADATA_KEY]["last_stop_reason"] == "done"


async def test_acp_close_reports_last_prompt_failure(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))

    with pytest.raises(RuntimeError, match="transient failure"):
        await agent.prompt(created.session_id, [text_block("fail")])
    closed = await agent.close_session(created.session_id)

    snapshot = closed.field_meta[SESSION_METADATA_KEY]
    assert snapshot["last_stop_reason"] == "error"
    assert snapshot["final"] is True


async def test_acp_close_rejects_queued_prompt(monkeypatch, tmp_path):
    _Engine.instances.clear()
    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr("rlm.acp.RLMEngine", _Engine)
    agent = RLMACPAgent()
    agent.on_connect(_Client())  # type: ignore[arg-type]
    created = await _new_session(agent, str(tmp_path))
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
        initialized = await connection.initialize(
            PROTOCOL_VERSION, **_contract_metadata()
        )
        created = await connection.new_session(
            cwd=str(tmp_path),
            mcp_servers=[],
            **_runtime_metadata(lineage_session_id="wire-session"),
        )
        closed = await connection.close_session(created.session_id)

    assert initialized.agent_info.name == "rlm"
    assert created.field_meta[SESSION_METADATA_KEY]["status"] == "created"
    assert created.field_meta[SESSION_METADATA_KEY]["session_id"] == "wire-session"
    assert closed.field_meta[SESSION_METADATA_KEY]["status"] == "closed"
