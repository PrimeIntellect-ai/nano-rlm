from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor
from rlm.types import RLMResult, TokenUsage


def _config(
    *, max_depth: int = 2, max_concurrent: int = 4, max_calls: int = 16
) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url="http://interceptor", api_key="secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=max_depth,
            max_concurrent_subagents=max_concurrent,
            max_subagent_calls=max_calls,
        ),
    )


async def _start_child(supervisor, capability, scope, prompt):
    child = supervisor._spawn(
        supervisor._caller(capability, scope), scope, prompt, None, False
    )

    async def result():
        await child.done.wait()
        payload = await supervisor._agent_operation(
            {
                "op": "agent.result",
                "capability": capability,
                "scope_id": scope,
                "agent_id": child.id,
            }
        )
        from rlm.broker import result_from_payload

        return result_from_payload(payload)

    return asyncio.create_task(result())


@dataclass
class _EngineState:
    active: int = 0
    peak: int = 0
    cancelled: int = 0


class _FastEngine:
    state = _EngineState()

    def __init__(self, *, runtime_config, session, **kwargs):
        self.runtime_config = runtime_config
        self.session = session

    async def prompt(self, prompt: str) -> RLMResult:
        state = self.state
        state.active += 1
        state.peak = max(state.peak, state.active)
        try:
            await asyncio.sleep(0.02)
            return RLMResult(
                answer=f"child:{prompt}",
                session_dir=self.session.dir,
                usage=TokenUsage(prompt_tokens=2, completion_tokens=1),
                turns=1,
            )
        finally:
            state.active -= 1

    async def aclose(self):
        self.session.close()


class _SometimesBlockingEngine(_FastEngine):
    started = asyncio.Event()

    async def prompt(self, prompt: str) -> RLMResult:
        if prompt != "wait":
            return await super().prompt(prompt)
        state = self.state
        state.active += 1
        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            state.cancelled += 1
            raise
        finally:
            state.active -= 1


class _NestedEngine:
    def __init__(
        self,
        *,
        runtime_config,
        session,
        supervisor,
        invocation_id,
        **kwargs,
    ):
        self.runtime_config = runtime_config
        self.session = session
        self.supervisor = supervisor
        self.invocation_id = invocation_id

    async def prompt(self, prompt: str) -> RLMResult:
        depth = self.runtime_config.invocation.depth
        if depth == self.runtime_config.policy.max_depth:
            return RLMResult(answer=f"leaf:{depth}", session_dir=self.session.dir)
        scope = await self.supervisor.open_scope(self.invocation_id)
        endpoint = self.supervisor.endpoint_for(self.invocation_id)
        try:
            task = await _start_child(
                self.supervisor, endpoint.capability, scope, prompt
            )
            result = await task
        finally:
            await self.supervisor.close_scope(scope)
        return RLMResult(
            answer=f"depth:{depth}>{result.answer}", session_dir=self.session.dir
        )

    async def aclose(self):
        self.session.close()


async def test_parallel_children_respect_depth_capacity(tmp_path):
    _FastEngine.state = _EngineState()
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=2, max_concurrent=4),
        cwd=str(tmp_path),
        engine_factory=_FastEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        tasks = [
            await _start_child(supervisor, endpoint.capability, scope, str(i))
            for i in range(6)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert [result.answer for result in results] == [f"child:{i}" for i in range(6)]
    assert _FastEngine.state.peak == 2
    assert supervisor.total_calls == 6


async def test_total_call_limit_is_atomic(tmp_path):
    _FastEngine.state = _EngineState()
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1, max_concurrent=2, max_calls=2),
        cwd=str(tmp_path),
        engine_factory=_FastEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        tasks = [
            await _start_child(supervisor, endpoint.capability, scope, str(i))
            for i in range(2)
        ]
        with pytest.raises(RuntimeError, match="call limit"):
            await _start_child(supervisor, endpoint.capability, scope, "excess")
        results = await asyncio.gather(*tasks)
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert supervisor.total_calls == 2
    assert len(results) == 2


async def test_cancel_while_awaiting_capacity_closes_child_session(
    tmp_path, monkeypatch
):
    """A child cancelled between session creation and engine start leaks nothing."""
    from rlm import supervisor as supervisor_module

    created: list[Session] = []

    class _RecordingSession(Session):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            created.append(self)

    monkeypatch.setattr(supervisor_module, "Session", _RecordingSession)
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1, max_concurrent=2),
        cwd=str(tmp_path),
        engine_factory=_FastEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        semaphore = supervisor._semaphores[1]
        await semaphore.acquire()
        await semaphore.acquire()
        child = supervisor._spawn(
            supervisor._caller(endpoint.capability, scope), scope, "x", None, False
        )
        await supervisor._terminate(child)
        assert child.status == "cancelled"
        semaphore.release()
        semaphore.release()
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert len(created) == 1
    assert created[0]._msg_file.closed
    assert supervisor._invocations == {}
    assert supervisor._capabilities == {}
    assert supervisor.semantic_edges.snapshot() == {"edges": []}


async def test_child_factory_failure_publishes_no_semantic_edge(tmp_path):
    class FailingEngine:
        def __init__(self, **kwargs):
            raise RuntimeError("engine init failed")

    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1),
        cwd=str(tmp_path),
        engine_factory=FailingEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        task = await _start_child(supervisor, endpoint.capability, scope, "x")
        with pytest.raises(RuntimeError, match="engine init failed"):
            await task
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert supervisor.semantic_edges.snapshot() == {"edges": []}


async def test_failed_child_returns_last_committed_request_to_parent(tmp_path):
    class PartiallyFailingEngine:
        def __init__(self, *, supervisor, invocation_id, **kwargs):
            self.semantic_edges = supervisor.semantic_edges
            self.invocation_id = invocation_id

        async def prompt(self, prompt: str) -> RLMResult:
            request_id = self.semantic_edges.start_request(self.invocation_id)
            self.semantic_edges.finish_request(request_id)
            raise RuntimeError("child failed")

        async def aclose(self):
            pass

    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1),
        cwd=str(tmp_path),
        engine_factory=PartiallyFailingEngine,
    )
    await supervisor.start()
    parent_request = supervisor.semantic_edges.start_request(supervisor.root_id)
    supervisor.semantic_edges.finish_request(parent_request)
    scope = await supervisor.open_scope(supervisor.root_id, parent_request)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        child = supervisor._spawn(
            supervisor._caller(endpoint.capability, scope), scope, "x", None, False
        )
        await child.done.wait()
        unrelated_request = supervisor.semantic_edges.start_request(supervisor.root_id)
        supervisor.semantic_edges.finish_request(unrelated_request)
        assert not any(
            edge["type"] == "subagent_return"
            for edge in supervisor.semantic_edges.snapshot()["edges"]
        )
        with pytest.raises(RuntimeError, match="child failed"):
            await supervisor._agent_operation(
                {
                    "op": "agent.result",
                    "capability": endpoint.capability,
                    "scope_id": scope,
                    "agent_id": child.id,
                }
            )
        resumed_request = supervisor.semantic_edges.start_request(supervisor.root_id)
        supervisor.semantic_edges.finish_request(resumed_request)
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    edges = supervisor.semantic_edges.snapshot()["edges"]
    child_request = next(
        edge["target_request_id"] for edge in edges if edge["type"] == "subagent_call"
    )
    assert {
        "source_request_id": child_request,
        "target_request_id": resumed_request,
        "type": "subagent_return",
    } in edges


async def test_saturated_nested_calls_do_not_deadlock(tmp_path):
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=3, max_concurrent=3),
        cwd=str(tmp_path),
        engine_factory=_NestedEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        tasks = [
            await _start_child(supervisor, endpoint.capability, scope, str(i))
            for i in range(2)
        ]
        results = await asyncio.wait_for(asyncio.gather(*tasks), timeout=2)
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert [result.answer for result in results] == [
        "depth:1>depth:2>leaf:3",
        "depth:1>depth:2>leaf:3",
    ]
    assert supervisor.total_calls == 6


async def test_real_kernel_uses_agent_handles(monkeypatch, session):
    _FastEngine.state = _EngineState()
    for name in (
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "PRIME_API_KEY",
        "PRIME_TEAM_ID",
        "RLM_API_KEY",
        "RLM_BASE_URL",
    ):
        monkeypatch.setenv(name, f"secret-{name}")
    config = _config(max_depth=1)
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=_FastEngine,
    )
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": """
import os, subprocess, rlm.config
secret_names = {
    'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'PRIME_API_KEY',
    'PRIME_TEAM_ID', 'RLM_API_KEY', 'RLM_BASE_URL',
}
child = await rlm.agent.spawn('hello', name='first')
other = await rlm.agent.spawn('again')
await asyncio.gather(child.wait(), other.wait())
child = await child.result()
other = await other.result()
subprocess_env = subprocess.check_output(['env'], text=True)
print(child.answer, other.answer)
print(all(name not in os.environ for name in secret_names))
print(all(f'{name}=' not in subprocess_env for name in secret_names))
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
        runtime_config=config,
        supervisor=supervisor,
        invocation_id=supervisor.root_id,
    )
    try:
        result = await engine.run("delegate")
    finally:
        await supervisor.aclose()

    assert result.answer == "done"
    assert tool_result(client).strip().splitlines() == [
        "child:hello child:again",
        "True",
        "True",
    ]
    assert supervisor.total_calls == 2


async def test_real_kernel_cell_cancellation_preserves_child_until_explicit_cancel(
    session,
):
    _SometimesBlockingEngine.state = _EngineState()
    _SometimesBlockingEngine.started = asyncio.Event()
    config = _config(max_depth=1)
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=_SometimesBlockingEngine,
    )
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "worker = await rlm.agent.spawn('wait', name='worker'); await worker.wait(timeout=30)"
                        },
                    )
                ]
            ),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "worker = await rlm.agent.get('worker'); assert (await worker.info()).status == 'running'; await worker.cancel(); child = await rlm.agent.spawn('next'); await child.wait(); print((await child.result()).answer)"
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
        runtime_config=config,
        supervisor=supervisor,
        invocation_id=supervisor.root_id,
    )

    pending = asyncio.create_task(engine.prompt("delegate"))
    await asyncio.wait_for(_SometimesBlockingEngine.started.wait(), timeout=5)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await asyncio.wait_for(pending, timeout=10)

    try:
        result = await engine.prompt("retry")
    finally:
        await engine.aclose()
        await supervisor.aclose()

    assert result.answer == "done"
    tool_messages = [
        message for message in client.calls[-1]["messages"] if message["role"] == "tool"
    ]
    assert tool_messages[-1]["content"].strip() == "child:next"
    assert _SometimesBlockingEngine.state.cancelled == 1
    assert supervisor.active_calls == 0


async def test_persistent_handles_history_permissions_and_subtree_lifetime(tmp_path):
    from rlm.agent import AgentHandle

    engines = []

    def factory(**kwargs):
        client = DummyClient(
            [
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "saved = 42"})]
                ),
                DummyMessage(content="ready"),
                DummyMessage(tool_calls=[DummyToolCall("wait", {"timeout": 300})]),
            ]
        )
        engine = RLMEngine(client=client, **kwargs)
        engines.append(engine)
        return engine

    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(),
        cwd=str(tmp_path),
        engine_factory=factory,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    parent = supervisor._invocations[supervisor.root_id]
    try:
        child = supervisor._spawn(parent, scope, "research", "researcher", True)
        await asyncio.wait_for(child.done.wait(), 10)
        assert child.status == "idle"
        assert child.engine is not None and not child.engine._closed
        assert (
            AgentHandle(child.id, child.session.dir)
            .history()
            .user_messages()[0]["content"]
            == "research"
        )
        with pytest.raises(ValueError, match="reserved"):
            supervisor._spawn(parent, scope, "duplicate", "researcher", False)
        await supervisor.close_scope(scope)
        scope = await supervisor.open_scope(parent.id)
        assert supervisor._child(parent, "researcher").id == child.id
        child_scope = await supervisor.open_scope(child.id)
        grandchild = supervisor._spawn(child, child_scope, "nested", "researcher", True)
        await asyncio.wait_for(grandchild.done.wait(), 10)
        with pytest.raises(PermissionError):
            supervisor._child(parent, grandchild.id)
        request = {
            "op": "agent.list",
            "capability": parent.capability,
            "scope_id": scope,
            "recursive": True,
        }
        assert len(await supervisor._agent_operation(request)) == 2
        await supervisor._terminate(child)
        assert child.status == grandchild.status == "cancelled"
        assert all(engine._closed for engine in engines)
        assert child.session._msg_file.closed and grandchild.session._msg_file.closed
    finally:
        await supervisor.aclose()
        session.close()
