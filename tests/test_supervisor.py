from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result
from rlm import broker
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor, depth_capacities
from rlm.types import RLMResult, TokenUsage


def _config(
    *, max_depth: int = 2, max_concurrent: int = 4, max_calls: int = 16
) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig("http://interceptor", "secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=max_depth,
            max_concurrent_subagents=max_concurrent,
            max_subagent_calls=max_calls,
        ),
    )


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

    async def run(self, prompt: str) -> RLMResult:
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


class _BlockingEngine(_FastEngine):
    started = asyncio.Event()

    async def run(self, prompt: str) -> RLMResult:
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


class _SometimesBlockingEngine(_FastEngine):
    started = asyncio.Event()

    async def run(self, prompt: str) -> RLMResult:
        if prompt != "wait":
            return await super().run(prompt)
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

    async def run(self, prompt: str) -> RLMResult:
        depth = self.runtime_config.invocation.depth
        if depth == self.runtime_config.policy.max_depth:
            return RLMResult(answer=f"leaf:{depth}", session_dir=self.session.dir)
        scope = await self.supervisor.open_scope(self.invocation_id)
        endpoint = self.supervisor.endpoint_for(self.invocation_id)
        try:
            task = await self.supervisor._start_child(
                endpoint.capability, scope, prompt, {}
            )
            result = await task
        finally:
            await self.supervisor.close_scope(scope)
        return RLMResult(
            answer=f"depth:{depth}>{result.answer}", session_dir=self.session.dir
        )


class _ClosableClient:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


def test_depth_capacities_reserve_every_level():
    assert depth_capacities(max_depth=3, limit=7) == {1: 3, 2: 2, 3: 2}
    assert depth_capacities(max_depth=4, limit=3, root_depth=2) == {3: 2, 4: 1}
    with pytest.raises(ValueError, match="every recursive depth"):
        depth_capacities(max_depth=3, limit=2)


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
            await supervisor._start_child(endpoint.capability, scope, str(i), {})
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
            await supervisor._start_child(endpoint.capability, scope, str(i), {})
            for i in range(3)
        ]
        results = await asyncio.gather(*tasks)
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert supervisor.total_calls == 2
    assert (
        sum(result.answer == "[recursive call limit reached]" for result in results)
        == 1
    )


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
            await supervisor._start_child(endpoint.capability, scope, str(i), {})
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


async def test_broker_round_trip_and_invalid_capability(tmp_path):
    _FastEngine.state = _EngineState()
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1),
        cwd=str(tmp_path),
        engine_factory=_FastEngine,
    )
    await supervisor.start()
    socket_path = supervisor.endpoint_for(supervisor.root_id).socket_path
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    broker.configure(endpoint)
    broker.set_scope(scope)
    try:
        result = await broker.run("hello")
        broker.configure(broker.BrokerEndpoint(endpoint.socket_path, "invalid"))
        with pytest.raises(RuntimeError, match="invalid recursive RLM capability"):
            await broker.run("forbidden")
    finally:
        broker.set_scope(None)
        broker.configure(None)
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()

    assert result.answer == "child:hello"
    assert result.usage == TokenUsage(prompt_tokens=2, completion_tokens=1)
    assert not Path(socket_path).exists()


async def test_client_disconnect_cancels_child(tmp_path):
    _BlockingEngine.state = _EngineState()
    _BlockingEngine.started = asyncio.Event()
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1),
        cwd=str(tmp_path),
        engine_factory=_BlockingEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    broker.configure(supervisor.endpoint_for(supervisor.root_id))
    broker.set_scope(scope)
    pending = asyncio.create_task(broker.run("wait"))
    await asyncio.wait_for(_BlockingEngine.started.wait(), timeout=2)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    for _ in range(100):
        if supervisor.active_calls == 0:
            break
        await asyncio.sleep(0.01)
    try:
        assert supervisor.active_calls == 0
        assert _BlockingEngine.state.cancelled == 1
    finally:
        broker.set_scope(None)
        broker.configure(None)
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        session.close()


async def test_shutdown_closes_incomplete_broker_connections(tmp_path):
    session = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(max_depth=1),
        cwd=str(tmp_path),
        engine_factory=_FastEngine,
    )
    await supervisor.start()
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    reader, writer = await asyncio.open_unix_connection(endpoint.socket_path)
    for _ in range(100):
        if supervisor._connection_writers:
            break
        await asyncio.sleep(0.01)
    assert len(supervisor._connection_writers) == 1

    await supervisor.aclose()
    try:
        assert await asyncio.wait_for(reader.read(), timeout=1) == b""
        assert not Path(endpoint.socket_path).exists()
    finally:
        writer.close()
        await writer.wait_closed()
        session.close()


async def test_real_kernel_uses_brokered_rlm_callable(monkeypatch, session):
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
import os, subprocess, rlm.api, rlm.config
secret_names = {
    'OPENAI_API_KEY', 'OPENAI_BASE_URL', 'PRIME_API_KEY',
    'PRIME_TEAM_ID', 'RLM_API_KEY', 'RLM_BASE_URL',
}
child = await rlm('hello')
other = await rlm.api.run('again')
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


async def test_real_kernel_cancels_child_and_remains_reusable(session):
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
                tool_calls=[DummyToolCall("ipython", {"code": "await rlm('wait')"})]
            ),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {"code": "child = await rlm('next'); print(child.answer)"},
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


async def test_engine_closes_only_owned_inference_client(monkeypatch, tmp_path):
    owned = _ClosableClient()
    monkeypatch.setattr("rlm.engine.make_client", lambda *args: owned)
    owned_engine = RLMEngine(
        runtime_config=_config(max_depth=0),
        session=Session(tmp_path / "owned"),
    )
    await owned_engine.aclose()

    injected = _ClosableClient()
    injected_engine = RLMEngine(
        client=injected,  # type: ignore[arg-type]
        runtime_config=_config(max_depth=0),
        session=Session(tmp_path / "injected"),
    )
    await injected_engine.aclose()

    assert owned.closed is True
    assert injected.closed is False


async def test_sync_close_requires_await_for_owned_async_resources(
    monkeypatch, tmp_path
):
    monkeypatch.setattr("rlm.engine.make_client", lambda *args: _ClosableClient())
    engine = RLMEngine(
        runtime_config=_config(max_depth=0),
        session=Session(tmp_path / "root"),
    )

    with pytest.raises(RuntimeError, match="await engine.aclose"):
        engine.close()

    await engine.aclose()


async def test_engine_close_finishes_before_propagating_cancellation(
    monkeypatch, tmp_path
):
    started = asyncio.Event()
    release = asyncio.Event()

    class SlowClosableClient(_ClosableClient):
        async def close(self) -> None:
            started.set()
            await release.wait()
            await super().close()

    owned = SlowClosableClient()
    monkeypatch.setattr("rlm.engine.make_client", lambda *args: owned)
    engine = RLMEngine(
        runtime_config=_config(max_depth=0),
        session=Session(tmp_path / "root"),
    )
    waiter = asyncio.create_task(engine.aclose())
    await started.wait()
    waiter.cancel()
    await asyncio.sleep(0)
    assert waiter.done() is False

    release.set()
    with pytest.raises(asyncio.CancelledError):
        await waiter
    await engine.aclose()

    assert owned.closed is True
