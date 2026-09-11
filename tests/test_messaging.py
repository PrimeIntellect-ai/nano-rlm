from __future__ import annotations

import asyncio

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.history import history
from rlm.supervisor import SessionTreeSupervisor
from test_supervisor import _config


def _tool(name, **arguments):
    return DummyMessage(tool_calls=[DummyToolCall(name, arguments)])


async def test_real_kernels_queue_steer_report_wait_and_resume(session):
    children = []

    def factory(**kwargs):
        client = DummyClient(
            [
                _tool(
                    "ipython",
                    code=f"import asyncio\nfrom pathlib import Path\nwhile not Path({str(session.dir / 'instructions-ready')!r}).exists():\n    await asyncio.sleep(0.01)\nsaved = 41; await rlm.agent.send_to_parent('progress')",
                ),
                DummyMessage(content="initial answer"),
                _tool("ipython", code="saved += 1; print(saved)"),
                DummyMessage(content="queued answer"),
                _tool(
                    "ipython",
                    code="print(saved); await rlm.agent.send_to_parent('resumed')",
                ),
                DummyMessage(content="followup answer"),
            ]
        )
        children.append(client)
        return RLMEngine(client=client, **kwargs)

    config = _config(max_depth=1)
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=factory,
    )
    client = DummyClient(
        [
            _tool(
                "ipython",
                code=f"worker = await rlm.agent.spawn('initial task', name='worker', persistent=True); await worker.send('queued task'); await worker.steer('steering task'); from pathlib import Path; Path({str(session.dir / 'instructions-ready')!r}).touch()",
            ),
            _tool("wait", timeout=10),
            _tool(
                "ipython",
                code="""
await worker.wait(timeout=10)
assert (await worker.result()).answer == 'queued answer'
events = await rlm.inbox.list()
assert len(events) == 2
assert all('content' not in event for event in events)
for event in events:
    await rlm.inbox.read(event['id'])
assert await rlm.inbox.list() == []
assert len(await rlm.inbox.list(unread_only=False)) == 2
await worker.send('followup task')
""",
            ),
            _tool("wait", timeout=10),
            _tool(
                "ipython",
                code="""
await worker.wait(timeout=10)
assert (await worker.result()).answer == 'followup answer'
for event in await rlm.inbox.list():
    await rlm.inbox.read(event['id'])
print('MESSAGING_OK')
""",
            ),
            _tool("wait", timeout=0),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=config,
        supervisor=supervisor,
        invocation_id=supervisor.root_id,
    )
    try:
        result = await engine.run("coordinate")
        assert result.answer == "done"
        logs = history(session.dir)
        tool_text = "\n".join(
            e["content"] for e in logs.events if e["type"] == "tool_result"
        )
        assert "AssertionError" not in tool_text
        assert "Traceback" not in tool_text
        assert "MESSAGING_OK" in tool_text
        assert logs.messages[-2]["content"] == "Wait timed out."
        child = next(
            agent
            for agent in supervisor._invocations.values()
            if agent.parent_id == supervisor.root_id
        )
        child_history = history(child.session.dir)
        instructions = [
            e["message"]["content"]
            for e in child_history.events
            if e["type"] == "parent_message"
        ]
        assert any("steering task" in text for text in instructions)
        assert any("queued task" in text for text in instructions)
        assert any("followup task" in text for text in instructions)
        messages = child_history.messages
        initial_answer = next(
            i for i, m in enumerate(messages) if m.get("content") == "initial answer"
        )
        queued_instruction = next(
            i
            for i, m in enumerate(messages)
            if m.get("content") == "Parent instruction:\nqueued task"
        )
        steering_instruction = next(
            i
            for i, m in enumerate(messages)
            if m.get("content") == "Parent instruction:\nsteering task"
        )
        assert steering_instruction < initial_answer < queued_instruction
        assert any(
            e["type"] == "tool_result" and e["content"].strip() == "42"
            for e in child_history.events
        )
        assert (
            len([e for e in logs.events if e["type"] == "supervisor_notification"]) > 0
        )
        assert len(supervisor._invocations[supervisor.root_id].inbox) == 4
        assert all(e["read"] for e in supervisor._invocations[supervisor.root_id].inbox)
        edge_types = {
            edge["type"] for edge in supervisor.semantic_edges.snapshot()["edges"]
        }
        assert {
            "agent_message",
            "subagent_call",
            "subagent_return",
        } <= edge_types
    finally:
        await supervisor.aclose()


@pytest.mark.parametrize("budget", ["turns", "tokens"])
@pytest.mark.parametrize("state", ["idle", "running"])
async def test_followup_rejected_at_budget_limit(session, budget, state):
    from test_supervisor import _SometimesBlockingEngine

    _SometimesBlockingEngine.started = asyncio.Event()

    config = _config()
    config = config.model_copy(
        update={"policy": config.policy.model_copy(update={f"max_total_{budget}": 10})}
    )
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=_SometimesBlockingEngine,
    )
    await supervisor.start()
    try:
        scope = await supervisor.open_scope(supervisor.root_id)
        parent = supervisor._invocations[supervisor.root_id]
        child = supervisor._spawn(
            parent, scope, "wait" if state == "running" else "initial", "worker", True
        )
        await asyncio.wait_for(
            _SometimesBlockingEngine.started.wait()
            if state == "running"
            else child.done.wait(),
            5,
        )
        assert child.status == state
        previous = child.result
        setattr(supervisor, f"_total_{budget}", 10)
        for op in ("agent.send", "agent.steer"):
            with pytest.raises(RuntimeError, match="tree budget exhausted"):
                await supervisor._agent_operation(
                    {
                        "op": op,
                        "capability": parent.capability,
                        "scope_id": scope,
                        "agent_id": child.id,
                        "message": "follow up",
                    }
                )
        assert not child.instructions
        assert child.result is previous
        assert child.done.is_set() == (state == "idle")
        assert not (child.session.dir / "inbox.jsonl").exists()
    finally:
        await supervisor.aclose()


@pytest.mark.parametrize("terminal_status", ["failed", "cancelled"])
async def test_auto_wake_preserves_completed_result_for_waiter(
    session, terminal_status
):
    from types import SimpleNamespace
    from rlm.types import RLMResult, TokenUsage

    started = asyncio.Event()
    finish = asyncio.Event()
    resumed = asyncio.Event()
    calls = 0

    async def prompt(task, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            started.set()
            await finish.wait()
            return RLMResult(
                answer="first answer",
                session_dir=session.dir,
                usage=TokenUsage(),
                turns=1,
            )
        resumed.set()
        await asyncio.Future()

    async def close():
        pass

    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(),
        cwd=str(session.dir),
        engine_factory=lambda **kwargs: SimpleNamespace(prompt=prompt, aclose=close),
    )
    await supervisor.start()
    try:
        scope = await supervisor.open_scope(supervisor.root_id)
        parent = supervisor._invocations[supervisor.root_id]
        child = supervisor._spawn(parent, scope, "task", "worker", True)
        await started.wait()
        request = {
            "capability": parent.capability,
            "scope_id": scope,
            "agent_id": child.id,
        }
        assert (
            await supervisor._agent_operation({**request, "op": "agent.result"}) is None
        )
        waiter = asyncio.create_task(
            supervisor._agent_operation({**request, "op": "agent.wait", "timeout": 5})
        )
        await asyncio.sleep(0)
        supervisor._publish(
            child, supervisor._event(parent, "agent.message", "new activity", None)
        )
        finish.set()
        await asyncio.wait_for(waiter, 5)
        await asyncio.wait_for(resumed.wait(), 5)
        assert not child.done.is_set()
        assert parent.inbox[-1]["type"] == "agent.completed"
        result = await supervisor._agent_operation({**request, "op": "agent.result"})
        assert result["answer"] == "first answer"
        child.status = terminal_status
        child.error = "follow-up stopped"
        assert not child.done.is_set()
        with pytest.raises(RuntimeError, match="follow-up stopped"):
            await supervisor._agent_operation({**request, "op": "agent.result"})
        child.status = "running"
    finally:
        await supervisor.aclose()


async def test_inbox_persistence_failure_preserves_delivery(session, monkeypatch):
    from pathlib import Path

    supervisor = SessionTreeSupervisor(
        root_session=session, runtime_config=_config(), cwd=str(session.dir)
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    owner = supervisor._invocations[supervisor.root_id]
    original = Path.open

    def fail_journal(path, *args, **kwargs):
        if path.name == "inbox.jsonl":
            raise OSError("journal unavailable")
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", fail_journal)
    try:
        event = supervisor._event(owner, "agent.completed", {"agent_id": "child"}, None)
        supervisor._publish(owner, event)
        assert owner.changed.is_set()
        assert (
            "Events remain available in memory only"
            in supervisor.inbox_notification(owner.id)
        )
        delivered = await supervisor._agent_operation(
            dict(
                op="inbox.read",
                capability=endpoint.capability,
                scope_id=scope,
                event_id=event["id"],
            )
        )
        assert delivered["content"] == {"agent_id": "child"}
        assert delivered["read"]
    finally:
        await supervisor.aclose()


@pytest.mark.parametrize("budget", ["turns", "tokens"])
@pytest.mark.parametrize("persistent", [False, True])
@pytest.mark.parametrize("tool_reply", [False, True])
async def test_running_instruction_failure_at_tree_budget(
    session, budget, persistent, tool_reply
):
    import json

    started, release = asyncio.Event(), asyncio.Event()

    class Client(DummyClient):
        async def create(self, **kwargs):
            started.set()
            await release.wait()
            return await super().create(**kwargs)

    reply = _tool("add", a=1, b=2) if tool_reply else DummyMessage(content="finished")
    client = Client([reply])
    config = _config().model_copy(update={"builtin_tools": ("add",)})
    config = config.model_copy(
        update={
            "policy": config.policy.model_copy(
                update={f"max_total_{budget}": 1 if budget == "turns" else 2}
            )
        }
    )
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=lambda **kwargs: RLMEngine(client=client, **kwargs),
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    parent = supervisor._invocations[supervisor.root_id]
    child = supervisor._spawn(parent, scope, "task", "worker", persistent)
    try:
        await asyncio.wait_for(started.wait(), 5)
        ids = []
        for op in ("agent.send", "agent.steer"):
            ids.append(
                await supervisor._agent_operation(
                    dict(
                        op=op,
                        capability=parent.capability,
                        scope_id=scope,
                        agent_id=child.id,
                        message="must not be marked delivered",
                    )
                )
            )
        release.set()
        await asyncio.wait_for(child.done.wait(), 5)
        failures = [
            e["content"] for e in parent.inbox if e["type"] == "agent.delivery_failed"
        ]
        assert {e["message_id"] for e in failures} == set(ids)
        assert all(
            e["agent_id"] == child.id and e["reason"] == "tree_budget_exhausted"
            for e in failures
        )
        assert not child.instructions and len(client.calls) == 1
        records = [
            json.loads(line)
            for line in (child.session.dir / "inbox.jsonl").read_text().splitlines()
        ]
        assert {
            e["event_id"] for e in records if e["type"] == "instruction_failed"
        } == set(ids)
        assert not any(e["type"] == "instructions_delivered" for e in records)
        assert not any(
            "must not be marked delivered" in str(m.get("content", ""))
            for m in history(child.session.dir).messages
        )
    finally:
        release.set()
        await supervisor.aclose()
