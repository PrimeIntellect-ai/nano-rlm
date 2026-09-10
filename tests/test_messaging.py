from __future__ import annotations

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
                    code="saved = 41; await rlm.agent.send_to_parent('progress')",
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
                code="worker = await rlm.agent.spawn('initial task', name='worker', persistent=True); await worker.send('queued task'); await worker.steer('steering task')",
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
        assert initial_answer < queued_instruction
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
            "agent_queue",
            "agent_steer",
            "agent_report",
            "agent_completion",
            "subagent_call",
            "subagent_return",
        } <= edge_types
    finally:
        await supervisor.aclose()
