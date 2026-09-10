from __future__ import annotations

import asyncio
import json

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.supervisor import SessionTreeSupervisor
from rlm.tools.ipython import IPythonREPL
from test_supervisor import _config


def _tool(code):
    return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])


async def test_kernel_exit_preserves_supervisor_resources(session):
    def factory(**kwargs):
        return RLMEngine(
            client=DummyClient(
                [
                    _tool(
                        "child_variable = 42; await rlm.agent.send_to_parent('still here')"
                    ),
                    DummyMessage(content="ready"),
                    _tool("assert child_variable == 42"),
                    DummyMessage(content="resumed"),
                ]
            ),
            **kwargs,
        )

    config = _config(max_depth=1)
    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=config,
        cwd=str(session.dir),
        engine_factory=factory,
    )
    client = DummyClient(
        [
            _tool("""
from pathlib import Path
parent_variable = 99
child = await rlm.agent.spawn('keep state', name='worker', persistent=True)
await child.wait(timeout=10)
events = await rlm.inbox.list()
await rlm.inbox.read(events[0]['id'])
job = await rlm.shell.run('sleep 0.2; printf SURVIVED')
with Path('side-effect').open('a') as stream:
    stream.write('once')
import os
os._exit(7)
"""),
            _tool("""
from pathlib import Path
assert 'parent_variable' not in globals()
assert Path('side-effect').read_text() == 'once'
assert len(await rlm.agent.list()) == 1
child = await rlm.agent.get('worker')
assert (await child.result()).answer == 'ready'
assert child.history().user_messages()
await child.send('verify your Python state')
await child.wait(timeout=10)
assert (await child.result()).answer == 'resumed'
jobs = await rlm.shell.list()
assert len(jobs) == 1
job = await rlm.shell.get(jobs[0].id)
assert (await job.read()).text == 'SURVIVED'
events = await rlm.inbox.list(unread_only=False)
assert sum(event['read'] for event in events) == 1
assert any(not event['read'] for event in events)
for event in await rlm.inbox.list():
    await rlm.inbox.read(event['id'])
print('RECOVERY_OK')
"""),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=config,
        cwd=str(session.dir),
        supervisor=supervisor,
    )
    try:
        await engine.prompt("Verify crash recovery")
        records = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
        ]
        notices = [record for record in records if record["type"] == "kernel_recovery"]
        assert len(notices) == 1
        assert "has been restarted" in notices[0]["message"]["content"]
        assert "not replayed" in notices[0]["message"]["content"]
        assert any(r.get("content", "").strip() == "RECOVERY_OK" for r in records)
        assert any(
            "has been restarted" in m.get("content", "")
            for m in client.calls[1]["messages"]
        )
    finally:
        await engine.aclose()
        await supervisor.aclose()


async def test_failed_restart_is_bounded_and_does_not_execute_cell(
    session, monkeypatch
):
    repl = IPythonREPL(cwd=str(session.dir), session=session)
    await asyncio.to_thread(repl.start)
    attempts = []

    def fail():
        attempts.append(1)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(repl, "restart_kernel", fail)
    repl._recovery_failed = True
    try:
        for _ in range(4):
            assert "not executed" in await asyncio.to_thread(
                repl.execute, "raise AssertionError('must not execute')"
            )
        assert len(attempts) == 3
        notices = repl.take_recovery_notices()
        assert len(notices) == 4
        assert all("restart failed" in notice for notice in notices[:3])
        assert "limit was reached" in notices[-1]
        assert all("has been restarted" not in notice for notice in notices)
    finally:
        await asyncio.to_thread(repl.shutdown)
