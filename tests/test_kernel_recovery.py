from __future__ import annotations

import asyncio
import json

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.supervisor import SessionTreeSupervisor
from rlm.tools.ipython import IPythonREPL, _KernelDied
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


async def test_cancel_during_scope_setup_does_not_submit_cell(session, monkeypatch):
    from rlm.broker import BrokerEndpoint

    repl = IPythonREPL(cwd=str(session.dir), session=session)
    await asyncio.to_thread(repl.start)
    execute_silent = repl._execute_silent

    def cancel_after_setup(code, **kwargs):
        execute_silent("pass")
        repl._interrupt_requested.set()

    monkeypatch.setattr(repl, "_execute_silent", cancel_after_setup)
    repl.broker_endpoint = BrokerEndpoint("unused", "unused")
    try:
        assert await asyncio.to_thread(repl.execute, "must_not_exist = 1") == ""
        repl.broker_endpoint = None
        assert "False" in await asyncio.to_thread(
            repl.execute, "print('must_not_exist' in globals())"
        )
    finally:
        await asyncio.to_thread(repl.shutdown)


async def test_interrupt_ignores_setup_idle_and_recovers_unresponsive_cell(session):
    repl = IPythonREPL(cwd=str(session.dir), session=session)
    await asyncio.to_thread(repl.start)
    try:
        # Silent setup leaves an unrelated idle message on the IOPub channel.
        await asyncio.to_thread(repl._execute_silent, "saved_before_restart = 42")
        msg_id = repl._kc.execute(
            "import signal, time; signal.signal(signal.SIGINT, signal.SIG_IGN); "
            "print('started', flush=True); time.sleep(30)"
        )
        pending = []
        while True:
            message = await asyncio.to_thread(repl._kc.get_iopub_msg, timeout=5)
            if (
                message["parent_header"].get("msg_id") == msg_id
                and message["msg_type"] == "stream"
            ):
                break
            pending.append(message)
        # Put the observed setup messages back ahead of the cell's remaining messages.
        from unittest.mock import patch

        get_message = repl._kc.get_iopub_msg

        def with_stale_messages(*args, **kwargs):
            return pending.pop(0) if pending else get_message(*args, **kwargs)

        with patch.object(repl._kc, "get_iopub_msg", with_stale_messages):
            await asyncio.to_thread(repl._interrupt_and_recover, msg_id)
        assert any("has been restarted" in n for n in repl.take_recovery_notices())
        assert "False" in await asyncio.to_thread(
            repl.execute, "print('saved_before_restart' in globals())"
        )
    finally:
        await asyncio.to_thread(repl.shutdown)


@pytest.mark.parametrize(
    "error",
    [TimeoutError("setup timed out"), RuntimeError("setup failed"), _KernelDied()],
)
async def test_scope_failure_recovers_without_executing_cell(
    session, monkeypatch, error
):
    from rlm.broker import BrokerEndpoint

    repl = IPythonREPL(cwd=str(session.dir), session=session)
    await asyncio.to_thread(repl.start)
    execute_silent = repl._execute_silent

    def fail_scope(code, *, interruptible=False):
        if interruptible:
            raise error
        return execute_silent(code)

    monkeypatch.setattr(repl, "_execute_silent", fail_scope)
    repl.broker_endpoint = BrokerEndpoint("unused", "unused")
    try:
        result = await asyncio.to_thread(repl.execute, "must_not_exist = 1")
        assert "not executed" in result
        notices = repl.take_recovery_notices()
        assert len(notices) == 1 and "has been restarted" in notices[0]
        assert "was not submitted" in notices[0]
        repl.broker_endpoint = None
        assert "False" in await asyncio.to_thread(
            repl.execute, "print('must_not_exist' in globals())"
        )
    finally:
        await asyncio.to_thread(repl.shutdown)


async def test_cancel_interrupts_pending_scope_setup(session, monkeypatch):
    from rlm.broker import BrokerEndpoint

    repl = IPythonREPL(cwd=str(session.dir), session=session)
    await asyncio.to_thread(repl.start)
    execute_silent = repl._execute_silent
    marker = session.dir / "setup-started"

    def slow_scope(code, *, interruptible=False):
        if interruptible:
            code = f"from pathlib import Path; import time; Path({str(marker)!r}).touch(); time.sleep(30)"
        return execute_silent(code, interruptible=interruptible)

    monkeypatch.setattr(repl, "_execute_silent", slow_scope)
    repl.broker_endpoint = BrokerEndpoint("unused", "unused")
    pending = asyncio.create_task(asyncio.to_thread(repl.execute, "must_not_exist = 1"))
    try:

        async def started():
            while not marker.exists():
                await asyncio.sleep(0.01)

        await asyncio.wait_for(started(), 5)
        repl._interrupt_requested.set()
        assert await asyncio.wait_for(asyncio.shield(pending), 5) == ""
        repl.broker_endpoint = None
        assert "False" in await asyncio.to_thread(
            repl.execute, "print('must_not_exist' in globals())"
        )
    finally:
        repl._interrupt_requested.set()
        await pending
        await asyncio.to_thread(repl.shutdown)
