from __future__ import annotations

import asyncio
import json

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.shell_jobs import ShellJobs
from test_supervisor import _config


async def test_bash_capture_limits_failure_and_cleanup(tmp_path, monkeypatch):
    monkeypatch.setattr("rlm.shell_jobs.MAX_OUTPUT_BYTES", 8)
    monkeypatch.setattr("rlm.shell_jobs.DRAIN_SECONDS", 0.05)
    events = []
    jobs = ShellJobs(events.append)

    def start(command, cwd=None):
        info = jobs.start(
            owner_id="owner",
            command=command,
            cwd=str(cwd or tmp_path),
            directory=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            source_request_id=None,
        )
        return jobs.get("owner", info["id"])

    try:
        job = start(
            "a=(hello world); [[ ${#a[@]} == 2 ]] && printf '%s' \"${a[*]}\"; printf '!'; exit 7"
        )
        await asyncio.wait_for(asyncio.shield(job.task), 5)
        assert job.info.exit_code == 7
        assert job.info.output_complete
        assert job.info.output_truncated
        assert jobs.read(job, 0, 4)["text"] == "hell"
        assert jobs.read(job, 4, 4) == {
            "text": "o wo",
            "next_cursor": 8,
            "done": True,
            "truncated": True,
        }
        with pytest.raises(PermissionError):
            jobs.get("another-agent", job.info.id)
        failed = start("true", tmp_path / "missing")
        await failed.task
        assert failed.info.status == "failed"
        assert failed.info.error
        held = start("sleep 30 & echo $! > descendant.pid")
        await asyncio.wait_for(asyncio.shield(held.task), 5)
        assert not held.info.output_complete
        assert held.info.output_truncated
        cancelled = start("sleep 30")
        await asyncio.sleep(0.05)
        await asyncio.wait_for(jobs.cancel(cancelled), 5)
        assert cancelled.info.status == "cancelled"
        assert len(events) == 4
        metadata = json.loads(
            (tmp_path / "jobs" / job.info.id / "meta.json").read_text()
        )
        assert metadata["exit_code"] == 7
    finally:
        await jobs.close()


async def test_real_kernel_shell_handle_recovery_and_inbox(session, monkeypatch):
    monkeypatch.setenv("SHELL_TEST_PRIVATE_KEY", "must-not-leak")

    def tool(code):
        return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])

    client = DummyClient(
        [
            tool(
                "job = await rlm.shell.run('sleep 0.2; [[ -z ${SHELL_TEST_PRIVATE_KEY+x} ]] || exit 90; values=(one two); [[ ${#values[@]} == 2 ]] && printf BASH_OK'); saved_id = job.id"
            ),
            tool(
                "del job; job = await rlm.shell.get(saved_id); assert len(await rlm.shell.list()) == 1"
            ),
            DummyMessage(tool_calls=[DummyToolCall("wait", {"timeout": 5})]),
            tool("""
events = await rlm.inbox.list()
assert len(events) == 1
event = await rlm.inbox.read(events[0]['id'])
assert event['type'] == 'shell.completed'
assert event['content']['job_id'] == saved_id
assert (await job.info()).exit_code == 0
output = await job.read()
assert output.text == 'BASH_OK' and output.done
assert (await job.read()).text == output.text
try:
    await rlm.shell.run('git log --all')
except RuntimeError:
    pass
else:
    raise AssertionError('Git policy was bypassed')
await rlm.shell.run('sleep 30')
print('SHELL_OK')
"""),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,
        session=session,
        runtime_config=_config(max_depth=0),
        cwd=str(session.dir),
    )
    try:
        await engine.prompt("Exercise Bash jobs")
        records = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
        ]
        assert any(
            r.get("type") == "tool_result"
            and r.get("content", "").strip() == "SHELL_OK"
            for r in records
        )
    finally:
        supervisor = engine._supervisor
        await engine.aclose()
    jobs = list(supervisor._shell_jobs.jobs.values())
    assert jobs[-1].info.status == "cancelled"
    assert all(job.task.done() for job in jobs)
