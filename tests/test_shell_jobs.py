from __future__ import annotations

import asyncio
import json
import subprocess

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
        assert jobs.read(job, 99, 4) == {
            "text": "",
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
    monkeypatch.setattr("rlm.supervisor.DEFAULT_RUN_TIMEOUT", 0.5)

    def tool(code):
        return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])

    client = DummyClient(
        [
            tool(
                "job = await rlm.shell.start('sleep 0.2; [[ -z ${SHELL_TEST_PRIVATE_KEY+x} ]] || exit 90; values=(one two); [[ ${#values[@]} == 2 ]] && printf BASH_OK'); saved_id = job.id"
            ),
            tool(
                "del job; job = await rlm.shell.get(saved_id); assert len(await rlm.shell.list()) == 1"
            ),
            DummyMessage(tool_calls=[DummyToolCall("wait", {"timeout": 5})]),
            tool(r"""
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
    await rlm.shell.start('git log --all')
except RuntimeError:
    pass
else:
    raise AssertionError('Git policy was bypassed')
result = await rlm.shell.run("values=(one two); printf '%s' \"${values[*]}\"; exit 7")
assert result.text == 'one two' and result.exit_code == 7 and not result.ok
argv = await rlm.shell.run(['printf', '%s %s', 'a b', 'c'])
assert (await rlm.shell.getenv()) == {}
overlay = await rlm.shell.setenv(STUDY_PERSIST='1')
assert overlay == {'STUDY_PERSIST': '1'} and (await rlm.shell.getenv()) == overlay
envres = await rlm.shell.run('printf "%s-%s" "$STUDY_PERSIST" "$PER_CALL"', env={'PER_CALL': '2'})
assert envres.ok and envres.text == '1-2'
assert (await rlm.shell.run('printf "%s" "$PER_CALL"')).text == ''
assert argv.ok and argv.text == 'a b c'
for bad in (['ls', 3], []):
    try:
        await rlm.shell.run(bad)
    except TypeError:
        pass
    else:
        raise AssertionError(f'bad argv accepted: {bad!r}')
assert not result.truncated and result.error is None
assert (await (await rlm.shell.get(result.job_id)).read()).text == result.text
result = await rlm.shell.run("printf '%20000s' x")
assert result.truncated and result.text.endswith('x') and 'bytes omitted' in result.text
assert result.text.startswith(' ' * 8192) and len(result.text) < 16384 + 200
job = await rlm.shell.get(result.job_id)
past = await job.read(cursor=10**6)
assert past.text == '' and past.done and past.next_cursor == 20000
big = await job.read(cursor=0, max_bytes=10**6)
assert len(big.text.encode()) == 20000 and big.done  # max_bytes clamped to 64 KiB, output is 20000 bytes
failed = await rlm.shell.run('true', cwd='missing-directory')
assert failed.exit_code is None and failed.error
assert not [e for e in await rlm.inbox.list() if e['type'] == 'shell.completed'], 'run() must not post inbox events'
import asyncio
listed = await rlm.shell.list()
assert listed[-1].handle().id == listed[-1].id and not hasattr(listed[-1], 'read')
assert (await listed[-1].handle().info()).id == listed[-1].id
untimed = await rlm.shell.run('printf server; sleep 30')  # no timeout= -> DEFAULT_RUN_TIMEOUT
assert untimed.timed_out and untimed.text == 'server'
timed = await rlm.shell.run('printf partial; sleep 30', timeout=0.3)
assert timed.timed_out and timed.exit_code is None and timed.text == 'partial'
assert 'timed out' in timed.error
timed_job = await rlm.shell.start('sleep 30', timeout=0.2)
await asyncio.sleep(0.6)
info = await timed_job.info()
assert info.status == 'timed_out' and info.timeout == 0.2 and info.timed_out
try:
    await rlm.shell.run('true', timeout=-1)
except ValueError:
    pass
else:
    raise AssertionError('negative timeout accepted')
waiting = asyncio.create_task(rlm.shell.run('sleep 30'))
while len(await rlm.shell.list()) < 8:
    await asyncio.sleep(0.01)
waiting.cancel()
try:
    await waiting
except asyncio.CancelledError:
    pass
jobs = await rlm.shell.list()
assert jobs[-1].status in ('starting', 'running')
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
        tool_outputs = [
            r.get("content", "") for r in records if r.get("type") == "tool_result"
        ]
        assert any(t.strip() == "SHELL_OK" for t in tool_outputs), tool_outputs[-1][
            -1500:
        ]
    finally:
        supervisor = engine._supervisor
        await engine.aclose()
    jobs = list(supervisor._shell_jobs.jobs.values())
    assert jobs[-1].info.status == "cancelled"
    assert all(job.task.done() for job in jobs)


@pytest.mark.parametrize("failure", ["metadata", "publication"])
async def test_job_failure_does_not_skip_tree_cleanup(session, monkeypatch, failure):
    from pathlib import Path

    from rlm.supervisor import SessionTreeSupervisor
    from test_supervisor import _SometimesBlockingEngine

    _SometimesBlockingEngine.started = asyncio.Event()

    supervisor = SessionTreeSupervisor(
        root_session=session,
        runtime_config=_config(),
        cwd=str(session.dir),
        engine_factory=_SometimesBlockingEngine,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    parent = supervisor._invocations[supervisor.root_id]
    child = supervisor._spawn(parent, scope, "child", "child", True)
    await asyncio.wait_for(child.done.wait(), 5)
    child_scope = await supervisor.open_scope(child.id)
    grandchild = supervisor._spawn(child, child_scope, "wait", "nested", True)
    await asyncio.wait_for(_SometimesBlockingEngine.started.wait(), 5)
    broker_dir = supervisor._broker_dir
    jobs = supervisor._shell_jobs
    for owner in (parent, child):
        jobs.start(
            owner_id=owner.id,
            command="sleep 30",
            cwd=str(session.dir),
            directory=owner.session.dir,
            env={"PATH": "/usr/bin:/bin"},
            source_request_id=None,
        )
    await asyncio.sleep(0.05)

    if failure == "metadata":
        write_text = Path.write_text

        def fail_metadata(path, *args, **kwargs):
            if path.name == "meta.json" and path.parent.parent.name == "jobs":
                raise OSError("job metadata failed")
            return write_text(path, *args, **kwargs)

        monkeypatch.setattr(Path, "write_text", fail_metadata)
    else:

        def fail_publication(job):
            raise OSError("job publication failed")

        monkeypatch.setattr(jobs, "publish", fail_publication)

    await supervisor._terminate(child)
    assert child.engine is None and grandchild.engine is None
    assert child.session._msg_file.closed and grandchild.session._msg_file.closed
    await supervisor.aclose()
    assert all(f"job {failure} failed" in job.info.error for job in jobs.jobs.values())
    assert all(job.task.done() for job in jobs.jobs.values())
    assert not broker_dir.exists()
    assert supervisor._server is None
    assert not supervisor._capabilities


async def test_job_metadata_failure_still_publishes_completion(tmp_path, monkeypatch):
    from pathlib import Path
    from rlm.shell_jobs import ShellJobs

    published = []
    jobs = ShellJobs(lambda job: published.append(job.snapshot()))
    info = jobs.start(
        owner_id="owner",
        command="printf done",
        cwd=str(tmp_path),
        directory=tmp_path,
        env={"PATH": "/usr/bin:/bin"},
        source_request_id=None,
    )
    job = jobs.jobs[info["id"]]
    Path(job.info.output_path).with_name("meta.json").mkdir()
    with pytest.raises(OSError):
        await job.task
    assert job.finished is not None
    assert published[0]["status"] == "completed"
    assert "Metadata persistence failed" in published[0]["error"]
    assert jobs.read(job, 0, 1024)["text"] == "done"


async def test_job_timeout_kills_process_group(tmp_path, monkeypatch):
    monkeypatch.setattr("rlm.shell_jobs.DRAIN_SECONDS", 0.05)
    events = []
    jobs = ShellJobs(events.append)
    try:
        info = jobs.start(
            owner_id="owner",
            command="printf started; sleep 30 & sleep 30; printf never",
            cwd=str(tmp_path),
            directory=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            source_request_id=None,
            timeout=0.3,
        )
        job = jobs.get("owner", info["id"])
        assert job.info.timeout == 0.3
        await asyncio.wait_for(asyncio.shield(job.task), 5)
        assert job.info.status == "timed_out"
        assert job.info.exit_code is None
        assert "timed out after 0.3s" in job.info.error
        assert jobs.read(job, 0, 64)["text"] == "started"
        assert len(events) == 1
        untimed = jobs.start(
            owner_id="owner",
            command="printf ok",
            cwd=str(tmp_path),
            directory=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            source_request_id=None,
        )
        untimed_job = jobs.get("owner", untimed["id"])
        await asyncio.wait_for(asyncio.shield(untimed_job.task), 5)
        assert untimed_job.info.status == "completed"
        assert untimed_job.info.timeout is None
    finally:
        await jobs.close()


async def test_job_that_exited_before_the_deadline_tick_keeps_its_exit_code(
    tmp_path, monkeypatch
):
    """Regression: the deadline check must not relabel an already-exited process."""
    monkeypatch.setattr("rlm.shell_jobs.DRAIN_SECONDS", 0.05)
    real_popen = subprocess.Popen

    def exited_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        process.wait(5)  # the process is finished before the loop's first tick
        return process

    monkeypatch.setattr("rlm.shell_jobs.subprocess.Popen", exited_popen)
    jobs = ShellJobs(lambda job: None)
    try:
        info = jobs.start(
            owner_id="owner",
            command="printf done; exit 3",
            cwd=str(tmp_path),
            directory=tmp_path,
            env={"PATH": "/usr/bin:/bin"},
            source_request_id=None,
            timeout=1e-6,  # already expired when the loop first looks at it
        )
        job = jobs.get("owner", info["id"])
        await asyncio.wait_for(asyncio.shield(job.task), 5)
        assert job.info.status == "completed"
        assert job.info.exit_code == 3
        assert job.info.error is None
        assert jobs.read(job, 0, 64)["text"] == "done"
    finally:
        await jobs.close()
