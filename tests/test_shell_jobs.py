from __future__ import annotations

import asyncio
import json
import subprocess

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.shell_jobs import ShellJobs
from test_supervisor import _config


async def test_result_yields_before_cell_deadline(session):
    config = _config()
    config = config.model_copy(
        update={"policy": config.policy.model_copy(update={"exec_timeout": 2})}
    )
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "job = await rlm.shell.run('printf small; sleep 30', yield_after=0)"
                        },
                    )
                ]
            ),
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": "result = await job.result(); assert result.running; assert not result.truncated; print('YIELDED')"
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)
    try:
        await engine.prompt("Run a job")
        outputs = [m["content"] for m in session.messages if m["role"] == "tool"]
        assert any("YIELDED" in text for text in outputs), outputs
        assert not any("execution timed out" in text for text in outputs)
    finally:
        await engine.aclose()


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
    monkeypatch.setattr("rlm.supervisor.RUN_DETACH_SECONDS", 0.5)

    def tool(code):
        return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])

    client = DummyClient(
        [
            tool(
                "job = await rlm.shell.run('sleep 1.5; [[ -z ${SHELL_TEST_PRIVATE_KEY+x} ]] || exit 90; values=(one two); [[ ${#values[@]} == 2 ]] && printf BASH_OK', yield_after=0); saved_id = job.id"
            ),
            tool(
                "del job; job = await rlm.shell.get(saved_id); assert len(await rlm.shell.list()) == 1"
            ),
            DummyMessage(
                tool_calls=[DummyToolCall("wait", {"timeout": 400})]
            ),  # clamped
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
    await rlm.shell.run('git log --all', yield_after=0)
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
piped = await rlm.shell.run('(printf out; exit 3) | tail -1')
assert piped.exit_code == 3 and piped.text == 'out' and not piped.ok  # pipefail on
assert argv.ok and argv.text == 'a b c'
for bad in (['ls', 3], []):
    try:
        await rlm.shell.run(bad)
    except TypeError:
        pass
    else:
        raise AssertionError(f'bad argv accepted: {bad!r}')
assert not result.truncated and result.error is None
assert (await (await rlm.shell.get(result.id)).read()).text == result.text
result = await rlm.shell.run("printf '%20000s' x")
assert result.truncated and result.text.endswith('x') and 'bytes omitted' in result.text
assert result.text.startswith(' ' * 8192) and len(result.text) < 16384 + 200
job = await rlm.shell.get(result.id)
past = await job.read(cursor=10**6)
assert past.text == '' and past.done and past.next_cursor == 20000
big = await job.read(cursor=0, max_bytes=10**6)
assert len(big.text.encode()) == 20000 and big.done  # max_bytes clamped to 64 KiB, output is 20000 bytes
failed = await rlm.shell.run('true', cwd='missing-directory')
assert failed.exit_code is None and failed.error
assert not [e for e in await rlm.inbox.list() if e['type'] == 'shell.completed'], 'run() must not post inbox events'
import asyncio
listed = await rlm.shell.list()
assert not hasattr(listed[-1], 'read') and not hasattr(listed[-1], 'handle')
assert (await (await rlm.shell.get(listed[-1].id)).info()).id == listed[-1].id
snap = await rlm.shell.run('sleep 1; printf slow', yield_after=0.2)  # yielding leaves the job running
assert snap.running and snap.exit_code is None and not snap.ok and snap.text.startswith('[yielded after')
bounded = await snap.result()  # result() waits for it
assert bounded.ok and bounded.text == 'slow' and not bounded.running and bounded.id == snap.id
assert (await snap.result()).text == 'slow'  # repeatable
detached = await rlm.shell.run('printf server; sleep 30')  # detaches after RUN_DETACH_SECONDS
assert detached.running and not detached.ok and detached.exit_code is None
assert detached.text.startswith('[yielded after') and detached.text.endswith('server')
assert f'rlm.shell.get("{detached.id}")).result()' in detached.text and 'await job.' not in detached.text  # no invented variable name
still = await detached.result(yield_after=0.2)  # bounded wait, still running
assert still.running and still.text.startswith('[yielded after 0.2 s')
bg = await rlm.shell.get(detached.id)
assert bg.running and bg.text == '' and (await bg.info()).status == 'running'
await bg.cancel()
assert (await bg.info()).status == 'cancelled'
assert (await rlm.hints.muted()) == []
assert (await rlm.hints.mute('run-detach')) == ['run-detach']
muted_run = await rlm.shell.run('sleep 30')  # detaches again, but the hint is muted now
assert muted_run.running
await muted_run.cancel()
assert (await rlm.hints.unmute('run-detach')) == []
quick = await rlm.shell.run('printf BG', yield_after=0)
assert quick.running and quick.text.startswith('[started: job ' + quick.id) and quick.exit_code is None and not quick.truncated
assert f'await (await rlm.shell.get("{quick.id}")).result()' in quick.text  # copyable collection expression
res = await quick.result()
assert res.ok and res.text == 'BG' and (await quick.result()).text == 'BG'
a_job = await rlm.shell.run('sleep 0.2; printf A', yield_after=0)
b_job = await rlm.shell.run('sleep 0.1; printf B', yield_after=0)
a_res, b_res = await asyncio.gather(a_job.result(), b_job.result())
assert a_res.text == 'A' and b_res.text == 'B' and a_res.ok and b_res.ok
all_completed = [e for e in await rlm.inbox.list(unread_only=False) if e['type'] == 'shell.completed']
read_before = {e['id'] for e in all_completed if e['read']}
completed = [await rlm.inbox.read(e['id']) for e in all_completed]
by_job = {e['content']['job_id']: e['content'] for e in completed}
assert by_job[quick.id]['text'] == 'BG' and by_job[quick.id]['exit_code'] == 0  # the event carries the tail
assert by_job[detached.id]['status'] == 'cancelled'
assert {e['content']['job_id'] for e in completed if e['id'] in read_before} >= {quick.id, a_job.id, b_job.id, snap.id}  # result() marked them read
assert detached.id not in {e['content']['job_id'] for e in completed if e['id'] in read_before}  # cancelled without result(): still unread, but quiet
bounded_wait = await rlm.shell.run('printf partial; sleep 30', yield_after=0.3)  # a wait bound never kills
assert bounded_wait.running and bounded_wait.exit_code is None and bounded_wait.text.endswith('partial')
assert (await bounded_wait.info()).status == 'running' and (await bounded_wait.info()).timeout is None
await bounded_wait.cancel()
assert (await bounded_wait.info()).status == 'cancelled'
long_wait = await rlm.shell.run('printf capped', yield_after=10**6)  # above the cap: clamped to 300 s, finishes anyway
assert long_wait.ok and long_wait.text == 'capped'
timed = await rlm.shell.run('printf partial; sleep 30', timeout=0.3)  # timeout= kills; the default wait outlives it
assert timed.timed_out and not timed.running and timed.exit_code is None and timed.text == 'partial' and 'timed out' in timed.error
killed = await rlm.shell.run('printf part2; sleep 30', yield_after=0.2, timeout=0.8)  # wait ends first, kill lands later
assert killed.running and not killed.timed_out
killed = await killed.result()
assert killed.timed_out and not killed.running and killed.exit_code is None and killed.text == 'part2'
timed_job = await rlm.shell.run('sleep 30', yield_after=0, timeout=0.2)
await asyncio.sleep(0.6)
info = await timed_job.info()
assert info.status == 'timed_out' and info.timeout == 0.2 and info.timed_out
try:
    await rlm.shell.run('true', yield_after=-1)
except ValueError:
    pass
else:
    raise AssertionError('negative yield_after accepted')
try:
    await rlm.shell.run('true', background=True)
except TypeError as exc:
    assert 'yield_after=0' in str(exc), str(exc)
else:
    raise AssertionError('background kwarg accepted')
try:
    await rlm.shell.run('true', yield_after=-1)
except ValueError:
    pass
else:
    raise AssertionError('negative timeout accepted')
for _ in range(4):
    await rlm.shell.run('cd . && HINT_VAR=/tmp/x true')  # the third prefix earns one env-prefix hint
before = len(await rlm.shell.list())
waiting = asyncio.create_task(rlm.shell.run('sleep 30'))
while len(await rlm.shell.list()) <= before:
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
        assert any(
            t.startswith("Note: wait timeout clamped from 400 to 300")
            for t in tool_outputs
        )
        hint_msgs = [
            str(m.get("content", ""))
            for call in client.calls
            for m in call["messages"]
            if m.get("role") == "user"
            and "command still running (.running is True" in str(m.get("content", ""))
        ]
        assert hint_msgs, "detach hint never reached the model"
        assert ")).result()" in hint_msgs[0] and "await job." not in hint_msgs[0]
        wait_hints = [
            str(m.get("content", ""))
            for m in client.calls[-1]["messages"]
            if m.get("role") == "user"
            and "You called wait while holding running job" in str(m.get("content", ""))
        ]
        assert (
            len(wait_hints) == 1 and 'rlm.hints.mute("wait-held-job")' in wait_hints[0]
        )
        assert 'rlm.hints.mute("run-detach")' in hint_msgs[0]
        assert len(set(hint_msgs)) == 1  # the second detach happened after mute()
        env_hints = [
            str(m.get("content", ""))
            for call in client.calls
            for m in call["messages"]
            if m.get("role") == "user" and "HINT_VAR=" in str(m.get("content", ""))
        ]
        assert env_hints and "rlm.shell.setenv(HINT_VAR='/tmp/x')" in env_hints[0]
        assert 'rlm.hints.mute("env-prefix")' in env_hints[0]
        assert (
            len(set(env_hints)) == 1
        )  # hinted once per variable, not on the fourth use
    finally:
        supervisor = engine._supervisor
        owner = supervisor._invocations[supervisor.root_id]
        await engine.aclose()
    jobs = list(supervisor._shell_jobs.jobs.values())
    assert jobs[-1].info.status == "cancelled"
    assert all(job.task.done() for job in jobs)
    records = [
        json.loads(line)
        for line in (session.dir / "inbox.jsonl").read_text().splitlines()
    ]
    recorded_reads = {e["event_id"] for e in records if e["type"] == "read"}
    assert all(
        event["id"] in recorded_reads
        for event in owner.inbox
        if event["read"] and event["type"] == "shell.completed"
    )


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
