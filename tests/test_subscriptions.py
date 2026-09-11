from __future__ import annotations

import asyncio
import json

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall
from rlm.engine import RLMEngine
from rlm.subscriptions import Subscriptions
from rlm.supervisor import SessionTreeSupervisor
from test_supervisor import _config


async def _until(predicate):
    async def wait():
        while not predicate():
            await asyncio.sleep(0.02)

    await asyncio.wait_for(wait(), 5)


async def test_subscription_batching_registration_and_cancellation():
    events = []
    registry = Subscriptions(
        lambda sub, kind, payload: events.append((kind, payload)), lambda sub: None
    )
    sub = registry.register("owner", "agent", "child", cursor=5)
    registry.activity("agent", "child", 0, 5)
    assert not sub.pending
    registry.activity("agent", "child", 3, 8)
    registry.activity("agent", "child", 8, 12)
    await _until(lambda: events)
    assert events == [("watch.agent", {"start": 5, "end": 12})]
    with pytest.raises(PermissionError):
        registry.get("other", sub.info.id)
    registry.activity("agent", "child", 12, 15)
    await registry.cancel(sub)
    await asyncio.sleep(0.25)
    assert len(events) == 1
    assert sub.info.status == "cancelled"


async def test_path_watcher_batches_changes_and_reports_removal(tmp_path):
    events = []
    registry = Subscriptions(
        lambda sub, kind, payload: events.append((kind, payload)), lambda sub: None
    )
    target = tmp_path / "watched"
    target.mkdir()
    sub = await registry.path("owner", target, False)
    try:
        for name in ["first", ".hidden"]:
            (target / name).write_text("a")
            (target / name).write_text("b")
        await _until(lambda: events)
        changed = {
            path
            for kind, payload in events
            if kind == "watch.path"
            for path in payload["paths"]
        }
        assert {str(target / "first"), str(target / ".hidden")} <= changed
        assert len(events) == 1
        for child in target.iterdir():
            child.unlink()
        target.rmdir()
        await _until(lambda: sub.info.status == "failed")
        assert events[-1][0] == "watch.failed"
        assert sub.info.error
        target.mkdir()
        (target / "later").write_text("no replay")
        count = len(events)
        await asyncio.sleep(0.25)
        assert len(events) == count
    finally:
        await registry.close()


def _tool(code):
    return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])


async def test_real_kernel_subscriptions_survive_restart(session):
    def factory(**kwargs):
        return RLMEngine(
            client=DummyClient(
                [
                    DummyMessage(content="ready"),
                    _tool("print('child progress')"),
                    DummyMessage(content="finished"),
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
Path('watched').mkdir()
child = await rlm.agent.spawn('initial', name='worker', persistent=True)
await child.wait(timeout=10)
activity = await rlm.watch.agent(child)
files = await rlm.watch.path('watched')
job = await rlm.shell.run('sleep 0.5; printf output; printf changed > watched/result')
output = await rlm.watch.job(job)
await child.send('continue')
import os
os._exit(7)
"""),
            _tool("""
import asyncio
for _ in range(100):
    events = await rlm.inbox.list()
    if {'watch.agent', 'watch.job', 'watch.path'} <= {e['type'] for e in events}:
        break
    await asyncio.sleep(0.05)
else:
    raise AssertionError('subscription events missing')
subscriptions = await rlm.watch.list()
assert len(subscriptions) == 3
assert all(s.status == ('completed' if s.kind == 'job' else 'active') for s in subscriptions)
child = await rlm.agent.get('worker')
for item in events:
    if not item['type'].startswith('watch.'):
        continue
    event = await rlm.inbox.read(item['id'])
    assert event['subscription_id'] in {s.id for s in subscriptions}
    content = event['content']
    if item['type'] == 'watch.agent':
        assert child.history().messages[content['start']:content['end']]
    elif item['type'] == 'watch.job':
        job = await rlm.shell.get(content['target'])
        chunk = await job.read(cursor=content['start'])
        assert 'output' in chunk.text
    elif item['type'] == 'watch.path':
        assert any(path.endswith('/watched/result') for path in content['paths'])
for info in subscriptions:
    watch = await rlm.watch.get(info.id)
    assert (await watch.cancel()).status == ('completed' if info.kind == 'job' else 'cancelled')
print('WATCHES_OK')
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
        await engine.prompt("Exercise subscriptions")
        records = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
        ]
        assert any(
            record.get("content", "").strip() == "WATCHES_OK" for record in records
        )
        ledger = [
            json.loads(line)
            for line in (session.dir / "inbox.jsonl").read_text().splitlines()
        ]
        assert any(record["type"] == "subscription" for record in ledger)
    finally:
        await engine.aclose()
        await supervisor.aclose()
    assert all(
        sub.info.status in {"cancelled", "completed"}
        for sub in supervisor._subscriptions.items.values()
    )


async def test_finished_targets_flush_and_release_capacity(monkeypatch):
    events = []
    registry = Subscriptions(
        lambda sub, kind, payload: events.append(payload), lambda sub: None
    )
    monkeypatch.setattr("rlm.subscriptions.MAX_ACTIVE_SUBSCRIPTIONS", 1)
    sub = registry.register("owner", "job", "first")
    registry.activity("job", "first", 0, 12)
    registry.finish("job", "first")
    assert events == [{"start": 0, "end": 12}]
    assert sub.info.status == "completed" and sub.timer is None
    registry.register("owner", "job", "second")
    already_done = registry.register("owner", "job", "first", completed=True)
    assert already_done.info.status == "completed"
    await registry.close()


async def test_cancel_pending_path_registration(tmp_path, monkeypatch):
    registry = Subscriptions(lambda *args: None, lambda sub: None)

    async def pending(sub, target):
        await asyncio.Future()

    monkeypatch.setattr(registry, "_watch_path", pending)
    task = asyncio.create_task(registry.path("owner", tmp_path, False))
    await asyncio.sleep(0)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    sub = next(iter(registry.items.values()))
    assert sub.info.status == "cancelled" and sub.task.done()
