from __future__ import annotations

import asyncio

import pytest

from rlm import broker, gather, run


async def test_gather_cancellation_releases_cell_lease(monkeypatch):
    monkeypatch.setattr(broker, "_endpoint", broker.BrokerEndpoint("unused", "test"))
    monkeypatch.setattr(broker, "_scope_id", "cell")
    started = asyncio.Event()
    running = []
    cancelled = []

    async def request(payload, wait):
        running.append(wait)
        if len(running) == 2:
            started.set()
        try:
            await asyncio.Future()
        finally:
            cancelled.append(wait)

    monkeypatch.setattr(broker, "_request", request)

    async def cell():
        with broker.cell_execution():
            group = gather(run("one"), run("two"))
            await asyncio.sleep(0)
            assert not running
            await group

    task = asyncio.create_task(cell())
    await asyncio.wait_for(started.wait(), timeout=1)
    assert all(wait.exclusive for wait in running)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task

    assert len(cancelled) == 2
    assert not any(wait.exclusive for wait in running)
    assert broker._cell_task is None


async def test_gather_rejects_non_subagent_calls(monkeypatch):
    monkeypatch.setattr(broker, "_endpoint", broker.BrokerEndpoint("unused", "test"))
    monkeypatch.setattr(broker, "_scope_id", "cell")
    child = run("one")
    other = asyncio.sleep(0)
    try:
        with pytest.raises(TypeError, match="only calls returned by rlm"):
            await gather(child, other)
    finally:
        child.close()
        other.close()
