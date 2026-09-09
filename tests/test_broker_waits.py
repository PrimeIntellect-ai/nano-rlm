from __future__ import annotations

import asyncio

import pytest

from rlm import broker


@pytest.mark.parametrize("return_exceptions", [False, True])
async def test_gather_cancellation_releases_cell_lease(monkeypatch, return_exceptions):
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
    observed = []
    with broker.cell_execution():
        group = broker._subagent_aware_gather(
            broker.run("one"), broker.run("two"), return_exceptions=return_exceptions
        )
        assert asyncio.isfuture(group)
        await started.wait()
        assert not any(wait.exclusive for wait in running)

        def cancel():
            observed.extend(wait.exclusive for wait in running)
            group.cancel("stop")

        asyncio.get_running_loop().call_soon(cancel)
        with pytest.raises(asyncio.CancelledError):
            await group

        assert observed == [True, True]
        assert len(cancelled) == 2
        assert not any(wait.exclusive for wait in running)
        assert group.done()
        assert not group.cancelled()
        assert not group.cancel()
    assert broker._cell_task is None
