"""Recursive agent concurrency limits."""

import asyncio
import fcntl
import multiprocessing
import os

import pytest

import rlm.api
from rlm.concurrency import (
    SUBAGENT_CONCURRENCY_ENV,
    _slot_indices,
    max_concurrent_subagents,
    subagent_slot,
)
from rlm.types import RLMResult


def _probe_lock_from_another_process(lock_path):
    fd = os.open(lock_path, os.O_RDWR)
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise SystemExit(1) from None
    finally:
        os.close(fd)


def test_limit_reserves_capacity_for_each_depth(monkeypatch):
    monkeypatch.setenv(SUBAGENT_CONCURRENCY_ENV, "5")

    assert max_concurrent_subagents(max_depth=2) == 5
    assert list(_slot_indices(depth=1, max_depth=2, limit=5)) == [0, 1, 2]
    assert list(_slot_indices(depth=2, max_depth=2, limit=5)) == [3, 4]


def test_limit_cannot_be_smaller_than_recursion_depth(monkeypatch):
    monkeypatch.setenv(SUBAGENT_CONCURRENCY_ENV, "1")

    with pytest.raises(ValueError, match="must be at least RLM_MAX_DEPTH"):
        max_concurrent_subagents(max_depth=2)


async def test_same_depth_calls_wait_for_a_shared_slot(tmp_path):
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    second_entered = asyncio.Event()

    async def first():
        async with subagent_slot(tmp_path, depth=1, max_depth=1, limit=1):
            first_entered.set()
            await release_first.wait()

    async def second():
        async with subagent_slot(tmp_path, depth=1, max_depth=1, limit=1):
            second_entered.set()

    first_task = asyncio.create_task(first())
    await first_entered.wait()
    second_task = asyncio.create_task(second())
    await asyncio.sleep(0.1)
    assert not second_entered.is_set()

    release_first.set()
    await asyncio.gather(first_task, second_task)
    assert second_entered.is_set()


async def test_nested_depth_has_reserved_slot(tmp_path):
    child_dir = tmp_path / "sub-child"
    child_dir.mkdir()
    async with subagent_slot(tmp_path, depth=1, max_depth=2, limit=2):
        await asyncio.wait_for(
            _enter_slot_once(child_dir, depth=2, max_depth=2, limit=2),
            timeout=1,
        )

    assert (tmp_path / ".subagent-slots").is_dir()
    assert not (child_dir / ".subagent-slots").exists()


async def test_slot_is_shared_across_processes(tmp_path):
    async with subagent_slot(tmp_path, depth=1, max_depth=1, limit=1):
        lock_path = tmp_path / ".subagent-slots" / "slot-0.lock"
        process = multiprocessing.get_context("spawn").Process(
            target=_probe_lock_from_another_process, args=(lock_path,)
        )
        process.start()
        await asyncio.to_thread(process.join, 5)

    assert not process.is_alive()
    assert process.exitcode == 1


async def test_api_bounds_live_subagents(monkeypatch, tmp_path):
    parent_dir = tmp_path / "session"
    parent_dir.mkdir()
    monkeypatch.setenv("RLM_SESSION_DIR", str(parent_dir))
    monkeypatch.setenv("RLM_DEPTH", "1")
    monkeypatch.setenv("RLM_MAX_DEPTH", "1")
    monkeypatch.setenv(SUBAGENT_CONCURRENCY_ENV, "1")

    first_entered = asyncio.Event()
    release = asyncio.Event()
    active = 0
    max_active = 0

    class BlockingEngine:
        def __init__(self, *, session):
            self.session = session

        async def run(self, prompt):
            nonlocal active, max_active
            active += 1
            max_active = max(max_active, active)
            first_entered.set()
            await release.wait()
            active -= 1
            self.session.close()
            return RLMResult(answer=prompt)

    monkeypatch.setattr(rlm.api, "RLMEngine", BlockingEngine)
    first_task = asyncio.create_task(rlm.api.run("first"))
    await first_entered.wait()
    second_task = asyncio.create_task(rlm.api.run("second"))
    await asyncio.sleep(0.1)

    assert max_active == 1
    assert len(list(parent_dir.glob("sub-*"))) == 1

    release.set()
    await asyncio.gather(first_task, second_task)
    assert len(list(parent_dir.glob("sub-*"))) == 2


async def _enter_slot_once(tmp_path, *, depth, max_depth, limit):
    async with subagent_slot(tmp_path, depth=depth, max_depth=max_depth, limit=limit):
        return
