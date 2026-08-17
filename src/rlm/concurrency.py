"""Cross-process admission control for recursive agents."""

from __future__ import annotations

import asyncio
import fcntl
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator


DEFAULT_MAX_CONCURRENT_SUBAGENTS = 4
SUBAGENT_CONCURRENCY_ENV = "RLM_MAX_CONCURRENT_SUBAGENTS"
_POLL_INTERVAL_SECONDS = 0.05


def max_concurrent_subagents(max_depth: int) -> int:
    """Read and validate the per-session-tree sub-agent concurrency bound."""
    raw = os.environ.get(
        SUBAGENT_CONCURRENCY_ENV, str(DEFAULT_MAX_CONCURRENT_SUBAGENTS)
    )
    try:
        limit = int(raw)
    except ValueError as exc:
        raise ValueError(f"{SUBAGENT_CONCURRENCY_ENV} must be an int") from exc
    if limit <= 0:
        raise ValueError(f"{SUBAGENT_CONCURRENCY_ENV} must be positive")
    if max_depth > limit:
        raise ValueError(
            f"{SUBAGENT_CONCURRENCY_ENV} ({limit}) must be at least "
            f"RLM_MAX_DEPTH ({max_depth})"
        )
    return limit


def _slot_indices(depth: int, max_depth: int, limit: int) -> range:
    """Assign disjoint capacity to each depth so nested waits cannot deadlock."""
    slots_per_depth, extra = divmod(limit, max_depth)
    start = (depth - 1) * slots_per_depth + min(depth - 1, extra)
    width = slots_per_depth + (1 if depth <= extra else 0)
    return range(start, start + width)


def _root_session_dir(session_dir: Path | str, depth: int) -> Path:
    root = Path(session_dir).resolve()
    for _ in range(depth - 1):
        root = root.parent
    return root


@asynccontextmanager
async def subagent_slot(
    session_dir: Path | str,
    *,
    depth: int,
    max_depth: int,
    limit: int,
) -> AsyncIterator[None]:
    """Hold one crash-safe slot for a live recursive agent."""
    if depth <= 0 or depth > max_depth:
        yield
        return

    lock_dir = _root_session_dir(session_dir, depth) / ".subagent-slots"
    lock_dir.mkdir(parents=True, exist_ok=True)
    slot_paths = [
        lock_dir / f"slot-{index}.lock"
        for index in _slot_indices(depth, max_depth, limit)
    ]

    acquired_fd: int | None = None
    while acquired_fd is None:
        for slot_path in slot_paths:
            fd = os.open(slot_path, os.O_CREAT | os.O_RDWR, 0o600)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                os.close(fd)
            else:
                acquired_fd = fd
                break
        if acquired_fd is None:
            await asyncio.sleep(_POLL_INTERVAL_SECONDS)

    try:
        yield
    finally:
        fcntl.flock(acquired_fd, fcntl.LOCK_UN)
        os.close(acquired_fd)
