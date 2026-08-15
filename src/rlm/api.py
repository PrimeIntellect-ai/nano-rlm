"""Public Python API for running rlm agents."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from rlm.concurrency import max_concurrent_subagents, subagent_slot
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult


def _parent_session_dir() -> Path | None:
    """Return the parent session when called from a recursive agent kernel."""
    parent_dir = os.environ.get("RLM_SESSION_DIR")
    depth = int(os.environ.get("RLM_DEPTH", "0"))
    if parent_dir and depth > 0:
        return Path(parent_dir)
    return None


@asynccontextmanager
async def _admit_subagent(parent_dir: Path | None) -> AsyncIterator[None]:
    if parent_dir is None:
        yield
        return

    depth = int(os.environ.get("RLM_DEPTH", "0"))
    max_depth = int(os.environ.get("RLM_MAX_DEPTH", "0"))
    limit = max_concurrent_subagents(max_depth)
    async with subagent_slot(parent_dir, depth=depth, max_depth=max_depth, limit=limit):
        yield


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent."""
    parent_dir = _parent_session_dir()
    async with _admit_subagent(parent_dir):
        if "session" not in kwargs and parent_dir is not None:
            kwargs["session"] = Session(Session.child_dir(parent_dir))
        engine = RLMEngine(**kwargs)
        return await engine.run(prompt)
