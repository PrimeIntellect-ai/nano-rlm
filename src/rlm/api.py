"""Public Python API for running rlm agents."""

import os

from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.types import RLMResult


def _child_session(
    parent_dir: str | os.PathLike | None = None, depth: int | None = None
) -> Session | None:
    """If inside a parent session (depth > 0), create a child under it."""
    if parent_dir is None:
        parent_dir = os.environ.get("RLM_SESSION_DIR")
    if depth is None:
        depth = int(os.environ.get("RLM_DEPTH", "0"))
    if parent_dir and depth > 0:
        return Session(Session.child_dir(parent_dir))
    return None


async def run(prompt: str, **kwargs) -> RLMResult:
    """Run a single rlm agent."""
    depth = kwargs.pop("_depth", None)
    max_depth = kwargs.pop("_max_depth", None)
    parent_session_dir = kwargs.pop("_parent_session_dir", None)
    if parent_session_dir is not None:
        child = _child_session(parent_session_dir, depth)
        if child:
            kwargs["session"] = child
    elif "session" not in kwargs:
        child = _child_session(depth=depth)
        if child:
            kwargs["session"] = child
    if depth is not None:
        kwargs["depth"] = depth
    if max_depth is not None:
        kwargs["max_depth"] = max_depth
    engine = RLMEngine(**kwargs)
    return await engine.run(prompt)
