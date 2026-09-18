"""Skill binding inside the IPython kernel.

The kernel bootstrap imports this module to expose skills as callables in the user
namespace; ``rlm.harness.load_skills`` reuses it to bring authored packages into a
running session.
"""

from __future__ import annotations

import functools
import importlib
import inspect
import json
import os
import sys
import time
import types
from pathlib import Path
from typing import Any

from rlm.tools.skills import AuthoredSkill, list_authored_skills


def log_programmatic_call(tool_name: str, source: str) -> None:
    # Matches the line format written by install.sh's bash wrapper so
    # ProgrammaticToolCallStats.from_log parses both sources identically.
    session_dir = os.environ.get("RLM_SESSION_DIR", "")
    if not session_dir:
        return
    try:
        with open(os.path.join(session_dir, "programmatic_tool_calls.jsonl"), "a") as f:
            f.write(
                json.dumps(
                    {"tool": tool_name, "source": source, "timestamp": time.time()}
                )
                + "\n"
            )
    except OSError:
        pass


class CallableModule(types.ModuleType):
    # Make `await <skill>(...)` shorthand for `await <skill>.run(...)`.
    # __call__ is looked up on the type, not the instance, so the
    # override has to live on the class.
    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return await self.run(*args, **kwargs)


def wrap_callable(mod: types.ModuleType, log_source: str | None, register: bool = True):
    """A module clone whose ``run`` is logged as a programmatic call and whose
    signature/docstring mirror ``run`` so ``help(<skill>)`` shows the real API.
    ``log_source`` is ``'python'`` for skills; brokered skills (None) are counted by
    the supervisor."""
    wrapped = CallableModule(mod.__name__)
    wrapped.__dict__.update(mod.__dict__)
    if log_source is not None:
        original_run = wrapped.run

        @functools.wraps(original_run)
        async def logged_run(*args: Any, **kwargs: Any) -> Any:
            log_programmatic_call(mod.__name__, log_source)
            return await original_run(*args, **kwargs)

        wrapped.run = logged_run
    wrapped.__signature__ = inspect.signature(wrapped.run)
    wrapped.__doc__ = wrapped.run.__doc__
    if register:
        sys.modules[mod.__name__] = wrapped
    return wrapped


class BrokenSkill:
    """Stands in for an authored package that breaks the skill contract or fails to
    import: calling it explains why, and the kernel keeps working so the agent can
    fix the package."""

    def __init__(self, name: str, reason: str, cause: BaseException | None = None):
        self.__name__ = name
        self.reason = reason
        self.__doc__ = f"authored skill {name!r} is unusable: {reason}"
        self._cause = cause

    async def __call__(self, *args: Any, **kwargs: Any) -> Any:
        raise RuntimeError(self.__doc__) from self._cause


def bind_authored(skill: AuthoredSkill, namespace: dict[str, Any]) -> str | None:
    """(Re)import one authored package and bind it by name into ``namespace``.
    Returns the reason it is unusable, or None."""
    if skill.error is not None:
        namespace[skill.name] = BrokenSkill(skill.name, skill.error)
        return skill.error
    if skill.path not in sys.path:
        sys.path.append(skill.path)
    try:
        existing = sys.modules.get(skill.name)
        if isinstance(existing, types.ModuleType) and not isinstance(
            existing, CallableModule
        ):
            module = importlib.reload(existing)
        else:
            sys.modules.pop(skill.name, None)
            module = importlib.import_module(skill.name)
    except Exception as error:
        reason = f"import failed: {error}"
        namespace[skill.name] = BrokenSkill(skill.name, reason, error)
        return reason
    if not inspect.iscoroutinefunction(getattr(module, "run", None)):
        reason = "it must define `async def run(...)`"
        namespace[skill.name] = BrokenSkill(skill.name, reason)
        return reason
    if not module.run.__doc__ and skill.skill_md:
        # SKILL.md stands in for a missing docstring so help(<name>) stays useful.
        module.run.__doc__ = Path(skill.skill_md).read_text()
    namespace[skill.name] = wrap_callable(module, "python")
    return None


def load_authored(
    skills_dir: str | None,
    namespace: dict[str, Any],
    names: tuple[str, ...] = (),
    *,
    reserved: set[str] | None = None,
) -> dict[str, str | None]:
    """Bind every authored package (or just ``names``) into ``namespace``.

    Returns ``{name: reason}`` with None for a usable skill. Names in ``reserved``
    (installed or MCP-generated skills) are never rebound.
    """
    outcome: dict[str, str | None] = {}
    found = {skill.name: skill for skill in list_authored_skills(skills_dir)}
    wanted = names or tuple(found)
    for name in wanted:
        if reserved and name in reserved:
            outcome[name] = f"{name!r} is an installed or generated skill"
        elif name not in found:
            outcome[name] = f"no package {name!r} under {skills_dir or '(unset)'}"
        else:
            outcome[name] = bind_authored(found[name], namespace)
    return outcome
