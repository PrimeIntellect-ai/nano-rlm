"""Built-in ``bash`` skill — run a shell command from the REPL.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await bash(command="...")`` (or ``await bash("...")``). Identical semantics to the
``bash`` tool — same ``bash -c`` execution, git-history guard, and output contract —
via the shared runner in ``rlm.tools.bash``.
"""

from __future__ import annotations

import asyncio
import os

from rlm.tools.bash import run_bash


def _default_timeout() -> int:
    raw = os.environ.get("RLM_EXEC_TIMEOUT", "")
    return int(raw) if raw.isdigit() and int(raw) > 0 else 300


async def run(command: str, timeout: int | None = None) -> str:
    """Run a shell command and return its combined output.

    Args:
        command: The shell command to run (executed with ``bash -c``).
        timeout: Seconds before the command is killed (default: the harness
            exec timeout via ``RLM_EXEC_TIMEOUT``, else 300).

    Returns:
        stdout and stderr of the command (with the exit code when nonzero).
    """
    return await asyncio.to_thread(run_bash, command, timeout or _default_timeout())
