"""Built-in ``bash`` skill — run a shell command from the REPL.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await bash(command="...")`` (or ``await bash("...")``). One command per call, a fresh
``bash -c`` subshell, with the same git-history guard as shell tool paths.
"""

from __future__ import annotations

import asyncio
import os
import subprocess

from rlm.tools.git_block import find_blocked_command, refusal


def _default_timeout() -> int:
    raw = os.environ.get("RLM_EXEC_TIMEOUT", "")
    return int(raw) if raw.isdigit() and int(raw) > 0 else 300


def _run_bash(command: str, timeout: int) -> str:
    """Execute via ``bash -c`` (never /bin/sh) and return combined output."""
    blocked = find_blocked_command(command)
    if blocked:
        return refusal(blocked)
    try:
        proc = subprocess.run(
            ["bash", "-c", command],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return f"Error: command timed out after {timeout}s"
    out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    out = out.strip() or "(no output)"
    if len(out) > _OUTPUT_LIMIT:
        out = (
            out[:_OUTPUT_LIMIT] + f"\n... [truncated {len(out) - _OUTPUT_LIMIT} chars]"
        )
    return out


async def run(command: str, timeout: int | None = None) -> str:
    """Run a shell command and return its combined output.

    Args:
        command: The shell command to run (executed with ``bash -c``).
        timeout: Seconds before the command is killed (default: the harness
            exec timeout via ``RLM_EXEC_TIMEOUT``, else 300).

    Returns:
        stdout and stderr of the command (with the exit code when nonzero).
    """
    return await asyncio.to_thread(_run_bash, command, timeout or _default_timeout())
