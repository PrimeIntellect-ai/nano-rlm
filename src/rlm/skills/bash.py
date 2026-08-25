"""Built-in ``bash`` skill — run a shell command from the REPL.

Enabled via ``RLM_SKILLS``; pre-imported into the IPython kernel so the agent calls
``await bash(command="...")`` (or ``await bash("...")``). One command per call, fresh
subshell, like a plain bash-agent tool but surfaced as a skill.
"""

from __future__ import annotations

import asyncio


async def run(command: str, timeout: int = 120) -> str:
    """Run a shell command and return its combined output.

    Args:
        command: The shell command to run.
        timeout: Seconds before the command is killed.

    Returns:
        stdout and stderr of the command (with the exit code when nonzero).
    """
    proc = await asyncio.create_subprocess_exec(
        "bash",
        "-c",
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        out_b, err_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        return f"Error: command timed out after {timeout}s"
    out = out_b.decode(errors="replace")
    err = err_b.decode(errors="replace")
    result = out + (("\n" + err) if err else "")
    if proc.returncode != 0:
        result += f"\n[exit code {proc.returncode}]"
    result = result.strip() or "(no output)"
    if len(result) > 30_000:
        result = result[:30_000] + f"\n... [truncated {len(result)-30_000} chars]"
    return result
