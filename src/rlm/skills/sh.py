"""Built-in ``sh`` skill — raw-shell cell magic with captured output.

Enabled via ``RLM_SKILLS``. Registers a ``%%sh`` cell magic in the kernel: the cell
body is passed to bash VERBATIM (no Python string quoting/escaping), and the combined
output is returned as the cell value — so ``out = _`` (or ``%%sh out``) captures it
as a normal Python string. Also usable programmatically: ``await sh("...")``.
"""

from __future__ import annotations

import asyncio
import subprocess


def _run_shell(command: str, timeout: int = 120) -> str:
    proc = subprocess.run(
        ["bash", "-c", command],
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    out = out.strip() or "(no output)"
    if len(out) > 30_000:
        out = out[:30_000] + f"\n... [truncated {len(out) - 30_000} chars]"
    return out


async def run(command: str, timeout: int = 120) -> str:
    """Run a shell command and return its combined output as a string."""
    return await asyncio.to_thread(_run_shell, command, timeout)




def _log_ptc() -> None:
    """Log a %%sh magic invocation to programmatic_tool_calls.jsonl (PTC accounting)."""
    import json as _json
    import os as _os
    import time as _time

    session_dir = _os.environ.get("RLM_SESSION_DIR", "")
    if not session_dir:
        return
    try:
        with open(_os.path.join(session_dir, "programmatic_tool_calls.jsonl"), "a") as f:
            f.write(_json.dumps({"tool": "sh", "source": "magic", "timestamp": _time.time()}) + "\n")
    except OSError:
        pass

def _register_magic() -> None:
    try:
        from IPython import get_ipython
    except ImportError:
        return
    ip = get_ipython()
    if ip is None:
        return

    def sh_cell(line: str, cell: str):
        """%%sh [varname] — run the raw cell body in bash; output returned (and
        assigned to varname when given)."""
        _log_ptc()
        output = _run_shell(cell)
        name = line.strip()
        if name:
            ip.user_ns[name] = output
            # keep context lean when captured: show a short preview, not the whole blob
            preview = output if len(output) <= 2000 else output[:2000] + "\n... [captured in full]"
            print(f"[captured to {name}, {len(output)} chars]\n{preview}")
            return None
        return output

    ip.register_magic_function(sh_cell, magic_kind="cell", magic_name="sh")


_register_magic()
