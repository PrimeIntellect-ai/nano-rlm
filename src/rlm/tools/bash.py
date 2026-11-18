"""Native ``bash`` builtin tool.

Runs one shell command per call in a fresh subshell, like a plain bash agent.
Selected via the runtime contract's ``builtin_tools`` (the default tool set is
ipython alone).
"""

from __future__ import annotations

import io
import os
import selectors
import signal
import subprocess
import sys
import threading
import time
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_command, refusal

# Once bash exits, output still in flight is collected for this long; anything
# holding the pipes after that is a background job and is not waited for.
_EXIT_DRAIN_GRACE = 0.5
_POLL_INTERVAL = 0.05
_OWN_PROCESS_GROUP: dict[str, Any] = (
    {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
)

BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output (stdout and stderr).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
        },
    },
}

EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a string in a file. old_str must occur exactly once in the file; "
            "the file is rewritten with old_str replaced by new_str."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to edit."},
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (must be unique).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


def _discard_until_eof(fd: int) -> None:
    # Keeps a pipe readable for a background job that still writes to it, so
    # the job never gets EPIPE/SIGPIPE after the call has returned.
    try:
        while os.read(fd, 65536):
            pass
    finally:
        os.close(fd)


def _decode(data: bytes) -> str:
    """Decode like ``subprocess`` text mode: locale encoding, universal newlines."""
    return io.TextIOWrapper(io.BytesIO(data), errors="replace").read()


def run_bash(
    command: str,
    timeout: int,
    cwd: str | None = None,
    allow_git: bool | None = None,
) -> str:
    """Guarded ``bash -c`` execution shared by the bash tool and the bash skill.

    Returns when bash itself exits, not when every process that inherited its
    stdout/stderr has closed them, so background jobs (``cmd &``, servers,
    ``nohup``) do not hold the call open until the timeout. On timeout the
    command's whole process group is killed.
    """
    if not isinstance(command, str) or not command.strip():
        return "Error: empty command"
    blocked = find_blocked_command(command, allow_git=allow_git)
    if blocked:
        return refusal(blocked)
    deadline = time.monotonic() + timeout
    proc = subprocess.Popen(
        ["bash", "-c", command],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        cwd=cwd or None,
        **_OWN_PROCESS_GROUP,
    )
    stdout_fd, stderr_fd = proc.stdout.fileno(), proc.stderr.fileno()
    chunks: dict[int, list[bytes]] = {stdout_fd: [], stderr_fd: []}
    exited_at = None
    with selectors.DefaultSelector() as sel:
        for fd in chunks:
            sel.register(fd, selectors.EVENT_READ)
        while sel.get_map():
            now = time.monotonic()
            if exited_at is None and proc.poll() is not None:
                exited_at = now
            if exited_at is not None:
                wait = exited_at + _EXIT_DRAIN_GRACE - now
            else:
                wait = deadline - now
            if wait <= 0:
                break
            for key, _ in sel.select(min(_POLL_INTERVAL, wait)):
                data = os.read(key.fd, 65536)
                if data:
                    chunks[key.fd].append(data)
                else:
                    sel.unregister(key.fd)
        for fd in list(sel.get_map()):
            threading.Thread(
                target=_discard_until_eof, args=(os.dup(fd),), daemon=True
            ).start()
    try:
        proc.wait(timeout=max(deadline - time.monotonic(), 0))
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait()
        return f"Error: command timed out after {timeout}s"
    finally:
        proc.stdout.close()
        proc.stderr.close()
    stdout = _decode(b"".join(chunks[stdout_fd]))
    stderr = _decode(b"".join(chunks[stderr_fd]))
    out = stdout + (("\n" + stderr) if stderr else "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out.strip() or "(no output)"


class BashTool:
    """One shell command per call, fresh subshell, cwd-anchored."""

    name = "bash"

    def schema(self) -> dict[str, Any]:
        return BASH_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return ToolOutcome(
            content=run_bash(
                args.get("command", ""),
                context.exec_timeout,
                cwd=context.cwd,
                allow_git=context.allow_git,
            )
        )


class EditTool:
    """Single-occurrence string replacement, mirroring the edit skill semantics."""

    name = "edit"

    def schema(self) -> dict[str, Any]:
        return EDIT_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        path, old, new = args.get("path"), args.get("old_str"), args.get("new_str")
        if not path or old is None or new is None:
            return ToolOutcome(content="Error: path, old_str and new_str are required")
        if context.cwd and not os.path.isabs(path):
            path = os.path.join(context.cwd, path)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            return ToolOutcome(content=f"Error: cannot read {path}: {e}")
        count = text.count(old)
        if count == 0:
            return ToolOutcome(content=f"Error: old_str not found in {path}")
        if count > 1:
            return ToolOutcome(
                content=f"Error: old_str occurs {count} times in {path}; must be unique"
            )
        open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
        return ToolOutcome(content=f"Edited {path} (1 replacement).")
