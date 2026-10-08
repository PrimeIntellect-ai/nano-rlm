"""Native ``bash`` builtin tool.

Runs one shell command per call in a fresh subshell, like a plain bash agent.
Selected via the runtime contract's ``builtin_tools`` (the default tool set is
ipython alone).
"""

from __future__ import annotations

import os
import selectors
import signal
import time
import contextlib
import subprocess
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_command, refusal

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


def run_bash(
    command: str,
    timeout: int,
    cwd: str | None = None,
    allow_git: bool | None = None,
) -> str:
    """Guarded ``bash -c`` execution shared by the bash tool and the bash skill."""
    if not isinstance(command, str) or not command.strip():
        return "Error: empty command"
    blocked = find_blocked_command(command, allow_git=allow_git)
    if blocked:
        return refusal(blocked)
    limit = 1024 * 1024
    deadline = time.monotonic() + timeout
    parts = {"stdout": bytearray(), "stderr": bytearray()}
    reason = ""
    with (
        subprocess.Popen(
            ["bash", "-c", command],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or None,
            start_new_session=True,
        ) as proc,
        selectors.DefaultSelector() as selector,
    ):
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        try:
            while selector.get_map():
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    reason = f"Error: command timed out after {timeout}s"
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    available = limit - sum(map(len, parts.values()))
                    parts[key.data].extend(data[:available])
                    if len(data) >= available:
                        reason = (
                            "[command output exceeded 1 MiB; process group terminated]"
                        )
                        break
                if reason:
                    break
            if not reason:
                try:
                    proc.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = f"Error: command timed out after {timeout}s"
        finally:
            if reason or proc.poll() is None:
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
    stdout, stderr = (
        parts[key].decode(errors="replace") for key in ("stdout", "stderr")
    )
    out = stdout + (("\n" + stderr) if stderr else "")
    if reason:
        out += "\n" + reason
    elif proc.returncode != 0:
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
