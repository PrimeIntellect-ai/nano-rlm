"""Native ``bash`` builtin tool.

Runs one shell command per call in a fresh subshell, like a plain bash agent.
Enabled by default alongside ipython; override the tool set with ``RLM_BUILTIN_TOOLS``.
"""

from __future__ import annotations

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

_OUTPUT_LIMIT = 30_000


def _clip(text: str) -> str:
    if len(text) <= _OUTPUT_LIMIT:
        return text
    return text[:_OUTPUT_LIMIT] + f"\n... [truncated {len(text) - _OUTPUT_LIMIT} chars]"


class BashTool:
    """One shell command per call, fresh subshell, cwd-anchored."""

    name = "bash"

    def schema(self) -> dict[str, Any]:
        return BASH_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        command = args.get("command", "")
        if not isinstance(command, str) or not command.strip():
            return ToolOutcome(content="Error: empty command")
        blocked = find_blocked_command(command)
        if blocked:
            return ToolOutcome(content=refusal(blocked))
        try:
            proc = subprocess.run(
                ["bash", "-c", command],
                capture_output=True,
                text=True,
                timeout=context.exec_timeout,
                cwd=context.cwd or None,
            )
            out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
            if proc.returncode != 0:
                out += f"\n[exit code {proc.returncode}]"
            return ToolOutcome(content=_clip(out.strip() or "(no output)"))
        except subprocess.TimeoutExpired:
            return ToolOutcome(
                content=f"Error: command timed out after {context.exec_timeout}s"
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
