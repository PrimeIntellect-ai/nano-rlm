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
