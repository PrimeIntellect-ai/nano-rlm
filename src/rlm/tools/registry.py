"""Builtin tool registry.

rlm's default builtin tool is the persistent IPython REPL. Shell work and file edits
go through it (``!cmd`` / ``%%bash``, Python, or the built-in ``edit`` skill).

``RLM_BUILTIN_TOOLS`` (comma-separated: ``ipython``, ``bash``, ``edit``) overrides the
tool set for interface ablations, e.g. ``bash,edit`` replicates the plain bash-agent
interface, ``bash,edit,ipython`` adds the REPL alongside.
"""

from __future__ import annotations

import os

from rlm.tools.base import BuiltinTool
from rlm.tools.bash import BashTool, EditTool
from rlm.tools.ipython import IpythonTool

_AVAILABLE: dict[str, BuiltinTool] = {
    "ipython": IpythonTool(),
    "bash": BashTool(),
    "edit": EditTool(),
}


def _selected() -> tuple[BuiltinTool, ...]:
    spec = os.environ.get("RLM_BUILTIN_TOOLS", "").strip()
    if not spec:
        return (_AVAILABLE["ipython"],)
    names = [n.strip() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in _AVAILABLE]
    if unknown:
        raise ValueError(f"RLM_BUILTIN_TOOLS: unknown tool(s) {unknown}; available: {sorted(_AVAILABLE)}")
    return tuple(_AVAILABLE[n] for n in names)


def get_active_builtin_tools() -> list[BuiltinTool]:
    """Return the active builtin-tool instances (default: just ipython)."""
    return list(_selected())


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas for the active builtins."""
    return [tool.schema() for tool in _selected()]


def get_builtin_tool(name: str) -> BuiltinTool | None:
    """Look up a builtin tool handler by name (None if unknown)."""
    for tool in _selected():
        if tool.name == name:
            return tool
    return None
