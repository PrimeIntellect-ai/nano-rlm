"""Builtin tool registry.

rlm exposes a single builtin tool: the persistent IPython REPL. Shell work and file edits
go through it (``!cmd`` / ``%%bash``, Python, or the built-in ``edit`` skill). The tool set
is not configurable.
"""

from __future__ import annotations

from rlm.tools.base import BuiltinTool
from rlm.tools.ipython import IpythonTool

_BUILTIN_TOOLS: tuple[BuiltinTool, ...] = (IpythonTool(),)
_TOOLS_BY_NAME: dict[str, BuiltinTool] = {tool.name: tool for tool in _BUILTIN_TOOLS}


def get_active_builtin_tools(exec_timeout: int = 300) -> list[BuiltinTool]:
    """Return registered tools with an engine-specific IPython schema."""
    return [
        IpythonTool(exec_timeout) if name == "ipython" else tool
        for name, tool in _TOOLS_BY_NAME.items()
    ]


def get_active_tools() -> list[dict]:
    """Return OpenAI tool schemas for the active builtins."""
    return [tool.schema() for tool in _BUILTIN_TOOLS]


def get_builtin_tool(name: str) -> BuiltinTool | None:
    """Look up a builtin tool handler by name (None if unknown)."""
    return _TOOLS_BY_NAME.get(name)
