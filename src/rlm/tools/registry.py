"""Builtin tool registry.

rlm's default tool set is native ``bash`` and ``edit`` tools plus the persistent
IPython REPL. Shell and edits are ALSO available as REPL skills (``await bash(...)``,
``await edit(...)``) for mixing with Python in one cell. ``RLM_BUILTIN_TOOLS``
(comma-separated) overrides the set, e.g. ``ipython`` for a REPL-only agent.
"""

from __future__ import annotations

import os

from rlm.tools.base import BuiltinTool
from rlm.tools.bash import BashTool, EditTool
from rlm.tools.ipython import IpythonTool

# All registered tools by name. Tests (and extensions) may add entries; names
# listed in RLM_BUILTIN_TOOLS or the default set become active.
_TOOLS_BY_NAME: dict[str, BuiltinTool] = {
    "bash": BashTool(),
    "edit": EditTool(),
    "ipython": IpythonTool(),
}
_DEFAULT = ("bash", "edit", "ipython")


def _selected() -> tuple[BuiltinTool, ...]:
    spec = os.environ.get("RLM_BUILTIN_TOOLS", "").strip()
    if spec:
        names = [n.strip() for n in spec.split(",") if n.strip()]
        unknown = [n for n in names if n not in _TOOLS_BY_NAME]
        if unknown:
            raise ValueError(
                f"RLM_BUILTIN_TOOLS: unknown tool(s) {unknown}; "
                f"available: {sorted(_TOOLS_BY_NAME)}"
            )
        return tuple(_TOOLS_BY_NAME[n] for n in names)
    # default: the standard set plus any extra registered tools (fixtures/extensions)
    extras = [n for n in _TOOLS_BY_NAME if n not in _DEFAULT]
    return tuple(_TOOLS_BY_NAME[n] for n in [*_DEFAULT, *extras])


def get_active_builtin_tools() -> list[BuiltinTool]:
    """Return the active builtin-tool instances (default: bash + ipython)."""
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
