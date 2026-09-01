"""Builtin tool registry.

rlm's default tool set is the persistent IPython REPL as the sole tool, with
shell and edits available as pre-imported REPL skills (``await bash(...)``,
``await edit(...)``). The runtime contract's ``builtin_tools`` list overrides
the tool set directly (e.g. ``["bash", "edit", "ipython"]`` for a native tool
agent).
"""

from __future__ import annotations

from collections.abc import Sequence

from rlm.tools.base import BuiltinTool
from rlm.tools.bash import BashTool, EditTool
from rlm.tools.fetch import FetchTool
from rlm.tools.ipython import IpythonTool

# All registered tools by name. Tests (and extensions) may add entries; names
# listed in the contract's `builtin_tools` or the default set become active.
_TOOLS_BY_NAME: dict[str, BuiltinTool] = {
    "bash": BashTool(),
    "edit": EditTool(),
    "fetch": FetchTool(),
    "ipython": IpythonTool(),
}
# NOT an activation list — activation comes from the contract's `builtin_tools`
# (None = DEFAULT_TOOLS). _STOCK only marks the shipped tools so they are excluded
# from the extras rule below, which auto-activates entries registered at runtime
# (test fixtures/extensions). A stock tool outside the default set (`fetch` —
# network-capable) therefore runs only when `builtin_tools` names it.
_STOCK = ("bash", "edit", "fetch", "ipython")
DEFAULT_TOOLS = ("ipython",)


def _selected(names: Sequence[str] | None) -> tuple[BuiltinTool, ...]:
    if names is not None:
        unknown = [n for n in names if n not in _TOOLS_BY_NAME]
        if unknown:
            raise ValueError(
                f"builtin_tools: unknown tool(s) {unknown}; "
                f"available: {sorted(_TOOLS_BY_NAME)}"
            )
        return tuple(_TOOLS_BY_NAME[n] for n in names)
    # default: ipython plus any extra registered tools (fixtures/extensions).
    extras = [n for n in _TOOLS_BY_NAME if n not in _STOCK]
    return tuple(_TOOLS_BY_NAME[n] for n in [*DEFAULT_TOOLS, *extras])


def get_active_builtin_tools(
    exec_timeout: int = 300, names: Sequence[str] | None = None
) -> list[BuiltinTool]:
    """Return the active tools (the contract's `builtin_tools`, else the
    default set), with an engine-specific IPython schema."""
    return [
        IpythonTool(exec_timeout) if tool.name == "ipython" else tool
        for tool in _selected(names)
    ]


def get_active_tools(names: Sequence[str] | None = None) -> list[dict]:
    """Return OpenAI tool schemas for the active builtins."""
    return [tool.schema() for tool in _selected(names)]


def get_builtin_tool(
    name: str, names: Sequence[str] | None = None
) -> BuiltinTool | None:
    """Look up an active builtin tool handler by name (None if unknown or inactive)."""
    for tool in _selected(names):
        if tool.name == name:
            return tool
    return None
