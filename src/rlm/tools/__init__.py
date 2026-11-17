"""Builtin tools, uploaded-skill discovery, and REPL helpers."""

from rlm.tools.base import BuiltinTool, ToolContext, ToolOutcome
from rlm.tools.discovery import get, search, servers
from rlm.tools.ipython import IPythonREPL
from rlm.tools.registry import (
    get_active_builtin_tools,
    get_active_tools,
    get_builtin_tool,
)
from rlm.tools.skills import (
    SKILLS_DIR,
    SKILLS_ROOT,
    SkillDoc,
    discover_skill_docs,
    discover_skills,
    get_installed_skills,
)

__all__ = [
    "BuiltinTool",
    "IPythonREPL",
    "SKILLS_DIR",
    "SkillDoc",
    "SKILLS_ROOT",
    "ToolContext",
    "ToolOutcome",
    "discover_skill_docs",
    "discover_skills",
    "get_active_builtin_tools",
    "get_active_tools",
    "get_builtin_tool",
    "get",
    "search",
    "servers",
    "get_installed_skills",
]
