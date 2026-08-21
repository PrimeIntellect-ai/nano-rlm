"""System prompt construction."""

from __future__ import annotations

import os

from typing import TYPE_CHECKING

from rlm.tools.git_block import allow_git

if TYPE_CHECKING:
    from rlm.tools.base import BuiltinTool


# Importable names of the base toolkit declared in pyproject.toml.
# Surfaced in the system prompt so the agent knows what's available
# without probing — keep in sync with the dependency list.
BASE_TOOLKIT = (
    "requests",
    "httpx",
    "yaml",
    "tomli",
    "dotenv",
    "pandas",
    "numpy",
    "scipy",
    "bs4",
    "lxml",
    "pydantic",
)

SHELL_TOOL_NAMES = frozenset({"ipython"})
GIT_HISTORY_GUARD_PROMPT = (
    "Do not cheat by using online solutions or hints specific to this task, or "
    "by copying or inferring solutions from other branches, tags, remotes, "
    "reflogs, or broad git history in the project. Broad-history `git log` "
    "options such as `--all`, `-all`, `--branches`, `--remotes`, `--tags`, "
    "`--glob`, `--alternate-refs`, `--reflog`, `--walk-reflogs`, or `-g` will "
    "be refused."
)
IPYTHON_CONTROL_PROMPT = (
    "IPython is the agent's long-lived notebook: a persistent control "
    "environment for reasoning, context management, state, tool orchestration, "
    "and recursive subcalls. Use it to keep intermediate variables, inspect "
    "and transform outputs, write small helper functions, and preserve useful "
    "state across turns or compaction.\n\n"
    "Do not assume IPython is the native runtime of the external thing being "
    "investigated. A repository, package, service, dataset, paper, website, "
    "benchmark, or API may have its own environment and normal interface. "
    "Evaluate external systems through their own interface, then use IPython "
    "to coordinate the process and analyze what comes back.\n\n"
    "When running shell commands from IPython, use `%%bash` cells. If you use "
    "`%%bash`, it must be the first line of the code cell: no comments, "
    "spaces, blank lines, imports, or Python statements before it. Avoid "
    "`!cmd` shell escapes for project commands so shell behavior is explicit "
    "and multi-line commands share one shell context.\n\n"
    "Important: do not install dependencies into the IPython kernel just to "
    "make an external project import or run there. If a project import, test, "
    "script, CLI, or dependency check is needed, run it through that project's "
    "own environment and normal command interface. For example, in a Python "
    "repo use its documented commands, `uv run ...`, `.venv/bin/python ...`, "
    "or the active project interpreter from the repo root. Treat failures from "
    "that native environment as the relevant result."
    "\n\n"
    "Use Python for reading, searching, and editing files — it gives you "
    "reusable variables you can slice, filter, and act on without re-reading. "
    "Always assign read/search results to named variables so you can revisit "
    "them later."
)
EDIT_SKILL_PROMPT = (
    "For targeted existing-file edits, prefer the pre-imported async `edit` "
    "skill from IPython: `old = '''...'''; new = '''...'''; await "
    'edit(path="pkg/file.py", old_str=old, new_str=new)`. Use exact '
    "old/new strings; if the text contains triple double quotes, use triple "
    "single-quoted variables or build `old`/`new` from inspected file slices."
)
SEARCH_SKILL_PROMPT = (
    "For web search, use the pre-imported async `search` skill from IPython: "
    '`await search(query="...")`. Results come back as title, URL, and snippet; '
    "assign the result to a variable so you can revisit it. To cover "
    "several angles at once, fan out with `asyncio.gather(search(...), search(...))`."
)


MINIMAL_IPYTHON_PROMPT = (
    "Run shell commands with `%%bash` as the very first line of an IPython cell; "
    "this is a uv-managed venv (use `uv run ...` / `uv pip install ...`, no pip module)."
)


def _minimal_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    messages_path: str,
    *,
    allow_recursion: bool,
    active_tools: list["BuiltinTool"],
    cli_skills: list[str] | None = None,
) -> str:
    """A stripped system prompt: factual affordances only, no behavioral steering.

    Used to measure how much the default prose steers the model. Keeps the role line,
    environment facts, enabled skills/recursion, the git guard, and one terse bash/venv
    line; drops IPYTHON_CONTROL_PROMPT and the "break down / iterate" framing.
    """
    parts: list[str] = [
        "You are a general purpose agent that uses code to solve tasks.",
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {messages_path}",
        f"Pre-installed Python packages: {', '.join(BASE_TOOLKIT)}.",
    ]
    if installed_skills:
        installed = ", ".join(f"`{skill}`" for skill in installed_skills)
        parts.append(f"Installed skills (pre-imported): {installed}.")
    if allow_recursion:
        parts.append(
            "A callable `rlm` is in your namespace: `await rlm('sub-task')` spawns a recursive sub-agent."
        )
    if _has_tool(active_tools, "ipython"):
        parts.extend(["", MINIMAL_IPYTHON_PROMPT])
    if _should_include_git_history_guard(active_tools):
        parts.extend(["", GIT_HISTORY_GUARD_PROMPT])
    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])
    return "\n".join(parts)




BASH_EDIT_REPLICA = (
    "You are a coding agent. You have access to a bash tool for running shell commands."
)
BASH_EDIT_EDIT_LINE = (
    "You also have an edit tool for single-occurrence string replacement in a file."
)
BASH_EDIT_IPYTHON_LINE = (
    "You also have an ipython tool: a persistent Python REPL (variables persist across calls)."
)
BASH_EDIT_EDIT_SKILL_LINE = (
    "In the ipython tool, edit files with `await edit(path=..., old_str=..., new_str=...)` "
    "(single-occurrence string replacement)."
)


BASH_EDIT_BASH_SKILL_LINE = (
    "In the ipython tool, run shell commands with `await bash(command=...)` — do not use "
    "`%%bash` or `!` escapes."
)


def _bashedit_prompt(installed_skills, *, active_tools):
    """Replicate the verifiers bash-harness system prompt, extended only by one line per
    extra capability (ipython tool / edit skill). For interface ablations."""
    names = {tool.name for tool in active_tools}
    if "bash" in names:
        parts = [BASH_EDIT_REPLICA]
    else:
        parts = ["You are a coding agent."]
    if "edit" in names:
        parts.append(BASH_EDIT_EDIT_LINE)
    if "ipython" in names:
        parts.append(BASH_EDIT_IPYTHON_LINE)
        if "bash" not in names and "bash" in (installed_skills or []):
            parts.append(BASH_EDIT_BASH_SKILL_LINE)
        elif "bash" in names and "bash" in (installed_skills or []):
            parts.append(
                "Inside the ipython tool you can also run shell with `await bash(command=...)` — "
                "it returns the output as a string, useful when mixing shell and Python in one cell."
            )
        elif "bash" not in names and os.environ.get("RLM_BASHEDIT_BASH_MAGIC"):
            parts.append(
                "In the ipython tool, run shell commands with `%%bash` as the very first line of a cell."
            )
        if "edit" in (installed_skills or []):
            parts.append(BASH_EDIT_EDIT_SKILL_LINE)
    notice = os.environ.get("RLM_COMPACT_NOTICE", "")
    if notice in ("know", "stash"):
        parts.append(
            "The conversation may be compacted when it grows long; the IPython kernel survives "
            "compaction — variables persist, the transcript does not."
        )
    if notice == "stash":
        parts.append(
            "Keep key findings (paths, root causes, plans) in named Python variables so they survive."
        )
    return " ".join(parts)


def _ultraterse_prompt(cwd, installed_skills, *, allow_recursion, active_tools):
    """The absolute-minimum prompt: role + cwd + one line naming the REPL and edit skill."""
    parts = [
        "You are an agent that solves tasks by running code.",
        f"Working directory: {cwd}",
        "Your tool is a persistent Python REPL (use `%%bash` at the top of a cell for shell).",
    ]
    if "edit" in (installed_skills or []):
        parts.append("Edit files with `await edit(path=..., old_str=..., new_str=...)`.")
    if allow_recursion:
        parts.append("Spawn a sub-agent with `await rlm('sub-task')`.")
    if _should_include_git_history_guard(active_tools):
        parts.append(GIT_HISTORY_GUARD_PROMPT)
    return "\n".join(parts)


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    messages_path: str,
    *,
    allow_recursion: bool,
    active_tools: list[BuiltinTool],
    cli_skills: list[str] | None = None,
) -> str:
    """Build the system prompt.

    Layout: role → environment (cwd, log path, skills) → capabilities
    (recursion) → tool API. Keep it tight: the model also receives the
    per-tool schemas, so redundant tool guidance here just inflates
    every request.
    """
    if os.environ.get("RLM_PROMPT_TACTIC") == "bashedit":
        return _bashedit_prompt(installed_skills, active_tools=active_tools)
    if os.environ.get("RLM_PROMPT_TACTIC") == "ultraterse":
        return _ultraterse_prompt(cwd, installed_skills, allow_recursion=allow_recursion, active_tools=active_tools)
    if os.environ.get("RLM_MINIMAL_PROMPT"):
        return _minimal_prompt(
            cwd, skills_dir, installed_skills, messages_path,
            allow_recursion=allow_recursion, active_tools=active_tools, cli_skills=cli_skills,
        )
    parts: list[str] = [
        "You are a general purpose agent that uses code to solve tasks.",
        "You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.",
        "When you are done, stop calling tools and state your final answer.",
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {messages_path}",
        f"Pre-installed Python packages: {', '.join(BASE_TOOLKIT)}.",
        "Install additional packages with `uv pip install <pkg>` (this is a uv-managed venv with no pip module).",
    ]

    skill_lines: list[str] = []
    if skills_dir:
        skill_lines.append(
            f"Local skills live under {skills_dir}. Read their SKILL.md files when helpful."
        )
    if installed_skills:
        installed = ", ".join(f"`{skill}`" for skill in installed_skills)
        skill_lines.append(f"Installed skills (pre-imported): {installed}.")
        skill_lines.append(
            "Each skill is an async function by the same name. "
            "Inspect with `help(<skill>)` or `inspect.signature(<skill>.run)`."
        )
        if cli_skills:
            commands = ", ".join(f"`{skill}`" for skill in cli_skills)
            skill_lines.append(
                f"Skills with shell commands: {commands}. "
                "Discover CLI usage with `<skill> --help`."
            )
        if "edit" in installed_skills:
            skill_lines.append(EDIT_SKILL_PROMPT)
        if "search" in installed_skills:
            skill_lines.append(SEARCH_SKILL_PROMPT)
    if skill_lines:
        parts.extend(["", *skill_lines])

    if allow_recursion:
        parts.extend(
            [
                "",
                "A callable `rlm` is already in your global namespace — call it directly with `await rlm('sub-task')` to spawn a recursive sub-agent. Returns an `RLMResult` with `.answer` (string), `.usage`, `.turns`, and `.session_dir`.",
                "For parallel sub-agents, use normal Python async patterns such as `await asyncio.gather(rlm('task1'), rlm('task2'))`.",
            ]
        )

    if _has_tool(active_tools, "ipython"):
        parts.extend(["", IPYTHON_CONTROL_PROMPT])

    if _should_include_git_history_guard(active_tools):
        parts.extend(["", GIT_HISTORY_GUARD_PROMPT])

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    return "\n".join(parts)


def _should_include_git_history_guard(active_tools: list["BuiltinTool"]) -> bool:
    if allow_git():
        return False
    return any(tool.name in SHELL_TOOL_NAMES for tool in active_tools)


def _has_tool(active_tools: list["BuiltinTool"], name: str) -> bool:
    return any(tool.name == name for tool in active_tools)
