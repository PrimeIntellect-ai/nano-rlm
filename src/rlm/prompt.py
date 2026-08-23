"""System prompt construction."""

from __future__ import annotations

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
PROJECT_ENV_PROMPT = (
    "The ipython kernel is an isolated venv without the project's packages — "
    "never import project modules there. Everything that executes project code "
    "(tests, repros, imports) goes through bash with the project's interpreter."
)
IPYTHON_CONTROL_PROMPT = (
    "Run shell commands with `%%bash` as the very first line of a code cell "
    "(no comments, imports, or statements before it). " + PROJECT_ENV_PROMPT
)
EDIT_SKILL_PROMPT = (
    "Inside ipython you can also edit files with the pre-imported async `edit` "
    'skill: `await edit(path="pkg/file.py", old_str=..., new_str=...)` — handy '
    "for multiline or quote-heavy replacements built from Python strings."
)
SEARCH_SKILL_PROMPT = (
    "For web search, use the pre-imported async `search` skill from IPython: "
    '`await search(query="...")`. Results come back as title, URL, and snippet; '
    "assign the result to a variable so you can revisit it. To cover "
    "several angles at once, fan out with `asyncio.gather(search(...), search(...))`."
)


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
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
    has_bash = _has_tool(active_tools, "bash")
    has_edit = _has_tool(active_tools, "edit")
    has_ipython = _has_tool(active_tools, "ipython")
    role = "You are a coding agent."
    if has_bash:
        role += " You have access to a bash tool for running shell commands."
    if has_edit:
        role += (
            " You also have an edit tool for single-occurrence string "
            "replacement in a file."
        )
    if has_ipython:
        role += (
            " You also have an ipython tool: a persistent Python REPL "
            "(variables, imports, and function definitions persist across calls)."
        )
    parts: list[str] = [
        role,
        "When you are done, stop calling tools and state your final answer.",
        "",
        f"Working directory: {cwd}",
        "Conversation log: $RLM_SESSION_DIR/messages.jsonl",
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
        if "bash" in installed_skills and _has_tool(active_tools, "bash"):
            skill_lines.append(
                "Inside ipython you can also run shell with `await bash(command=...)` — "
                "it returns the output as a string, useful when mixing shell and Python "
                "in one cell or avoiding shell quoting."
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

    if has_ipython and not has_bash:
        parts.extend(["", IPYTHON_CONTROL_PROMPT])
    elif has_ipython:
        parts.extend(["", PROJECT_ENV_PROMPT])

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
