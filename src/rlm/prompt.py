"""System prompt construction."""

from __future__ import annotations

from typing import TYPE_CHECKING

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


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    *,
    allow_recursion: bool,
    allow_git: bool,
    active_tools: list[BuiltinTool],
    shell_skills: list[str] | None = None,
) -> str:
    """Build the system prompt.

    Layout: role → environment (cwd, log path, skills) → capabilities
    (recursion) → tool API. Keep it tight: the model also receives the
    per-tool schemas, so redundant tool guidance here just inflates
    every request.
    """
    parts: list[str] = [
        "You are a general purpose agent that uses code to solve tasks.",
        "You solve tasks by breaking down problems into sub-tasks, writing and executing code, observing results, and iterating one step at a time.",
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
        shell_skill_set = set(shell_skills or [])
        if shell_skill_set:
            names = ", ".join(f"`{name}`" for name in sorted(shell_skill_set))
            skill_lines.append(
                f"Shell-enabled installed skills: {names}. Discover CLI usage with "
                "`<skill> --help`. Other listed skills are IPython-only."
            )
        else:
            skill_lines.append("The listed skills are IPython-only.")
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

    if _should_include_git_history_guard(active_tools, allow_git):
        parts.extend(["", GIT_HISTORY_GUARD_PROMPT])

    if active_tools:
        parts.extend(["", "Call at most one built-in tool per turn."])

    return "\n".join(parts)


def _should_include_git_history_guard(
    active_tools: list["BuiltinTool"], allow_git: bool
) -> bool:
    if allow_git:
        return False
    return any(tool.name in SHELL_TOOL_NAMES for tool in active_tools)


def _has_tool(active_tools: list["BuiltinTool"], name: str) -> bool:
    return any(tool.name == name for tool in active_tools)
