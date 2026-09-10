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
PROJECT_ENV_PROMPT = (
    "The ipython kernel is an isolated venv without the project's packages — "
    "never import project modules there. Everything that executes project code "
    "(tests, repros, imports) goes through bash with the project's interpreter."
)
IPYTHON_CONTROL_PROMPT = (
    "Run shell commands with `%%bash` as the very first line of a code cell "
    "(no comments, imports, or statements before it). " + PROJECT_ENV_PROMPT
)
KERNEL_PACKAGES_PROMPT = (
    "Pre-installed in the kernel venv: " + ", ".join(BASE_TOOLKIT) + ". "
    "Install extra packages with `!uv pip install <pkg>` in a code cell — that "
    "targets the kernel venv (a uv-managed venv with no pip module)."
)
BASH_SKILL_PROMPT = (
    "Run shell with `out = await bash('''command here''')` — always "
    "triple-quote the command so shell quotes and multi-line scripts never "
    "need escaping. It returns the output as a string; no need for "
    "`subprocess` or `%%bash`. Chain related commands with && in one call."
)
BASH_SKILL_WITH_TOOL_PROMPT = (
    "Inside ipython you can also run shell with `await bash(command=...)` — "
    "it returns the output as a string, useful when mixing shell and Python "
    "in one cell or avoiding shell quoting."
)
EDIT_SKILL_PROMPT = (
    "Inside ipython you can also edit files with the pre-imported async `edit` "
    'skill: `await edit(path="pkg/file.py", old_str=..., new_str=...)` — handy '
    "for multiline or quote-heavy replacements built from Python strings."
)
SEARCH_SKILL_PROMPT = (
    "For web search, use the pre-imported async `search` skill from IPython: "
    '`await search(query="...")` returns one formatted text string containing '
    "titles, URLs, and snippets, not a list of records. Assign it to a variable "
    "and print it to read the results. To cover "
    "several angles at once, fan out with `asyncio.gather(search(...), search(...))`."
)
FETCH_SKILL_PROMPT = (
    "To read a specific webpage, use the pre-imported async `fetch` skill: "
    '`await fetch(url="...")` returns the webpage as cleaned text. It can be used '
    "to open URLs from `search` results."
)

# One curated line per built-in skill, appended generically for whatever is enabled.
BUILTIN_SKILL_PROMPTS: dict[str, str] = {
    "bash": BASH_SKILL_PROMPT,
    "edit": EDIT_SKILL_PROMPT,
    "search": SEARCH_SKILL_PROMPT,
    "fetch": FETCH_SKILL_PROMPT,
}


def build_system_prompt(
    cwd: str,
    skills_dir: str | None,
    installed_skills: list[str],
    *,
    depth: int = 0,
    session_dir: str | None = None,
    allow_recursion: bool,
    allow_git: bool,
    active_tools: list[BuiltinTool],
    shell_skills: list[str] | None = None,
) -> str:
    """Build the system prompt.

    Layout: role → environment (cwd, log path, skills, kernel venv) →
    capabilities (recursion) → guards. Keep it tight: the model also receives
    the per-tool schemas, so redundant tool guidance here just inflates
    every request.
    """
    has_bash = _has_tool(active_tools, "bash")
    has_edit = _has_tool(active_tools, "edit")
    has_ipython = _has_tool(active_tools, "ipython")
    if depth > 0:
        role = (
            "You are a coding agent, spawned as a sub-agent: your caller "
            "delegated a task to you and can inspect your history. Do "
            "exactly that task; don't widen the scope."
        )
    else:
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
    if depth > 0:
        done_line = (
            "When the task is done, stop calling tools and state your final "
            "answer. Make it a "
            "complete, self-contained result — the answer plus the evidence "
            "needed to trust it (sources, file paths, values)."
        )
    else:
        done_line = "When you are done, stop calling tools and state your final answer."
    log_dir = session_dir or "$RLM_SESSION_DIR"
    parts: list[str] = [
        role,
        done_line,
        "",
        f"Working directory: {cwd}",
        f"Conversation log: {log_dir}/messages.jsonl",
        "Read the live conversation ledger with `from rlm import history; h = history()`. "
        "`h.messages[i]` addresses a session-wide message; `h.windows[w].messages[i]` addresses "
        "a message within a context window. Indices are zero-based. Compaction and rollback "
        "start new windows; earlier windows remain available. `h.user_messages()` returns "
        "original user inputs. `history(session_dir=path)` reads another session directory. "
        "Call `history(...)` again for a "
        "fresh snapshot. `h.events` includes spawn prompts and rollback markers. "
        "Full tool outputs and their shortened context versions have separate message indices. "
        "Search or print selected records to recover missing context; avoid printing the whole ledger. "
        "Failed attempts remain as history: `prompt_rollback.prompt_id` identifies their user record.",
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
            "Each skill is an async function by the same name; "
            "inspect one with `help(<skill>)`."
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
        for name in installed_skills:
            if prompt := _builtin_skill_prompt(name, active_tools):
                skill_lines.append(prompt)
    if skill_lines:
        parts.extend(["", *skill_lines])

    if has_ipython and not has_bash and "bash" not in (installed_skills or []):
        parts.extend(["", IPYTHON_CONTROL_PROMPT, KERNEL_PACKAGES_PROMPT])
    elif has_ipython:
        parts.extend(["", PROJECT_ENV_PROMPT, KERNEL_PACKAGES_PROMPT])

    if allow_recursion:
        parts.extend(
            [
                "",
                "The `rlm` package is available in Python. `child = await rlm.agent.spawn(task='...', name='researcher')` registers a child and returns a handle immediately; the child continues across cells.",
                "Use `await rlm.agent.list()` for child metadata, or `recursive=True` for descendants. Recover a direct child's handle with `await rlm.agent.get('researcher')` or its ID. Names are unique among siblings and reserved for this session.",
                "`await child.info()` reads status/task/timing; `child.history()` reads its conversation. `await child.result()` returns an RLMResult with .answer, .usage, .turns, .session_dir, or None while pending; failed/cancelled agents raise. `await child.wait(timeout=30)` waits at most that many seconds and returns current metadata. Waits use the cell's normal timeout and never cancel the agent. Avoid busy polling.",
                "`await child.cancel()` terminates the child and its descendants. An ordinary child releases its kernel after answering. `persistent=True` retains an idle kernel after answering; Use `await child.send(message)` to queue an instruction until an answer or explicit wait, and `await child.steer(message)` for the next model/tool boundary. Terminating a parent terminates all its descendants. Only direct children can be controlled; descendant listing grants no control.",
            ]
        )

    if has_ipython and (allow_recursion or depth > 0):
        parts.extend(
            [
                "",
                "Supervisor inbox: `await rlm.inbox.list()` lists unread event metadata without reading payloads. `await rlm.inbox.read(event_id)` retrieves a payload and marks it read; `list(unread_only=False)` includes read events. Child completion is automatic; reports use `await rlm.agent.send_to_parent(message)`. Children cannot steer parents or message siblings.",
                "Call the native `wait` tool when you have no work until a new event arrives. It suspends inference without occupying IPython. Already-announced unread events do not wake it repeatedly. Parent instructions are pushed automatically; reports and completion events require inbox reads. Notifications contain only an unread count. A final answer ends this ACP prompt; use wait to remain available. Queued instructions are delivered at an answer or wait boundary, steering at the next model/tool boundary; active tools are not interrupted.",
            ]
        )

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


def _builtin_skill_prompt(name: str, active_tools: list["BuiltinTool"]) -> str | None:
    """The curated line for one enabled built-in skill (None for uploaded skills).
    `bash` swaps its guidance when the native bash tool is also active — the skill
    is then the secondary shell path."""
    if name == "bash" and _has_tool(active_tools, "bash"):
        return BASH_SKILL_WITH_TOOL_PROMPT
    return BUILTIN_SKILL_PROMPTS.get(name)


def _has_tool(active_tools: list["BuiltinTool"], name: str) -> bool:
    return any(tool.name == name for tool in active_tools)
