"""System prompt construction."""

from __future__ import annotations

import json
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

SHELL_TOOL_NAMES = frozenset({"ipython", "bash"})
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
    "Run background Bash commands with `job = await rlm.shell.run(command, cwd=...)`. "
    + PROJECT_ENV_PROMPT
)
KERNEL_PACKAGES_PROMPT = (
    "Pre-installed in the kernel venv: " + ", ".join(BASE_TOOLKIT) + ". "
    "Install extra packages with `!uv pip install <pkg>` in a code cell — that "
    "targets the kernel venv (a uv-managed venv with no pip module)."
)
BASH_SKILL_PROMPT = (
    "For short, blocking shell work, use `out = await bash('''command here''')` — always "
    "triple-quote the command so shell quotes and multi-line scripts never "
    "need escaping. It returns the output as a string; no need for "
    "`subprocess` or `%%bash`. Use rlm.shell.run for supervisor-owned background work."
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


RUNTIME_PROMPT = """## Runtime
Your execution environment is a persistent IPython REPL: each `ipython` call runs a cell in
the same kernel, so variables, imports and functions persist. Run quick shell commands (ls,
grep, cat, sed, git, a single test file — anything under about a minute) inline with
`!command` or a `%%bash` cell; the output comes back in the same turn. Do not wrap shell in
Python `subprocess`.

A supervisor outside the kernel owns background Bash jobs, agents, your inbox and
subscriptions, driven through the pre-imported async `rlm` API (top-level `await`). Handles
are references to supervisor-owned resources: ending a cell, losing a variable or a kernel
restart does not cancel them — recover them with `rlm.shell.get/list`, `rlm.agent.get/list`,
`rlm.watch.get/list`. After a kernel recovery notice your Python state is gone but jobs,
agents and inbox survive; never blindly re-run the interrupted cell.

## Background Bash jobs
`job = await rlm.shell.run(command, cwd=...)` starts a supervisor-owned Bash job and returns
at once. Use it only for genuinely long work (full test suites, builds, long repro loops) and
keep working while it runs. `info = await job.info()` gives .status (starting, running,
completed, failed, cancelled), .exit_code, .output_complete, .output_truncated, .error.
`chunk = await job.read(cursor=0, max_bytes=16384)` gives .text, .next_cursor, .done,
.truncated (cursors are bytes; at most 65536 bytes per read; 16 MiB retained per job).
`await job.cancel()` kills the process group. `await rlm.shell.list()` returns JobInfo
records, not handles — use `await rlm.shell.get(info.id)` to get a handle. No stdin/PTY.
Jobs die with their owner, so keep Bash alive until its work is finished.

## Inbox and waiting
Job completion arrives as an inbox event of type `shell.completed` whose content carries
job_id, status and exit_code. `await rlm.inbox.list()` returns unread events as dicts (id,
type, sender_id, created_at, read); `event = await rlm.inbox.read(event_id)` returns the
dict with ["content"] and marks it read. Notifications only show an unread count. When
nothing is actionable until a job or agent finishes, call the native `wait` tool (timeout at
most 300 s) — never sleep-poll in Python. Typical loop: start the job → do other work →
`wait` → read the shell.completed event → `job = await rlm.shell.get(job_id)` → info/read.

## Subscriptions
`await rlm.watch.job(job)` and `await rlm.watch.path(path, recursive=False)` post inbox
events (`watch.job` with a start:end byte range, `watch.path` with changed paths) for new
output or file changes; `rlm.watch.list()`, `rlm.watch.get(id)`, `await handle.cancel()`.
Completion notifications need no subscription. `help(rlm.shell.run)` etc. give details.
"""

AGENT_PROMPT = """## Delegation
`child = await rlm.agent.spawn(task, name="researcher", persistent=False)` returns an
AgentHandle immediately; give the child a self-contained task and expected result. Names are
unique among siblings. `await rlm.agent.list()` returns AgentInfo records (.id, .name, .status,
...); recover a handle with `await rlm.agent.get(name_or_id)`. `await child.info()`,
`await child.result()` (RLMResult with .answer/.usage/.turns, None while pending, raises on
failure), `child.history()`, `await child.cancel()`. Completion posts an `agent.completed`
inbox event (content.agent_id, content.status). `persistent=True` keeps an idle child for
follow-ups: `await child.send(message)` queues work, `await child.steer(message)` redirects
at the next model/tool boundary. `await child.wait(timeout=30)` blocks inside the cell;
prefer the native `wait` tool. `await rlm.watch.agent(child)` posts `watch.agent` events
with start:end indices into `child.history().messages`. Only direct children can be
controlled; terminating a parent ends its subtree.
"""

HISTORY_PROMPT = """## Conversation history
`from rlm import history; h = history()` snapshots your ledger: `h.messages[i]` (session-wide
index), `h.windows[w].messages[i]` (within a context window), `h.user_messages()` (original
inputs), `h.events` (spawn and rollback records). Compaction and rollback open new windows;
earlier records stay addressable. Search or print selected records, not the whole ledger.
"""


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
    task_instructions: str | None = None,
    extra_instructions: str | None = None,
    agent_info: dict | None = None,
) -> str:
    """Compose task instructions with the guide for this agent's actual runtime."""
    has_ipython = _has_tool(active_tools, "ipython")
    can_delegate = has_ipython and allow_recursion
    parts = [
        task_instructions
        if task_instructions is not None
        else "You are an agent. Complete the user's task using the available tools."
    ]
    if extra_instructions:
        parts.append(extra_instructions)
    parts.append("## Agent context")
    if agent_info:
        parts.append(
            "Supervisor identity: "
            + json.dumps(
                {
                    key: agent_info[key]
                    for key in ("id", "parent_id", "name", "persistent")
                }
            )
        )
    if depth > 0:
        parts.append(
            "You are a sub-agent. Work on the task delegated by your immediate parent; do not widen its scope. Return a self-contained answer with relevant evidence, sources, paths, and uncertainties."
        )
        if agent_info and agent_info["persistent"]:
            parts.append(
                "After answering, you become idle and retain your kernel for follow-up instructions or inbox arrivals. You do not need to wait merely to keep your persistent session alive."
            )
        else:
            parts.append(
                "After your final answer, your runtime and descendants are terminated. Finish necessary child/job work before answering."
            )
    else:
        parts.append(
            "You are the root agent. A final answer returns control to the caller. Do not claim completion while required work remains pending."
        )
    parts.extend(
        [
            f"Working directory: {cwd}",
            f"Conversation log: {session_dir or '$RLM_SESSION_DIR'}/messages.jsonl",
        ]
    )
    parts.append(
        "Available native tools: "
        + ", ".join(
            [tool.name for tool in active_tools] + (["wait"] if has_ipython else [])
        )
        + ". Call at most one native tool per model step."
    )
    if has_ipython:
        parts.extend(
            [
                RUNTIME_PROMPT,
                HISTORY_PROMPT,
                IPYTHON_CONTROL_PROMPT,
                KERNEL_PACKAGES_PROMPT,
            ]
        )
        if can_delegate:
            parts.append(AGENT_PROMPT)
        else:
            parts.append(
                "Delegation is disabled at this depth. Work directly with your available tools."
            )
        if depth > 0:
            parts.append(
                "Use `await rlm.agent.send_to_parent(message)` to put a report in your immediate parent's inbox; it returns an event ID. Your parent chooses when to read it. Parent instructions are pushed automatically: queued input at an answer/wait boundary, steering at the next model/tool boundary. You cannot steer your parent or message siblings. If you have children, their reports enter your own pull-based inbox in the same way."
            )
        else:
            parts.append(
                "You have no parent: rlm.agent.send_to_parent is unavailable. Child reports arrive as agent.message inbox events; read their content when useful."
            )
    if _has_tool(active_tools, "bash"):
        parts.append(
            "The native bash tool executes a command to completion or timeout and returns text. It does not return a background-job handle."
        )
    if _has_tool(active_tools, "edit"):
        parts.append(
            "Use the native edit tool for exact single-occurrence string replacement; its schema describes the arguments."
        )
    if skills_dir:
        parts.append(
            f"Local skills live under {skills_dir}. Read their SKILL.md files when helpful."
        )
    shell_skill_set = set(shell_skills or [])
    if installed_skills and has_ipython:
        parts.append(
            "Installed skills (pre-imported): "
            + ", ".join(f"`{name}`" for name in installed_skills)
            + ". Each is async; use help(skill) for its signature."
        )
        for name in installed_skills:
            if guidance := _builtin_skill_prompt(name, active_tools):
                parts.append(guidance)
    if shell_skill_set and (has_ipython or _has_tool(active_tools, "bash")):
        parts.append(
            "Shell-enabled installed skills: "
            + ", ".join(f"`{name}`" for name in sorted(shell_skill_set))
            + ". Discover CLI usage with `<skill> --help`. Other listed skills are IPython-only."
        )
    if _should_include_git_history_guard(active_tools, allow_git):
        parts.append(GIT_HISTORY_GUARD_PROMPT)
    return "\n\n".join(parts)


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
