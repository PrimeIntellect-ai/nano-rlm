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
    "Use `rlm.shell.run` for quick Bash results and `rlm.shell.start` for background work. "
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
    "need escaping. It returns the output as a string, useful for further Python processing. "
    "Use rlm.shell.start for supervisor-owned background work."
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


RUNTIME_PROMPT = """## Runtime and ownership
You have a persistent IPython REPL as your execution environment. Each `ipython` tool call
runs a cell in the same kernel, so variables, imports, and functions remain available to
later cells. Use Python to program over tools and coordinate concurrent work.

A supervisor runs outside your IPython kernel. It manages agents, background Bash jobs,
message delivery, and subscriptions. The pre-imported `rlm` Python API lets you ask it to
create, inspect, and control these resources; execute async API calls with top-level `await`.
Handles stored in Python variables are references to supervisor-owned resources.
Finishing or cancelling a cell, losing a handle variable, or restarting IPython does not
cancel those resources. Recover handles through their registries. Terminating an agent
cleans up its children, jobs, and subscriptions.

Parent instructions are delivered automatically. Child reports and watcher events enter
your inbox; lightweight notifications let you choose when to read them. Agents share a
filesystem as trusted collaborators; handles enforce orchestration ownership, not
filesystem isolation.

A kernel recovery notice means Python variables/imports/in-kernel tasks were lost.
Reconstruct them and recover handles through the registries below. Never blindly repeat
an interrupted cell: file writes and accepted spawn/send/job requests may already have
happened. Compaction alone preserves the kernel and supervisor resources.

## Bash commands and background jobs
Use `result = await rlm.shell.run(command, cwd=...)` for quick commands whose results you
need immediately. It waits for Bash and output capture to finish. For example:
```python
result = await rlm.shell.run("git status --short")
print(result.text, result.exit_code)
```
Commands can contain multiline Bash scripts. The result has .text (combined stdout/stderr,
up to 16 KiB), .exit_code, .job_id, .truncated, and .error. Nonzero exit codes are returned;
startup/capture failures populate .error. If .truncated is true, recover the job with
`await rlm.shell.get(result.job_id)` to read more captured output or inspect metadata.

Use `job = await rlm.shell.start(command, cwd=...)` for long commands or work you want to
run alongside other tasks. It returns a JobHandle promptly, before completion. Use job
handles for output subscriptions, reading progress, and cancellation.
Both calls run supervisor-owned Bash, with no stdin/PTY. Default cwd is this agent's
working directory; relative cwd resolves against it. Cancelling a cell awaiting run()
stops waiting but leaves the job running. Jobs survive kernel restarts; recover their IDs
with shell.list(). Both calls publish shell.completed inbox events.
`await rlm.shell.list()` returns JobInfo objects; `await rlm.shell.get(job_id)` recovers
a handle. `job.id` is stable. `await job.info()` returns metadata with .status,
.exit_code, .output_complete, .output_truncated, and .error. Status is starting, running, completed, failed, or cancelled. Nonzero exit codes are
completed processes; failed means startup/capture failure. `await job.cancel()` stops
the process group. Owner termination cancels its jobs, including background descendants.
Keep Bash alive until its work finishes; detached processes are outside this guarantee.

`chunk = await job.read(cursor=0, max_bytes=16384)` returns .text, .next_cursor, .done,
and .truncated. Reads are repeatable, not consuming; save next_cursor for the next read.
Cursors count bytes, not characters. Reads allow at most 65536 bytes and capture retains
16 MiB per job. .done means the job ended and this read reached the retained output's end;
it does not imply success or exhaustive output. Check the job's exit code and capture flags.
IPython `!`/`%%bash` and any enabled blocking bash skill/tool are not supervisor-owned jobs.

Agent cleanup errors are reported separately in AgentInfo.cleanup_error; completed answers remain readable. Use cancel() to retry unfinished cleanup.

## Inbox and waiting
`await rlm.inbox.list()` returns unread event dictionaries: ["id"], ["type"],
["sender_id"], ["created_at"], ["read"]. Listing does not mark events read.
`event = await rlm.inbox.read(event_id)` returns a dictionary with ["content"] and
marks it read. `list(unread_only=False)` includes read events; reads are repeatable.
A read flag means retrieved, not completed or acted upon.

Supervisor notifications show only an unread count. You choose when to inspect payloads.
When work remains but nothing is actionable, call the native `wait` tool (outside Python),
with timeout at most 300 seconds. It suspends inference without holding a cell open.
New arrivals wake it; already-announced unread events do not. Inspect existing unread
events before waiting for more. Avoid polling/sleep loops in Python to wait for agents/jobs.

Bash completion arrives automatically as type `shell.completed`, with
`event["content"]["job_id"]`; recover the job to read output and inspect its outcome.
For example, start a job in one cell:
```python
job = await rlm.shell.start("uv run pytest tests/", cwd="/workspace/project")
```
Continue other work, or call native `wait`. In a later cell, inspect relevant arrivals:
```python
for item in await rlm.inbox.list():
    if item["type"] == "shell.completed":
        event = await rlm.inbox.read(item["id"])
        job = await rlm.shell.get(event["content"]["job_id"])
        info = await job.info()
        chunk = await job.read()
        print(info.status, info.exit_code, chunk.text)
```
Adapt the command and cwd to the actual task. Continue reading with next_cursor if needed.

## Subscriptions
`await rlm.watch.job(job)` observes newly captured output from an owned job.
Subscriptions become completed after the target permanently terminates and pending
activity is delivered. Watching an already-finished target returns a completed
subscription. Watches of idle persistent agents remain active.
`await rlm.watch.path(path, recursive=False)` observes an existing file/directory;
relative paths use this agent's cwd. Recursion is opt-in. Both return handles with .id
and async .cancel(). `await rlm.watch.list()` returns SubscriptionInfo objects with
.id, .kind, .target, .status, .error; `await rlm.watch.get(id)` recovers a handle.
No events selector is needed. Completion notifications require no subscription.

Watches observe future activity, batching arrivals over 200 ms. An inbox event carries
["subscription_id"] and ["content"]["target"]. `watch.job` content has an exclusive
start:end byte range; `watch.path` content has paths and truncated. Read the referenced
output/files when useful. `watch.failed` explains failure in `event["content"]["error"]`; inspect
metadata and register again after fixing the cause. Observed path removal stops its watch.
Cancellation stops future events and drops an unpublished batch, retaining published
inbox events. Owner termination cancels subscriptions. Limits are 64 active / 1024 total
subscriptions per tree; oversized path batches report truncation explicitly.

Use `help(rlm.shell.start)`, `help(rlm.watch.path)`, or `help(type(handle))` for signatures
and details. Objects use attributes; inbox events and history messages are dictionaries.
"""

AGENT_PROMPT = """## Delegation
`child = await rlm.agent.spawn(task, name="researcher", persistent=False)` returns
an AgentHandle immediately. Give the child a self-contained task, relevant constraints,
and an expected result. Names are unique among siblings and reserved for the session.
`await rlm.agent.list()` returns AgentInfo objects with .id, .parent_id, .name, .task,
.status, .persistent, .session_dir, and timing. `recursive=True` also lists descendants;
only direct children can be controlled. Finished children remain discoverable. Recover a direct child with
`await rlm.agent.get(name_or_id)`. Reassigning/deleting a Python handle does not stop it.

`await child.info()` reads metadata. `await child.result()` returns an RLMResult
(.answer, .usage, .turns, .session_dir), or None before its first answer. The latest answer
remains available while a persistent child runs again; use info/wait for current activity.
Terminal failure/cancellation raises.
Child completion/failure posts `agent.completed` automatically;
`event["content"]["agent_id"]` identifies the child and ["status"] gives its state. Inspect the event and recover the handle rather than assuming success.
`child.history()` returns a fresh history snapshot. `await child.cancel()` terminates
that child and its descendants. Terminating a parent ends its whole subtree.

Use persistent=True for follow-up work: the child becomes idle after answering and retains
its kernel/conversation. `await child.send(message)` queues work until an answer or native
wait boundary; `await child.steer(message)` delivers at the next model/tool boundary during
ongoing work. Neither interrupts running code. These IDs acknowledge acceptance, not
processing. An exhausted tree budget rejects new instructions. If accepted instructions
become undeliverable, your inbox receives `agent.delivery_failed` with content fields
agent_id, message_id, and reason (for example, tree_budget_exhausted).
`await child.wait(timeout=30)` waits inside the Python cell and returns AgentInfo, not the
result; prefer native wait when you have no other work. Cell timeouts still apply.

`await rlm.watch.agent(child)` watches a direct child's conversation after complete
assistant/tool steps, including final answers. Its `watch.agent` event content identifies
the child via target and gives start:end indices for
`child.history().messages[start:end]`. It observes progress without waiting for an explicit
report. Read history, then steer if needed; the subscription itself does not direct the child.
"""

HISTORY_PROMPT = """## Conversation history
`from rlm import history; h = history()` reads a snapshot of your ledger.
`h.messages[i]` addresses a session-wide message; `h.windows[w].messages[i]` addresses
one within a context window. Indices are zero-based. Messages are dictionaries with
role/content/tool fields. `h.user_messages()` returns original user inputs, distinct
from generated summaries and supervisor notices. `history(session_dir=path)` reads an
explicit session; use a child handle's .history() when available. Reload for fresh state.

Compaction and rollback start new windows; earlier records remain addressable. Full tool
outputs and shortened context versions have separate indices. `h.events` contains spawn
and rollback records; prompt_rollback.prompt_id identifies a rolled-back user attempt.
History records what happened, not proof that side effects were undone. Recover exact
instructions and evidence by searching/selectively printing records, not the entire ledger.
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
