"""System prompt construction."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from rlm.tools.base import BuiltinTool
    from rlm.tools.skills import SkillDoc


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
    "reflogs, or broad git history in the project, or by cloning other "
    "repositories to read their history. Git commands that reach beyond the "
    "current branch are refused: `git log --all` and similar history options, "
    "history subcommands given another ref (`git show origin/main`), listing "
    "branches, tags, remotes or the reflog, and `clone`, `fetch` or `pull`."
)
PROJECT_ENV_PROMPT = (
    "The ipython kernel is an isolated venv without the project's packages — "
    "never import project modules there. Everything that executes project code "
    "(tests, repros, imports) goes through bash with the project's interpreter."
)
IPYTHON_CONTROL_PROMPT = (
    "Use `rlm.shell.run` for Bash; `yield_after=` is how long it waits before yielding a handle (0 = at once) and "
    "`await job.result()` collects any job. Use `rlm.services.start` for a server or daemon that must keep "
    "running after you finish. " + PROJECT_ENV_PROMPT
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
    "Use rlm.shell.run(..., yield_after=0) for supervisor-owned background work, and "
    "rlm.services.start for a server or daemon that must keep running after you finish."
)
BASH_SKILL_WITH_TOOL_PROMPT = (
    "Inside ipython you can also run shell with `await bash(command=...)` — "
    "it returns the output as a string, useful when mixing shell and Python "
    "in one cell or avoiding shell quoting."
)
EDIT_SKILL_PROMPT = (
    "Change existing files with the pre-imported async `edit` tool, not with "
    '`str.replace` + `write_text`: `await edit(path="pkg/file.py", old_str=..., new_str=...)` '
    "replaces exactly one occurrence and raises ValueError when old_str is absent or "
    "appears more than once, so a stale or ambiguous hunk cannot be applied silently. "
    "Several hunks go in one cell:\n"
    "```python\n"
    "for old, new in [(OLD_IMPORTS, NEW_IMPORTS), (OLD_CALL, NEW_CALL)]:\n"
    '    print(await edit(path="src/pkg/module.py", old_str=old, new_str=new))\n'
    "```\n"
    "Use write_text only for files you create."
)
SEARCH_SKILL_PROMPT = (
    "For web search, use the pre-imported async `search` tool from IPython: "
    '`await search(query="...")` returns one formatted text string containing '
    "titles, URLs, and snippets, not a list of records. Assign it to a variable "
    "and print it to read the results. To cover "
    "several angles at once, fan out with `asyncio.gather(search(...), search(...))`."
)
FETCH_SKILL_PROMPT = (
    "To read a specific webpage, use the pre-imported async `fetch` tool: "
    '`await fetch(url="...")` returns the webpage as cleaned text. It can be used '
    "to open URLs from `search` results."
)

MCP_DISCOVERY_PROMPT = """## MCP tool discovery
Use `await rlm.tools.servers()` for server names, tool counts and available descriptions. Search by keywords
with `page = await rlm.tools.search("calendar events", limit=5, server=None)`.
Use a non-empty keyword query and a limit from 1 to 20; an empty query cannot list tools.
`page["matches"]` contains id, server and a short description, without argument schemas.
Each page has at most 20 matches. If `page["next_cursor"]` is not None, retrieve more
with the same query and server and `cursor=page["next_cursor"]`. Stop when it is None.
If no match fits, try other keywords or filter by server. Load a selected tool with
`tool = await rlm.tools.get(page["matches"][0]["id"])`. Inspect `tool.schema` for the full
JSON input schema (nested fields, enums and constraints), or `help(tool)` for its
Python signature and description, then call `await tool(...)` using that schema.
Save callables in variables to reuse them, and use asyncio.gather for independent calls.
MCP tools are loaded async functions: use `await tool(...)`, not guessed methods such as
`tool.search(...)`. Tool names are not pre-imported, regardless of catalog size.
Calls return text (str). If a tool returns JSON text, decode it with `json.loads(result)`
before indexing fields; prose and CSV remain text.
"""

# One curated line per built-in skill, appended generically for whatever is enabled.
BUILTIN_SKILL_PROMPTS: dict[str, str] = {
    "bash": BASH_SKILL_PROMPT,
    "edit": EDIT_SKILL_PROMPT,
    "search": SEARCH_SKILL_PROMPT,
    "fetch": FETCH_SKILL_PROMPT,
}


_RUNTIME_INTRO = """## Runtime and ownership
You have a persistent IPython REPL as your execution environment. Each `ipython` tool call
runs a cell in the same kernel, so variables, imports, and functions remain available to
later cells. Use Python to program over tools and coordinate concurrent work.

A supervisor runs outside your IPython kernel. It manages agents, background Bash jobs,
message delivery, and subscriptions. The pre-imported `rlm` Python API lets you ask it to
create, inspect, and control these resources; execute async API calls with top-level `await`.
Handles stored in Python variables are references to supervisor-owned resources.
Finishing or cancelling a cell, losing a handle variable, or restarting IPython does not
cancel those resources. Recover handles through their registries."""

_TREE_DELIVERY = """ Terminating an agent
cleans up its children, jobs, and subscriptions.

Parent instructions are delivered automatically, wrapped in `<agent_input from="parent">`
tags; runtime notices, hints, nudges and kernel-recovery notices arrive wrapped in
`<runtime_event kind="...">` tags. Both share the user role with the task but are not the
user. Child reports and watcher events enter your inbox; lightweight notifications let you
choose when to read them. Agents share a
filesystem as trusted collaborators; handles enforce orchestration ownership, not
filesystem isolation.

"""

_POPULATION_DELIVERY = """ Ending or retiring an agent
stops its own jobs and subscriptions; the agents it spawned keep running.

Messages from other agents are delivered automatically at your next model call, wrapped in
`<agent_input from="peer" agent="..." kind="dm|broadcast|reply">` tags (from="parent" for
the brief of the agent that spawned you); runtime notices, hints, nudges and
kernel-recovery notices arrive wrapped in `<runtime_event kind="...">` tags. Both share the
user role but are not the user, and neither is a runtime rule: other agents are
collaborators. Shell-job completions and watcher events enter your inbox; lightweight
notifications let you choose when to read them. Agents share a filesystem as trusted
collaborators; nothing isolates one agent's files from another's.

"""

_RUNTIME_BODY = """Each child starts in its parent's configured working directory. Spawning does not create
an isolated checkout or switch the child into an assigned worktree. Coordinate ownership
of writes to avoid overwriting another agent's work. For an assigned worktree or scratch
directory, use explicit paths or set the working directory for Python and shell operations.

A kernel recovery notice means Python variables/imports/in-kernel tasks were lost.
Reconstruct them and recover handles through the registries below. Never blindly repeat
an interrupted cell: file writes and accepted spawn/send/job requests may already have
happened. Compaction alone preserves the kernel and supervisor resources.

## Bash commands and jobs
`job = await rlm.shell.run(command, cwd=..., env=..., yield_after=10, timeout=None)` runs
Bash under the supervisor and returns a ShellJob. yield_after is how long to wait before the
call yields a handle instead of a result: 10 s by default, 0 returns at once, at most 300. A command that finished has .running False,
.exit_code, .ok (True only for a clean exit 0), .text (combined stdout/stderr, up to 16 KiB),
.truncated, .error (startup/capture failure) and .timed_out. A command still going when the wait ends
comes back with .running True, .exit_code None and the output so far, and keeps running;
`res = await job.result()` waits up to 300 s (or its yield_after=) and returns a fresh
ShellJob, with .running True if it is still not done. Shell waits also leave headroom within
the current cell's remaining execution time. result() is repeatable and never
consumes output. A ShellJob is a snapshot: its .running and .text do not change by
themselves, result() returns a fresh one. result() already waits, so never call native
`wait` for a job you hold. Nothing is killed by yielding; timeout=, if given, kills the process
group after that many seconds (.timed_out True, .exit_code None), and `await job.cancel()`
stops a job at any time. The command is a Bash string or an argv list.
```python
r = await rlm.shell.run("git status --short")                  # finished within 10 s
if not r.ok:
    print("FAILED", r.exit_code, r.text)
job = await rlm.shell.run("go test ./...", cwd="/workspace/project", yield_after=0)   # handle at once
# ... other work in this or later cells ...
res = await job.result()                                        # waits (up to 300 s) for the finished job
print(res.exit_code, res.text)
first, second = await asyncio.gather(job_a.result(), job_b.result())   # several jobs at once
```
Act on the exit code before reading the text: a nonzero .exit_code means the command failed
even if .text looks plausible, so branch on it rather than only printing it. Pass cwd=
instead of prefixing `cd dir &&`. Environment variables the project needs (PYTHONPATH,
GOFLAGS, ...) are set once with `await rlm.shell.setenv(PYTHONPATH="/app/lib")` and apply
to every later run(); `env={...}` applies to one call. Startup/capture failures populate
.error. Bash runs with pipefail on: a pipeline's exit code is that of its rightmost failing
stage, so `pytest ... | tail -20` reports pytest's failure, and a `grep` with no match makes
the pipeline exit 1.

.text is capped at 16 KiB: longer output keeps its first and last 8 KiB around a marker
that names the omitted byte range, so the first failure and the final summary both survive
without piping through `tail`. Prefer `grep -n`, `sed -n 'A,Bp'`, and `head` over printing
whole files; print the parts needed for your next decision. A pipeline cut short by `head`
exits 141 (SIGPIPE): the producer was interrupted, not failed. If .truncated is true, the
output is retained (up to 16 MiB per job): `chunk = await job.read(cursor=8192,
max_bytes=65536)` returns .text, .next_cursor, .done and .truncated; cursors count bytes.
Multi-line commands (heredocs, `python -c`, small scripts) are fine. Write the command as a
triple-quoted raw string so quotes, backslashes and newlines inside need no escaping, using
the triple quote the text does not contain (`r'''...'''` for text with `\"\"\"`, `r\"\"\"...\"\"\"`
for text with `'''`):
```python
r = await rlm.shell.run(r'''python3 - <<'EOF'
import package
print(package.__version__, "quotes 'inside' need no escaping")
EOF''', cwd="/workspace/project")
```
Longer scripts and scratch tests go to a file with the same delimiter rule, then run with the
project's own toolchain:
```python
from pathlib import Path
script = Path("/tmp/repro/check.py")
script.parent.mkdir(parents=True, exist_ok=True)
script.write_text(r\"\"\"import package
'''Docstrings and backslashes inside are fine: the outer delimiter is a raw triple double quote.'''
print(package.__version__)
\"\"\")
r = await rlm.shell.run(["python3", str(script)], cwd="/workspace/project")
```
The supervisor API is the only shell: do not call subprocess or os.system from the kernel.
Reuse Python variables and helpers across cells.

Jobs run supervisor-owned Bash with no stdin/PTY. Default cwd is this agent's working
directory; relative cwd resolves against it. Cancelling a cell that is awaiting run() or
result() stops waiting but leaves the job running. Jobs survive kernel restarts:
`await rlm.shell.list()` returns JobInfo snapshots (.id, .command, .status, .exit_code,
.timeout, .error; status is starting, running, completed, failed, cancelled, or timed_out;
nonzero exit codes are completed processes, failed means startup/capture failure), and
`await rlm.shell.get(job_id)` recovers the ShellJob for a lost variable.
`await job.cancel()` stops the process group. Owner termination cancels its jobs. A job's
processes, including anything it started with `&` or `nohup`, are stopped when the job ends,
and running jobs are stopped when you finish. Keep Bash alive until its work finishes. Before
your final answer, `await rlm.shell.list()` must show no running job whose result you still need.
Only a job handed back with .running True posts a shell.completed inbox event when it ends;
finished results post nothing. Those events are quiet: they wake native `wait` but are not
counted in the unread notice, because `await job.result()` already collects the job. IPython `!`/`%%bash` and any enabled blocking bash
tool are not supervisor-owned jobs.

Agent cleanup errors are reported separately in AgentInfo.cleanup_error; completed answers remain readable. Use cancel() to retry unfinished cleanup.

## Services
A process that must keep running after you finish, such as a server or daemon you are asked
to leave running, is a service, not a job: `svc = await rlm.services.start(name, command,
cwd=..., env=...)` starts it in its own session with its output appended to a log file, and
it keeps running after your session ends. It returns a Service snapshot (.name, .pid, .log,
.running, .exit_code); `await svc.tail(50)` reads the end of its log and `await svc.stop()`
stops it. start() returns at once, so check the service came up (tail() or connect to it)
before relying on it. `await rlm.services.list()` and `await rlm.services.get(name)` recover
services after kernel restarts. Starting a name that is still running raises.
```python
svc = await rlm.services.start("api", "python3 server.py --port 8080", cwd="/app")
r = await rlm.shell.run("curl -sf localhost:8080/health")       # check it answers
if not r.ok:
    print(await svc.tail(50))
```

## Inbox and waiting
`await rlm.inbox.list()` returns unread event dictionaries: ["id"], ["type"],
["sender_id"], ["created_at"], ["read"]; listed items carry no ["content"] and listing
does not mark events read. `event = await rlm.inbox.read(event_id)` returns a dictionary
with ["content"] and marks it read. Content is always a dictionary: index its keys, do not
slice it. An `agent.message` a child sent has ["agent_id"], ["name"] and ["text"].
`list(unread_only=False)` includes read events; reads are repeatable.
A read flag means retrieved, not completed or acted upon.

Supervisor notifications show the unread count when it changes, plus occasional one-line hints about
runtime use, each with a tag; `await rlm.hints.mute("tag")` stops a hint you have understood.
You choose when to inspect payloads.
When work remains but nothing is actionable and you hold no running job (a held job is
collected with `await job.result()`), call the native `wait` tool (outside Python), with
timeout at most 300 seconds. It suspends inference without holding a cell open.
New arrivals wake it; already-announced unread events do not. Inspect existing unread
events before waiting for more. Avoid polling/sleep loops in Python to wait for agents/jobs.

A job that was handed back running posts a quiet `shell.completed` when it ends, with content
job_id, status, exit_code and text (the last 4 KiB of output); `await job.result()` marks it
read. Keep the job variable and call `result()` when you need the outcome; the inbox route is
only for a job whose variable you lost after native `wait`:
```python
for item in await rlm.inbox.list():
    if item["type"] == "shell.completed":
        event = await rlm.inbox.read(item["id"])
        res = await (await rlm.shell.get(event["content"]["job_id"])).result()
        print(res.exit_code, res.text)
```

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

If unsure of an API's arguments, inspect its signature before calling it:
`help(rlm.shell.run)`, `help(rlm.watch.path)`, or `help(type(handle))`.
Objects use attributes; inbox events and history messages are dictionaries.
"""

RUNTIME_PROMPT = _RUNTIME_INTRO + _TREE_DELIVERY + _RUNTIME_BODY
POPULATION_RUNTIME_PROMPT = _RUNTIME_INTRO + _POPULATION_DELIVERY + _RUNTIME_BODY

AGENT_PROMPT = """## Delegation
`child = await rlm.agent.spawn(task, name="researcher", persistent=False)` returns
an AgentHandle (.id, .name, .session_dir) immediately. Give the child a self-contained task, relevant constraints,
and an expected result. Names are unique among siblings and reserved for the session.
`await rlm.agent.list()` returns AgentInfo objects with .id, .parent_id, .name, .task,
.status, .persistent, .turns, .session_dir, and timing. `recursive=True` also lists descendants;
only direct children can be controlled. Finished children remain discoverable. Recover a direct child with
`await rlm.agent.get(name_or_id)`. Reassigning/deleting a Python handle does not stop it.

`await child.info()` reads metadata. `await child.result()` waits up to 300 s (or its
yield_after=) for the child to finish and returns an AgentResult (.status, .answer, .usage,
.turns, .session_dir, .running). If the child is still working on its first answer, .answer is
None and .running is True: keep going and collect later, or from the `agent.completed` event.
A persistent child's latest answer stays available while it runs again. Terminal
failure/cancellation raises.
Child completion/failure posts `agent.completed` automatically; its content has
["agent_id"], ["name"], ["status"], ["turns"], ["error"] and ["answer"] (the last 4 KiB of
the child's answer), so the event alone tells you what came back; `await child.result()`
has the full answer. Inspect the event rather than assuming success.
`await child.history()` returns a fresh history snapshot. `await child.cancel()` terminates
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
`(await child.history()).messages[start:end]`. `await rlm.watch.agent(child, every_turns=10)` (and/or
`every_tokens=50000`) instead posts a `watch.progress` event each time the child's own model
calls or new tokens cross the next multiple, with ["turns"], ["tokens"], ["name"], ["status"]
and the same start:end slice — the way to keep a long-running child in view: read its recent
history, then `await child.steer("report what you have and stop")` if it should wrap up. The
subscription itself does not direct the child.
"""

DELEGATION_PROMPT = """## Delegating work
When a task has independent parts that benefit from separate investigation, delegate them to subagents while you work on another part. Give each child a bounded, self-contained brief, relevant context, and a clear expected result. Avoid duplicating the same investigation yourself unless independent verification is useful.
Use persistent=False for a one-shot assignment and persistent=True when you expect follow-ups on the same context.
For longer assignments, register await rlm.watch.agent(child, every_turns=10) to receive progress updates. Inspect the referenced slice of (await child.history()).messages when you need to assess progress. Steer a child if its scope drifts or its remaining work is unlikely to help; cancel work you no longer need.
Collect reports with result = await child.result(). Check result.running before treating the assignment as finished; a persistent child may retain an earlier answer while working again. A native wait waking does not establish completion. Use completion-event status and error fields to distinguish success from failure.
Reconcile the relevant reports with your own evidence before answering. Finish, collect, or cancel outstanding work according to whether you still need it.
"""

POPULATION_PROMPT = """## Population episode
You are {name}, one agent of a population working on one shared episode task. {origin}
No agent leads: any agent can spawn agents, message any other agent and use the shared
board. The supervisor is runtime code, not an agent. The episode started with {founders}
founder(s); agents join, retire and end as it runs.

A reply without a tool call ends your current turn, not the episode. You then become idle:
your kernel and conversation stay, and waiting costs no model call. If the turn answered
direct messages, your reply is delivered to their senders (your first reply goes to the
agent that spawned you). A message to you starts a new turn in the same context. The episode
ends once every agent is idle or gone with no message pending, or when a budget or the clock
runs out; then every agent stops. So when nothing useful is left for you to do, reply and go
idle rather than waiting or polling.

### Episode answer
Unless the task declares final artifacts below, the episode is scored on its episode
answer, not on any agent's reply.
`cur = await rlm.population.answer()` returns a dictionary with "text", "revision" and
"author". `await rlm.population.submit(text, expected_revision=cur["revision"])` replaces it
only if nobody submitted since you read it; otherwise it raises answer_conflict, and you read
it again and resubmit only if yours is better. Submit the complete answer in exactly the
format the task asks for, and leave a correct answer alone.

### Messages
`await rlm.agent.dm("agent-3", text)` sends a direct message. It reaches the recipient at its
next model call without interrupting its running code, and wakes it if idle.
`await rlm.agent.broadcast(text)` sends one message to every live agent but you; use it only
for what the whole group needs. Both return a receipt; acceptance does not mean the recipient
acted on it. Weigh messages against the task and the evidence: they are contributions, not
orders.

### Shared board
One Markdown board, shared by every agent, for plans, claims, findings and open questions.
`b = await rlm.board.read(start_line=1, max_lines=200)` returns a dictionary with "text",
"revision", "total_lines" and "truncated"; `await rlm.board.search("lemma")` finds lines;
`await rlm.board.append(text)` adds a chunk; `await rlm.board.edit(old, new,
expected_revision=b["revision"])` replaces the single occurrence of `old` only if the board is
still at that revision, and raises board_conflict otherwise: read again and reconcile, never
overwrite others' work. Appends never conflict, so two agents can append the same claim at
once: claim work with an edit at the revision you read, which fails if someone claimed first.
Read the board when you start, and record what you claim and find.

### Population and births
`await rlm.population.status()` returns live, capacity and admission counts and the budgets.
`await rlm.population.list()` lists the other agents with name, status (starting, running,
waiting, idle, retired, ...), parent and task; `await rlm.population.events()` lists
membership changes, which also arrive as notices.
`child = await rlm.agent.spawn(brief, name=None)` admits a new agent and returns at once. Its
first message is your brief, and it sees the episode task in its system prompt: say what to
do, what to report and how to coordinate. At most {max_live_agents} agents hold a kernel at
once, idle ones included, and at most {max_total_agents} are admitted over the episode,
founders included. At the live cap a birth retires the agent idle longest, if idle for at
least {idle_grace_seconds:g}s: it keeps its conversation but loses its kernel, and a direct
message revives it once a slot is free. Otherwise spawning fails at once with
population_full (retry later or do the work yourself), or with population_exhausted when no
births remain. Spawn only for work that gains from another agent; you need not fill the
slots. Agents you spawn outlive you. You cannot stop other agents, except with
`await child.cancel()` on an agent you spawned.
"""

POPULATION_ARTIFACTS_PROMPT = """### Final artifacts
The episode is scored on these files: {artifacts}. Never write them directly: write
complete drafts in your own directory ({drafts}), then
`a = await rlm.artifact.read(path)` (returns "revision", "author", "exists", "sha256") and
`await rlm.artifact.publish(path, draft_path, expected_revision=a["revision"])`, which
replaces the file only if nobody published since you read it and raises artifact_conflict
otherwise. Read the published file before improving on it, and leave a correct one alone."""

POPULATION_HISTORY_PROMPT = """`h = await rlm.population.history("agent-3", start=0, limit=20, query=None)` reads a page
of another agent's conversation (with its task and last answer) without waking it, also
when it is retired."""

HISTORY_PROMPT = """## Conversation history
`from rlm import history; h = await history()` reads a snapshot of your ledger.
`h.messages[i]` addresses a session-wide message; `h.windows[w].messages[i]` addresses
one within a context window. Indices are zero-based. Messages are dictionaries with
role/content/tool fields. `h.user_messages()` returns original user inputs, distinct
from generated summaries and supervisor notices. `history(session_dir=path)` reads an
explicit session; use `await child.history()` when available. Reload for fresh state.

Compaction and rollback start new windows; earlier records remain addressable. Full tool
outputs and shortened context versions have separate indices. `h.events` contains spawn
and rollback records; prompt_rollback.prompt_id identifies a rolled-back user attempt.
History records what happened, not proof that side effects were undone. Recover exact
instructions and evidence by searching/selectively printing records, not the entire ledger.
"""


def build_system_prompt(
    cwd: str,
    installed_skills: list[str],
    *,
    depth: int = 0,
    session_dir: str | None = None,
    allow_recursion: bool,
    delegation_prompt: bool = False,
    allow_git: bool,
    active_tools: list[BuiltinTool],
    shell_skills: list[str] | None = None,
    skill_docs: list["SkillDoc"] | None = None,
    task_instructions: str | None = None,
    extra_instructions: str | None = None,
    agent_info: dict | None = None,
    mcp_servers: list[dict] | None = None,
) -> str:
    """Compose task instructions with the guide for this agent's actual runtime."""
    has_ipython = _has_tool(active_tools, "ipython")
    population = agent_info.get("population") if agent_info else None
    can_delegate = has_ipython and allow_recursion and population is None
    parts = [
        task_instructions
        if task_instructions is not None
        else "You are an agent. Complete the user's task using the available tools."
    ]
    if extra_instructions:
        parts.append(extra_instructions)
    if population:
        parts.append("## Episode task\n" + population["task"])
    if has_ipython and any(server["tool_count"] > 0 for server in mcp_servers or []):
        parts.append(MCP_DISCOVERY_PROMPT)
        parts.append("MCP servers: " + json.dumps(mcp_servers))
    if can_delegate and delegation_prompt:
        parts.append(DELEGATION_PROMPT)
    parts.append("## Agent context")
    if population:
        parts.append(
            "Supervisor identity: "
            + json.dumps(
                {
                    "id": agent_info["id"],
                    "name": agent_info["name"],
                    "parent": population["parent_name"],
                    "founder": population["founder"],
                }
            )
        )
    elif agent_info:
        parts.append(
            "Supervisor identity: "
            + json.dumps(
                {
                    key: agent_info[key]
                    for key in ("id", "parent_id", "name", "persistent")
                }
            )
        )
    if population:
        pass  # The population guide below describes this agent's role.
    elif depth > 0:
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
                POPULATION_RUNTIME_PROMPT if population else RUNTIME_PROMPT,
                HISTORY_PROMPT,
                IPYTHON_CONTROL_PROMPT,
                KERNEL_PACKAGES_PROMPT,
            ]
        )
        if population:
            parts.append(_population_prompt(agent_info["name"], population))
        elif can_delegate:
            parts.append(AGENT_PROMPT)
        else:
            parts.append(
                "Delegation is disabled at this depth. Work directly with your available tools."
            )
        if population:
            pass
        elif depth > 0:
            parts.append(
                "Your final answer is your deliverable: it reaches your parent automatically as an `agent.completed` event and through `result()`. You finish by replying without calling any tool; that reply is your final answer, so put the complete report there. Do not send the final report with `await rlm.agent.send_to_parent(message)`; use it only for interim findings, blockers or questions while you are still working. It returns an event ID and your parent chooses when to read it. Parent instructions are pushed automatically: queued input at an answer/wait boundary, steering at the next model/tool boundary. You cannot steer your parent or message siblings. If you have children, their reports enter your own pull-based inbox in the same way."
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
    if skill_docs:
        parts.append(_skill_docs_prompt(skill_docs))
    shell_skill_set = set(shell_skills or [])
    if installed_skills and has_ipython:
        parts.append(
            "Python tools (pre-imported): "
            + ", ".join(f"`{name}`" for name in installed_skills)
            + ". Each is async; use help(tool) for its signature."
        )
        for name in installed_skills:
            if guidance := _builtin_skill_prompt(name, active_tools):
                parts.append(guidance)
    if shell_skill_set and (has_ipython or _has_tool(active_tools, "bash")):
        parts.append(
            "Shell-enabled Python tools: "
            + ", ".join(f"`{name}`" for name in sorted(shell_skill_set))
            + ". Discover CLI usage with `<tool> --help`. Other listed Python tools are IPython-only."
        )
    if _should_include_git_history_guard(active_tools, allow_git):
        parts.append(GIT_HISTORY_GUARD_PROMPT)
    return "\n\n".join(parts)


def _population_prompt(name: str, population: dict) -> str:
    origin = (
        "You are one of the founders admitted when the episode started."
        if population["founder"]
        else f"You were spawned by {population['parent_name']}, whose brief is your first message."
    )
    text = POPULATION_PROMPT.format(
        name=name,
        origin=origin,
        founders=population["founders"],
        max_live_agents=population["max_live_agents"],
        max_total_agents=population["max_total_agents"],
        idle_grace_seconds=population["idle_grace_seconds"],
    )
    if population["artifacts"]:
        text += "\n" + POPULATION_ARTIFACTS_PROMPT.format(
            artifacts=", ".join(f"`{path}`" for path in population["artifacts"]),
            drafts=population["drafts"],
        )
    if population["allow_agent_lookup_agent_history"]:
        text += "\n" + POPULATION_HISTORY_PROMPT
    return text


SKILLS_PROMPT = """## Skills
A skill is a set of local instructions to follow that is stored in a `SKILL.md` file. Below is the list of skills that can be used. Each entry includes a name, description, and file path so you can open the source for full instructions when using a specific skill.
### Available skills
{skills}
### How to use skills
- Trigger rules: If the user names a skill OR the task clearly matches a skill's description shown above, you must use that skill for that turn.
- How to use a skill: read its `SKILL.md` completely before taking task actions. Resolve relative paths in it against the directory that holds it. If it ships `scripts/`, prefer running them over retyping their code.
- Missing/blocked: If a skill's file can't be read, say so briefly and continue with the best fallback."""


def _skill_docs_prompt(skill_docs: list["SkillDoc"]) -> str:
    lines = "\n".join(
        f"- {doc.name}: {doc.description} (file: {doc.location})"
        if doc.description
        else f"- {doc.name}: (file: {doc.location})"
        for doc in skill_docs
    )
    return SKILLS_PROMPT.format(skills=lines)


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
