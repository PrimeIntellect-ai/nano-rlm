# nano-rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents. For a full-fledged coding agent built on the same RLM principles, see [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent).

By default the model gets a single built-in tool, `ipython`: a persistent IPython kernel for Python, Bash commands via `rlm.shell.run`, and background jobs via `rlm.shell.start`. File edits, shell work, and orchestration all go through it. The runtime contract's `builtin_tools` list can select a different tool set (`bash`, `edit`, `fetch`, `ipython`) for native tool-calling runs.

For convenience, rlm ships built-in *skills* that can be enabled per session via the runtime contract's `skills` list (off by default): `edit` (single-occurrence string replacement), `search` (web search via Serper, needs `SERPER_API_KEY`), and `fetch` (retrieve a URL as cleaned text). Enabled skills are pre-imported into the IPython kernel like any other skill (see [Skills](#skills)), so the agent calls `await edit(path=..., old_str=..., new_str=...)`, `await search(query=...)`, or `await fetch(url=...)`. `fetch` also exists as a native builtin tool with the same semantics, for tool-calling runs (opt-in via the contract's `builtin_tools`).

Context compaction is on by default: the engine compacts when 16k tokens remain below an advertised model context window. Termination comes from the default tree-wide budget of 1M new tokens (`max_total_tokens`). The policy can set an explicit `summarize_at_tokens` threshold. The IPython kernel keeps running across compaction, so REPL state survives (see [Compaction](#compaction)).

Inside IPython, the `rlm` package is available in the namespace. When recursion is allowed, `await rlm.agent.spawn(task=...)` returns a handle to a supervisor-owned agent that continues across cells. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

## Install

```bash
git clone https://github.com/PrimeIntellect-ai/nano-rlm.git
cd nano-rlm
uv sync
source .venv/bin/activate
```

## CLI

rlm runs exclusively as an [Agent Client Protocol](https://agentclientprotocol.com/) agent:

```bash
rlm --acp
```

There is no standalone prompt mode: every session is created by an ACP client that
supplies the full runtime configuration (see below). Skill CLIs provided by the host
environment are on `$PATH` inside the kernel (e.g. `websearch --queries "..."` when
the `websearch` skill is installed).

## Agent Client Protocol

Each ACP session owns one persistent RLM engine. Repeated `session/prompt`
requests retain both the model conversation and the live IPython kernel. The
agent accepts text prompts and stdio or streamable HTTP MCP servers, including
HTTP request headers. It supports cancellation and `session/close`; cancelling
a turn leaves its session available for the next prompt. It intentionally does
not advertise `session/load`: an arbitrary live Python kernel cannot be
reconstructed after the ACP process exits, so clients must keep the process
alive for the lifetime of a session.

RLM's ACP surface is a versioned training contract. `initialize` advertises the exact
`ai.prime.rlm/contract-v1` marker in its response `_meta`; clients must require
it, then provide one complete `ai.prime.rlm/runtime-v1` object in
`session/new._meta`. The runtime object contains the ACP session ID, model,
provider, execution policy, prompt configuration, enabled built-in skills,
optional builtin tool selection,
explicit kernel environment, and optional search credential. Nullable and
disabled values are sent explicitly as `null` or empty collections. Missing,
partial, unknown, or unsupported contracts are rejected; ACP sessions never
fall back to process environment configuration.

Credentials travel over the private ACP stdio channel and are never echoed.
The `session/close` response carries one authoritative, credential-free
snapshot of cumulative usage, metrics, tool-call stats, supervisor counters,
and limits under `ai.prime.rlm/session-v1`.

Every actual model call carries a standard HTTP `Idempotency-Key` header that
stays stable across SDK and outer retries (retry attempts are distinguished by
`x-stainless-retry-count`), so an inference proxy can deduplicate replayed
requests. Both header names are reserved and rejected in provider
configuration.

Model calls also carry a private `X-ACP-Model-Request-ID` correlation header.
RLM publishes sparse, labeled relationships between those request IDs under
`ai.prime.acp/semantic-edges-v1`: `continuation`, `subagent_call`,
`subagent_return`, and `compaction`. `continuation` preserves same-agent causal
order even when a consumer's physical token-prefix graph splits. ACP consumers
can resolve the request IDs onto their own message nodes while harnesses that do
not understand the extension ignore it.

## Python API (inside a session)

`rlm.agent` exposes `spawn`, `list`, and `get` inside a running session. These async calls contact the session supervisor; spawning waits only for registration, not task completion. `history()` also supports explicit session-directory loading outside a running session.

## Configuration

All runtime configuration enters through the `ai.prime.rlm/runtime-v1` contract object
(model, provider credentials, execution policy, prompt configuration, skills, builtin tools, kernel
environment, search credential). Prompt configuration is role-aware: optional
`subagent_append_to_system_prompt` (nodes that can still recurse) and
`leaf_append_to_system_prompt` (depth == max_depth) override `append_to_system_prompt`
for sub-agents, each falling back to the next-more-general tier. Recursive children inherit the parent's configuration
in-memory (`model_copy`); nothing is re-read from the process environment.

`system_prompt_path` supplies task instructions in place of the default task role.
The runtime guide is always appended, including when a custom prompt file is used.
The role-appropriate append instructions are included between the task instructions
and runtime guide. This keeps tool/API documentation and lifecycle rules available
to root agents, persistent children, and leaves. Credentials are not included in
agent identity metadata.

The generated guide distinguishes Python state from supervisor-owned resources,
shows inbox dictionaries versus handle/metadata objects, and explains waiting,
completion, history recovery, jobs, and subscriptions. Delegation instructions are
shown only when IPython and recursion are available. Tool descriptions follow the
same shell execution guidance.

Checkpoint prompts preserve task requirements, evidence, outstanding assignments,
jobs, subscriptions, event actions, output cursors, and history references. Commands
and edits are included only when relevant. Compaction thresholds, summary validation,
and context retention policy are unchanged.

The process environment configures only process infrastructure:

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |

## Recursion

The supervisor owns each agent's identity, task, runtime, and lifetime. Python variables hold handles, so losing a variable or ending a cell does not stop its agent.

```python
researcher = await rlm.agent.spawn(
    task="Check authentication behavior", name="researcher"
)
other = await rlm.agent.spawn(task="Check login behavior", name="login")

agents = await rlm.agent.list()          # Direct children, including completed agents
info = await researcher.info()          # Fresh metadata snapshot
h = researcher.history()                # Fresh conversation snapshot
researcher = await rlm.agent.get("researcher")  # Recover by sibling name or ID
```

Names are unique among siblings and remain reserved for the session, including after completion. Immutable IDs are shown alongside names. Metadata includes the parent ID, initial task, status, persistence flag, creation time, elapsed lifetime in seconds, session directory, and any failure. `list(recursive=True)` includes descendants, but only direct children can be retrieved as handles or controlled through the supervisor.

```python
status = await researcher.wait(timeout=30)
result = await researcher.result()
if result is not None:
    print(result.answer)
```

`result()` returns the latest successful `RLMResult`, including while a persistent agent is running again, or `None` before its first answer. Terminal failure or cancellation raises. Use `info()` or `wait()` for current activity; a retained answer does not mean a follow-up has finished. `wait()` returns current metadata after an outcome or its timeout (default 30 seconds, range 0–300). It waits inside the Python cell and uses the cell's normal execution timeout. Cancelling or timing out a wait does not cancel the agent.

`await researcher.cancel()` terminates the agent and its descendants and waits for cleanup. Ordinary agents release their kernels after answering. An agent spawned with `persistent=True` becomes idle after answering and retains its conversation and kernel; a parent instruction or new inbox event wakes it. Parent termination tears down all descendants, including persistent agents. Closing the ACP session tears down the tree. Cancelling an individual prompt or cell leaves its accepted children registered and recoverable.

Status is `starting` while waiting for capacity, `running` during execution, `waiting` during a native event wait, `idle` for a persistent agent that has answered, or `completed`, `failed`, or `cancelled` after termination. Completed metadata, results, and history remain accessible for the supervisor's lifetime. Shared filesystem access is trusted; handle permissions are orchestration controls, not filesystem isolation.

Depth, concurrency, total-spawn, and shared token/turn limits remain supervisor-enforced. Rejected spawns raise before registration. Persistent idle kernels release inference capacity but still count toward the session's total-spawn limit. A child return enters the ACP semantic trace when the parent retrieves its result, so background completion is not mistaken for result consumption.

### Messages and event waiting

```python
await researcher.send("Also check logout")   # Deliver after its answer or native wait
await researcher.steer("Focus on login first")  # Deliver at the next model/tool boundary

# Inside the child:
await rlm.agent.send_to_parent("Found a missing permission check")

# Inside its parent:
for event in await rlm.inbox.list():
    report = await rlm.inbox.read(event["id"])
    print(report["type"], report["content"])
```

Parent instructions are pushed into the child's conversation. Steering does not interrupt a running model request or tool. Queued messages wait until the child answers or calls the native `wait` tool. Both operations wake an idle persistent child; sending to a terminated child raises.

Reports and `agent.completed` events enter the parent's inbox at every depth. The model sees an unread count before each inference step and chooses when to retrieve payloads. `list()` returns metadata without marking events read; `read(id)` returns the payload and marks it read. `list(unread_only=False)` includes previously read events. Completion payloads identify the agent and status; retrieve the answer with its handle's `result()`.

ACP dependency edges use `agent_message` for delivered instructions and explicitly read inbox events. Spawning uses `subagent_call`; retrieving an answer uses `subagent_return`. Delivery policy and event type remain in the event records. An unread-count notification alone creates no `agent_message` edge.

The native `wait` tool suspends inference without occupying an IPython cell, until a new inbox arrival, parent instruction, or timeout (default 300 seconds, range 0–300). It also yields to queued instructions. Already-announced unread events do not repeatedly wake it. A final root answer returns control to the ACP caller; use `wait` to keep the current prompt available for events.

The supervisor owns inboxes and instruction queues. Each agent's `inbox.jsonl` records arrivals, instruction delivery, and explicit reads; delivered messages and notifications also enter `messages.jsonl`. Message payloads are limited to 65,536 JSON-encoded UTF-8 bytes. Reports are rejected once an inbox contains 10,000 events, and pending instruction queues have the same limit; lifecycle events remain recordable. This state lasts while the supervisor lives. Restarting a crashed IPython kernel and restoring a crashed supervisor are separate features.

## Compaction

There is no model-driven compaction tool. Compaction is on by default and unlimited (the policy's `compaction` field turns it off, `max_compactions` caps it); the default 1M `max_total_tokens` tree budget keeps sessions bounded, since each compaction cycle itself spends new tokens. The engine reads the model context window from the provider's `/models` response and compacts when 16k tokens remain below it; small windows keep at least half. Set `summarize_at_tokens` to pin the threshold explicitly. Without a known window or explicit threshold, proactive compaction stays off, but a provider overflow still triggers it reactively. A tool result larger than 20KB is truncated to its head and tail before it enters the conversation, with a warning naming the original size.

The engine asks the model for a plain-text handoff summary and resumes the task on a fresh branch seeded with that summary; reasoning is never part of it. A provider overflow (a 400 or 413 naming a context limit) triggers the same compaction reactively from the current state. A rejected checkpoint request falls back to the last state that passed a threshold check - by definition a state with a full reserve of room - and an empty or tool-calling reply is resampled; after three failed attempts the run ends cleanly with what the conversation holds. An overflow with no history beyond the task propagates: the task alone approaches the window and there is nothing to reclaim.

The IPython kernel keeps running across the compaction, so all variables, imports, and in-memory data are preserved. The model is told to mention important variable names in its summary so the resumed branch knows what is available. The same policy applies to the main agent and all recursive agents.

A session creates a fresh ledger and refuses to reopen an existing `messages.jsonl` for writing. Existing histories remain readable. The session owns active messages and their indices; the engine uses context snapshots. A ledger write or flush failure makes the writer unusable and prevents further prompts. Rollback restores in-memory context and counters even if its ledger write fails. Supervisor crash recovery is not supported.

The working conversation remains in the session's append-only `messages.jsonl`, which the model can search and process from Python. Message records have stable event `id` fields, zero-based `message_index` values, and a `message` object containing the role, content, and tool-call fields. Full tool outputs and shortened context versions have separate indices; the shortened tool record links to its `source_message_index`. System messages and installed summaries are also logged. Checkpoint prompts and raw checkpoint responses are omitted: this ledger serves agent context recovery. Verifiers, the ACP consumer, records model requests/responses through its interception endpoint and receives semantic links and metrics over ACP, independently of this file. A `user` record begins a prompt attempt; `prompt_rollback.prompt_id` identifies a failed or cancelled attempt, whose records remain available as history.

Each `context_window` record declares a zero-based `window` index, a `reason` (`start`, `compaction`, or `rollback`), and the ordered `message_indices` that seed that window. Subsequent message records with a `window` field append to it. Closed windows never change. Rollback opens a new window containing the restored context, preserving addresses in the failed window. The `turn` field is an execution-loop counter, not a message or window index.

Each agent writes its own `messages.jsonl` in its session directory. Message and context-window indices are local to that agent. Agent discovery belongs to the supervisor; the history reader accepts an explicit session directory.

The kernel can inspect its own history or a child's, including while the child is running:

```python
from rlm import history

h = history()                        # Defaults to $RLM_SESSION_DIR
requests = h.user_messages()          # Original inputs, including rolled-back attempts
message = h.messages[3]               # Session-wide message index
earlier = h.windows[0].messages       # Initial working context
child_history = history(session_dir="/path/to/child-session")
message = child_history.windows[4].messages[2]  # If that child has reached window 4
```

Snapshots contain complete records as of the read; call `history(...)` again to observe new activity. `h.events` exposes lifecycle records, including each child's spawn prompt and rollback markers. The compacted context points to this API so the model can retrieve omitted details without putting the whole transcript back in context.

## Session Directory

Every invocation writes to `$RLM_HOME/sessions/<id>/`. Nested session directories mirror the call tree.

```text
.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── sub-d4e5/
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/
└── sub-h8i9/
```

These artifacts are consumable for debugging, visualization, or training-data extraction.

## Skills

`rlm` ships a small set of built-in skills enabled per session via the runtime contract's `skills` list (`edit`, `search`, `fetch`; see [MCP tools as skills](#mcp-tools-as-skills) for the related MCP path). `edit` runs in the kernel. Credentialed `search` runs in the supervisor through the capability broker, so `SERPER_API_KEY` is unavailable to IPython and its subprocesses; it returns title/URL/snippet for a single query (`await search(query="...")`). Additional skills are supplied by the host environment: before `install.sh` runs, the environment places skill packages under `/task/rlm-skills/<name>/`, and `install.sh` installs them alongside `rlm` so they're both importable and on `$PATH`.

From IPython, import a skill and call its async `run(...)` entrypoint:

```python
import websearch

help(websearch)  # signature + docstring
results = await websearch(queries=["latest jupyter_client release"])
```

Uploaded `rlm-skill-*` packages also expose their declared console command. Session-generated built-in and MCP proxy skills are IPython-only:

```bash
websearch --queries "latest jupyter_client release"
```

### Skill contract

A skill is a normal Python package laid out like this:

```text
<name>/
├── SKILL.md
├── pyproject.toml
└── src/
    └── <name>/
        ├── __init__.py
        └── <name>.py
```

Required public surface:

- async `run(...)`: the single entrypoint. Type-annotated parameters and a Google-style docstring (`Args:`, `Returns:`) drive both the Python API and the CLI.

The `pyproject.toml` points its console script at the shared CLI entry:

```toml
[project.scripts]
<name> = "rlm.skill:cli"
```

`rlm.skill:cli` reads `sys.argv[0]`, imports the matching module, and uses [tyro](https://brentyi.github.io/tyro/) to build the argparse CLI from `run`'s signature. The return value of `run` is printed; a raise surfaces as a normal Python traceback.

Naming expectations (all match):

- skill directory name: `<name>`
- distribution name in `pyproject.toml`: `rlm-skill-<name>`
- import name: `<name>`
- console script name: `<name>`

Keyword arguments on `run(...)` and CLI flags line up automatically — `queries: list[str]` becomes `--queries` on the CLI.

Dependencies go in the skill's own `pyproject.toml`; declare `rlm` there so the shared CLI entry resolves. Version conflicts between skills installed side-by-side are the user's responsibility.

### Local development

For running `rlm` against a specific skill set outside of a sandbox-orchestrated environment, create a `/task/rlm-skills/` directory (or bind-mount one) and place skill packages there before running `install.sh`. The rlm repo ships no skills by default; look at the `rlm-swe` or `rlm-deepdive` environments for working skill packages to copy.

### MCP tools as skills

A host harness can wire task-specific [MCP](https://modelcontextprotocol.io) tool servers to `rlm` through the ACP session (a standard `mcpServers` config shape). Streamable HTTP and stdio transports are supported:

```json
{
  "mcpServers": {
    "remote": {"url": "http://127.0.0.1:8000/mcp"},
    "local": {
      "command": "/path/to/tool-server",
      "args": ["--stdio"],
      "env": {"API_KEY": "..."}
    }
  }
}
```

Programmatically, pass `mcp_servers={"tools": "http://127.0.0.1:8000/mcp"}` to `RLMEngine`. An HTTP server may instead be `{"url": "...", "headers": {"Authorization": "..."}}` when it needs request headers; stdio servers use the same `command` / `args` / `env` shape shown above.

At startup `rlm` connects to each server, lists its tools, and generates one skill per tool (named `<server>_<tool>`, e.g. `tools_add_event`). These join the installed skills — pre-imported into the IPython namespace as async functions the agent calls programmatically, with a signature built from the tool's input schema:

```python
help(tools_add_event)  # signature (typed from the schema) + the tool's description
await tools_add_event(day="monday", title="standup")
```

Each call connects using the configured transport, invokes the tool, and returns its text content (a tool-reported error is raised as `RuntimeError`). Discovery and invocation run in the session supervisor. The generated modules contain only the public tool schema and an opaque capability; server URLs, headers, commands, and environment variables are not copied into the kernel environment or session artifacts. The kernel imports these proxy modules from the session directory and reaches the supervisor over the session-local broker. Unlike installed skills, MCP skills are IPython-only — they're not exposed as shell commands.

## Kernel

The IPython kernel always runs in rlm's own Python (`sys.executable`). `install.sh` puts `rlm` and all discovered skills into the same `uv tool install` environment, so `from rlm import run`, `import edit`, etc. work natively from inside an IPython cell.

The kernel starts from a small platform environment (`PATH`, home/user/shell, locale, temporary-directory, certificate, and virtual-environment variables) plus the contract's explicit `kernel_env` mapping. It receives private Jupyter/IPython config directories and does not inherit the rest of the supervisor process environment. This de-ambients credentials; it is not hostile-code containment because the kernel still shares the sandbox user, filesystem, process namespace, and network with the supervisor.

To exercise packages from the target project's `.venv` (e.g. running its test suite), shell out from an IPython cell: `!./.venv/bin/python3 -m pytest`. The kernel itself stays isolated from whatever project venv the agent is working on — no cross-cell state involving sandbox packages.

## Developing

After `uv sync`, enable the pre-commit hooks:

```bash
uv run pre-commit install
```

Lint and format manually:

```bash
uv run ruff check --fix .
uv run ruff format .
```

## Testing

Install dev dependencies and run the suite:

```bash
uv sync --group dev
uv run pytest tests/
```

Agent execution status and cleanup are separate: `info.cleanup_error` reports failed resource cleanup without hiding a completed result. Calling `cancel()` again retries unfinished cleanup.

`send()` and `steer()` return acceptance IDs, not processing acknowledgements. They reject instructions when the tree budget is already exhausted. If an accepted instruction cannot be delivered because the budget runs out or the child terminates, the supervisor records `instruction_failed` in the child's inbox journal and sends the parent an `agent.delivery_failed` event containing `agent_id`, `message_id`, and `reason`. Failed instructions are removed from the pending queue.

### Bash commands and background jobs

For quick commands, wait for the result in the current cell:

```python
result = await rlm.shell.run("git status --short")
print(result.text, result.exit_code)
```

`run(command, cwd=..., timeout=..., env=...)` returns `ok`, combined stdout/stderr `text` (up to
16 KiB: the first and last 8 KiB around an omitted-range marker when longer), `exit_code`,
`job_id`, `truncated`, `error`, and `timed_out`. The command is a Bash string or an argv
list. Nonzero exit codes are returned; startup/capture errors populate `error`; an exceeded
`timeout` (seconds) kills the process group and sets `timed_out`; a `run()` still going after 60 s
returns early with `running=True` and partial text while the job continues in the background. If output is truncated,
recover the job with `await rlm.shell.get(result.job_id)` to read more or inspect its
metadata. Cancelling the waiting cell leaves the job running and discoverable with
`shell.list()`. Both calls run Bash under the supervisor; only `start()` publishes a
`shell.completed` inbox event. `await rlm.shell.setenv(NAME='value')` sets variables for every later `run()`/`start()` of the
agent (`getenv()` reads the overlay; per-call `env=` wins). The kernel and Bash jobs inherit the launching process's environment
(in a sandbox, the image's ENV) minus a blocklist: credential-looking names, values with
URL-embedded credentials, variables that would redirect the kernel's own interpreter or
venv (PYTHONHOME, UV_PYTHON, ...), and agent/daemon sockets.

For long commands or work that should run alongside other tasks, use `start()`:

```python
job = await rlm.shell.start("uv run pytest tests/", cwd="/workspace/project")
job_id = job.id
await job.info()

# In a later cell, recover the same job and inspect its output.
job = await rlm.shell.get(job_id)
chunk = await job.read(cursor=0, max_bytes=16384)
print(chunk.text)
# Continue from chunk.next_cursor; reading is repeatable.
await rlm.shell.list()
```

Use the native `wait` tool when there is no other work. Termination produces a
`shell.completed` inbox event with the job ID, status, exit code, and output
completeness flags. Read it with `rlm.inbox.read(event_id)`. A nonzero Bash exit
code is a completed process; startup/capture errors have status `failed`.

`await job.cancel()` terminates its process group. Jobs survive cell completion
and lost handle variables; owner termination cancels them. Commands run in
`/bin/bash --noprofile --norc -c`, with closed stdin and combined stdout/stderr.
Relative working directories resolve against the agent's working directory.
The task environment and Git-history policy also apply to these commands.

The supervisor permits 32 active jobs and 1,024 total jobs per session tree.
Each job retains up to 16 MiB in `<session>/jobs/<id>/output.bin`; excess output
is drained and discarded. Reads default to 16 KiB and allow at most 64 KiB.
Cursors count bytes, and text decodes as UTF-8 with replacement. Terminal metadata
is saved alongside output in `meta.json`. Output files remain after job cleanup.

Normal completion waits for stdout/stderr EOF. If descendants keep the pipe open
more than two seconds after Bash exits, capture stops and reports
`output_complete=False` and `output_truncated=True`. Remaining processes in the
job's process group are killed when the job ends. Keep Bash alive until its work
finishes; detached processes that escape the process group are not managed.

There is no PTY, stdin API, or redirection of IPython's `!` / `%%bash` magics.

### IPython crash recovery

If the kernel exits during a cell, the harness starts a fresh kernel under the
same agent identity and resumes the conversation with a recovery notice. A dead
kernel discovered before a cell starts is also restarted; that cell is returned
unexecuted so the model can first reconstruct its Python state.

The conversation, history, child agents, shell jobs, and inbox (including read
state) remain owned by the live supervisor. Recover handles with
`rlm.agent.list/get` and `rlm.shell.list/get`. Python variables, imports, and
in-kernel tasks are lost. The interrupted cell is never replayed automatically:
its file writes or accepted spawn/send/job requests may already have happened.
Inspect those resources before retrying an operation.

Recovery notices are separate `kernel_recovery` history records and active
conversation messages, so truncating tool output cannot hide them. A kernel
restart required after an unresponsive timeout/interrupt produces the same notice.
A successful interrupt that preserves the kernel does not claim variables were lost.

There are at most three recovery attempts without a subsequently completed cell.
A failed startup reports failure; a later IPython call may retry within that
budget. Exhausting the budget leaves IPython unavailable, while the agent can
still respond or use native waiting. Supervisor crash recovery is not supported.
Kernel death is detected during execution or at the next IPython call; this does
not introduce a background kernel-health subscription.

### Subscriptions

Watch future activity without keeping a Python cell running:

```python
activity = await rlm.watch.agent(researcher)
output = await rlm.watch.job(job)
files = await rlm.watch.path("/workspace/results", recursive=False)

subscriptions = await rlm.watch.list()
watch = await rlm.watch.get(subscriptions[0].id)
await watch.cancel()
```

Agent watches observe direct children after complete assistant/tool steps,
including final answers. Job watches observe newly captured stdout/stderr from
jobs owned by the caller. Path watches observe existing files or directories;
relative paths resolve against the agent's working directory. Hidden files are
included, and directory recursion is opt-in. Child/job completion already
notifies the owner automatically and needs no subscription.

Events enter the same pull-based inbox and carry `subscription_id`. Their
`content` contains a `target` (agent ID, job ID, or absolute path) plus:

| Event | References |
| --- | --- |
| `watch.agent` | `start` and exclusive `end` message indices: `researcher.history().messages[start:end]` |
| `watch.job` | `start` and exclusive `end` byte cursors for `job.read(cursor=start)` |
| `watch.path` | Changed `paths` and a `truncated` flag |
| `watch.failed` | An `error` explaining why the subscription stopped |

Subscriptions start with future activity, not historical replay. Agent and job
ranges can be coalesced across several steps/chunks; changes are batched over
200 ms. Agent ranges exclude records written before registration. Filesystem
registration waits until the backend is watching before returning the handle.
Observed removal of the watched path ends the subscription with a failure event;
register a new watch after recreating it.

Handles and subscriptions survive kernel restart. Cancellation stops future
events and drops any unpublished batch; existing inbox events remain readable.
Owner termination cancels its subscriptions. Registration and state transitions
are journaled in the owner's `inbox.jsonl`; supervisor restart recovery remains
out of scope.

The tree permits 64 active subscriptions and 1,024 total registrations. Changed
path lists are capped at 32 KiB of JSON with explicit truncation. A subscription
stops with a failure event when the inbox event limit is reached. There are no
model-authored callbacks, event selectors, or output predicates.

Activity subscriptions become `completed` after their target permanently terminates, flushing pending activity and releasing their active slot. Watches of idle persistent agents remain active. Watching an already-finished target returns a completed subscription.
