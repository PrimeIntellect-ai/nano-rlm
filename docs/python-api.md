# Python API

Inside the kernel, the `rlm` package is how an agent reaches beyond its own cell: it starts and steers sub-agents, runs shell jobs and services, watches for activity, and reads its own history. These are async calls to the [supervisor](architecture.md#components), so what they create outlives the cell, the variable holding it, and even a kernel crash.

| Module | For |
|---|---|
| [`rlm.agent`](#agents) | Spawning sub-agents and steering, waiting for and cancelling them |
| [`rlm.inbox`](#inbox-and-waiting) | Reports, completions and watch events, pulled when the agent chooses |
| [`rlm.shell`](#shell-jobs) | Bash commands, short or in the background |
| [`rlm.services`](#services) | Processes that must keep running after the agent finishes |
| [`rlm.watch`](#watches) | Notifications about an agent's steps, a job's output or changes under a path |
| [`rlm.history`](#history) | Reading this agent's or a child's conversation |
| [`rlm.hints`](#hints) | Muting the supervisor's one-line usage hints |
| `rlm.tools` | Searching and loading MCP tools ([Tools](tools.md#mcp-tools)) |

## Agents

Agents are available when recursion is allowed (`max_depth` above the agent's depth). A Python variable holds a handle; the supervisor owns the agent, so ending a cell or losing the variable does not stop it.

```python
researcher = await rlm.agent.spawn(task="Check authentication behavior", name="researcher")
other = await rlm.agent.spawn(task="Check login behavior", name="login")

agents = await rlm.agent.list()                  # direct children, finished ones included
info = await researcher.info()                   # a fresh metadata snapshot
h = await researcher.history()                   # a fresh conversation snapshot
researcher = await rlm.agent.get("researcher")   # recover a handle by name or ID
```

`spawn` returns once the child is registered, not when it finishes; a rejected spawn (depth, spawn or budget limits) raises before registration. Names are unique among siblings and stay reserved for the session. `list(recursive=True)` includes descendants, but only direct children can be retrieved as handles and controlled. Metadata includes the ID, name, parent, task, status, persistence, creation time, elapsed seconds, session directory and any failure.

**Results.**

```python
result = await researcher.result(yield_after=60)
if result.running:
    print("still working, answer so far:", result.answer)
else:
    print(result.answer)
```

`result()` waits up to `yield_after` seconds (default 300) and returns an `AgentResult` with `status`, `answer`, `usage`, `turns`, `session_dir` and `running`. `answer` is `None` while the agent works on its first answer; a persistent agent's latest answer stays available while it runs again. A failed or cancelled agent raises. `wait(timeout=30)` returns current metadata after an outcome or its timeout (0 to 300 seconds); it waits inside the cell, under the cell's own timeout. Neither cancels the agent.

**Status** is `starting` while waiting for capacity, `running`, `waiting` during a native wait, `idle` for a persistent agent that has answered, and `completed`, `failed` or `cancelled` once it has ended. Finished agents' metadata, results and history stay readable while the session lives.

**Lifetime.** An agent releases its kernel after answering. One spawned with `persistent=True` goes `idle` instead, keeping its conversation and kernel, and wakes on a parent instruction or a new inbox event. `await researcher.cancel()` ends an agent and its descendants and waits for cleanup; calling it again retries cleanup that did not finish, and `info.cleanup_error` reports a cleanup failure without hiding a completed result. A parent's end ends all its descendants, persistent ones included, and closing the ACP session ends the tree. Cancelling a single prompt or cell leaves its accepted children registered and recoverable. Idle persistent agents hold no inference slot but count toward `max_subagent_calls`.

**Messages.**

```python
await researcher.send("Also check logout")         # delivered after its answer or at a native wait
await researcher.steer("Focus on login first")     # delivered at its next model/tool boundary

# inside the child:
await rlm.agent.send_to_parent("Found a missing permission check")
```

Parent instructions enter the child's conversation; neither kind interrupts a running model call or tool. Both wake an idle persistent child, and sending to an agent that has ended raises. `send` and `steer` return acceptance IDs, not acknowledgements, and refuse when the tree's budget is spent. An accepted instruction that cannot be delivered (the budget ran out, or the child ended) is recorded as `instruction_failed` in the child's inbox journal and reported to the parent as an `agent.delivery_failed` event with `agent_id`, `message_id` and `reason`.

## Population episodes

A session configured with a [population policy](runtime-contract.md#population-episodes) runs as one episode of peer agents. Every peer is a depth-0 agent with the same authority; no agent leads.

**Spawning.** `await rlm.agent.spawn(brief, independent=True)` admits a new depth-0 peer: not beholden to you (no lifecycle events to you, no cascade when either of you ends), the spawn recorded for provenance and spend attribution, and you get a handle to observe it (`info()`, `history()`, `result()`) and message it like any peer. `await rlm.agent.spawn(brief)` (the default) spawns a beholden child one depth deeper instead: it reports to you, ends with you and counts against the tree limits; the population's own caps count only peers. At the live cap an admission retires the longest-idle peer past its grace period (conversation kept, kernel released); a direct message revives it when a slot is free.

**Messaging is inbox-plus-pull; nothing is pushed into a conversation.** Messages land in the recipient's `rlm.inbox`, one coalesced, time-signatured notice announces them at the recipient's next model boundary, and a periodic reminder repeats while peer messages stay unread. Check the inbox before ending a turn.

```python
receipt = await rlm.agent.dm("agent-2", "please compute 6*7")   # lands in agent-2's inbox
await rlm.agent.broadcast("midpoint check")                    # every live peer but you
print(await rlm.agent.delivery(receipt["message_id"]))
# {'message_id': ..., 'to': 'agent-2', 'kind': 'agent.dm', 'status': 'accepted', ...}
# status: accepted -> delivered (read) -> answered (answer = the reply's tail, pulled),
# or failed with a reason.
```

Do not wait on a wake for a reply: the answering peer DMs you back (into your inbox) or you read the receipt and `rlm.population.history`.

**Population, board and answer.**

```python
await rlm.population.status()                        # live, capacity, admissions, budgets
await rlm.population.list()                          # peers: name, status, spawned_by, task
await rlm.population.events()                        # membership ledger, in order

b = await rlm.board.read()                           # shared Markdown board
await rlm.board.append("agent-1: claiming the proof")  # or board.edit(old, new, expected_revision=...)

cur = await rlm.population.answer()                  # {"text": ..., "revision": ..., "author": ...}
await rlm.population.submit("42", expected_revision=cur["revision"])  # CAS, scored
```

Board edits and the episode answer are revision-checked compare-and-swap: a conflict means someone else wrote first — read again and reconcile, never overwrite. Declared final files are only replaced through `await rlm.artifact.publish(path, draft, expected_revision=...)`.

**Episode end.** The episode runs until no agent can act again (or a budget or the wall clock runs out), then every agent stops — jobs, subscriptions and services included — and the scored state is the submitted answer and/or the workspace as the task's runner sees it after the session closes.

## Inbox and waiting

Reports, completions and watch events arrive in the agent's inbox. The model sees the unread count whenever it changes and decides when to read:

```python
for event in await rlm.inbox.list():                 # metadata, unread only by default
    report = await rlm.inbox.read(event["id"])       # the payload, now marked read
    if report["type"] == "agent.message":
        print(report["content"]["name"], report["content"]["text"])
```

`list(unread_only=False)` includes events already read. A child's `agent.completed` event names the agent and its status; its answer comes from the handle's `result()`.

The native **`wait`** tool suspends inference without holding a cell, until a new inbox event, a parent instruction or its timeout (default 300 seconds). Events already announced do not wake it again. An agent with nothing else to do waits instead of polling; a final root answer instead returns control to the client.

Inboxes and instruction queues belong to the supervisor. Each agent's `inbox.jsonl` journals arrivals, deliveries and reads, and delivered messages and notifications also enter its `messages.jsonl`. Message payloads are limited to 65,536 JSON bytes. An inbox takes up to 10,000 events and a pending instruction queue as many; lifecycle events are always recorded.

## Shell jobs

`rlm.shell.run` runs a Bash command under the supervisor. A short command returns its result in the same cell:

```python
result = await rlm.shell.run("git status --short")
print(result.text, result.exit_code)
```

A long one keeps running in the background while the cell moves on:

```python
job = await rlm.shell.run("uv run pytest tests/", cwd="/workspace/project", yield_after=0)
# ... other work ...
res = await job.result()
print(res.exit_code, res.text)

a, b = await asyncio.gather(job_a.result(), job_b.result())   # several at once
job = await rlm.shell.get(job_id)                              # recover in a later cell
chunk = await job.read(cursor=0, max_bytes=16384)              # continue from chunk.next_cursor
```

`run(command, cwd=..., env=..., yield_after=10, timeout=None)` takes a Bash string or an argv list and returns a `ShellJob` snapshot: `id`, `running`, `ok`, `exit_code`, `text` (combined stdout and stderr, up to 16 KiB: the first and last 8 KiB around a marker when longer), `truncated`, `timed_out` and `error`.

- **Yielding.** `run` waits `yield_after` seconds (default 10, 0 returns at once, at most 300) before handing back a job that is still running, with `running=True` and its output so far. Yielding never stops the job. `await job.result(yield_after=...)` waits up to 300 seconds per call and returns the finished snapshot, or the job again if it is still running; it is repeatable and never consumes output.
- **Timeout and cancel.** `timeout` kills the job's process group after that many seconds and sets `timed_out`. `await job.cancel()` terminates the process group.
- **Output.** Each job keeps up to 16 MiB in `<session>/jobs/<id>/output.bin`, readable with `job.read(cursor, max_bytes)` (16 KiB by default, 64 KiB at most; cursors count bytes, text decodes as UTF-8 with replacement). A finished job's metadata is kept beside it.
- **Completion events.** A job handed back while still running publishes a quiet `shell.completed` event when it ends, with its ID, status, exit code, completeness flags and last 4 KiB of output. Quiet events wake `wait` but are not counted as unread, and `result()` marks them read.
- **Environment.** `await rlm.shell.setenv(NAME="value")` sets variables for every later `run` of this agent; `getenv()` reads them back, and a call's own `env=` wins. Jobs inherit the agent's process environment minus credential-looking names, URLs with embedded credentials, variables that would redirect rlm's interpreter or virtual environment, and agent sockets.
- **Execution.** Commands run in `/bin/bash --noprofile --norc -c`, with stdin closed and stdout and stderr combined. Relative directories resolve against the agent's working directory, and the Git guard applies. A nonzero exit code is a completed job; failing to start or capture is status `failed`, with `error` set. There is no PTY or stdin API, and IPython's `!` and `%%bash` do not go through `rlm.shell`.

Jobs survive the end of their cell, lost variables and kernel restarts; `rlm.shell.list()` finds them. A job ends with its owner, and when it ends the processes left in its process group are killed. A job counts as finished when stdout and stderr close: if descendants keep them open more than two seconds after Bash exits, capture stops with `output_complete=False`. Processes that leave the job's process group are not managed; anything that must keep running is a [service](#services). A session tree allows 32 running jobs and 1,024 in total.

## Services

A service is a process that must keep running after the agent finishes, such as a server or daemon a task asks to leave up.

```python
svc = await rlm.services.start("api", "python -m http.server 8000", cwd="/workspace/site")
print(await svc.tail())          # the last lines of its log
svc = await rlm.services.get("api")
await svc.stop(grace=5)          # SIGTERM to its process group, SIGKILL after the grace period
```

`start(name, command, cwd=..., env=...)` returns at once; check that the service came up with `tail()` or by connecting to it. The service runs in a session of its own, with stdin from `/dev/null` and stdout and stderr appended to its log, and only `stop()` ends it. Names are 1 to 64 letters, digits, `.`, `_` or `-`; starting a name that is still running raises. The environment and default working directory are those of `rlm.shell.run`. `get(name)` returns a fresh snapshot (`name`, `pid`, `command`, `cwd`, `log`, `running`, `exit_code`, `started_at`) and works after kernel restarts; `list()` returns every service started in this workspace, running or not.

## Watches

A watch turns future activity into inbox events, without keeping a cell running:

```python
activity = await rlm.watch.agent(researcher)       # its completed steps, final answers included
output = await rlm.watch.job(job)                   # newly captured output
files = await rlm.watch.path("/workspace/results", recursive=False)

subscriptions = await rlm.watch.list()
watch = await rlm.watch.get(subscriptions[0].id)
await watch.cancel()
```

Agent watches cover direct children; job watches cover the caller's own jobs; path watches cover an existing file or directory, hidden files included, with recursion opt-in. A child's or job's completion already notifies its owner without a watch.

Each event carries its `subscription_id`, and its `content` names the `target` (an agent ID, job ID or absolute path) plus:

| Event | References |
|---|---|
| `watch.agent` | `start` and exclusive `end` message indices: `(await researcher.history()).messages[start:end]` |
| `watch.job` | `start` and exclusive `end` byte cursors for `job.read(cursor=start)` |
| `watch.path` | The changed `paths` and a `truncated` flag |
| `watch.failed` | An `error` saying why the watch stopped |

Watches see only activity after registration; there is no replay. Agent and job ranges can merge several steps or chunks, and path changes are batched over 200 ms. A path watch returns once the backend is watching; removing the watched path ends it with `watch.failed`. Watches survive kernel restarts, end with their owner, and complete once their target has ended (flushing pending activity); a watch on an idle persistent agent stays active, and one on an already finished target returns completed. Cancelling drops any unpublished batch but keeps delivered events. Registrations and state changes are journaled in the owner's `inbox.jsonl`. A tree allows 64 active watches and 1,024 registrations, changed-path lists are capped at 32 KiB of JSON, and a watch stops with `watch.failed` when its owner's inbox is full.

## History

An agent can read its own conversation, or a child's, as it stands:

```python
from rlm import history

h = await history()                       # this agent's ledger
requests = h.user_messages()              # original inputs, rolled-back attempts included
message = h.messages[3]                   # by session-wide message index
earlier = h.windows[0].messages           # the first context window
child = await researcher.history()        # a child's, even while it runs
```

Each call returns a complete snapshot as of the read; call again to see new activity. `h.events` holds the lifecycle records too, such as each child's spawn prompt and rollback markers. `history(session_dir=...)` reads any session directory, also outside a session. [Context and history](context.md#the-ledger) describes the records.

## Hints

The supervisor attaches a one-line hint to the next turn when something worth knowing happened, for example a blocking `run()` that detached into a background job. Each hint names a tag; `await rlm.hints.mute("tag")` stops that hint for this agent, `unmute` restores it, and `muted()` lists the muted tags.
