# Runtime contract

nano-rlm is configured by its client, once per session, through one versioned object. There is no config file and no environment fallback: a training run gets exactly the agent it asked for, and the response tells it exactly what that agent spent.

`rlm --acp` is an [Agent Client Protocol](https://agentclientprotocol.com/) agent over stdio. There is no standalone prompt mode.

## Handshake

1. `initialize` returns `ai.prime.rlm/contract-v1: true` in its response `_meta`. Clients must require it. The same `_meta` advertises steering (`steering.supported`) and split execution (`ai.prime.rlm/execution-v1`, with its transports and tools).
2. `session/new` carries one complete `ai.prime.rlm/runtime-v1` object in its `_meta`. Nullable and disabled values are sent explicitly as `null` or empty collections. Missing, partial, unknown or unsupported contracts are rejected.
3. `session/prompt` runs a turn. Repeated prompts continue the same conversation and the same live kernel. Cancelling a turn leaves the session available for the next prompt.
4. `session/close` tears down the agent tree and returns the final [session snapshot](#session-snapshot).

The agent accepts text prompts, and stdio or streamable-HTTP MCP servers (with HTTP headers) in `session/new`. It does not advertise `session/load`: a live Python kernel cannot be rebuilt after the process exits, so a client keeps the process alive for the session's lifetime.

## The runtime object

| Field | Type | Meaning |
| --- | --- | --- |
| `session_id` | string | The client's ID for the session, echoed in snapshots (`[A-Za-z0-9._:-]`, at most 128 characters) |
| `model` | string | Model name sent to the provider |
| `provider` | object | `base_url`, `api_key`, `headers`, `max_retries` (default 5). `Idempotency-Key`, `x-stainless-retry-count` and the ACP tracing headers are reserved |
| `policy` | object | Budgets, recursion, compaction and timeouts, see [Policy](#policy). Omitted fields take their defaults |
| `system_prompt_path` | string or null | A file whose instructions replace the default task role |
| `append_to_system_prompt` | string or null | Instructions appended for every agent |
| `subagent_append_to_system_prompt` | string or null | Override for sub-agents that can still recurse |
| `leaf_append_to_system_prompt` | string or null | Override for agents at `max_depth` |
| `skills` | list | Built-in Python tools to pre-import: `bash`, `edit`, `fetch`, `search` ([Tools](tools.md#built-in-python-tools)) |
| `builtin_tools` | list or null | Native tools; null means `["ipython"]` ([Tools](tools.md#native-tools)) |
| `kernel_env` | object | Variables added to the kernel's minimal environment |
| `search_api_key` | string or null | Serper key for `search`, held by the supervisor |
| `network_block` | list | Domains, wildcard hosts or URL origins that `search` results must not point to |
| `execution_command` | list or null | A trusted argv starting `python -m rlm.execution`, for [split execution](split-execution.md) over stdio |
| `execution_cwd` | string or null | The tools' working directory in split execution |
| `execution_tcp` | object or null | `host`, `port` and `token` of a worker for split execution over TCP |

Recursive children inherit their parent's configuration in memory, one level deeper; nothing is re-read.

The only process environment variable is `RLM_HOME` (default `~/.rlm`): the root for session directories and host-supplied tool packages (`$RLM_HOME/skills`).

## Policy

| Field | Default | Meaning |
| --- | --- | --- |
| `max_total_tokens` | 1,000,000 | New tokens across the whole tree: each call's completion plus uncached prompt. Once reached, every engine stops before its next model call and no more agents are spawned. The default terminator |
| `max_total_turns` | null | Model calls across the tree, compaction excluded |
| `max_depth` | 1 | How deep agents can recurse; 0 disables `rlm.agent.spawn` |
| `max_concurrent_subagents` | 4 | Concurrent model calls across the tree, root and compaction calls included |
| `max_subagent_calls` | null | Total spawns across the tree |
| `delegation_prompt` | false | Add guidance on when and how to delegate, for agents that can |
| `exec_timeout` | 300 | Seconds a tool call may run; an `ipython` call may ask for up to 600 |
| `max_tokens` | null | Completion tokens one agent may spend; reaching it ends the turn |
| `max_tool_output_bytes` | null (20 KB) | How much of one tool result enters the conversation |
| `compaction` | true | Compact the context when it runs out of room, and recover from provider overflows |
| `summarize_at_tokens` | null | Compaction threshold; null derives it from the provider's advertised context window |
| `summary_budget_tokens` | 24,576 | Room a summary reply gets, reasoning included, capped at half a known window; a summary request that would leave less is trimmed first |
| `max_compactions` | null | Compactions per session |
| `max_compaction_attempts` | 5 | Summary attempts within one compaction |
| `allow_git` | false | Lift the guard that refuses Git commands reaching beyond the current branch (other refs, branch and tag listings, `clone`, `fetch`, `pull`) |
| `population` | null | Run the session as one [population episode](#population-episodes) of peer agents |

**Model-call admission.** `max_concurrent_subagents` bounds calls across the tree. Queued calls closer to the root go first, FIFO within a depth, and calls in flight are not preempted, so continuous shallow work can delay deeper agents. Tools, child waits and the backoff between outer retries hold no slot; the SDK's own retries stay within one. The limit is per tree: a cap across trees belongs in the runner or inference gateway.

**Model transport.** All model calls, including compaction, request Chat Completions streaming with usage reporting. nano-rlm assembles the complete reply before using it; tools never execute from partial deltas. Providers must support SSE streaming. The inference slot remains held until the stream is consumed and closed. Interrupted streams and transient stream errors retry the whole call with the same request ID; partial replies do not enter the conversation or token totals. Streaming keeps an active generation from leaving the HTTP connection idle, but does not bypass absolute request-duration limits or stalls before the first byte.

## Prompts

`system_prompt_path` supplies task instructions in place of the default task role. The runtime guide is always appended, so root agents, persistent children and leaves all get the tool and API documentation and the lifecycle rules. The role's append instructions sit between the task instructions and the guide; sub-agents and leaves fall back to the next more general tier when their own is unset.

The guide distinguishes Python state from supervisor-owned resources and explains waiting, completion, history, jobs and subscriptions. Delegation guidance appears only when IPython and recursion are both available.

## Steering

During a running turn, `_session/steering` (with `sessionId` and text `prompt` blocks) adds a user message at the next model step, after any tool results; running tools are not interrupted. It wakes the native `wait` tool. The request returns `injected` once the message is in the conversation, or `promptRequired` with reason `noRunningTurn` when no turn can take it; idle steering never starts a turn. Clients can set `_meta.steering.idleBehavior` to `promptRequired`.

An optional `messageId` deduplicates retries within the session; reusing an ID with different content fails. Injected messages survive prompt rollback and stay in the user-request history that compaction keeps. Queued messages that cannot be applied stay unconsumed.

## Session snapshot

Prompt responses and the `session/close` response carry one credential-free snapshot under `ai.prime.rlm/session-v1`: the session ID and last stop reason, the model, turns, cumulative token usage, metrics (compactions, sub-agent calls, programmatic tool calls, ...), supervisor counters and the effective limits. The `session/close` snapshot is authoritative.

The ACP stop reason says when a limit cut a prompt short; the snapshot's `last_stop_reason` names the exact cause. A population episode additionally carries a `population` block with the admission counts, membership counters, episode answer and per-agent usage, and its `last_stop_reason` is the episode's stop: `quiescent` (no agent can act again), `extinct` (every admitted agent ended), `max_agent_turns`, `max_seconds` or `agent_failed` (an infrastructure failure of one agent ends the episode as an error after the summary is written):

| `last_stop_reason` | Cause | ACP stop reason |
| --- | --- | --- |
| `done` | The agent answered | `end_turn` |
| `max_total_turns` | The tree reached `max_total_turns` | `max_turn_requests` |
| `max_total_tokens` | The tree reached `max_total_tokens` | `max_tokens` |
| `token_budget` | The agent reached its `max_tokens` | `max_tokens` |
| `compaction_failed` | A compaction failed with the context full | `max_tokens` |
| `format_violation` | The final summary response was unusable | `end_turn` |
| `cancelled` | The client cancelled the prompt | `cancelled` |

## Population episodes

A non-null `population` policy turns the session into one episode of peer agents instead of one root agent's tree: the single `session/prompt` runs the whole episode, steering is refused, and the returned answer is the episode answer, not any agent's reply. Set `policy.population` to the object below; omitted fields take their defaults.

| Field | Default | Meaning |
|---|---|---|
| `founders` | 1 | Peer agents admitted when the episode starts |
| `max_live_agents` | 8 | Peers holding a kernel at once, idle ones included |
| `max_total_agents` | 32 | Peers admitted over the episode; revival is not a new admission |
| `idle_grace_seconds` | 60 | How long an idle peer keeps its kernel before an admission may retire it |
| `reminder_seconds` | 60 | Repeat the inbox notice at this interval while a peer is idle with unread peer messages; 0 disables |
| `max_agent_turns` | null | Model calls one agent may make over the episode; at the cap it stops for good |
| `max_seconds` | null | Episode wall-clock limit (`stop_reason=max_seconds`) |
| `scored_on` | "answer" | What the task scores: the CAS-submitted episode answer, or the final workspace state |
| `artifacts` | [] | Final files, relative to the working directory, replaced only through `rlm.artifact.publish` |
| `allow_agent_lookup_agent_history` | false | Let peers read each other's conversation histories |

**Agents.** Founders and every `rlm.agent.spawn(..., independent=True)` are depth-0 peers: same authority, not beholden to their spawner, no lifecycle events to it, no cascade when either ends; the spawn edge is recorded for provenance and spend attribution, and the spawner keeps a handle to observe and message the peer. The default `rlm.agent.spawn(...)` inside an episode still spawns a beholden child one depth deeper: it reports to its parent, ends with it and counts against `max_depth` and `max_subagent_calls`, while the population caps count only peers.

**Messaging is inbox-plus-pull.** Direct messages, broadcasts, membership changes and job completions land in the recipient's `rlm.inbox`; nothing is pushed into a conversation. One coalesced, time-signatured notice (`[inbox 14:03] 3 messages waiting (2 dm, 1 broadcast)`) announces arrivals at the recipient's next model boundary, waking it if idle, and `reminder_seconds` repeats the notice while peer messages stay unread. Senders poll receipts with `rlm.agent.delivery` — accepted, delivered (read), answered (the reply's tail, pulled) or failed — and never wait on a wake for a reply.

**Lifecycle and budgets.** At the live cap an admission retires the longest-idle peer past its grace period: its kernel, jobs, subscriptions and services stop, its conversation and spend stay. A direct message revives it when a slot is free. Every retire or stop is owned by the supervisor, so no process of the episode outlives it: jobs, subscriptions and services all stop at episode end. Episode budgets are the tree-wide `max_total_tokens`/`max_total_turns` plus `max_seconds`, accounted across every agent including retired ones; a turn-cap death never records a synthetic answer.

**Scoring.** The scorer stays in the task runner, outside every agent's reach. `scored_on="answer"` is scored on the answer submitted through `rlm.population.submit`; `scored_on="workspace"` on the workspace state after the session closes. The episode answer and declared artifacts are supervisor-owned compare-and-swap state, the `population_summary.json` written at episode end records the submitted answer, artifact revisions and per-agent usage, and a runner snapshots the workspace after `session/close` returns.

## Tracing headers

Every model call carries a standard `Idempotency-Key` header that stays stable across SDK and outer retries (attempts differ in `x-stainless-retry-count`), so an inference proxy can deduplicate replays.

Each call also carries a private `X-ACP-Model-Request-ID`. Responses publish sparse, labeled edges between those request IDs under `ai.prime.acp/semantic-edges-v1`:

| Edge | Meaning |
| --- | --- |
| `continuation` | The same agent's next call, in causal order even where a consumer's token-prefix graph splits |
| `subagent_call` | A spawn |
| `subagent_return` | The parent retrieving a child's result |
| `agent_message` | A delivered instruction, or an inbox event the agent explicitly read; an unread count alone creates none |
| `compaction` | A compaction call |
| `compaction_attempt` | A summary attempt branching from the agent's work |
| `format_violation` | The terminal unusable summary; retained for zero-reward training |

`X-ACP-Agent-Session-ID`, also on every call, names the calling agent's session in the tree: the root's is the ACP session, each sub-agent's its own. A session-affinity router can key on it to keep each agent's growing prefix on one engine without pinning a whole tree, whose agents run concurrently, to that engine.

Clients that do not understand the extension ignore it.
