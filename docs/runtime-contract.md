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
|---|---|---|
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
|---|---|---|
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
| `max_compactions` | null | Compactions per session |
| `max_compaction_attempts` | 5 | Summary attempts within one compaction |
| `allow_git` | false | Lift the guard that refuses Git commands reaching beyond the current branch (other refs, branch and tag listings, `clone`, `fetch`, `pull`) |

**Model-call admission.** `max_concurrent_subagents` bounds calls across the tree. Queued calls closer to the root go first, FIFO within a depth, and calls in flight are not preempted, so continuous shallow work can delay deeper agents. Tools, child waits and the backoff between outer retries hold no slot; the SDK's own retries stay within one. The limit is per tree: a cap across trees belongs in the runner or inference gateway.

## Prompts

`system_prompt_path` supplies task instructions in place of the default task role. The runtime guide is always appended, so root agents, persistent children and leaves all get the tool and API documentation and the lifecycle rules. The role's append instructions sit between the task instructions and the guide; sub-agents and leaves fall back to the next more general tier when their own is unset.

The guide distinguishes Python state from supervisor-owned resources and explains waiting, completion, history, jobs and subscriptions. Delegation guidance appears only when IPython and recursion are both available.

## Steering

During a running turn, `_session/steering` (with `sessionId` and text `prompt` blocks) adds a user message at the next model step, after any tool results; running tools are not interrupted. It wakes the native `wait` tool. The request returns `injected` once the message is in the conversation, or `promptRequired` with reason `noRunningTurn` when no turn can take it; idle steering never starts a turn. Clients can set `_meta.steering.idleBehavior` to `promptRequired`.

An optional `messageId` deduplicates retries within the session; reusing an ID with different content fails. Injected messages survive prompt rollback and stay in the user-request history that compaction keeps. Queued messages that cannot be applied stay unconsumed.

## Session snapshot

Prompt responses and the `session/close` response carry one credential-free snapshot under `ai.prime.rlm/session-v1`: the session ID and last stop reason, the model, turns, cumulative token usage, metrics (compactions, sub-agent calls, programmatic tool calls, ...), supervisor counters and the effective limits. The `session/close` snapshot is authoritative.

The ACP stop reason says when a limit cut a prompt short; the snapshot's `last_stop_reason` names the exact cause:

| `last_stop_reason` | Cause | ACP stop reason |
|---|---|---|
| `done` | The agent answered | `end_turn` |
| `max_total_turns` | The tree reached `max_total_turns` | `max_turn_requests` |
| `max_total_tokens` | The tree reached `max_total_tokens` | `max_tokens` |
| `token_budget` | The agent reached its `max_tokens` | `max_tokens` |
| `compaction_failed` | A compaction failed with the context full | `max_tokens` |
| `format_violation` | The final summary response was unusable | `end_turn` |
| `cancelled` | The client cancelled the prompt | `cancelled` |

## Tracing headers

Every model call carries a standard `Idempotency-Key` header that stays stable across SDK and outer retries (attempts differ in `x-stainless-retry-count`), so an inference proxy can deduplicate replays.

Each call also carries a private `X-ACP-Model-Request-ID`. Responses publish sparse, labeled edges between those request IDs under `ai.prime.acp/semantic-edges-v1`:

| Edge | Meaning |
|---|---|
| `continuation` | The same agent's next call, in causal order even where a consumer's token-prefix graph splits |
| `subagent_call` | A spawn |
| `subagent_return` | The parent retrieving a child's result |
| `agent_message` | A delivered instruction, or an inbox event the agent explicitly read; an unread count alone creates none |
| `compaction` | A compaction call |
| `compaction_attempt` | A summary attempt branching from the agent's work |
| `format_violation` | The terminal unusable summary; retained for zero-reward training |

`X-ACP-Agent-Session-ID`, also on every call, names the calling agent's session in the tree: the root's is the ACP session, each sub-agent's its own. A session-affinity router can key on it to keep each agent's growing prefix on one engine without pinning a whole tree, whose agents run concurrently, to that engine.

Clients that do not understand the extension ignore it.
