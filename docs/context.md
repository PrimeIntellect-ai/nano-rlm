# Context and history

A session has no fixed length. When the conversation outgrows the model's context window, the engine compacts it into a handoff summary and carries on, while the kernel keeps every variable and the full record stays on disk, readable from Python. The model loses nothing it cannot get back.

## Compaction

Compaction is on by default and unlimited: `compaction: false` turns it off and `max_compactions` caps it. The default `max_total_tokens` budget still bounds the session, since each compaction spends new tokens too.

**When.** The engine reads the model's context window from the provider's `/models` response and compacts once the summary budget (`summary_budget_tokens`, 24k by default) plus 8k tokens remain; small windows keep at least half. `summarize_at_tokens` pins the threshold instead. Without a known window or explicit threshold, compaction happens only when the provider reports an overflow (a 400 or 413 naming a context limit), which also triggers it with a threshold set.

**How.** The engine asks the model for a plain-text handoff summary, without reasoning, and resumes on a fresh context seeded with it. The checkpoint prompt asks the summary to keep the task's requirements, evidence, outstanding assignments, jobs, subscriptions, pending event actions, output cursors and history references, plus the commands and edits that matter, and to name the important kernel variables so the resumed context knows what exists. The summary gets `summary_budget_tokens` of the window, reasoning included, capped at half a known window to leave room for the retained input: a summary request that would leave it less is trimmed first, oldest and cheapest context first. Tool outputs give way to a stub, then reasoning is dropped (tool calls stay as the record of what was done), then whole exchanges are replaced by one omission marker. The task or previous summary and the newest exchange are always kept, and a newest reply too large for what remains keeps its beginning and end. These share a budget across content and reasoning, at most 4k tokens from the beginning and 16k from the end, reduced further to fit the available space. An empty or tool-calling reply is resampled. A rejected checkpoint request may fall back to the last state that passed a threshold check. When compacting an interrupted reply, every retry retains that reply's ends; if the protected context cannot fit, compaction fails without replacing the live context. After `max_compaction_attempts` failures the run ends cleanly with what the conversation holds. An overflow with nothing beyond the task to drop propagates, because there is nothing to reclaim.

**Replies that run into the window.** A reply cut off by the advertised context window stays in the conversation, minus any tool call it was cut off in, and compaction runs at once with the summary asked to record where that reply left off and what came next. The session continues from the summary, so every reply it produced is one the conversation builds on. A reply stopped by a completion cap below the window is the turn's answer and ends the prompt like any reply without a tool call. Without an advertised window the two cannot be told apart, and a length-cut reply ends the prompt.

For training, set `max_compaction_attempts: 1`. An unusable final summary (empty, tool-calling, or not normally terminated) ends the prompt with `format_violation`. Its terminal attempt is advertised as a `format_violation` semantic edge so a trainer can retain those tokens and assign zero reward. Earlier discarded attempts remain separate. A rejected provider request or an exhausted context-recovery path remains `compaction_failed`, without a format-violation edge.

The kernel keeps running through compaction, so variables, imports and in-memory data survive it. The same policy applies to every agent in the tree.

## Tool output

A tool result longer than 20 KB (`max_tool_output_bytes`) enters the conversation as its first and last 10 KB around a notice with the whole output's token estimate, line count and omitted bytes. The full result stays in the ledger.

A command or cell always runs to its end, however much it prints. Its result keeps at most the first and last 512 KiB of output (bash divides that between stdout and stderr) and records how much more the whole output measured, so the notice reports the whole output's size either way. The ledger keeps the retained ends with a marker where the middle was dropped. `rlm.shell` jobs keep up to 16 MiB each in their own files ([Shell jobs](python-api.md#shell-jobs)).

## The ledger

Each agent appends its conversation to `messages.jsonl` in its session directory. The ledger serves the agent's own recovery: it is what `history()` reads and what compacted contexts point to. Clients that trace model calls record requests and responses at their own proxy and receive metrics and semantic edges over ACP, independent of this file.

- **Messages.** Message records carry a stable event `id`, a zero-based `message_index` and a `message` with the role, content and tool calls. System messages and installed summaries are recorded; checkpoint prompts and raw checkpoint responses are not. A tool result truncated for the context has a separate record linking back through `source_message_index`.
- **Windows.** A `context_window` record opens a window with a zero-based `window` index, a `reason` (`start`, `compaction` or `rollback`) and the ordered `message_indices` that seed it. Later records with a `window` field append to it, and closed windows never change. A failed or cancelled prompt rolls back by opening a new window with the restored context, so addresses in the failed window stay valid.
- **Prompts and turns.** A `user` record begins a prompt attempt, and `prompt_rollback.prompt_id` marks one that failed or was cancelled; its records remain as history. The `turn` field counts loop iterations and is not an index.
- **Other records.** Lifecycle records such as `kernel_recovery`, child spawns and delivered inbox notifications are in the same file.

Indices are local to each agent. A session opens a fresh ledger and refuses to reopen an existing `messages.jsonl` for writing. If a write fails (for example, the disk is full), the session cuts the file back to its last whole record. It keeps the unwritten records in memory and writes them with the next record, so the session continues and the file holds only whole records. The session must write every record before it finishes, or finishing fails. If the session cannot cut back a failed write, the session refuses further prompts. Rollback restores the in-memory context even if its own record cannot be written.

## Reading history

```python
from rlm import history

h = await history()                        # this agent's ledger ($RLM_SESSION_DIR)
requests = h.user_messages()                # original inputs, rolled-back attempts included
message = h.messages[3]                     # by message index
earlier = h.windows[0].messages             # the first context window
other = await history(session_dir="/path/to/session")
message = other.windows[4].messages[2]      # if that session has reached window 4
```

A snapshot holds the complete records as of the read; read again to see new activity. `h.events` holds the lifecycle records, including each child's spawn prompt and rollback markers. For a child, `await child.history()` goes through the supervisor ([Agents](python-api.md#agents)). Compacted contexts point to this API, so the model can recover details without putting the transcript back in context.

## Session directory

Every agent writes `$RLM_HOME/sessions/<id>/`, and children nest under their parent, mirroring the call tree:

```text
~/.rlm/sessions/abc123/
├── meta.json
├── messages.jsonl
├── inbox.jsonl
├── jobs/<job-id>/{output.bin, meta.json}
├── sub-d4e5/
│   ├── meta.json
│   ├── messages.jsonl
│   └── sub-f6g7/
└── sub-h8i9/
```

These files are meant for debugging, visualization and extracting training data.
