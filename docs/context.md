# Context and history

A session has no fixed length. When the conversation outgrows the model's context window, the engine compacts it into a handoff summary and carries on, while the kernel keeps every variable and the full record stays on disk, readable from Python. The model loses nothing it cannot get back.

## Compaction

Compaction is on by default and unlimited: `compaction: false` turns it off and `max_compactions` caps it. The default `max_total_tokens` budget still bounds the session, since each compaction spends new tokens too.

**When.** The engine reads the model's context window from the provider's `/models` response and compacts once 16k tokens remain; small windows keep at least half. `summarize_at_tokens` pins the threshold instead. Without a known window or explicit threshold, compaction happens only when the provider reports an overflow (a 400 or 413 naming a context limit), which also triggers it with a threshold set.

**How.** The engine asks the model for a plain-text handoff summary, without reasoning, and resumes on a fresh context seeded with it. The checkpoint prompt asks the summary to keep the task's requirements, evidence, outstanding assignments, jobs, subscriptions, pending event actions, output cursors and history references, plus the commands and edits that matter, and to name the important kernel variables so the resumed context knows what exists. An empty or tool-calling reply is resampled. A rejected checkpoint request falls back to the last state that passed a threshold check, which by definition has a full reserve of room. After `max_compaction_attempts` failures the run ends cleanly with what the conversation holds. An overflow with nothing beyond the task to drop propagates, because there is nothing to reclaim.

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

Indices are local to each agent. A session opens a fresh ledger and refuses to reopen an existing `messages.jsonl` for writing. A failed write or flush makes the session refuse further prompts; rollback restores the in-memory context even if its own record cannot be written.

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
