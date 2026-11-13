# Context and history

A session has no fixed length. When the conversation outgrows the model's context window, the engine compacts it into a handoff summary and carries on, while the kernel keeps every variable and the full record stays on disk, readable from Python. The model loses nothing it cannot get back.

## Compaction

Compaction is on by default and unlimited: `compaction: false` turns it off and `max_compactions` caps it. The default `max_total_tokens` budget still bounds the session, since each compaction spends new tokens too.

**When.** The engine reads the real context window from the provider's `/models` response. Its default work cutoff is that window minus `summary_budget_tokens` (24,576), reserving space for both the summary instructions and reply, including reasoning. Small windows reserve half. `summarize_at_tokens` can request earlier compaction.

Providers advertising `supports_max_context_tokens: true` on the model card enforce `max_context_tokens` as an exact prompt-plus-completion limit: they cap generation at the limit minus the rendered prompt length, preserve smaller completion caps, and reject input already at the limit with a context-overflow error. Work requests use the smaller of the configured threshold and the default cutoff; summary requests use the real window. This cuts work mid-turn without consuming the summary reserve. Verifiers TrainClient requires support for this extension. Other OpenAI-compatible providers keep between-turn threshold checks and recovery at the real window; they do not promise an exact early cutoff. Without a known window or explicit threshold, compaction is overflow-reactive only.

**Input crossing the cutoff.** A new tool result or user input can cross the work cutoff without exceeding the real window. On an exact-cutoff provider, the engine summarizes that full input first, without trimming it to the work limit. Input that exceeds the real window cannot be summarized in one request: overflow recovery shortens the summary input on subsequent attempts. Tool output limits still apply independently.

**How.** The engine asks for a plain-text handoff summary and resumes on a fresh context seeded with its final text. The summary preserves requirements, progress, pending work, history references and useful kernel variables. On an exact-cutoff provider, the full conversation reaches the first summary attempt; the instructions consume part of the reserve, and generation gets the remaining room, up to `summary_budget_tokens` including reasoning. A large new input can consume some of that reserve too, so 24k is headroom, not a guarantee that every summary will finish.

Providers without the extension estimate and trim the summary input up front to leave the reply budget. Overflow retries can also shorten it: old tool outputs give way to a stub, then old reasoning is dropped, then whole old exchanges become an omission marker. The task or previous summary and the newest exchange are protected; an oversized newest reply keeps its beginning and end, with a shared budget across content and reasoning of at most 4k beginning and 16k ending tokens, reduced to fit. A retry may use the last accepted snapshot, except when it would lose an interrupted reply. If the protected context cannot fit, compaction fails without replacing the live context. Empty, tool-calling or unfinished summaries are retried up to `max_compaction_attempts`.

**Replies that reach the cutoff.** A reply cut off at the work limit stays in the conversation, minus any tool call it was cut off in, and compaction runs at once with the summary asked to record where it left off and what came next. A reply stopped by a smaller completion cap ends the prompt like any reply without a tool call. Without an advertised window the two cannot be told apart, and a length-cut reply ends the prompt.

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
