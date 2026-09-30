# Experimental execution replay

`ExecutionTape` records one RLM session tree and re-executes its native tools in
fresh kernels using the original model responses. Child engines inherit the tape
from the supervisor. No inference request is sent during replay, including for
compaction. This is a Python API prototype; the ACP contract is unchanged.

## Running it

Run the example **inside an already prepared sandbox**. It does not provision a
sandbox or restore its filesystem. Record and replay with the same nano-rlm
revision, dependencies, skills, initial files, environment and workspace path.
Keep tapes and session directories outside the task workspace.

Provide a `RuntimeConfig` JSON file, for example:

```json
{
  "model": "your-model",
  "provider": {"base_url": "http://your-provider/v1", "api_key": "your-key"},
  "invocation": {},
  "policy": {"compaction": false}
}
```

```sh
uv run examples/replay.py record /artifacts/run.jsonl \
  --config /config/runtime.json --session /artifacts/recorded \
  --cwd /task --prompt 'Solve the task in /task'
```

Restore the starting sandbox state, copy the tape into the fresh sandbox, and run:

```sh
uv run examples/replay.py replay /artifacts/run.jsonl \
  --config /config/runtime.json --session /artifacts/replayed \
  --cwd /task --prompt 'Solve the task in /task'
```

The replay provider configuration may use a dummy key; it is not contacted.
Compaction requires an explicit `summarize_at_tokens` value so startup does not
query the provider for its context window. All other execution configuration must
match. The caller remains responsible for supplying the original environment.

For an existing engine integration, pass
`execution_tape=ExecutionTape(path, mode="record")` (or `"replay"`) to the root
`RLMEngine`. After the engine and its children have closed, call `tape.finish()`
to seal the recording or verify complete replay consumption. Always call
`tape.close()` in cleanup. For a caller-owned supervisor, pass the tape to that
supervisor instead; its engines inherit it. A tape is owned by one event loop.

## What is recorded and checked

The append-only JSONL contains a version header, root/child identities, timestamped
boundary events, and a completion marker. The model events contain full request
bodies and full responses, including usage and compaction responses. Tool events
contain native tool arguments and the full output before context truncation.
Session identities, child parentage, spawn arguments, and parent-local spawn
ordinals let fresh children consume their own model responses.

Replay checks model inputs, native tool inputs, outputs, prompt results, and the
recorded order of model/tool boundaries across actors. An actor that arrives early
waits for the preceding actor. A parent cell can remain running while child
engines execute: the scheduler never waits for the parent cell to finish before
allowing its children to start. A missing expected actor fails after 30 seconds
without journal progress rather than waiting indefinitely.

Session directory paths are normalized only for comparisons. Model responses and
executable code are preserved exactly. Transport request headers are excluded
from matching; fresh semantic request IDs are not execution identities. Runtime
provider credentials and kernel environment values are not serialized as config.
Prompts, code, responses and tool output are stored verbatim and may themselves
contain sensitive data.

A mismatch latches a tree-wide `ReplayDivergence`; a caught child failure cannot
turn into a successful replay. Incomplete recordings and unconsumed events are
rejected. Recording files are created exclusively and never overwritten.

## Guarantee and limits

This is strict checking at harness boundaries, not deterministic OS execution.
Matching observations do not prove identical hidden state or filesystem contents.
Tools really execute, including their network calls and side effects. Only model
responses are substituted. Restore initial inputs before each replay; replaying
in the original mutated workspace is not equivalent.

The integration tests exercise two concurrent child engines writing separate
files, persistent parent variables, compaction with a live kernel, changed tool
output, changed requests, incomplete tapes and cross-actor failure propagation.
They use real IPython kernels and scripted model responses.

The prototype does not capture or schedule individual operations inside a cell,
programmatic skill calls, broker messages, inbox delivery, processes, random
numbers, clocks or network responses. Persistent-agent messaging, subscriptions,
shared-file races and time-dependent observations can therefore diverge. Native
model errors and prompt cancellation make a recording unsealable. Errors returned
as ordinary tool output can replay if the output matches. Existing transcript
logs cannot be converted into a complete tape.

The next layer is a broker operation journal with stable operation IDs and
recorded delivery/spawn/join/cancellation decisions, followed by capture of
external observations and a sandbox snapshot manifest. Exact replay of arbitrary
Python and shell races additionally requires control below the harness boundary.
