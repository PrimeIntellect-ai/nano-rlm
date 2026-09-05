# nano-rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents. For a full-fledged coding agent built on the same RLM principles, see [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent).

The model gets a single built-in tool, `ipython`: a persistent IPython kernel for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`. The tool set is not configurable. File edits, shell work, and orchestration all go through it.

For convenience, rlm ships built-in *skills* that can be enabled per session via the runtime contract's `skills` list (off by default): `edit` (single-occurrence string replacement), `search` (web search via Serper, needs `SERPER_API_KEY`), and `fetch` (retrieve a URL as cleaned text). Enabled skills are pre-imported into the IPython kernel like any other skill (see [Skills](#skills)), so the agent calls `await edit(path=..., old_str=..., new_str=...)`, `await search(query=...)`, or `await fetch(url=...)`. `fetch` also exists as a native builtin tool with the same semantics, for tool-calling runs (opt-in via `RLM_BUILTIN_TOOLS`).

Context compaction is on by default: the engine compacts when 16k tokens remain below an advertised model context window. Termination comes from the default tree-wide budget of 1M new tokens (`max_total_tokens`). The policy can set an explicit `summarize_at_tokens` threshold. The IPython kernel keeps running across compaction, so REPL state survives (see [Compaction](#compaction)).

Inside the IPython session, a callable `rlm` is pre-injected into the namespace. When recursion is allowed, the model can call `await rlm(...)` to spawn sub-agents. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

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

Inside a running session's IPython kernel, `rlm.run("sub-task")` (or the pre-injected
`rlm(...)` callable) spawns a recursive sub-agent through the session's broker. There is
no standalone entry point: outside a session the call raises.

## Configuration

All runtime configuration enters through the `ai.prime.rlm/runtime-v1` contract object
(model, provider credentials, execution policy, prompt configuration, skills, kernel
environment, search credential). Prompt configuration is role-aware: optional
`subagent_append_to_system_prompt` (nodes that can still recurse) and
`leaf_append_to_system_prompt` (depth == max_depth) override `append_to_system_prompt`
for sub-agents, each falling back to the next-more-general tier. Recursive children inherit the parent's configuration
in-memory (`model_copy`); nothing is re-read from the process environment.

The process environment configures only process infrastructure:

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |

## Recursion

Each agent runs inside a persistent IPython kernel with an already-running event loop. A callable `rlm` is pre-injected into the kernel namespace, so recursive calls are just `await`:

```python
result = await rlm("verify the fix")
```

The result is an `RLMResult` with `.answer`, `.task`, `.usage`, `.turns`, and `.session_dir`.
The harness sets `.task` to the exact prompt passed to that invocation, including limit results;
it is not generated by the child. For a follow-up `engine.prompt(...)`, it identifies that
follow-up request. Caller requests are also retained verbatim across context compaction,
independently of child reports or generated checkpoint summaries.

For parallel sub-agents, use normal async Python:

```python
import asyncio

results = await asyncio.gather(
    rlm("check auth.py"),
    rlm("check login.py"),
)
```

Recursive calls are created by a session-local supervisor rather than by the IPython kernel. The supervisor assigns depth and session ancestry, enforces the concurrency and total-call limits, and cancels descendants when their parent cell or session closes. When recursion is disabled by depth, the system prompt does not advertise these APIs and child runs beyond the depth limit fail immediately.

The IPython execution timeout measures active cell execution. Time spent responsively awaiting a supervisor-owned sub-agent does not consume that budget, nor does an `asyncio.gather` containing only sub-agent calls. Kernel CPU, skills, ordinary waits, mixed gathers, background sub-agent tasks, and periods without broker heartbeats still do. Recursive work remains bounded independently by the tree resource policy and the runtime hosting the rollout.

## Compaction

There is no model-driven compaction tool. Compaction is on by default and unlimited (the policy's `compaction` field turns it off, `max_compactions` caps it); the default 1M `max_total_tokens` tree budget keeps sessions bounded, since each compaction cycle itself spends new tokens. The engine reads the model context window from the provider's `/models` response and compacts when 16k tokens remain below it; small windows keep at least half. Set `summarize_at_tokens` to pin the threshold explicitly. Without a known window or explicit threshold, proactive compaction stays off, but a provider overflow still triggers it reactively. A tool result larger than 20KB is truncated to its head and tail before it enters the conversation, with a warning naming the original size.

The engine asks the model for a plain-text handoff summary and resumes the task on a fresh branch seeded with that summary; reasoning is never part of it. A provider overflow (a 400 or 413 naming a context limit) triggers the same compaction reactively from the current state. A rejected checkpoint request falls back to the last state that passed a threshold check - by definition a state with a full reserve of room - and an empty or tool-calling reply is resampled; after three failed attempts the run ends cleanly with what the conversation holds. An overflow with no history beyond the task propagates: the task alone approaches the window and there is nothing to reclaim.

The IPython kernel keeps running across the compaction, so all variables, imports, and in-memory data are preserved. The model is told to mention important variable names in its summary so the resumed branch knows what is available. The same policy applies to the main agent and all recursive agents.

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
