# nano-rlm

A minimal CLI coding agent with a persistent IPython execution environment and optional recursive sub-agents. For a full-fledged coding agent built on the same RLM principles, see [prime-agent](https://github.com/PrimeIntellect-ai/prime-agent).

The model gets a single built-in tool, `ipython`: a persistent IPython kernel for Python, shell commands via `!command`, and multi-line shell scripts via `%%bash`. The tool set is not configurable. File edits, shell work, and orchestration all go through it.

For convenience, rlm ships built-in *skills* that can be enabled per run via `RLM_SKILLS` (comma-separated, off by default): `edit` (single-occurrence string replacement) and `search` (web search via Serper, needs `SERPER_API_KEY`). Enabled skills are pre-imported into the IPython kernel like any other skill (see [Skills](#skills)), so the agent calls `await edit(path=..., old_str=..., new_str=...)` or `await search(query=...)`.

Context compaction is optional. When enabled, the engine compacts when 16k tokens remain below the model context window when the provider advertises it. `RLM_SUMMARIZE_AT_TOKENS` sets an explicit threshold. The IPython kernel keeps running across compaction, so REPL state survives (see [Compaction](#compaction)).

Inside the IPython session, a callable `rlm` is pre-injected into the namespace. When recursion is allowed, the model can call `await rlm(...)` to spawn sub-agents. Skills supplied by the host environment (see [Skills](#skills)) are importable directly by name, e.g. `import websearch`.

## Install

```bash
git clone https://github.com/PrimeIntellect-ai/nano-rlm.git
cd nano-rlm
uv sync
source .venv/bin/activate
```

## CLI

```bash
rlm "fix the auth bug in login.py"

# Override model
RLM_MODEL=openai/gpt-5-mini rlm "refactor the parser"

# Append extra instructions to the generated system prompt
RLM_APPEND_TO_SYSTEM_PROMPT="Always run tests before finishing." rlm "solve the task"

# Replace the generated system prompt from a file
RLM_SYSTEM_PROMPT_PATH=/tmp/system.txt rlm "solve the task"
```

Skill CLIs provided by the host environment are on `$PATH` and invoked the same way (e.g. `websearch --queries "latest jupyter_client release"` when the `websearch` skill is installed).

## Agent Client Protocol

RLM can run as an [Agent Client Protocol](https://agentclientprotocol.com/)
agent over stdio:

```bash
rlm --acp
```

Each ACP session owns one persistent RLM engine. Repeated `session/prompt`
requests retain both the model conversation and the live IPython kernel. The
agent accepts text prompts and stdio or streamable HTTP MCP servers, including
HTTP request headers. It supports cancellation and `session/close`; cancelling
a turn leaves its session available for the next prompt. It intentionally does
not advertise `session/load`: an arbitrary live Python kernel cannot be
reconstructed after the ACP process exits, so clients must keep the process
alive for the lifetime of a session.

RLM's ACP surface is a versioned training contract, not a compatibility layer
over the standalone CLI. `initialize` advertises the exact
`ai.prime.rlm/contract-v1` marker in its response `_meta`; clients must require
it, then provide one complete `ai.prime.rlm/runtime-v1` object in
`session/new._meta`. The runtime object contains the lineage session ID, model,
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

## Python SDK

```python
import asyncio
import rlm

result = asyncio.run(rlm.run("fix the bug"))
```

## Standalone configuration

The CLI and Python API resolve standalone configuration from environment
variables. ACP sessions ignore these runtime fields and require the explicit
versioned contract described above.

| Variable | Default | Description |
| ---------- | --------- | ------------- |
| `RLM_HOME` | `~/.rlm` | Root directory for sessions and data |
| `RLM_MODEL` | `openai/gpt-5-mini` | Model name (PI Inference slug). Override with `--model` or `RLM_MODEL` for OpenAI/Anthropic direct (e.g. `gpt-4o`, `claude-sonnet-4-5`) |
| `RLM_API_KEY` / `RLM_BASE_URL` | — / SDK default (`https://api.openai.com/v1`) | Explicit override (highest priority). Independent: setting `RLM_API_KEY` alone targets the SDK default endpoint; set `RLM_BASE_URL` too for a custom endpoint. For PI, use `PRIME_API_KEY` (below) which owns the full pair. |
| `SERPER_API_KEY` | — | API key for the built-in `search` skill (Serper backend). Resolved by the supervisor and not copied into the kernel. |
| `PRIME_API_KEY` | — | PI Inference pair: targets `https://api.pinference.ai/api/v1` and forwards `PRIME_TEAM_ID` as `X-Prime-Team-ID` when set. |
| `OPENAI_API_KEY` / `OPENAI_BASE_URL` | resolved at startup | OpenAI pair (covers OpenAI direct and verifiers' rollout tunnel). Provider precedence: explicit → PI → OpenAI. Keys are scoped to their own base URL so an `OPENAI_API_KEY` lying around can't leak to PI Inference. |
| `RLM_SKILLS` | — | Comma-separated built-in skills to enable (`edit`, `search`); pre-imported into the kernel. Unknown names raise. See [Skills](#skills). |
| `RLM_MCP_CONFIG` | — | Standard `mcpServers` config (streamable HTTP or stdio); each server's tools become pre-imported IPython skills (`<server>_<tool>`). See [MCP tools as skills](#mcp-tools-as-skills). |
| `RLM_KERNEL_ENV` | `{}` | JSON object of task variables explicitly passed to IPython and its subprocesses. Supervisor, provider, MCP, and broker configuration names are reserved. |
| `RLM_MAX_DEPTH` | `0` | Max recursion depth (`0` means no sub-agents) |
| `RLM_MAX_CONCURRENT_SUBAGENTS` | `max(4, RLM_MAX_DEPTH)` | Maximum live recursive agents in a session tree. Capacity is reserved per depth to prevent nested-call deadlocks. |
| `RLM_MAX_SUBAGENT_CALLS` | `64` | Maximum accepted recursive calls across the complete session tree. |
| `RLM_EXEC_TIMEOUT` | `300` | Seconds per IPython execution |
| `RLM_COMPACTION` | — | Set to `1` to enable context compaction. |
| `RLM_SUMMARIZE_AT_TOKENS` | — | Proactive compaction threshold for the current prompt, completion, and pending tool result. Setting it also enables compaction. When unset, enabled compaction uses 90% of an advertised context window. |
| `RLM_MAX_TOKENS` | `0` | Optional completion-token budget (`0` disables) |
| `RLM_APPEND_TO_SYSTEM_PROMPT` | — | Extra instructions appended to the generated system prompt |
| `RLM_SYSTEM_PROMPT_PATH` | — | Path to a file whose contents fully replace the generated system prompt |
| `RLM_ALLOW_GIT` | — | Set to `1` to disable the restricted git-history guard. When unset, shell-capable prompts tell agents not to use task-specific online hints or solutions from other git history, and broad-history `git log` options such as `--all` are refused. |
| `RLM_SDK_MAX_RETRIES` | `5` | Per-request retry count passed to the OpenAI SDK (in addition to the call-site retry wrapper that rides out longer outages). |

`RLM_SYSTEM_PROMPT_PATH` takes precedence over `RLM_APPEND_TO_SYSTEM_PROMPT`. CLI flags override env vars: `rlm --model gpt-5-mini --append-to-system-prompt "..." --system-prompt-path /tmp/system.txt "prompt"`.

## Recursion

Each agent runs inside a persistent IPython kernel with an already-running event loop. A callable `rlm` is pre-injected into the kernel namespace, so recursive calls are just `await`:

```python
result = await rlm("verify the fix")
```

The result is an `RLMResult` with `.answer`, `.usage`, `.turns`, and `.session_dir`. For parallel sub-agents, use normal async Python:

```python
import asyncio

results = await asyncio.gather(
    rlm("check auth.py"),
    rlm("check login.py"),
)
```

Recursive calls are created by a session-local supervisor rather than by the IPython kernel. The supervisor assigns depth and session ancestry, enforces the concurrency and total-call limits, and cancels descendants when their parent cell or session closes. When recursion is disabled by depth, the system prompt does not advertise these APIs and child runs beyond the depth limit fail immediately.

## Compaction

There is no model-driven compaction tool. Set `RLM_COMPACTION=1` to enable compaction. The engine reads the model context window from the provider's `/models` response and compacts when 16k tokens remain below it (small windows keep at least half). Set `RLM_SUMMARIZE_AT_TOKENS` to pin the threshold explicitly. Without a known window or explicit threshold, compaction stays off and an overflow propagates. A tool result larger than 20KB is truncated to its head and tail before it enters the conversation, with a warning naming the original size.

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

`rlm` ships a small set of built-in skills enabled per run via `RLM_SKILLS` (`edit`, `search`; see [MCP tools as skills](#mcp-tools-as-skills) for the related MCP path). `edit` runs in the kernel. Credentialed `search` runs in the supervisor through the capability broker, so `SERPER_API_KEY` is unavailable to IPython and its subprocesses; it returns title/URL/snippet for a single query (`await search(query="...")`). Additional skills are supplied by the host environment: before `install.sh` runs, the environment places skill packages under `/task/rlm-skills/<name>/`, and `install.sh` installs them alongside `rlm` so they're both importable and on `$PATH`.

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

A host harness can wire task-specific [MCP](https://modelcontextprotocol.io) tool servers to `rlm` by setting `RLM_MCP_CONFIG` to a standard `mcpServers` config. Streamable HTTP and stdio transports are supported:

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

Programmatically, pass `mcp_servers={"tools": "http://127.0.0.1:8000/mcp"}` to `RLMEngine` / `rlm.run` instead (it takes precedence over `RLM_MCP_CONFIG`). An HTTP server may instead be `{"url": "...", "headers": {"Authorization": "..."}}` when it needs request headers; stdio servers use the same `command` / `args` / `env` shape shown above.

At startup `rlm` connects to each server, lists its tools, and generates one skill per tool (named `<server>_<tool>`, e.g. `tools_add_event`). These join the installed skills — pre-imported into the IPython namespace as async functions the agent calls programmatically, with a signature built from the tool's input schema:

```python
help(tools_add_event)  # signature (typed from the schema) + the tool's description
await tools_add_event(day="monday", title="standup")
```

Each call connects using the configured transport, invokes the tool, and returns its text content (a tool-reported error is raised as `RuntimeError`). Discovery and invocation run in the session supervisor. The generated modules contain only the public tool schema and an opaque capability; server URLs, headers, commands, and environment variables are not copied into the kernel environment or session artifacts. The kernel imports these proxy modules from the session directory and reaches the supervisor over the session-local broker. Unlike installed skills, MCP skills are IPython-only — they're not exposed as shell commands.

## Kernel

The IPython kernel always runs in rlm's own Python (`sys.executable`). `install.sh` puts `rlm` and all discovered skills into the same `uv tool install` environment, so `from rlm import run`, `import edit`, etc. work natively from inside an IPython cell.

The kernel starts from a small platform environment (`PATH`, home/user/shell, locale, temporary-directory, certificate, and virtual-environment variables) plus the explicit `RLM_KERNEL_ENV` mapping. It receives private Jupyter/IPython config directories and does not inherit the rest of the supervisor process environment. This de-ambients credentials; it is not hostile-code containment because the kernel still shares the sandbox user, filesystem, process namespace, and network with the supervisor.

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

## Interactive Mode

Running `rlm` with no prompts enters a placeholder interactive mode. The TUI is not implemented yet.
