# Concepts

The design pillars of nano-rlm, per the [official documentation](https://deepwiki.com/PrimeIntellect-ai/nano-rlm) and the source.

## 1. Minimal tool interface

Instead of a large suite of specialized tools (`read_file`, `write_file`, `execute_shell`, …), the model gets a small set of high-capability built-ins — with the `ipython` tool as the centerpiece. The active set is chosen by tooling presets (`RLM_TOOLING` / `RLM_BUILTIN_TOOLS`):

```mermaid
flowchart LR
    MODEL["LLM"] -->|"one tool call per turn"| REG{"Tool registry"}
    REG --> BASH["bash<br/>fresh bash -c subshell"]
    REG --> EDIT["edit<br/>single-occurrence replace"]
    REG --> IPY["ipython<br/>persistent kernel"]
    IPY --> PY["Python"]
    IPY --> SH["!cmd / %%bash"]
    IPY --> SK["skills + await rlm()"]
```

The `dual` preset (default) exposes all three tools plus the `bash`/`edit` skills. Because `ipython` runs a real kernel, the agent has the full power of Python — file manipulation, data processing, HTTP, shell escapes — without bespoke tools.

## 2. Persistent state

The IPython kernel stays alive for the entire session lifecycle. Variables, imports, and in-memory data survive across turns — and across context compactions.

```mermaid
flowchart TB
    subgraph S["Session lifecycle"]
        T1["turn 1<br/>df = pandas.read_csv(...)"] --> T2["turn 2<br/>df.describe()"]
        T2 --> C{"context too large?"}
        C -->|yes| COMP["compaction:<br/>messages := [system, summary]"]
        COMP --> T3["turn 3<br/>df still in memory ✓"]
        C -->|no| T3
    end
    K[("single persistent kernel<br/>never restarted by compaction")] -.-> T1 & T2 & T3
```

The kernel is also credential-isolated: `build_kernel_env()` passes a whitelist of environment variables (PATH, HOME, locale, cert vars, explicit `RLM_KERNEL_ENV` entries) — credentials are de-ambiented by omission, not filtered by name.

## 3. Context compaction

When a turn's `prompt_tokens` reaches `RLM_SUMMARIZE_AT_TOKENS`, the engine triggers a compaction turn:

```mermaid
sequenceDiagram
    participant E as RLMEngine
    participant L as LLM
    participant K as IPython kernel

    E->>E: prompt_tokens ≥ summarize_at_tokens<br/>(and max_compactions not exhausted)
    E->>E: append CHECKPOINT_COMPACTION_PROMPT
    E->>L: summary call (tool_choice="none")
    L-->>E: handoff summary
    E->>E: messages := [system, framing + summary]<br/>log compaction event + CompactionApplied metric
    Note over K: kernel untouched —<br/>all variables intact
    E->>L: loop resumes from summary
```

If the agent stalls on the summary, a `REPL_RESTART_NOTE` reminds it that the kernel state is still available.

## 4. Native recursion

Agents spawn sub-agents by calling `await rlm("...")` inside their own kernel — hierarchical task decomposition without external orchestration.

```mermaid
sequenceDiagram
    participant K as Parent kernel
    participant B as Broker (unix socket)
    participant S as SessionTreeSupervisor
    participant E2 as Child RLMEngine

    K->>B: rlm.run(prompt)<br/>rlm.broker over framed JSON RPC
    B->>S: BrokerRunRequest
    S->>S: validate capability + scope<br/>check depth / call limits
    S->>E2: spawn (depth + 1, child session dir)
    E2-->>S: RLMResult(answer, usage, metrics)
    S-->>B: response
    B-->>K: RLMResult
```

- Depth and concurrency are bounded (`RLM_MAX_DEPTH`, `RLM_MAX_SUBAGENT_CALLS`, per-depth semaphores).
- Each child gets its own session directory (`sub-<uuid8>/`) and kernel.
- `asyncio.gather` over multiple `rlm()` calls gives parallel sub-agents.

## 5. Skills system

Three skill sources are unified into pre-imported, callable modules in the kernel namespace:

```mermaid
flowchart TB
    subgraph Sources
        BI["Built-in<br/>rlm/skills/<br/>bash · edit · search"]
        INST["Installed<br/>rlm-skill-* packages<br/>(uv tool install --with-editable)"]
        MCP["MCP servers<br/>RLM_MCP_CONFIG"]
    end
    BI -->|"re-export module"| DIR["session dir<br/>*.py modules"]
    MCP -->|"proxy module<br/>__rlm_brokered__ = True"| DIR
    INST --> KER
    DIR --> KER["Kernel startup injection<br/>_CallableModule wrapper"]
    KER -->|"await skill(...) == await skill.run(...)"| AGENT["Agent code"]
    AGENT -->|"brokered: skill.call RPC"| SUP["Supervisor<br/>(credentials stay out of kernel)"]
```

Name collisions across sources raise an error at discovery. Every programmatic skill call is logged to `programmatic_tool_calls.jsonl` for telemetry.

## 6. Git history guard

Active unless `RLM_ALLOW_GIT=1`. It blocks broad-history `git log` (e.g. `--all`, `--reflog`, `--branches`) — not git in general — so evaluation agents can't mine task solutions from history.

```mermaid
flowchart LR
    CODE["agent-generated code"] --> CHK{"find_blocked_*"}
    CHK -->|"bash tool / !cmd / %%bash"| SH["shell parser:<br/>split && || ; |<br/>shlex per segment"]
    CHK -->|"python code"| AST["AST visitor:<br/>tracks subprocess/os aliases<br/>literal argv only"]
    SH --> DEC{blocked?}
    AST --> DEC
    DEC -->|yes| REFUSE["REFUSAL_TEMPLATE error"]
    DEC -->|no| EXEC["execute normally"]
```

Documented bypasses (`getattr`, multi-hop reassignment, `__import__`) are accepted as out of scope — the guard is a tripwire, not a sandbox.
