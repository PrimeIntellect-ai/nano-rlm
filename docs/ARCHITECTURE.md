# Architecture

System shape of nano-rlm, grounded in `src/rlm/`. Diagrams use Mermaid and render natively on GitHub.

## System overview

`RLMEngine` (`src/rlm/engine.py`) is the central orchestrator. It bridges natural-language reasoning with a concrete execution space: a persistent IPython kernel, a small set of built-in tools, and a session record on disk.

```mermaid
flowchart TB
    subgraph Entry["Entry points"]
        CLI["rlm CLI<br/>cli.py"]
        ACP["ACP server<br/>acp.py"]
        API["Python API<br/>rlm.run — api.py"]
    end

    subgraph Core["Agent core"]
        ENGINE["RLMEngine<br/>engine.py"]
        CLIENT["LLM client<br/>client.py"]
        PROMPT["System prompt builder<br/>prompt.py"]
    end

    subgraph Exec["Execution environment"]
        REG["Tool registry<br/>tools/registry.py"]
        BASH["bash tool"]
        EDIT["edit tool"]
        IPTOOL["ipython tool"]
        REPL["IPythonREPL<br/>tools/ipython.py"]
        KERNEL[("Persistent<br/>IPython kernel")]
        GUARD["GitHistoryGuard<br/>tools/git_block.py"]
    end

    subgraph Rec["Recursion"]
        SUP["SessionTreeSupervisor<br/>supervisor.py"]
        BRK["Broker<br/>broker.py<br/>(unix socket, framed JSON RPC)"]
    end

    subgraph State["Persistence"]
        SES["Session<br/>session.py"]
        FS[("$RLM_HOME/sessions/&lt;id&gt;<br/>meta.json · messages.jsonl")]
    end

    CLI --> ENGINE
    ACP --> ENGINE
    API --> ENGINE
    ENGINE <--> CLIENT
    PROMPT --> ENGINE
    ENGINE --> REG
    REG --> BASH & EDIT & IPTOOL
    BASH & IPTOOL --> GUARD
    IPTOOL --> REPL
    REPL <--> KERNEL
    KERNEL <--> BRK
    BRK <--> SUP
    SUP -->|spawns| ENGINE
    ENGINE --> SES
    SES --> FS
```

## Key components

| Component | Responsibility | Source |
| ----------- | --------------- | -------- |
| `RLMEngine` | Agent loop: conversation history, tool dispatch, auto-compaction, recursion depth | `src/rlm/engine.py` |
| `IPythonREPL` | Lifecycle of the persistent Jupyter kernel (ZMQ/IPC transport) | `src/rlm/tools/ipython.py` |
| Tool registry | Active built-in tools: `bash`, `edit`, `ipython` | `src/rlm/tools/registry.py` |
| `Session` | On-disk persistence: `meta.json`, `messages.jsonl`, metrics aggregation | `src/rlm/session.py` |
| Skills system | Discovers and injects callable skills (built-in, installed, MCP) into the kernel | `src/rlm/skills/`, `src/rlm/mcp.py` |
| `GitHistoryGuard` | Blocks broad git-history access (`git log --all`, `--reflog`, …) in shell and Python | `src/rlm/tools/git_block.py` |
| `SessionTreeSupervisor` | Owns the recursion tree, depth semaphores, brokered skill calls | `src/rlm/supervisor.py` |
| Broker | Framed JSON RPC over a unix socket between kernels and the supervisor | `src/rlm/broker.py` |

## The turn loop

`_run_loop()` (`engine.py:368`) is an act–observe cycle. Exactly one built-in tool call is allowed per turn.

```mermaid
sequenceDiagram
    participant E as RLMEngine
    participant L as LLM (client.py)
    participant T as Tool (bash/edit/ipython)
    participant S as Session

    loop for each turn
        E->>L: chat.completions.create(messages, tools)<br/>Idempotency-Key: call_id
        L-->>E: assistant message (+ usage)
        E->>S: log_assistant
        alt no tool calls
            E->>E: stop_reason = "done"
        else more than one tool call
            E->>E: error feedback to all calls
        else one tool call
            E->>T: execute(args, ToolContext)<br/>(asyncio.shield + to_thread)
            T-->>E: ToolOutcome
            E->>S: log_tool_result
        end
        opt prompt_tokens ≥ RLM_SUMMARIZE_AT_TOKENS
            E->>L: compaction call (tool_choice="none")
            L-->>E: handoff summary
            E->>E: messages := [system, summary]<br/>kernel NOT restarted
            E->>S: log compaction event
        end
    end
```

Tool cancellation is cooperative: on interruption the engine interrupts the kernel (`repl.interrupt()`), and recovers by restarting the kernel only if it does not return to idle within 2 seconds.

## Recursion tree

Sub-agents never spawn engines themselves — recursion always goes through the supervisor.

```mermaid
flowchart LR
    subgraph K0["Root kernel (depth 0)"]
        R0["await rlm('sub-task')"]
    end
    R0 -->|rlm.run RPC| BROKER["Broker<br/>(unix socket)"]
    BROKER --> SUP["SessionTreeSupervisor"]
    SUP -->|"capability + scope check<br/>depth & call limits"| CHK{allowed?}
    CHK -->|no| LIM["RLMResult<br/>[depth limit reached]"]
    CHK -->|yes| SEM["depth semaphore<br/>depth_capacities()"]
    SEM --> E1["Child RLMEngine<br/>depth + 1"]
    E1 --> K1[("Child kernel<br/>sub-&lt;uuid8&gt;/")]
    K1 -->|"RLMResult(answer, usage, …)"| R0
```

Guard rails: `RLM_MAX_DEPTH` (default 0 = no sub-agents), `RLM_MAX_SUBAGENT_CALLS` (default 64 per tree), `RLM_MAX_CONCURRENT_SUBAGENTS` (per-depth semaphores so nested calls cannot deadlock).

## Session layout on disk

```mermaid
flowchart TB
    HOME["$RLM_HOME (default ~/.rlm)"] --> S["sessions/&lt;uuid12&gt;/"]
    S --> M["meta.json<br/>id · model · depth · status<br/>usage · metrics · answer_preview"]
    S --> J["messages.jsonl<br/>assistant · tool_result · sub_spawn<br/>compaction · prompt_rollback · done"]
    S --> P["programmatic_tool_calls.jsonl<br/>skill calls from kernel/CLI wrappers"]
    S --> SK["skill modules<br/>(built-in re-exports, MCP proxies)"]
    S --> C1["sub-&lt;uuid8&gt;/<br/>(child session, same layout)"]
    C1 --> C2["sub-&lt;uuid8&gt;/ …"]
```

`meta.json` is written atomically (`.tmp` + rename). `aggregate_child_metrics()` recursively merges child stats into the parent session.
