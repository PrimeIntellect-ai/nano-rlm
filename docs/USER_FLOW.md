# User Flow

How a user (or a host application) drives nano-rlm, from invocation to a finished session.

## Entry flow

The `rlm` console script (`cli.py:main`) has three modes:

```mermaid
flowchart TD
    START["rlm [flags] [prompt]"] --> FLAGS["apply CLI flags<br/>--model → RLM_MODEL<br/>--system-prompt-path → RLM_SYSTEM_PROMPT_PATH<br/>--append-to-system-prompt → RLM_APPEND_TO_SYSTEM_PROMPT"]
    FLAGS --> MODE{mode?}
    MODE -->|"--acp (no prompt)"| ACP["serve_acp()<br/>ACP over stdio"]
    MODE -->|"prompt given"| HEADLESS["rlm.run(prompt)<br/>print answer"]
    MODE -->|"no prompt"| TUI["interactive mode<br/>(TUI placeholder)"]
```

CLI flags simply write environment variables before config resolution, so precedence is always: CLI flag > environment > default.

## Headless run

The most common flow — one prompt, one answer:

```mermaid
sequenceDiagram
    participant U as User
    participant C as rlm CLI
    participant E as RLMEngine
    participant L as LLM
    participant K as IPython kernel
    participant S as Session

    U->>C: rlm "fix the failing test"
    C->>E: rlm.run(prompt)
    E->>S: create session dir<br/>write meta.json
    E->>K: start kernel<br/>inject skills + rlm callable
    loop act–observe turns
        E->>L: messages + tool schemas
        L-->>E: assistant msg + tool call
        E->>K: execute code
        K-->>E: output
        E->>S: log assistant + tool_result
    end
    L-->>E: final answer (no tool calls)
    E->>S: finalize meta.json<br/>(usage, metrics, answer_preview)
    E-->>C: RLMResult
    C-->>U: print answer
```

## Stop reasons

Every run ends with a `stop_reason` recorded in metrics:

```mermaid
flowchart LR
    RUN["run loop"] --> D["done<br/>model stopped calling tools"]
    RUN --> TB["token_budget<br/>RLM_MAX_TOKENS exhausted"]
    RUN --> DL["depth_limit<br/>depth > RLM_MAX_DEPTH"]
    RUN --> RTL["request_too_large<br/>context exceeded model limit<br/>and compaction unavailable/exhausted"]
```

## Recursion flow (user perspective)

With `RLM_MAX_DEPTH ≥ 1`, the agent can decompose work itself:

```mermaid
sequenceDiagram
    participant U as User
    participant P as Parent agent (depth 0)
    participant C1 as Child agent (depth 1)
    participant C2 as Child agent (depth 1)

    U->>P: "audit the repo and summarize"
    P->>P: plan: split by subsystem
    par parallel sub-agents via asyncio.gather
        P->>C1: await rlm("audit src/rlm/tools")
    and
        P->>C2: await rlm("audit src/rlm/engine.py")
    end
    C1-->>P: RLMResult(answer, usage)
    C2-->>P: RLMResult(answer, usage)
    P->>P: synthesize findings
    P-->>U: final answer<br/>(metrics aggregate child sessions)
```

## ACP flow (host applications)

`rlm --acp` speaks the Agent Client Protocol over stdio for editor/agent-host integration:

```mermaid
sequenceDiagram
    participant H as ACP client (editor)
    participant A as RLMACPAgent
    participant E as RLMEngine

    H->>A: initialize
    A-->>H: capabilities + ai.prime.rlm/contract-v1
    H->>A: session/new<br/>_meta: ai.prime.rlm/runtime-v1<br/>(session_id, model, provider, policy, …)
    A->>E: create Session + RLMEngine
    loop prompting
        H->>A: session/prompt (text blocks)
        A->>E: engine.prompt(text)
        E-->>A: answer + usage
        A-->>H: session_update + PromptResponse<br/>_meta: session snapshot
    end
    H->>A: session/close
    A->>E: aclose() (finalize session)
    A-->>H: ai.prime.rlm/session-v1 snapshot<br/>(authoritative, credential-free)
```

ACP sessions bypass environment-variable config entirely: the runtime contract in `_meta` is the full configuration. There is no `session/load` by design — sessions are not resumable over ACP.
