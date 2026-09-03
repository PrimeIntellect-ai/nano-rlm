# Prompt profiles

Validated per-model system prompts for `RLM_SYSTEM_PROMPT` (full replacement of the built prompt).

## `glm_minimal.txt` (~460 chars)

For GLM-5.3-class models with `RLM_TOOLING=skills` and the tool-output cap disabled.

Validated on the 32-task SWE-rebench slice (final n=30):

| config | solve | turns | gen tok | reason/turn |
|---|---|---|---|---|
| bash_edit reference (verifiers 0.3.0) | 0.78 | 43.0 | 17.1k | 281 |
| rlm, default prompt (skills preset) | 0.66 | 59.0 | 26.7k | 328 |
| rlm, this profile | 0.70-0.75 | 54.6 | 20.8k | 248 |

Every sentence is ablation-backed: minimal framing (+~0.1 solve vs the default prompt for
GLM — it pays a compliance tax on environment prose), the assign-and-print idiom
(eliminated ~150 wasted `print(out)` follow-up turns, -7 turns/rollout), triple-quoting
(kernel SyntaxErrors), and the kernel-venv fact (ModuleNotFoundError dead-ends).

Usage (verifiers):

```toml
[env.agent.harness]
id = "rlm"

[env.agent.harness.env]
RLM_TOOLING = "skills"
RLM_TOOL_OUTPUT_MAX_BYTES = "0"
RLM_SYSTEM_PROMPT = "<contents of glm_minimal.txt>"
```

Kimi-k3-class models should keep the default prompt (they measurably benefit from the full
environment specification; the same prose that costs GLM ~0.1 solve fixed kimi's 2x
reasoning inflation). Prompt fingerprints are per-model; measure before porting.
