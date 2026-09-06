# Prompt profiles

Validated per-model system prompts, used via the ACP runtime contract's `system_prompt_path`
(full replacement of the built prompt) — e.g. point it at a file containing the profile.

## `glm_minimal.txt` (~460 chars)

For GLM-5.3-class models with the skills tooling preset and tool-output truncation disabled
(`policy.max_tool_output_bytes` unset/high).

Validated on a 32-task SWE-rebench slice (n=30):

| config | solve | turns | gen tok | reason/turn |
|---|---|---|---|---|
| bash_edit reference (verifiers) | 0.78 | 43.0 | 17.1k | 281 |
| rlm, default prompt (skills preset) | 0.66 | 59.0 | 26.7k | 328 |
| rlm, this profile | 0.70-0.75 | 54.6 | 20.8k | 248 |

Every sentence is ablation-backed: minimal framing (+~0.1 solve vs the default prompt for
GLM — it pays a compliance tax on environment prose), the assign-and-print idiom
(eliminated ~150 wasted `print(out)` follow-up turns, -7 turns/rollout), triple-quoting
(kernel SyntaxErrors), and the kernel-venv fact (ModuleNotFoundError dead-ends).

Kimi-k3-class models should keep the default prompt (they measurably benefit from the full
environment specification; the same prose that costs GLM ~0.1 solve fixed kimi's 2x
reasoning inflation). Prompt fingerprints are per-model; measure before porting.

A deploy branch with these defaults baked in (pin-and-go, not for merge) is maintained at
`deploy/glm53-syngen`.
