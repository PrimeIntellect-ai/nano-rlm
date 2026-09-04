"""Validated runtime configuration for RLM engines.

Configuration enters an rlm process exactly once, through the versioned ACP
runtime contract (``ai.prime.rlm/runtime-v1``); recursive children inherit it
in-memory via ``model_copy``. There is no environment-variable resolution.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from rlm.semantic import ACP_EXTENSION_HEADER_NAMES


class _ConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class ProviderConfig(_ConfigModel):
    """Credentials and transport options for one inference provider."""

    base_url: str | None
    api_key: str = Field(min_length=1, repr=False)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)
    max_retries: int = Field(default=5, ge=0)

    @field_validator("headers")
    @classmethod
    def _reserve_transport_headers(cls, headers: dict[str, str]) -> dict[str, str]:
        reserved_names = {
            "idempotency-key",
            "x-stainless-retry-count",
            *(header.lower() for header in ACP_EXTENSION_HEADER_NAMES),
        }
        reserved = sorted(name for name in headers if name.lower() in reserved_names)
        if reserved:
            raise ValueError(f"provider headers contain reserved names: {reserved}")
        return headers


class InvocationContext(_ConfigModel):
    """Trusted identity of one engine within a recursive session tree."""

    depth: int = Field(default=0, ge=0)

    def child(self) -> InvocationContext:
        return InvocationContext(depth=self.depth + 1)


class ExecutionPolicy(_ConfigModel):
    """Resource and context-management policy for one RLM engine."""

    max_depth: int = Field(default=1, ge=0)
    max_total_turns: int | None = Field(default=None, gt=0)
    """Tree-total turn budget (one turn = one work-loop model call, any engine; compaction
    calls don't count). Once reached, every engine stops before its next model call
    (stop_reason=max_total_turns). None = uncapped."""
    max_total_tokens: int | None = Field(default=1_000_000, gt=0)
    """Tree-total budget of NEW tokens across the whole recursive session tree, live: each
    model call contributes its completion plus uncached prompt tokens (the cached context
    prefix re-billed every call is not new work). Once reached, every engine stops before
    its next model call (stop_reason=max_total_tokens) and no further sub-agents are
    spawned. This budget is the default terminator (compaction never runs out of context,
    so without it a stuck session would run forever). None = unbounded."""
    max_tool_output_bytes: int | None = Field(default=None, gt=0)
    """Byte budget for a single tool result entering the conversation (middle truncation,
    head + tail, with a warning naming the original size). None = the built-in 20KB
    default; an explicit value overrides it in either direction."""
    exec_timeout: int = Field(default=300, gt=0)
    """Active execution budget for one IPython cell. Direct broker waits and gathers made
    only of broker calls do not consume it; CPU, ordinary waits, and mixed work do."""
    max_tokens: int | None = Field(default=None, gt=0)
    compaction: bool = True
    """Compact the context once it outgrows ``summarize_at_tokens`` (and recover from
    provider context-overflow errors by checkpointing). On by default; set False to
    let an overflowing session fail instead."""
    summarize_at_tokens: int | None = Field(default=None, gt=0)
    """Compaction threshold. None = auto-discover from the provider's advertised
    context window (~16k tokens of headroom); when the provider advertises no
    window, compaction stays overflow-reactive only."""
    max_compactions: int | None = Field(default=None, gt=0)
    """Compactions per session before the engine stops compacting. None (default) =
    unlimited: every compaction cycle itself spends new tokens, so ``max_total_tokens``
    still bounds the session."""
    max_compaction_attempts: int = Field(default=5, gt=0)
    """Summary-generation attempts within one compaction cycle."""
    max_concurrent_subagents: int = Field(default=4, gt=0)
    max_subagent_calls: int = Field(default=64, gt=0)
    allow_git: bool = False

    @model_validator(mode="after")
    def _validate_concurrency(self) -> Self:
        if self.max_concurrent_subagents < self.max_depth:
            raise ValueError("max_concurrent_subagents must be at least max_depth")
        return self


class RuntimeConfig(_ConfigModel):
    """Configuration resolved once at an RLM process boundary."""

    model: str = Field(min_length=1)
    provider: ProviderConfig
    invocation: InvocationContext
    policy: ExecutionPolicy
    system_prompt_path: str | None = None
    append_to_system_prompt: str | None = None
    subagent_append_to_system_prompt: str | None = None
    leaf_append_to_system_prompt: str | None = None
    skills: tuple[str, ...] = ()
    kernel_env: tuple[tuple[str, str], ...] = Field(default=(), repr=False)
    search_api_key: str | None = Field(default=None, repr=False)

    @property
    def resolved_append_to_system_prompt(self) -> str | None:
        """The append for this engine's role in the session tree.

        root (depth 0)                            -> append_to_system_prompt
        node (depth >= 1, can still recurse)      -> subagent_append_to_system_prompt
        leaf (depth == max_depth, cannot recurse) -> leaf_append_to_system_prompt

        Each tier falls back to the next-more-general one (leaf -> subagent -> root),
        so any unset append preserves the prior single-append behavior. Mirrors the
        allow_recursion gating that build_system_prompt uses for the built-in rlm hint.
        """
        if self.invocation.depth == 0:
            return self.append_to_system_prompt
        if (
            self.invocation.depth >= self.policy.max_depth
            and self.leaf_append_to_system_prompt is not None
        ):
            return self.leaf_append_to_system_prompt
        if self.subagent_append_to_system_prompt is not None:
            return self.subagent_append_to_system_prompt
        return self.append_to_system_prompt
