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
    exec_timeout: int = Field(default=300, gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    summarize_at_tokens: int | None = Field(default=256_000, gt=0)
    max_compactions: int | None = Field(default=None, gt=0)
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
    skills: tuple[str, ...] = ()
    kernel_env: tuple[tuple[str, str], ...] = Field(default=(), repr=False)
    search_api_key: str | None = Field(default=None, repr=False)
