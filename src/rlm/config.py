"""Validated runtime configuration for RLM engines."""

from __future__ import annotations

import json
import os
from typing import Mapping

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from typing_extensions import Self

from rlm.lineage import LINEAGE_HEADER_NAMES
from rlm.tools.registry import preset_skills


PI_INFERENCE_BASE_URL = "https://api.pinference.ai/api/v1"
KERNEL_ENV_CONFIG_ENV = "RLM_KERNEL_ENV"


def _optional_positive_int(value: str | int | None, name: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(f"{name} must be an int")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be an int (got {value!r})") from exc
    return parsed if parsed > 0 else None


def _positive_int(value: str | int, name: str) -> int:
    parsed = _optional_positive_int(value, name)
    if parsed is None:
        raise ValueError(f"{name} must be positive")
    return parsed


def _summarize_at_tokens(value: str | int | None) -> int | None:
    """Unset -> the 256k default; "" or "0" -> disabled; else a positive threshold."""
    if value is None:
        return 256_000
    if value in ("", "0", 0):
        return None
    parsed = _optional_positive_int(value, "summarize_at_tokens")
    if parsed is None:
        raise ValueError(f"summarize_at_tokens must be positive (got {value})")
    return parsed


def _kernel_env(value: str | None) -> tuple[tuple[str, str], ...]:
    if not value:
        return ()
    parsed = json.loads(value)
    if not isinstance(parsed, dict) or not all(
        isinstance(key, str) and isinstance(item, str) for key, item in parsed.items()
    ):
        raise ValueError(f"{KERNEL_ENV_CONFIG_ENV} must be a JSON object of strings")
    return tuple(parsed.items())


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
            *(header.lower() for header in LINEAGE_HEADER_NAMES),
        }
        reserved = sorted(name for name in headers if name.lower() in reserved_names)
        if reserved:
            raise ValueError(f"provider headers contain reserved names: {reserved}")
        return headers

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProviderConfig:
        env = os.environ if environ is None else environ
        max_retries = int(env.get("RLM_SDK_MAX_RETRIES", "5"))
        if api_key := env.get("RLM_API_KEY"):
            return cls(
                base_url=env.get("RLM_BASE_URL"),
                api_key=api_key,
                max_retries=max_retries,
            )
        if api_key := env.get("PRIME_API_KEY"):
            headers = {}
            if team_id := env.get("PRIME_TEAM_ID"):
                headers["X-Prime-Team-ID"] = team_id
            return cls(
                base_url=PI_INFERENCE_BASE_URL,
                api_key=api_key,
                headers=headers,
                max_retries=max_retries,
            )
        if env.get("OPENAI_API_KEY"):
            return cls(
                base_url=env.get("OPENAI_BASE_URL"),
                api_key=env["OPENAI_API_KEY"],
                max_retries=max_retries,
            )
        return cls(
            base_url=PI_INFERENCE_BASE_URL,
            api_key="EMPTY",
            max_retries=max_retries,
        )


class InvocationContext(_ConfigModel):
    """Trusted identity of one engine within a recursive session tree."""

    depth: int = Field(default=0, ge=0)

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> InvocationContext:
        env = os.environ if environ is None else environ
        return cls(depth=int(env.get("RLM_DEPTH", "0")))

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

    @classmethod
    def from_env(
        cls,
        *,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        env = os.environ if environ is None else environ
        raw_skills = env.get("RLM_SKILLS")
        max_depth = int(env.get("RLM_MAX_DEPTH", "1"))
        default_concurrency = max(4, max_depth)
        max_concurrent_subagents = _positive_int(
            env.get("RLM_MAX_CONCURRENT_SUBAGENTS", str(default_concurrency)),
            "RLM_MAX_CONCURRENT_SUBAGENTS",
        )
        if max_depth > max_concurrent_subagents:
            raise ValueError(
                "RLM_MAX_CONCURRENT_SUBAGENTS must be at least RLM_MAX_DEPTH"
            )
        return cls(
            model=env.get("RLM_MODEL", "openai/gpt-5-mini"),
            provider=ProviderConfig.from_env(env),
            invocation=InvocationContext.from_env(env),
            policy=ExecutionPolicy(
                max_depth=max_depth,
                exec_timeout=int(env.get("RLM_EXEC_TIMEOUT", "300")),
                max_tokens=_optional_positive_int(
                    env.get("RLM_MAX_TOKENS"), "RLM_MAX_TOKENS"
                ),
                summarize_at_tokens=_summarize_at_tokens(
                    env.get("RLM_SUMMARIZE_AT_TOKENS")
                ),
                max_compactions=_optional_positive_int(
                    env.get("RLM_MAX_COMPACTIONS"), "RLM_MAX_COMPACTIONS"
                ),
                max_concurrent_subagents=max_concurrent_subagents,
                max_subagent_calls=_positive_int(
                    env.get("RLM_MAX_SUBAGENT_CALLS", "64"),
                    "RLM_MAX_SUBAGENT_CALLS",
                ),
                allow_git=env.get("RLM_ALLOW_GIT") == "1",
            ),
            system_prompt_path=env.get("RLM_SYSTEM_PROMPT_PATH"),
            append_to_system_prompt=env.get("RLM_APPEND_TO_SYSTEM_PROMPT"),
            subagent_append_to_system_prompt=env.get(
                "RLM_SUBAGENT_APPEND_TO_SYSTEM_PROMPT"
            ),
            leaf_append_to_system_prompt=env.get("RLM_LEAF_APPEND_TO_SYSTEM_PROMPT"),
            skills=(
                tuple(s.strip() for s in raw_skills.split(",") if s.strip())
                if raw_skills is not None
                else preset_skills()
            ),
            kernel_env=_kernel_env(env.get(KERNEL_ENV_CONFIG_ENV)),
            search_api_key=env.get("SERPER_API_KEY"),
        )
