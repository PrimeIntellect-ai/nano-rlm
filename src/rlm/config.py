"""Validated runtime configuration for RLM engines."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Mapping


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
    parsed = _optional_positive_int(value, "summarize_at_tokens")
    if value not in (None, "") and parsed is None:
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


@dataclass(frozen=True)
class ProviderConfig:
    """Credentials and transport options for one inference provider."""

    base_url: str | None
    api_key: str | None = field(repr=False)
    headers: dict[str, str] = field(default_factory=dict, repr=False)
    max_retries: int = 5

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> ProviderConfig:
        env = os.environ if environ is None else environ
        max_retries = int(env.get("RLM_SDK_MAX_RETRIES", "5"))
        if api_key := env.get("RLM_API_KEY"):
            return cls(env.get("RLM_BASE_URL"), api_key, max_retries=max_retries)
        if api_key := env.get("PRIME_API_KEY"):
            headers = {}
            if team_id := env.get("PRIME_TEAM_ID"):
                headers["X-Prime-Team-ID"] = team_id
            return cls(
                PI_INFERENCE_BASE_URL,
                api_key,
                headers=headers,
                max_retries=max_retries,
            )
        if env.get("OPENAI_API_KEY"):
            return cls(
                env.get("OPENAI_BASE_URL"),
                env["OPENAI_API_KEY"],
                max_retries=max_retries,
            )
        return cls(PI_INFERENCE_BASE_URL, "EMPTY", max_retries=max_retries)


@dataclass(frozen=True)
class InvocationContext:
    """Trusted identity of one engine within a recursive session tree."""

    depth: int = 0

    def __post_init__(self) -> None:
        if self.depth < 0:
            raise ValueError("depth must be non-negative")

    @classmethod
    def from_env(cls, environ: Mapping[str, str] | None = None) -> InvocationContext:
        env = os.environ if environ is None else environ
        return cls(depth=int(env.get("RLM_DEPTH", "0")))

    def child(self) -> InvocationContext:
        return InvocationContext(depth=self.depth + 1)


@dataclass(frozen=True)
class ExecutionPolicy:
    """Resource and context-management policy for one RLM engine."""

    max_depth: int = 0
    exec_timeout: int = 300
    max_output: int = -1
    max_tokens: int | None = None
    summarize_at_tokens: int | None = None
    max_compactions: int | None = None
    max_concurrent_subagents: int = 4
    max_subagent_calls: int = 64

    def __post_init__(self) -> None:
        if self.max_depth < 0:
            raise ValueError("max_depth must be non-negative")
        if self.exec_timeout <= 0:
            raise ValueError("exec_timeout must be positive")
        if self.max_output == 0 or self.max_output < -1:
            raise ValueError("max_output must be positive, or -1 to disable truncation")
        for name in ("max_tokens", "summarize_at_tokens", "max_compactions"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive")
        if self.max_concurrent_subagents <= 0:
            raise ValueError("max_concurrent_subagents must be positive")
        if self.max_concurrent_subagents < self.max_depth:
            raise ValueError("max_concurrent_subagents must be at least max_depth")
        if self.max_subagent_calls <= 0:
            raise ValueError("max_subagent_calls must be positive")


@dataclass(frozen=True)
class RuntimeConfig:
    """Configuration resolved once at an RLM process boundary."""

    model: str
    provider: ProviderConfig
    invocation: InvocationContext
    policy: ExecutionPolicy
    system_prompt_path: str | None = None
    append_to_system_prompt: str | None = None
    skills: tuple[str, ...] = ()
    kernel_env: tuple[tuple[str, str], ...] = field(default=(), repr=False)
    search_api_key: str | None = field(default=None, repr=False)

    @classmethod
    def from_env(
        cls,
        *,
        model: str | None = None,
        summarize_at_tokens: int | None = None,
        system_prompt_path: str | None = None,
        append_to_system_prompt: str | None = None,
        environ: Mapping[str, str] | None = None,
    ) -> RuntimeConfig:
        env = os.environ if environ is None else environ
        max_output = int(env.get("RLM_MAX_OUTPUT", "-1"))
        if max_output == 0:
            raise ValueError(
                "RLM_MAX_OUTPUT must be positive, or -1 to disable truncation"
            )
        raw_summarize = (
            env.get("RLM_SUMMARIZE_AT_TOKENS")
            if summarize_at_tokens is None
            else summarize_at_tokens
        )
        raw_skills = env.get("RLM_SKILLS", "")
        max_depth = int(env.get("RLM_MAX_DEPTH", "0"))
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
            model=model or env.get("RLM_MODEL", "openai/gpt-5-mini"),
            provider=ProviderConfig.from_env(env),
            invocation=InvocationContext.from_env(env),
            policy=ExecutionPolicy(
                max_depth=max_depth,
                exec_timeout=int(env.get("RLM_EXEC_TIMEOUT", "300")),
                max_output=max_output,
                max_tokens=_optional_positive_int(
                    env.get("RLM_MAX_TOKENS"), "RLM_MAX_TOKENS"
                ),
                summarize_at_tokens=_summarize_at_tokens(raw_summarize),
                max_compactions=_optional_positive_int(
                    env.get("RLM_MAX_COMPACTIONS"), "RLM_MAX_COMPACTIONS"
                ),
                max_concurrent_subagents=max_concurrent_subagents,
                max_subagent_calls=_positive_int(
                    env.get("RLM_MAX_SUBAGENT_CALLS", "64"),
                    "RLM_MAX_SUBAGENT_CALLS",
                ),
            ),
            system_prompt_path=system_prompt_path or env.get("RLM_SYSTEM_PROMPT_PATH"),
            append_to_system_prompt=append_to_system_prompt
            or env.get("RLM_APPEND_TO_SYSTEM_PROMPT"),
            skills=tuple(s.strip() for s in raw_skills.split(",") if s.strip()),
            kernel_env=_kernel_env(env.get(KERNEL_ENV_CONFIG_ENV)),
            search_api_key=env.get("SERPER_API_KEY"),
        )
