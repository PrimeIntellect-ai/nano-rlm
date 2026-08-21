"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
import os
from typing import Any, Awaitable, Callable

from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from pydantic import ValidationError

from rlm.types import TokenUsage

_RETRYABLE: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    APIResponseValidationError,
    ValidationError,
    ConnectionResetError,
)

# Widely-spaced delays (seconds) between attempts; total ~5 min wall budget.
_RETRY_DELAYS: tuple[int, ...] = (15, 30, 60, 90, 120)


PI_INFERENCE_BASE_URL = "https://api.pinference.ai/api/v1"


def resolve_provider() -> tuple[str | None, str | None, dict[str, str]]:
    """Pick the first provider whose key is set: ``(base_url, api_key, headers)``.

    Each provider is a self-contained pair so a key never reaches a base
    URL it wasn't issued for:

    1. **Explicit** — ``RLM_API_KEY`` (pairs with ``RLM_BASE_URL`` if set,
       otherwise SDK default = ``api.openai.com``). Set both for a
       non-OpenAI custom endpoint.
    2. **PI Inference** — ``PRIME_API_KEY`` at PI's base, with
       ``PRIME_TEAM_ID`` forwarded as ``X-Prime-Team-ID``.
    3. **OpenAI** — ``OPENAI_API_KEY`` set: delegate to AsyncOpenAI's
       native env handling (``OPENAI_API_KEY`` + ``OPENAI_BASE_URL``).
       Covers OpenAI direct and verifiers' rollout tunnel both.

    Falls back to PI + ``"EMPTY"`` so the SDK can't silently inherit
    ``OPENAI_API_KEY`` and ship it to the PI default base.
    """
    if api_key := os.environ.get("RLM_API_KEY"):
        return os.environ.get("RLM_BASE_URL"), api_key, {}
    if api_key := os.environ.get("PRIME_API_KEY"):
        headers: dict[str, str] = {}
        if team_id := os.environ.get("PRIME_TEAM_ID"):
            headers["X-Prime-Team-ID"] = team_id
        return PI_INFERENCE_BASE_URL, api_key, headers
    if os.environ.get("OPENAI_API_KEY"):
        return None, None, {}
    return PI_INFERENCE_BASE_URL, "EMPTY", {}


def make_client(depth: int | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from environment variables.

    See ``resolve_provider`` for provider precedence. Tags every outbound
    request with ``X-RLM-Depth: <RLM_DEPTH>`` so an interceptor (e.g.
    verifiers' interception server) can distinguish parent-agent calls
    (depth 0) from sub-agent calls (depth >= 1) and decide whether to
    record them in the rollout's trajectory.
    """
    base_url, api_key, extra_headers = resolve_provider()
    rlm_depth = os.environ.get("RLM_DEPTH", "0") if depth is None else str(depth)
    headers = {"X-RLM-Depth": rlm_depth, **extra_headers}
    return AsyncOpenAI(
        base_url=base_url,
        api_key=api_key,
        max_retries=int(os.environ.get("RLM_SDK_MAX_RETRIES", 5)),
        default_headers=headers,
    )


async def call_with_retries(
    func: Callable[..., Awaitable[Any]], /, **kwargs: Any
) -> Any:
    """Call ``func(**kwargs)`` with widely-spaced retries on transient errors.

    Extends the SDK's retry set with ``NotFoundError`` to ride out intermittent
    tunnel/proxy 404s that the SDK itself does not retry.
    """
    for delay in _RETRY_DELAYS:
        try:
            return await func(**kwargs)
        except _RETRYABLE:
            await asyncio.sleep(delay)
    return await func(**kwargs)


def extract_usage(response) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )
