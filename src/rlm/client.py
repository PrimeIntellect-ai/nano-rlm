"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
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

from rlm.config import InvocationContext, ProviderConfig
from rlm.types import TokenUsage

RLM_SESSION_ID_HEADER = "X-RLM-Session-ID"
RLM_INVOCATION_ID_HEADER = "X-RLM-Invocation-ID"
RLM_PARENT_INVOCATION_ID_HEADER = "X-RLM-Parent-Invocation-ID"
RLM_SEGMENT_ID_HEADER = "X-RLM-Segment-ID"
RLM_CALL_ID_HEADER = "X-RLM-Call-ID"
RLM_PARENT_CALL_ID_HEADER = "X-RLM-Parent-Call-ID"
RLM_DEPTH_HEADER = "X-RLM-Depth"
RLM_CALL_KIND_HEADER = "X-RLM-Call-Kind"
RLM_LINEAGE_VERSION_HEADER = "X-RLM-Lineage-Version"

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


def resolve_provider() -> tuple[str | None, str | None, dict[str, str]]:
    """Pick the first provider whose key is set: ``(base_url, api_key, headers)``.

    Each provider is a self-contained pair so a key never reaches a base
    URL it wasn't issued for:

    1. **Explicit** — ``RLM_API_KEY`` (pairs with ``RLM_BASE_URL`` if set,
       otherwise SDK default = ``api.openai.com``). Set both for a
       non-OpenAI custom endpoint.
    2. **PI Inference** — ``PRIME_API_KEY`` at PI's base, with
       ``PRIME_TEAM_ID`` forwarded as ``X-Prime-Team-ID``.
    3. **OpenAI** — ``OPENAI_API_KEY`` set: capture ``OPENAI_API_KEY`` and
       ``OPENAI_BASE_URL`` into the trusted provider configuration. Covers
       OpenAI direct and verifiers' rollout tunnel both.

    Falls back to PI + ``"EMPTY"`` so the SDK can't silently inherit
    ``OPENAI_API_KEY`` and ship it to the PI default base.
    """
    provider = ProviderConfig.from_env()
    return provider.base_url, provider.api_key, provider.headers.copy()


def make_client(
    provider: ProviderConfig | None = None,
    invocation: InvocationContext | None = None,
) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from explicit or environment configuration.

    See ``resolve_provider`` for provider precedence. Tags every outbound
    request with ``X-RLM-Depth: <RLM_DEPTH>`` so an interceptor (e.g.
    verifiers' interception server) can distinguish parent-agent calls
    (depth 0) from sub-agent calls (depth >= 1) and decide whether to
    record them in the rollout's trajectory.
    """
    provider = provider or ProviderConfig.from_env()
    invocation = invocation or InvocationContext.from_env()
    reserved = sorted(
        name for name in provider.headers if name.lower().startswith("x-rlm-")
    )
    if reserved:
        raise ValueError(f"provider headers contain reserved names: {reserved}")
    headers = {**provider.headers, RLM_DEPTH_HEADER: str(invocation.depth)}
    return AsyncOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        max_retries=provider.max_retries,
        default_headers=headers,
    )


def model_call_headers(
    *,
    session_id: str,
    invocation_id: str,
    parent_invocation_id: str | None,
    segment_id: str,
    call_id: str,
    parent_call_id: str | None,
    depth: int,
    call_kind: str,
) -> dict[str, str]:
    """Build provenance headers for one logical model call."""
    headers = {
        RLM_LINEAGE_VERSION_HEADER: "1",
        RLM_SESSION_ID_HEADER: session_id,
        RLM_INVOCATION_ID_HEADER: invocation_id,
        RLM_SEGMENT_ID_HEADER: segment_id,
        RLM_CALL_ID_HEADER: call_id,
        RLM_DEPTH_HEADER: str(depth),
        RLM_CALL_KIND_HEADER: call_kind,
    }
    if parent_invocation_id is not None:
        headers[RLM_PARENT_INVOCATION_ID_HEADER] = parent_invocation_id
    if parent_call_id is not None:
        headers[RLM_PARENT_CALL_ID_HEADER] = parent_call_id
    return headers


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
