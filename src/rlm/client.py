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

from rlm.config import ProviderConfig
from rlm.semantic import (
    ACP_EXTENSION_HEADER_NAMES,
    MODEL_REQUEST_ID_HEADER,
)
from rlm.types import TokenUsage

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
RETRY_COUNT_HEADER = "x-stainless-retry-count"

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


def make_client(provider: ProviderConfig) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from an explicit provider configuration."""
    reserved = sorted(
        name
        for name in provider.headers
        if name.lower()
        in {
            IDEMPOTENCY_KEY_HEADER.lower(),
            RETRY_COUNT_HEADER,
            *(header.lower() for header in ACP_EXTENSION_HEADER_NAMES),
        }
    )
    if reserved:
        raise ValueError(f"provider headers contain reserved names: {reserved}")
    return AsyncOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        max_retries=provider.max_retries,
        default_headers=provider.headers,
    )


def model_call_headers(request_id: str) -> dict[str, str]:
    """Build transport headers for one idempotent, attributable model call."""
    return {
        IDEMPOTENCY_KEY_HEADER: request_id,
        MODEL_REQUEST_ID_HEADER: request_id,
    }


async def call_with_retries(
    func: Callable[..., Awaitable[Any]], /, **kwargs: Any
) -> Any:
    """Call ``func(**kwargs)`` with widely-spaced retries on transient errors.

    Extends the SDK's retry set with ``NotFoundError`` to ride out intermittent
    tunnel/proxy 404s that the SDK itself does not retry.
    """
    for attempt in range(len(_RETRY_DELAYS) + 1):
        if attempt:
            await asyncio.sleep(_RETRY_DELAYS[attempt - 1])
        attempt_kwargs = kwargs
        if attempt:
            attempt_kwargs = dict(kwargs)
            headers = dict(attempt_kwargs.get("extra_headers") or {})
            headers[RETRY_COUNT_HEADER] = str(attempt)
            attempt_kwargs["extra_headers"] = headers
        try:
            return await func(**attempt_kwargs)
        except _RETRYABLE:
            if attempt == len(_RETRY_DELAYS):
                raise


def extract_usage(response) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )
