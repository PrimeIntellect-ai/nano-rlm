"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit

import anthropic
import certifi
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from pydantic import ValidationError

from rlm.config import ProviderConfig
from rlm.anthropic import create_anthropic_completion
from rlm.semantic import (
    ACP_EXTENSION_HEADER_NAMES,
    MODEL_REQUEST_ID_HEADER,
)
from rlm.types import TokenUsage

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
RETRY_COUNT_HEADER = "x-stainless-retry-count"
ModelClient = AsyncOpenAI | anthropic.AsyncAnthropic
MODEL_STATUS_ERRORS = (APIStatusError, anthropic.APIStatusError)

_RETRYABLE: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    NotFoundError,
    RateLimitError,
    APIResponseValidationError,
    ValidationError,
    ConnectionResetError,
    anthropic.APIConnectionError,
    anthropic.APITimeoutError,
    anthropic.InternalServerError,
    anthropic.NotFoundError,
    anthropic.RateLimitError,
    anthropic.APIResponseValidationError,
)

# Widely-spaced delays (seconds) between attempts; total ~5 min wall budget.
_RETRY_DELAYS: tuple[int, ...] = (15, 30, 60, 90, 120)


class ModelTransportError(Exception):
    """A model connection failed after its transport retries were exhausted."""


def make_client(provider: ProviderConfig) -> ModelClient:
    """Create a provider client from an explicit configuration."""
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
    if provider.api_format == "anthropic" or (
        provider.api_format == "auto"
        and urlsplit(provider.base_url or "").hostname == "api.anthropic.com"
    ):
        base_url = provider.base_url or "https://api.anthropic.com"
        # The Anthropic SDK includes /v1 in each resource path.
        if base_url and base_url.rstrip("/").endswith("/v1"):
            base_url = base_url.rstrip("/")[:-3]
        return anthropic.AsyncAnthropic(
            base_url=base_url,
            api_key=provider.api_key,
            max_retries=provider.max_retries,
            default_headers=provider.headers,
            http_client=anthropic.DefaultAsyncHttpxClient(verify=certifi.where()),
        )
    return AsyncOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        max_retries=provider.max_retries,
        default_headers=provider.headers,
        # Minimal task images may lack a system CA bundle.
        http_client=DefaultAsyncHttpxClient(verify=certifi.where()),
    )


def model_call_headers(request_id: str) -> dict[str, str]:
    """Build transport headers for one idempotent, attributable model call."""
    return {
        IDEMPOTENCY_KEY_HEADER: request_id,
        MODEL_REQUEST_ID_HEADER: request_id,
    }


async def create_completion(
    *, client: ModelClient, provider: ProviderConfig, **request: Any
) -> Any:
    """Complete one model request using the engine's chat-message representation."""
    if isinstance(client, anthropic.AsyncAnthropic):
        return await create_anthropic_completion(client, provider=provider, **request)
    return await client.chat.completions.create(**request)


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
        except _RETRYABLE as error:
            if attempt == len(_RETRY_DELAYS):
                if isinstance(
                    error,
                    (
                        APIConnectionError,
                        anthropic.APIConnectionError,
                        ConnectionResetError,
                    ),
                ):
                    raise ModelTransportError(
                        f"{type(error).__name__}: {error}"
                    ) from error
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
