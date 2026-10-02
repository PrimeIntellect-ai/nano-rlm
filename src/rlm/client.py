"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
import math
import random
import time
from email.utils import parsedate_to_datetime
from typing import Any, Awaitable, Callable

import certifi
from openai import (
    APIConnectionError,
    APIStatusError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
)

from rlm.config import ProviderConfig
from rlm.semantic import (
    ACP_EXTENSION_HEADER_NAMES,
    MODEL_REQUEST_ID_HEADER,
)
from rlm.types import TokenUsage

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
RETRY_COUNT_HEADER = "x-stainless-retry-count"

_RETRY_DELAYS = (0.5, 1, 2, 4, 8)


def _is_tunnel_unavailable(error: BaseException) -> bool:
    return (
        isinstance(error, APIStatusError)
        and error.status_code == 404
        and "text/html" in error.response.headers.get("content-type", "")
        and "Tunnel not found or no longer active." in error.response.text
    )


def _retry_delay(error: Exception, attempt: int) -> float | None:
    if isinstance(error, APIStatusError):
        headers = error.response.headers
        if headers.get("x-should-retry") == "false":
            return None
        if headers.get("x-should-retry") != "true" and not (
            _is_tunnel_unavailable(error)
            or error.status_code in (408, 409, 429)
            or error.status_code >= 500
        ):
            return None
        try:
            if "retry-after-ms" in headers:
                delay = float(headers["retry-after-ms"]) / 1000
            else:
                value = headers.get("retry-after", "")
                try:
                    delay = float(value)
                except ValueError:
                    delay = parsedate_to_datetime(value).timestamp() - time.time()
            if math.isfinite(delay) and delay >= 0:
                return delay
        except (ValueError, TypeError, OverflowError):
            pass
    elif not isinstance(error, (APIConnectionError, ConnectionResetError)):
        return None
    return _RETRY_DELAYS[min(attempt, len(_RETRY_DELAYS) - 1)] * random.uniform(0.75, 1)


class ModelTransportError(Exception):
    """A model connection failed after its transport retries were exhausted."""


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
        max_retries=0,
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


async def call_with_retries(
    func: Callable[..., Awaitable[Any]], /, *, max_retries: int = 5, **kwargs: Any
) -> Any:
    """Retry one model request without replaying completed turns or tools."""
    for attempt in range(max_retries + 1):
        attempt_kwargs = dict(kwargs)
        headers = dict(attempt_kwargs.get("extra_headers") or {})
        headers[RETRY_COUNT_HEADER] = str(attempt)
        attempt_kwargs["extra_headers"] = headers
        try:
            return await func(**attempt_kwargs)
        except (APIStatusError, APIConnectionError, ConnectionResetError) as error:
            delay = _retry_delay(error, attempt)
            if attempt == max_retries or delay is None:
                if isinstance(
                    error, (APIConnectionError, ConnectionResetError)
                ) or _is_tunnel_unavailable(error):
                    raise ModelTransportError(
                        f"{type(error).__name__}: {error}"
                    ) from error
                raise
            await asyncio.sleep(delay)


def extract_usage(response) -> TokenUsage:
    """Extract token usage from an API response."""
    usage = response.usage
    if usage is None:
        return TokenUsage()
    return TokenUsage(
        prompt_tokens=usage.prompt_tokens or 0,
        completion_tokens=usage.completion_tokens or 0,
    )
