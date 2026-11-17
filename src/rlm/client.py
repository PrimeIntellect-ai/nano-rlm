"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
from typing import Any, Awaitable, Callable

import certifi
import httpx
from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APIStatusError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from openai.lib.streaming.chat import ChatCompletionStreamState
from openai.types.chat import ChatCompletion
from pydantic import ValidationError

from rlm.config import ProviderConfig
from rlm.semantic import (
    ACP_EXTENSION_HEADER_NAMES,
    AGENT_SESSION_ID_HEADER,
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

_STREAM_ERROR_STATUSES = {
    "context_length_exceeded": 400,
    "invalid_request_error": 400,
    "invalid_api_key": 401,
    "authentication_error": 401,
    "permission_error": 403,
    "not_found_error": 404,
    "request_too_large": 413,
    "rate_limit_exceeded": 429,
    "rate_limit_error": 429,
    "server_error": 500,
    "api_error": 500,
    "overloaded_error": 503,
}


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
        max_retries=provider.max_retries,
        default_headers=provider.headers,
        # Minimal task images may lack a system CA bundle.
        http_client=DefaultAsyncHttpxClient(verify=certifi.where()),
    )


def model_call_headers(request_id: str, session_id: str) -> dict[str, str]:
    """Build transport headers for one idempotent, attributable model call.
    `session_id` is the calling agent's session in the tree: the root's is the ACP
    session, each sub-agent's its own."""
    return {
        IDEMPOTENCY_KEY_HEADER: request_id,
        MODEL_REQUEST_ID_HEADER: request_id,
        AGENT_SESSION_ID_HEADER: session_id,
    }


def _append_reasoning_details(details: list[dict], delta: list[dict]) -> None:
    # Provider reasoning blocks may omit indexes and repeat stable IDs on each delta.
    for detail in delta:
        previous = details[-1] if details else {}
        field = {"reasoning.text": "text", "reasoning.summary": "summary"}.get(
            detail.get("type")
        )
        if (
            field
            and detail.get("type") == previous.get("type")
            and all(
                previous.get(key) is None
                or detail.get(key) is None
                or previous[key] == detail[key]
                for key in ("id", "index", "format")
            )
        ):
            previous[field] = (previous.get(field) or "") + (detail.get(field) or "")
            for key in ("id", "index", "format", "signature"):
                if detail.get(key) is not None:
                    previous[key] = detail[key]
        else:
            details.append(dict(detail))


def _stream_error_status(body: dict) -> int:
    # Explicit HTTP statuses take precedence over symbolic provider codes and types.
    for key in ("status_code", "code"):
        value = str(body.get(key))
        if value.isdigit() and 400 <= int(value) <= 599:
            return int(value)
    for key in ("code", "type"):
        value = body.get(key)
        if isinstance(value, str) and value in _STREAM_ERROR_STATUSES:
            return _STREAM_ERROR_STATUSES[value]
    return 502


async def stream_completion(client: AsyncOpenAI, **request: Any) -> ChatCompletion:
    """Consume one complete model attempt before returning or releasing its slot."""
    stream = await client.chat.completions.create(
        **request, stream=True, stream_options={"include_usage": True}
    )
    state = ChatCompletionStreamState()
    completion = None
    usage = None
    roles: dict[int, str] = {}
    reasoning_details: dict[int, list[dict]] = {}
    try:
        async with stream:
            async for chunk in stream:
                for choice in chunk.choices:
                    if choice.finish_reason == "error":
                        raise APIConnectionError(
                            message=f"Model stream failed: {chunk.model_dump_json()}",
                            request=stream.response.request,
                        )
                    if choice.delta.role:
                        roles[choice.index] = choice.delta.role
                        choice.delta.role = None
                    extras = choice.delta.model_extra or {}
                    details = extras.pop("reasoning_details", None)
                    if details:
                        _append_reasoning_details(
                            reasoning_details.setdefault(choice.index, []), details
                        )
                # No structured-output parsing: length-limited replies belong to compaction.
                state.handle_chunk(chunk)
                completion = state.current_completion_snapshot
                usage = chunk.usage or usage
    except httpx.TransportError as error:
        raise APIConnectionError(
            message=str(error), request=stream.response.request
        ) from error
    except APIError as error:
        if type(error) is not APIError:
            raise
        # The SDK reports HTTP-200 SSE error events as a plain APIError.
        body = error.body if isinstance(error.body, dict) else {}
        status = _stream_error_status(body)
        error_type = (
            RateLimitError
            if status == 429
            else NotFoundError
            if status == 404
            else InternalServerError
            if status >= 500
            else APIStatusError
        )
        raise error_type(
            str(error),
            response=httpx.Response(status, request=stream.response.request),
            body=error.body,
        ) from error
    if (
        completion is None
        or not completion.choices
        or any(choice.finish_reason is None for choice in completion.choices)
    ):
        raise APIConnectionError(
            message="Model stream ended before a completion finished",
            request=stream.response.request,
        )
    completion.usage = usage
    for choice in completion.choices:
        choice.message.role = roles.get(choice.index, "assistant")
        for tool_call in choice.message.tool_calls or []:
            # Delta indexes are transport metadata, not assistant-message fields.
            if tool_call.model_extra is not None:
                tool_call.model_extra.pop("index", None)
        if choice.index in reasoning_details:
            choice.message.reasoning_details = reasoning_details[choice.index]
    return completion


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
                if isinstance(error, (APIConnectionError, ConnectionResetError)):
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
