"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
import logging
from collections.abc import Iterable, Mapping
from typing import Any, Awaitable, Callable
from urllib.parse import urlsplit, urlunsplit

import certifi
from openai import (
    APIConnectionError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    DefaultAsyncHttpxClient,
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

logger = logging.getLogger(__name__)

IDEMPOTENCY_KEY_HEADER = "Idempotency-Key"
RETRY_COUNT_HEADER = "x-stainless-retry-count"
_BODY_PREVIEW_LIMIT = 512
_SAFE_REQUEST_HEADERS = frozenset(
    {
        "accept",
        "content-type",
        IDEMPOTENCY_KEY_HEADER.lower(),
        MODEL_REQUEST_ID_HEADER.lower(),
        RETRY_COUNT_HEADER,
    }
)
_SAFE_RESPONSE_HEADERS = frozenset(
    {
        "cf-ray",
        "content-type",
        "request-id",
        "retry-after",
        "x-correlation-id",
        "x-request-id",
    }
)

_RETRYABLE: tuple[type[BaseException], ...] = (
    APIConnectionError,
    APITimeoutError,
    InternalServerError,
    RateLimitError,
    APIResponseValidationError,
    ValidationError,
    ConnectionResetError,
)

# Widely-spaced delays (seconds) between attempts; total ~5 min wall budget.
_RETRY_DELAYS: tuple[int, ...] = (15, 30, 60, 90, 120)


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


def model_call_headers(request_id: str) -> dict[str, str]:
    """Build transport headers for one idempotent, attributable model call."""
    return {
        IDEMPOTENCY_KEY_HEADER: request_id,
        MODEL_REQUEST_ID_HEADER: request_id,
    }


def _safe_url(url: Any) -> str:
    parts = urlsplit(str(url))
    host = parts.hostname or ""
    if ":" in host:
        host = f"[{host}]"
    if parts.port is not None:
        host = f"{host}:{parts.port}"
    return urlunsplit((parts.scheme, host, parts.path, "", ""))


def _safe_headers(
    headers: Mapping[str, str], allowed: frozenset[str]
) -> dict[str, str]:
    return {name: value for name, value in headers.items() if name.lower() in allowed}


def _strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        if value:
            yield value
    elif isinstance(value, Mapping):
        for nested in value.values():
            yield from _strings(nested)
    elif isinstance(value, (list, tuple)):
        for nested in value:
            yield from _strings(nested)


def _body_preview(text: str, sensitive: Iterable[str]) -> str:
    preview = text[:65536]
    for value in sorted(set(sensitive), key=len, reverse=True):
        preview = preview.replace(value, "<redacted>")
    return preview[:_BODY_PREVIEW_LIMIT]


def _request_sensitive_values(request: Mapping[str, Any], client: Any) -> Iterable[str]:
    for name, value in request.items():
        if name not in {"model", "extra_headers"}:
            yield from _strings(value)
    for name, value in (request.get("extra_headers") or {}).items():
        if name.lower() not in _SAFE_REQUEST_HEADERS:
            yield from _strings(value)
            if isinstance(value, str) and " " in value:
                yield from _strings(value.split(" ", 1)[1])
    yield from _strings(getattr(client, "api_key", None))


def _request_url(error: NotFoundError, client: Any) -> str:
    request = getattr(error.response, "request", None)
    if request is not None:
        return _safe_url(request.url)
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    return _safe_url(f"{base_url}/chat/completions")


def _log_not_found(
    error: NotFoundError,
    *,
    attempt: int,
    request: Mapping[str, Any],
    client: Any,
) -> None:
    request_headers = request.get("extra_headers") or {}
    response_headers = _safe_headers(error.response.headers, _SAFE_RESPONSE_HEADERS)
    request_id = request_headers.get(
        MODEL_REQUEST_ID_HEADER
    ) or error.response.headers.get("x-request-id")
    sensitive = _request_sensitive_values(request, client)
    logger.warning(
        "model request returned 404: attempt=%d url=%s model=%r status=%d "
        "request_id=%s request_headers=%s response_headers=%s body_preview=%r",
        attempt,
        _request_url(error, client),
        request.get("model"),
        error.status_code,
        request_id,
        _safe_headers(request_headers, _SAFE_REQUEST_HEADERS),
        response_headers,
        _body_preview(error.response.text, sensitive),
    )


async def _models_preflight(
    client: Any, *, model: Any, request: Mapping[str, Any], request_id: str | None
) -> None:
    """Diagnose a persistent 404 without changing the request's retry behavior."""
    if client is None:
        return
    base_url = str(getattr(client, "base_url", "")).rstrip("/")
    url = _safe_url(f"{base_url}/models")
    sensitive = list(_request_sensitive_values(request, client))
    try:
        page = await client.models.with_options(max_retries=0).list()
    except Exception as error:  # noqa: BLE001 - diagnostics must preserve the original 404
        response = getattr(error, "response", None)
        text = response.text if response is not None else ""
        logger.warning(
            "model 404 preflight failed: url=%s expected_model=%r status=%s "
            "request_id=%s error_type=%s body_preview=%r",
            url,
            model,
            getattr(error, "status_code", None),
            request_id,
            type(error).__name__,
            _body_preview(text, sensitive),
        )
        return
    model_ids = {getattr(item, "id", None) for item in page.data}
    logger.warning(
        "model 404 preflight: url=%s expected_model=%r status=200 "
        "model_present=%s model_count=%d request_id=%s",
        url,
        model,
        model in model_ids,
        len(model_ids),
        request_id,
    )


async def call_with_retries(
    func: Callable[..., Awaitable[Any]], /, client: Any = None, **kwargs: Any
) -> Any:
    """Call ``func(**kwargs)`` with widely-spaced retries on transient errors.

    A 404 gets one quick retry. Persistent tunnel/registration 404s must instead
    reach the outer rollout retry, which creates a fresh tunnel rather than
    spending many minutes retrying a dead endpoint.
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
        except NotFoundError as error:
            _log_not_found(
                error, attempt=attempt + 1, request=attempt_kwargs, client=client
            )
            if attempt >= 1:
                headers = attempt_kwargs.get("extra_headers") or {}
                response_request_id = error.response.headers.get("x-request-id")
                await _models_preflight(
                    client,
                    model=attempt_kwargs.get("model"),
                    request=attempt_kwargs,
                    request_id=headers.get(MODEL_REQUEST_ID_HEADER)
                    or response_request_id,
                )
                raise
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
