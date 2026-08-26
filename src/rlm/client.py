"""Thin LLM client wrapper. Extracts token usage from responses."""

import asyncio
import logging
import re
from collections.abc import Mapping
from typing import Any, Awaitable, Callable

from openai import (
    APIConnectionError,
    APIError,
    APIResponseValidationError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    NotFoundError,
    RateLimitError,
)
from pydantic import ValidationError

from rlm.config import ProviderConfig
from rlm.types import TokenUsage

logger = logging.getLogger(__name__)

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

CONTEXT_WINDOW_FIELDS = (
    "max_model_len",
    "context_length",
    "context_window",
    "max_context_length",
)
CONTEXT_WINDOW_PATTERNS = (
    re.compile(
        r"(?:maximum|max(?:imum)?)[^.\n]{0,40}(?:context length|context window)"
        r"[^\d]{0,20}([\d,]+)",
        re.IGNORECASE,
    ),
    re.compile(
        r"[\"']?(?:max_model_len|context_length)[\"']?\s*[:=]\s*([\d,]+)",
        re.IGNORECASE,
    ),
)

_context_window_cache: dict[tuple[str, str], int | None] = {}
_context_window_lock = asyncio.Lock()


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


def make_client(provider: ProviderConfig | None = None) -> AsyncOpenAI:
    """Create an AsyncOpenAI client from explicit or environment configuration."""
    provider = provider or ProviderConfig.from_env()
    reserved = sorted(
        name
        for name in provider.headers
        if name.lower() in {IDEMPOTENCY_KEY_HEADER.lower(), RETRY_COUNT_HEADER}
    )
    if reserved:
        raise ValueError(f"provider headers contain reserved names: {reserved}")
    return AsyncOpenAI(
        base_url=provider.base_url,
        api_key=provider.api_key,
        max_retries=provider.max_retries,
        default_headers=provider.headers,
    )


def model_call_headers(call_id: str) -> dict[str, str]:
    """Build transport headers for one idempotent model call."""
    return {IDEMPOTENCY_KEY_HEADER: call_id}


def model_context_window(payload: Mapping[str, Any], model: str) -> int | None:
    """Read a context-window extension from the requested model card."""
    for card in payload.get("data") or []:
        if not isinstance(card, Mapping) or card.get("id") != model:
            continue
        for field in CONTEXT_WINDOW_FIELDS:
            value = card.get(field)
            if isinstance(value, int) and not isinstance(value, bool) and value > 0:
                return value
        break
    return None


def context_window_from_error(error: BaseException) -> int | None:
    """Extract a context window from common provider overflow diagnostics."""
    details = f"{error} {getattr(error, 'body', '') or ''}"
    for pattern in CONTEXT_WINDOW_PATTERNS:
        match = pattern.search(details)
        if match:
            return int(match.group(1).replace(",", ""))
    return None


def compaction_threshold(context_window: int) -> int:
    """Reserve ten percent of the model context for checkpointing."""
    return max(1, context_window * 9 // 10)


async def resolve_compaction_threshold(client: AsyncOpenAI, model: str) -> int | None:
    """Discover a proactive compaction threshold from model metadata."""
    key = (str(getattr(client, "base_url", "")), model)
    if key in _context_window_cache:
        window = _context_window_cache[key]
        return compaction_threshold(window) if window is not None else None

    async with _context_window_lock:
        if key not in _context_window_cache:
            try:
                payload = await client.get("/models", cast_to=dict)
                _context_window_cache[key] = model_context_window(payload, model)
            except (APIError, AttributeError) as error:
                logger.debug("model context-window lookup failed: %s", error)
                _context_window_cache[key] = None

    window = _context_window_cache[key]
    return compaction_threshold(window) if window is not None else None


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
