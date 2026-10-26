import asyncio

import httpx
import pytest
from openai import NotFoundError, RateLimitError

from rlm.client import call_with_retries


def status_error(cls, status_code):
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", "https://example.com"),
        text="not found",
    )
    return cls("not found", response=response, body={"error": "not found"})


async def immediate_sleep(_seconds):
    return None


def test_not_found_gets_only_one_quick_retry(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    calls = 0

    async def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise status_error(NotFoundError, 404)

    with pytest.raises(NotFoundError):
        asyncio.run(call_with_retries(fail))
    assert calls == 2


def test_other_transient_errors_keep_existing_retry_budget(monkeypatch):
    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    calls = 0

    async def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise status_error(RateLimitError, 429)

    with pytest.raises(RateLimitError):
        asyncio.run(call_with_retries(fail))
    assert calls == 6
