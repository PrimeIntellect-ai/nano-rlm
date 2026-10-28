import asyncio
import logging
from types import SimpleNamespace

import httpx
import pytest
from openai import NotFoundError, RateLimitError

from rlm.client import call_with_retries


def status_error(
    cls,
    status_code,
    *,
    url="https://example.com/v1/chat/completions",
    text="not found",
    headers=None,
):
    response = httpx.Response(
        status_code,
        request=httpx.Request("POST", url),
        text=text,
        headers=headers,
    )
    return cls("not found", response=response, body={"error": text})


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


class FakeModels:
    def __init__(self, model_ids):
        self.model_ids = model_ids
        self.calls = 0

    def with_options(self, **kwargs):
        self.options = kwargs
        return self

    async def list(self):
        self.calls += 1
        return SimpleNamespace(
            data=[SimpleNamespace(id=model_id) for model_id in self.model_ids]
        )


class FakeClient:
    def __init__(self, model_ids=(), *, api_key="client-secret"):
        self.base_url = "https://example.com/bad/v1"
        self.api_key = api_key
        self.models = FakeModels(model_ids)


def test_transient_not_found_retries_without_preflight(monkeypatch, caplog):
    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    calls = 0
    client = FakeClient(("expected-model",))

    async def attempt(**kwargs):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise status_error(NotFoundError, 404)
        return "ok"

    with caplog.at_level(logging.WARNING, logger="rlm.client"):
        result = asyncio.run(
            call_with_retries(
                attempt,
                client=client,
                model="expected-model",
                extra_headers={"X-ACP-Model-Request-ID": "request-1"},
            )
        )
    assert result == "ok"
    assert calls == 2
    assert client.models.calls == 0
    assert "attempt=1" in caplog.text
    assert "request_id=request-1" in caplog.text


@pytest.mark.parametrize(
    "model_ids,expected",
    [
        (("expected-model",), "model_present=True"),
        (("other-model",), "model_present=False"),
    ],
    ids=["wrong-path", "wrong-model"],
)
def test_persistent_not_found_logs_sanitized_route_and_preflights_models(
    monkeypatch, caplog, model_ids, expected
):
    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    calls = 0
    client = FakeClient(model_ids)

    async def fail(**kwargs):
        nonlocal calls
        calls += 1
        raise status_error(
            NotFoundError,
            404,
            url="https://user:password@example.com/bad/v1/chat/completions?key=url-secret",
            headers={"X-Request-ID": "provider-request", "X-Secret": "header-secret"},
        )

    with caplog.at_level(logging.WARNING, logger="rlm.client"):
        with pytest.raises(NotFoundError):
            asyncio.run(
                call_with_retries(
                    fail,
                    client=client,
                    model="expected-model",
                    extra_headers={
                        "X-ACP-Model-Request-ID": "request-2",
                        "Authorization": "Bearer auth-secret",
                    },
                )
            )
    assert calls == 2
    assert client.models.calls == 1
    assert client.models.options == {"max_retries": 0}
    assert "https://example.com/bad/v1/chat/completions" in caplog.text
    assert "https://example.com/bad/v1/models" in caplog.text
    assert "expected_model='expected-model'" in caplog.text
    assert expected in caplog.text
    assert "request_id=request-2" in caplog.text
    assert "provider-request" in caplog.text
    assert "url-secret" not in caplog.text
    assert "password" not in caplog.text
    assert "header-secret" not in caplog.text
    assert "auth-secret" not in caplog.text


def test_not_found_logs_redact_request_payload_and_credentials(monkeypatch, caplog):
    monkeypatch.setattr(asyncio, "sleep", immediate_sleep)
    secrets = ("prompt-secret", "tool-secret", "client-secret", "auth-secret")
    client = FakeClient(("expected-model",))

    async def fail(**kwargs):
        raise status_error(
            NotFoundError,
            404,
            text="provider echoed prompt-secret tool-secret client-secret auth-secret",
        )

    with caplog.at_level(logging.WARNING, logger="rlm.client"):
        with pytest.raises(NotFoundError):
            asyncio.run(
                call_with_retries(
                    fail,
                    client=client,
                    model="expected-model",
                    messages=[{"role": "user", "content": "prompt-secret"}],
                    tools=[{"function": {"description": "tool-secret"}}],
                    extra_headers={"Authorization": "Bearer auth-secret"},
                )
            )
    for secret in secrets:
        assert secret not in caplog.text
    assert "<redacted>" in caplog.text
