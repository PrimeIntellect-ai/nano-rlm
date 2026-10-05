"""Native Anthropic request conversion and transport."""

from copy import deepcopy
import json

from anthropic import AsyncAnthropic
import httpx
from openai import AsyncOpenAI
import pytest

from conftest import make_runtime_config
from rlm.anthropic import _messages
from rlm.client import make_client
from rlm.compaction import discover_threshold
from rlm.config import ExecutionPolicy, ProviderConfig
from rlm.engine import RLMEngine


def _response(content, stop_reason="end_turn"):
    return httpx.Response(
        200,
        json={
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": content,
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 10,
                "cache_read_input_tokens": 900,
                "cache_creation_input_tokens": 90,
                "output_tokens": 50,
            },
        },
    )


@pytest.mark.parametrize(
    "base_url, api_format, native, expected_url",
    [
        ("https://api.anthropic.com/v1/", "auto", True, "https://api.anthropic.com/"),
        ("https://api.anthropic.com", "auto", True, "https://api.anthropic.com/"),
        ("http://native/anthropic/v1", "anthropic", True, "http://native/anthropic/"),
        (None, "anthropic", True, "https://api.anthropic.com/"),
        (
            "https://api.anthropic.com/v1",
            "openai",
            False,
            "https://api.anthropic.com/v1/",
        ),
        ("http://proxy/v1", "auto", False, "http://proxy/v1/"),
    ],
)
async def test_provider_selects_native_anthropic_and_normalizes_url(
    monkeypatch, base_url, api_format, native, expected_url
):
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "http://untrusted-env")
    provider = ProviderConfig(base_url=base_url, api_key="test", api_format=api_format)
    client = make_client(provider)
    try:
        assert isinstance(client, AsyncAnthropic if native else AsyncOpenAI)
        assert str(client.base_url).rstrip("/") == expected_url.rstrip("/")
    finally:
        await client.close()


def test_native_messages_preserve_tool_pairing_and_do_not_mutate_history():
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": None},
        {"role": "user", "content": "continue"},
        {
            "role": "assistant",
            "tool_calls": [
                {
                    "id": "call_1",
                    "type": "function",
                    "function": {
                        "name": "bash",
                        "arguments": '{"command": "echo hello"}',
                    },
                },
            ],
        },
        {"role": "tool", "tool_call_id": "call_1", "content": "hello"},
        {"role": "user", "content": "runtime notice"},
    ]
    original = deepcopy(messages)

    system, conversation = _messages(messages)

    assert system == [{"type": "text", "text": "instructions"}]
    assert conversation == [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "task"},
                {"type": "text", "text": "continue"},
            ],
        },
        {
            "role": "assistant",
            "content": [
                {
                    "type": "tool_use",
                    "id": "call_1",
                    "name": "bash",
                    "input": {"command": "echo hello"},
                },
            ],
        },
        {
            "role": "user",
            "content": [
                {"type": "tool_result", "tool_use_id": "call_1", "content": "hello"},
                {"type": "text", "text": "runtime notice"},
            ],
        },
    ]
    conversation[-1]["content"][0]["content"] = "changed"
    assert messages == original


@pytest.mark.parametrize("mode", ["5m", "1h", "off"])
async def test_native_cache_reaches_wire_and_counts_full_prompt_but_only_new_tokens(
    mode,
):
    requests = []
    content = [
        {"type": "thinking", "thinking": "reasoning", "signature": "signed"},
        {"type": "text", "text": "running"},
        {
            "type": "tool_use",
            "id": "toolu_1",
            "name": "bash",
            "input": {"command": "echo hello"},
        },
    ]

    def respond(request):
        requests.append(request)
        return _response(content, "tool_use")

    provider = ProviderConfig(
        base_url="https://api.anthropic.com/v1",
        api_key="test",
        prompt_cache=mode,
        max_output_tokens=1024,
    )
    messages = [
        {"role": "system", "content": "instructions"},
        {"role": "user", "content": "task"},
    ]
    original = deepcopy(messages)
    async with AsyncAnthropic(
        base_url="https://api.anthropic.com",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        engine = RLMEngine(
            client=client,
            runtime_config=make_runtime_config(
                model="claude-sonnet-4-6", provider=provider
            ),
        )
        engine._active_tool_schemas = []
        response, usage = await engine._call_model(messages)
        assistant = response.choices[0].message
        replay = [
            *messages,
            assistant.model_dump(exclude_none=True),
            {"role": "tool", "tool_call_id": "toolu_1", "content": "hello"},
        ]
        await engine._call_model(replay)

    body = json.loads(requests[0].content)
    assert requests[0].url.path == "/v1/messages"
    assert requests[0].headers["x-api-key"] == "test"
    assert requests[0].headers["anthropic-version"] == "2023-06-01"
    assert (
        requests[0].headers["Idempotency-Key"]
        == requests[0].headers["X-ACP-Model-Request-ID"]
    )
    assert body["max_tokens"] == 1024
    if mode == "off":
        assert "cache_control" not in body
        assert "cache_control" not in body["system"][-1]
    else:
        control = {"type": "ephemeral", **({"ttl": "1h"} if mode == "1h" else {})}
        assert body["cache_control"] == control
        assert body["system"][-1]["cache_control"] == control
    assert messages == original
    assert response.choices[0].finish_reason == "tool_calls"
    assert assistant.content == "running"
    assert assistant.tool_calls[0].id == "toolu_1"
    assert json.loads(assistant.tool_calls[0].function.arguments) == {
        "command": "echo hello"
    }
    assert json.loads(requests[1].content)["messages"][-2]["content"] == content
    assert usage.prompt_tokens == 1000
    assert usage.completion_tokens == 50
    assert response.usage.prompt_tokens_details.cached_tokens == 900
    assert response.usage.prompt_tokens_details.cache_write_tokens == 90
    assert engine._own_new_tokens == 300
    assert engine._total_usage.total == 2100


async def test_native_model_metadata_resolves_alias_context_window(monkeypatch):
    monkeypatch.setattr("rlm.compaction._window_cache", {})
    requests = []

    def respond(request):
        requests.append(request)
        return httpx.Response(
            200,
            json={
                "id": "claude-test-20260101",
                "type": "model",
                "display_name": "Claude Test",
                "created_at": "2026-01-01T00:00:00Z",
                "max_input_tokens": 100000,
            },
        )

    async with AsyncAnthropic(
        base_url="http://native",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        assert await discover_threshold(client, "claude-test") == 100000 - 16384
        assert await discover_threshold(client, "claude-test") == 100000 - 16384
    assert len(requests) == 1
    assert requests[0].url.path == "/v1/models/claude-test"


@pytest.mark.parametrize(
    "overflow_message",
    [
        "prompt is too long: 100001 tokens > 100000 maximum",
        "input length and max_tokens exceed context limit: 95000 + 8192 > 100000",
        "input length and `max_tokens` exceed context limit: 95000 + 8192 > 100000",
    ],
)
async def test_native_retries_and_overflow_compaction_keep_cache_and_request_identity(
    monkeypatch, session, overflow_message
):
    monkeypatch.setattr("rlm.client._RETRY_DELAYS", (0,))
    requests = []

    def respond(request):
        requests.append(request)
        if len(requests) == 1:
            return httpx.Response(
                429,
                json={
                    "type": "error",
                    "error": {"type": "rate_limit_error", "message": "retry"},
                },
            )
        if len(requests) == 2:
            return _response(
                [
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "bash",
                        "input": {"command": "printf ready"},
                    }
                ],
                "tool_use",
            )
        if len(requests) == 3:
            return httpx.Response(
                400,
                json={
                    "type": "error",
                    "error": {
                        "type": "invalid_request_error",
                        "message": overflow_message,
                    },
                },
            )
        return _response(
            [{"type": "text", "text": "summary" if len(requests) == 4 else "done"}]
        )

    provider = ProviderConfig(
        base_url="http://native", api_key="test", api_format="anthropic", max_retries=0
    )
    async with AsyncAnthropic(
        base_url="http://native",
        api_key="test",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(respond)),
    ) as client:
        engine = RLMEngine(
            client=client,
            session=session,
            runtime_config=make_runtime_config(
                model="claude-test",
                provider=provider,
                builtin_tools=("bash",),
                policy=ExecutionPolicy(
                    max_depth=0, compaction=True, summarize_at_tokens=90000
                ),
            ),
        )
        try:
            result = await engine.prompt("do work")
        finally:
            await engine.aclose()

    assert result.answer == "done"
    assert len(requests) == 5
    assert (
        requests[0].headers["Idempotency-Key"] == requests[1].headers["Idempotency-Key"]
    )
    assert requests[1].headers["x-stainless-retry-count"] == "1"
    assert requests[0].content == requests[1].content
    assert len({request.headers["Idempotency-Key"] for request in requests}) == 4
    bodies = [json.loads(request.content) for request in requests]
    assert all(body["cache_control"] == {"type": "ephemeral"} for body in bodies)
    assert bodies[1]["tool_choice"] == {
        "type": "auto",
        "disable_parallel_tool_use": True,
    }
    assert bodies[3]["tool_choice"] == {"type": "none"}
    assert bodies[3]["system"] == bodies[4]["system"]
    assert engine._metrics.num_compactions == 1
    assert engine._own_new_tokens == 450
    assert engine._total_usage.prompt_tokens == 3000
    assert all(
        isinstance(message.get("content", ""), str) for message in session.messages
    )
