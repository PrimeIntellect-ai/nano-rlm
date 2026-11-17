import json

import httpx
import pytest
from openai import APIStatusError, AsyncOpenAI

from rlm import client as model_client
from rlm.client import call_with_retries, model_call_headers, stream_completion
from rlm.compaction import is_context_overflow


def chunk(delta=None, finish_reason=None, **extra):
    return {
        "id": "reply",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "m",
        "choices": []
        if delta is None
        else [{"index": 0, "delta": delta, "finish_reason": finish_reason}],
        **extra,
    }


def sse(events):
    return b"".join(
        b"data: "
        + (event.encode() if isinstance(event, str) else json.dumps(event).encode())
        + b"\n\n"
        for event in events
    )


def connect(handle):
    return AsyncOpenAI(
        api_key="test",
        base_url="http://model.test/v1",
        max_retries=0,
        http_client=httpx.AsyncClient(transport=httpx.MockTransport(handle)),
    )


@pytest.mark.parametrize("finish_reason", ["stop", "length", "tool_calls"])
async def test_stream_preserves_response_fields(finish_reason):
    usage = {
        "prompt_tokens": 11,
        "completion_tokens": 7,
        "total_tokens": 18,
        "prompt_tokens_details": {"cached_tokens": 5},
    }
    events = [
        chunk(
            {
                "role": "assistant",
                "reasoning": "Think ",
                "reasoning_content": "Think ",
                "reasoning_details": [
                    {"type": "reasoning.text", "id": "r", "text": "Think "}
                ],
            }
        ),
        chunk(
            {
                "role": "assistant",
                "reasoning": "more",
                "reasoning_content": "more",
                "reasoning_details": [
                    {"type": "reasoning.text", "id": "r", "text": "more"}
                ],
            }
        ),
        chunk({"content": "Hello", "refusal": "Cannot "}),
        chunk(
            {
                "refusal": "do that",
                "tool_calls": [
                    {
                        "index": 0,
                        "id": "call",
                        "type": "function",
                        "function": {"name": "echo", "arguments": '{"text":'},
                    }
                ],
            }
        ),
        chunk({"tool_calls": [{"index": 0, "function": {"arguments": '"hi"}'}}]}),
        chunk({}, finish_reason),
        chunk(usage=usage),
        "[DONE]",
    ]
    responses = []

    async def handle(request):
        body = json.loads(request.content)
        assert body["stream"] is True
        assert body["stream_options"] == {"include_usage": True}
        payload = b": keepalive\n\n" + sse(events)

        async def pieces():
            for start in range(0, len(payload), 7):
                yield payload[start : start + 7]

        response = httpx.Response(200, content=pieces())
        responses.append(response)
        return response

    async with connect(handle) as client:
        response = await stream_completion(client, model="m", messages=[])
    message = response.choices[0].message
    assert message.role == "assistant"
    assert message.content == "Hello"
    assert message.refusal == "Cannot do that"
    assert message.reasoning == message.reasoning_content == "Think more"
    assert message.reasoning_details == [
        {"type": "reasoning.text", "id": "r", "text": "Think more"}
    ]
    assert message.tool_calls[0].function.arguments == '{"text":"hi"}'
    assert "index" not in message.tool_calls[0].model_dump()
    assert response.choices[0].finish_reason == finish_reason
    assert response.usage.model_dump(exclude_none=True) == usage
    assert responses[0].is_closed


@pytest.mark.parametrize(
    "failure",
    [
        "truncated",
        "timeout",
        "disconnect",
        "error_event",
        "rate_limit",
        "overloaded",
        "unknown_error",
        "error_finish",
    ],
)
async def test_stream_failure_retries_the_whole_attempt(monkeypatch, failure):
    monkeypatch.setattr(model_client, "_RETRY_DELAYS", (0,))
    requests = []
    responses = []

    async def handle(request):
        requests.append(request)
        attempt = len(requests)

        async def pieces():
            if attempt == 1:
                yield sse([chunk({"content": "discard me"})])
                if failure == "timeout":
                    raise httpx.ReadTimeout("stalled")
                if failure == "disconnect":
                    raise httpx.RemoteProtocolError("disconnected")
                errors = {
                    "error_event": {"code": 429},
                    "rate_limit": {"code": "rate_limit_exceeded"},
                    "overloaded": {"type": "overloaded_error"},
                    "unknown_error": {"code": "unknown_provider_error"},
                }
                if failure in errors:
                    yield sse(
                        [{"error": {"message": "failed", **errors[failure]}}, "[DONE]"]
                    )
                if failure == "error_finish":
                    yield sse([chunk({}, "error"), "[DONE]"])
                return
            yield sse([chunk({"content": "complete"}, "stop"), "[DONE]"])

        response = httpx.Response(200, content=pieces())
        responses.append(response)
        return response

    async with connect(handle) as client:
        response = await call_with_retries(
            stream_completion,
            client=client,
            model="m",
            messages=[],
            extra_headers=model_call_headers("request-id", "session-id"),
        )
    assert response.choices[0].message.content == "complete"
    assert len(requests) == 2
    assert [r.headers["Idempotency-Key"] for r in requests] == ["request-id"] * 2
    assert requests[1].headers["x-stainless-retry-count"] == "1"
    assert all(r.is_closed for r in responses)


@pytest.mark.parametrize(
    "error,status,overflow",
    [
        ({"code": 400}, 400, False),
        (
            {"code": "context_length_exceeded", "type": "invalid_request_error"},
            400,
            True,
        ),
        (
            {
                "type": "invalid_request_error",
                "message": "maximum context length is 4096 tokens",
            },
            400,
            True,
        ),
        ({"type": "invalid_request_error"}, 400, False),
        ({"code": "invalid_api_key"}, 401, False),
        ({"type": "permission_error"}, 403, False),
        ({"type": "request_too_large"}, 413, True),
        ({"status_code": "413", "code": "context_length_exceeded"}, 413, True),
        ({"status_code": 401, "code": "context_length_exceeded"}, 401, False),
        ({"code": 200, "type": "invalid_request_error"}, 400, False),
    ],
)
async def test_nonretryable_stream_error_preserves_status(
    monkeypatch, error, status, overflow
):
    monkeypatch.setattr(model_client, "_RETRY_DELAYS", (0,))
    calls = []

    async def handle(request):
        calls.append(request)
        return httpx.Response(
            200,
            content=sse(
                [
                    {"error": {"message": "invalid request", **error}},
                    "[DONE]",
                ]
            ),
        )

    async with connect(handle) as client:
        with pytest.raises(APIStatusError) as raised:
            await call_with_retries(
                stream_completion, client=client, model="m", messages=[]
            )
    assert raised.value.status_code == status
    assert is_context_overflow(raised.value) is overflow
    assert len(calls) == 1
