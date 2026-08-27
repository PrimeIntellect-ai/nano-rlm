from __future__ import annotations

from copy import deepcopy
from typing import Any

import httpx
from openai import BadRequestError
import pytest

from conftest import (
    DummyChoice,
    DummyClient,
    DummyMessage,
    DummyResponse,
    DummyToolCall,
    DummyUsage,
)
from rlm.compaction import DROPPED_TOOL_RESULT, context_error
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor


def _response(
    message: DummyMessage,
    *,
    prompt_tokens: int = 1,
    completion_tokens: int = 1,
    finish_reason: str = "stop",
) -> DummyResponse:
    return DummyResponse(
        choices=[DummyChoice(message=message, finish_reason=finish_reason)],
        usage=DummyUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        ),
    )


def _overflow() -> BadRequestError:
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://interceptor/v1/chat/completions"),
    )
    return BadRequestError(
        "This model's maximum context length is 4096 tokens.",
        response=response,
        body={
            "error": {"message": "This model's maximum context length is 4096 tokens."}
        },
    )


def test_threshold_is_learned_from_provider_error():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://interceptor/v1/chat/completions"),
    )
    error = BadRequestError(
        "maximum context length is 32,768 tokens",
        response=response,
        body={"error": {"message": "maximum context length is 32,768 tokens"}},
    )

    assert context_error(error) == (True, 16_384)


class _ScriptedClient(DummyClient):
    def __init__(
        self,
        actions: list[DummyResponse | BaseException],
        *,
        max_model_len: int | None = None,
    ):
        super().__init__([])
        self.actions = list(actions)
        self.max_model_len = max_model_len
        self.base_url = f"http://scripted-{id(self)}"

    async def get(self, path: str, **kwargs: Any) -> dict:
        assert path == "/models"
        card = {"id": "test-model"}
        if self.max_model_len is not None:
            card["max_model_len"] = self.max_model_len
        return {"data": [card]}

    async def create(self, **kwargs: Any) -> DummyResponse:
        self.calls.append(deepcopy(kwargs))
        if not self.actions:
            raise AssertionError("script exhausted")
        action = self.actions.pop(0)
        if isinstance(action, BaseException):
            raise action
        return action


def _config(
    *,
    max_depth: int = 0,
    summarize_at_tokens: int | None = None,
    compaction: bool = True,
):
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url=None, api_key="test-key"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            max_depth=max_depth,
            max_concurrent_subagents=max(4, max_depth),
            compaction=compaction,
            summarize_at_tokens=summarize_at_tokens,
        ),
    )


async def test_tool_result_overflow_compacts_and_retries(session):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "print('x' * 4000)"})]
                )
            ),
            _overflow(),
            _overflow(),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )

    try:
        result = await engine.run("produce a large tool result")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    checkpoint_messages = client.calls[3]["messages"]
    assert any(
        message.get("content") == DROPPED_TOOL_RESULT for message in checkpoint_messages
    )
    assert client.calls[3]["tool_choice"] == "none"


async def test_context_overflow_propagates_when_compaction_is_disabled(session):
    client = _ScriptedClient([_overflow()])
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(compaction=False),
    )

    try:
        with pytest.raises(BadRequestError):
            await engine.run("overflow without compaction")
    finally:
        engine.close()

    assert engine._metrics.num_compactions == 0


async def test_decode_context_limit_uses_discovered_threshold(session):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[DummyToolCall("ipython", {"code": "print('ready')"})]
                )
            ),
            _response(
                DummyMessage(content="partial decode"),
                prompt_tokens=95,
                completion_tokens=5,
                finish_reason="length",
            ),
            _response(DummyMessage(content="summary")),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=112,
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )

    try:
        result = await engine.run("fill the remaining context")
    finally:
        engine.close()

    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1
    checkpoint_messages = client.calls[2]["messages"]
    assert all(
        message.get("content") != "partial decode" for message in checkpoint_messages
    )


async def test_subagent_recovers_from_context_overflow(tmp_path):
    clients: list[_ScriptedClient] = []
    engines: list[RLMEngine] = []

    def engine_factory(**kwargs: Any) -> RLMEngine:
        client = _ScriptedClient(
            [
                _overflow(),
                _response(DummyMessage(content="summary")),
                _response(DummyMessage(content="child done")),
            ]
        )
        engine = RLMEngine(client=client, **kwargs)  # type: ignore[arg-type]
        clients.append(client)
        engines.append(engine)
        return engine

    config = _config(max_depth=1)
    root = Session(tmp_path / "root")
    supervisor = SessionTreeSupervisor(
        root_session=root,
        runtime_config=config,
        cwd=str(tmp_path),
        engine_factory=engine_factory,
    )
    await supervisor.start()
    scope = await supervisor.open_scope(supervisor.root_id)
    endpoint = supervisor.endpoint_for(supervisor.root_id)
    try:
        task = await supervisor._start_child(
            endpoint.capability, scope, "recover in the child"
        )
        result = await task
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        root.close()

    assert result.answer == "child done"
    assert engines[0].depth == 1
    assert engines[0]._metrics.num_compactions == 1
    assert len(clients[0].calls) == 3
