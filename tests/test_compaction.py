from __future__ import annotations

import json
from copy import deepcopy
from types import SimpleNamespace
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
from rlm.compaction import (
    SUMMARY_FRAMING,
    CompactionFailed,
    is_context_overflow,
)
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.history import history
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


def test_overflow_detection_is_status_gated():
    response = httpx.Response(
        400,
        request=httpx.Request("POST", "http://interceptor/v1/chat/completions"),
    )
    error = BadRequestError(
        "maximum context length is 32,768 tokens",
        response=response,
        body={"error": {"message": "maximum context length is 32,768 tokens"}},
    )

    assert is_context_overflow(error)


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

    @property
    def models(self):
        outer = self

        class _Models:
            async def list(self):
                extra = (
                    {"max_model_len": outer.max_model_len}
                    if outer.max_model_len is not None
                    else {}
                )
                card = SimpleNamespace(id="test-model", model_extra=extra)
                return SimpleNamespace(data=[card])

        return _Models()

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
    max_compaction_attempts: int = 5,
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
            max_compaction_attempts=max_compaction_attempts,
        ),
    )


async def test_compaction_attempt_limit_is_configurable(session):
    client = _ScriptedClient(
        [
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
            _response(DummyMessage(tool_calls=[DummyToolCall("ipython", {})])),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(max_compaction_attempts=2),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "prompt"},
    ]

    try:
        with pytest.raises(CompactionFailed, match="after 2 attempts"):
            await engine._compact_branch(messages, turn=0)
    finally:
        engine.close()

    assert len(client.calls) == 2


async def test_tool_result_overflow_compacts_and_retries(session):
    client = _ScriptedClient(
        [
            _response(
                DummyMessage(
                    tool_calls=[
                        DummyToolCall("ipython", {"code": "print('x' * 40000)"})
                    ]
                )
            ),
            _overflow(),
            _overflow(),
            _response(DummyMessage(content="summary")),
            _response(
                DummyMessage(
                    tool_calls=[
                        DummyToolCall(
                            "ipython",
                            {
                                "code": (
                                    "from rlm import history\n"
                                    "h = history()\n"
                                    "print(h.user_messages()[0]['content'])\n"
                                    "print(len(next(r['message']['content'] for r in h.events if r['type'] == 'tool_result')))\n"
                                    "print(h.windows[0].messages[1]['content'])"
                                )
                            },
                        )
                    ]
                )
            ),
            _response(DummyMessage(content="done")),
        ],
        max_model_len=32_768,
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
    assert client.calls[3]["tool_choice"] == "none"
    retry_messages = client.calls[4]["messages"]
    assert len(retry_messages) == 2
    assert retry_messages[1]["content"].startswith(SUMMARY_FRAMING)
    assert str(session.dir / "messages.jsonl") in retry_messages[1]["content"]
    records = [
        json.loads(line)
        for line in (session.dir / "messages.jsonl").read_text().splitlines()
    ]
    tool_records = [entry for entry in records if entry["type"] == "tool_result"]
    assert tool_records[0]["message"] == {
        "role": "tool",
        "tool_call_id": "call_0",
        "content": "x" * 40000 + "\n",
    }
    assistant = next(entry for entry in records if entry["type"] == "assistant")
    assert (
        assistant["message"]["tool_calls"][0]["id"]
        == tool_records[0]["message"]["tool_call_id"]
    )
    assert "40001" in tool_records[1]["content"]
    assert "produce a large tool result" in tool_records[1]["content"]
    assert (
        next(entry for entry in records if entry["type"] == "system")["message"]["role"]
        == "system"
    )
    assert not any(entry["type"].startswith("checkpoint_") for entry in records)
    ledger = history(session.dir)
    assert ledger.windows[0].messages == client.calls[1]["messages"]
    assert ledger.windows[1].messages[:2] == retry_messages
    assert ledger.windows[1].messages == session.messages
    assert ledger.windows[0].message_indices[0] == ledger.windows[1].message_indices[0]


async def test_overflow_recovers_without_discovered_threshold(session):
    """Reactive compaction works when the provider advertises no context window."""
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
        ],
        max_model_len=None,
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

    assert engine.summarize_at_tokens is None
    assert result.answer == "done"
    assert engine._metrics.num_compactions == 1


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
                _response(
                    DummyMessage(
                        tool_calls=[DummyToolCall("ipython", {"code": "print('hi')"})]
                    )
                ),
                _overflow(),
                _response(DummyMessage(content="summary")),
                _response(DummyMessage(content="child done")),
            ],
            max_model_len=32_768,
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
        child = supervisor._spawn(
            supervisor._caller(endpoint.capability, scope),
            scope,
            "recover in the child",
            None,
            False,
        )
        await child.done.wait()
        result = child.result
    finally:
        await supervisor.close_scope(scope)
        await supervisor.aclose()
        root.close()

    assert result.answer == "child done"
    assert engines[0].depth == 1
    assert engines[0]._metrics.num_compactions == 1
    assert len(clients[0].calls) == 4
