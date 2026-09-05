from __future__ import annotations

from copy import deepcopy
import json
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
    CompactionFailed,
    checkpoint_rejection_reason,
    is_context_overflow,
)
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


@pytest.mark.parametrize(
    "metadata,finish,text,strict,reason",
    [
        (
            {"version": 1, "status": "complete", "reason": None},
            "stop",
            "summary",
            True,
            None,
        ),
        (
            {"version": 1, "status": "incomplete", "reason": "unfinished_reasoning"},
            "stop",
            "misparsed reasoning",
            False,
            "unfinished_reasoning",
        ),
        (
            {"version": 1, "status": "invalid", "reason": "malformed_tool_call"},
            "stop",
            "text",
            False,
            "malformed_tool_call",
        ),
        (
            {"version": 1, "status": "complete", "reason": None},
            "length",
            "partial",
            True,
            "output_truncated",
        ),
        (
            {"version": 1, "status": "complete", "reason": None},
            "stop",
            "",
            True,
            "missing_final_output",
        ),
        (None, "stop", "summary", False, None),
        (None, "stop", "summary", True, "missing_completion_status"),
        (None, "length", "partial", False, "output_truncated"),
        (None, None, "summary", False, "non_final_termination"),
        (
            {"version": 1, "status": "unknown", "reason": "parser_unavailable"},
            "stop",
            "summary",
            False,
            None,
        ),
        (
            {"version": 1, "status": "unknown", "reason": "parser_unavailable"},
            "stop",
            "summary",
            True,
            "parser_unavailable",
        ),
        (
            {"version": 1, "status": "unknown", "reason": "unknown_termination"},
            "stop",
            "summary",
            False,
            "unknown_termination",
        ),
        (
            {"version": 2, "status": "complete"},
            "stop",
            "summary",
            False,
            "unsupported_completion_metadata",
        ),
        (
            {"version": True, "status": "complete"},
            "stop",
            "summary",
            False,
            "invalid_completion_metadata",
        ),
        (
            {"version": 1, "status": []},
            "stop",
            "summary",
            False,
            "invalid_completion_metadata",
        ),
        ([], "stop", "summary", False, "invalid_completion_metadata"),
    ],
)
def test_checkpoint_completion_contract(metadata, finish, text, strict, reason):
    choice = SimpleNamespace(
        vf_completion=metadata,
        finish_reason=finish,
        message=SimpleNamespace(content=text, tool_calls=None),
    )
    assert (
        checkpoint_rejection_reason(choice, require_completion_status=strict) == reason
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
    require_compaction_completion_status: bool = False,
    max_total_tokens: int | None = 1_000_000,
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
            require_compaction_completion_status=require_compaction_completion_status,
            max_total_tokens=max_total_tokens,
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


@pytest.mark.parametrize("succeeds", [False, True])
async def test_rejected_checkpoint_preserves_history_budget_and_edges(
    session, succeeds
):
    bad = _response(
        DummyMessage(content="unfinished reasoning"),
        prompt_tokens=10,
        completion_tokens=3,
    )
    bad.choices[0].vf_completion = {
        "version": 1,
        "status": "incomplete",
        "reason": "unfinished_reasoning",
    }
    good = _response(
        DummyMessage(content="validated summary"), prompt_tokens=10, completion_tokens=4
    )
    good.choices[0].vf_completion = {"version": 1, "status": "complete", "reason": None}
    client = _ScriptedClient(
        [
            _response(DummyMessage(content="work")),
            bad,
            good if succeeds else bad,
            *([_response(DummyMessage(content="resumed"))] if succeeds else []),
        ]
    )
    config = _config(
        max_compaction_attempts=2, require_compaction_completion_status=True
    )
    engine = RLMEngine(client=client, session=session, runtime_config=config)
    engine._user_prompts = ["original task"]
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "original task"},
        {"role": "assistant", "content": "work"},
    ]
    original = deepcopy(messages)
    try:
        await engine._call_model(messages)
        if succeeds:
            await engine._compact_branch(messages, turn=1)
            assert len(messages) == 2
            assert "validated summary" in messages[1]["content"]
            assert "original task" in messages[1]["content"]
            assert "unfinished reasoning" not in messages[1]["content"]
            await engine._call_model(messages)
        else:
            with pytest.raises(CompactionFailed):
                await engine._compact_branch(messages, turn=1)
            assert messages == original
        assert engine._metrics.num_compaction_attempts == 2
        assert engine._metrics.num_failed_compaction_attempts == (1 if succeeds else 2)
        assert engine._own_new_tokens == (31 if succeeds else 28)
        assert client.calls[1]["messages"] == client.calls[2]["messages"]
        edges = engine._semantic_edges.snapshot()["edges"]
        assert sum(e["type"] == "compaction_attempt" for e in edges) == 2
        assert sum(e["type"] == "compaction" for e in edges) == int(succeeds)
        rejected_id = client.calls[1]["extra_headers"]["X-ACP-Model-Request-ID"]
        assert not any(e["source_request_id"] == rejected_id for e in edges)
        events = [
            json.loads(line)
            for line in (session.dir / "messages.jsonl").read_text().splitlines()
        ]
        attempts = [e for e in events if e["type"] == "compaction_attempt"]
        assert len(attempts) == 2
        assert attempts[0]["reason"] == "unfinished_reasoning"
        assert attempts[0]["request_id"] == rejected_id
    finally:
        engine.close()


async def test_compaction_does_not_retry_after_spending_tree_budget(session):
    client = _ScriptedClient(
        [
            _response(DummyMessage(content=None), prompt_tokens=2, completion_tokens=3),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=_config(max_total_tokens=5)
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
    ]
    original = deepcopy(messages)
    try:
        with pytest.raises(CompactionFailed, match="max_total_tokens"):
            await engine._compact_branch(messages, turn=0)
        assert messages == original
        assert len(client.calls) == 1
        assert engine._own_new_tokens == 5
        assert engine._metrics.num_compaction_attempts == 1
        assert engine._metrics.num_failed_compaction_attempts == 1
    finally:
        engine.close()


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
    assert result.task == "produce a large tool result"
    assert engine._metrics.num_compactions == 1
    assert client.calls[3]["tool_choice"] == "none"
    # The retried work call runs on the rebuilt branch: system + framed summary.
    retry_messages = client.calls[4]["messages"]
    assert len(retry_messages) == 2
    assert "summary" in retry_messages[1]["content"]
    assert "produce a large tool result" in retry_messages[1]["content"]


async def test_repeated_compaction_retains_user_requests_despite_bad_summary(session):
    client = _ScriptedClient(
        [
            _response(DummyMessage(content="first answer")),
            _response(DummyMessage(content="<tool_call>ipython</tool_call>")),
            _response(DummyMessage(content="summary without the task")),
            _response(DummyMessage(content="second answer")),
            _response(DummyMessage(content="third summary")),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=_config(),
    )
    try:
        await engine.prompt("Identify the original site. Preserve this exact question.")
        # Exercise two checkpoints on the same branch without another work turn.
        await engine._compact_branch(engine._messages, turn=0)
        await engine._compact_branch(engine._messages, turn=0)
        content = engine._messages[1]["content"]
        assert (
            content.count("Identify the original site. Preserve this exact question.")
            == 1
        )
        assert "summary without the task" in content
        result = await engine.prompt("Include the decisive source URL.")
        assert result.task == "Include the decisive source URL."
        await engine._compact_branch(engine._messages, turn=1)
        content = engine._messages[1]["content"]
        assert "Identify the original site. Preserve this exact question." in content
        assert "Include the decisive source URL." in content
        assert "summary without the task" not in content
    finally:
        await engine.aclose()


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
    assert len(clients[0].calls) == 4
