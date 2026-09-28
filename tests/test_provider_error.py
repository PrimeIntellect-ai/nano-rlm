"""A model/provider fault escaping a turn is tagged so the host records a provider error."""

from __future__ import annotations

import httpx
import pytest
from acp import RequestError, text_block
from openai import APIConnectionError

from rlm.acp import RLMACPAgent, _SessionState
from rlm.client import is_provider_error


def test_is_provider_error_direct():
    assert is_provider_error(ConnectionResetError())
    assert is_provider_error(
        APIConnectionError(request=httpx.Request("POST", "http://x"))
    )


def test_is_provider_error_follows_cause_and_context():
    cause = RuntimeError("wrapped")
    cause.__cause__ = ConnectionResetError()
    assert is_provider_error(cause)
    ctx = RuntimeError("wrapped")
    ctx.__context__ = APIConnectionError(request=httpx.Request("POST", "http://x"))
    assert is_provider_error(ctx)


def test_is_provider_error_rejects_non_provider():
    assert not is_provider_error(RuntimeError("engine bug"))
    # A full-disk ledger write failure is an environment fault, not a provider one.
    ledger = RuntimeError("session ledger is unusable after a write failure")
    ledger.__cause__ = OSError(28, "No space left on device")
    assert not is_provider_error(ledger)


class _FakeEngine:
    def __init__(self, exc: BaseException) -> None:
        self._exc = exc
        self.stop_reason = None

    async def prompt(self, _text: str):
        raise self._exc


async def _run_turn(exc: BaseException):
    agent = RLMACPAgent()
    agent._sessions["s"] = _SessionState(engine=_FakeEngine(exc), session_id="s")
    await agent.prompt(session_id="s", prompt=[text_block("hi")])


async def test_provider_error_becomes_structured_request_error():
    with pytest.raises(RequestError) as excinfo:
        await _run_turn(APIConnectionError(request=httpx.Request("POST", "http://x")))
    error = excinfo.value
    assert isinstance(error.data, dict)
    assert error.data["kind"] == "provider"
    assert error.data["retryable"] is True
    assert "APIConnectionError" in error.data["details"]


async def test_non_provider_error_is_reraised_unchanged():
    with pytest.raises(RuntimeError, match="session ledger is unusable"):
        await _run_turn(
            RuntimeError("session ledger is unusable after a write failure")
        )
