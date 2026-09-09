"""Framed local RPC used by model-controlled IPython kernels."""

from __future__ import annotations

import asyncio
import functools
import inspect
import json
import keyword
import struct
import threading
import time
import weakref
from collections.abc import Coroutine, Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from rlm.types import RLMResult


MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024
BROKER_HEARTBEAT_INTERVAL_SECONDS = 0.25
BROKER_HEARTBEAT_GRACE_SECONDS = 1.0


@dataclass(frozen=True)
class BrokerEndpoint:
    socket_path: str
    capability: str


@dataclass(frozen=True)
class BrokerWaitSnapshot:
    """Thread-safe view of broker work awaited by one IPython cell."""

    active: bool
    responsive: bool
    process_time: float | None


class BrokerWaitTracker:
    """Track live broker operations and kernel liveness for one cell."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._heartbeats: dict[str, tuple[float, bool]] = {}
        self._process_time: float | None = None

    def start(self, operation_id: str) -> None:
        with self._lock:
            self._heartbeats[operation_id] = (0.0, False)

    def heartbeat(
        self, operation_id: str, process_time: float, exclusive_wait: bool
    ) -> None:
        with self._lock:
            if operation_id not in self._heartbeats:
                return
            self._heartbeats[operation_id] = (time.monotonic(), exclusive_wait)
            if self._process_time is None or process_time > self._process_time:
                self._process_time = process_time

    def finish(self, operation_id: str) -> None:
        with self._lock:
            self._heartbeats.pop(operation_id, None)

    def snapshot(self) -> BrokerWaitSnapshot:
        with self._lock:
            active = bool(self._heartbeats)
            freshest_exclusive = max(
                (
                    heartbeat_at
                    for heartbeat_at, exclusive in self._heartbeats.values()
                    if exclusive
                ),
                default=0.0,
            )
            return BrokerWaitSnapshot(
                active=active,
                responsive=(
                    freshest_exclusive > 0
                    and time.monotonic() - freshest_exclusive
                    <= BROKER_HEARTBEAT_GRACE_SECONDS
                ),
                process_time=self._process_time,
            )


_endpoint: BrokerEndpoint | None = None
_scope_id: str | None = None
_cell_task: asyncio.Task[Any] | None = None
_original_gather = asyncio.gather


@dataclass
class _SubagentWait:
    scope_id: str
    leases: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)

    @property
    def exclusive(self) -> bool:
        return self.scope_id == _scope_id and self.leases > 0


_subagent_calls: weakref.WeakKeyDictionary[Coroutine[Any, Any, Any], _SubagentWait] = (
    weakref.WeakKeyDictionary()
)


@contextmanager
def cell_execution() -> Iterator[None]:
    """Bind timeout leases to the task actually executing IPython user code."""
    global _cell_task
    previous_task = _cell_task
    _cell_task = asyncio.current_task()
    try:
        yield
    finally:
        _cell_task = previous_task


@contextmanager
def _exclusive_wait(*waits: _SubagentWait) -> Iterator[None]:
    eligible = (
        [wait for wait in waits if wait.scope_id == _scope_id]
        if _cell_task is not None and asyncio.current_task() is _cell_task
        else []
    )
    for wait in eligible:
        wait.leases += 1
        wait.changed.set()
    try:
        yield
    finally:
        for wait in eligible:
            wait.leases -= 1
            wait.changed.set()


_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class BrokerRunRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    op: Literal["rlm.run"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    prompt: str


class BrokerSkillRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    op: Literal["skill.call"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    skill_capability: Annotated[str, Field(min_length=1)]
    arguments: dict[str, Any]


BrokerRequest = Annotated[
    BrokerRunRequest | BrokerSkillRequest,
    Field(discriminator="op"),
]
_REQUEST_ADAPTER = TypeAdapter(BrokerRequest)


class BrokerHeartbeat(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    op: Literal["wait.heartbeat"]
    process_time: Annotated[float, Field(ge=0)]
    exclusive_wait: bool


_HEARTBEAT_ADAPTER = TypeAdapter(BrokerHeartbeat)


class _BrokerSuccess(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    result: dict[str, Any] | str


class _BrokerFailure(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    error: str


BrokerResponse = _BrokerSuccess | _BrokerFailure
_RESPONSE_ADAPTER = TypeAdapter(BrokerResponse)


_RESULT_ADAPTER = TypeAdapter(RLMResult)
_RESULT_FIELDS = {"answer", "session_dir", "usage", "turns"}
_USAGE_FIELDS = {"prompt_tokens", "completion_tokens"}


def parse_request(value: dict[str, Any]) -> BrokerRequest:
    try:
        return _REQUEST_ADAPTER.validate_python(value)
    except ValidationError:
        raise ValueError("invalid broker request") from None


def parse_heartbeat(value: dict[str, Any]) -> BrokerHeartbeat:
    try:
        return _HEARTBEAT_ADAPTER.validate_python(value)
    except ValidationError:
        raise ValueError("invalid broker heartbeat") from None


def result_to_payload(result: RLMResult) -> dict[str, Any]:
    return _RESULT_ADAPTER.dump_python(result, mode="json")


def result_from_payload(value: dict[str, Any]) -> RLMResult:
    usage = value.get("usage")
    if set(value) != _RESULT_FIELDS or not isinstance(usage, dict):
        raise RuntimeError("invalid response from RLM supervisor")
    if set(usage) != _USAGE_FIELDS:
        raise RuntimeError("invalid response from RLM supervisor")
    try:
        return _RESULT_ADAPTER.validate_python(value)
    except ValidationError:
        raise RuntimeError("invalid response from RLM supervisor") from None


def configure(endpoint: BrokerEndpoint | None) -> None:
    global _endpoint
    _endpoint = endpoint
    if endpoint is not None:
        _install_asyncio_hooks()


def is_configured() -> bool:
    return _endpoint is not None


def set_scope(scope_id: str | None) -> None:
    global _scope_id
    _scope_id = scope_id


async def read_frame(
    reader: asyncio.StreamReader, max_bytes: int = MAX_REQUEST_BYTES
) -> dict[str, Any]:
    header = await reader.readexactly(4)
    size = struct.unpack(">I", header)[0]
    if size > max_bytes:
        raise ValueError(f"broker frame exceeds {max_bytes} bytes")
    payload = await reader.readexactly(size)
    value = json.loads(payload)
    if not isinstance(value, dict):
        raise ValueError("broker frame must contain a JSON object")
    return value


async def write_frame(
    writer: asyncio.StreamWriter,
    value: dict[str, Any],
    max_bytes: int = MAX_RESPONSE_BYTES,
) -> None:
    payload = json.dumps(value, separators=(",", ":")).encode()
    if len(payload) > max_bytes:
        raise ValueError(f"broker frame exceeds {max_bytes} bytes")
    writer.write(struct.pack(">I", len(payload)) + payload)
    await writer.drain()


def run(prompt: str) -> Coroutine[Any, Any, RLMResult]:
    """Run a recursive RLM through the trusted session supervisor."""
    if _endpoint is None or _scope_id is None:
        raise RuntimeError("recursive RLM calls are unavailable outside an active cell")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    payload = BrokerRunRequest(
        op="rlm.run",
        capability=_endpoint.capability,
        scope_id=_scope_id,
        prompt=prompt,
    )

    wait = _SubagentWait(_scope_id)

    async def invoke() -> RLMResult:
        with _exclusive_wait(wait):
            response = await _request(payload, wait)
        result = response.get("result")
        if not isinstance(result, dict):
            raise RuntimeError("invalid response from RLM supervisor")
        return result_from_payload(result)

    call = invoke()
    _subagent_calls[call] = wait
    return call


def call_skill(capability: str, arguments: dict[str, Any]) -> Coroutine[Any, Any, str]:
    """Invoke a supervisor-owned skill through its opaque capability."""
    if _endpoint is None or _scope_id is None:
        raise RuntimeError("brokered calls are unavailable outside an active cell")
    payload = BrokerSkillRequest(
        op="skill.call",
        capability=_endpoint.capability,
        scope_id=_scope_id,
        skill_capability=capability,
        arguments=arguments,
    )

    async def invoke() -> str:
        response = await _request(payload)
        result = response.get("result")
        if not isinstance(result, str):
            raise RuntimeError("invalid skill response from RLM supervisor")
        return result

    return invoke()


def make_skill(descriptor: dict[str, Any]):
    """Build a callable coroutine from a public brokered-skill descriptor."""
    capability = descriptor["capability"]
    description = descriptor["description"]
    schema = descriptor["input_schema"]

    def run(**kwargs: Any) -> Coroutine[Any, Any, str]:
        return call_skill(capability, kwargs)

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    parameters = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required else None,
            annotation=_JSON_TO_PY.get(value.get("type"), inspect.Parameter.empty),
        )
        for name, value in properties.items()
        if name.isidentifier() and not keyword.iskeyword(name)
    ]
    parameters.sort(
        key=lambda parameter: parameter.default is not inspect.Parameter.empty
    )
    run.__signature__ = inspect.Signature(parameters)
    run.__doc__ = description
    return run


async def _request(
    payload: BrokerRequest, wait: _SubagentWait | None = None
) -> BrokerResponse:
    if _endpoint is None:
        raise RuntimeError("brokered calls are unavailable outside an active cell")
    reader, writer = await asyncio.open_unix_connection(_endpoint.socket_path)
    heartbeat_task: asyncio.Task[None] | None = None
    try:
        await write_frame(
            writer,
            payload,
            MAX_REQUEST_BYTES,
        )
        heartbeat_task = asyncio.create_task(_send_heartbeats(writer, wait))
        raw_response = await read_frame(reader, MAX_RESPONSE_BYTES)
    finally:
        if heartbeat_task is not None:
            heartbeat_task.cancel()
            await asyncio.gather(heartbeat_task, return_exceptions=True)
        writer.close()
        await writer.wait_closed()
    try:
        response = _RESPONSE_ADAPTER.validate_python(raw_response)
    except ValidationError:
        raise RuntimeError("invalid response from RLM supervisor") from None
    if "error" in response:
        raise RuntimeError(response["error"])
    return response


async def _send_heartbeats(
    writer: asyncio.StreamWriter, wait: _SubagentWait | None
) -> None:
    while True:
        if wait is not None:
            wait.changed.clear()
        await write_frame(
            writer,
            {
                "op": "wait.heartbeat",
                "process_time": time.process_time(),
                "exclusive_wait": wait is not None and wait.exclusive,
            },
            MAX_REQUEST_BYTES,
        )
        if wait is None:
            await asyncio.sleep(BROKER_HEARTBEAT_INTERVAL_SECONDS)
        else:
            try:
                await asyncio.wait_for(
                    wait.changed.wait(), BROKER_HEARTBEAT_INTERVAL_SECONDS
                )
            except asyncio.TimeoutError:
                pass


class _SubagentGather(asyncio.Future):
    """An eager gather whose lease lasts only for a cell-task await."""

    def __init__(self, gather: asyncio.Future, waits: list[_SubagentWait]) -> None:
        super().__init__(loop=gather.get_loop())
        self._gather = gather
        self._waits = waits
        gather.add_done_callback(self._complete)

    def _complete(self, gather: asyncio.Future) -> None:
        if gather.cancelled():
            super().cancel()
        elif (error := gather.exception()) is not None:
            self.set_exception(error)
        else:
            self.set_result(gather.result())

    def cancel(self, msg: str | None = None) -> bool:
        return self._gather.cancel(msg=msg)

    def __await__(self):
        with _exclusive_wait(*self._waits):
            return (yield from super().__await__())

    __iter__ = __await__


def _is_subagent_call(value: Any) -> bool:
    try:
        return value in _subagent_calls
    except TypeError:
        return False


@functools.wraps(asyncio.gather)
def _subagent_aware_gather(*aws: Any, **kwargs: Any):
    if aws and all(_is_subagent_call(aw) for aw in aws):
        return _SubagentGather(
            _original_gather(*aws, **kwargs), [_subagent_calls[aw] for aw in aws]
        )
    return _original_gather(*aws, **kwargs)


def _install_asyncio_hooks() -> None:
    if asyncio.gather is _subagent_aware_gather:
        return
    asyncio.gather = _subagent_aware_gather
