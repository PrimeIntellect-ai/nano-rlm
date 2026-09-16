"""Framed local RPC used by model-controlled IPython kernels."""

from __future__ import annotations

import asyncio
import inspect
import json
import keyword
import struct
import time
from dataclasses import dataclass
from typing import Annotated, Any, Literal

from pydantic import ConfigDict, Field, TypeAdapter, ValidationError
from typing_extensions import TypedDict

from rlm.types import RLMResult


MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class BrokerEndpoint:
    socket_path: str
    capability: str


_endpoint: BrokerEndpoint | None = None
_scope_id: str | None = None
_cell_deadline: float | None = None

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class _BrokerRequestBase(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]


class BrokerAgentSpawnRequest(_BrokerRequestBase):
    op: Literal["agent.spawn"]
    task: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)] | None
    persistent: bool


class BrokerAgentGetRequest(_BrokerRequestBase):
    op: Literal["agent.get"]
    name_or_id: Annotated[str, Field(min_length=1)]


class BrokerAgentListRequest(_BrokerRequestBase):
    op: Literal["agent.list"]
    recursive: bool


class BrokerAgentHandleRequest(_BrokerRequestBase):
    op: Literal["agent.info", "agent.result", "agent.cancel"]
    agent_id: Annotated[str, Field(min_length=1)]


class BrokerAgentWaitRequest(_BrokerRequestBase):
    op: Literal["agent.wait"]
    agent_id: Annotated[str, Field(min_length=1)]
    timeout: Annotated[float, Field(ge=0, le=300, allow_inf_nan=False)]


class BrokerAgentMessageRequest(_BrokerRequestBase):
    op: Literal["agent.send", "agent.steer"]
    agent_id: Annotated[str, Field(min_length=1)]
    message: Annotated[str, Field(min_length=1)]


class BrokerAgentReportRequest(_BrokerRequestBase):
    op: Literal["agent.report"]
    message: Annotated[str, Field(min_length=1)]


class BrokerInboxListRequest(_BrokerRequestBase):
    op: Literal["inbox.list"]
    unread_only: bool


class BrokerInboxReadRequest(_BrokerRequestBase):
    op: Literal["inbox.read"]
    event_id: Annotated[str, Field(min_length=1)]


class BrokerShellRunRequest(_BrokerRequestBase):
    op: Literal["shell.run"]
    command: Annotated[str, Field(min_length=1, max_length=65_536)]
    cwd: str | None
    yield_after: Annotated[float, Field(ge=0)] | None
    timeout: Annotated[float, Field(gt=0)] | None
    env: dict[str, str] | None


class BrokerShellResultRequest(_BrokerRequestBase):
    op: Literal["shell.result"]
    job_id: Annotated[str, Field(min_length=1)]
    yield_after: Annotated[float, Field(ge=0)] | None


class BrokerShellEnvRequest(_BrokerRequestBase):
    op: Literal["shell.setenv", "shell.getenv"]
    variables: dict[str, str] | None


class BrokerHintsRequest(_BrokerRequestBase):
    op: Literal["hints.mute", "hints.unmute", "hints.muted"]
    tags: list[Annotated[str, Field(min_length=1, max_length=64)]]


class BrokerShellListRequest(_BrokerRequestBase):
    op: Literal["shell.list"]


class BrokerShellHandleRequest(_BrokerRequestBase):
    op: Literal["shell.info", "shell.cancel"]
    job_id: Annotated[str, Field(min_length=1)]


class BrokerShellReadRequest(_BrokerRequestBase):
    op: Literal["shell.read"]
    job_id: Annotated[str, Field(min_length=1)]
    cursor: Annotated[int, Field(ge=0)]
    max_bytes: Annotated[int, Field(ge=1, le=65_536)]


class BrokerWatchAgentRequest(_BrokerRequestBase):
    op: Literal["watch.agent"]
    agent_id: Annotated[str, Field(min_length=1)]


class BrokerWatchJobRequest(_BrokerRequestBase):
    op: Literal["watch.job"]
    job_id: Annotated[str, Field(min_length=1)]


class BrokerWatchPathRequest(_BrokerRequestBase):
    op: Literal["watch.path"]
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    recursive: bool


class BrokerWatchListRequest(_BrokerRequestBase):
    op: Literal["watch.list"]


class BrokerWatchHandleRequest(_BrokerRequestBase):
    op: Literal["watch.get", "watch.cancel"]
    subscription_id: Annotated[str, Field(min_length=1)]


class BrokerSkillRequest(_BrokerRequestBase):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=False)
    op: Literal["skill.call"]
    skill_capability: Annotated[str, Field(min_length=1)]
    arguments: dict[str, Any]


BrokerRequest = Annotated[
    BrokerAgentSpawnRequest
    | BrokerAgentGetRequest
    | BrokerAgentListRequest
    | BrokerAgentHandleRequest
    | BrokerAgentWaitRequest
    | BrokerSkillRequest
    | BrokerAgentMessageRequest
    | BrokerAgentReportRequest
    | BrokerInboxListRequest
    | BrokerInboxReadRequest
    | BrokerShellRunRequest
    | BrokerShellResultRequest
    | BrokerShellListRequest
    | BrokerShellHandleRequest
    | BrokerShellEnvRequest
    | BrokerHintsRequest
    | BrokerShellReadRequest
    | BrokerWatchAgentRequest
    | BrokerWatchJobRequest
    | BrokerWatchPathRequest
    | BrokerWatchListRequest
    | BrokerWatchHandleRequest,
    Field(discriminator="op"),
]
_REQUEST_ADAPTER = TypeAdapter(BrokerRequest)


class _BrokerSuccess(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    result: dict[str, Any] | list[dict[str, Any]] | str | None


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


def set_scope(scope_id: str | None, timeout: float | None = None) -> None:
    global _scope_id
    _scope_id = scope_id
    global _cell_deadline
    _cell_deadline = None if timeout is None else time.monotonic() + timeout


def shell_wait(seconds: float) -> float:
    """Leave time for the broker response before the execution kernel interrupts."""
    if _cell_deadline is None:
        return seconds
    return min(seconds, max(0.0, _cell_deadline - time.monotonic() - 1.0))


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


async def agent_request(op: str, **arguments: Any) -> Any:
    """Invoke an agent operation using the current cell's identity."""
    if _endpoint is None or _scope_id is None:
        raise RuntimeError("agent operations are unavailable outside an active cell")
    request = parse_request(
        {
            "op": op,
            "capability": _endpoint.capability,
            "scope_id": _scope_id,
            **arguments,
        }
    )
    response = await _request(request)
    return response["result"]


async def call_skill(capability: str, arguments: dict[str, Any]) -> str:
    """Invoke a supervisor-owned skill through its opaque capability."""
    if _endpoint is None or _scope_id is None:
        raise RuntimeError("brokered calls are unavailable outside an active cell")
    response = await _request(
        {
            "op": "skill.call",
            "capability": _endpoint.capability,
            "scope_id": _scope_id,
            "skill_capability": capability,
            "arguments": arguments,
        }
    )
    result = response.get("result")
    if not isinstance(result, str):
        raise RuntimeError("invalid skill response from RLM supervisor")
    return result


def make_skill(descriptor: dict[str, Any]):
    """Build a callable coroutine from a public brokered-skill descriptor."""
    capability = descriptor["capability"]
    name = descriptor.get("name") or "skill"
    description = descriptor["description"]
    schema = descriptor["input_schema"]

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    parameters = [
        inspect.Parameter(
            field,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            default=inspect.Parameter.empty if field in required else None,
            annotation=_JSON_TO_PY.get(value.get("type"), inspect.Parameter.empty),
        )
        for field, value in properties.items()
        if field.isidentifier() and not keyword.iskeyword(field)
    ]
    parameters.sort(
        key=lambda parameter: parameter.default is not inspect.Parameter.empty
    )
    signature = inspect.Signature(parameters)
    names = [parameter.name for parameter in parameters]

    async def run(*args: Any, **kwargs: Any) -> str:
        if len(args) > len(names):
            raise TypeError(f"{name}{signature} takes {len(names)} arguments")
        for field, value in zip(names, args):
            if field in kwargs:
                raise TypeError(f"{name}{signature} got '{field}' twice")
            kwargs[field] = value
        return await call_skill(capability, kwargs)

    run.__name__ = run.__qualname__ = name
    run.__signature__ = signature
    run.__doc__ = description
    return run


async def _request(payload: BrokerRequest) -> BrokerResponse:
    if _endpoint is None:
        raise RuntimeError("brokered calls are unavailable outside an active cell")
    reader, writer = await asyncio.open_unix_connection(_endpoint.socket_path)
    try:
        await write_frame(
            writer,
            payload,
            MAX_REQUEST_BYTES,
        )
        raw_response = await read_frame(reader, MAX_RESPONSE_BYTES)
    finally:
        writer.close()
        await writer.wait_closed()
    try:
        response = _RESPONSE_ADAPTER.validate_python(raw_response)
    except ValidationError:
        raise RuntimeError("invalid response from RLM supervisor") from None
    if "error" in response:
        raise RuntimeError(response["error"])
    return response
