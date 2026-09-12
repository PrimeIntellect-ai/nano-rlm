"""Framed local RPC used by model-controlled IPython kernels."""

from __future__ import annotations

import asyncio
import inspect
import json
import keyword
import struct
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

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


class BrokerAgentSpawnRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.spawn"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    task: Annotated[str, Field(min_length=1)]
    name: Annotated[str, Field(min_length=1)] | None
    persistent: bool


class BrokerAgentGetRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.get"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    name_or_id: Annotated[str, Field(min_length=1)]


class BrokerAgentListRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.list"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    recursive: bool


class BrokerAgentHandleRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.info", "agent.result", "agent.cancel"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    agent_id: Annotated[str, Field(min_length=1)]


class BrokerAgentWaitRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.wait"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    agent_id: Annotated[str, Field(min_length=1)]
    timeout: Annotated[float, Field(ge=0, le=300, allow_inf_nan=False)]


class BrokerAgentMessageRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.send", "agent.steer"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    agent_id: Annotated[str, Field(min_length=1)]
    message: Annotated[str, Field(min_length=1)]


class BrokerAgentReportRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["agent.report"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    message: Annotated[str, Field(min_length=1)]


class BrokerInboxListRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["inbox.list"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    unread_only: bool


class BrokerInboxReadRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["inbox.read"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    event_id: Annotated[str, Field(min_length=1)]


class BrokerShellRunRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["shell.run", "shell.start"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    command: Annotated[str, Field(min_length=1, max_length=65_536)]
    cwd: str | None
    timeout: Annotated[float, Field(gt=0)] | None
    env: dict[str, str] | None


class BrokerShellEnvRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["shell.setenv", "shell.getenv"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    variables: dict[str, str] | None


class BrokerHintsRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["hints.mute", "hints.unmute", "hints.muted"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    tags: list[Annotated[str, Field(min_length=1, max_length=64)]]


class BrokerShellListRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["shell.list"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]


class BrokerShellHandleRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["shell.info", "shell.cancel"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]


class BrokerShellReadRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["shell.read"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]
    cursor: Annotated[int, Field(ge=0)]
    max_bytes: Annotated[int, Field(ge=1, le=65_536)]


class BrokerWatchAgentRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["watch.agent"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    agent_id: Annotated[str, Field(min_length=1)]


class BrokerWatchJobRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["watch.job"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    job_id: Annotated[str, Field(min_length=1)]


class BrokerWatchPathRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["watch.path"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    path: Annotated[str, Field(min_length=1, max_length=4096)]
    recursive: bool


class BrokerWatchListRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["watch.list"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]


class BrokerWatchHandleRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid", strict=True)
    op: Literal["watch.get", "watch.cancel"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
    subscription_id: Annotated[str, Field(min_length=1)]


class BrokerSkillRequest(TypedDict):
    __pydantic_config__ = ConfigDict(extra="forbid")
    op: Literal["skill.call"]
    capability: Annotated[str, Field(min_length=1)]
    scope_id: Annotated[str, Field(min_length=1)]
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
    description = descriptor["description"]
    schema = descriptor["input_schema"]

    async def run(**kwargs: Any) -> str:
        return await call_skill(capability, kwargs)

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
