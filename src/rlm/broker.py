"""Framed local RPC used by model-controlled IPython kernels."""

from __future__ import annotations

import asyncio
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rlm.types import RLMResult, TokenUsage


MAX_REQUEST_BYTES = 1024 * 1024
MAX_RESPONSE_BYTES = 16 * 1024 * 1024


@dataclass(frozen=True)
class BrokerEndpoint:
    socket_path: str
    capability: str


_endpoint: BrokerEndpoint | None = None
_scope_id: str | None = None


def configure(endpoint: BrokerEndpoint | None) -> None:
    global _endpoint
    _endpoint = endpoint


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


def result_to_json(result: RLMResult) -> dict[str, Any]:
    return {
        "answer": result.answer,
        "session_dir": str(result.session_dir) if result.session_dir else None,
        "usage": {
            "prompt_tokens": result.usage.prompt_tokens,
            "completion_tokens": result.usage.completion_tokens,
        },
        "turns": result.turns,
    }


def result_from_json(value: dict[str, Any]) -> RLMResult:
    usage = value.get("usage") or {}
    session_dir = value.get("session_dir")
    return RLMResult(
        answer=str(value.get("answer", "")),
        session_dir=Path(session_dir) if session_dir else None,
        usage=TokenUsage(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
        ),
        turns=int(value.get("turns", 0)),
    )


async def run(prompt: str, **kwargs: Any) -> RLMResult:
    """Run a recursive RLM through the trusted session supervisor."""
    if _endpoint is None or _scope_id is None:
        raise RuntimeError("recursive RLM calls are unavailable outside an active cell")
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    reader, writer = await asyncio.open_unix_connection(_endpoint.socket_path)
    try:
        await write_frame(
            writer,
            {
                "op": "rlm.run",
                "capability": _endpoint.capability,
                "scope_id": _scope_id,
                "prompt": prompt,
                "options": kwargs,
            },
            MAX_REQUEST_BYTES,
        )
        response = await read_frame(reader, MAX_RESPONSE_BYTES)
    finally:
        writer.close()
        await writer.wait_closed()
    if error := response.get("error"):
        raise RuntimeError(str(error))
    result = response.get("result")
    if not isinstance(result, dict):
        raise RuntimeError("invalid response from RLM supervisor")
    return result_from_json(result)
