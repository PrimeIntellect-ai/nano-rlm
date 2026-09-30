"""Opt-in execution tapes for strict replay at engine tool/model boundaries."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Awaitable, Callable
from collections import defaultdict
from pathlib import Path
from typing import Any, Literal

from openai.types.chat import ChatCompletion


class ReplayDivergence(RuntimeError):
    """Execution no longer matches the recorded tape."""


class ExecutionTape:
    """One tape per session tree, used on its supervisor's event loop.

    Replay executes tools, substitutes model responses, and checks boundary order.
    It does not virtualize clocks, filesystem races, network calls, or subprocesses.
    Use a fresh sandbox with the original inputs before replaying untrusted code.
    """

    def __init__(self, path: Path, *, mode: Literal["record", "replay"]):
        if mode not in ("record", "replay"):
            raise ValueError("mode must be record or replay")
        self.mode = mode
        self.path = Path(path)
        self._cursor = 0
        self._changed = asyncio.Event()
        self._failure: ReplayDivergence | None = None
        self._paths: dict[str, str] = {}
        self._counts: dict[str, int] = defaultdict(int)
        self._used_identities: set[str] = set()
        self._closed = False
        if mode == "record":
            self._file = self.path.open("x", encoding="utf-8")
            self._write({"type": "header", "version": 1})
            self._events: list[dict] = []
            self._identities: dict[str, dict] = {}
        else:
            self._file = None
            records = [json.loads(line) for line in self.path.read_text().splitlines()]
            if not records or records[0] != {"type": "header", "version": 1}:
                raise ValueError("unsupported execution tape")
            if records[-1].get("type") != "complete":
                raise ValueError("execution tape is incomplete")
            self._identities = {
                row["key"]: row for row in records if row["type"] == "identity"
            }
            self._events = [row for row in records if row["type"] == "event"]

    def _write(self, row: dict) -> None:
        self._file.write(json.dumps(row, ensure_ascii=False) + "\n")
        self._file.flush()

    def check(self) -> None:
        if self._failure is not None:
            raise self._failure
        if self._closed:
            raise RuntimeError("execution tape is closed")

    def abort(self, message: str) -> None:
        """Latch failures even when a supervisor catches a child exception."""
        if self._failure is None:
            self._failure = ReplayDivergence(message)
        self._changed.set()

    def _fail(self, message: str) -> None:
        self.abort(message)
        raise self._failure

    def identity(self, key: str, **attributes: Any) -> dict:
        """Allocate stable root/child IDs; child keys are parent-local ordinals."""
        self.check()
        if key in self._used_identities:
            self._fail(f"duplicate identity: {key}")
        self._used_identities.add(key)
        if self.mode == "record":
            row = {
                "type": "identity",
                "key": key,
                "id": uuid.uuid4().hex,
                "attributes": attributes,
            }
            self._write(row)
        else:
            row = self._identities.get(key)
            if row is None or row["attributes"] != attributes:
                self._fail(f"identity diverged: {key}")
        return row

    def child(self, parent: str, **attributes: Any) -> str:
        self._counts[parent] += 1
        return self.identity(f"{parent}/{self._counts[parent]}", **attributes)["id"]

    def bind_path(self, path: str | Path, label: str) -> None:
        self._paths[str(Path(path).resolve())] = f"<rlm:{label}>"

    def _normalize(self, value: Any) -> Any:
        if isinstance(value, str):
            for path in sorted(self._paths, key=len, reverse=True):
                value = value.replace(path, self._paths[path])
            return value
        if isinstance(value, list):
            return [self._normalize(item) for item in value]
        if isinstance(value, dict):
            return {key: self._normalize(item) for key, item in value.items()}
        return value

    async def event(self, actor: str, kind: str, data: Any = None, *, supply=False):
        """Match a boundary, waiting for other actors to reach their next boundary.

        A bounded wait turns missing actors or changed control flow into a failure.
        The timeout is a liveness guard, not a reproduction of original timing.
        """
        self.check()
        normalized = data if kind == "model.end" else self._normalize(data)
        if self.mode == "record":
            self._write(
                {
                    "type": "event",
                    "actor": actor,
                    "kind": kind,
                    "time": time.time(),
                    "data": normalized,
                }
            )
            return data
        while True:
            self.check()
            if self._cursor >= len(self._events):
                self._fail(f"unexpected {actor}:{kind} after end of tape")
            expected = self._events[self._cursor]
            if expected["actor"] == actor:
                if expected["kind"] != kind or (
                    not supply and expected["data"] != normalized
                ):
                    self._fail(
                        f"event {self._cursor}: {actor}:{kind} diverged "
                        f"(expected {expected['kind']}; inspect {self.path})"
                    )
                self._cursor += 1
                self._changed.set()
                self._changed = asyncio.Event()
                return expected["data"]
            changed = self._changed
            try:
                await asyncio.wait_for(changed.wait(), timeout=30)
            except asyncio.TimeoutError:
                self._fail(
                    f"event {self._cursor}: waiting for "
                    f"{expected['actor']}:{expected['kind']}, got {actor}:{kind}"
                )

    async def model(
        self,
        actor: str,
        request: dict,
        invoke: Callable[[], Awaitable[ChatCompletion]],
    ) -> ChatCompletion:
        # Transport request IDs are freshly allocated by the semantic graph.
        comparable = {k: v for k, v in request.items() if k != "extra_headers"}
        await self.event(actor, "model.start", comparable)
        if self.mode == "replay":
            payload = await self.event(actor, "model.end", supply=True)
            return ChatCompletion.model_validate_json(payload)
        try:
            response = await invoke()
        except BaseException:
            self.abort(f"{actor}: model errors/cancellation are not replayable")
            raise
        # Response bytes remain executable inputs, not path-normalized comparisons.
        payload = response.model_dump(mode="json")
        await self.event(actor, "model.end", json.dumps(payload))
        return response

    def finish(self) -> None:
        """Seal a successful recording, or require complete replay consumption."""
        self.check()
        if self.mode == "replay":
            if self._cursor != len(self._events):
                self._fail(
                    f"unconsumed tape: {len(self._events) - self._cursor} events"
                )
            if self._used_identities != set(self._identities):
                self._fail("unconsumed session identities")
        else:
            self._write({"type": "complete"})
        self.close()

    def close(self) -> None:
        """Close without sealing on failure; incomplete recordings cannot replay."""
        if self._file is not None:
            self._file.close()
        self._closed = True
