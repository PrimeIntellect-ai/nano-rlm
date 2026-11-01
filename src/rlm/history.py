"""Read-only snapshots of a session's message ledger and context windows."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


HISTORY_PAGE_BYTES = 8 * 1024 * 1024
"""JSON bytes of ledger records one page of another agent's history holds, beyond its
first record: half a frame, so a page always crosses the broker and execution wire."""


def read_records(path: Path) -> Iterator[dict]:
    with path.open("rb") as stream:
        for line in stream:
            # A live writer may not have finished its final record yet.
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


def read_page(
    path: Path, offset: int, limit: int = HISTORY_PAGE_BYTES
) -> tuple[list[dict], int | None]:
    """The complete records from byte `offset` that fit `limit` bytes of JSON, at least
    one, and the offset of the record after them, or None at the ledger's end."""
    records: list[dict] = []
    size = 0
    with path.open("rb") as stream:
        stream.seek(offset)
        for line in stream:
            if not line.endswith(b"\n"):
                break
            record = json.loads(line)
            size += len(json.dumps(record))
            if records and size > limit:
                return records, offset
            records.append(record)
            offset += len(line)
    return records, None


@dataclass
class ContextWindow:
    index: int
    reason: str
    message_indices: list[int]
    _messages: list[dict] = field(repr=False)

    @property
    def messages(self) -> list[dict]:
        return [self._messages[index] for index in self.message_indices]


class History:
    """A point-in-time snapshot. Read again to observe subsequent live activity."""

    def __init__(self, session_dir: str | Path):
        self.session_dir = Path(session_dir).resolve()
        self._load(list(read_records(self.session_dir / "messages.jsonl")))

    @classmethod
    def from_events(cls, session_dir: str | Path, events: list[dict]) -> History:
        """Construct a history snapshot received from the owning supervisor."""
        history = cls.__new__(cls)
        history.session_dir = Path(session_dir)
        history._load(events)
        return history

    def _load(self, events: list[dict]) -> None:
        self.events = events
        self.messages: list[dict] = []
        self.windows: list[ContextWindow] = []
        for event in self.events:
            if "message_index" in event:
                index = event["message_index"]
                if index != len(self.messages):
                    raise ValueError("non-contiguous message indices in ledger")
                self.messages.append(event["message"])
                if "window" in event:
                    self.windows[event["window"]].message_indices.append(index)
            if event["type"] == "context_window":
                if event["window"] != len(self.windows):
                    raise ValueError("non-contiguous context windows in ledger")
                indices = event["message_indices"]
                if any(index < 0 or index >= len(self.messages) for index in indices):
                    raise ValueError("context window references an unknown message")
                self.windows.append(
                    ContextWindow(
                        event["window"], event["reason"], list(indices), self.messages
                    )
                )

    def user_messages(self) -> list[dict]:
        """Original user inputs, including attempts identified by rollback events."""
        return [
            self.messages[event["message_index"]]
            for event in self.events
            if event["type"] == "user" and "message_index" in event
        ]


async def history(session_dir: str | Path | None = None) -> History:
    """Read a session's ledger; defaults to this kernel's RLM session directory.
    Async like every other rlm call, though it reads the local ledger without the broker."""
    if session_dir is None:
        session_dir = os.environ.get("RLM_SESSION_DIR")
        if not session_dir:
            raise RuntimeError("pass a session directory when outside an RLM kernel")
    return History(session_dir)
