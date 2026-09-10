"""Read-only snapshots of a session's message ledger and context windows."""

from __future__ import annotations

import json
import os
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path


def read_records(path: Path) -> Iterator[dict]:
    with path.open("rb") as stream:
        for line in stream:
            # A live writer may not have finished its final record yet.
            if not line.endswith(b"\n"):
                break
            yield json.loads(line)


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
        self.events = list(read_records(self.session_dir / "messages.jsonl"))
        self.messages: list[dict] = []
        self.windows: list[ContextWindow] = []
        self.child_session_dirs: list[Path] = []
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
            if event["type"] == "sub_spawn":
                self.child_session_dirs.append(self.session_dir / event["child_dir"])

    def user_messages(self) -> list[dict]:
        """Original user inputs, including attempts identified by rollback events."""
        return [
            self.messages[event["message_index"]]
            for event in self.events
            if event["type"] == "user" and "message_index" in event
        ]


def history(session_dir: str | Path | None = None) -> History:
    """Read a session's ledger; defaults to this kernel's RLM session directory."""
    if session_dir is None:
        session_dir = os.environ.get("RLM_SESSION_DIR")
        if not session_dir:
            raise RuntimeError("pass a session directory when outside an RLM kernel")
    return History(session_dir)
