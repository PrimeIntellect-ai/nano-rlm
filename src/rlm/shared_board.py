"""The population's shared Markdown board, owned by the supervisor.

Every method runs synchronously on the supervisor's event loop, so each operation
observes and replaces one consistent revision. Writes are atomic file replacements,
journaled with their author, and deduplicated by the caller's request ID.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path

INITIAL_BOARD = """# Shared board

Shared by every agent in this episode. Record plans, claims, findings and open
questions here; read the latest revision before editing.
"""
MAX_CHUNK_BYTES = 65_536
MAX_BOARD_BYTES = 16 * 1024 * 1024
MAX_READ_LINES = 1000
MAX_SEARCH_MATCHES = 100


class BoardConflict(RuntimeError):
    pass


class SharedBoard:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.journal = path.with_suffix(".jsonl")
        self.revision = 0
        self.text = INITIAL_BOARD
        self.writes = 0
        self._receipts: dict[tuple[str, str], tuple[str, dict]] = {}
        self._write(INITIAL_BOARD)

    @property
    def sha256(self) -> str:
        return hashlib.sha256(self.text.encode()).hexdigest()

    def _write(self, text: str) -> None:
        tmp = self.path.with_name(f".{self.path.name}.tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, self.path)

    def _check_revision(self, revision: int | None) -> None:
        if revision is not None and revision != self.revision:
            raise BoardConflict(
                f"board_conflict: you read revision {revision}, the board is at "
                f"revision {self.revision}; read it again"
            )

    def read(
        self, start_line: int = 1, max_lines: int = 200, revision: int | None = None
    ) -> dict:
        self._check_revision(revision)
        lines = self.text.splitlines()
        max_lines = min(max_lines, MAX_READ_LINES)
        start = max(start_line, 1) - 1
        page = lines[start : start + max_lines]
        return {
            "text": "\n".join(page),
            "start_line": start + 1,
            "end_line": start + len(page),
            "total_lines": len(lines),
            "truncated": start + len(page) < len(lines),
            "revision": self.revision,
            "sha256": self.sha256,
        }

    def search(self, query: str, limit: int = 20) -> dict:
        matches = [
            {"line": number, "text": line}
            for number, line in enumerate(self.text.splitlines(), start=1)
            if query in line
        ]
        limit = min(limit, MAX_SEARCH_MATCHES)
        return {
            "matches": matches[:limit],
            "total_matches": len(matches),
            "revision": self.revision,
        }

    def append(self, author: str, text: str, request_id: str | None = None) -> dict:
        def apply() -> str:
            chunk = text if text.endswith("\n") else text + "\n"
            if len(chunk.encode()) > MAX_CHUNK_BYTES:
                raise ValueError(
                    f"board appends are limited to {MAX_CHUNK_BYTES} bytes"
                )
            separator = "" if self.text.endswith("\n") else "\n"
            return self.text + separator + chunk

        return self._commit(author, "append", {"text": text}, request_id, None, apply)

    def edit(
        self,
        author: str,
        old: str,
        new: str,
        expected_revision: int,
        request_id: str | None = None,
    ) -> dict:
        def apply() -> str:
            if not old:
                raise ValueError("old must be non-empty")
            count = self.text.count(old)
            if count != 1:
                raise ValueError(
                    f"old must occur exactly once on the board; it occurs {count} times"
                )
            return self.text.replace(old, new, 1)

        return self._commit(
            author,
            "edit",
            {"old": old, "new": new, "expected_revision": expected_revision},
            request_id,
            expected_revision,
            apply,
        )

    def _commit(
        self, author, op, arguments, request_id, expected_revision, apply
    ) -> dict:
        fingerprint = hashlib.sha256(
            json.dumps([op, arguments], sort_keys=True).encode()
        ).hexdigest()
        key = (author, request_id) if request_id else None
        if key in self._receipts:
            previous, receipt = self._receipts[key]
            if previous != fingerprint:
                raise ValueError(
                    "request_id was already used for a different board write"
                )
            return receipt
        self._check_revision(expected_revision)
        before = self.sha256
        text = apply()
        if len(text.encode()) > MAX_BOARD_BYTES:
            raise ValueError(f"the board is limited to {MAX_BOARD_BYTES} bytes")
        self._write(text)
        self.text = text
        self.revision += 1
        self.writes += 1
        receipt = {"revision": self.revision, "sha256": self.sha256, "bytes": len(text)}
        with self.journal.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {
                        "revision": self.revision,
                        "author": author,
                        "op": op,
                        "request_id": request_id,
                        "before_sha256": before,
                        "after_sha256": receipt["sha256"],
                        "timestamp": time.time(),
                        **arguments,
                    }
                )
                + "\n"
            )
        if key is not None:
            self._receipts[key] = (fingerprint, receipt)
        return receipt
