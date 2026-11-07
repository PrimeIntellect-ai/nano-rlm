"""Session directory management. Writes meta.json + messages.jsonl."""

from __future__ import annotations

import json
import os
import time
import uuid
from pathlib import Path

from rlm.types import ChildSessionAggregate, ProgrammaticToolCallStats


class Session:
    def __init__(self, session_dir: Path | None = None):
        if session_dir is None:
            sid = uuid.uuid4().hex[:12]
            rlm_home = Path(os.environ.get("RLM_HOME") or Path.home() / ".rlm")
            session_dir = rlm_home / "sessions" / sid
        # Session writes must remain valid if the working directory changes.
        self.dir = Path(session_dir).resolve()
        self.dir.mkdir(parents=True, exist_ok=True)
        fd = os.open(
            self.dir / "messages.jsonl",
            os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_APPEND,
            0o644,
        )
        # Unbuffered, so a failed write leaves no bytes queued for a later flush.
        self._msg_file = open(fd, "ab", buffering=0)
        self._size = 0
        self._unwritten = bytearray()
        self._write_error: OSError | None = None
        self._message_count = 0
        self._window = -1
        self._context: list[tuple[int, dict]] = []
        self._context_started = False

    @property
    def message_count(self) -> int:
        """Number of recorded messages; the next stable message index."""
        return self._message_count

    @property
    def messages(self) -> list[dict]:
        """Snapshot of the active context, in model-request order."""
        return [message for _, message in self._context]

    @property
    def context_indices(self) -> tuple[int, ...]:
        return tuple(index for index, _ in self._context)

    def check_writable(self) -> None:
        if self._write_error is not None:
            raise OSError(
                f"session ledger is unusable after a write failure: {self._write_error}"
            ) from self._write_error
        if self._msg_file.closed:
            raise RuntimeError("session ledger is closed")

    def write_meta(self, **kwargs):
        """Write meta.json atomically."""
        meta_path = self.dir / "meta.json"
        if meta_path.exists():
            existing = json.loads(meta_path.read_text())
            existing.update(kwargs)
            data = existing
        else:
            data = kwargs
        tmp = self.dir / "meta.json.tmp"
        tmp.write_text(json.dumps(data, indent=2, default=str))
        tmp.rename(meta_path)

    def log(self, entry: dict, *, in_context: bool = False) -> int | None:
        """Append a line to messages.jsonl."""
        self.check_writable()
        if in_context and not self._context_started:
            self.replace_context([], reason="start")
        index = None
        if "message" in entry:
            index = self._message_count
            entry["message_index"] = index
            if in_context:
                entry["window"] = self._window
        entry.setdefault("timestamp", time.time())
        entry.setdefault("id", uuid.uuid4().hex)
        self._unwritten += (json.dumps(entry, default=str) + "\n").encode()
        self._write_unwritten()
        if index is not None:
            self._message_count += 1
            if in_context:
                self._context.append((index, entry["message"]))
        return index

    def _write_unwritten(self) -> OSError | None:
        """Append the records the file doesn't hold yet. A failed write (e.g. a
        full disk) is cut back to the last whole record and kept for the next
        attempt, so a transient failure doesn't end the session. Only a failed
        cut leaves the file torn, and then the ledger refuses further writes."""
        written = 0
        try:
            while written < len(self._unwritten):
                written += self._msg_file.write(self._unwritten[written:])
        except OSError as exc:
            try:
                self._msg_file.truncate(self._size)
            except OSError as truncate_error:
                self._write_error = truncate_error
                raise truncate_error from exc
            return exc
        self._size += written
        self._unwritten.clear()
        return None

    def flush(self) -> None:
        """Write every logged record to the file, or raise why it can't."""
        self.check_writable()
        error = self._write_unwritten()
        if error is not None:
            raise OSError(
                f"session ledger could not write {len(self._unwritten)} bytes: {error}"
            ) from error

    def replace_context(
        self, messages: list[dict], *, reason: str, indices: list[int] | None = None
    ) -> int:
        """Start a new window without changing any previous window's addresses."""
        if indices is None:
            known = {id(message): index for index, message in self._context}
            indices = []
            for message in messages:
                index = known.get(id(message))
                if index is None:
                    index = self.log({"type": "context_message", "message": message})
                indices.append(index)
        context = list(zip(indices, messages, strict=True))
        try:
            self.log(
                {
                    "type": "context_window",
                    "window": self._window + 1,
                    "reason": reason,
                    "message_indices": indices,
                }
            )
        finally:
            # Restore memory even if recording a rollback poisons the writer.
            self._context = context
        self._window += 1
        self._context_started = True
        return self._window

    def log_assistant(self, turn: int, tool_calls: list[dict] | None, message: dict):
        entry = {"type": "assistant", "turn": turn, "message": message}
        if tool_calls:
            entry["tool_calls"] = tool_calls
        if message.get("content"):
            entry["content"] = message["content"]
        self.log(entry, in_context=True)

    def log_tool_result(
        self,
        turn: int,
        tool: str,
        content: str,
        duration: float,
        *,
        call_id: str,
        context_content: str | None = None,
    ) -> dict:
        message = {"role": "tool", "tool_call_id": call_id, "content": content}
        truncated = context_content is not None and context_content != content
        source_index = self.log(
            {
                "type": "tool_result",
                "turn": turn,
                "tool": tool,
                "content": content,
                "duration": round(duration, 3),
                "message": message,
            },
            in_context=not truncated,
        )
        if truncated:
            message = {**message, "content": context_content}
            self.log(
                {
                    "type": "context_message",
                    "source_message_index": source_index,
                    "message": message,
                },
                in_context=True,
            )
        return message

    def log_sub_spawn(self, child_name: str, command: str, *, prompt: str):
        self.log(
            {
                "type": "sub_spawn",
                "child_dir": child_name,
                "command": command,
                "prompt": prompt,
            }
        )

    def aggregate_child_metrics(
        self, field: str = "programmatic_tool_call_stats"
    ) -> ChildSessionAggregate:
        """Aggregate finalized metadata or live logs across recursive children."""

        def subtree_stats(child_dir: Path) -> ProgrammaticToolCallStats:
            meta_path = child_dir / "meta.json"
            try:
                meta = json.loads(meta_path.read_text())
            except FileNotFoundError:
                meta = None
            if meta is not None and field in meta:
                return ProgrammaticToolCallStats.from_meta(meta, field)

            stats = ProgrammaticToolCallStats.from_log(
                child_dir / "programmatic_tool_calls.jsonl"
            )
            for grandchild in child_dir.glob("sub-*"):
                if grandchild.is_dir():
                    stats = stats.merge(subtree_stats(grandchild))
            return stats

        aggregate = ChildSessionAggregate()
        aggregate.num_sessions = sum(
            1 for path in self.dir.rglob("sub-*") if path.is_dir()
        )
        for child_dir in self.dir.glob("sub-*"):
            if child_dir.is_dir():
                aggregate.absorb(subtree_stats(child_dir))
        return aggregate

    def finalize(
        self,
        answer: str,
        usage: dict | None = None,
        turns: int = 0,
        metrics=None,
        trusted_direct_tool_stats: ProgrammaticToolCallStats | None = None,
        trusted_child_tool_stats: ProgrammaticToolCallStats | None = None,
    ):
        entry = {"type": "done", "answer": answer[:1000]}
        if usage:
            entry["usage"] = usage
        if turns:
            entry["turns"] = turns
        self.log(entry)

        meta_update = {"status": "done", "answer_preview": answer[:200], "turns": turns}
        if usage:
            meta_update["usage"] = usage
        if metrics is not None:
            local_direct_tool_stats = ProgrammaticToolCallStats.from_log(
                self.dir / "programmatic_tool_calls.jsonl"
            )
            local_child = self.aggregate_child_metrics(
                "local_programmatic_tool_call_stats"
            )
            local_child_tool_stats = local_child.tool_call_stats
            direct_tool_stats = local_direct_tool_stats
            if trusted_direct_tool_stats is not None:
                direct_tool_stats = direct_tool_stats.merge(trusted_direct_tool_stats)
            if trusted_child_tool_stats is not None:
                child_tool_stats = local_child_tool_stats.merge(
                    trusted_child_tool_stats
                )
            else:
                child_tool_stats = self.aggregate_child_metrics().tool_call_stats

            metrics.apply_programmatic_tool_call_stats(
                direct_tool_stats, child_tool_stats, local_child.num_sessions
            )

            meta_update["metrics"] = metrics.to_dict()
            meta_update["programmatic_tool_call_stats"] = direct_tool_stats.merge(
                child_tool_stats
            ).to_dict()
            meta_update["local_programmatic_tool_call_stats"] = (
                local_direct_tool_stats.merge(local_child_tool_stats).to_dict()
            )
        self.flush()
        self.write_meta(**meta_update)
        self._msg_file.close()

    @staticmethod
    def child_dir(parent_dir: Path | str) -> Path:
        """Create and return a new child session directory under parent_dir."""
        child_id = uuid.uuid4().hex[:8]
        child = Path(parent_dir) / f"sub-{child_id}"
        child.mkdir()
        return child

    def close(self):
        if self._msg_file.closed:
            return
        try:
            if self._write_error is None:
                self.flush()
        finally:
            self._msg_file.close()
