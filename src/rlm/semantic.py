"""ACP model-request correlation and semantic edges."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from typing import Literal


CompactionStatus = Literal["completed", "failed", "cancelled"]

MODEL_REQUEST_ID_HEADER = "X-ACP-Model-Request-ID"
ACP_EXTENSION_HEADER_NAMES = (MODEL_REQUEST_ID_HEADER,)


@dataclass(frozen=True)
class Compaction:
    compaction_id: str
    session_id: str


@dataclass(frozen=True)
class _PendingEdge:
    source_request_id: str
    type: str


@dataclass(frozen=True)
class _Checkpoint:
    pending_edges: tuple[_PendingEdge, ...]
    last_request_id: str | None
    spawn_claimed: bool


@dataclass
class _Session:
    parent_session_id: str | None
    spawned_by_request_id: str | None
    spawn_claimed: bool = False
    last_request_id: str | None = None
    pending_edges: list[_PendingEdge] = field(default_factory=list)
    returned: bool = False


@dataclass
class _Request:
    session_id: str
    inbound_edges: list[_PendingEdge]
    compaction_id: str | None = None


@dataclass
class _Compaction:
    session_id: str
    source_request_id: str | None
    summary_request_id: str | None = None


class SemanticEdgeTracker:
    """Semantic request edges shared by every engine in one recursive tree."""

    def __init__(self) -> None:
        self._sessions: dict[str, _Session] = {}
        self._requests: dict[str, _Request] = {}
        self._compactions: dict[str, _Compaction] = {}
        self._edges: list[dict[str, str]] = []

    def register_session(
        self,
        session_id: str,
        *,
        parent_session_id: str | None,
        spawned_by_request_id: str | None = None,
    ) -> None:
        if session_id in self._sessions:
            return
        if parent_session_id is None and spawned_by_request_id is not None:
            raise ValueError("root session cannot have a spawning request")
        self._sessions[session_id] = _Session(
            parent_session_id=parent_session_id,
            spawned_by_request_id=spawned_by_request_id,
        )

    def checkpoint(self, session_id: str) -> _Checkpoint:
        """Capture the semantic continuation point before a resumable prompt."""
        session = self._sessions[session_id]
        return _Checkpoint(
            pending_edges=tuple(session.pending_edges),
            last_request_id=session.last_request_id,
            spawn_claimed=session.spawn_claimed,
        )

    def restore(self, session_id: str, checkpoint: _Checkpoint) -> None:
        """Restore the semantic continuation point after a rolled-back prompt."""
        session = self._sessions[session_id]
        session.pending_edges = list(checkpoint.pending_edges)
        session.last_request_id = checkpoint.last_request_id
        session.spawn_claimed = checkpoint.spawn_claimed

    def start_request(
        self,
        session_id: str,
        *,
        compaction_id: str | None = None,
    ) -> str:
        session = self._sessions[session_id]
        if compaction_id is not None:
            compaction = self._compactions[compaction_id]
            if compaction.session_id != session_id:
                raise ValueError("compaction does not belong to session")
            if compaction.summary_request_id is not None:
                raise ValueError("compaction already has a summary request")
            inbound = (
                [_PendingEdge(compaction.source_request_id, "compaction_attempt")]
                if compaction.source_request_id is not None
                else []
            )
        else:
            inbound = session.pending_edges
            session.pending_edges = []
            if not session.spawn_claimed and session.spawned_by_request_id is not None:
                inbound.append(
                    _PendingEdge(session.spawned_by_request_id, "subagent_call")
                )
                session.spawn_claimed = True
            if session.last_request_id is not None and not any(
                edge.source_request_id == session.last_request_id for edge in inbound
            ):
                inbound.append(_PendingEdge(session.last_request_id, "continuation"))

        request_id = uuid.uuid4().hex
        self._requests[request_id] = _Request(session_id, inbound, compaction_id)
        if compaction_id is not None:
            compaction.summary_request_id = request_id
        return request_id

    def finish_request(self, request_id: str) -> None:
        request = self._requests.pop(request_id)
        for inbound in request.inbound_edges:
            self._edges.append(
                {
                    "source_request_id": inbound.source_request_id,
                    "target_request_id": request_id,
                    "type": inbound.type,
                }
            )
        if request.compaction_id is None:
            self._sessions[request.session_id].last_request_id = request_id

    def fail_request(self, request_id: str) -> None:
        request = self._requests.pop(request_id)
        session = self._sessions[request.session_id]
        if request.compaction_id is not None:
            compaction = self._compactions.get(request.compaction_id)
            if compaction is not None and compaction.summary_request_id == request_id:
                compaction.summary_request_id = None
            return
        session.pending_edges = request.inbound_edges + session.pending_edges

    def release_summary_request(self, compaction_id: str) -> None:
        """Release a committed summary attempt so another attempt can start."""
        compaction = self._compactions[compaction_id]
        if compaction.summary_request_id is None:
            raise ValueError("compaction has no summary request to reject")
        compaction.summary_request_id = None

    def begin_compaction(self, session_id: str) -> Compaction:
        session = self._sessions[session_id]
        compaction_id = uuid.uuid4().hex
        self._compactions[compaction_id] = _Compaction(
            session_id=session_id,
            source_request_id=session.last_request_id,
        )
        return Compaction(compaction_id=compaction_id, session_id=session_id)

    def finish_compaction(self, compaction_id: str, status: CompactionStatus) -> None:
        compaction = self._compactions.pop(compaction_id)
        if status != "completed":
            return
        summary_request_id = compaction.summary_request_id
        session = self._sessions[compaction.session_id]
        if summary_request_id is None:
            raise ValueError(
                "completed compaction requires a successful summary request"
            )
        session.last_request_id = summary_request_id
        session.pending_edges.append(_PendingEdge(summary_request_id, "compaction"))

    def finish_subagent(self, session_id: str) -> None:
        session = self._sessions[session_id]
        if session.returned:
            return
        if session.parent_session_id is None:
            raise ValueError("root session cannot return to a parent")
        if session.last_request_id is None:
            session.returned = True
            return
        parent = self._sessions[session.parent_session_id]
        parent.pending_edges.append(
            _PendingEdge(session.last_request_id, "subagent_return")
        )
        session.returned = True

    def snapshot(self) -> dict[str, list[dict[str, str]]]:
        return {"edges": [edge.copy() for edge in self._edges]}
