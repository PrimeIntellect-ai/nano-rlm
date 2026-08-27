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


@dataclass
class _Compaction:
    session_id: str
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

    def checkpoint(self, session_id: str) -> tuple[_PendingEdge, ...]:
        """Capture unpublished inbound edges before a resumable prompt."""
        return tuple(self._sessions[session_id].pending_edges)

    def restore(self, session_id: str, checkpoint: tuple[_PendingEdge, ...]) -> None:
        """Discard unpublished transitions created by a rolled-back prompt."""
        self._sessions[session_id].pending_edges = list(checkpoint)

    def start_request(
        self,
        session_id: str,
        *,
        compaction_id: str | None = None,
    ) -> str:
        session = self._sessions[session_id]
        inbound = session.pending_edges
        session.pending_edges = []
        if not session.spawn_claimed and session.spawned_by_request_id is not None:
            inbound.append(_PendingEdge(session.spawned_by_request_id, "subagent_call"))
            session.spawn_claimed = True

        request_id = uuid.uuid4().hex
        self._requests[request_id] = _Request(session_id, inbound)
        if compaction_id is not None:
            compaction = self._compactions[compaction_id]
            if compaction.session_id != session_id:
                raise ValueError("compaction does not belong to session")
            if compaction.summary_request_id is not None:
                raise ValueError("compaction already has a summary request")
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
        self._sessions[request.session_id].last_request_id = request_id

    def fail_request(self, request_id: str) -> None:
        request = self._requests.pop(request_id)
        session = self._sessions[request.session_id]
        session.pending_edges = request.inbound_edges + session.pending_edges

    def begin_compaction(self, session_id: str) -> Compaction:
        compaction_id = uuid.uuid4().hex
        self._compactions[compaction_id] = _Compaction(session_id=session_id)
        return Compaction(compaction_id=compaction_id, session_id=session_id)

    def finish_compaction(self, compaction_id: str, status: CompactionStatus) -> None:
        compaction = self._compactions.pop(compaction_id)
        if status != "completed":
            return
        summary_request_id = compaction.summary_request_id
        session = self._sessions[compaction.session_id]
        if summary_request_id is None or session.last_request_id != summary_request_id:
            raise ValueError(
                "completed compaction requires a successful summary request"
            )
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
