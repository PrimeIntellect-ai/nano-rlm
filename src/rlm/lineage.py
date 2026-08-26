"""Stable model-call lineage for recursive sessions and context compactions."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal


ContextTransition = Literal["root", "spawn", "compact"]
RequestKind = Literal["turn", "compaction"]
SessionStatus = Literal["running", "completed", "failed", "cancelled"]
CompactionStatus = Literal["in_progress", "completed", "failed", "cancelled"]

REQUEST_ID_HEADER = "X-RLM-Request-ID"
SESSION_ID_HEADER = "X-RLM-Session-ID"
PARENT_SESSION_ID_HEADER = "X-RLM-Parent-Session-ID"
CONTEXT_ID_HEADER = "X-RLM-Context-ID"
PREVIOUS_CONTEXT_ID_HEADER = "X-RLM-Previous-Context-ID"
TRANSITION_HEADER = "X-RLM-Transition"
COMPACTION_ID_HEADER = "X-RLM-Compaction-ID"
DEPTH_HEADER = "X-RLM-Depth"
LINEAGE_HEADER_NAMES = (
    REQUEST_ID_HEADER,
    SESSION_ID_HEADER,
    PARENT_SESSION_ID_HEADER,
    CONTEXT_ID_HEADER,
    PREVIOUS_CONTEXT_ID_HEADER,
    TRANSITION_HEADER,
    COMPACTION_ID_HEADER,
    DEPTH_HEADER,
)


@dataclass(frozen=True)
class RequestProvenance:
    request_id: str
    session_id: str
    parent_session_id: str | None
    context_id: str
    previous_context_id: str | None
    transition: ContextTransition
    compaction_id: str | None
    depth: int


@dataclass(frozen=True)
class Compaction:
    compaction_id: str
    session_id: str
    source_context_id: str
    target_context_id: str


class LineageTracker:
    """Append-only provenance shared by every engine in one recursive tree."""

    def __init__(self) -> None:
        self._sessions: dict[str, dict] = {}
        self._contexts: dict[str, dict] = {}
        self._pending_contexts: dict[str, dict] = {}
        self._compactions: dict[str, dict] = {}
        self._requests: dict[str, dict] = {}
        self._active_contexts: dict[str, str] = {}

    def register_session(
        self,
        session_id: str,
        *,
        parent_session_id: str | None,
        depth: int,
        spawned_by_request_id: str | None = None,
    ) -> str:
        existing = self._sessions.get(session_id)
        if existing is not None:
            return existing["initial_context_id"]

        context_id = uuid.uuid4().hex
        session = {
            "session_id": session_id,
            "depth": depth,
            "initial_context_id": context_id,
            "status": "running",
        }
        if parent_session_id is not None:
            session["parent_session_id"] = parent_session_id
        if spawned_by_request_id is not None:
            session["spawned_by_request_id"] = spawned_by_request_id
        self._sessions[session_id] = session

        context = {
            "context_id": context_id,
            "session_id": session_id,
            "transition": "root" if parent_session_id is None else "spawn",
        }
        self._contexts[context_id] = context
        self._active_contexts[session_id] = context_id
        return context_id

    def set_session_status(self, session_id: str, status: SessionStatus) -> None:
        self._sessions[session_id]["status"] = status

    def active_context(self, session_id: str) -> str:
        return self._active_contexts[session_id]

    def restore_context(self, session_id: str, context_id: str) -> None:
        context = self._contexts[context_id]
        if context["session_id"] != session_id:
            raise ValueError("context does not belong to session")
        self._active_contexts[session_id] = context_id

    def start_request(
        self,
        session_id: str,
        *,
        kind: RequestKind,
        compaction_id: str | None = None,
    ) -> RequestProvenance:
        if kind == "compaction" and compaction_id is None:
            raise ValueError("compaction requests require a compaction ID")
        session = self._sessions[session_id]
        context_id = self._active_contexts[session_id]
        context = self._contexts[context_id]
        effective_compaction_id = compaction_id or context.get("compaction_id")
        request_id = uuid.uuid4().hex
        request = {
            "request_id": request_id,
            "session_id": session_id,
            "context_id": context_id,
            "kind": kind,
        }
        if effective_compaction_id is not None:
            request["compaction_id"] = effective_compaction_id
        self._requests[request_id] = request

        if kind == "compaction":
            self._compactions[compaction_id]["summary_request_id"] = request_id

        return RequestProvenance(
            request_id=request_id,
            session_id=session_id,
            parent_session_id=session.get("parent_session_id"),
            context_id=context_id,
            previous_context_id=context.get("previous_context_id"),
            transition=context["transition"],
            compaction_id=effective_compaction_id,
            depth=session["depth"],
        )

    def begin_compaction(self, session_id: str) -> Compaction:
        compaction_id = uuid.uuid4().hex
        source_context_id = self._active_contexts[session_id]
        target_context_id = uuid.uuid4().hex
        self._pending_contexts[compaction_id] = {
            "context_id": target_context_id,
            "session_id": session_id,
            "previous_context_id": source_context_id,
            "transition": "compact",
            "compaction_id": compaction_id,
        }
        self._compactions[compaction_id] = {
            "compaction_id": compaction_id,
            "session_id": session_id,
            "source_context_id": source_context_id,
            "target_context_id": target_context_id,
            "status": "in_progress",
        }
        return Compaction(
            compaction_id=compaction_id,
            session_id=session_id,
            source_context_id=source_context_id,
            target_context_id=target_context_id,
        )

    def finish_compaction(self, compaction_id: str, status: CompactionStatus) -> None:
        compaction = self._compactions[compaction_id]
        if status == "in_progress":
            raise ValueError("a compaction cannot finish in progress")
        compaction["status"] = status
        self._contexts[compaction["target_context_id"]] = self._pending_contexts.pop(
            compaction_id
        )
        if status == "completed":
            self._active_contexts[compaction["session_id"]] = compaction[
                "target_context_id"
            ]

    def snapshot(self) -> dict[str, list[dict]]:
        return {
            "sessions": [entry.copy() for entry in self._sessions.values()],
            "contexts": [entry.copy() for entry in self._contexts.values()],
            "compactions": [entry.copy() for entry in self._compactions.values()],
            "requests": [entry.copy() for entry in self._requests.values()],
        }
