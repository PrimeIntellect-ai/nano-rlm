"""Supervisor-backed agent discovery and handles for IPython programs."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rlm import broker
from rlm.history import History, history
from rlm.types import RLMResult


@dataclass(frozen=True)
class AgentInfo:
    id: str
    parent_id: str | None
    name: str | None
    task: str
    status: Literal[
        "starting", "running", "waiting", "idle", "completed", "failed", "cancelled"
    ]
    persistent: bool
    created_at: float
    elapsed_seconds: float
    session_dir: Path
    error: str | None

    @classmethod
    def from_payload(cls, payload: dict) -> AgentInfo:
        return cls(**{**payload, "session_dir": Path(payload["session_dir"])})


@dataclass(frozen=True)
class AgentHandle:
    id: str
    session_dir: Path

    async def info(self) -> AgentInfo:
        """Read current metadata from the supervisor."""
        return AgentInfo.from_payload(
            await broker.agent_request("agent.info", agent_id=self.id)
        )

    def history(self) -> History:
        """Read a fresh snapshot of this agent's local conversation history."""
        return history(session_dir=self.session_dir)

    async def result(self) -> RLMResult | None:
        """Return the answer, or None while pending; raise for failure/cancellation."""
        payload = await broker.agent_request("agent.result", agent_id=self.id)
        return broker.result_from_payload(payload) if payload is not None else None

    async def wait(self, timeout: float = 30) -> AgentInfo:
        """Wait up to timeout seconds for an outcome, then return current metadata.

        This waits inside the Python cell. Timing out or cancelling the wait does
        not cancel the agent. The cell's normal execution timeout still applies.
        """
        return AgentInfo.from_payload(
            await broker.agent_request("agent.wait", agent_id=self.id, timeout=timeout)
        )

    async def cancel(self) -> AgentInfo:
        """Terminate this agent and its descendants, releasing their runtimes."""
        return AgentInfo.from_payload(
            await broker.agent_request("agent.cancel", agent_id=self.id)
        )

    async def send(self, message: str) -> str:
        """Queue an instruction for the child's next answer/wait boundary."""
        return await broker.agent_request(
            "agent.send", agent_id=self.id, message=message
        )

    async def steer(self, message: str) -> str:
        """Insert an instruction at the next model/tool boundary without interruption."""
        return await broker.agent_request(
            "agent.steer", agent_id=self.id, message=message
        )


async def spawn(
    task: str, *, name: str | None = None, persistent: bool = False
) -> AgentHandle:
    """Register a child and return its handle before the child finishes.

    Sibling names remain reserved for the session. Persistent children keep their
    kernel after answering; all children terminate when their parent terminates.
    """
    info = AgentInfo.from_payload(
        await broker.agent_request(
            "agent.spawn", task=task, name=name, persistent=persistent
        )
    )
    return AgentHandle(info.id, info.session_dir)


async def get(name_or_id: str) -> AgentHandle:
    """Recover a direct child's handle by sibling name or immutable ID."""
    info = AgentInfo.from_payload(
        await broker.agent_request("agent.get", name_or_id=name_or_id)
    )
    return AgentHandle(info.id, info.session_dir)


async def send_to_parent(message: str) -> str:
    """Place a report in the immediate parent's inbox; the parent chooses when to read it."""
    return await broker.agent_request("agent.report", message=message)


async def list(*, recursive: bool = False) -> builtins.list[AgentInfo]:
    """List direct children, including finished agents; optionally include descendants.

    Discovering a descendant's ID does not grant control of that agent.
    """
    payload = await broker.agent_request("agent.list", recursive=recursive)
    return [AgentInfo.from_payload(item) for item in payload]
