"""Supervisor-backed agent discovery and handles for IPython programs."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rlm import broker
from rlm.history import History, history
from rlm.types import AgentResult


@dataclass(frozen=True)
class AgentInfo:
    id: str
    parent_id: str | None
    name: str | None
    task: str
    status: Literal[
        "starting",
        "running",
        "waiting",
        "idle",
        "retiring",
        "retired",
        "completed",
        "failed",
        "cancelled",
    ]
    persistent: bool
    created_at: float
    elapsed_seconds: float
    session_dir: Path
    error: str | None
    turns: int = 0
    cleanup_error: str | None = None

    @classmethod
    def from_payload(cls, payload: dict) -> AgentInfo:
        return cls(**{**payload, "session_dir": Path(payload["session_dir"])})


@dataclass(frozen=True)
class AgentHandle:
    id: str
    session_dir: Path
    name: str | None = None

    async def info(self) -> AgentInfo:
        """Read current metadata from the supervisor."""
        return AgentInfo.from_payload(
            await broker.agent_request("agent.info", agent_id=self.id)
        )

    async def history(self) -> History:
        """Read a fresh snapshot of this agent's local conversation history."""
        return await history(session_dir=self.session_dir)

    async def result(self, *, yield_after: float = 300) -> AgentResult:
        """Wait up to yield_after seconds for the child to finish, then return its latest state.

        .answer is None while the child is still working on its first answer (.running
        is True); a persistent child's latest answer stays available while it runs
        again. Terminal failure/cancellation raises."""
        payload = await broker.agent_request(
            "agent.result", agent_id=self.id, yield_after=yield_after
        )
        return broker.result_from_payload(payload)

    async def wait(self, timeout: float = 30) -> AgentInfo:
        """Wait up to timeout seconds for an outcome, then return current metadata.

        This waits inside the Python cell. Timing out or cancelling the wait does
        not cancel the agent. The cell's normal execution timeout still applies.
        """
        return AgentInfo.from_payload(
            await broker.agent_request("agent.wait", agent_id=self.id, timeout=timeout)
        )

    async def cancel(self) -> AgentInfo:
        """Terminate the subtree, or retry unfinished cleanup after it has finished."""
        return AgentInfo.from_payload(
            await broker.agent_request("agent.cancel", agent_id=self.id)
        )

    async def send(self, message: str) -> str:
        """Accept an instruction for the next answer/wait boundary and return its ID.

        Undeliverable instructions produce agent.delivery_failed in the parent inbox.
        """
        return await broker.agent_request(
            "agent.send", agent_id=self.id, message=message
        )

    async def steer(self, message: str) -> str:
        """Accept an instruction for the next model/tool boundary and return its ID.

        Active tools are not interrupted. Delivery failures go to the parent inbox.
        """
        return await broker.agent_request(
            "agent.steer", agent_id=self.id, message=message
        )


async def spawn(
    task: str,
    *,
    name: str | None = None,
    persistent: bool = False,
    request_id: str | None = None,
) -> AgentHandle:
    """Register a child and return its handle before the child finishes.

    Sibling names remain reserved for the session. Persistent children keep their
    kernel after answering; all children terminate when their parent terminates.
    Retrying with the same request_id returns the first outcome instead of spawning twice.
    """
    info = AgentInfo.from_payload(
        await broker.agent_request(
            "agent.spawn",
            task=task,
            name=name,
            persistent=persistent,
            request_id=request_id,
        )
    )
    return AgentHandle(info.id, info.session_dir, info.name)


async def get(name_or_id: str) -> AgentHandle:
    """Recover a direct child's handle by sibling name or immutable ID."""
    info = AgentInfo.from_payload(
        await broker.agent_request("agent.get", name_or_id=name_or_id)
    )
    return AgentHandle(info.id, info.session_dir, info.name)


async def send_to_parent(message: str) -> str:
    """Place a report in the immediate parent's inbox; the parent chooses when to read it."""
    return await broker.agent_request("agent.report", message=message)


async def dm(agent: str, message: str, *, message_id: str | None = None) -> dict:
    """Send a direct message to another agent of the population, by name or ID.

    It reaches the recipient at its next model call and wakes it if idle; a retired
    recipient is revived when a slot is free. Returns a receipt; retrying with the same
    message_id returns the original receipt instead of sending twice."""
    return await broker.agent_request(
        "agent.dm", agent_id=agent, message=message, message_id=message_id
    )


async def broadcast(message: str, *, message_id: str | None = None) -> dict:
    """Send one message to every live agent of the population except the caller.

    Returns the names it was delivered to and any per-recipient rejections. Retired
    agents are not revived by a broadcast."""
    return await broker.agent_request(
        "agent.broadcast", message=message, message_id=message_id
    )


async def delivery(message_id: str) -> dict:
    """The delivery state of a message you sent: "status" is accepted (queued),
    delivered (in the recipient's conversation) or failed (with a "reason")."""
    return await broker.agent_request("agent.delivery", message_id=message_id)


async def list(*, recursive: bool = False) -> builtins.list[AgentInfo]:
    """List direct children, including finished agents; optionally include descendants.

    Discovering a descendant's ID does not grant control of that agent.
    """
    payload = await broker.agent_request("agent.list", recursive=recursive)
    return [AgentInfo.from_payload(item) for item in payload]
