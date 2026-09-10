"""Supervisor-owned subscriptions that publish references into the caller's inbox."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Literal

from rlm import broker
from rlm.agent import AgentHandle
from rlm.shell import JobHandle


@dataclass(frozen=True)
class SubscriptionInfo:
    id: str
    owner_id: str
    kind: Literal["agent", "job", "path"]
    target: str
    recursive: bool
    status: Literal["active", "cancelled", "failed"]
    created_at: float
    error: str | None


@dataclass(frozen=True)
class SubscriptionHandle:
    id: str

    async def cancel(self) -> SubscriptionInfo:
        """Stop future events; already published inbox events remain readable."""
        return SubscriptionInfo(
            **await broker.agent_request("watch.cancel", subscription_id=self.id)
        )


async def agent(target: AgentHandle) -> SubscriptionHandle:
    """Watch a direct child's future conversation activity after complete steps."""
    info = await broker.agent_request("watch.agent", agent_id=target.id)
    return SubscriptionHandle(info["id"])


async def job(target: JobHandle) -> SubscriptionHandle:
    """Watch future captured output from a Bash job owned by this agent."""
    info = await broker.agent_request("watch.job", job_id=target.id)
    return SubscriptionHandle(info["id"])


async def path(path: str, *, recursive: bool = False) -> SubscriptionHandle:
    """Watch an existing file/directory; relative paths use the agent's cwd.

    Bursts are batched over 200 ms. Removing the watched path ends the
    subscription with a failure event; register a new watch after recreation.
    """
    info = await broker.agent_request("watch.path", path=path, recursive=recursive)
    return SubscriptionHandle(info["id"])


async def get(subscription_id: str) -> SubscriptionHandle:
    """Recover a subscription owned by this agent."""
    info = await broker.agent_request("watch.get", subscription_id=subscription_id)
    return SubscriptionHandle(info["id"])


async def list() -> builtins.list[SubscriptionInfo]:
    """List this agent's subscriptions, including stopped subscriptions."""
    return [
        SubscriptionInfo(**item) for item in await broker.agent_request("watch.list")
    ]
