"""Population membership, the episode answer, and peer histories, through the supervisor."""

from __future__ import annotations

import builtins

from rlm import broker


async def status() -> dict:
    """Live, idle and retired counts, slots, admissions, and the episode budgets."""
    return await broker.agent_request("population.status")


async def list(
    *, include_self: bool = False, include_ended: bool = False
) -> builtins.list[dict]:
    """The agents of the population: name, id, parent, status, whether a message reaches
    them ("messageable"; retired agents need a free slot to revive), idle time and task."""
    return await broker.agent_request(
        "population.list", include_self=include_self, include_ended=include_ended
    )


async def events(after: int = 0, limit: int = 100) -> builtins.list[dict]:
    """Membership changes in order (joined, retired, revived, ended, answer), each with
    the population counts at that moment. Pass the last seen "seq" as after."""
    return await broker.agent_request("population.events", after=after, limit=limit)


async def answer() -> dict:
    """The current episode answer: "text" ("" until submitted), "revision" (0 until
    submitted) and "author"."""
    return await broker.agent_request("population.answer")


async def submit(text: str, *, expected_revision: int) -> dict:
    """Replace the episode answer if it is still at expected_revision; raises
    answer_conflict otherwise. The episode is scored on this answer."""
    return await broker.agent_request(
        "population.submit", text=text, expected_revision=expected_revision
    )


async def history(
    agent: str, start: int = 0, limit: int = 20, query: str | None = None
) -> dict:
    """Read another agent's conversation without waking it, when the episode allows it.

    Returns "messages" (index, role, text; long texts truncated), "total", "next" (the
    start of the following page, or None), and the agent's "task" and "last_answer".
    query keeps only messages containing it."""
    return await broker.agent_request(
        "population.history", agent=agent, start=start, limit=limit, query=query
    )
