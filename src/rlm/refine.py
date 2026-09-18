"""Ask the supervisor to refine this agent's continual harness.

Refinement never runs mid-cell: ``run()`` returns at once and the pass happens at the
next model-call boundary, after which the system prompt is rebuilt and a runtime notice
describes the applied edits. Calling ``run()`` again before then only replaces the
request.
"""

from rlm import broker


async def run(
    instructions: str | None = None,
    *,
    global_: bool = False,
    rollback_id: str | None = None,
) -> dict:
    """Schedule a refinement pass.

    Args:
        instructions: optional focus for the pass (an observation, a lesson to record).
        global_: write the shared global store instead of this session's local store.
        rollback_id: undo a previous refinement by its id instead of planning new edits.

    Returns:
        ``{"scheduled": True}`` or ``{"scheduled": False, "reason": ...}``.
    """
    return await broker.agent_request(
        "refine.run",
        instructions=instructions,
        global_=global_,
        rollback_id=rollback_id,
    )


async def status() -> dict:
    """``{"pending": bool, "in_flight": bool}`` for this agent's refinement."""
    return await broker.agent_request(
        "refine.status", instructions=None, global_=False, rollback_id=None
    )
