"""Supervisor hints: one-line notes about how the runtime was used, and the switch to mute them.

The supervisor attaches a hint to the next turn when something worth knowing happened (for
example a blocking run() detached into a background job). Each hint carries a tag; an agent
that has understood a hint mutes its tag so it is not repeated.
"""

from rlm import broker


async def mute(*tags: str) -> list[str]:
    """Stop hints with these tags for this agent; returns the muted tags."""
    return (await broker.agent_request("hints.mute", tags=_tags(tags)))["muted"]


async def unmute(*tags: str) -> list[str]:
    """Re-enable hints with these tags; returns the muted tags that remain."""
    return (await broker.agent_request("hints.unmute", tags=_tags(tags)))["muted"]


async def muted() -> list[str]:
    """The tags currently muted for this agent."""
    return (await broker.agent_request("hints.muted", tags=[]))["muted"]


def _tags(tags) -> list[str]:
    if not all(isinstance(t, str) and t for t in tags):
        raise TypeError("hint tags must be non-empty strings")
    return list(tags)
