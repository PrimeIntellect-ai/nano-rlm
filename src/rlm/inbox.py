"""Pull-based access to supervisor-owned reports and completion events."""

from rlm import broker


async def list(*, unread_only: bool = True) -> list[dict]:
    """List event metadata without retrieving payloads or changing read state."""
    return await broker.agent_request("inbox.list", unread_only=unread_only)


async def read(event_id: str) -> dict:
    """Retrieve an event and mark it read; previously read events remain available."""
    return await broker.agent_request("inbox.read", event_id=event_id)
