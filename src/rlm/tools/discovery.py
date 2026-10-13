"""On-demand access to supervisor-owned MCP tools."""

from typing import Any

from rlm import broker


async def servers() -> list[dict[str, Any]]:
    """List server names, tool counts and available descriptions, without schemas."""
    return await broker.agent_request("tools.servers")


async def search(
    query: str,
    *,
    limit: int = 5,
    server: str | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Search non-empty keywords (limit 1..20). Page with the same query and server."""
    return await broker.agent_request(
        "tools.search", query=query, limit=limit, server=server, cursor=cursor
    )


async def get(name: str):
    """Load an async callable returning str. Inspect .schema or help() before use."""
    descriptor = await broker.agent_request("tools.get", name=name)
    return broker.make_skill(descriptor)
