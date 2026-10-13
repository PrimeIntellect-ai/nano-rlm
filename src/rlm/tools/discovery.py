"""On-demand access to supervisor-owned MCP tools."""

from typing import Any

from rlm import broker


async def servers() -> list[dict[str, Any]]:
    """List MCP server names and tool counts without retrieving tool schemas."""
    return await broker.agent_request("tools.servers")


async def search(
    query: str,
    *,
    limit: int = 5,
    server: str | None = None,
    cursor: str | None = None,
) -> dict[str, Any]:
    """Return matches and next_cursor. Continue with the same query and server."""
    return await broker.agent_request(
        "tools.search", query=query, limit=limit, server=server, cursor=cursor
    )


async def get(name: str):
    """Load a tool by search-result id. Use .schema or help() before calling it."""
    descriptor = await broker.agent_request("tools.get", name=name)
    return broker.make_skill(descriptor)
