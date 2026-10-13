"""On-demand access to supervisor-owned MCP tools."""

from typing import Any

from rlm import broker


async def servers() -> list[dict[str, Any]]:
    """List MCP server names and tool counts without retrieving tool schemas."""
    return await broker.agent_request("tools.servers")


async def search(
    query: str, *, limit: int = 5, server: str | None = None
) -> list[dict[str, str]]:
    """Search tool names/descriptions by keywords; return at most 20 summaries."""
    return await broker.agent_request(
        "tools.search", query=query, limit=limit, server=server
    )


async def get(name: str):
    """Load a tool by search-result id. Use .schema or help() before calling it."""
    descriptor = await broker.agent_request("tools.get", name=name)
    return broker.make_skill(descriptor)
