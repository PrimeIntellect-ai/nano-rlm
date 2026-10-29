"""A large, paginated MCP catalog with a nested tool schema."""

import asyncio
import json
import os

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

server = Server(
    "catalog-test", instructions="Find calendar events and catalog operations."
)
COUNT = 3000
PAGE_SIZE = 100
SCHEMA = {
    "type": "object",
    "properties": {
        "filter": {
            "type": "object",
            "properties": {"status": {"type": "string", "enum": ["confirmed"]}},
            "required": ["status"],
        }
    },
    "required": ["filter"],
}


@server.list_tools()
async def list_tools(request: types.ListToolsRequest) -> types.ListToolsResult:
    cursor = request.params.cursor if request and request.params else None
    start = int((cursor or "0").partition(":")[0])
    if os.environ.get("ENDLESS_EMPTY_PAGES"):
        return types.ListToolsResult(
            tools=[],
            nextCursor=f"{start + 1}:" + "x" * int(os.environ.get("CURSOR_BYTES", "0")),
        )
    end = min(start + PAGE_SIZE, COUNT)
    return types.ListToolsResult(
        tools=[
            types.Tool(
                name="calendarEvents" if i == COUNT - 1 else f"operation_{i}",
                description=(
                    "Find calendar events by confirmation status."
                    if i == COUNT - 1
                    else "Unrelated operation. " * 60
                ),
                inputSchema=SCHEMA,
            )
            for i in range(start, end)
        ],
        nextCursor=("100" if os.environ.get("REPEAT_CURSOR") else str(end))
        if end < COUNT
        else None,
    )


@server.call_tool(validate_input=False)
async def call_tool(name: str, arguments: dict) -> list[types.TextContent]:
    if name != "calendarEvents":
        raise ValueError("unknown tool")
    if arguments["filter"]["status"] != "confirmed":
        raise ValueError("invalid status")
    return [types.TextContent(type="text", text=json.dumps(arguments))]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
