"""A large, paginated MCP catalog with a nested tool schema."""

import asyncio
import json
import os

from mcp import types
from mcp.server.lowlevel import Server
from mcp.server.stdio import stdio_server

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


async def list_tools(
    context, params: types.PaginatedRequestParams | None
) -> types.ListToolsResult:
    cursor = params.cursor if params else None
    start = int((cursor or "0").partition(":")[0])
    if os.environ.get("ENDLESS_EMPTY_PAGES"):
        return types.ListToolsResult(
            tools=[],
            next_cursor=f"{start + 1}:"
            + "x" * int(os.environ.get("CURSOR_BYTES", "0")),
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
                input_schema=SCHEMA,
            )
            for i in range(start, end)
        ],
        next_cursor=("100" if os.environ.get("REPEAT_CURSOR") else str(end))
        if end < COUNT
        else None,
    )


async def call_tool(
    context, params: types.CallToolRequestParams
) -> types.CallToolResult:
    if params.name != "calendarEvents":
        raise ValueError("unknown tool")
    if params.arguments["filter"]["status"] != "confirmed":
        raise ValueError("invalid status")
    return types.CallToolResult(
        content=[types.TextContent(type="text", text=json.dumps(params.arguments))]
    )


server = Server(
    "catalog-test",
    instructions="Find calendar events and catalog operations.",
    on_list_tools=list_tools,
    on_call_tool=call_tool,
)


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
