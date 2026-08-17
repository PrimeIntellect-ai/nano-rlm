"""Stdio MCP server used to verify cancellation and subsequent reuse."""

import asyncio
import os
from pathlib import Path

from mcp.server.fastmcp import FastMCP

server = FastMCP("blocking-stdio-test")


@server.tool()
async def wait(marker: str) -> str:
    Path(marker).write_text(str(os.getpid()))
    await asyncio.sleep(30)
    return "finished"


@server.tool()
def echo(text: str) -> str:
    return text


if __name__ == "__main__":
    server.run(transport="stdio")
