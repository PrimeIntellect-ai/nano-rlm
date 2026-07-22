"""Small stdio MCP server used by the transport integration test."""

import os

from mcp.server.fastmcp import FastMCP

server = FastMCP("stdio-test")


@server.tool()
def echo(text: str) -> str:
    inherited = all(name in os.environ for name in ("HOME", "PATH"))
    return f"{os.environ['TEST_PREFIX']}:{inherited}:{text}"


if __name__ == "__main__":
    server.run(transport="stdio")
