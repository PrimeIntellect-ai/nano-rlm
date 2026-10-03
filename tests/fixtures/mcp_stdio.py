"""Small stdio MCP server used by the transport integration test."""

import os

import base64

from mcp.server.fastmcp import FastMCP, Image

server = FastMCP("stdio-test")


@server.tool()
def echo(text: str) -> str:
    inherited = all(name in os.environ for name in ("HOME", "PATH"))
    return f"{os.environ['TEST_PREFIX']}:{inherited}:{os.getcwd()}:{text}"


@server.tool()
def pixel() -> list:
    return ["one pixel", Image(data=base64.b64decode(PIXEL_PNG), format="png")]


PIXEL_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="


if __name__ == "__main__":
    server.run(transport="stdio")
