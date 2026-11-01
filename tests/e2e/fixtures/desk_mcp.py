"""A stdio MCP server whose results only a real call produces: `echo` stamps what it
returns, and `lookup` fails with a token in its error."""

from mcp.server.fastmcp import FastMCP

ECHO_STAMP = "ok-7f3"
ERROR_TOKEN = "err-5c2"

server = FastMCP("desk")


@server.tool()
def echo(message: str) -> str:
    """Echo the message back, stamped so the caller can prove the tool ran."""
    return f"{message} [{ECHO_STAMP}]"


@server.tool()
def lookup(order_id: str) -> str:
    """Look up an order by its id."""
    raise ValueError(f"Order {order_id} not found [{ERROR_TOKEN}]")


if __name__ == "__main__":
    server.run()
