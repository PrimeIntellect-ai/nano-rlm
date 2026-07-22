"""MCP tool servers exposed as pre-imported IPython skills.

A v1 harness can wire task-specific tool servers to the agent over MCP. rlm has no MCP
client in its native tool-call loop; instead each MCP tool becomes a *skill* — a generated
async function the agent calls straight from the IPython REPL (``await tools_add_event(...)``),
the same programmatic tool-call (PTC) path as installed skills. The servers arrive as a
standard ``mcpServers`` URL map in ``RLM_MCP_CONFIG`` (set by the verifiers rlm harness).

Each tool is written to its own flat module in the session directory (added to the kernel's
``sys.path``); the module's ``run`` carries a signature built from the tool's input schema so
``help()`` / ``inspect.signature`` show the real arguments.
"""

from __future__ import annotations

import inspect
import json
import os
import re
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool

MCP_CONFIG_ENV = "RLM_MCP_CONFIG"

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}


MCPServer = str | dict[str, Any]


def load_mcp_servers() -> dict[str, MCPServer]:
    """Parse ``RLM_MCP_CONFIG`` into normalized streamable HTTP servers."""
    raw = os.environ.get(MCP_CONFIG_ENV)
    servers = json.loads(raw)["mcpServers"] if raw else {}
    return {
        name: (
            {"url": spec["url"], "headers": spec["headers"]}
            if spec.get("headers")
            else spec["url"]
        )
        for name, spec in servers.items()
    }


def _skill_name(server: str, tool: str) -> str:
    """A valid Python identifier (importable + REPL call name) for a server's tool."""
    ident = re.sub(r"\W", "_", f"{server}_{tool}")
    return f"_{ident}" if ident[:1].isdigit() else ident


def _server_connection(server: MCPServer) -> tuple[str, dict[str, str]]:
    if isinstance(server, str):
        return server, {}
    return str(server["url"]), dict(server.get("headers") or {})


def dump_mcp_servers(servers: dict[str, MCPServer]) -> str:
    """Serialize servers as a standard ``mcpServers`` configuration."""
    specs = {}
    for name, server in servers.items():
        url, headers = _server_connection(server)
        specs[name] = {"url": url, **({"headers": headers} if headers else {})}
    return json.dumps({"mcpServers": specs})


async def discover_tools(
    servers: dict[str, MCPServer],
) -> dict[str, tuple[str, Tool]]:
    """List each server's tools using one discovery session per server."""
    found: dict[str, tuple[str, Tool]] = {}
    for server, spec in servers.items():
        url, headers = _server_connection(spec)
        async with (
            streamablehttp_client(url, headers=headers or None) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            await session.initialize()
            for tool in (await session.list_tools()).tools:
                found[_skill_name(server, tool.name)] = (server, tool)
    return found


async def call_tool(
    url: str, name: str, arguments: dict, headers: dict[str, str] | None = None
) -> str:
    """Call an MCP tool over streamable HTTP and return its text content.

    Raises ``RuntimeError`` on a tool-reported error, so a failed call surfaces as an
    exception in the REPL rather than a silently-wrong return value.
    """
    async with (
        streamablehttp_client(url, headers=headers) as (read, write, _),
        ClientSession(read, write) as session,
    ):
        await session.initialize()
        result = await session.call_tool(name, arguments or {})
    text = "\n".join(
        getattr(block, "text", "") or str(block) for block in result.content
    )
    if result.isError:
        raise RuntimeError(text or f"MCP tool {name!r} failed")
    return text


def build_signature(schema: dict) -> inspect.Signature:
    """A keyword-only signature mirroring a tool's input schema (required params first).

    Non-identifier property names are skipped (still reachable via ``**kwargs``).
    """
    properties, required = schema.get("properties", {}), set(schema.get("required", []))
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required else None,
            annotation=_JSON_TO_PY.get(prop.get("type"), inspect.Parameter.empty),
        )
        for name, prop in properties.items()
        if name.isidentifier()
    ]
    params.sort(key=lambda p: p.default is not inspect.Parameter.empty)
    return inspect.Signature(params)


def make_skill(server: str, tool: Tool):
    """Build the async ``run`` for a generated skill module from an MCP ``Tool``.

    Generated modules are a single statement (``run = make_skill(server, Tool(...))``). The
    returned coroutine forwards keyword args to the tool; its ``__signature__`` and
    ``__doc__`` come from the tool so ``help()`` / ``inspect.signature`` expose the real API.
    """

    async def run(**kwargs):
        try:
            spec = load_mcp_servers()[server]
        except KeyError as exc:
            raise RuntimeError(f"MCP server {server!r} is not configured") from exc
        url, headers = _server_connection(spec)
        return await call_tool(url, tool.name, kwargs, headers)

    run.__signature__ = build_signature(tool.inputSchema)
    run.__doc__ = tool.description or f"MCP tool {tool.name!r}."
    return run


_MODULE_TEMPLATE = '''\
"""{summary}"""

from mcp.types import Tool

from rlm.mcp import make_skill

run = make_skill({server!r}, Tool.model_validate({tool!r}))
'''


def write_skill_modules(
    found: dict[str, tuple[str, Tool]], dest_dir: Path
) -> list[str]:
    """Write one importable ``<skill>.py`` per discovered tool into ``dest_dir``.

    Returns the generated skill names (importable once ``dest_dir`` is on ``sys.path``).
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    for name, (server, tool) in found.items():
        summary = next(
            iter((tool.description or "").splitlines()), f"MCP tool {tool.name}."
        )
        source = _MODULE_TEMPLATE.format(
            summary=summary.replace('"""', "'''"),
            server=server,
            tool=tool.model_dump(mode="json"),
        )
        (dest_dir / f"{name}.py").write_text(source)
    return list(found)


async def generate_mcp_skills(
    servers: dict[str, MCPServer], dest_dir: Path
) -> list[str]:
    """Discover all tools on ``servers`` and write them as skill modules in ``dest_dir``."""
    return write_skill_modules(await discover_tools(servers), dest_dir)


def list_skill_modules(skills_dir: Path) -> list[str]:
    """Names of the MCP-skill modules in ``skills_dir`` (its top-level ``.py`` files).

    The directory the kernel imports from is the source of truth, so callers read it back
    instead of threading the generated names around.
    """
    return sorted(path.stem for path in skills_dir.glob("*.py"))
