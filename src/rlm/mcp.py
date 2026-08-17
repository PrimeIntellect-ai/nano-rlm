"""Supervisor-owned MCP discovery, registration, and invocation."""

from __future__ import annotations

import inspect
import json
import keyword
import os
import re
import secrets
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client

MCP_CONFIG_ENV = "RLM_MCP_CONFIG"
MAX_MCP_TOOLS = 128
MAX_MCP_SKILL_NAME_CHARS = 96
MAX_MCP_DESCRIPTOR_BYTES = 1024 * 1024

_JSON_TO_PY = {
    "string": str,
    "integer": int,
    "number": float,
    "boolean": bool,
    "array": list,
    "object": dict,
}

MCPServer = str | dict[str, Any]


@dataclass(frozen=True)
class MCPToolDescriptor:
    """Public information exposed to an IPython kernel for one MCP tool."""

    capability: str
    name: str
    description: str
    input_schema: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "capability": self.capability,
            "name": self.name,
            "description": self.description,
            "input_schema": self.input_schema,
        }


@dataclass(frozen=True)
class _RegisteredMCPTool:
    descriptor: MCPToolDescriptor
    server: MCPServer
    tool_name: str


class MCPToolError(RuntimeError):
    """An error intentionally returned by an MCP tool."""


class MCPRegistry:
    """Keep MCP transport configuration private and expose opaque tool capabilities."""

    def __init__(self, servers: dict[str, MCPServer], cwd: str | None = None) -> None:
        self._servers = {
            name: _normalize_server(spec) for name, spec in servers.items()
        }
        self._cwd = cwd
        self._tools: dict[str, _RegisteredMCPTool] | None = None

    async def discover(self) -> tuple[MCPToolDescriptor, ...]:
        if self._tools is not None:
            return tuple(entry.descriptor for entry in self._tools.values())

        registered: dict[str, _RegisteredMCPTool] = {}
        names: set[str] = set()
        descriptor_bytes = 0
        for server_name, server in self._servers.items():
            try:
                async with _client_session(server, self._cwd) as session:
                    await session.initialize()
                    tools = (await session.list_tools()).tools
            except Exception:
                raise RuntimeError(
                    f"MCP server {server_name!r} discovery failed"
                ) from None
            for tool in tools:
                if len(registered) >= MAX_MCP_TOOLS:
                    raise ValueError(f"MCP tool count exceeds {MAX_MCP_TOOLS}")
                name = _skill_name(server_name, tool.name)
                if not name.isidentifier() or keyword.iskeyword(name):
                    raise ValueError(
                        f"MCP tool name {name!r} is not a Python identifier"
                    )
                if len(name) > MAX_MCP_SKILL_NAME_CHARS:
                    raise ValueError(
                        f"MCP tool name exceeds {MAX_MCP_SKILL_NAME_CHARS} characters"
                    )
                if name in names:
                    raise ValueError(f"duplicate normalized MCP tool name: {name!r}")
                names.add(name)
                capability = secrets.token_urlsafe(24)
                descriptor = MCPToolDescriptor(
                    capability=capability,
                    name=name,
                    description=tool.description or f"MCP tool {tool.name!r}.",
                    input_schema=dict(tool.inputSchema),
                )
                descriptor_bytes += len(
                    json.dumps(descriptor.to_dict(), separators=(",", ":")).encode()
                )
                if descriptor_bytes > MAX_MCP_DESCRIPTOR_BYTES:
                    raise ValueError(
                        "MCP public tool metadata exceeds "
                        f"{MAX_MCP_DESCRIPTOR_BYTES} bytes"
                    )
                registered[capability] = _RegisteredMCPTool(
                    descriptor=descriptor,
                    server=server,
                    tool_name=tool.name,
                )
        self._tools = registered
        return tuple(entry.descriptor for entry in registered.values())

    def descriptor(self, capability: str) -> MCPToolDescriptor:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        try:
            return self._tools[capability].descriptor
        except KeyError as exc:
            raise PermissionError("unknown MCP tool capability") from exc

    async def call(self, capability: str, arguments: dict[str, Any]) -> str:
        descriptor = self.descriptor(capability)
        entry = self._tools[capability]
        try:
            return await call_tool(
                entry.server,
                entry.tool_name,
                arguments,
                self._cwd,
            )
        except MCPToolError as exc:
            raise RuntimeError(str(exc)) from None
        except Exception:
            raise RuntimeError(f"MCP tool {descriptor.name!r} is unavailable") from None

    def write_skill_modules(
        self, dest_dir: Path, reserved_names: Iterable[str] = ()
    ) -> list[str]:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        reserved = set(reserved_names)
        descriptors = [entry.descriptor for entry in self._tools.values()]
        conflicts = sorted(
            descriptor.name for descriptor in descriptors if descriptor.name in reserved
        )
        if conflicts:
            raise ValueError(
                f"MCP tool names conflict with existing skills: {conflicts}"
            )

        dest_dir.mkdir(parents=True, exist_ok=True)
        for descriptor in descriptors:
            source = _MODULE_TEMPLATE.format(
                description=descriptor.description,
                descriptor=descriptor.to_dict(),
            )
            (dest_dir / f"{descriptor.name}.py").write_text(source)
        return [descriptor.name for descriptor in descriptors]


def load_mcp_servers() -> dict[str, MCPServer]:
    """Parse ``RLM_MCP_CONFIG`` into normalized HTTP or stdio servers."""
    raw = os.environ.get(MCP_CONFIG_ENV)
    servers = json.loads(raw)["mcpServers"] if raw else {}
    return {name: _normalize_server(spec) for name, spec in servers.items()}


def _normalize_server(server: MCPServer) -> MCPServer:
    if isinstance(server, str):
        return server
    if "url" in server:
        headers = dict(server.get("headers") or {})
        return (
            {"url": str(server["url"]), "headers": headers}
            if headers
            else str(server["url"])
        )
    return {
        "command": str(server["command"]),
        "args": list(server.get("args") or []),
        "env": dict(server.get("env") or {}),
    }


def _skill_name(server: str, tool: str) -> str:
    """Return the normalized Python name for a server tool."""
    ident = re.sub(r"\W", "_", f"{server}_{tool}")
    return f"_{ident}" if ident[:1].isdigit() else ident


def _http_connection(server: MCPServer) -> tuple[str, dict[str, str]]:
    if isinstance(server, str):
        return server, {}
    if "url" not in server:
        raise TypeError("expected an HTTP MCP server")
    return str(server["url"]), dict(server.get("headers") or {})


def dump_mcp_servers(servers: dict[str, MCPServer]) -> str:
    """Serialize servers as a standard ``mcpServers`` configuration."""
    specs = {}
    for name, server in servers.items():
        normalized = _normalize_server(server)
        if isinstance(normalized, str):
            specs[name] = {"url": normalized}
        else:
            specs[name] = normalized
    return json.dumps({"mcpServers": specs})


@asynccontextmanager
async def _client_session(
    server: MCPServer, cwd: str | None = None
) -> AsyncIterator[ClientSession]:
    normalized = _normalize_server(server)
    if isinstance(normalized, str) or "url" in normalized:
        url, headers = _http_connection(normalized)
        async with (
            streamablehttp_client(url, headers=headers or None) as (read, write, _),
            ClientSession(read, write) as session,
        ):
            yield session
        return

    params = StdioServerParameters(
        command=normalized["command"],
        args=normalized["args"],
        env=normalized["env"],
        cwd=cwd,
    )
    async with (
        stdio_client(params) as (read, write),
        ClientSession(read, write) as session,
    ):
        yield session


async def call_tool(
    server: MCPServer,
    name: str,
    arguments: dict[str, Any],
    cwd: str | None = None,
) -> str:
    """Call an MCP tool and return its text content."""
    async with _client_session(server, cwd) as session:
        await session.initialize()
        result = await session.call_tool(name, arguments or {})
    text = "\n".join(
        getattr(block, "text", "") or str(block) for block in result.content
    )
    if result.isError:
        raise MCPToolError(text or f"MCP tool {name!r} failed")
    return text


def build_signature(schema: dict[str, Any]) -> inspect.Signature:
    """Return a keyword-only Python signature for a JSON input schema."""
    properties, required = schema.get("properties", {}), set(schema.get("required", []))
    params = [
        inspect.Parameter(
            name,
            inspect.Parameter.KEYWORD_ONLY,
            default=inspect.Parameter.empty if name in required else None,
            annotation=_JSON_TO_PY.get(prop.get("type"), inspect.Parameter.empty),
        )
        for name, prop in properties.items()
        if name.isidentifier() and not keyword.iskeyword(name)
    ]
    params.sort(key=lambda parameter: parameter.default is not inspect.Parameter.empty)
    return inspect.Signature(params)


_MODULE_TEMPLATE = """\
from rlm.broker import make_skill

__doc__ = {description!r}
__rlm_brokered__ = True
run = make_skill({descriptor!r})
"""


def list_skill_modules(skills_dir: Path) -> list[str]:
    """Return names of generated skill modules in ``skills_dir``."""
    return sorted(path.stem for path in skills_dir.glob("*.py"))
