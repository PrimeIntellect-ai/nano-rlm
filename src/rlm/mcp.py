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
from typing import Annotated, Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

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


class _MCPConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class MCPHTTPServer(_MCPConfigModel):
    url: str = Field(min_length=1)
    headers: dict[str, str] = Field(default_factory=dict, repr=False)

    @field_validator("url")
    @classmethod
    def _require_http_url(cls, url: str) -> str:
        if not url.startswith(("http://", "https://")):
            raise ValueError("MCP URL must use HTTP or HTTPS")
        return url


class MCPStdioServer(_MCPConfigModel):
    command: str = Field(min_length=1)
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] = Field(default_factory=dict, repr=False)


MCPServer = MCPHTTPServer | MCPStdioServer
_MCP_SERVERS_ADAPTER = TypeAdapter(dict[Annotated[str, Field(min_length=1)], MCPServer])


class _MCPServersDocument(_MCPConfigModel):
    mcp_servers: dict[Annotated[str, Field(min_length=1)], MCPServer] = Field(
        alias="mcpServers"
    )


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
        self._servers = validate_mcp_servers(servers)
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
        descriptors = [entry.descriptor for entry in self._tools.values()]
        return write_skill_modules(descriptors, dest_dir, reserved_names)


def load_mcp_servers() -> dict[str, MCPServer]:
    """Parse ``RLM_MCP_CONFIG`` into validated HTTP or stdio servers."""
    raw = os.environ.get(MCP_CONFIG_ENV)
    if not raw:
        return {}
    return _MCPServersDocument.model_validate_json(raw).mcp_servers


def _skill_name(server: str, tool: str) -> str:
    """Return the normalized Python name for a server tool."""
    ident = re.sub(r"\W", "_", f"{server}_{tool}")
    return f"_{ident}" if ident[:1].isdigit() else ident


def validate_mcp_servers(servers: dict[str, Any]) -> dict[str, MCPServer]:
    return _MCP_SERVERS_ADAPTER.validate_python(servers)


def dump_mcp_servers(servers: dict[str, MCPServer]) -> str:
    """Serialize servers as a standard ``mcpServers`` configuration."""
    document = _MCPServersDocument(mcpServers=validate_mcp_servers(servers))
    return document.model_dump_json(by_alias=True, exclude_defaults=True)


@asynccontextmanager
async def _client_session(
    server: MCPServer, cwd: str | None = None
) -> AsyncIterator[ClientSession]:
    if isinstance(server, MCPHTTPServer):
        async with (
            streamablehttp_client(server.url, headers=server.headers or None) as (
                read,
                write,
                _,
            ),
            ClientSession(read, write) as session,
        ):
            yield session
        return

    params = StdioServerParameters(
        command=server.command,
        args=server.args,
        env=server.env,
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


def write_skill_modules(
    descriptors: Iterable[MCPToolDescriptor],
    dest_dir: Path,
    reserved_names: Iterable[str] = (),
) -> list[str]:
    """Write public brokered-skill descriptors as importable proxy modules."""
    descriptor_list = list(descriptors)
    reserved = set(reserved_names)
    names = [descriptor.name for descriptor in descriptor_list]
    duplicates = sorted(name for name in set(names) if names.count(name) > 1)
    conflicts = sorted(name for name in names if name in reserved)
    if duplicates:
        raise ValueError(f"duplicate brokered skill names: {duplicates}")
    if conflicts:
        raise ValueError(
            f"brokered skill names conflict with existing skills: {conflicts}"
        )

    dest_dir.mkdir(parents=True, exist_ok=True)
    for descriptor in descriptor_list:
        source = _MODULE_TEMPLATE.format(
            description=descriptor.description,
            descriptor=descriptor.to_dict(),
        )
        (dest_dir / f"{descriptor.name}.py").write_text(source)
    return names


def list_skill_modules(skills_dir: Path) -> list[str]:
    """Return names of generated skill modules in ``skills_dir``."""
    return sorted(path.stem for path in skills_dir.glob("*.py"))
