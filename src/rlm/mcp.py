"""Supervisor-owned MCP discovery, registration, and invocation."""

from __future__ import annotations

import hmac
import inspect
import json
import keyword
import re
import secrets
from collections.abc import AsyncIterator, Iterable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import Tool
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator

from rlm.broker import json_annotation

MAX_MCP_SKILL_NAME_CHARS = 96
MAX_MCP_DESCRIPTOR_BYTES = 1024 * 1024
MAX_MCP_CATALOG_BYTES = 64 * 1024 * 1024
MAX_MCP_DISCOVERY_PAGES = 4096


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
_MCP_SERVERS_ADAPTER = TypeAdapter(dict[str, MCPServer])


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
class _CatalogMCPTool:
    server_name: str
    tool: Tool


class _CatalogError(ValueError):
    """Invalid or oversized public MCP catalog metadata."""


class MCPToolError(RuntimeError):
    """An error intentionally returned by an MCP tool."""


class MCPRegistry:
    """Keep MCP transport configuration private and expose opaque tool capabilities."""

    def __init__(self, servers: dict[str, MCPServer], cwd: str | None = None) -> None:
        self._servers = validate_mcp_servers(servers)
        self._cwd = cwd
        self._tools: dict[str, _CatalogMCPTool] | None = None
        self._loaded: dict[str, MCPToolDescriptor] = {}
        self._capabilities: dict[str, str] = {}
        self._server_descriptions: dict[str, str] = {}
        self._index: dict[str, dict[str, int]] = {}
        self._cursor_key = secrets.token_bytes(32)

    async def discover(self) -> None:
        if self._tools is not None:
            return

        catalog: dict[str, _CatalogMCPTool] = {}
        server_descriptions: dict[str, str] = {}
        catalog_bytes = 0
        for server_name, server in self._servers.items():
            catalog_error: _CatalogError | None = None
            try:
                async with _client_session(server, self._cwd) as session:
                    info = await session.initialize()
                    server_descriptions[server_name] = (
                        getattr(info.serverInfo, "description", None)
                        or info.instructions
                        or ""
                    )[:1000]
                    cursor = None
                    cursors: set[str] = set()
                    for _ in range(MAX_MCP_DISCOVERY_PAGES):
                        page = await session.list_tools(cursor=cursor)
                        for tool in page.tools:
                            try:
                                catalog_bytes = _add_catalog_tool(
                                    catalog, server_name, tool, catalog_bytes
                                )
                            except _CatalogError as exc:
                                # Close the SDK task group normally before reporting metadata errors.
                                catalog_error = exc
                                break
                        if catalog_error is not None:
                            break
                        cursor = page.nextCursor
                        if cursor is None:
                            break
                        if cursor in cursors:
                            raise ValueError("MCP pagination cursor repeated")
                        catalog_bytes += len(cursor.encode())
                        if catalog_bytes > MAX_MCP_CATALOG_BYTES:
                            catalog_error = _CatalogError(
                                "MCP catalog pagination metadata exceeds "
                                f"{MAX_MCP_CATALOG_BYTES} bytes"
                            )
                            break
                        cursors.add(cursor)
                    else:
                        catalog_error = _CatalogError(
                            f"MCP tool listing exceeded {MAX_MCP_DISCOVERY_PAGES} pages"
                        )
            except Exception:
                raise RuntimeError(
                    f"MCP server {server_name!r} discovery failed"
                ) from None
            if catalog_error is not None:
                raise catalog_error
        self._tools = catalog
        self._server_descriptions = server_descriptions
        for name, entry in catalog.items():
            for token in _search_tokens(entry.tool.description or ""):
                self._index.setdefault(token, {})[name] = 1
            for token in _search_tokens(name):
                self._index.setdefault(token, {})[name] = 4

    def servers(self) -> list[dict[str, Any]]:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        counts = dict.fromkeys(self._servers, 0)
        for entry in self._tools.values():
            counts[entry.server_name] += 1
        return [
            {
                "name": name,
                "tool_count": count,
                **(
                    {"description": self._server_descriptions[name]}
                    if self._server_descriptions[name]
                    else {}
                ),
            }
            for name, count in counts.items()
        ]

    def get(self, name: str) -> MCPToolDescriptor:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        if name in self._loaded:
            return self._loaded[name]
        try:
            entry = self._tools[name]
        except KeyError:
            raise ValueError(f"unknown MCP tool: {name!r}") from None
        descriptor = MCPToolDescriptor(
            capability=secrets.token_urlsafe(24),
            name=name,
            description=entry.tool.description or f"MCP tool {entry.tool.name!r}.",
            input_schema=dict(entry.tool.inputSchema),
        )
        self._loaded[name] = descriptor
        self._capabilities[descriptor.capability] = name
        return descriptor

    def search(
        self,
        query: str,
        limit: int = 5,
        server: str | None = None,
        cursor: str | None = None,
    ) -> dict[str, Any]:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        if server is not None and server not in self._servers:
            raise ValueError(f"unknown MCP server: {server!r}")
        offset = 0
        if cursor is not None:
            raw_offset = cursor.partition(":")[0]
            if not cursor.isascii() or not raw_offset.isdecimal():
                raise ValueError("invalid MCP search cursor")
            offset = int(raw_offset)
            if not secrets.compare_digest(
                cursor, self._search_cursor(query, server, offset)
            ):
                raise ValueError("invalid MCP search cursor for this query or server")
        scores: dict[str, float] = {}
        for token in sorted(_search_tokens(query)):
            matches = self._index.get(token, {})
            weight = log(1 + len(self._tools) / (1 + len(matches)))
            for name, boost in matches.items():
                if server is None or self._tools[name].server_name == server:
                    scores[name] = scores.get(name, 0) + weight * boost
        ranked = sorted(scores, key=lambda name: (-scores[name], name))
        matches = [
            {
                "id": name,
                "server": self._tools[name].server_name,
                "description": (
                    self._tools[name].tool.description
                    or f"MCP tool {self._tools[name].tool.name!r}."
                )[:1000],
            }
            for name in ranked[offset : offset + limit]
        ]
        next_offset = offset + len(matches)
        return {
            "matches": matches,
            "next_cursor": self._search_cursor(query, server, next_offset)
            if next_offset < len(ranked)
            else None,
        }

    def _search_cursor(self, query: str, server: str | None, offset: int) -> str:
        payload = json.dumps([query, server, offset], separators=(",", ":")).encode()
        signature = hmac.digest(self._cursor_key, payload, "sha256").hex()
        return f"{offset}:{signature}"

    def descriptor(self, capability: str) -> MCPToolDescriptor:
        if self._tools is None:
            raise RuntimeError("MCP tools have not been discovered")
        try:
            return self._loaded[self._capabilities[capability]]
        except KeyError as exc:
            raise PermissionError("unknown MCP tool capability") from exc

    async def call(self, capability: str, arguments: dict[str, Any]) -> str:
        descriptor = self.descriptor(capability)
        entry = self._tools[descriptor.name]
        try:
            return await call_tool(
                self._servers[entry.server_name],
                entry.tool.name,
                arguments,
                self._cwd,
            )
        except MCPToolError as exc:
            raise RuntimeError(str(exc)) from None
        except Exception:
            raise RuntimeError(f"MCP tool {descriptor.name!r} is unavailable") from None


def _add_catalog_tool(
    catalog: dict[str, _CatalogMCPTool],
    server_name: str,
    tool: Tool,
    catalog_bytes: int,
) -> int:
    name = _skill_name(server_name, tool.name)
    if not name.isidentifier() or keyword.iskeyword(name):
        raise _CatalogError(f"MCP tool name {name!r} is not a Python identifier")
    if len(name) > MAX_MCP_SKILL_NAME_CHARS:
        raise _CatalogError(
            f"MCP tool name exceeds {MAX_MCP_SKILL_NAME_CHARS} characters"
        )
    if name in catalog:
        raise _CatalogError(f"duplicate normalized MCP tool name: {name!r}")
    size = len(tool.model_dump_json(by_alias=True, exclude_none=True).encode())
    if size > MAX_MCP_DESCRIPTOR_BYTES:
        raise _CatalogError(
            f"MCP public tool metadata exceeds {MAX_MCP_DESCRIPTOR_BYTES} bytes"
        )
    catalog_bytes += size
    if catalog_bytes > MAX_MCP_CATALOG_BYTES:
        raise _CatalogError(
            f"MCP catalog metadata exceeds {MAX_MCP_CATALOG_BYTES} bytes"
        )
    catalog[name] = _CatalogMCPTool(server_name, tool)
    return catalog_bytes


def _search_tokens(text: str) -> set[str]:
    text = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", text)
    return set(re.findall(r"[^\W_]+", text.lower()))


def _skill_name(server: str, tool: str) -> str:
    """Return the normalized Python name for a server tool; an unnamed server keeps tool names bare."""
    ident = re.sub(r"\W", "_", f"{server}_{tool}" if server else tool)
    return f"_{ident}" if ident[:1].isdigit() else ident


def validate_mcp_servers(servers: dict[str, Any]) -> dict[str, MCPServer]:
    return _MCP_SERVERS_ADAPTER.validate_python(servers)


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
            annotation=json_annotation(prop.get("type")),
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
