"""Agent Client Protocol transport for RLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Any

from acp import (
    PROTOCOL_VERSION,
    Agent,
    InitializeResponse,
    NewSessionResponse,
    PromptResponse,
    RequestError,
    run_agent,
    text_block,
    update_agent_message,
)
from acp.interfaces import Client
from acp.schema import (
    AcpMcpServer,
    AgentCapabilities,
    AudioContentBlock,
    ClientCapabilities,
    CloseSessionResponse,
    EmbeddedResourceContentBlock,
    HttpMcpServer,
    ImageContentBlock,
    Implementation,
    McpCapabilities,
    McpServerStdio,
    PromptCapabilities,
    ResourceContentBlock,
    SessionCapabilities,
    SessionCloseCapabilities,
    SseMcpServer,
    TextContentBlock,
    Usage,
)

from rlm.engine import RLMEngine
from rlm.mcp import MCPServer
from rlm.session import Session


@dataclass
class _SessionState:
    engine: RLMEngine
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_task: asyncio.Task | None = None
    closing: bool = False


def _mcp_servers(
    servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None,
) -> dict[str, MCPServer]:
    resolved: dict[str, MCPServer] = {}
    for server in servers or []:
        if isinstance(server, HttpMcpServer):
            headers = {header.name: header.value for header in server.headers}
            resolved[server.name] = (
                {"url": server.url, "headers": headers} if headers else server.url
            )
        elif isinstance(server, McpServerStdio):
            resolved[server.name] = {
                "command": server.command,
                "args": list(server.args),
                "env": {item.name: item.value for item in server.env},
            }
        else:
            raise RequestError.invalid_params(
                {"reason": "RLM supports stdio and streamable HTTP MCP servers only"}
            )
    return resolved


def _prompt_text(
    prompt: list[
        TextContentBlock
        | ImageContentBlock
        | AudioContentBlock
        | ResourceContentBlock
        | EmbeddedResourceContentBlock
    ],
) -> str:
    if any(not isinstance(block, TextContentBlock) for block in prompt):
        raise RequestError.invalid_params(
            {"reason": "RLM currently accepts text prompt blocks only"}
        )
    text = "".join(block.text for block in prompt)
    if not text:
        raise RequestError.invalid_params({"reason": "prompt has no text"})
    return text


class RLMACPAgent(Agent):
    """Expose persistent RLM engines as ACP sessions."""

    def __init__(self) -> None:
        self._client: Client
        self._sessions: dict[str, _SessionState] = {}

    def on_connect(self, conn: Client) -> None:
        self._client = conn

    async def initialize(
        self,
        protocol_version: int,
        client_capabilities: ClientCapabilities | None = None,
        client_info: Implementation | None = None,
        **kwargs: Any,
    ) -> InitializeResponse:
        return InitializeResponse(
            protocol_version=PROTOCOL_VERSION,
            agent_capabilities=AgentCapabilities(
                prompt_capabilities=PromptCapabilities(),
                mcp_capabilities=McpCapabilities(http=True),
                session_capabilities=SessionCapabilities(
                    close=SessionCloseCapabilities()
                ),
            ),
            agent_info=Implementation(name="rlm", title="RLM", version=version("rlm")),
        )

    async def new_session(
        self,
        cwd: str,
        additional_directories: list[str] | None = None,
        mcp_servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio]
        | None = None,
        **kwargs: Any,
    ) -> NewSessionResponse:
        if additional_directories:
            raise RequestError.invalid_params(
                {"reason": "RLM does not support additional session directories"}
            )
        session = Session()
        session_id = session.dir.name
        self._sessions[session_id] = _SessionState(
            engine=RLMEngine(
                cwd=cwd,
                session=session,
                mcp_servers=_mcp_servers(mcp_servers),
            )
        )
        return NewSessionResponse(session_id=session_id)

    async def prompt(
        self,
        session_id: str,
        prompt: list[
            TextContentBlock
            | ImageContentBlock
            | AudioContentBlock
            | ResourceContentBlock
            | EmbeddedResourceContentBlock
        ],
        **kwargs: Any,
    ) -> PromptResponse:
        state = self._sessions.get(session_id)
        if state is None:
            raise RequestError.resource_not_found(session_id)

        async with state.lock:
            if state.closing:
                raise RequestError.resource_not_found(session_id)
            task = asyncio.create_task(state.engine.prompt(_prompt_text(prompt)))
            state.prompt_task = task
            try:
                result = await task
            except asyncio.CancelledError:
                return PromptResponse(stop_reason="cancelled")
            finally:
                state.prompt_task = None

            await self._client.session_update(
                session_id=session_id,
                update=update_agent_message(text_block(result.answer)),
            )
            stop_reason = (
                "max_tokens"
                if state.engine.stop_reason == "token_budget"
                else "end_turn"
            )
            return PromptResponse(
                stop_reason=stop_reason,
                usage=Usage(
                    total_tokens=result.usage.total,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                ),
            )

    async def cancel(self, session_id: str, **kwargs: Any) -> None:
        state = self._sessions.get(session_id)
        if state is not None and state.prompt_task is not None:
            state.prompt_task.cancel()

    async def close_session(
        self, session_id: str, **kwargs: Any
    ) -> CloseSessionResponse:
        state = self._sessions.pop(session_id, None)
        if state is None:
            raise RequestError.resource_not_found(session_id)
        state.closing = True
        if state.prompt_task is not None:
            state.prompt_task.cancel()
        async with state.lock:
            state.engine.close()
        return CloseSessionResponse()

    async def shutdown(self) -> None:
        for session_id in list(self._sessions):
            await self.close_session(session_id)


async def serve_acp() -> None:
    """Serve RLM over ACP on stdin/stdout until the client disconnects."""
    agent = RLMACPAgent()
    try:
        # agent-client-protocol 0.11 gates session/close routing behind this flag.
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.shutdown()
