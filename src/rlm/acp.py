"""Agent Client Protocol transport for RLM."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from importlib.metadata import version
from typing import Annotated, Any

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
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from rlm.engine import RLMEngine
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.mcp import MCPHTTPServer, MCPServer, MCPStdioServer
from rlm.session import Session

SESSION_METADATA_KEY = "ai.prime.rlm/session-v1"
RUNTIME_METADATA_KEY = "ai.prime.rlm/runtime-v1"


class _ContractModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class _RuntimeMetadata(_ContractModel):
    session_id: str = Field(
        pattern=r"^[A-Za-z0-9._:-]{1,128}$",
    )
    model: str = Field(min_length=1)
    provider: ProviderConfig
    policy: ExecutionPolicy
    system_prompt_path: str | None
    append_to_system_prompt: str | None
    skills: list[Annotated[str, Field(min_length=1)]]
    kernel_env: dict[str, str]
    search_api_key: str | None


class _UsageSnapshot(_ContractModel):
    prompt_tokens: int = Field(ge=0)
    completion_tokens: int = Field(ge=0)
    total_tokens: int = Field(ge=0)


class _ProgrammaticToolCallSnapshot(_ContractModel):
    python_total: int = Field(ge=0)
    bash_total: int = Field(ge=0)
    by_tool_python: dict[str, int]
    by_tool_bash: dict[str, int]


class _SupervisorSnapshot(_ContractModel):
    subagent_calls: int = Field(ge=0)
    active_subagent_calls: int = Field(ge=0)


class _LimitsSnapshot(_ContractModel):
    max_depth: int = Field(ge=0)
    max_concurrent_subagents: int = Field(gt=0)
    max_subagent_calls: int = Field(gt=0)
    max_tokens: int | None = Field(default=None, gt=0)
    summarize_at_tokens: int | None = Field(default=None, gt=0)
    max_compactions: int | None = Field(default=None, gt=0)
    max_tool_output_chars: int | None = Field(default=None, gt=0)
    allow_git: bool


class _SessionSnapshot(_ContractModel):
    session_id: str = Field(pattern=r"^[A-Za-z0-9._:-]{1,128}$")
    last_stop_reason: str | None
    model: str = Field(min_length=1)
    turns: int = Field(ge=0)
    usage: _UsageSnapshot
    metrics: dict[str, int | float]
    programmatic_tool_call_stats: _ProgrammaticToolCallSnapshot
    supervisor: _SupervisorSnapshot
    limits: _LimitsSnapshot


@dataclass
class _SessionState:
    engine: RLMEngine
    session_id: str
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_task: asyncio.Task | None = None
    delivery_task: asyncio.Task | None = None
    close_task: asyncio.Task[dict[str, Any]] | None = None
    closing: bool = False
    last_stop_reason: str | None = None


def _request_is_cancelling(awaited: asyncio.Task[Any]) -> bool:
    current = asyncio.current_task()
    cancelling = getattr(current, "cancelling", None)
    if cancelling is not None:
        return bool(cancelling())
    return not awaited.cancelled()


async def _cancel_and_wait(task: asyncio.Task[Any]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _session_metadata(state: _SessionState) -> dict[str, Any]:
    snapshot = {
        "session_id": state.session_id,
        "last_stop_reason": state.last_stop_reason,
        **state.engine.execution_snapshot(),
    }
    validated = _SessionSnapshot.model_validate(snapshot)
    return {SESSION_METADATA_KEY: validated.model_dump(mode="json")}


def _validation_fields(error: ValidationError) -> list[str]:
    return [".".join(str(part) for part in item["loc"]) for item in error.errors()]


def _runtime_config(field_meta: Any) -> tuple[RuntimeConfig, str]:
    if not isinstance(field_meta, dict):
        raise RequestError.invalid_params({"reason": "ACP _meta must be an object"})
    try:
        payload = _RuntimeMetadata.model_validate(
            field_meta.get(RUNTIME_METADATA_KEY),
        )
    except ValidationError as error:
        raise RequestError.invalid_params(
            {
                "reason": (
                    f"{RUNTIME_METADATA_KEY} has invalid fields: "
                    f"{_validation_fields(error)}"
                )
            }
        ) from error

    return (
        RuntimeConfig(
            model=payload.model,
            provider=payload.provider,
            invocation=InvocationContext(),
            policy=payload.policy,
            system_prompt_path=payload.system_prompt_path,
            append_to_system_prompt=payload.append_to_system_prompt,
            skills=tuple(payload.skills),
            kernel_env=tuple(payload.kernel_env.items()),
            search_api_key=payload.search_api_key,
        ),
        payload.session_id,
    )


def _mcp_servers(
    servers: list[HttpMcpServer | SseMcpServer | AcpMcpServer | McpServerStdio] | None,
) -> dict[str, MCPServer]:
    resolved: dict[str, MCPServer] = {}
    for server in servers or []:
        if isinstance(server, HttpMcpServer):
            headers = {header.name: header.value for header in server.headers}
            resolved[server.name] = MCPHTTPServer(
                url=server.url,
                headers=headers,
            )
        elif isinstance(server, McpServerStdio):
            resolved[server.name] = MCPStdioServer(
                command=server.command,
                args=list(server.args),
                env={item.name: item.value for item in server.env},
            )
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
        resolved_mcp_servers = _mcp_servers(mcp_servers)
        runtime_config, external_session_id = _runtime_config(kwargs)
        session = Session()
        session_id = session.dir.name
        try:
            engine = RLMEngine(
                cwd=cwd,
                session=session,
                mcp_servers=resolved_mcp_servers,
                runtime_config=runtime_config,
            )
        except BaseException:
            session.close()
            raise
        state = _SessionState(engine=engine, session_id=external_session_id)
        self._sessions[session_id] = state
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
                result = await asyncio.shield(task)
            except asyncio.CancelledError:
                if _request_is_cancelling(task):
                    await _cancel_and_wait(task)
                    raise
                state.last_stop_reason = "cancelled"
                return PromptResponse(stop_reason="cancelled")
            except Exception:
                state.last_stop_reason = "error"
                raise
            finally:
                state.prompt_task = None

            stop_reason = (
                "max_tokens"
                if state.engine.stop_reason == "token_budget"
                else "end_turn"
            )
            state.last_stop_reason = state.engine.stop_reason or stop_reason
            if state.closing:
                return PromptResponse(stop_reason="cancelled")

            delivery = asyncio.create_task(
                self._client.session_update(
                    session_id=session_id,
                    update=update_agent_message(text_block(result.answer)),
                )
            )
            state.delivery_task = delivery
            try:
                await asyncio.shield(delivery)
            except asyncio.CancelledError:
                if _request_is_cancelling(delivery):
                    await _cancel_and_wait(delivery)
                    raise
                if not state.closing:
                    raise
                return PromptResponse(stop_reason="cancelled")
            finally:
                state.delivery_task = None

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
        state = self._sessions.get(session_id)
        if state is None:
            raise RequestError.resource_not_found(session_id)
        if state.close_task is None:
            state.closing = True
            state.close_task = asyncio.create_task(
                self._close_session(session_id, state)
            )
        metadata = await asyncio.shield(state.close_task)
        return CloseSessionResponse(field_meta=metadata)

    async def _close_session(
        self, session_id: str, state: _SessionState
    ) -> dict[str, Any]:
        try:
            if state.prompt_task is not None:
                state.prompt_task.cancel()
            if state.delivery_task is not None:
                state.delivery_task.cancel()
            async with state.lock:
                await state.engine.aclose()
            return _session_metadata(state)
        finally:
            if self._sessions.get(session_id) is state:
                self._sessions.pop(session_id)

    async def shutdown(self) -> None:
        results = await asyncio.gather(
            *(self.close_session(session_id) for session_id in list(self._sessions)),
            return_exceptions=True,
        )
        if error := next(
            (result for result in results if isinstance(result, BaseException)), None
        ):
            raise error


async def serve_acp() -> None:
    """Serve RLM over ACP on stdin/stdout until the client disconnects."""
    agent = RLMACPAgent()
    try:
        # session/close is currently part of ACP's unstable extension set.
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.shutdown()
