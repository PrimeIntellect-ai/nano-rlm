"""Agent Client Protocol transport for RLM."""

from __future__ import annotations

import asyncio
import re
from dataclasses import dataclass, field, replace
from importlib.metadata import version
from typing import Any, Literal

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
from rlm.config import ProviderConfig, RuntimeConfig
from rlm.mcp import MCPServer
from rlm.session import Session
from rlm.tools.ipython import RESERVED_KERNEL_ENV_NAMES

SESSION_METADATA_KEY = "ai.prime.rlm/session-v1"
CAPABILITIES_METADATA_KEY = "ai.prime.rlm/capabilities-v1"
RUNTIME_METADATA_KEY = "ai.prime.rlm/runtime-v1"
SessionStatus = Literal["created", "idle", "closing", "closed"]
_LINEAGE_ID_RE = re.compile(r"[A-Za-z0-9._:-]{1,128}")


@dataclass
class _SessionState:
    engine: RLMEngine
    lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    prompt_task: asyncio.Task | None = None
    delivery_task: asyncio.Task | None = None
    close_task: asyncio.Task[dict[str, Any]] | None = None
    closing: bool = False
    last_stop_reason: str | None = None
    snapshot_sequence: int = 0


def _request_is_cancelling(awaited: asyncio.Task[Any]) -> bool:
    current = asyncio.current_task()
    cancelling = getattr(current, "cancelling", None)
    if cancelling is not None:
        return bool(cancelling())
    return not awaited.cancelled()


async def _cancel_and_wait(task: asyncio.Task[Any]) -> None:
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)


def _session_metadata(
    state: _SessionState, status: SessionStatus, *, final: bool = False
) -> dict[str, Any]:
    snapshot = {
        "schema_version": 1,
        "sequence": state.snapshot_sequence,
        "status": status,
        "final": final,
        "last_stop_reason": state.last_stop_reason,
        **state.engine.execution_snapshot(),
    }
    state.snapshot_sequence += 1
    return {SESSION_METADATA_KEY: snapshot}


def _runtime_config(
    field_meta: Any,
) -> tuple[RuntimeConfig, str | None]:
    config = RuntimeConfig.from_env()
    if field_meta is None:
        return config, None
    if not isinstance(field_meta, dict):
        raise RequestError.invalid_params({"reason": "ACP _meta must be an object"})
    payload = field_meta.get(RUNTIME_METADATA_KEY)
    if payload is None:
        return config, None
    if not isinstance(payload, dict):
        raise RequestError.invalid_params(
            {"reason": f"{RUNTIME_METADATA_KEY} must be an object"}
        )
    unknown = sorted(
        set(payload)
        - {"lineage_session_id", "provider", "kernel_env", "search_api_key"}
    )
    if unknown:
        raise RequestError.invalid_params(
            {"reason": f"{RUNTIME_METADATA_KEY} has unknown fields: {unknown}"}
        )

    lineage_session_id = payload.get("lineage_session_id")
    if lineage_session_id is not None and (
        not isinstance(lineage_session_id, str)
        or _LINEAGE_ID_RE.fullmatch(lineage_session_id) is None
    ):
        raise RequestError.invalid_params(
            {"reason": "lineage_session_id must be a bounded opaque identifier"}
        )

    provider = config.provider
    if "provider" in payload:
        provider_payload = payload["provider"]
        if not isinstance(provider_payload, dict):
            raise RequestError.invalid_params({"reason": "provider must be an object"})
        provider_unknown = sorted(
            set(provider_payload) - {"base_url", "api_key", "headers", "max_retries"}
        )
        if provider_unknown:
            raise RequestError.invalid_params(
                {"reason": f"provider has unknown fields: {provider_unknown}"}
            )
        base_url = provider_payload.get("base_url")
        api_key = provider_payload.get("api_key")
        headers = provider_payload.get("headers", {})
        max_retries = provider_payload.get("max_retries", provider.max_retries)
        if base_url is not None and not isinstance(base_url, str):
            raise RequestError.invalid_params(
                {"reason": "provider base_url is invalid"}
            )
        if not isinstance(api_key, str) or not api_key:
            raise RequestError.invalid_params({"reason": "provider api_key is invalid"})
        if not isinstance(headers, dict) or not all(
            isinstance(key, str) and isinstance(value, str)
            for key, value in headers.items()
        ):
            raise RequestError.invalid_params(
                {"reason": "provider headers are invalid"}
            )
        if (
            not isinstance(max_retries, int)
            or isinstance(max_retries, bool)
            or max_retries < 0
        ):
            raise RequestError.invalid_params(
                {"reason": "provider max_retries is invalid"}
            )
        try:
            provider = ProviderConfig(base_url, api_key, headers, max_retries)
        except ValueError as error:
            raise RequestError.invalid_params({"reason": str(error)}) from error

    kernel_env = payload.get("kernel_env")
    if kernel_env is None:
        resolved_kernel_env = config.kernel_env
    elif not isinstance(kernel_env, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in kernel_env.items()
    ):
        raise RequestError.invalid_params({"reason": "kernel_env must map strings"})
    else:
        resolved_kernel_env = tuple(kernel_env.items())
        reserved_kernel_env = sorted(RESERVED_KERNEL_ENV_NAMES.intersection(kernel_env))
        if reserved_kernel_env:
            raise RequestError.invalid_params(
                {"reason": f"kernel_env contains reserved names: {reserved_kernel_env}"}
            )

    search_api_key = payload.get("search_api_key", config.search_api_key)
    if search_api_key is not None and not isinstance(search_api_key, str):
        raise RequestError.invalid_params({"reason": "search_api_key is invalid"})
    return (
        replace(
            config,
            provider=provider,
            kernel_env=resolved_kernel_env,
            search_api_key=search_api_key,
        ),
        lineage_session_id,
    )


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
            field_meta={CAPABILITIES_METADATA_KEY: {"session_snapshot_versions": [1]}},
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
        runtime_config, lineage_session_id = _runtime_config(kwargs)
        session = Session()
        session_id = session.dir.name
        try:
            engine = RLMEngine(
                cwd=cwd,
                session=session,
                mcp_servers=resolved_mcp_servers,
                runtime_config=runtime_config,
                lineage_session_id=lineage_session_id,
            )
        except BaseException:
            session.close()
            raise
        state = _SessionState(engine=engine)
        self._sessions[session_id] = state
        return NewSessionResponse(
            session_id=session_id,
            field_meta=_session_metadata(state, "created"),
        )

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
                return PromptResponse(
                    stop_reason="cancelled",
                    field_meta=_session_metadata(
                        state, "closing" if state.closing else "idle"
                    ),
                )
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
                return PromptResponse(
                    stop_reason="cancelled",
                    field_meta=_session_metadata(state, "closing"),
                )

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
                return PromptResponse(
                    stop_reason="cancelled",
                    field_meta=_session_metadata(state, "closing"),
                )
            finally:
                state.delivery_task = None

            return PromptResponse(
                stop_reason=stop_reason,
                usage=Usage(
                    total_tokens=result.usage.total,
                    input_tokens=result.usage.prompt_tokens,
                    output_tokens=result.usage.completion_tokens,
                ),
                field_meta=_session_metadata(state, "idle"),
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
            return _session_metadata(state, "closed", final=True)
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
        # agent-client-protocol 0.11 gates session/close routing behind this flag.
        await run_agent(agent, use_unstable_protocol=True)
    finally:
        await agent.shutdown()
