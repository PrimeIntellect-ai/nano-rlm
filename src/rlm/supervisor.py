"""Trusted lifecycle manager for one recursive RLM session tree."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import tempfile
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, replace
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.broker import BrokerEndpoint, read_frame, result_to_json, write_frame
from rlm.config import RuntimeConfig
from rlm.mcp import (
    MCPRegistry,
    MCPServer,
    MCPToolDescriptor,
    write_skill_modules,
)
from rlm.session import Session
from rlm.skills.search import run_with_api_key as run_search
from rlm.types import ProgrammaticToolCallStats, RLMResult

if TYPE_CHECKING:
    from rlm.engine import RLMEngine


MAX_BROKER_CONNECTIONS = 128
BROKER_INITIAL_FRAME_TIMEOUT_SECONDS = 5


@dataclass
class _Invocation:
    id: str
    parent_id: str | None
    capability: str
    session: Session
    runtime_config: RuntimeConfig
    cwd: str
    mcp_servers: dict[str, MCPServer]


@dataclass
class _Scope:
    invocation_id: str
    parent_call_id: str | None
    segment_id: str | None
    tasks: set[asyncio.Task[Any]]


def depth_capacities(max_depth: int, limit: int, root_depth: int = 0) -> dict[int, int]:
    """Reserve capacity per descendant depth so nested calls cannot deadlock."""
    levels = max_depth - root_depth
    if levels <= 0:
        return {}
    if limit < levels:
        raise ValueError("subagent concurrency must cover every recursive depth")
    per_level, extra = divmod(limit, levels)
    return {
        depth: per_level + (1 if offset < extra else 0)
        for offset, depth in enumerate(range(root_depth + 1, max_depth + 1))
    }


class SessionTreeSupervisor:
    """Own child engines, recursion limits, and the kernel broker endpoint."""

    def __init__(
        self,
        *,
        root_session: Session,
        runtime_config: RuntimeConfig,
        cwd: str,
        mcp_servers: dict[str, MCPServer] | None = None,
        engine_factory: Callable[..., RLMEngine] | None = None,
        root_invocation_id: str | None = None,
        lineage_session_id: str | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._server: asyncio.AbstractServer | None = None
        self._broker_dir: Path | None = None
        self._socket_path: str | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._total_calls = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._child_tasks: set[asyncio.Task[RLMResult]] = set()
        self._connection_tasks: set[asyncio.Task[None]] = set()
        self._connection_writers: set[asyncio.StreamWriter] = set()
        self._scopes: dict[str, _Scope] = {}
        self._mcp_registry = MCPRegistry(mcp_servers, cwd) if mcp_servers else None
        self._brokered_skills: dict[
            str,
            tuple[
                MCPToolDescriptor,
                Callable[[dict[str, Any]], Awaitable[str]],
            ],
        ] = {}
        self._root_config = runtime_config
        if "search" in runtime_config.skills:
            capability = secrets.token_urlsafe(24)
            descriptor = MCPToolDescriptor(
                capability=capability,
                name="search",
                description=(
                    "Run a web search via Serper and return formatted title, URL, "
                    "and snippet results."
                ),
                input_schema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string"},
                        "num_results": {"type": "integer"},
                    },
                    "required": ["query"],
                },
            )
            self._brokered_skills[capability] = (descriptor, self._call_search)

        root_id = root_invocation_id or uuid.uuid4().hex
        self.session_id = lineage_session_id or root_session.dir.name
        root = _Invocation(
            id=root_id,
            parent_id=None,
            capability=secrets.token_urlsafe(32),
            session=root_session,
            runtime_config=runtime_config,
            cwd=cwd,
            mcp_servers=dict(mcp_servers or {}),
        )
        self.root_id = root_id
        self._invocations = {root_id: root}
        self._parents = {root_id: None}
        self._tool_stats: dict[str, ProgrammaticToolCallStats] = {}
        self._capabilities = {root.capability: root_id}
        capacities = depth_capacities(
            runtime_config.policy.max_depth,
            runtime_config.policy.max_concurrent_subagents,
            runtime_config.invocation.depth,
        )
        self._semaphores = {
            depth: asyncio.Semaphore(capacity) for depth, capacity in capacities.items()
        }

    @property
    def total_calls(self) -> int:
        return self._total_calls

    @property
    def active_calls(self) -> int:
        return len(self._child_tasks)

    async def start(self) -> None:
        if self._server is not None:
            return
        if self._closed:
            raise RuntimeError("session supervisor is closed")
        if self._mcp_registry is not None:
            for descriptor in await self._mcp_registry.discover():
                self._brokered_skills[descriptor.capability] = (
                    descriptor,
                    partial(self._mcp_registry.call, descriptor.capability),
                )
        self._broker_dir = Path(tempfile.mkdtemp(prefix="rlm-brk-"))
        os.chmod(self._broker_dir, 0o700)
        self._socket_path = str(self._broker_dir / "b.sock")
        self._server = await asyncio.start_unix_server(
            self._accept_connection, path=self._socket_path
        )
        os.chmod(self._socket_path, 0o600)

    def write_brokered_skill_modules(
        self, dest_dir: Path, reserved_names: Iterable[str] = ()
    ) -> list[str]:
        descriptors = [entry[0] for entry in self._brokered_skills.values()]
        return write_skill_modules(descriptors, dest_dir, reserved_names)

    async def _call_search(self, arguments: dict[str, Any]) -> str:
        return await run_search(self._root_config.search_api_key, **arguments)

    def programmatic_tool_call_stats(
        self, invocation_id: str
    ) -> tuple[ProgrammaticToolCallStats, ProgrammaticToolCallStats]:
        direct = ProgrammaticToolCallStats().merge(
            self._tool_stats.get(invocation_id, ProgrammaticToolCallStats())
        )
        descendants = ProgrammaticToolCallStats()
        for candidate_id, stats in self._tool_stats.items():
            parent_id = self._parents.get(candidate_id)
            while parent_id is not None:
                if parent_id == invocation_id:
                    descendants = descendants.merge(stats)
                    break
                parent_id = self._parents.get(parent_id)
        return direct, descendants

    def _accept_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        if self._closed or self._close_task is not None:
            writer.close()
            return
        if len(self._connection_tasks) >= MAX_BROKER_CONNECTIONS:
            writer.close()
            return
        self._connection_writers.add(writer)
        task = asyncio.create_task(self._handle_connection(reader, writer))
        self._connection_tasks.add(task)
        task.add_done_callback(self._connection_tasks.discard)

    def endpoint_for(self, invocation_id: str) -> BrokerEndpoint:
        if self._socket_path is None:
            raise RuntimeError("session supervisor has not started")
        invocation = self._invocations[invocation_id]
        return BrokerEndpoint(self._socket_path, invocation.capability)

    async def open_scope(
        self,
        invocation_id: str,
        parent_call_id: str | None = None,
        segment_id: str | None = None,
    ) -> str:
        async with self._lock:
            if self._closed or invocation_id not in self._invocations:
                raise RuntimeError("recursive invocation is no longer active")
            scope_id = secrets.token_urlsafe(24)
            self._scopes[scope_id] = _Scope(
                invocation_id, parent_call_id, segment_id, set()
            )
            return scope_id

    async def close_scope(self, scope_id: str) -> None:
        async with self._lock:
            scope = self._scopes.pop(scope_id, None)
            tasks = list(scope.tasks) if scope else []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def _start_child(
        self, capability: str, scope_id: str, prompt: str, options: dict[str, Any]
    ) -> asyncio.Task[RLMResult]:
        if options:
            raise ValueError("recursive RLM options are not supported")
        async with self._lock:
            parent_id = self._capabilities.get(capability)
            scope = self._scopes.get(scope_id)
            if parent_id is None or scope is None or scope.invocation_id != parent_id:
                raise PermissionError("invalid recursive RLM capability")
            parent = self._invocations[parent_id]
            child_depth = parent.runtime_config.invocation.depth + 1
            if child_depth > parent.runtime_config.policy.max_depth:
                return asyncio.create_task(
                    self._limit_result(parent, "depth limit reached")
                )
            if self._total_calls >= parent.runtime_config.policy.max_subagent_calls:
                return asyncio.create_task(
                    self._limit_result(parent, "recursive call limit reached")
                )
            self._total_calls += 1
            task = asyncio.create_task(
                self._run_child(
                    parent_id, prompt, scope.parent_call_id, scope.segment_id
                )
            )
            self._tasks.add(task)
            self._child_tasks.add(task)
            scope.tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(self._child_tasks.discard)
            task.add_done_callback(scope.tasks.discard)
            return task

    async def _start_skill_call(
        self,
        capability: str,
        scope_id: str,
        skill_capability: str,
        arguments: dict[str, Any],
    ) -> asyncio.Task[str]:
        async with self._lock:
            invocation_id = self._capabilities.get(capability)
            scope = self._scopes.get(scope_id)
            if (
                invocation_id is None
                or scope is None
                or scope.invocation_id != invocation_id
            ):
                raise PermissionError("invalid broker capability")
            try:
                descriptor, handler = self._brokered_skills[skill_capability]
            except KeyError as exc:
                raise PermissionError("unknown brokered skill capability") from exc
            stats = self._tool_stats.setdefault(
                invocation_id, ProgrammaticToolCallStats()
            )
            stats.python_total += 1
            stats.by_tool_python[descriptor.name] = (
                stats.by_tool_python.get(descriptor.name, 0) + 1
            )
            task = asyncio.create_task(handler(arguments))
            self._tasks.add(task)
            scope.tasks.add(task)
            task.add_done_callback(self._tasks.discard)
            task.add_done_callback(scope.tasks.discard)
            return task

    async def _limit_result(self, parent: _Invocation, message: str) -> RLMResult:
        return RLMResult(answer=f"[{message}]", session_dir=parent.session.dir)

    async def _run_child(
        self,
        parent_id: str,
        prompt: str,
        parent_call_id: str | None,
        segment_id: str | None,
    ) -> RLMResult:
        parent = self._invocations[parent_id]
        child_context = parent.runtime_config.invocation.child()
        semaphore = self._semaphores[child_context.depth]
        async with semaphore:
            child_session = Session(Session.child_dir(parent.session.dir))
            child_id = uuid.uuid4().hex
            child_config = replace(
                parent.runtime_config,
                invocation=child_context,
            )
            child = _Invocation(
                id=child_id,
                parent_id=parent_id,
                capability=secrets.token_urlsafe(32),
                session=child_session,
                runtime_config=child_config,
                cwd=parent.cwd,
                mcp_servers=parent.mcp_servers,
            )
            parent.session.log_sub_spawn(child_session.dir.name, "(brokered rlm())")
            async with self._lock:
                if self._closed:
                    child_session.close()
                    raise asyncio.CancelledError
                self._invocations[child_id] = child
                self._parents[child_id] = parent_id
                self._capabilities[child.capability] = child_id
            try:
                factory = self._engine_factory
                if factory is None:
                    from rlm.engine import RLMEngine

                    factory = RLMEngine
                engine = factory(
                    cwd=child.cwd,
                    session=child.session,
                    mcp_servers=child.mcp_servers,
                    runtime_config=child.runtime_config,
                    supervisor=self,
                    invocation_id=child.id,
                    parent_invocation_id=parent_id,
                    parent_call_id=parent_call_id,
                    lineage_session_id=self.session_id,
                    segment_id=segment_id,
                )
                return await engine.run(prompt)
            finally:
                async with self._lock:
                    self._capabilities.pop(child.capability, None)
                    self._invocations.pop(child.id, None)
                child_session.close()

    async def _handle_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        operation_task: asyncio.Task[Any] | None = None
        disconnect_task: asyncio.Task[bytes] | None = None
        try:
            try:
                request = await asyncio.wait_for(
                    read_frame(reader), timeout=BROKER_INITIAL_FRAME_TIMEOUT_SECONDS
                )
            except asyncio.TimeoutError:
                raise TimeoutError("broker request timed out") from None
            operation = request.get("op")
            if operation not in {"rlm.run", "skill.call"}:
                raise ValueError("unknown broker operation")
            capability = request.get("capability")
            scope_id = request.get("scope_id")
            if not all(isinstance(value, str) for value in (capability, scope_id)):
                raise ValueError("invalid broker request")
            if operation == "rlm.run":
                prompt = request.get("prompt")
                options = request.get("options", {})
                if not isinstance(prompt, str) or not isinstance(options, dict):
                    raise ValueError("invalid recursive RLM request")
                operation_task = await self._start_child(
                    capability, scope_id, prompt, options
                )
            else:
                skill_capability = request.get("skill_capability")
                arguments = request.get("arguments", {})
                if not isinstance(skill_capability, str) or not isinstance(
                    arguments, dict
                ):
                    raise ValueError("invalid skill request")
                operation_task = await self._start_skill_call(
                    capability,
                    scope_id,
                    skill_capability,
                    arguments,
                )
            disconnect_task = asyncio.create_task(reader.read(1))
            done, _ = await asyncio.wait(
                {operation_task, disconnect_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            if disconnect_task in done and operation_task not in done:
                operation_task.cancel()
                await asyncio.gather(operation_task, return_exceptions=True)
                return
            result = await operation_task
            if isinstance(result, RLMResult):
                result = result_to_json(result)
            await write_frame(writer, {"result": result})
        except Exception as exc:
            if not writer.is_closing():
                try:
                    await write_frame(writer, {"error": str(exc)})
                except (ConnectionError, OSError, asyncio.IncompleteReadError):
                    pass
        finally:
            if disconnect_task is not None:
                disconnect_task.cancel()
                await asyncio.gather(disconnect_task, return_exceptions=True)
            writer.close()
            try:
                await writer.wait_closed()
            except (ConnectionError, OSError):
                pass
            self._connection_writers.discard(writer)

    async def aclose(self) -> None:
        if self._closed and self._close_task is None:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._aclose_impl())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.done():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _aclose_impl(self) -> None:
        self._closed = True
        if self._server is not None:
            self._server.close()
        for scope_id in list(self._scopes):
            await self.close_scope(scope_id)
        tasks = list(self._tasks)
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        connection_writers = list(self._connection_writers)
        for writer in connection_writers:
            writer.close()
        connection_tasks = list(self._connection_tasks)
        for task in connection_tasks:
            task.cancel()
        if connection_tasks:
            await asyncio.gather(*connection_tasks, return_exceptions=True)
        if connection_writers:
            await asyncio.gather(
                *(writer.wait_closed() for writer in connection_writers),
                return_exceptions=True,
            )
        if self._server is not None:
            await self._server.wait_closed()
            self._server = None
        self._capabilities.clear()
        self._invocations.clear()
        if self._broker_dir is not None and self._broker_dir.exists():
            shutil.rmtree(self._broker_dir)
        self._broker_dir = None
        self._socket_path = None
