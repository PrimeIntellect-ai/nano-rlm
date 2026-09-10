"""Trusted lifecycle manager for one recursive RLM session tree."""

from __future__ import annotations

import asyncio
import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.broker import (
    BrokerEndpoint,
    parse_request,
    read_frame,
    result_to_payload,
    write_frame,
)
from rlm.config import RuntimeConfig
from rlm.semantic import SemanticEdgeTracker
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
    spawned_by_request_id: str | None = None
    name: str | None = None
    task: str = ""
    persistent: bool = False
    status: str = "starting"
    created_at: float = field(default_factory=time.time)
    started_at: float = field(default_factory=time.monotonic)
    finished_at: float | None = None
    error: str | None = None
    result: RLMResult | None = None
    engine: RLMEngine | None = None
    runner: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)


@dataclass
class _Scope:
    invocation_id: str
    tasks: set[asyncio.Task[Any]]
    request_id: str | None = None


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
        semantic_edges: SemanticEdgeTracker | None = None,
    ) -> None:
        self._engine_factory = engine_factory
        self._server: asyncio.AbstractServer | None = None
        self._broker_dir: Path | None = None
        self._socket_path: str | None = None
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None
        self._lock = asyncio.Lock()
        self._total_calls = 0
        # live tree totals, pushed by every engine per model call: work-loop turns,
        # and NEW tokens (completion + uncached prompt)
        self._total_turns = 0
        self._total_tokens = 0
        self._tasks: set[asyncio.Task[Any]] = set()
        self._child_tasks: set[asyncio.Task[None]] = set()
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
        root = _Invocation(
            id=root_id,
            parent_id=None,
            capability=secrets.token_urlsafe(32),
            session=root_session,
            runtime_config=runtime_config,
            cwd=cwd,
            mcp_servers=dict(mcp_servers or {}),
            status="running",
            persistent=True,
        )
        self.root_id = root_id
        self.semantic_edges = semantic_edges or SemanticEdgeTracker()
        self.semantic_edges.register_session(
            root_id,
            parent_session_id=None,
        )
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
    def total_turns(self) -> int:
        """Live tree-total work-loop turns (every engine's model calls)."""
        return self._total_turns

    @property
    def total_tokens(self) -> int:
        """Live tree-total NEW tokens (completion + uncached prompt, every engine)."""
        return self._total_tokens

    def record_call(self, tokens: int) -> None:
        """Count one work-loop model call: a tree turn plus its new tokens."""
        self._total_turns += 1
        self._total_tokens += tokens

    def record_usage(self, tokens: int) -> None:
        """Add new tokens without a turn (compaction/checkpoint calls)."""
        self._total_tokens += tokens

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
        self, invocation_id: str, request_id: str | None = None
    ) -> str:
        async with self._lock:
            if self._closed or invocation_id not in self._capabilities.values():
                raise RuntimeError("recursive invocation is no longer active")
            scope_id = secrets.token_urlsafe(24)
            self._scopes[scope_id] = _Scope(invocation_id, set(), request_id)
            return scope_id

    async def close_scope(self, scope_id: str) -> None:
        async with self._lock:
            scope = self._scopes.pop(scope_id, None)
            tasks = list(scope.tasks) if scope else []
        for task in tasks:
            task.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    def _caller(self, capability: str, scope_id: str) -> _Invocation:
        parent_id = self._capabilities.get(capability)
        scope = self._scopes.get(scope_id)
        if (
            self._closed
            or parent_id is None
            or scope is None
            or scope.invocation_id != parent_id
        ):
            raise PermissionError("invalid agent capability or inactive cell")
        return self._invocations[parent_id]

    def _child(self, parent: _Invocation, name_or_id: str) -> _Invocation:
        children = [
            agent
            for agent in self._invocations.values()
            if agent.parent_id == parent.id
        ]
        for agent in children:
            if agent.id == name_or_id:
                return agent
        for agent in children:
            if agent.name == name_or_id:
                return agent
        raise PermissionError("agent is not a direct child of the caller")

    def _info(self, agent: _Invocation) -> dict[str, Any]:
        return {
            "id": agent.id,
            "parent_id": agent.parent_id,
            "name": agent.name,
            "task": agent.task,
            "status": agent.status,
            "persistent": agent.persistent,
            "created_at": agent.created_at,
            "elapsed_seconds": (agent.finished_at or time.monotonic())
            - agent.started_at,
            "session_dir": str(agent.session.dir),
            "error": agent.error,
        }

    def _spawn(
        self,
        parent: _Invocation,
        scope_id: str,
        task: str,
        name: str | None,
        persistent: bool,
    ) -> _Invocation:
        policy = parent.runtime_config.policy
        context = parent.runtime_config.invocation.child()
        if name is not None and any(
            agent.parent_id == parent.id and (agent.name == name or agent.id == name)
            for agent in self._invocations.values()
        ):
            raise ValueError("agent name is already reserved among these siblings")
        if context.depth > policy.max_depth:
            raise RuntimeError("depth limit reached")
        if self._total_calls >= policy.max_subagent_calls:
            raise RuntimeError("subagent call limit reached")
        if (
            policy.max_total_turns is not None
            and self._total_turns >= policy.max_total_turns
        ):
            raise RuntimeError("turn budget reached")
        if (
            policy.max_total_tokens is not None
            and self._total_tokens >= policy.max_total_tokens
        ):
            raise RuntimeError("token budget reached")
        child = _Invocation(
            id=uuid.uuid4().hex,
            parent_id=parent.id,
            capability=secrets.token_urlsafe(32),
            session=Session(Session.child_dir(parent.session.dir)),
            runtime_config=parent.runtime_config.model_copy(
                update={"invocation": context}
            ),
            cwd=parent.cwd,
            mcp_servers=parent.mcp_servers,
            spawned_by_request_id=self._scopes[scope_id].request_id,
            name=name,
            task=task,
            persistent=persistent,
        )
        try:
            child.session.write_meta(**self._info(child))
            parent.session.log_sub_spawn(
                child.session.dir.name, "rlm.agent.spawn", prompt=task
            )
        except BaseException:
            child.session.close()
            raise
        self._total_calls += 1
        self._invocations[child.id] = child
        self._parents[child.id] = parent.id
        self._capabilities[child.capability] = child.id
        self.semantic_edges.register_session(
            child.id,
            parent_session_id=parent.id,
            spawned_by_request_id=child.spawned_by_request_id,
        )
        child.runner = asyncio.create_task(self._run_child(child))
        self._tasks.add(child.runner)
        self._child_tasks.add(child.runner)
        child.runner.add_done_callback(self._tasks.discard)
        child.runner.add_done_callback(self._child_tasks.discard)
        return child

    async def _agent_operation(self, request: dict) -> Any:
        parent = self._caller(request["capability"], request["scope_id"])
        op = request["op"]
        if op == "agent.spawn":
            return self._info(
                self._spawn(
                    parent,
                    request["scope_id"],
                    request["task"],
                    request["name"],
                    request["persistent"],
                )
            )
        if op == "agent.list":
            agents = []
            for agent in self._invocations.values():
                ancestor = agent.parent_id
                while ancestor is not None:
                    if ancestor == parent.id:
                        agents.append(self._info(agent))
                        break
                    if not request["recursive"]:
                        break
                    ancestor = self._parents[ancestor]
            return agents
        child = self._child(parent, request.get("name_or_id", request.get("agent_id")))
        if op == "agent.wait":
            if not child.done.is_set() and request["timeout"] > 0:
                try:
                    await asyncio.wait_for(
                        child.done.wait(), timeout=request["timeout"]
                    )
                except asyncio.TimeoutError:
                    pass
        elif op == "agent.cancel":
            await self._terminate(child)
        elif op == "agent.result":
            if not child.done.is_set():
                return None
            # Publish return edges only when the parent actually retrieves the outcome.
            self.semantic_edges.finish_subagent(child.id)
            if child.status in {"failed", "cancelled"}:
                raise RuntimeError(child.error or "agent cancelled")
            return result_to_payload(child.result)
        return self._info(child)

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

    async def _run_child(self, child: _Invocation) -> None:
        try:
            async with self._semaphores[child.runtime_config.invocation.depth]:
                child.status = "running"
                factory = self._engine_factory
                if factory is None:
                    from rlm.engine import RLMEngine

                    factory = RLMEngine
                child.engine = factory(
                    cwd=child.cwd,
                    session=child.session,
                    mcp_servers=child.mcp_servers,
                    runtime_config=child.runtime_config,
                    supervisor=self,
                    invocation_id=child.id,
                )
                child.result = await child.engine.prompt(child.task)
                child.status = "idle" if child.persistent else "completed"
        except asyncio.CancelledError:
            child.status = "cancelled"
        except Exception as exc:
            child.status = "failed"
            child.error = str(exc)
        finally:
            try:
                if child.status != "idle":
                    await self._release_agent(child)
                else:
                    child.session.write_meta(**self._info(child))
            except Exception as exc:
                child.status = "failed"
                child.error = f"agent cleanup failed: {exc}"
            finally:
                child.done.set()

    async def _release_agent(self, child: _Invocation) -> None:
        self._capabilities.pop(child.capability, None)
        for scope_id, scope in list(self._scopes.items()):
            if scope.invocation_id == child.id:
                await self.close_scope(scope_id)
        try:
            for descendant in list(self._invocations.values()):
                if descendant.parent_id == child.id:
                    await self._terminate(descendant)
        finally:
            try:
                if child.engine is not None:
                    await child.engine.aclose()
                    child.engine = None
            finally:
                child.finished_at = time.monotonic()
                child.session.close()
                child.session.write_meta(**self._info(child))

    async def _terminate(self, child: _Invocation) -> None:
        if child.done.is_set() and child.status != "idle":
            return
        if child.stop_task is None:
            child.stop_task = asyncio.create_task(self._stop_agent(child))
            self._tasks.add(child.stop_task)
            child.stop_task.add_done_callback(self._tasks.discard)
        await asyncio.shield(child.stop_task)

    async def _stop_agent(self, child: _Invocation) -> None:
        if child.runner is not None and not child.runner.done():
            if child.status in {"starting", "running"}:
                child.runner.cancel()
            await asyncio.gather(child.runner, return_exceptions=True)
        if not child.done.is_set() or child.status == "idle":
            child.status = "cancelled"
            try:
                await self._release_agent(child)
            finally:
                child.done.set()

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
            request = parse_request(request)
            if request["op"] == "skill.call":
                operation_task = await self._start_skill_call(
                    request["capability"],
                    request["scope_id"],
                    request["skill_capability"],
                    request["arguments"],
                )
            else:
                self._caller(request["capability"], request["scope_id"])
                operation_task = asyncio.create_task(self._agent_operation(request))
                scope = self._scopes[request["scope_id"]]
                scope.tasks.add(operation_task)
                operation_task.add_done_callback(scope.tasks.discard)
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
                result = result_to_payload(result)
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
        for child in list(self._invocations.values()):
            if child.parent_id == self.root_id:
                await self._terminate(child)
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
