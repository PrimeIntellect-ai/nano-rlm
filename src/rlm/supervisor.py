"""Trusted lifecycle manager for one recursive RLM session tree."""

from __future__ import annotations

import asyncio
import json
import os
import secrets
import shutil
import tempfile
import time
import uuid
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import asdict, dataclass, field
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
from rlm.shell_jobs import ShellJob, ShellJobs
from rlm.subscriptions import Subscription, Subscriptions
from rlm.tools.ipython import build_kernel_env
from rlm.tools.git_block import find_blocked_command, refusal
from rlm.session import Session
from rlm.skills.search import run_with_api_key as run_search
from rlm.types import ProgrammaticToolCallStats, RLMResult

if TYPE_CHECKING:
    from rlm.engine import RLMEngine


MAX_BROKER_CONNECTIONS = 128
BROKER_INITIAL_FRAME_TIMEOUT_SECONDS = 5
MAX_INBOX_EVENTS = 10_000
MAX_MESSAGE_BYTES = 65_536


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
    cleanup_error: str | None = None
    released: bool = False
    result: RLMResult | None = None
    result_request_id: str | None = None
    engine: RLMEngine | None = None
    runner: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    inbox: list[dict] = field(default_factory=list)
    instructions: list[dict] = field(default_factory=list)
    inbox_error: str | None = None
    announced: int = 0
    changed: asyncio.Event = field(default_factory=asyncio.Event)


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
        self._subscriptions = Subscriptions(
            self._publish_subscription, self._record_subscription
        )
        self._shell_jobs = ShellJobs(self._publish_job, self._publish_job_output)
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
            "cleanup_error": agent.cleanup_error,
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

    def _record_event(self, agent: _Invocation, record: dict) -> None:
        try:
            with (agent.session.dir / "inbox.jsonl").open(
                "a", encoding="utf-8"
            ) as stream:
                stream.write(json.dumps(record) + "\n")
        except OSError as exc:
            agent.inbox_error = f"Inbox persistence failed: {exc}. Events remain available in memory only."

    def _event(
        self, sender: _Invocation, kind: str, content: Any, request_id: str | None
    ) -> dict:
        if len(json.dumps(content).encode("utf-8")) > MAX_MESSAGE_BYTES:
            raise ValueError("message exceeds 65536 bytes")
        return {
            "id": uuid.uuid4().hex,
            "type": kind,
            "sender_id": sender.id,
            "created_at": time.time(),
            "content": content,
            "source_request_id": request_id,
            "read": False,
        }

    def _budget_exhausted(self, agent: _Invocation) -> bool:
        policy = agent.runtime_config.policy
        return (
            policy.max_total_turns is not None
            and self._total_turns >= policy.max_total_turns
        ) or (
            policy.max_total_tokens is not None
            and self._total_tokens >= policy.max_total_tokens
        )

    def _wake_agent(self, agent: _Invocation) -> None:
        agent.changed.set()
        if agent.parent_id is None or agent.status != "idle":
            return
        if self._budget_exhausted(agent):
            self._fail_instructions(agent, "tree_budget_exhausted")
            return
        agent.done.clear()
        agent.status = "starting"
        agent.runner = asyncio.create_task(self._run_child(agent))
        self._tasks.add(agent.runner)
        self._child_tasks.add(agent.runner)
        agent.runner.add_done_callback(self._tasks.discard)
        agent.runner.add_done_callback(self._child_tasks.discard)

    def _publish(self, target: _Invocation, event: dict) -> str:
        if event["type"] == "agent.message" and len(target.inbox) >= MAX_INBOX_EVENTS:
            raise RuntimeError("inbox event limit reached")
        self._record_event(target, event)
        target.inbox.append(event)
        self._wake_agent(target)
        return event["id"]

    def _fail_instructions(self, agent: _Invocation, reason: str) -> None:
        pending, agent.instructions = agent.instructions, []
        parent = self._invocations.get(agent.parent_id)
        for instruction in pending:
            self._record_event(
                agent,
                {
                    "type": "instruction_failed",
                    "event_id": instruction["id"],
                    "reason": reason,
                },
            )
            if parent is not None and parent.capability in self._capabilities:
                self._publish(
                    parent,
                    self._event(
                        agent,
                        "agent.delivery_failed",
                        {
                            "agent_id": agent.id,
                            "message_id": instruction["id"],
                            "reason": reason,
                        },
                        None,
                    ),
                )

    def take_instructions(
        self, invocation_id: str, *, include_queue: bool = False
    ) -> list[dict]:
        agent = self._invocations[invocation_id]
        if self._budget_exhausted(agent):
            return []
        selected = [
            event
            for event in agent.instructions
            if include_queue or event["type"] == "steer"
        ]
        if selected:
            self._record_event(
                agent,
                {
                    "type": "instructions_delivered",
                    "event_ids": [e["id"] for e in selected],
                },
            )
            agent.instructions = [
                event for event in agent.instructions if event not in selected
            ]
            for event in selected:
                self.semantic_edges.deliver_message(
                    agent.id,
                    event["source_request_id"],
                    edge_type="agent_message",
                )
        return selected

    def inbox_notification(self, invocation_id: str) -> str | None:
        agent = self._invocations[invocation_id]
        agent.announced = len(agent.inbox)
        count = sum(not event["read"] for event in agent.inbox)
        notices = []
        if agent.inbox_error:
            notices.append(agent.inbox_error)
        if count:
            notices.append(
                f"Inbox: {count} unread events. Use rlm.inbox.list() and rlm.inbox.read(event_id) to inspect them."
            )
        return "Supervisor: " + " ".join(notices) if notices else None

    async def wait_for_events(self, invocation_id: str, timeout: float) -> str:
        agent = self._invocations[invocation_id]

        def ready():
            return len(agent.inbox) > agent.announced or bool(agent.instructions)

        agent.changed.clear()
        agent.status = "waiting"
        try:
            if not ready() and timeout > 0:
                await asyncio.wait_for(agent.changed.wait(), timeout=timeout)
            return (
                "New supervisor events or instructions are available."
                if ready()
                else "Wait timed out."
            )
        except asyncio.TimeoutError:
            return "Wait timed out."
        finally:
            agent.status = "running"

    def _record_subscription(self, sub: Subscription) -> None:
        self._record_event(
            self._invocations[sub.info.owner_id],
            {"type": "subscription", "subscription": asdict(sub.info)},
        )

    def _publish_subscription(
        self, sub: Subscription, kind: str, content: dict
    ) -> None:
        owner = self._invocations[sub.info.owner_id]
        if self._closed or owner.capability not in self._capabilities:
            return
        if kind != "watch.failed" and len(owner.inbox) >= MAX_INBOX_EVENTS:
            raise RuntimeError("inbox event limit reached; subscription stopped")
        event = self._event(owner, kind, {"target": sub.info.target, **content}, None)
        event["subscription_id"] = sub.info.id
        self._publish(owner, event)

    def agent_step(self, agent_id: str, start: int) -> None:
        self._subscriptions.activity(
            "agent", agent_id, start, self._invocations[agent_id].session.message_count
        )

    def _publish_job_output(self, job: ShellJob, start: int) -> None:
        self._subscriptions.activity("job", job.info.id, start, job.info.output_bytes)

    async def _watch_operation(self, parent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "watch.agent":
            child = self._child(parent, request["agent_id"])
            sub = self._subscriptions.register(
                parent.id,
                "agent",
                child.id,
                cursor=child.session.message_count,
                completed=child.status in {"completed", "failed", "cancelled"},
            )
        elif op == "watch.job":
            job = self._shell_jobs.get(parent.id, request["job_id"])
            sub = self._subscriptions.register(
                parent.id,
                "job",
                job.info.id,
                cursor=job.info.output_bytes,
                completed=job.finished is not None,
            )
        elif op == "watch.path":
            target = (Path(parent.cwd) / request["path"]).resolve()
            sub = await self._subscriptions.path(
                parent.id, target, request["recursive"]
            )
        elif op == "watch.list":
            return [
                asdict(sub.info)
                for sub in self._subscriptions.items.values()
                if sub.info.owner_id == parent.id
            ]
        else:
            sub = self._subscriptions.get(parent.id, request["subscription_id"])
            if op == "watch.cancel":
                return await self._subscriptions.cancel(sub)
        return asdict(sub.info)

    def _publish_job(self, job: ShellJob) -> None:
        self._subscriptions.finish("job", job.info.id)
        owner = self._invocations[job.info.owner_id]
        if self._closed or owner.capability not in self._capabilities:
            return
        self._publish(
            owner,
            self._event(
                owner,
                "shell.completed",
                {
                    "job_id": job.info.id,
                    "status": job.info.status,
                    "exit_code": job.info.exit_code,
                    "output_complete": job.info.output_complete,
                    "output_truncated": job.info.output_truncated,
                    "error": job.info.error,
                },
                job.source_request_id,
            ),
        )

    async def _shell_operation(self, parent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "shell.run":
            command = request["command"]
            if not command.strip():
                raise ValueError("empty command")
            blocked = find_blocked_command(
                command, allow_git=parent.runtime_config.policy.allow_git
            )
            if blocked:
                raise PermissionError(refusal(blocked))
            cwd = Path(parent.cwd) / (request["cwd"] or ".")
            return self._shell_jobs.start(
                owner_id=parent.id,
                command=command,
                cwd=str(cwd.resolve()),
                directory=parent.session.dir,
                env=build_kernel_env(dict(parent.runtime_config.kernel_env)),
                source_request_id=self._scopes[request["scope_id"]].request_id,
            )
        if op == "shell.list":
            return [
                job.snapshot()
                for job in self._shell_jobs.jobs.values()
                if job.info.owner_id == parent.id
            ]
        job = self._shell_jobs.get(parent.id, request["job_id"])
        if op == "shell.read":
            return self._shell_jobs.read(job, request["cursor"], request["max_bytes"])
        if op == "shell.cancel":
            return await self._shell_jobs.cancel(job)
        return job.snapshot()

    async def _agent_operation(self, request: dict) -> Any:
        parent = self._caller(request["capability"], request["scope_id"])
        op = request["op"]
        if op.startswith("watch."):
            return await self._watch_operation(parent, request)
        if op.startswith("shell."):
            return await self._shell_operation(parent, request)
        if op == "inbox.list":
            return [
                {
                    key: event[key]
                    for key in ("id", "type", "sender_id", "created_at", "read")
                }
                for event in parent.inbox
                if not request["unread_only"] or not event["read"]
            ]
        if op == "inbox.read":
            event = next(
                (e for e in parent.inbox if e["id"] == request["event_id"]), None
            )
            if event is None:
                raise ValueError("unknown inbox event")
            if not event["read"]:
                self._record_event(parent, {"type": "read", "event_id": event["id"]})
                event["read"] = True
                self.semantic_edges.deliver_message(
                    parent.id,
                    event["source_request_id"]
                    if event["type"].startswith("agent.")
                    else None,
                    edge_type="agent_message",
                )
            return dict(event)
        if op == "agent.report":
            if parent.parent_id is None:
                raise PermissionError("root agent has no parent")
            target = self._invocations[parent.parent_id]
            if target.capability not in self._capabilities:
                raise RuntimeError("parent is no longer active")
            return self._publish(
                target,
                self._event(
                    parent,
                    "agent.message",
                    request["message"],
                    self._scopes[request["scope_id"]].request_id,
                ),
            )
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
        if op in {"agent.send", "agent.steer"}:
            if child.status in {"completed", "failed", "cancelled"}:
                raise RuntimeError("agent is no longer active")
            if self._budget_exhausted(child):
                raise RuntimeError("cannot send instruction: tree budget exhausted")
            if len(child.instructions) >= MAX_INBOX_EVENTS:
                raise RuntimeError("instruction queue limit reached")
            event = self._event(
                parent,
                "steer" if op == "agent.steer" else "queue",
                request["message"],
                self._scopes[request["scope_id"]].request_id,
            )
            self._record_event(child, event)
            child.instructions.append(event)
            self._wake_agent(child)
            return event["id"]
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
            if child.status in {"failed", "cancelled"}:
                self.semantic_edges.finish_subagent(child.id)
                raise RuntimeError(child.error or "agent cancelled")
            if child.result is None:
                return None
            self.semantic_edges.finish_subagent(
                child.id, request_id=child.result_request_id
            )
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
                if child.engine is None:
                    child.engine = factory(
                        cwd=child.cwd,
                        session=child.session,
                        mcp_servers=child.mcp_servers,
                        runtime_config=child.runtime_config,
                        supervisor=self,
                        invocation_id=child.id,
                    )
                    child.result = await child.engine.prompt(child.task)
                else:
                    instructions = self.take_instructions(child.id, include_queue=True)
                    prompt = (
                        "\n\n".join(event["content"] for event in instructions)
                        if instructions
                        else "Supervisor: new inbox events are available."
                    )
                    child.result = await child.engine.prompt(
                        prompt,
                        message_type="parent_message"
                        if instructions
                        else "supervisor_notification",
                        event_ids=[e["id"] for e in instructions],
                    )
                child.result_request_id = self.semantic_edges.last_request_id(child.id)
                child.status = "idle" if child.persistent else "completed"
                if child.status == "idle":
                    child.session.write_meta(**self._info(child))
        except asyncio.CancelledError:
            child.status = "cancelled"
        except BaseException as exc:
            child.status = "failed"
            child.error = str(exc) or type(exc).__name__
            if not isinstance(exc, Exception):
                raise
        finally:
            try:
                if self._budget_exhausted(child):
                    self._fail_instructions(child, "tree_budget_exhausted")
                if child.status != "idle":
                    await self._release_agent(child)
            except Exception:
                pass  # Cleanup errors remain queryable and can be retried by cancel().
            finally:
                child.done.set()
                parent = self._invocations.get(child.parent_id)
                if parent is not None and parent.capability in self._capabilities:
                    self._publish(
                        parent,
                        self._event(
                            child,
                            "agent.completed",
                            {"agent_id": child.id, "status": child.status},
                            self.semantic_edges.last_request_id(child.id),
                        ),
                    )
                if child.status == "idle" and (
                    child.instructions or len(child.inbox) > child.announced
                ):
                    self._wake_agent(child)

    async def _close_agent_subscriptions(self, agent_id: str) -> None:
        try:
            self._subscriptions.finish("agent", agent_id)
        finally:
            await self._subscriptions.close(agent_id)

    async def _release_agent(self, child: _Invocation) -> None:
        self._fail_instructions(child, f"agent_{child.status}")
        self._capabilities.pop(child.capability, None)
        results = await asyncio.gather(
            self._close_agent_subscriptions(child.id),
            self._shell_jobs.close(child.id),
            *(
                self.close_scope(scope_id)
                for scope_id, scope in list(self._scopes.items())
                if scope.invocation_id == child.id
            ),
            *(
                self._terminate(descendant)
                for descendant in list(self._invocations.values())
                if descendant.parent_id == child.id
            ),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        try:
            if child.engine is not None:
                await child.engine.aclose()
                child.engine = None
        except BaseException as exc:
            errors.append(exc)
        child.finished_at = child.finished_at or time.monotonic()
        child.cleanup_error = (
            "; ".join(str(e) or type(e).__name__ for e in errors) or None
        )
        try:
            if child.engine is None:
                child.session.close()
            child.session.write_meta(**self._info(child))
        except BaseException as exc:
            errors.append(exc)
            child.cleanup_error = "; ".join(str(e) or type(e).__name__ for e in errors)
        child.released = not errors
        if errors:
            raise errors[0]

    async def _terminate(self, child: _Invocation) -> None:
        if child.released:
            return
        if child.stop_task is None or child.stop_task.done():
            child.stop_task = asyncio.create_task(self._stop_agent(child))
            self._tasks.add(child.stop_task)
            child.stop_task.add_done_callback(self._tasks.discard)
        await asyncio.shield(child.stop_task)

    async def _stop_agent(self, child: _Invocation) -> None:
        if child.runner is not None and not child.runner.done():
            if child.status in {"starting", "running", "waiting"}:
                child.runner.cancel()
            await asyncio.gather(child.runner, return_exceptions=True)
        if not child.released:
            if child.status in {"starting", "running", "waiting", "idle"}:
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
        results = await asyncio.gather(
            *(self.close_scope(scope_id) for scope_id in list(self._scopes)),
            return_exceptions=True,
        )
        results.extend(
            await asyncio.gather(
                *(
                    self._terminate(child)
                    for child in list(self._invocations.values())
                    if child.parent_id == self.root_id
                ),
                self._subscriptions.close(),
                self._shell_jobs.close(),
                return_exceptions=True,
            )
        )
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
        for result in results:
            if isinstance(result, BaseException):
                raise result
