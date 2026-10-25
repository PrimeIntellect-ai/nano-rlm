"""Trusted lifecycle manager for one recursive RLM session tree."""

from __future__ import annotations

import asyncio
import heapq
import itertools
import json
import logging
import math
import os
import re
import secrets
import shlex
import shutil
import tempfile
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Iterable
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass, field
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.broker import (
    BrokerEndpoint,
    parse_request,
    read_frame,
    agent_result_to_payload,
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
from rlm.shell_jobs import (
    RUN_BLOCK_MAX_SECONDS,
    RUN_DETACH_SECONDS,
    RUN_TEXT_BYTES,
    JobRecord,
    ShellJobs,
)
from rlm.service_manager import ServiceManager
from rlm.provenance import agent_input, runtime_event
from rlm.shared_board import SharedBoard
from rlm.subscriptions import Subscription, Subscriptions
from rlm.tools.ipython import build_kernel_env
from rlm.tools.git_block import find_blocked_command, refusal
from rlm.history import History
from rlm.session import Session
from rlm.skills.search import run_with_api_key as run_search
from rlm.types import AgentResult, ProgrammaticToolCallStats, RLMResult, TokenUsage

# Per-agent limits for runtime hints.
MAX_LONG_RUN_NOTES = 3
MAX_WAIT_HELD_JOB_HINTS = 2
# shell.completed events carry this much of the end of the output.
COMPLETED_TAIL_BYTES = 4 * 1024
# Hint once per variable after this many command prefixes.
ENV_PREFIX_HINT_AFTER = 3
_ENV_ASSIGNMENT_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$", re.S)


def _leading_env_assignments(command: str) -> list[tuple[str, str]]:
    """Read leading assignments after an optional `cd dir &&` and `env`."""
    first = re.split(r"\s*(?:&&|\|\||;|\|)\s*", command.strip(), maxsplit=1)
    segment = first[0]
    if segment.startswith("cd ") and len(first) > 1:
        segment = re.split(r"\s*(?:&&|\|\||;|\|)\s*", first[1].strip(), maxsplit=1)[0]
    try:
        words = shlex.split(segment)
    except ValueError:
        words = segment.split()
    if words and words[0] == "env":
        words = words[1:]
    out: list[tuple[str, str]] = []
    for word in words:
        match = _ENV_ASSIGNMENT_RE.match(word)
        if not match:
            break
        out.append((match.group(1), match.group(2)))
    return out


logger = logging.getLogger(__name__)

_AUTO_NAME_RE = re.compile(r"agent-\d+")


def _describe_membership(event: dict) -> str:
    name = event["agent"]
    kind = event["type"]
    if kind == "joined":
        if event.get("parent"):
            return f"{name} joined (spawned by {event['parent']})"
        return f"{name} joined as a founder"
    if kind == "retired":
        return f"{name} retired to free a slot (a DM revives it)"
    if kind == "revived":
        return f"{name} was revived"
    if kind == "answer":
        return f"{name} submitted episode answer revision {event['revision']}"
    return f"{name} ended ({event['status']})"


def _message_text(message: dict) -> str:
    """A history message as text: its content plus any tool-call arguments."""
    content = message.get("content")
    text = (
        content if isinstance(content, str) else json.dumps(content) if content else ""
    )
    for call in message.get("tool_calls") or []:
        function = call.get("function") or {}
        text += f"\n[{function.get('name')}] {function.get('arguments', '')}"
    return text.strip()


if TYPE_CHECKING:
    from rlm.engine import RLMEngine


MAX_BROKER_CONNECTIONS = 128
BROKER_INITIAL_FRAME_TIMEOUT_SECONDS = 5
MAX_INBOX_EVENTS = 10_000
MAX_MESSAGE_BYTES = 65_536
# Population agents in these states hold a live slot; "retiring" also holds one until
# its cleanup finishes. Retired agents keep their conversation and can be revived.
RESIDENT = frozenset({"starting", "running", "waiting", "idle"})
ENDED = frozenset({"completed", "failed", "cancelled"})
# Agents in these states can still act, so the population is not quiescent.
ACTIVE = frozenset({"starting", "running", "waiting", "retiring"})
# Inputs pushed into a population agent's conversation at its next model boundary.
PEER_INPUTS = frozenset({"dm", "broadcast", "reply"})
MAX_NOTICE_EVENTS = 8
HISTORY_TEXT_BYTES = 4096


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
    shell_env: dict[str, str] = field(default_factory=dict)  # rlm.shell.setenv overlay
    notes: list[tuple[str, str]] = field(
        default_factory=list
    )  # (tag, text) for next turn
    muted_hints: set[str] = field(default_factory=set)  # rlm.hints.mute()
    long_run_notes: int = 0
    env_prefixes: dict[str, int] = field(default_factory=dict)  # VAR -> prefix count
    env_prefix_hinted: set[str] = field(default_factory=set)
    result_request_id: str | None = None
    engine: RLMEngine | None = None
    runner: asyncio.Task[None] | None = None
    stop_task: asyncio.Task[None] | None = None
    done: asyncio.Event = field(default_factory=asyncio.Event)
    inbox: list[dict] = field(default_factory=list)
    instructions: list[dict] = field(default_factory=list)
    inbox_error: str | None = None
    announced: int = 0  # events seen by the last notice/wait (all types; wakes wait)
    announced_loud: int = 0  # events that count toward the unread notice
    announced_quiet: int = 0
    wait_hints: int = 0  # wait-held-job hints spent
    turns: int = 0
    new_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    unread_announced: int = 0  # unread count in the last inbox notice
    changed: asyncio.Event = field(default_factory=asyncio.Event)
    # Population mode
    idle_since: float | None = None  # monotonic start of the current idle period
    kernel_generation: int = 0
    requesters: list[str] = field(default_factory=list)  # DM senders awaiting a reply
    failure: BaseException | None = None
    membership_seen: int = 0  # population events already announced to this agent


@dataclass
class _Scope:
    invocation_id: str
    tasks: set[asyncio.Task[Any]]
    request_id: str | None = None


class _InferencePool:
    """Inference permits ordered by depth then arrival; requests are not preempted."""

    def __init__(self, capacity: int) -> None:
        self.available = capacity
        self._tickets = itertools.count()
        self._waiters: list[tuple[int, int, asyncio.Future[None]]] = []

    async def acquire(self, depth: int) -> None:
        waiter = asyncio.get_running_loop().create_future()
        heapq.heappush(self._waiters, (depth, next(self._tickets), waiter))
        self._drain()
        try:
            await waiter
        except asyncio.CancelledError:
            if not waiter.cancelled():
                self.release()  # Admission was granted before cancellation arrived.
            else:
                self._drain()
            raise

    def release(self) -> None:
        self.available += 1
        self._drain()

    def _drain(self) -> None:
        while self._waiters and (self.available or self._waiters[0][2].cancelled()):
            _, _, waiter = heapq.heappop(self._waiters)
            if not waiter.cancelled():
                self.available -= 1
                waiter.set_result(None)


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
        self._mcp_capabilities: set[str] = set()
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
        # Services outlive this supervisor, so their records live outside the session directory.
        self._services = ServiceManager(
            Path(os.environ.get("RLM_SERVICES_DIR", "/tmp/rlm-services"))
        )
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
        self._inference = _InferencePool(runtime_config.policy.max_concurrent_subagents)
        # Population mode: the root record stands for the episode, not a model agent.
        self._population = runtime_config.policy.population
        self._population_open = False
        self._episode_task = ""
        self._admitted = 0
        self._reserved_births = 0
        self._peak_live = 0
        self._counters = {
            key: 0
            for key in (
                "births_denied",
                "retirements",
                "revivals",
                "dms",
                "broadcasts",
                "replies",
                "answer_submissions",
            )
        }
        self._membership: list[dict] = []
        self._message_receipts: dict[tuple[str, str], tuple[str, Any]] = {}
        self._board: SharedBoard | None = None
        self._answer: dict[str, Any] = {"text": "", "revision": 0, "author": None}
        self._activity = asyncio.Event()

    @asynccontextmanager
    async def inference_slot(self, invocation_id: str) -> AsyncIterator[None]:
        agent = self._invocations[invocation_id]
        # Peers queue FIFO: depth priority would favor founders over their offspring.
        depth = 0 if self._population else agent.runtime_config.invocation.depth
        await self._inference.acquire(depth)
        try:
            yield
        finally:
            self._inference.release()

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

    def record_call(
        self, tokens: int, agent_id: str | None = None, usage: TokenUsage | None = None
    ) -> None:
        """Count one work-loop model call: a tree turn plus its new tokens, and with the
        calling agent's id its own counters as well."""
        self._total_turns += 1
        self.record_usage(tokens, agent_id, usage)
        agent = self._invocations.get(agent_id) if agent_id else None
        if agent is not None:
            agent.turns += 1
            agent.new_tokens += tokens

    def record_usage(
        self, tokens: int, agent_id: str | None = None, usage: TokenUsage | None = None
    ) -> None:
        """Add new tokens without a turn (compaction/checkpoint calls); usage also
        counts toward the calling agent's billed prompt and completion tokens."""
        self._total_tokens += tokens
        agent = self._invocations.get(agent_id) if agent_id else None
        if agent is not None and usage is not None:
            agent.prompt_tokens += usage.prompt_tokens
            agent.completion_tokens += usage.completion_tokens

    @property
    def active_calls(self) -> int:
        return len(self._child_tasks)

    async def start(self) -> None:
        if self._server is not None:
            return
        if self._closed:
            raise RuntimeError("session supervisor is closed")
        if self._mcp_registry is not None:
            await self._mcp_registry.discover()
        self._broker_dir = Path(tempfile.mkdtemp(prefix="rlm-brk-"))
        os.chmod(self._broker_dir, 0o700)
        self._socket_path = str(self._broker_dir / "b.sock")
        self._server = await asyncio.start_unix_server(
            self._accept_connection, path=self._socket_path
        )
        os.chmod(self._socket_path, 0o600)
        if self._population:
            self._board = SharedBoard(
                self._invocations[self.root_id].session.dir / "board.md"
            )

    def write_brokered_skill_modules(
        self, dest_dir: Path, reserved_names: Iterable[str] = ()
    ) -> list[str]:
        descriptors = [
            entry[0]
            for capability, entry in self._brokered_skills.items()
            if capability not in self._mcp_capabilities
        ]
        return write_skill_modules(descriptors, dest_dir, reserved_names)

    def mcp_server_summaries(self) -> list[dict[str, Any]]:
        return self._mcp_registry.servers() if self._mcp_registry else []

    async def _call_search(self, arguments: dict[str, Any]) -> str:
        return await run_search(
            self._root_config.search_api_key,
            network_block=self._root_config.network_block,
            **arguments,
        )

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

    def agent_context(self, invocation_id: str) -> dict[str, Any]:
        """Runtime identity for this agent's prompt, without broker credentials."""
        agent = self._invocations[invocation_id]
        context = {
            "id": agent.id,
            "parent_id": agent.parent_id,
            "name": agent.name,
            "persistent": agent.persistent,
        }
        if self._population:
            parent = self._invocations.get(agent.parent_id)
            context["population"] = {
                "founder": agent.parent_id is None,
                "parent_name": parent.name if parent else None,
                "task": self._episode_task,
                **self._population.model_dump(),
            }
        return context

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
            "turns": agent.turns,
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
        if (
            policy.max_subagent_calls is not None
            and self._total_calls >= policy.max_subagent_calls
        ):
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
        self._start_runner(child)
        return child

    # --- Population mode ---------------------------------------------------------

    def _peers(self) -> list[_Invocation]:
        return [
            agent for agent in self._invocations.values() if agent.id != self.root_id
        ]

    def _used_slots(self) -> int:
        return sum(
            agent.status in RESIDENT or agent.status == "retiring"
            for agent in self._peers()
        )

    def _job_owners(self) -> set[str]:
        return {
            job.info.owner_id
            for job in self._shell_jobs.jobs.values()
            if job.finished is None
        }

    def _awaiting_replies(self) -> set[str]:
        """Agents whose direct message has not been answered yet."""
        awaiting = set()
        for agent in self._peers():
            awaiting.update(agent.requesters)
            awaiting.update(
                event["sender_id"]
                for event in agent.instructions
                if event["type"] == "dm"
            )
        return awaiting

    def _lookup(self, name_or_id: str) -> _Invocation:
        for agent in self._peers():
            if name_or_id in (agent.id, agent.name):
                return agent
        raise ValueError(f"no agent named {name_or_id!r} in this population")

    def population_counts(self) -> dict[str, Any]:
        policy = self._population
        peers = self._peers()
        used = self._used_slots()
        return {
            "live": used,
            "capacity": policy.max_live_agents,
            "available_slots": max(policy.max_live_agents - used, 0),
            "idle": sum(agent.status == "idle" for agent in peers),
            "retired": sum(agent.status == "retired" for agent in peers),
            "ended": sum(agent.status in ENDED for agent in peers),
            "admitted": self._admitted,
            "max_total_agents": policy.max_total_agents,
            "remaining_total_agents": policy.max_total_agents
            - self._admitted
            - self._reserved_births,
            "admission_open": self._population_open,
        }

    def _announce(self, kind: str, agent: _Invocation, **fields: Any) -> None:
        """Record a membership change; peers see it as a notice at their next model call."""
        event = {
            "seq": len(self._membership) + 1,
            "type": kind,
            "agent": agent.name,
            "agent_id": agent.id,
            "time": time.time(),
            **fields,
            **self.population_counts(),
        }
        self._membership.append(event)
        root_dir = self._invocations[self.root_id].session.dir
        with (root_dir / "population.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event) + "\n")
        self._activity.set()

    def _membership_notice(self, agent: _Invocation) -> str | None:
        events = [
            event
            for event in self._membership[agent.membership_seen :]
            if event["agent_id"] != agent.id
        ]
        agent.membership_seen = len(self._membership)
        if not events:
            return None
        shown = events[-MAX_NOTICE_EVENTS:]
        parts = [_describe_membership(event) for event in shown]
        if len(events) > len(shown):
            parts.insert(
                0,
                f"{len(events) - len(shown)} earlier changes (see rlm.population.events())",
            )
        counts = self.population_counts()
        return (
            "Population: "
            + "; ".join(parts)
            + f". Now {counts['live']}/{counts['capacity']} live, "
            f"{counts['admitted']}/{counts['max_total_agents']} admitted."
        )

    def _check_birth(self, name: str | None) -> None:
        policy = self._population
        if not self._population_open:
            raise RuntimeError("population_closed: the episode is ending")
        if self._budget_exhausted(self._invocations[self.root_id]):
            raise RuntimeError(
                "budget_exhausted: the episode's token or turn budget is spent"
            )
        if self._admitted + self._reserved_births >= policy.max_total_agents:
            raise RuntimeError(
                f"population_exhausted: all {policy.max_total_agents} admissions of "
                "this episode are used, so no agent can be born. Continue with the "
                "existing agents."
            )
        if name is not None:
            if _AUTO_NAME_RE.fullmatch(name):
                raise ValueError(
                    "names of the form agent-<number> are assigned automatically"
                )
            if any(name in (agent.name, agent.id) for agent in self._peers()):
                raise ValueError(f"agent name {name!r} is already taken")

    def _claim_retiree(self, caller: _Invocation) -> _Invocation:
        """Mark the longest-idle agent past its grace period as retiring, or fail at once."""
        policy = self._population
        now = time.monotonic()
        busy = self._job_owners() | self._awaiting_replies()
        idle = [
            agent
            for agent in self._peers()
            if agent.status == "idle"
            and agent is not caller
            and not agent.instructions
            and agent.id not in busy
        ]
        eligible = [
            agent
            for agent in idle
            if now - agent.idle_since >= policy.idle_grace_seconds
        ]
        if eligible:
            retiree = min(eligible, key=lambda agent: (agent.idle_since, agent.id))
            retiree.status = "retiring"
            return retiree
        self._counters["births_denied"] += 1
        counts = self.population_counts()
        if idle:
            wait = min(
                policy.idle_grace_seconds - (now - agent.idle_since) for agent in idle
            )
            reason = (
                f"every idle agent is within its {policy.idle_grace_seconds:g}s grace "
                f"period; a slot can be reclaimed in about {math.ceil(wait)}s"
            )
        else:
            reason = "no agent is idle"
        raise RuntimeError(
            f"population_full: {counts['live']}/{counts['capacity']} agents are live and "
            f"{reason}. Continue the work yourself or retry after a membership change."
        )

    async def _retire(self, agent: _Invocation) -> None:
        """Release a retiring agent's kernel, jobs and subscriptions; keep its conversation."""
        self._capabilities.pop(agent.capability, None)
        results = await asyncio.gather(
            self._subscriptions.close(agent.id),
            self._shell_jobs.close(agent.id),
            return_exceptions=True,
        )
        errors = [result for result in results if isinstance(result, BaseException)]
        if agent.engine is not None:
            try:
                await agent.engine.release_kernel()
            except Exception as exc:
                errors.append(exc)
        if errors:
            # An agent whose resources could not be released cannot be revived safely.
            agent.cleanup_error = "; ".join(str(e) or type(e).__name__ for e in errors)
            agent.status = "failed"
            agent.error = f"retirement cleanup failed: {agent.cleanup_error}"
            agent.failure = errors[0]
            self._announce("ended", agent, status=agent.status, error=agent.error)
            raise errors[0]

    def _finish_retirement(self, agent: _Invocation) -> bool:
        """Free a retiring agent's slot once its cleanup succeeded; False if it ended meanwhile."""
        if agent.status != "retiring":
            return False
        agent.status = "retired"
        agent.idle_since = None
        self._counters["retirements"] += 1
        agent.session.write_meta(**self._info(agent))
        return True

    def _settle_retirement(self, agent: _Invocation, task: asyncio.Task) -> None:
        if not task.cancelled() and task.exception() is None:
            if self._finish_retirement(agent):
                self._announce("retired", agent, reason="capacity")

    async def _admit(
        self, parent: _Invocation, task: str, name: str | None, request_id: str | None
    ) -> _Invocation:
        """Admit one spawned agent, retiring an idle peer first if the live cap is full."""
        self._check_birth(name)
        if self._used_slots() >= self._population.max_live_agents:
            retiree = self._claim_retiree(parent)
            self._reserved_births += 1
            retirement = asyncio.create_task(self._retire(retiree))
            self._tasks.add(retirement)
            retirement.add_done_callback(self._tasks.discard)
            try:
                await asyncio.shield(retirement)
            except asyncio.CancelledError:
                # The caller left; the slot frees once cleanup finishes.
                retirement.add_done_callback(partial(self._settle_retirement, retiree))
                raise
            finally:
                self._reserved_births -= 1
            # Hand the slot over without yielding, so no other spawn can take it, and
            # announce the retirement with the counts after the handoff.
            retired = self._finish_retirement(retiree)
            try:
                self._check_birth(name)
                self._total_calls += 1
                return self._create_agent(
                    task, name, parent=parent, spawned_by_request_id=request_id
                )
            finally:
                if retired:
                    self._announce("retired", retiree, reason="capacity")
        self._total_calls += 1
        return self._create_agent(
            task, name, parent=parent, spawned_by_request_id=request_id
        )

    def _create_agent(
        self,
        task: str,
        name: str | None,
        *,
        parent: _Invocation | None,
        spawned_by_request_id: str | None = None,
    ) -> _Invocation:
        root = self._invocations[self.root_id]
        source = parent or root
        agent = _Invocation(
            id=uuid.uuid4().hex,
            parent_id=parent.id if parent else None,
            capability=secrets.token_urlsafe(32),
            session=Session(Session.child_dir(root.session.dir)),
            runtime_config=source.runtime_config.model_copy(
                update={"invocation": source.runtime_config.invocation.child()}
            )
            if parent
            else root.runtime_config,
            cwd=source.cwd,
            mcp_servers=source.mcp_servers,
            spawned_by_request_id=spawned_by_request_id,
            name=name or f"agent-{self._admitted + 1}",
            task=task,
            persistent=True,
            requesters=[parent.id] if parent else [],
            membership_seen=len(self._membership),
        )
        try:
            agent.session.write_meta(**self._info(agent))
            if parent is not None:
                parent.session.log_sub_spawn(
                    agent.session.dir.name, "rlm.agent.spawn", prompt=task
                )
        except BaseException:
            agent.session.close()
            raise
        self._admitted += 1
        self._invocations[agent.id] = agent
        self._parents[agent.id] = agent.parent_id
        self._capabilities[agent.capability] = agent.id
        self.semantic_edges.register_session(
            agent.id,
            parent_session_id=agent.parent_id,
            spawned_by_request_id=spawned_by_request_id,
        )
        self._peak_live = max(self._peak_live, self._used_slots())
        self._announce("joined", agent, parent=parent.name if parent else None)
        self._start_runner(agent)
        return agent

    def admit_founders(self, task: str) -> None:
        """Open the population on the episode task and admit every founder at once."""
        if self._population_open or self._admitted:
            raise RuntimeError("the population has already started")
        self._episode_task = task
        self._population_open = True
        count = self._population.founders
        founders = [
            self._create_agent(
                f"You are agent-{index + 1}, one of {count} founders admitted at the "
                "start of this episode. Begin work on the episode task.",
                None,
                parent=None,
            )
            for index in range(count)
        ]
        # Founders know each other from their prompts; notices start after the cohort.
        for founder in founders:
            founder.membership_seen = len(self._membership)

    def close_population(self) -> None:
        """Close admission and stop waking agents; the episode is ending."""
        self._population_open = False

    async def stop_population(self) -> None:
        """Stop every agent; afterwards no model call or tool of the episode runs."""
        self.close_population()
        peers = self._peers()
        results = await asyncio.gather(
            *(self._terminate(agent) for agent in peers), return_exceptions=True
        )
        for agent, result in zip(peers, results):
            if isinstance(result, BaseException):
                logger.warning("rlm: cleanup of %s failed: %s", agent.name, result)

    def population_failures(self) -> list[BaseException]:
        return [agent.failure for agent in self._peers() if agent.failure is not None]

    @property
    def activity(self) -> asyncio.Event:
        """Set whenever an agent finishes a turn or membership changes."""
        return self._activity

    def population_quiescent(self) -> bool:
        """True when no agent can act again: none runs, starts, retires or has input queued."""
        job_owners = self._job_owners()
        return all(
            agent.status not in ACTIVE
            and (agent.runner is None or agent.runner.done())
            and not (agent.status == "idle" and agent.instructions)
            and agent.id not in job_owners
            for agent in self._peers()
        )

    def _revive(self, agent: _Invocation) -> None:
        """Give a retired agent a live slot again; its next turn starts a fresh kernel."""
        policy = self._population
        if agent.released:
            raise RuntimeError(f"{agent.name} has ended")
        if self._used_slots() >= policy.max_live_agents:
            hint = (
                " Its history is readable with rlm.population.history()."
                if policy.allow_agent_lookup_agent_history
                else ""
            )
            raise RuntimeError(
                f"retired_no_capacity: {agent.name} is retired and no slot is free "
                f"({self._used_slots()}/{policy.max_live_agents} live), so the message "
                f"was not delivered. Retry after a membership change or continue without it.{hint}"
            )
        if self._budget_exhausted(agent):
            raise RuntimeError(
                "budget_exhausted: the episode's token or turn budget is spent"
            )
        agent.capability = secrets.token_urlsafe(32)
        self._capabilities[agent.capability] = agent.id
        agent.kernel_generation += 1
        agent.status = "idle"
        agent.idle_since = time.monotonic()
        self._counters["revivals"] += 1
        self._peak_live = max(self._peak_live, self._used_slots())
        self._announce("revived", agent)

    def _message(
        self,
        sender: _Invocation,
        target: _Invocation,
        kind: str,
        text: str,
        request_id: str | None,
    ) -> dict:
        """Queue a message for target's next model boundary, reviving it if retired."""
        if not self._population_open:
            raise RuntimeError("population_closed: the episode is ending")
        if target is sender:
            raise ValueError("an agent cannot message itself")
        if target.status in ENDED:
            raise RuntimeError(f"{target.name} has ended ({target.status})")
        if target.status == "retiring":
            raise RuntimeError(
                f"agent_retiring: {target.name} is releasing its slot; retry shortly"
            )
        if len(target.instructions) >= MAX_INBOX_EVENTS:
            raise RuntimeError(f"{target.name}'s message queue is full")
        event = self._event(sender, kind, text, request_id)
        if target.status == "retired":
            self._revive(target)
        self._record_event(target, event)
        target.instructions.append(event)
        self._wake_agent(target)
        return event

    def _once(self, sender: _Invocation, message_id: str | None, payload: Any, send):
        """Run send once per sender and message_id; a retry returns the first receipt."""
        if message_id is None:
            return send()
        key = (sender.id, message_id)
        fingerprint = json.dumps(payload, sort_keys=True)
        if key in self._message_receipts:
            previous, receipt = self._message_receipts[key]
            if previous != fingerprint:
                raise ValueError("message_id was already used for a different message")
            return receipt
        receipt = send()
        self._message_receipts[key] = (fingerprint, receipt)
        return receipt

    def _dm(self, sender, scope_id, target_name, text, message_id) -> dict:
        def send() -> dict:
            target = self._lookup(target_name)
            revived = target.status == "retired"
            event = self._message(
                sender, target, "dm", text, self._scopes[scope_id].request_id
            )
            self._counters["dms"] += 1
            return {
                "message_id": event["id"],
                "to": target.name,
                "status": "accepted",
                "revived": revived,
            }

        return self._once(sender, message_id, ["dm", target_name, text], send)

    def _broadcast(self, sender, scope_id, text, message_id) -> dict:
        def send() -> dict:
            delivered, rejected = [], []
            for target in self._peers():
                if target is sender or target.status not in RESIDENT:
                    continue
                try:
                    self._message(
                        sender,
                        target,
                        "broadcast",
                        text,
                        self._scopes[scope_id].request_id,
                    )
                    delivered.append(target.name)
                except RuntimeError as exc:
                    rejected.append({"to": target.name, "reason": str(exc)})
            self._counters["broadcasts"] += 1
            return {"delivered": delivered, "rejected": rejected}

        return self._once(sender, message_id, ["broadcast", text], send)

    def _reply(self, agent: _Invocation) -> None:
        """Push a finished turn's answer to the agents whose messages it answered."""
        requesters, agent.requesters = agent.requesters, []
        answer = agent.result.answer if agent.result is not None else ""
        if not answer:
            return
        data = answer.encode()
        if len(data) > MAX_MESSAGE_BYTES // 2:
            answer = "[answer truncated to its end]\n" + data[
                -(MAX_MESSAGE_BYTES // 2) :
            ].decode(errors="ignore")
        for requester_id in requesters:
            target = self._invocations[requester_id]
            try:
                self._message(
                    agent,
                    target,
                    "reply",
                    answer,
                    self.semantic_edges.last_request_id(agent.id),
                )
                self._counters["replies"] += 1
            except (RuntimeError, ValueError) as exc:
                self._record_event(
                    agent,
                    {
                        "type": "reply_undelivered",
                        "to": target.name,
                        "reason": str(exc),
                    },
                )

    def _submit_answer(
        self, agent: _Invocation, text: str, expected_revision: int
    ) -> dict:
        if not self._population_open:
            raise RuntimeError("population_closed: the episode answer is frozen")
        current = self._answer
        if expected_revision != current["revision"]:
            raise RuntimeError(
                f"answer_conflict: you expected revision {expected_revision} but the "
                f"episode answer is at revision {current['revision']} (by "
                f"{current['author']}). Read it with rlm.population.answer() and "
                "resubmit only if yours is better."
            )
        if text == current["text"]:
            return {"revision": current["revision"], "changed": False}
        self._answer = {
            "text": text,
            "revision": current["revision"] + 1,
            "author": agent.name,
        }
        self._counters["answer_submissions"] += 1
        self._announce("answer", agent, revision=self._answer["revision"])
        return {"revision": self._answer["revision"], "changed": True}

    @property
    def episode_answer(self) -> dict[str, Any]:
        return dict(self._answer)

    @property
    def board_text(self) -> str:
        return self._board.text if self._board else ""

    def _peer_history(
        self, name_or_id: str, start: int, limit: int, query: str | None
    ) -> dict:
        if not self._population.allow_agent_lookup_agent_history:
            raise PermissionError("peer history lookup is disabled in this episode")
        target = self._lookup(name_or_id)
        records = []
        for index, message in enumerate(History(target.session.dir).messages):
            if message.get("role") == "system":
                continue
            text = _message_text(message)
            if query is None or query in text:
                records.append(
                    {"index": index, "role": message.get("role"), "text": text}
                )
        page = records[start : start + limit]
        for record in page:
            data = record["text"].encode()
            if len(data) > HISTORY_TEXT_BYTES:
                record["text"] = data[:HISTORY_TEXT_BYTES].decode(errors="ignore")
                record["truncated"] = True
        end = start + len(page)
        return {
            "agent": target.name,
            "status": target.status,
            "task": target.task,
            "last_answer": target.result.answer if target.result else None,
            "total": len(records),
            "next": end if end < len(records) else None,
            "messages": page,
        }

    def _population_list(
        self, caller: _Invocation, include_self: bool, include_ended: bool
    ) -> list[dict]:
        now = time.monotonic()
        return [
            {
                "id": agent.id,
                "name": agent.name,
                "parent": self._invocations[agent.parent_id].name
                if agent.parent_id
                else None,
                "founder": agent.parent_id is None,
                "status": agent.status,
                "messageable": agent.status in RESIDENT or agent.status == "retired",
                "requires_revival": agent.status == "retired",
                "idle_seconds": None
                if agent.idle_since is None
                else round(now - agent.idle_since, 1),
                "turns": agent.turns,
                "task": agent.task[:300],
            }
            for agent in self._peers()
            if (include_self or agent is not caller)
            and (include_ended or agent.status not in ENDED)
        ]

    def _population_operation(self, agent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "population.status":
            policy = agent.runtime_config.policy
            return {
                **self.population_counts(),
                "you": agent.name,
                "board_revision": self._board.revision,
                "answer_revision": self._answer["revision"],
                "answer_author": self._answer["author"],
                "turns": self._total_turns,
                "max_total_turns": policy.max_total_turns,
                "new_tokens": self._total_tokens,
                "max_total_tokens": policy.max_total_tokens,
            }
        if op == "population.list":
            return self._population_list(
                agent, request["include_self"], request["include_ended"]
            )
        if op == "population.events":
            start = request["after"]
            return self._membership[start : start + request["limit"]]
        if op == "population.answer":
            return self.episode_answer
        if op == "population.submit":
            return self._submit_answer(
                agent, request["text"], request["expected_revision"]
            )
        return self._peer_history(
            request["agent"], request["start"], request["limit"], request.get("query")
        )

    def _board_operation(self, agent: _Invocation, request: dict) -> dict:
        board = self._board
        op = request["op"]
        if op == "board.read":
            return board.read(
                request["start_line"], request["max_lines"], request.get("revision")
            )
        if op == "board.search":
            return board.search(request["query"], request["limit"])
        if not self._population_open:
            raise RuntimeError("population_closed: the board is frozen")
        if op == "board.append":
            return board.append(agent.name, request["text"], request.get("request_id"))
        return board.edit(
            agent.name,
            request["old"],
            request["new"],
            request["expected_revision"],
            request.get("request_id"),
        )

    def population_snapshot(self) -> dict[str, Any]:
        """Episode accounting: membership, communication, and per-agent usage."""
        return {
            **self.population_counts(),
            "peak_live": self._peak_live,
            **self._counters,
            "board_revision": self._board.revision if self._board else 0,
            "board_bytes": len(self._board.text.encode()) if self._board else 0,
            "answer_revision": self._answer["revision"],
            "answer_author": self._answer["author"],
            "agents": [
                {
                    "id": agent.id,
                    "name": agent.name,
                    "parent": self._invocations[agent.parent_id].name
                    if agent.parent_id
                    else None,
                    "founder": agent.parent_id is None,
                    "status": agent.status,
                    "turns": agent.turns,
                    "prompt_tokens": agent.prompt_tokens,
                    "completion_tokens": agent.completion_tokens,
                    "new_tokens": agent.new_tokens,
                    "kernel_generation": agent.kernel_generation,
                    "error": agent.error,
                }
                for agent in self._peers()
            ],
        }

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
        if agent.id == self.root_id or agent.status != "idle" or self._closed:
            return
        if self._population and not self._population_open:
            return
        if self._budget_exhausted(agent):
            self._fail_instructions(agent, "tree_budget_exhausted")
            return
        agent.done.clear()
        agent.status = "starting"
        agent.idle_since = None
        self._start_runner(agent)

    def _start_runner(self, agent: _Invocation) -> None:
        agent.runner = asyncio.create_task(self._run_child(agent))
        self._tasks.add(agent.runner)
        self._child_tasks.add(agent.runner)
        agent.runner.add_done_callback(self._tasks.discard)
        agent.runner.add_done_callback(self._child_tasks.discard)
        agent.runner.add_done_callback(lambda _: self._activity.set())

    def _publish(self, target: _Invocation, event: dict) -> str:
        if event["type"] == "agent.message" and len(target.inbox) >= MAX_INBOX_EVENTS:
            raise RuntimeError("inbox event limit reached")
        self._record_event(target, event)
        target.inbox.append(event)
        self._wake_agent(target)
        return event["id"]

    def _fail_instructions(self, agent: _Invocation, reason: str) -> None:
        pending, agent.instructions = agent.instructions, []
        for instruction in pending:
            self._record_event(
                agent,
                {
                    "type": "instruction_failed",
                    "event_id": instruction["id"],
                    "reason": reason,
                },
            )
            sender = self._invocations.get(instruction["sender_id"])
            if sender is not None and sender.capability in self._capabilities:
                self._publish(
                    sender,
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
        # Queued parent instructions wait for an answer boundary; the rest are pushed
        # at the next model boundary.
        selected = [
            event
            for event in agent.instructions
            if include_queue or event["type"] != "queue"
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
                    edge_type="agent_reply"
                    if event["type"] == "reply"
                    else "agent_message",
                )
                if event["type"] == "dm" and event["sender_id"] not in agent.requesters:
                    agent.requesters.append(event["sender_id"])
        return selected

    def render_input(self, event: dict) -> tuple[dict, dict]:
        """The user-role message for one delivered agent input, plus its provenance."""
        sender = self._invocations.get(event["sender_id"])
        if event["type"] in PEER_INPUTS:
            return agent_input(
                event["content"],
                agent=sender.name if sender else event["sender_id"],
                kind=event["type"],
                sender="peer",
            )
        return agent_input(
            event["content"],
            agent=event.get("sender_id"),
            kind="steer" if event["type"] == "steer" else "instruction",
        )

    def _note_env_prefixes(self, agent: _Invocation, command: str) -> None:
        """Suggest setenv() after repeated assignments to the same variable."""
        for name, value in _leading_env_assignments(command):
            count = agent.env_prefixes.get(name, 0) + 1
            agent.env_prefixes[name] = count
            if count == ENV_PREFIX_HINT_AFTER and name not in agent.env_prefix_hinted:
                agent.env_prefix_hinted.add(name)
                self._hint(
                    agent,
                    "env-prefix",
                    f"`{name}=...` has prefixed {count} commands so far; "
                    f"`await rlm.shell.setenv({name}={value!r})` applies it to every "
                    "later run(), and `env={...}` to one call.",
                )

    def _note_wait_with_held_jobs(self, agent: _Invocation) -> None:
        """Suggest result() when native wait is called with outstanding jobs."""
        if (
            agent.wait_hints >= MAX_WAIT_HELD_JOB_HINTS
            or "wait-held-job" in agent.muted_hints
        ):
            return
        held = [
            job
            for job in self._shell_jobs.jobs.values()
            if job.info.owner_id == agent.id and job.finished is None and job.notify
        ]
        if not held:
            return
        agent.wait_hints += 1
        job = held[0]
        self._hint(
            agent,
            "wait-held-job",
            f"You called wait while holding running job {job.info.id}: "
            f"`res = {self._collect_expr(job)}` waits for it directly (up to 300 s per "
            "call) and returns its output. Native wait is for events from agents or "
            "watches, not for jobs you hold.",
        )

    def hint(self, invocation_id: str, tag: str, text: str) -> None:
        """Queue a tagged hint for an agent from outside the supervisor (engine-side observations)."""
        agent = self._invocations.get(invocation_id)
        if agent is not None:
            self._hint(agent, tag, text)

    @staticmethod
    def _hint(agent: _Invocation, tag: str, text: str) -> None:
        """Queue a tagged one-line hint for the agent's next turn unless the tag is muted."""
        if tag in agent.muted_hints:
            return
        agent.notes.append(
            (tag, f'{text} (Mute this hint with await rlm.hints.mute("{tag}").)')
        )

    def inbox_notice(self, invocation_id: str) -> dict | None:
        """Pending runtime notices for the agent's next turn as a structured record:
        {"text", "hints": [tags], "unread": int | None, "error": bool}. Clears them."""
        agent = self._invocations[invocation_id]
        # Quiet completions have a separate wait cursor and no unread-count notice.
        loud = [event for event in agent.inbox if event["type"] != "shell.completed"]
        new_events = len(loud) > agent.announced_loud
        agent.announced_loud = len(loud)
        agent.announced = len(agent.inbox)
        count = sum(not event["read"] for event in loud)
        notices: list[str] = []
        hints: list[str] = []
        if agent.notes:
            for tag, text in agent.notes:
                if tag not in agent.muted_hints:
                    notices.append(text)
                    hints.append(tag)
            agent.notes.clear()
        error = bool(agent.inbox_error)
        if agent.inbox_error:
            notices.append(agent.inbox_error)
        if self._population and (membership := self._membership_notice(agent)):
            notices.append(membership)
        # Announce unread counts only on changes or new arrivals.
        unread = None
        if count and (new_events or count != agent.unread_announced):
            unread = count
            notices.append(
                f"Inbox: {count} unread events. Use rlm.inbox.list() and rlm.inbox.read(event_id) to inspect them."
            )
        agent.unread_announced = count
        if not notices:
            return None
        return {
            "text": " ".join(notices),
            "hints": hints,
            "unread": unread,
            "error": error,
        }

    def inbox_notification(self, invocation_id: str) -> str | None:
        """The notices as one line (see inbox_notice for the structured form)."""
        notice = self.inbox_notice(invocation_id)
        return "Supervisor: " + notice["text"] if notice else None

    async def wait_for_events(self, invocation_id: str, timeout: float) -> str:
        agent = self._invocations[invocation_id]
        self._note_wait_with_held_jobs(agent)

        def ready():
            return (
                len(agent.inbox) > agent.announced
                or any(
                    not event["read"] and event["type"] == "shell.completed"
                    for event in agent.inbox[agent.announced_quiet :]
                )
                or bool(agent.instructions)
            )

        agent.changed.clear()
        agent.status = "waiting"
        try:
            if not ready() and timeout > 0:
                await asyncio.wait_for(agent.changed.wait(), timeout=timeout)
            available = ready()
            agent.announced_quiet = len(agent.inbox)
            return (
                "New supervisor events or instructions are available."
                if available
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
        """A child's assistant/tool step is fully logged: publish watch.agent activity and
        check the progress thresholds a parent subscribed to
        (watch.agent(every_turns=/every_tokens=)), so the event's slice includes the step."""
        agent = self._invocations[agent_id]
        end = agent.session.message_count
        self._subscriptions.activity("agent", agent_id, start, end)
        self._subscriptions.progress(
            agent.id,
            agent.turns,
            agent.new_tokens,
            end,
            {"name": agent.name, "status": agent.status},
        )

    def _publish_job_output(self, job: JobRecord, start: int) -> None:
        self._subscriptions.activity("job", job.info.id, start, job.info.output_bytes)

    async def _watch_operation(self, parent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "watch.agent":
            child = self._child(parent, request["agent_id"])
            thresholds = {
                key: request[key]
                for key in ("every_turns", "every_tokens")
                if request.get(key) is not None
            }
            sub = self._subscriptions.register(
                parent.id,
                "progress" if thresholds else "agent",
                child.id,
                cursor=child.session.message_count,
                completed=child.status in {"completed", "failed", "cancelled"},
                thresholds=thresholds or None,
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

    def _publish_job(self, job: JobRecord) -> None:
        self._subscriptions.finish("job", job.info.id)
        self._activity.set()
        owner = self._invocations[job.info.owner_id]
        if self._closed or owner.capability not in self._capabilities or not job.notify:
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
                    # Include a bounded output tail for inspecting the completion.
                    "text": self._shell_jobs.read(
                        job,
                        max(0, job.info.output_bytes - COMPLETED_TAIL_BYTES),
                        COMPLETED_TAIL_BYTES,
                    )["text"],
                },
                job.source_request_id,
            ),
        )

    @staticmethod
    def _collect_expr(job: JobRecord) -> str:
        """The exact expression that collects a job by id; markers and hints quote it
        verbatim rather than naming a variable the caller may not have."""
        return f'await (await rlm.shell.get("{job.info.id}")).result()'

    def _finished_payload(self, job: JobRecord) -> dict:
        output = self._shell_jobs.run_text(job)
        return {
            "id": job.info.id,
            "text": output["text"],
            "exit_code": job.info.exit_code,
            "truncated": output["truncated"],
            "error": job.info.error,
            "timed_out": job.info.status == "timed_out",
            "running": False,
        }

    def _running_payload(
        self, job: JobRecord, marker: str, *, partial: bool = True
    ) -> dict:
        output = (
            self._shell_jobs.read(job, 0, RUN_TEXT_BYTES)
            if marker and partial
            else {"text": "", "truncated": False}
        )
        return {
            "id": job.info.id,
            "text": marker + output["text"],
            "exit_code": None,
            "truncated": output["truncated"]
            or (bool(marker and partial) and job.info.output_bytes > RUN_TEXT_BYTES),
            "error": None,
            "timed_out": False,
            "running": True,
        }

    async def _shell_operation(self, parent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "shell.setenv":
            variables = request.get("variables") or {}
            if any(
                not k or "=" in k or "\0" in k or "\0" in v
                for k, v in variables.items()
            ):
                raise ValueError(
                    "environment variable names must be non-empty without '='"
                )
            parent.shell_env.update(variables)
            return dict(parent.shell_env)
        if op == "shell.getenv":
            return dict(parent.shell_env)
        if op == "shell.run":
            command = request["command"]
            if not command.strip():
                raise ValueError("empty command")
            blocked = find_blocked_command(
                command, allow_git=parent.runtime_config.policy.allow_git
            )
            if blocked:
                raise PermissionError(refusal(blocked))
            self._note_env_prefixes(parent, command)
            cwd = Path(parent.cwd) / (request["cwd"] or ".")
            yield_after = request.get("yield_after")
            block = (
                RUN_DETACH_SECONDS
                if yield_after is None
                else min(float(yield_after), RUN_BLOCK_MAX_SECONDS)
            )
            info = self._shell_jobs.start(
                owner_id=parent.id,
                command=command,
                cwd=str(cwd.resolve()),
                directory=parent.session.dir,
                env={
                    **build_kernel_env(dict(parent.runtime_config.kernel_env)),
                    **parent.shell_env,
                    **(request.get("env") or {}),
                },
                source_request_id=self._scopes[request["scope_id"]].request_id,
                # yield_after= bounds how long run() blocks; timeout= is the kill deadline.
                timeout=request.get("timeout"),
                # A result handed back synchronously needs no inbox event on top; a job
                # handed back while still running posts shell.completed when it ends.
                notify=block == 0,
            )
            job = self._shell_jobs.get(parent.id, info["id"])
            if block == 0:
                return self._running_payload(
                    job,
                    f"[started: job {job.info.id}; `res = {self._collect_expr(job)}` "
                    "(or .result() on this object) returns the output when it finishes]\n",
                    partial=False,
                )
            try:
                await asyncio.wait_for(asyncio.shield(job.task), block)
            except asyncio.TimeoutError:
                job.notify = True
                if (
                    yield_after is None
                    and parent.long_run_notes < MAX_LONG_RUN_NOTES
                    and "run-detach" not in parent.muted_hints
                ):
                    parent.long_run_notes += 1  # muted detaches do not spend the budget
                    self._hint(
                        parent,
                        "run-detach",
                        f"rlm.shell.run() returned after the default {RUN_DETACH_SECONDS:g} s "
                        "wait with the command still running (.running is True on the "
                        f"object it returned). Collect it with `res = {self._collect_expr(job)}` "
                        "(or .result() on that object; it waits up to 300 s per call, so no "
                        "native wait is needed). Pass yield_after=300 to wait longer up front, "
                        "or yield_after=0 to return at once for commands you expect to take long.",
                    )
                return self._running_payload(
                    job,
                    f"[yielded after {block:g} s: job {job.info.id} is still running; "
                    f"`res = {self._collect_expr(job)}` waits for it (up to 300 s per call; "
                    ".result() on this object does the same); .cancel() on it stops it]\n",
                )
            except Exception:
                if job.finished is None:
                    raise
            return self._finished_payload(job)
        if op == "shell.result":
            job = self._shell_jobs.get(parent.id, request["job_id"])
            yield_after = request.get("yield_after")
            block = (
                RUN_BLOCK_MAX_SECONDS
                if yield_after is None
                else min(float(yield_after), RUN_BLOCK_MAX_SECONDS)
            )
            if not job.task.done():
                try:
                    await asyncio.wait_for(asyncio.shield(job.task), block)
                except asyncio.TimeoutError:
                    job.notify = True
                    marker = (
                        f"[yielded after {block:g} s: job {job.info.id} is still running; "
                        f"`{self._collect_expr(job)}` waits again; .cancel() on it stops it]\n"
                        if block
                        else ""
                    )
                    return self._running_payload(job, marker)
                except Exception:
                    if job.finished is None:
                        raise
            # Collected through the handle: its completion event needs no separate read.
            for event in parent.inbox:
                if (
                    event["type"] == "shell.completed"
                    and not event["read"]
                    and event["content"].get("job_id") == job.info.id
                ):
                    self._record_event(
                        parent, {"type": "read", "event_id": event["id"]}
                    )
                    event["read"] = True
            return self._finished_payload(job)
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

    async def _services_operation(self, parent: _Invocation, request: dict) -> Any:
        op = request["op"]
        if op == "services.start":
            command = request["command"]
            if not command.strip():
                raise ValueError("empty command")
            blocked = find_blocked_command(
                command, allow_git=parent.runtime_config.policy.allow_git
            )
            if blocked:
                raise PermissionError(refusal(blocked))
            cwd = Path(parent.cwd) / (request["cwd"] or ".")
            return self._services.start(
                name=request["name"],
                command=command,
                cwd=str(cwd.resolve()),
                env={
                    **build_kernel_env(dict(parent.runtime_config.kernel_env)),
                    **parent.shell_env,
                    **(request.get("env") or {}),
                },
                owner_id=parent.id,
            )
        if op == "services.list":
            return self._services.list()
        if op == "services.get":
            return self._services.get(request["name"])
        if op == "services.tail":
            return self._services.tail(request["name"], request["lines"])
        if op == "services.stop":
            return await self._services.stop(request["name"], request["grace"])
        raise ValueError(f"unknown services operation {op!r}")

    async def _agent_operation(self, request: dict) -> Any:
        parent = self._caller(request["capability"], request["scope_id"])
        op = request["op"]
        if op.startswith("tools."):
            if self._mcp_registry is None:
                if op == "tools.servers":
                    return []
                if op == "tools.search":
                    if request.get("cursor") is not None:
                        raise ValueError("invalid MCP search cursor")
                    return {"matches": [], "next_cursor": None}
                raise ValueError("no MCP tools are configured")
            if op == "tools.servers":
                return self._mcp_registry.servers()
            if op == "tools.get":
                descriptor = self._mcp_registry.get(request["name"])
                if descriptor.capability not in self._mcp_capabilities:
                    self._brokered_skills[descriptor.capability] = (
                        descriptor,
                        partial(self._mcp_registry.call, descriptor.capability),
                    )
                    self._mcp_capabilities.add(descriptor.capability)
                return descriptor.to_dict()
            return self._mcp_registry.search(
                request["query"],
                request["limit"],
                request["server"],
                request.get("cursor"),
            )
        if op.startswith("watch."):
            return await self._watch_operation(parent, request)
        if op.startswith("shell."):
            return await self._shell_operation(parent, request)
        if op.startswith("services."):
            return await self._services_operation(parent, request)
        if op.startswith(("board.", "population.", "agent.dm", "agent.broadcast")):
            if not self._population:
                raise RuntimeError(f"{op} is only available in a population episode")
            if op.startswith("board."):
                return self._board_operation(parent, request)
            if op.startswith("population."):
                return self._population_operation(parent, request)
            if op == "agent.dm":
                return self._dm(
                    parent,
                    request["scope_id"],
                    request["agent_id"],
                    request["message"],
                    request.get("message_id"),
                )
            return self._broadcast(
                parent,
                request["scope_id"],
                request["message"],
                request.get("message_id"),
            )
        if op.startswith("hints."):
            tags = set(request.get("tags") or [])
            if op == "hints.mute":
                parent.muted_hints |= tags
            elif op == "hints.unmute":
                parent.muted_hints -= tags
            return {"muted": sorted(parent.muted_hints)}
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
        if self._population and op in {"agent.send", "agent.steer", "agent.report"}:
            raise RuntimeError(f"{op} is not used in a population: use rlm.agent.dm")
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
                    {
                        "agent_id": parent.id,
                        "name": parent.name,
                        "text": request["message"],
                    },
                    self._scopes[request["scope_id"]].request_id,
                ),
            )
        if op == "agent.spawn" and self._population:
            return self._info(
                await self._admit(
                    parent,
                    request["task"],
                    request["name"],
                    self._scopes[request["scope_id"]].request_id,
                )
            )
        if op in {"agent.get", "agent.info"} and self._population:
            return self._info(
                self._lookup(request.get("name_or_id") or request["agent_id"])
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
            if child.status == "cancelled" and not self._population:
                self.semantic_edges.record_subagent_cancel(
                    self._scopes[request["scope_id"]].request_id, child.id
                )
        elif op == "agent.result":
            yield_after = min(request["yield_after"], RUN_BLOCK_MAX_SECONDS)
            if not child.done.is_set() and yield_after > 0:
                try:
                    await asyncio.wait_for(child.done.wait(), timeout=yield_after)
                except asyncio.TimeoutError:
                    pass
            if child.status == "failed":
                self.semantic_edges.finish_subagent(
                    child.id, edge_type="subagent_failed"
                )
                raise RuntimeError(child.error or "agent failed")
            if child.status == "cancelled":
                if not self._population:
                    self.semantic_edges.record_subagent_cancel(
                        self._scopes[request["scope_id"]].request_id, child.id
                    )
                raise RuntimeError(child.error or "agent cancelled")
            if child.result is not None:
                self.semantic_edges.finish_subagent(
                    child.id, request_id=child.result_request_id
                )
            return agent_result_to_payload(
                AgentResult(
                    status=child.status,
                    answer=None if child.result is None else child.result.answer,
                    session_dir=child.session.dir,
                    usage=TokenUsage() if child.result is None else child.result.usage,
                    turns=child.turns,
                )
            )
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
            child.status = "running"
            factory = self._engine_factory
            if factory is None:
                from rlm.engine import RLMEngine

                factory = RLMEngine
            first_turn = child.engine is None
            if first_turn:
                child.engine = factory(
                    cwd=child.cwd,
                    session=child.session,
                    mcp_servers=child.mcp_servers,
                    runtime_config=child.runtime_config,
                    supervisor=self,
                    invocation_id=child.id,
                )
            if self._population:
                text, kwargs = self._population_turn(child, first_turn)
                child.result = await child.engine.prompt(text, **kwargs)
            elif first_turn:
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
                child.idle_since = time.monotonic()
                child.session.write_meta(**self._info(child))
        except asyncio.CancelledError:
            child.status = "cancelled"
        except BaseException as exc:
            child.status = "failed"
            child.error = str(exc) or type(exc).__name__
            child.failure = exc
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
                if self._population:
                    if child.status == "idle":
                        self._reply(child)
                elif parent is not None and parent.capability in self._capabilities:
                    self._publish(
                        parent,
                        self._event(
                            child,
                            "agent.completed",
                            {
                                "agent_id": child.id,
                                "name": child.name,
                                "status": child.status,
                                "answer": (
                                    child.result.answer[-COMPLETED_TAIL_BYTES:]
                                    if child.result is not None and child.result.answer
                                    else None
                                ),
                                "turns": (
                                    child.result.turns
                                    if child.result is not None
                                    else None
                                ),
                                "error": child.error,
                            },
                            self.semantic_edges.last_request_id(child.id),
                        ),
                    )
                if child.status == "idle" and (
                    child.instructions or len(child.inbox) > child.announced
                ):
                    self._wake_agent(child)

    def _population_turn(
        self, agent: _Invocation, first_turn: bool
    ) -> tuple[str, dict[str, Any]]:
        """The input that starts a population agent's turn, attributed to its source."""
        if first_turn and agent.parent_id is None:
            return agent.task, {
                "message_type": "assignment",
                "message": runtime_event("assignment", agent.task),
            }
        if first_turn:
            parent = self._invocations[agent.parent_id]
            return agent.task, {
                "message_type": "parent_message",
                "message": agent_input(
                    agent.task, agent=parent.name, kind="assignment"
                ),
            }
        events = self.take_instructions(agent.id, include_queue=True)
        if not events:
            text = "Supervisor: new inbox events are available."
            return text, {
                "message_type": "supervisor_notification",
                "message": runtime_event("notice", text),
            }
        rendered = [self.render_input(event) for event in events]
        text = "\n\n".join(message["content"] for message, _ in rendered)
        return text, {
            "message_type": "agent_message",
            "message": (
                {"role": "user", "content": text},
                {
                    "source": "agent",
                    "inputs": [provenance for _, provenance in rendered],
                },
            ),
            "event_ids": [event["id"] for event in events],
        }

    async def _close_agent_subscriptions(self, agent_id: str) -> None:
        try:
            self._subscriptions.finish("agent", agent_id)
            self._subscriptions.finish("progress", agent_id)
        finally:
            await self._subscriptions.close(agent_id)

    async def _release_agent(self, child: _Invocation) -> None:
        if self._population and self._population_open:
            self._announce("ended", child, status=child.status, error=child.error)
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
            # Population descendants outlive their ancestors.
            *(
                self._terminate(descendant)
                for descendant in list(self._invocations.values())
                if descendant.parent_id == child.id and not self._population
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
            if child.status in RESIDENT or child.status in {"retiring", "retired"}:
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
                    or (self._population and child.id != self.root_id)
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
