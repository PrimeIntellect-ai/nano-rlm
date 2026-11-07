from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from conftest import DummyMessage, DummyToolCall, DummyChoice, DummyResponse
from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    PopulationPolicy,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.shared_board import BoardConflict, SharedBoard
from rlm.swarm import Swarm
from rlm.types import RLMResult, TokenUsage


def _config(exec_timeout: int = 300, **population) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url="http://interceptor", api_key="secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(
            exec_timeout=exec_timeout, population=PopulationPolicy(**population)
        ),
    )


@dataclass
class _Agent:
    """A scripted agent's view of the supervisor during one turn."""

    supervisor: object
    invocation_id: str
    scope: str

    async def op(self, op: str, **arguments):
        agent = self.supervisor._invocations[self.invocation_id]
        if op == "agent.spawn":
            # The kernel API always sends these; parse_request would require them.
            arguments.setdefault("independent", False)
            arguments.setdefault("request_id", None)
        if op == "inbox.list":
            arguments.setdefault("unread_only", True)
        return await self.supervisor._agent_operation(
            {
                "op": op,
                "capability": agent.capability,
                "scope_id": self.scope,
                **arguments,
            }
        )

    def info(self):
        return self.supervisor._invocations[self.invocation_id]


class _ScriptEngine:
    """A population peer whose turns run scripted coroutines against the supervisor."""

    stop_reason = "done"

    def __init__(self, *, session, supervisor, invocation_id, **kwargs):
        self.session = session
        self.supervisor = supervisor
        self.invocation_id = invocation_id

    async def prompt(self, text, **kwargs):
        name = self.supervisor._invocations[self.invocation_id].name
        script = self.supervisor.script
        script.prompts.setdefault(name, []).append(text)
        notice = self.supervisor.inbox_notice(self.invocation_id)
        if notice:  # the engine's first call of the turn delivers the notice
            script.notices.setdefault(name, []).append(notice["text"])
        turns = script.turns.get(name, [])
        turn = len(script.prompts[name]) - 1
        scope = await self.supervisor.open_scope(self.invocation_id)
        try:
            answer = (
                await turns[turn](_Agent(self.supervisor, self.invocation_id, scope))
                if turn < len(turns)
                else f"{name} idle"
            )
        finally:
            await self.supervisor.close_scope(scope)
        # One model call per turn, as the real engine records it.
        self.supervisor.record_call(
            2, self.invocation_id, TokenUsage(prompt_tokens=10, completion_tokens=2)
        )
        self.stop_reason = script.stops.get(name, ["done"] * (turn + 1))[turn]
        return RLMResult(
            answer=answer,
            session_dir=self.session.dir,
            usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
            turns=1,
        )

    async def release_kernel(self):
        self.supervisor.script.released.append(
            self.supervisor._invocations[self.invocation_id].name
        )

    async def aclose(self):
        self.session.close()


@dataclass
class _Script:
    turns: dict[str, list] = field(default_factory=dict)  # name -> per-turn coroutines
    stops: dict[str, list[str]] = field(
        default_factory=dict
    )  # name -> per-turn reasons
    prompts: dict[str, list[str]] = field(default_factory=dict)
    notices: dict[str, list[str]] = field(default_factory=dict)
    released: list[str] = field(default_factory=list)


def _scripted_factory(script: _Script):
    def factory(**kwargs):
        engine = _ScriptEngine(**kwargs)
        return engine

    return factory


def _swarm(tmp_path, script: _Script, **population) -> Swarm:
    swarm = Swarm(
        cwd=str(tmp_path),
        session=Session(tmp_path / "episode"),
        runtime_config=_config(**population),
    )
    swarm.supervisor._engine_factory = _scripted_factory(script)
    swarm.supervisor.script = script
    return swarm


async def _wait_until(predicate, timeout=5.0):
    deadline = asyncio.get_event_loop().time() + timeout
    while not predicate():
        if asyncio.get_event_loop().time() > deadline:
            raise AssertionError("condition did not arrive")
        await asyncio.sleep(0.01)


async def test_independent_spawn_creates_a_depth0_peer_not_beholden(tmp_path):
    script = _Script()

    async def founder(agent):
        await agent.op(
            "agent.spawn", task="check 6*7", name="checker", independent=True
        )
        await agent.op(
            "agent.spawn", task="delegated work", name="helper", persistent=False
        )
        return "spawned"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=4, max_total_agents=4)
    supervisor = swarm.supervisor
    try:
        await asyncio.wait_for(swarm.prompt("compute 6*7"), 10)
        founder = supervisor._lookup("agent-1")
        peer = supervisor._lookup("checker")
        child = next(a for a in supervisor._population_agents() if a.name == "helper")
        founder_inbox = list(founder.inbox)
    finally:
        await swarm.aclose()

    # Independent: depth-0, parentless, provenance recorded for attribution only.
    assert peer.parent_id is None
    assert peer.runtime_config.invocation.depth == 0
    assert peer.spawned_by == founder.id
    # Beholden: a tree child of the peer that spawned it.
    assert child.parent_id == founder.id
    assert child.runtime_config.invocation.depth == 1
    # The peer's completion produced no lifecycle event for the spawner; the child's did.
    completed = [
        event["content"]["name"]
        for event in founder_inbox
        if event["type"] == "agent.completed"
    ]
    assert completed == ["helper"]
    snapshot = swarm.execution_snapshot()["population"]
    checker = next(a for a in snapshot["agents"] if a["name"] == "checker")
    assert checker["spawned_by"] == "agent-1" and checker["founder"] is False
    assert checker["parent"] is None


async def test_dms_land_in_the_inbox_as_one_coalesced_notice(tmp_path):
    script = _Script()
    seen = {}

    async def sender(agent):
        supervisor = agent.supervisor
        await _wait_until(lambda: supervisor._lookup("agent-2").status == "idle")
        await agent.op("agent.dm", agent_id="agent-2", message="what is 6*7?")
        await agent.op("agent.dm", agent_id="agent-2", message="and 9*6?")
        return "sent"

    async def waiting(agent):
        return "starting out"

    async def reader(agent):
        events = await agent.op("inbox.list")
        seen["events"] = events
        for event in events:
            content = (await agent.op("inbox.read", event_id=event["id"]))["content"]
            if "6*7" in content.get("text", ""):
                seen["dm"] = content
        await agent.op("population.submit", text="42", expected_revision=0)
        return "read and answered"

    script.turns = {"agent-1": [sender], "agent-2": [waiting, reader]}
    swarm = _swarm(tmp_path, script, founders=2, reminder_seconds=0)
    supervisor = swarm.supervisor
    try:
        result = await asyncio.wait_for(swarm.prompt("compute 6*7"), 10)
        receipts = {k: dict(v) for k, v in supervisor._deliveries.items()}
    finally:
        await swarm.aclose()

    assert result.answer == "42"
    # Both DMs arrived while agent-2 was idle; ONE coalesced notice woke it.
    notices = script.notices["agent-2"]
    assert len(notices) == 1
    assert "[inbox " in notices[0] and "2 dm" in notices[0]
    assert "waiting" in notices[0]
    # The DM content never entered a conversation, only the inbox.
    assert all("what is 6*7?" not in prompt for prompt in script.prompts["agent-2"])
    assert seen["dm"]["text"] == "what is 6*7?"
    assert seen["dm"]["name"] == "agent-1"
    # The sender's receipts were read and answered with the reply's tail, pulled.
    assert receipts and all(r["status"] == "answered" for r in receipts.values())
    assert all(r["answer"] == "read and answered" for r in receipts.values())


async def test_dm_receipt_answers_and_a_turn_cap_death_fails_it(tmp_path):
    script = _Script()
    outcomes = {}

    async def sender(agent):
        supervisor = agent.supervisor
        await _wait_until(lambda: supervisor._lookup("agent-2").status == "idle")
        first = await agent.op(
            "agent.dm", agent_id="agent-2", message="reply with 42", message_id="m1"
        )
        outcomes["first"] = first["message_id"]
        await _wait_until(
            lambda: supervisor._deliveries[outcomes["first"]]["status"] == "answered"
        )
        second = await agent.op(
            "agent.dm", agent_id="agent-2", message="one more thing", message_id="m2"
        )
        outcomes["second"] = second["message_id"]
        await _wait_until(lambda: supervisor._lookup("agent-2").status == "idle")
        with pytest.raises(RuntimeError, match="agent_turns_exhausted"):
            await agent.op(
                "agent.dm", agent_id="agent-2", message="never", message_id="m3"
            )
        return "asked"

    async def starting(agent):
        return "starting out"

    async def answers(agent):
        for event in await agent.op("inbox.list"):
            await agent.op("inbox.read", event_id=event["id"])
        await agent.op("population.submit", text="42", expected_revision=0)
        return "the answer is 42"

    async def reads_then_hits_the_cap(agent):
        for event in await agent.op("inbox.list"):
            await agent.op("inbox.read", event_id=event["id"])
        return "ran out of model calls"

    script.turns = {
        "agent-1": [sender],
        "agent-2": [starting, answers, reads_then_hits_the_cap],
    }
    script.stops = {
        "agent-2": ["done", "done", "max_agent_turns"],
    }
    swarm = _swarm(tmp_path, script, founders=2, max_agent_turns=3, reminder_seconds=0)
    supervisor = swarm.supervisor
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
        sender_agent = supervisor._lookup("agent-1")
        agent2 = supervisor._lookup("agent-2")
        answered = supervisor._delivery(sender_agent, outcomes["first"])
        failed = supervisor._delivery(sender_agent, outcomes["second"])
        capped = supervisor.spent_agent_turns(agent2.id)
    finally:
        await swarm.aclose()

    # The read-and-answered receipt carries the reply's tail, pulled.
    assert answered["status"] == "answered"
    assert answered["answer"] == "the answer is 42"
    # The turn-cap death failed the read message instead of answering it with the
    # salvage marker: no synthetic answer is ever recorded as a reply.
    assert failed["status"] == "failed"
    assert failed["reason"] == "recipient stopped: max_agent_turns"
    assert "turn budget reached" not in json.dumps(failed)
    assert all("turn budget reached" not in p for p in script.prompts["agent-1"])
    assert capped
    # A DM to a capped agent is refused outright.
    assert swarm.stop_reason in {"quiescent", "max_agent_turns"}


async def test_periodic_reminder_repeats_until_the_inbox_is_read(tmp_path):
    script = _Script()
    read_events = []

    async def sender(agent):
        await agent.op("agent.dm", agent_id="agent-2", message="please answer")
        return "sent"

    async def starting(agent):
        return "starting out"

    async def ignoring(agent):
        return "ignoring the inbox"

    async def procrastinator(agent):
        for event in await agent.op("inbox.list"):
            content = (await agent.op("inbox.read", event_id=event["id"]))["content"]
            read_events.append(content)
        await agent.op("population.submit", text="ok", expected_revision=0)
        return "read at last"

    script.turns = {
        "agent-1": [sender],
        "agent-2": [starting, ignoring, procrastinator],
    }
    swarm = _swarm(tmp_path, script, founders=2, reminder_seconds=0.05)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()

    supervisor = swarm.supervisor
    # The unread inbox kept the episode open; one reminder re-delivered the notice,
    # and reading it cleared the state.
    assert supervisor._counters["reminders"] >= 1
    assert any("please answer" in event["text"] for event in read_events)
    assert swarm.stop_reason == "quiescent"


async def test_unread_inbox_does_not_hold_a_closed_reminder_episode(tmp_path):
    script = _Script()

    async def sender(agent):
        await agent.op("agent.dm", agent_id="agent-2", message="never read")
        return "sent"

    async def never_reads(agent):
        return "busy elsewhere"

    script.turns = {"agent-1": [sender], "agent-2": [never_reads]}
    swarm = _swarm(tmp_path, script, founders=2, reminder_seconds=0)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    # With reminders disabled the notice was the recipient's chance; the episode ended.
    assert swarm.stop_reason == "quiescent"
    assert len(script.prompts["agent-2"]) == 1


async def test_full_population_retires_the_longest_idle_peer_for_a_birth(tmp_path):
    script = _Script()
    outcomes = {}

    async def spawner(agent):
        await agent.op("agent.spawn", task="first", name="first", independent=True)
        await asyncio.sleep(0.05)
        try:
            await agent.op(
                "agent.spawn", task="too soon", name="early", independent=True
            )
        except RuntimeError as error:
            outcomes["early"] = str(error)
        supervisor = agent.supervisor
        supervisor._lookup("first").idle_since -= 100
        await agent.op("agent.spawn", task="second", name="second", independent=True)
        await asyncio.sleep(0.05)
        try:
            await agent.op("agent.dm", agent_id="first", message="still there?")
        except RuntimeError as error:
            outcomes["dm"] = str(error)
        return "done"

    script.turns = {"agent-1": [spawner]}
    swarm = _swarm(
        tmp_path,
        script,
        founders=1,
        max_live_agents=2,
        max_total_agents=5,
        idle_grace_seconds=30,
        reminder_seconds=0,
    )
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()

    assert outcomes["early"].startswith("population_full")
    assert "grace period" in outcomes["early"]
    assert script.released == ["first"]
    # The caller and "second" hold both slots, so the retired peer cannot revive.
    assert outcomes["dm"].startswith("retired_no_capacity")
    summary = swarm.execution_snapshot()["population"]
    assert summary["retirements"] == 1
    assert summary["admitted"] == 3


async def test_dm_revives_a_retired_peer_only_with_a_free_slot(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        supervisor = agent.supervisor
        await agent.op("agent.spawn", task="a", name="a", independent=True)
        while supervisor._lookup("a").status != "idle":
            await asyncio.sleep(0.01)
        supervisor._lookup("a").idle_since -= 100
        await agent.op("agent.spawn", task="b", name="b", independent=True)
        try:
            await agent.op("agent.dm", agent_id="a", message="hello")
        except RuntimeError as error:
            outcomes["full"] = str(error)
        await supervisor._terminate(supervisor._lookup("b"))
        outcomes["revived"] = await agent.op(
            "agent.dm", agent_id="a", message="hello again"
        )
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(
        tmp_path,
        script,
        founders=1,
        max_live_agents=2,
        max_total_agents=3,
        idle_grace_seconds=1,
        reminder_seconds=0,
    )
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()

    assert outcomes["full"].startswith("retired_no_capacity")
    assert outcomes["revived"]["revived"] is True
    a = next(
        agent
        for agent in swarm.execution_snapshot()["population"]["agents"]
        if agent["name"] == "a"
    )
    assert a["kernel_generation"] == 1
    # The revived peer's new turn began with a generic wake prompt, not the DM text;
    # the message itself only ever lived in the inbox.
    assert script.prompts["a"][1].startswith("Supervisor:")
    assert "hello again" not in script.prompts["a"][1]


async def test_retirement_and_episode_stop_stop_services(tmp_path):
    script = _Script()
    stopped = []

    async def founder(agent):
        await agent.op(
            "services.start",
            name="founder-service",
            command="sleep 60",
            cwd=None,
            env=None,
        )
        await agent.op("agent.spawn", task="a", name="a", independent=True)
        supervisor = agent.supervisor
        await _wait_until(lambda: supervisor._lookup("a").status == "idle")
        supervisor._lookup("a").idle_since -= 100
        await agent.op("agent.spawn", task="b", name="b", independent=True)
        # Retiring "a" stopped its service; the founder's own survives its turns.
        stopped.append(
            [svc["name"] for svc in supervisor._services.list() if svc["running"]]
        )
        return "done"

    async def runs_a_service(agent):
        await agent.op(
            "services.start",
            name="a-service",
            command="sleep 60",
            cwd=None,
            env=None,
        )
        return "service up"

    script.turns = {"agent-1": [founder], "a": [runs_a_service]}
    swarm = _swarm(
        tmp_path, script, founders=1, max_live_agents=2, idle_grace_seconds=1
    )
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()

    supervisor = swarm.supervisor
    # Retiring "a" stopped its service while the founder's survived; the episode's stop
    # ended every remaining service, so nothing outlives the scored episode.
    assert stopped[0] == ["founder-service"]
    assert [svc["name"] for svc in supervisor._services.list() if svc["running"]] == []


async def test_cumulative_cap_blocks_births_but_not_messages(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        await agent.op("agent.spawn", task="x", name="x", independent=True)
        try:
            await agent.op("agent.spawn", task="y", name="y", independent=True)
        except RuntimeError as error:
            outcomes["exhausted"] = str(error)
        outcomes["dm"] = await agent.op("agent.dm", agent_id="x", message="go")
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(
        tmp_path,
        script,
        founders=1,
        max_live_agents=4,
        max_total_agents=2,
        reminder_seconds=0,
    )
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert outcomes["exhausted"].startswith("population_exhausted")
    assert outcomes["dm"]["status"] == "accepted"


async def test_concurrent_spawns_never_exceed_the_live_cap(tmp_path):
    script = _Script()
    outcomes = []

    async def founder(agent):
        async def spawn(name):
            try:
                await agent.op("agent.spawn", task="t", name=name, independent=True)
                outcomes.append("ok")
            except RuntimeError as error:
                outcomes.append(str(error).split(":")[0])

        await asyncio.gather(*(spawn(f"c{i}") for i in range(6)))
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=3, max_total_agents=10)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert outcomes.count("ok") == 2
    assert outcomes.count("population_full") == 4
    assert swarm.execution_snapshot()["population"]["peak_live"] == 3


async def test_episode_wall_clock_stops_running_agents(tmp_path):
    script = _Script()

    async def forever(agent):
        await asyncio.Future()

    script.turns = {"agent-1": [forever]}
    swarm = _swarm(tmp_path, script, founders=1, max_seconds=0.2)
    try:
        result = await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert swarm.stop_reason == "max_seconds"
    assert result.answer == ""


async def test_an_agent_failure_fails_the_episode(tmp_path):
    script = _Script()
    errors = {}

    async def founder(agent):
        supervisor = agent.supervisor
        await agent.op("agent.spawn", task="a", name="a", independent=True)
        while supervisor._lookup("a").status != "idle":
            await asyncio.sleep(0.01)
        supervisor._lookup("a").idle_since -= 100

        async def broken_release():
            raise OSError("kernel would not stop")

        supervisor._lookup("a").engine.release_kernel = broken_release
        try:
            await agent.op("agent.spawn", task="b", name="b", independent=True)
        except OSError as error:
            errors["spawn"] = str(error)
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(
        tmp_path, script, founders=1, max_live_agents=2, idle_grace_seconds=1
    )
    try:
        with pytest.raises(OSError, match="kernel would not stop"):
            await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert errors["spawn"] == "kernel would not stop"
    assert swarm.stop_reason == "agent_failed"
    a = swarm.execution_snapshot()["population"]["agents"][1]
    assert a["status"] == "failed"


async def test_running_shell_jobs_keep_the_episode_open(tmp_path):
    script = _Script()

    async def start_job(agent):
        await agent.op(
            "shell.run",
            command="sleep 1",
            cwd=None,
            yield_after=0,
            timeout=None,
            env=None,
        )
        return "waiting for the job"

    async def collect(agent):
        return "job finished"

    script.turns = {"agent-1": [start_job, collect]}
    swarm = _swarm(tmp_path, script, founders=1)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert len(script.prompts["agent-1"]) == 2


async def test_an_idle_agents_job_holds_the_episode_only_for_exec_timeout(tmp_path):
    script = _Script()

    async def leave_a_server(agent):
        await agent.op(
            "shell.run",
            command="sleep 1000",
            cwd=None,
            yield_after=0,
            timeout=None,
            env=None,
        )
        return "server left running"

    script.turns = {"agent-1": [leave_a_server]}
    swarm = _swarm(tmp_path, script, founders=1, exec_timeout=1)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert swarm.stop_reason == "quiescent"
    assert len(script.prompts["agent-1"]) == 1


async def test_a_job_finishing_mid_turn_wakes_its_idle_agent(tmp_path):
    script = _Script()

    async def keep_working(agent):
        await agent.op(
            "shell.run",
            command="sleep 0.1",
            cwd=None,
            yield_after=0,
            timeout=None,
            env=None,
        )
        await asyncio.sleep(0.5)
        # The engine reads the notices before every model call; a completion has none.
        assert agent.supervisor.inbox_notice(agent.invocation_id) is None
        return "I will look at the job when it completes"

    async def collect(agent):
        return "job finished"

    script.turns = {"agent-1": [keep_working, collect]}
    swarm = _swarm(tmp_path, script, founders=1)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert len(script.prompts["agent-1"]) == 2


async def test_peers_outlive_spawners_but_beholden_children_cascade(tmp_path):
    script = _Script()
    child_started = asyncio.Event()
    child_release = asyncio.Event()

    async def founder(agent):
        await agent.op("agent.spawn", task="work", name="child", persistent=False)
        await agent.op("agent.spawn", task="peer work", name="peer", independent=True)
        await child_started.wait()
        return "done"

    async def child(agent):
        child_started.set()
        await child_release.wait()
        return "child answer"

    async def peer(agent):
        await agent.op("population.submit", text="peer done", expected_revision=0)
        return "peer done"

    script.turns = {"agent-1": [founder], "child": [child], "peer": [peer]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=4, max_total_agents=4)
    supervisor = swarm.supervisor
    try:
        episode = asyncio.create_task(swarm.prompt("task"))
        await child_started.wait()
        founder = supervisor._lookup("agent-1")
        await founder.done.wait()
        # End the founder from outside, as a stop would: its beholden child cascades,
        # its independent peer does not.
        await supervisor._terminate(founder)
        child = next(a for a in supervisor._population_agents() if a.name == "child")
        child_state = child.status
        child_release.set()
        result = await asyncio.wait_for(episode, 10)
        # The episode's stop cancelled everyone; the pre-stop snapshot shows the peer
        # still idle, untouched by its spawner's end.
        agents = swarm.execution_snapshot()["population"]["agents"]
        peer_state = next(a for a in agents if a["name"] == "peer")["status"]
    finally:
        await swarm.aclose()
    assert founder.status == "cancelled"
    assert child_state == "cancelled"
    assert peer_state == "idle"
    assert result.answer == "peer done"


async def test_messages_are_idempotent_and_receipted(tmp_path):
    script = _Script()
    seen = {}

    async def founder(agent):
        supervisor = agent.supervisor
        await _wait_until(lambda: supervisor._lookup("agent-2").status == "idle")
        first = await agent.op(
            "agent.dm", agent_id="agent-2", message="hi", message_id="m1"
        )
        again = await agent.op(
            "agent.dm", agent_id="agent-2", message="hi", message_id="m1"
        )
        assert again == first
        with pytest.raises(ValueError, match="already used"):
            await agent.op(
                "agent.dm", agent_id="agent-2", message="other", message_id="m1"
            )
        seen["accepted"] = await agent.op(
            "agent.delivery", message_id=first["message_id"]
        )
        broadcast = await agent.op("agent.broadcast", message="all", message_id="b1")
        seen["broadcast"] = broadcast
        seen["first"] = first
        return "sent"

    async def starting(agent):
        return "starting out"

    async def reader(agent):
        await _wait_until(
            lambda: any(
                event["type"] == "agent.broadcast" for event in agent.info().inbox
            )
        )
        for event in await agent.op("inbox.list"):
            await agent.op("inbox.read", event_id=event["id"])
        return "hi back"

    script.turns = {"agent-1": [founder], "agent-2": [starting, reader]}
    swarm = _swarm(tmp_path, script, founders=2, reminder_seconds=0)
    supervisor = swarm.supervisor
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
        sender = supervisor._lookup("agent-1")
        answered = supervisor._delivery(sender, seen["first"]["message_id"])
        broadcast_receipts = [
            supervisor._delivery(sender, item["message_id"])
            for item in seen["broadcast"]["delivered"]
        ]
    finally:
        await swarm.aclose()
    assert seen["accepted"]["status"] == "accepted"
    assert answered["status"] == "answered" and answered["to"] == "agent-2"
    assert answered["answer"] == "hi back"
    assert [item["to"] for item in seen["broadcast"]["delivered"]] == ["agent-2"]
    assert all(r["status"] == "answered" for r in broadcast_receipts)
    assert swarm.execution_snapshot()["population"]["dms"] == 1


async def test_failed_requests_are_retriable_with_the_same_id(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        try:
            await agent.op("agent.dm", agent_id="agent-9", message="x", message_id="r1")
        except ValueError as error:
            outcomes["missing"] = str(error)
        receipt = await agent.op(
            "agent.dm", agent_id="agent-2", message="x", message_id="r1"
        )
        outcomes["retry"] = receipt
        return "sent"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=2, reminder_seconds=0)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    # A rejection is not remembered: the retry ran instead of replaying the error.
    assert "no agent named" in outcomes["missing"]
    assert outcomes["retry"]["status"] == "accepted"
    assert swarm.execution_snapshot()["population"]["dms"] == 1


async def test_spawn_retries_do_not_duplicate_births(tmp_path):
    script = _Script()

    async def founder(agent):
        results = await asyncio.gather(
            *(
                agent.op(
                    "agent.spawn",
                    task="t",
                    name="worker",
                    independent=True,
                    request_id="spawn-1",
                )
                for _ in range(3)
            )
        )
        assert len({result["id"] for result in results}) == 1
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1)
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert swarm.execution_snapshot()["population"]["admitted"] == 2


async def test_membership_changes_arrive_in_the_inbox_without_waking(tmp_path):
    script = _Script()
    release = asyncio.Event()

    async def founder(agent):
        supervisor = agent.supervisor
        await _wait_until(lambda: supervisor._lookup("agent-2").status == "idle")
        await agent.op("agent.spawn", task="join", name="new", independent=True)
        await release.wait()
        return "done"

    async def idle_peer(agent):
        release.set()
        return "idle"

    script.turns = {"agent-1": [founder], "agent-2": [idle_peer, idle_peer]}
    swarm = _swarm(tmp_path, script, founders=2, max_live_agents=3, reminder_seconds=0)
    supervisor = swarm.supervisor
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
        peer_events = list(supervisor._lookup("agent-2").inbox)
    finally:
        await swarm.aclose()
    # The join arrived as an inbox event; agent-2 was not woken for it: it saw the
    # event only when its founder's release woke it.
    membership = [event for event in peer_events if event["type"] == "population.event"]
    assert membership and any(
        "new joined" in event["content"]["summary"] for event in membership
    )
    # agent-2 ran exactly its first scripted turn: the join never woke it.
    assert len(script.prompts["agent-2"]) == 1
    events = json.loads(
        (Path(swarm.session.dir) / "population.jsonl").read_text().splitlines()[0]
    )
    assert events["type"] == "joined"


async def test_artifacts_are_published_by_revision(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        drafts = agent.info().session.dir / "drafts"
        (drafts / "a.txt").write_text("first")
        state = await agent.op("artifact.read", path="out/final.txt")
        assert state["revision"] == 0 and not state["exists"]
        await agent.op(
            "artifact.publish",
            path="out/final.txt",
            candidate_path=str(drafts / "a.txt"),
            expected_revision=0,
        )
        (drafts / "a.txt").write_text("second")
        with pytest.raises(RuntimeError, match="artifact_conflict"):
            await agent.op(
                "artifact.publish",
                path="out/final.txt",
                candidate_path=str(drafts / "a.txt"),
                expected_revision=0,
            )
        with pytest.raises(ValueError, match="not a final artifact"):
            await agent.op("artifact.read", path="other.txt")
        outcomes["state"] = await agent.op("artifact.read", path="out/final.txt")
        return "published"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1, artifacts=["out/final.txt"])
    try:
        await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert (tmp_path / "out" / "final.txt").read_text() == "first"
    assert (
        outcomes["state"]["revision"] == 1 and outcomes["state"]["author"] == "agent-1"
    )
    snapshot = swarm.execution_snapshot()["population"]
    assert snapshot["artifacts"] == {
        "out/final.txt": {"revision": 1, "author": "agent-1"}
    }


async def test_answer_conflicts_history_permission_and_budget_stop(tmp_path):
    script = _Script()

    async def founder(agent):
        await agent.op("population.submit", text="1", expected_revision=0)
        with pytest.raises(RuntimeError, match="answer_conflict"):
            await agent.op("population.submit", text="2", expected_revision=0)
        with pytest.raises(ValueError, match="raw string"):
            await agent.op("population.submit", text="\boxed{2}", expected_revision=1)
        with pytest.raises(PermissionError, match="disabled"):
            await agent.op("population.history", agent="agent-1", start=0, limit=5)
        agent.supervisor.record_call(10**9, agent.invocation_id)
        with pytest.raises(RuntimeError, match="budget_exhausted"):
            await agent.op("agent.spawn", task="t", name="late", independent=True)
        return "spent"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1)
    try:
        result = await asyncio.wait_for(swarm.prompt("task"), 10)
    finally:
        await swarm.aclose()
    assert result.answer == "1"
    assert swarm.stop_reason == "max_total_tokens"


def test_board_revisions_conflicts_and_retries(tmp_path):
    board = SharedBoard(tmp_path / "board.md")
    first = board.append("a", "plan: try induction", request_id="r1")
    assert board.append("a", "plan: try induction", request_id="r1") == first
    with pytest.raises(ValueError, match="different board write"):
        board.append("a", "other", request_id="r1")
    read = board.read()
    board.edit("b", "induction", "contradiction", expected_revision=read["revision"])
    with pytest.raises(BoardConflict):
        board.edit("c", "plan", "PLAN", expected_revision=read["revision"])
    with pytest.raises(ValueError, match="exactly once"):
        board.edit("c", "missing", "x", expected_revision=board.revision)
    assert "plan: try contradiction" in (tmp_path / "board.md").read_text()
    assert [
        json.loads(line)["author"] for line in board.journal.read_text().splitlines()
    ] == ["a", "b"]
    assert board.search("contradiction")["matches"][0]["line"] == read["total_lines"]


def test_population_prompt_replaces_tree_instructions():
    from rlm.tools import get_active_builtin_tools

    tools = get_active_builtin_tools(300, None)
    population = {
        "founder": False,
        "spawned_by": "agent-1",
        "task": "Prove the lemma.",
        "founders": 4,
        "max_live_agents": 8,
        "max_total_agents": 32,
        "idle_grace_seconds": 60.0,
        "reminder_seconds": 60.0,
        "allow_agent_lookup_agent_history": False,
        "max_seconds": None,
        "artifacts": ("proof/Solution.lean",),
        "drafts": "/s/drafts",
        "job_hold_seconds": 300,
        "scored_on": "answer",
        "max_agent_turns": None,
    }
    text = build_system_prompt(
        "/w",
        [],
        depth=0,
        session_dir="/s",
        allow_recursion=True,
        allow_git=False,
        active_tools=tools,
        agent_info={
            "id": "x",
            "parent_id": None,
            "name": "agent-5",
            "persistent": True,
            "population": population,
        },
    )
    assert "## Episode task\nProve the lemma." in text
    assert "spawned by agent-1" in text
    assert "independent=True" in text
    assert "not beholden to you" in text
    assert "ONE coalesced" in text
    assert "[inbox 14:03] 3 messages waiting" in text
    assert "rlm.agent.delivery" in text
    assert "rlm.population.submit" in text
    assert "rlm.population.history" not in text
    assert "`proof/Solution.lean`" in text and "/s/drafts" in text
    workspace = build_system_prompt(
        "/w",
        [],
        session_dir="/s",
        allow_recursion=True,
        allow_git=False,
        active_tools=tools,
        agent_info={
            "id": "x",
            "parent_id": None,
            "name": "agent-1",
            "persistent": True,
            "population": {
                **population,
                "founder": True,
                "spawned_by": None,
                "scored_on": "workspace",
                "max_agent_turns": 40,
                "artifacts": (),
            },
        },
    )
    assert "final state of the shared workspace" in workspace
    assert "rlm.population.submit" not in workspace
    assert "at most 40 model calls" in workspace
    for tree_rule in (
        "You are a sub-agent",
        "message siblings",
        "Terminating an agent\ncleans up its children",
        "Delegation is disabled",
        "You are the root agent",
        "agent.completed",
    ):
        assert tree_rule not in text


class _ReactiveClient:
    """Answers each model call from the conversation so far, whatever the timing."""

    def __init__(self, policy):
        self.policy = policy
        self.chat = self
        self.completions = self

    async def create(self, **kwargs):
        return DummyResponse(
            choices=[DummyChoice(message=self.policy(kwargs["messages"]))]
        )

    async def close(self):
        pass


def _text(messages) -> str:
    return "\n".join(str(message.get("content")) for message in messages)


def _cell(code: str) -> DummyMessage:
    return DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})])


ASK_CELL = """await rlm.board.append("agent-1: asked agent-2 for 6*7")
r = await rlm.agent.dm("agent-2", "please compute 6*7")
print(r["status"])
while True:
    d = await rlm.agent.delivery(r["message_id"])
    if d["status"] in {"answered", "failed"}:
        break
    await asyncio.sleep(0.05)
print(d["answer"])
"""

ANSWER_CELL = """dm = [e for e in await rlm.inbox.list() if e["type"] == "agent.dm"][0]
event = await rlm.inbox.read(dm["id"])
print(event["content"]["text"])
cur = await rlm.population.answer()
print(await rlm.population.submit(str(6 * 7), expected_revision=cur["revision"]))
print((await rlm.board.read())["text"])
"""


def _asker(messages):
    tools = sum(message["role"] == "tool" for message in messages)
    if tools == 0:
        return _cell(ASK_CELL)
    return DummyMessage(content="agent-2 answered from the receipt")


def _answerer(messages):
    tool_messages = [m for m in messages if m["role"] == "tool"]
    if not tool_messages:
        return _cell("print(await rlm.inbox.list())")
    last = str(tool_messages[-1].get("content"))
    if "'type': 'agent.dm'" in last and "revision" not in last:
        return _cell(ANSWER_CELL)
    if "revision" in last:
        return DummyMessage(content="the answer is 42")
    return _cell("print(await rlm.inbox.list())")


async def test_real_kernels_run_one_population_episode(tmp_path):
    policies = {"agent-1": _asker, "agent-2": _answerer}
    swarm = Swarm(
        cwd=str(tmp_path),
        session=Session(tmp_path / "episode"),
        runtime_config=_config(founders=2, max_live_agents=2, max_total_agents=2),
    )
    supervisor = swarm.supervisor

    def factory(**kwargs):
        name = supervisor._invocations[kwargs["invocation_id"]].name
        return RLMEngine(client=_ReactiveClient(policies[name]), **kwargs)

    supervisor._engine_factory = factory
    try:
        result = await asyncio.wait_for(swarm.prompt("compute 6*7"), 120)
    finally:
        await swarm.aclose()

    assert result.answer == "42"
    assert swarm.stop_reason == "quiescent"
    summary = json.loads((tmp_path / "episode" / "population_summary.json").read_text())
    assert summary["dms"] == 1
    assert "agent-1: asked agent-2 for 6*7" in summary["board"]
    # The DM reached its recipient through the inbox, never as a pushed conversation
    # record: it appears only in tool results (the sender's own cell code included).
    reads = 0
    for peer_dir in (tmp_path / "episode").glob("sub-*"):
        records = [
            json.loads(line)
            for line in (peer_dir / "messages.jsonl").read_text().splitlines()
        ]
        for record in records:
            if "please compute 6*7" not in json.dumps(record):
                continue
            if record.get("type") == "tool_result":
                reads += 1
            else:
                assert record.get("type") == "assistant"
    assert reads


async def test_acp_runs_a_population_session(monkeypatch, tmp_path):
    from acp import text_block

    from rlm import acp
    from test_acp import _Client, _initialize, _runtime_metadata

    script = _Script()

    async def founder(agent):
        await agent.op("population.submit", text="42", expected_revision=0)
        return "submitted"

    script.turns = {"agent-1": [founder]}

    class ScriptedSwarm(Swarm):
        def __init__(self, **kwargs):
            super().__init__(**kwargs)
            self.supervisor._engine_factory = _scripted_factory(script)
            self.supervisor.script = script

    monkeypatch.setenv("RLM_HOME", str(tmp_path / "rlm"))
    monkeypatch.setattr(acp, "Swarm", ScriptedSwarm)
    metadata = _runtime_metadata()
    policy = metadata[acp.RUNTIME_METADATA_KEY]["policy"]
    policy["population"] = {"founders": 2, "max_live_agents": 3, "artifacts": ["a.txt"]}
    client = _Client()
    agent = acp.RLMACPAgent()
    agent.on_connect(client)
    await _initialize(agent)
    created = await agent.new_session(str(tmp_path), **metadata)
    response = await agent.prompt(created.session_id, [text_block("compute")])
    closed = await agent.close_session(created.session_id)

    assert [update.content.text for _, update in client.updates] == ["42"]
    snapshot = response.field_meta[acp.SESSION_METADATA_KEY]
    assert snapshot["population"]["admitted"] == 2
    assert snapshot["metrics"]["population_answered"] == 1
    assert snapshot["last_stop_reason"] == "quiescent"
    assert closed.field_meta[acp.SESSION_METADATA_KEY]["population"]["admitted"] == 2
