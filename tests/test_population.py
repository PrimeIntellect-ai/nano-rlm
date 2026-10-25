from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass, field

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
from rlm.tools import get_active_builtin_tools
from rlm.types import RLMResult, TokenUsage


def _config(**population) -> RuntimeConfig:
    return RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(base_url="http://interceptor", api_key="secret"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(population=PopulationPolicy(**population)),
    )


@dataclass
class _Agent:
    """A scripted agent's view of the supervisor during one turn."""

    supervisor: object
    invocation_id: str
    scope: str

    async def op(self, op: str, **arguments):
        agent = self.supervisor._invocations[self.invocation_id]
        return await self.supervisor._agent_operation(
            {
                "op": op,
                "capability": agent.capability,
                "scope_id": self.scope,
                **arguments,
            }
        )


@dataclass
class _Script:
    turns: dict[str, list] = field(default_factory=dict)  # name -> per-turn coroutines
    prompts: dict[str, list[str]] = field(default_factory=dict)
    released: list[str] = field(default_factory=list)


def _scripted_factory(script: _Script):
    class Engine:
        def __init__(self, *, session, supervisor, invocation_id, **kwargs):
            self.session = session
            self.supervisor = supervisor
            self.invocation_id = invocation_id

        async def prompt(self, text, **kwargs):
            name = self.supervisor._invocations[self.invocation_id].name
            script.prompts.setdefault(name, []).append(text)
            turns = script.turns.get(name, [])
            turn = len(script.prompts[name]) - 1
            scope = await self.supervisor.open_scope(self.invocation_id)
            try:
                answer = (
                    await turns[turn](
                        _Agent(self.supervisor, self.invocation_id, scope)
                    )
                    if turn < len(turns)
                    else f"{name} idle"
                )
            finally:
                await self.supervisor.close_scope(scope)
            return RLMResult(
                answer=answer,
                session_dir=self.session.dir,
                usage=TokenUsage(prompt_tokens=10, completion_tokens=2),
                turns=1,
            )

        async def release_kernel(self):
            script.released.append(
                self.supervisor._invocations[self.invocation_id].name
            )

        async def aclose(self):
            self.session.close()

    return Engine


def _swarm(tmp_path, script: _Script, **population) -> Swarm:
    swarm = Swarm(
        cwd=str(tmp_path),
        session=Session(tmp_path / "episode"),
        runtime_config=_config(**population),
    )
    swarm.supervisor._engine_factory = _scripted_factory(script)
    return swarm


async def test_founders_message_reply_and_submit_the_answer(tmp_path):
    script = _Script()

    async def founder_one(agent):
        child = await agent.op(
            "agent.spawn", task="check 6*7", name="checker", persistent=False
        )
        assert child["parent_id"] is not None
        await agent.op("agent.dm", agent_id="agent-2", message="what is 6*7?")
        return "asked"

    async def founder_two_answers(agent):
        cur = await agent.op("population.answer")
        await agent.op(
            "population.submit", text="42", expected_revision=cur["revision"]
        )
        return "42"

    async def wait_for_dm(agent):
        return "waiting"

    script.turns = {
        "agent-1": [founder_one],
        "agent-2": [wait_for_dm, founder_two_answers],
    }
    swarm = _swarm(tmp_path, script, founders=2, max_live_agents=4, max_total_agents=4)
    try:
        result = await swarm.prompt("compute 6*7")
    finally:
        await swarm.aclose()

    assert result.answer == "42"
    assert swarm.stop_reason == "quiescent"
    # agent-2 answered the DM, so agent-1 got a reply turn; checker replied to its parent.
    assert any("42" in prompt for prompt in script.prompts["agent-1"][1:])
    assert script.prompts["checker"] == ["check 6*7"]
    snapshot = swarm.execution_snapshot()
    metrics = snapshot["metrics"]
    assert metrics["population_admitted"] == 3
    assert metrics["population_dms"] == 1
    assert metrics["population_replies"] >= 2
    assert metrics["population_answered"] == 1
    assert snapshot["population"]["answer"]["author"] == "agent-2"
    events = [
        json.loads(line)
        for line in (tmp_path / "episode" / "population.jsonl").read_text().splitlines()
    ]
    assert [event["type"] for event in events][:3] == ["joined", "joined", "joined"]


async def test_descendants_outlive_a_cancelled_parent(tmp_path):
    script = _Script()
    release = asyncio.Event()

    async def parent(agent):
        await agent.op("agent.spawn", task="work", name="child", persistent=False)
        return "spawned"

    async def child(agent):
        await release.wait()
        cur = await agent.op("population.answer")
        await agent.op(
            "population.submit", text="done", expected_revision=cur["revision"]
        )
        return "finished"

    script.turns = {"agent-1": [parent], "child": [child]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=3)
    try:
        episode = asyncio.create_task(swarm.prompt("task"))
        supervisor = swarm.supervisor
        while "child" not in [a.name for a in supervisor._peers()]:
            await asyncio.sleep(0.01)
        founder = supervisor._lookup("agent-1")
        await founder.done.wait()
        await supervisor._terminate(founder)
        assert founder.status == "cancelled"
        assert supervisor._lookup("child").status in {"starting", "running"}
        release.set()
        result = await episode
    finally:
        await swarm.aclose()
    assert result.answer == "done"
    assert swarm.execution_snapshot()["population"]["agents"][1]["status"] == "idle"


async def test_full_population_retires_the_longest_idle_agent_for_a_birth(tmp_path):
    script = _Script()
    outcomes = {}

    async def spawner(agent):
        await agent.op("agent.spawn", task="first", name="first", persistent=False)
        await asyncio.sleep(0.05)
        try:
            await agent.op(
                "agent.spawn", task="too soon", name="early", persistent=False
            )
        except RuntimeError as error:
            outcomes["early"] = str(error)
        supervisor = agent.supervisor
        supervisor._lookup("first").idle_since -= 100
        await agent.op("agent.spawn", task="second", name="second", persistent=False)
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
    )
    try:
        await swarm.prompt("task")
    finally:
        await swarm.aclose()

    assert outcomes["early"].startswith("population_full")
    assert "grace period" in outcomes["early"]
    assert script.released == ["first"]
    # The caller and "second" hold both slots, so the retired agent cannot revive.
    assert outcomes["dm"].startswith("retired_no_capacity")
    summary = swarm.execution_snapshot()["population"]
    assert summary["retirements"] == 1
    assert summary["admitted"] == 3


async def test_dm_revives_a_retired_agent_only_with_a_free_slot(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        supervisor = agent.supervisor
        await agent.op("agent.spawn", task="a", name="a", persistent=False)
        while supervisor._lookup("a").status != "idle":
            await asyncio.sleep(0.01)
        supervisor._lookup("a").idle_since -= 100
        await agent.op("agent.spawn", task="b", name="b", persistent=False)
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
    )
    try:
        await swarm.prompt("task")
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
    assert script.prompts["a"][1].count("hello again") == 1


async def test_cumulative_cap_blocks_births_but_not_messages(tmp_path):
    script = _Script()
    outcomes = {}

    async def founder(agent):
        await agent.op("agent.spawn", task="x", name="x", persistent=False)
        try:
            await agent.op("agent.spawn", task="y", name="y", persistent=False)
        except RuntimeError as error:
            outcomes["exhausted"] = str(error)
        outcomes["dm"] = await agent.op("agent.dm", agent_id="x", message="go")
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=4, max_total_agents=2)
    try:
        await swarm.prompt("task")
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
                await agent.op("agent.spawn", task="t", name=name, persistent=False)
                outcomes.append("ok")
            except RuntimeError as error:
                outcomes.append(str(error).split(":")[0])

        await asyncio.gather(*(spawn(f"c{i}") for i in range(6)))
        return "done"

    script.turns = {"agent-1": [founder]}
    swarm = _swarm(tmp_path, script, founders=1, max_live_agents=3, max_total_agents=10)
    try:
        await swarm.prompt("task")
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
        result = await swarm.prompt("task")
    finally:
        await swarm.aclose()
    assert swarm.stop_reason == "max_seconds"
    assert result.answer == ""


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
    tools = get_active_builtin_tools(300, None)
    population = {
        "founder": False,
        "parent_name": "agent-1",
        "task": "Prove the lemma.",
        "board_path": "/s/board.md",
        "founders": 4,
        "max_live_agents": 8,
        "max_total_agents": 32,
        "idle_grace_seconds": 60.0,
        "allow_agent_lookup_agent_history": False,
        "max_seconds": None,
    }
    text = build_system_prompt(
        "/w",
        [],
        depth=3,
        session_dir="/s",
        allow_recursion=False,
        allow_git=False,
        active_tools=tools,
        agent_info={
            "id": "x",
            "parent_id": "p",
            "name": "agent-5",
            "persistent": True,
            "population": population,
        },
    )
    assert "## Episode task\nProve the lemma." in text
    assert "spawned by agent-1" in text
    assert "rlm.population.submit" in text
    assert "rlm.population.history" not in text
    for tree_rule in (
        "You are a sub-agent",
        "message siblings",
        "Terminating an agent\ncleans up its children",
        "Delegation is disabled",
        "You are the root agent",
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


def _agent_one(messages):
    text = _text(messages)
    tools = sum(message["role"] == "tool" for message in messages)
    if tools == 0:
        return _cell(
            'await rlm.board.append("agent-1: asked agent-2 for 6*7")\n'
            'print((await rlm.agent.dm("agent-2", "please compute 6*7"))["status"])'
        )
    if 'kind="reply"' in text:
        return DummyMessage(content="agent-2 answered")
    return DummyMessage(content="waiting for agent-2")


def _agent_two(messages):
    text = _text(messages)
    tools = sum(message["role"] == "tool" for message in messages)
    if "please compute 6*7" not in text:
        return DummyMessage(content="nothing to do yet")
    if tools == 0:
        return _cell(
            "cur = await rlm.population.answer()\n"
            'print(await rlm.population.submit(str(6 * 7), expected_revision=cur["revision"]))\n'
            "print((await rlm.board.read())['text'])"
        )
    return DummyMessage(content="submitted 42")


async def test_real_kernels_run_one_population_episode(tmp_path):
    policies = {"agent-1": _agent_one, "agent-2": _agent_two}
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
    assert summary["dms"] == 1 and summary["replies"] == 1
    assert "agent-1: asked agent-2 for 6*7" in summary["board"]
    edges = swarm.execution_snapshot()["semantic_edges"]["edges"]
    assert {edge["type"] for edge in edges} >= {"agent_message", "agent_reply"}
