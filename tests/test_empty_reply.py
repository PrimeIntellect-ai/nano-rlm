from conftest import DummyClient, DummyMessage, DummyToolCall, make_runtime_config

from rlm.engine import (
    EMPTY_REPLY_NUDGE,
    MAX_EMPTY_REPLY_NUDGES,
    PLAN_REPLY_NUDGE,
    RLMEngine,
    _looks_like_plan,
)


def _records(session):
    import json

    return [
        json.loads(line)
        for line in (session.dir / "messages.jsonl").read_text().splitlines()
    ]


async def test_empty_reply_is_nudged_then_recovers(session):
    client = DummyClient(
        [
            DummyMessage(content=""),
            DummyMessage(tool_calls=[DummyToolCall("add", {"a": 1, "b": 2})]),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    result = await engine.run("add")
    assert result.answer == "done"
    nudges = [r for r in _records(session) if r.get("type") == "empty_reply_nudge"]
    assert len(nudges) == 1 and nudges[0]["message"]["content"] == EMPTY_REPLY_NUDGE
    # the nudge reached the model as a user message on the next call
    assert any(
        m.get("content") == EMPTY_REPLY_NUDGE for m in client.calls[1]["messages"]
    )


async def test_empty_reply_gives_up_after_budget_and_resets_per_prompt(session):
    empties = [DummyMessage(content="") for _ in range(MAX_EMPTY_REPLY_NUDGES + 1)]
    client = DummyClient(
        empties
        + [
            DummyMessage(content=""),
            DummyMessage(content="second"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    first = await engine.prompt("first")
    assert first.answer == ""
    assert len(client.calls) == MAX_EMPTY_REPLY_NUDGES + 1
    second = await engine.prompt("second")
    assert (
        second.answer == "second"
    )  # budget reset: the new turn's empty reply was nudged again
    nudges = [r for r in _records(session) if r.get("type") == "empty_reply_nudge"]
    assert len(nudges) == MAX_EMPTY_REPLY_NUDGES + 1


def test_looks_like_plan_heuristic():
    assert _looks_like_plan("Now let me apply all the changes:")
    assert _looks_like_plan(
        "Let me look at the unit tests related to the variable manager."
    )
    assert not _looks_like_plan(
        "The fix adds a `job_path` argument; all 12 tests pass."
    )
    assert not _looks_like_plan(
        "Let me summarize: " + "the change touches three files. " * 20
    )
    assert not _looks_like_plan("")


async def test_plan_like_reply_is_nudged_once(session):
    client = DummyClient(
        [
            DummyMessage(content="Now let me apply all the changes:"),
            DummyMessage(tool_calls=[DummyToolCall("add", {"a": 1, "b": 2})]),
            DummyMessage(content="Let me check the exports:"),  # second plan-like stop
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    result = await engine.run("add")
    assert result.answer == "done"
    nudges = [r for r in _records(session) if r.get("type") == "plan_reply_nudge"]
    assert len(nudges) == 2 and nudges[0]["message"]["content"] == PLAN_REPLY_NUDGE
    assert any(
        m.get("content") == PLAN_REPLY_NUDGE for m in client.calls[1]["messages"]
    )


async def test_plan_like_reply_budget_is_one_per_stretch(session):
    client = DummyClient(
        [
            DummyMessage(content="Let me look at the tests:"),
            DummyMessage(content="Let me look at the tests:"),  # accepted: budget spent
        ]
    )
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore
    result = await engine.run("add")
    assert result.answer == "Let me look at the tests:"
    assert len(client.calls) == 2
