from conftest import DummyClient, DummyMessage, DummyToolCall, make_runtime_config

from rlm.engine import EMPTY_REPLY_NUDGE, MAX_EMPTY_REPLY_NUDGES, RLMEngine


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
