"""Replay exercises real kernels and child engines without provider calls."""

import json

import pytest
from openai.types.chat import ChatCompletion

from conftest import make_runtime_config
from rlm.config import ExecutionPolicy
from rlm.engine import RLMEngine
from rlm.replay import ExecutionTape, ReplayDivergence
from rlm.session import Session


def response(content=None, code=None):
    message = {"role": "assistant", "content": content}
    if code is not None:
        message["tool_calls"] = [
            {
                "id": "cell",
                "type": "function",
                "function": {
                    "name": "ipython",
                    "arguments": json.dumps({"code": code}),
                },
            }
        ]
    return ChatCompletion.model_validate(
        {
            "id": "completion",
            "object": "chat.completion",
            "created": 0,
            "model": "dummy-model",
            "choices": [
                {
                    "index": 0,
                    "message": message,
                    "finish_reason": "tool_calls" if code else "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )


class ScriptedClient:
    def __init__(self, scripts):
        self.scripts = scripts
        self.calls = 0
        self.chat = self
        self.completions = self

    async def create(self, **request):
        self.calls += 1
        users = "\n".join(
            m.get("content", "") for m in request["messages"] if m["role"] == "user"
        )
        key = next(key for key in self.scripts if key in users)
        index = sum(m["role"] == "assistant" for m in request["messages"])
        return self.scripts[key][index]

    async def close(self):
        pass


async def run_tree(tmp_path, tape, label, *, prompt="ROOT_TASK"):
    session = Session(tmp_path / label)
    config = make_runtime_config(policy=ExecutionPolicy(compaction=False))
    engine = RLMEngine(
        cwd=str(tmp_path), session=session, runtime_config=config, execution_tape=tape
    )
    try:
        return await engine.run(prompt)
    finally:
        session.close()


async def test_replay_executes_nested_children_in_fresh_kernels(tmp_path, monkeypatch):
    scripts = {
        "ROOT_TASK": [
            response(
                code="""
x = 40
first = await rlm.agent.spawn('CHILD_A', name='a')
second = await rlm.agent.spawn('CHILD_B', name='b')
await asyncio.gather(first.wait(), second.wait())
print((await first.result()).answer, (await second.result()).answer)
"""
            ),
            response(code="print(x + 2)"),
            response("finished"),
        ],
        "CHILD_A": [
            response(
                code="from pathlib import Path; Path('a.txt').write_text('A'); print('A')"
            ),
            response("A"),
        ],
        "CHILD_B": [
            response(
                code="from pathlib import Path; Path('b.txt').write_text('B'); print('B')"
            ),
            response("B"),
        ],
    }
    client = ScriptedClient(scripts)
    monkeypatch.setattr("rlm.engine.make_client", lambda _: client)
    path = tmp_path / "tape.jsonl"
    recording = ExecutionTape(path, mode="record")
    try:
        original = await run_tree(tmp_path, recording, "original")
        recording.finish()
    finally:
        recording.close()
    assert client.calls == 7
    for name in ("a.txt", "b.txt"):
        (tmp_path / name).unlink()

    async def forbidden(**kwargs):
        pytest.fail("replay called the provider")

    monkeypatch.setattr(client, "create", forbidden)
    replay = ExecutionTape(path, mode="replay")
    try:
        result = await run_tree(tmp_path, replay, "replayed")
        replay.finish()
    finally:
        replay.close()
    assert result.answer == original.answer == "finished"
    assert (tmp_path / "a.txt").read_text() == "A"
    assert (tmp_path / "b.txt").read_text() == "B"
    records = [
        json.loads(line)
        for line in (tmp_path / "replayed/messages.jsonl").read_text().splitlines()
    ]
    assert any(
        row.get("type") == "tool_result" and row["content"].strip() == "42"
        for row in records
    )


async def test_replay_rejects_changed_tool_output(tmp_path, monkeypatch):
    client = ScriptedClient(
        {
            "ROOT_TASK": [
                response(
                    code="from pathlib import Path; print(Path('input').read_text())"
                ),
                response("done"),
            ]
        }
    )
    monkeypatch.setattr("rlm.engine.make_client", lambda _: client)
    (tmp_path / "input").write_text("before")
    path = tmp_path / "tape.jsonl"
    tape = ExecutionTape(path, mode="record")
    try:
        await run_tree(tmp_path, tape, "original")
        tape.finish()
    finally:
        tape.close()
    (tmp_path / "input").write_text("after")
    tape = ExecutionTape(path, mode="replay")
    try:
        with pytest.raises(ReplayDivergence, match="tool.end diverged"):
            await run_tree(tmp_path, tape, "replayed")
        with pytest.raises(ReplayDivergence):
            tape.finish()
    finally:
        tape.close()


def test_incomplete_recording_cannot_replay(tmp_path):
    path = tmp_path / "tape.jsonl"
    tape = ExecutionTape(path, mode="record")
    tape.close()
    with pytest.raises(ValueError, match="incomplete"):
        ExecutionTape(path, mode="replay")


async def test_replay_preserves_kernel_through_compaction(tmp_path, monkeypatch):
    scripted = [
        response(code="x = 41; print(x)"),
        response("x is 41; finish the calculation"),
        response(code="print(x + 1)"),
        response("done"),
    ]
    client = ScriptedClient({})

    async def create(**request):
        client.calls += 1
        return scripted.pop(0)

    client.create = create
    monkeypatch.setattr("rlm.engine.make_client", lambda _: client)
    config = make_runtime_config(
        policy=ExecutionPolicy(
            max_depth=0,
            compaction=True,
            summarize_at_tokens=1,
            max_compactions=1,
        )
    )
    path = tmp_path / "tape.jsonl"
    for mode in ("record", "replay"):
        session = Session(tmp_path / mode)
        tape = ExecutionTape(path, mode=mode)
        engine = RLMEngine(
            cwd=str(tmp_path),
            session=session,
            runtime_config=config,
            execution_tape=tape,
        )
        try:
            result = await engine.run("calculate")
            tape.finish()
            assert result.answer == "done"
            assert engine._metrics.num_compactions == 1
        finally:
            await engine.aclose()
            session.close()
            tape.close()
    assert client.calls == 4


async def test_replay_rejects_changed_request_and_unused_events(tmp_path):
    path = tmp_path / "tape.jsonl"
    tape = ExecutionTape(path, mode="record")
    await tape.event("root", "model.start", {"messages": ["original"]})
    tape.finish()
    tape = ExecutionTape(path, mode="replay")
    try:
        with pytest.raises(ReplayDivergence, match="model.start diverged"):
            await tape.event("root", "model.start", {"messages": ["changed"]})
    finally:
        tape.close()
    tape = ExecutionTape(path, mode="replay")
    try:
        with pytest.raises(ReplayDivergence, match="unconsumed"):
            tape.finish()
    finally:
        tape.close()


async def test_child_failure_poisoning_wakes_other_actors(tmp_path):
    import asyncio

    path = tmp_path / "tape.jsonl"
    tape = ExecutionTape(path, mode="record")
    await tape.event("child", "tool.end", "ok")
    await tape.event("parent", "tool.end", "ok")
    tape.finish()
    tape = ExecutionTape(path, mode="replay")
    waiting = asyncio.create_task(tape.event("parent", "tool.end", "ok"))
    await asyncio.sleep(0)
    try:
        with pytest.raises(ReplayDivergence):
            await tape.event("child", "tool.end", "different")
        with pytest.raises(ReplayDivergence):
            await asyncio.wait_for(waiting, 1)
    finally:
        tape.close()
