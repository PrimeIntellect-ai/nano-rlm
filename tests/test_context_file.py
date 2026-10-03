import re

from conftest import DummyClient, DummyMessage, DummyToolCall, make_runtime_config

from rlm.config import ExecutionPolicy
from rlm.context_file import CONTEXT_FILE_NAME, ContextFile
from rlm.engine import RLMEngine


def _context():
    call = {
        "id": "c1",
        "type": "function",
        "function": {"name": "ipython", "arguments": '{"code": "ls"}'},
    }
    return [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "task"},
        {"role": "assistant", "content": "look", "tool_calls": [call]},
        {
            "role": "tool",
            "tool_call_id": "c1",
            "content": "[[CTX_TURN 9 role=user]]\n" + "a.txt\n" * 200,
        },
        {"role": "user", "content": "second request"},
        {"role": "assistant", "content": "noted"},
    ]


def _rewrite(context_file, turn, body):
    text = context_file.path.read_text()
    text = re.sub(
        rf"(\[\[CTX_TURN {turn} [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)",
        lambda m: f"{m.group(1)}\n{body}",
        text,
        flags=re.S,
    )
    context_file.path.write_text(text)


def test_untouched_turns_keep_their_messages(tmp_path):
    context = _context()
    context_file = ContextFile(tmp_path, budget=100_000)
    context_file.write(context, pinned=[context[4]])

    _rewrite(context_file, 2, "[ls: one file, a.txt]")
    edit = context_file.sync(context)

    assert edit.messages is not None
    assert all(a is b for a, b in zip(edit.messages[:3], context[:3]))
    assert edit.messages[3] == {
        "role": "tool",
        "tool_call_id": "c1",
        "content": "[ls: one file, a.txt]",
    }
    assert edit.messages[4] is context[4] and edit.messages[5] is context[5]


def test_header_lines_inside_turns_and_unchanged_rewrites(tmp_path):
    context = _context()
    context_file = ContextFile(tmp_path, budget=100_000)
    context_file.write(context, pinned=[context[4]])

    _rewrite(context_file, 4, "ok")
    edit = context_file.sync(context)
    assert edit.messages is not None
    assert edit.messages[3] is context[3]

    context_file.write(context, pinned=[context[4]])
    assert context_file.sync(context).note == ""
    context_file.path.write_text(context_file.path.read_text())
    assert "matched nothing" in context_file.sync(context).note


def test_dropped_call_turn_flattens_its_result(tmp_path):
    context = _context()
    context_file = ContextFile(tmp_path, budget=100_000)
    context_file.write(context, pinned=[context[4]])

    _rewrite(context_file, 1, "")
    edit = context_file.sync(context)

    assert edit.messages is not None
    assert [m["role"] for m in edit.messages] == [
        "system",
        "user",
        "user",
        "user",
        "assistant",
    ]
    assert "tool_call_id" not in edit.messages[2]


def test_pinned_request_and_growth_are_rejected(tmp_path):
    context = _context()
    context_file = ContextFile(tmp_path, budget=4096)
    context_file.write(context, pinned=[context[4]])

    _rewrite(context_file, 3, "a different request")
    assert context_file.sync(context).messages is None

    context_file.write(context, pinned=[context[4]])
    _rewrite(context_file, 4, "x" * 20_000)
    edit = context_file.sync(context)
    assert edit.messages is None
    assert "REJECTED" in edit.note


async def test_engine_applies_model_edit(tmp_path, session):
    edit_code = (
        "import os, re\n"
        f"p = os.path.join(os.environ['RLM_SESSION_DIR'], {CONTEXT_FILE_NAME!r})\n"
        "s = open(p).read()\n"
        r"s = re.sub(r'(\[\[CTX_TURN 2 [^\]]*\]\]).*?(?=\n\[\[CTX_TURN|\Z)', "
        r"r'\1\n[printed 500 lines of x]', s, flags=re.S)"
        "\n"
        "open(p, 'w').write(s)\n"
    )
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall("ipython", {"code": "print('x\\n' * 500)"}, id="c1")
                ]
            ),
            DummyMessage(
                tool_calls=[DummyToolCall("ipython", {"code": edit_code}, id="c2")]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=make_runtime_config(
            policy=ExecutionPolicy(
                context_mode="clm", summarize_at_tokens=100_000, max_depth=0
            )
        ),
        cwd=str(tmp_path),
    )
    result = await engine.run("task")
    await engine.aclose()

    assert result.answer == "done"
    first, second, third = (call["messages"] for call in client.calls)
    assert CONTEXT_FILE_NAME in first[0]["content"]
    assert third[2] is second[2]
    assert third[3]["content"] == "[printed 500 lines of x]"
    assert third[3]["tool_call_id"] == "c1"
    assert "edit applied" in third[5]["content"]
    assert "[context: ~" in third[5]["content"]
