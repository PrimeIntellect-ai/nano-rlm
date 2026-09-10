from __future__ import annotations

import json

from rlm.history import history
from rlm.session import Session


def test_window_addresses_survive_compaction_and_rollback(tmp_path):
    session = Session(tmp_path)
    system = {"role": "system", "content": "system"}
    user = {"role": "user", "content": "original task"}
    answer = {"role": "assistant", "content": "work"}
    summary = {"role": "user", "content": "summary"}
    followup = {"role": "user", "content": "failed followup"}
    try:
        session.log({"type": "system", "message": system}, in_context=True)
        session.log({"type": "user", "message": user}, in_context=True)
        session.log_assistant(0, None, answer)
        first = history(tmp_path)
        session.replace_context([system, user, summary], reason="compaction")
        checkpoint = list(session.context_indices)
        session.log({"type": "user", "message": followup}, in_context=True)
        session.replace_context(
            [system, user, summary], reason="rollback", indices=checkpoint
        )
        current = history(tmp_path)
        assert first.windows[0].messages == [system, user, answer]
        assert current.windows[0].messages == first.windows[0].messages
        assert current.windows[1].messages == [system, user, summary, followup]
        assert current.windows[2].messages == [system, user, summary]
        assert current.windows[2].reason == "rollback"
        assert current.windows[0].message_indices == [0, 1, 2]
        assert current.windows[1].message_indices == [0, 1, 3, 4]
        assert current.windows[2].message_indices == [0, 1, 3]
        assert current.user_messages() == [user, followup]
    finally:
        session.close()


def test_full_tool_output_has_a_separate_index_from_context_text(tmp_path):
    session = Session(tmp_path)
    try:
        context_message = session.log_tool_result(
            0, "ipython", "full result", 0, call_id="tool-1", context_content="short"
        )
        snapshot = history(tmp_path)
        assert snapshot.messages == [
            {"role": "tool", "tool_call_id": "tool-1", "content": "full result"},
            context_message,
        ]
        assert snapshot.windows[0].message_indices == [1]
        assert snapshot.windows[0].messages == [context_message]
        derived = next(e for e in snapshot.events if e["type"] == "context_message")
        assert derived["source_message_index"] == 0
    finally:
        session.close()


def test_live_reader_ignores_unfinished_record_and_sees_it_on_next_read(tmp_path):
    path = tmp_path / "messages.jsonl"
    data = json.dumps(
        {
            "type": "user",
            "message_index": 0,
            "message": {"role": "user", "content": "新"},
        },
        ensure_ascii=False,
    ).encode()
    boundary = data.index("新".encode()) + 1
    path.write_bytes(data[:boundary])
    assert history(tmp_path).messages == []
    with path.open("ab") as stream:
        stream.write(data[boundary:] + b"\n")
    assert history(tmp_path).user_messages() == [{"role": "user", "content": "新"}]


def test_parent_can_discover_and_read_live_child_history(tmp_path):
    parent = Session(tmp_path / "parent")
    child = Session(Session.child_dir(parent.dir))
    try:
        parent.log_sub_spawn(child.dir.name, "(brokered rlm())", prompt="research")
        child.log(
            {"type": "user", "message": {"role": "user", "content": "research"}},
            in_context=True,
        )
        root = history(parent.dir)
        assert root.children == [child.dir]
        assert root.events[0]["prompt"] == "research"
        earlier = history(root.children[0])
        child.log_assistant(0, None, {"role": "assistant", "content": "working"})
        current = history(root.children[0])
        assert len(earlier.windows[0].messages) == 1
        assert current.windows[0].messages[-1] == {
            "role": "assistant",
            "content": "working",
        }
    finally:
        child.close()
        parent.close()
