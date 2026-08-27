"""Stable lineage for recursive sessions and compacted contexts."""

from __future__ import annotations

import asyncio

from rlm.client import model_call_headers
from rlm.lineage import LineageTracker


def test_root_and_child_requests_have_exact_parentage():
    lineage = LineageTracker()
    root_context_id = lineage.register_session(
        "trace-1", parent_session_id=None, depth=0
    )
    root_request = lineage.start_request("trace-1", kind="turn")
    child_context_id = lineage.register_session(
        "child-1",
        parent_session_id="trace-1",
        depth=1,
        spawned_by_request_id=root_request,
    )
    child_request = lineage.start_request("child-1", kind="turn")

    root_headers = model_call_headers(root_request)
    child_headers = model_call_headers(child_request)
    snapshot = lineage.snapshot()

    assert root_headers == {
        "Idempotency-Key": root_request,
        "X-ACP-Lineage-Request-ID": root_request,
    }
    assert child_headers == {
        "Idempotency-Key": child_request,
        "X-ACP-Lineage-Request-ID": child_request,
    }
    assert snapshot["sessions"][1]["spawned_by_request_id"] == root_request
    assert snapshot["requests"] == [
        {
            "request_id": root_request,
            "session_id": "trace-1",
            "context_id": root_context_id,
            "kind": "turn",
        },
        {
            "request_id": child_request,
            "session_id": "child-1",
            "context_id": child_context_id,
            "kind": "turn",
        },
    ]


def test_terminal_session_status_cannot_be_overwritten():
    lineage = LineageTracker()
    lineage.register_session("trace-1", parent_session_id=None, depth=0)

    lineage.set_session_status("trace-1", "completed")
    lineage.set_session_status("trace-1", "cancelled")
    lineage.set_session_status("trace-1", "failed")

    assert lineage.snapshot()["sessions"][0]["status"] == "completed"


def test_completed_compaction_starts_linked_context_epoch():
    lineage = LineageTracker()
    source_context_id = lineage.register_session(
        "trace-1", parent_session_id=None, depth=0
    )
    compaction = lineage.begin_compaction("trace-1")
    summary_request = lineage.start_request(
        "trace-1", kind="compaction", compaction_id=compaction.compaction_id
    )
    lineage.finish_compaction(compaction.compaction_id, "completed")
    resumed_request = lineage.start_request("trace-1", kind="turn")

    snapshot = lineage.snapshot()
    requests = {request["request_id"]: request for request in snapshot["requests"]}
    contexts = {context["context_id"]: context for context in snapshot["contexts"]}

    assert requests[summary_request] == {
        "request_id": summary_request,
        "session_id": "trace-1",
        "context_id": source_context_id,
        "kind": "compaction",
        "compaction_id": compaction.compaction_id,
    }
    assert requests[resumed_request]["context_id"] == compaction.target_context_id
    assert requests[resumed_request]["compaction_id"] == compaction.compaction_id
    assert contexts[compaction.target_context_id]["previous_context_id"] == (
        source_context_id
    )
    assert contexts[compaction.target_context_id]["transition"] == "compact"
    assert snapshot["compactions"] == [
        {
            "compaction_id": compaction.compaction_id,
            "session_id": "trace-1",
            "source_context_id": source_context_id,
            "target_context_id": compaction.target_context_id,
            "status": "completed",
            "summary_request_id": summary_request,
        }
    ]


async def test_concurrent_requests_receive_unique_stable_ids():
    lineage = LineageTracker()
    lineage.register_session("trace-1", parent_session_id=None, depth=0)

    async def start_request():
        await asyncio.sleep(0)
        provenance = lineage.start_request("trace-1", kind="turn")
        return model_call_headers(provenance)

    headers = await asyncio.gather(*(start_request() for _ in range(64)))
    request_ids = [item["X-ACP-Lineage-Request-ID"] for item in headers]

    assert len(set(request_ids)) == 64
    assert all(
        item["Idempotency-Key"] == item["X-ACP-Lineage-Request-ID"] for item in headers
    )
    assert [request["request_id"] for request in lineage.snapshot()["requests"]] == (
        request_ids
    )


def test_new_compaction_request_keeps_context_origin_in_manifest():
    lineage = LineageTracker()
    first_context_id = lineage.register_session(
        "trace-1", parent_session_id=None, depth=0
    )
    first_compaction = lineage.begin_compaction("trace-1")
    lineage.start_request(
        "trace-1",
        kind="compaction",
        compaction_id=first_compaction.compaction_id,
    )
    lineage.finish_compaction(first_compaction.compaction_id, "completed")

    ordinary_request = lineage.start_request("trace-1", kind="turn")
    second_compaction = lineage.begin_compaction("trace-1")
    second_summary = lineage.start_request(
        "trace-1",
        kind="compaction",
        compaction_id=second_compaction.compaction_id,
    )

    snapshot = lineage.snapshot()
    requests = {request["request_id"]: request for request in snapshot["requests"]}
    contexts = {context["context_id"]: context for context in snapshot["contexts"]}

    assert requests[ordinary_request]["compaction_id"] == first_compaction.compaction_id
    assert requests[second_summary]["context_id"] == first_compaction.target_context_id
    assert requests[second_summary]["compaction_id"] == second_compaction.compaction_id
    compacted = contexts[first_compaction.target_context_id]
    assert compacted["previous_context_id"] == first_context_id
    assert compacted["transition"] == "compact"


def test_failed_compaction_does_not_activate_target_context():
    lineage = LineageTracker()
    source_context_id = lineage.register_session(
        "trace-1", parent_session_id=None, depth=0
    )
    compaction = lineage.begin_compaction("trace-1")
    lineage.start_request(
        "trace-1", kind="compaction", compaction_id=compaction.compaction_id
    )
    lineage.finish_compaction(compaction.compaction_id, "failed")
    next_request = lineage.start_request("trace-1", kind="turn")
    snapshot = lineage.snapshot()

    assert snapshot["requests"][-1]["request_id"] == next_request
    assert snapshot["requests"][-1]["context_id"] == source_context_id
    assert snapshot["compactions"] == [
        {
            "compaction_id": compaction.compaction_id,
            "session_id": "trace-1",
            "source_context_id": source_context_id,
            "status": "failed",
            "summary_request_id": snapshot["requests"][0]["request_id"],
        }
    ]
    assert [context["context_id"] for context in snapshot["contexts"]] == [
        source_context_id
    ]
