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
        spawned_by_request_id=root_request.request_id,
    )
    child_request = lineage.start_request("child-1", kind="turn")

    root_headers = model_call_headers(root_request)
    child_headers = model_call_headers(child_request)
    snapshot = lineage.snapshot()

    assert root_headers == {
        "Idempotency-Key": root_request.request_id,
        "X-ACP-Lineage-Request-ID": root_request.request_id,
        "X-ACP-Lineage-Session-ID": "trace-1",
        "X-ACP-Lineage-Context-ID": root_context_id,
        "X-ACP-Lineage-Transition": "root",
        "X-ACP-Lineage-Depth": "0",
    }
    assert child_headers["X-ACP-Lineage-Parent-Session-ID"] == "trace-1"
    assert child_headers["X-ACP-Lineage-Context-ID"] == child_context_id
    assert child_headers["X-ACP-Lineage-Transition"] == "spawn"
    assert child_headers["X-ACP-Lineage-Depth"] == "1"
    assert snapshot["sessions"][1]["spawned_by_request_id"] == (root_request.request_id)


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

    summary_headers = model_call_headers(summary_request)
    resumed_headers = model_call_headers(resumed_request)
    snapshot = lineage.snapshot()

    assert summary_headers["X-ACP-Lineage-Context-ID"] == source_context_id
    assert summary_headers["X-ACP-Lineage-Compaction-ID"] == compaction.compaction_id
    assert "X-ACP-Lineage-Previous-Context-ID" not in summary_headers
    assert resumed_headers["X-ACP-Lineage-Context-ID"] == compaction.target_context_id
    assert resumed_headers["X-ACP-Lineage-Previous-Context-ID"] == source_context_id
    assert resumed_headers["X-ACP-Lineage-Transition"] == "compact"
    assert resumed_headers["X-ACP-Lineage-Compaction-ID"] == compaction.compaction_id
    assert snapshot["requests"][-1]["compaction_id"] == compaction.compaction_id
    assert snapshot["compactions"] == [
        {
            "compaction_id": compaction.compaction_id,
            "session_id": "trace-1",
            "source_context_id": source_context_id,
            "target_context_id": compaction.target_context_id,
            "status": "completed",
            "summary_request_id": summary_request.request_id,
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


def test_new_compaction_id_overrides_source_context_origin():
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

    ordinary_headers = model_call_headers(ordinary_request)
    summary_headers = model_call_headers(second_summary)
    assert ordinary_headers["X-ACP-Lineage-Compaction-ID"] == (
        first_compaction.compaction_id
    )
    assert (
        summary_headers["X-ACP-Lineage-Context-ID"]
        == first_compaction.target_context_id
    )
    assert summary_headers["X-ACP-Lineage-Previous-Context-ID"] == first_context_id
    assert summary_headers["X-ACP-Lineage-Transition"] == "compact"
    assert summary_headers["X-ACP-Lineage-Compaction-ID"] == (
        second_compaction.compaction_id
    )


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

    assert next_request.context_id == source_context_id
    assert snapshot["compactions"][0]["status"] == "failed"
    assert [context["context_id"] for context in snapshot["contexts"]] == [
        source_context_id,
        compaction.target_context_id,
    ]
