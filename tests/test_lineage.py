"""Semantic request edges for recursive sessions and compaction."""

from __future__ import annotations

import asyncio

from rlm.client import model_call_headers
from rlm.semantic import SemanticEdgeTracker


def _finish(lineage: SemanticEdgeTracker, session_id: str) -> str:
    request_id = lineage.start_request(session_id)
    lineage.finish_request(request_id)
    return request_id


def test_subagent_call_and_return_are_request_edges():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)
    parent_request = _finish(lineage, "root")
    lineage.register_session(
        "child",
        parent_session_id="root",
        spawned_by_request_id=parent_request,
    )
    child_request = _finish(lineage, "child")
    lineage.finish_subagent("child")
    resumed_request = _finish(lineage, "root")

    assert model_call_headers(parent_request) == {
        "Idempotency-Key": parent_request,
        "X-ACP-Model-Request-ID": parent_request,
    }
    assert lineage.snapshot() == {
        "edges": [
            {
                "source_request_id": parent_request,
                "target_request_id": child_request,
                "type": "subagent_call",
            },
            {
                "source_request_id": child_request,
                "target_request_id": resumed_request,
                "type": "subagent_return",
            },
        ]
    }


def test_concurrent_subagent_returns_share_the_consuming_request():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)
    parent_request = _finish(lineage, "root")
    child_requests = []
    for child_id in ("child-1", "child-2"):
        lineage.register_session(
            child_id,
            parent_session_id="root",
            spawned_by_request_id=parent_request,
        )
        child_requests.append(_finish(lineage, child_id))
        lineage.finish_subagent(child_id)
    resumed_request = _finish(lineage, "root")

    returns = [
        edge
        for edge in lineage.snapshot()["edges"]
        if edge["type"] == "subagent_return"
    ]
    assert {edge["source_request_id"] for edge in returns} == set(child_requests)
    assert {edge["target_request_id"] for edge in returns} == {resumed_request}


def test_completed_compaction_links_summary_to_resumed_request():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)
    _finish(lineage, "root")
    compaction = lineage.begin_compaction("root")
    summary_request = lineage.start_request(
        "root", compaction_id=compaction.compaction_id
    )
    lineage.finish_request(summary_request)
    lineage.finish_compaction(compaction.compaction_id, "completed")
    resumed_request = _finish(lineage, "root")

    assert lineage.snapshot()["edges"] == [
        {
            "source_request_id": summary_request,
            "target_request_id": resumed_request,
            "type": "compaction",
        }
    ]


def test_failed_compaction_publishes_no_transition():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)
    compaction = lineage.begin_compaction("root")
    summary_request = lineage.start_request(
        "root", compaction_id=compaction.compaction_id
    )
    lineage.fail_request(summary_request)
    lineage.finish_compaction(compaction.compaction_id, "failed")
    _finish(lineage, "root")

    assert lineage.snapshot() == {"edges": []}


def test_prompt_rollback_discards_unconsumed_compaction_transition():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)
    before = lineage.checkpoint("root")
    compaction = lineage.begin_compaction("root")
    summary_request = lineage.start_request(
        "root", compaction_id=compaction.compaction_id
    )
    lineage.finish_request(summary_request)
    lineage.finish_compaction(compaction.compaction_id, "completed")
    failed_target = lineage.start_request("root")
    lineage.fail_request(failed_target)
    lineage.restore("root", before)
    _finish(lineage, "root")

    assert lineage.snapshot() == {"edges": []}


async def test_concurrent_requests_receive_unique_stable_ids():
    lineage = SemanticEdgeTracker()
    lineage.register_session("root", parent_session_id=None)

    async def start_request():
        await asyncio.sleep(0)
        request_id = lineage.start_request("root")
        lineage.finish_request(request_id)
        return model_call_headers(request_id)

    headers = await asyncio.gather(*(start_request() for _ in range(64)))
    request_ids = [item["X-ACP-Model-Request-ID"] for item in headers]

    assert len(set(request_ids)) == 64
    assert all(
        item["Idempotency-Key"] == item["X-ACP-Model-Request-ID"] for item in headers
    )
    assert lineage.snapshot() == {"edges": []}
