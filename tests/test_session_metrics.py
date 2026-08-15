"""Sub-agent metric aggregation and recoverable metric snapshots."""

import json

from rlm.session import Session
from rlm.types import ProgrammaticToolCallStats, RLMMetrics


def _write_calls(session_dir, n_python, n_bash=0):
    lines = ['{"tool": "browsecomp_plus_search", "source": "python"}\n'] * n_python
    lines += ['{"tool": "edit", "source": "bash"}\n'] * n_bash
    lines += ["{malformed, exited mid-line\n"]  # from_log must tolerate this
    (session_dir / "programmatic_tool_calls.jsonl").write_text("".join(lines))


def test_aggregate_counts_unfinalized_children(tmp_path):
    s = Session(tmp_path / "session")
    child = s.dir / "sub-aaaa"
    child.mkdir()
    _write_calls(child, n_python=3, n_bash=1)  # no meta.json: never finalized
    grand = child / "sub-bbbb"  # nested descendant, also unfinalized
    grand.mkdir()
    _write_calls(grand, n_python=1)

    agg = s.aggregate_child_metrics()
    assert agg.num_sessions == 2
    assert agg.tool_call_stats.python_total == 4
    assert agg.tool_call_stats.bash_total == 1


def test_aggregate_counts_child_with_no_calls(tmp_path):
    s = Session(tmp_path / "session")
    (s.dir / "sub-empty").mkdir()  # spawned, never ran a call, no log file

    agg = s.aggregate_child_metrics()
    assert agg.num_sessions == 1
    assert agg.tool_call_stats.python_total == 0


def test_ptc_log_skips_non_object_json(tmp_path):
    log = tmp_path / "programmatic_tool_calls.jsonl"
    log.write_text('null\n[]\n"text"\n{"tool": "search", "source": "python"}\n')

    stats = ProgrammaticToolCallStats.from_log(log)

    assert stats.python_total == 1
    assert stats.by_tool_python == {"search": 1}


def test_checkpoint_metrics_persists_midrun(tmp_path):
    s = Session(tmp_path / "session")
    metrics = RLMMetrics()
    metrics._sub_rlm_enabled = True
    child = s.dir / "sub-cccc"
    child.mkdir()
    _write_calls(child, n_python=2)

    s.checkpoint_metrics(metrics, status="incomplete")

    meta = json.loads((s.dir / "meta.json").read_text())
    assert meta["status"] == "incomplete"
    assert meta["metrics"]["sub_rlm_num_calls"] == 1
    assert meta["metrics"]["sub_rlm_num_ptc_calls_python"] == 2
    assert meta["programmatic_tool_call_stats"]["python_total"] == 2

    # finalize afterwards still wins with the authoritative version
    s.finalize("answer", turns=3, metrics=metrics)
    meta = json.loads((s.dir / "meta.json").read_text())
    assert meta["status"] == "done"
    assert meta["metrics"]["sub_rlm_num_ptc_calls_python"] == 2


def test_sub_rlm_keys_gated_when_disabled(tmp_path):
    metrics = RLMMetrics()  # _sub_rlm_enabled defaults False (max_depth == 0)
    keys = metrics.to_dict().keys()
    assert not any(k.startswith("sub_rlm_") for k in keys)
    metrics._sub_rlm_enabled = True
    assert "sub_rlm_num_calls" in metrics.to_dict()
