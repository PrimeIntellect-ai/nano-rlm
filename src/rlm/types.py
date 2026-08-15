"""Core data types."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class TokenUsage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    @property
    def total(self) -> int:
        return self.prompt_tokens + self.completion_tokens


@dataclass(frozen=True)
class IpythonExecuted:
    input_chars: int
    input_loc: int


@dataclass(frozen=True)
class CompactionApplied:
    """Emitted when the engine auto-compacts context after a summary turn."""

    dropped_chars: int
    summary_chars: int
    turns_since_last_compaction: int


BuiltinMetricEvent = IpythonExecuted | CompactionApplied


@dataclass
class ProgrammaticToolCallStats:
    """Programmatic tool-call invocation attempts from the IPython REPL."""

    python_total: int = 0
    bash_total: int = 0
    by_tool_python: dict[str, int] = field(default_factory=dict)
    by_tool_bash: dict[str, int] = field(default_factory=dict)

    @classmethod
    def from_log(cls, log_path: Path) -> ProgrammaticToolCallStats:
        """Count programmatic tool calls from a session-local JSONL log.

        Untrusted input (written by child processes that may exit mid-line),
        so malformed entries are skipped.
        """
        stats = cls()
        try:
            f = open(log_path)
        except FileNotFoundError:
            return stats
        with f:
            for line in f:
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if not isinstance(entry, dict):
                    continue
                tool = entry.get("tool")
                source = entry.get("source")
                if not isinstance(tool, str) or source not in {"python", "bash"}:
                    continue

                if source == "python":
                    stats.python_total += 1
                    stats.by_tool_python[tool] = stats.by_tool_python.get(tool, 0) + 1
                else:
                    stats.bash_total += 1
                    stats.by_tool_bash[tool] = stats.by_tool_bash.get(tool, 0) + 1

        return stats

    @classmethod
    def from_meta(cls, meta: dict) -> ProgrammaticToolCallStats:
        """Load tool-call stats previously persisted via ``to_dict``."""
        raw = meta.get("programmatic_tool_call_stats", {})
        return cls(
            python_total=int(raw.get("python_total", 0)),
            bash_total=int(raw.get("bash_total", 0)),
            by_tool_python=dict(raw.get("by_tool_python", {})),
            by_tool_bash=dict(raw.get("by_tool_bash", {})),
        )

    def merge(self, other: ProgrammaticToolCallStats) -> ProgrammaticToolCallStats:
        """Return a merged copy of this stats object and *other*."""
        merged = ProgrammaticToolCallStats(
            python_total=self.python_total + other.python_total,
            bash_total=self.bash_total + other.bash_total,
            by_tool_python=self.by_tool_python.copy(),
            by_tool_bash=self.by_tool_bash.copy(),
        )
        for tool_name, count in other.by_tool_python.items():
            merged.by_tool_python[tool_name] = (
                merged.by_tool_python.get(tool_name, 0) + count
            )
        for tool_name, count in other.by_tool_bash.items():
            merged.by_tool_bash[tool_name] = (
                merged.by_tool_bash.get(tool_name, 0) + count
            )
        return merged

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ChildSessionAggregate:
    """Aggregated stats across all recursive descendants of a session."""

    tool_call_stats: ProgrammaticToolCallStats = field(
        default_factory=ProgrammaticToolCallStats
    )
    num_sessions: int = 0
    """Recursive count of descendant sub-agent sessions (spawned via rlm())."""

    def absorb(self, tool_stats: ProgrammaticToolCallStats) -> None:
        """Combine a child session's stats into this aggregate."""
        self.tool_call_stats = self.tool_call_stats.merge(tool_stats)


@dataclass
class RLMMetrics:
    """Metrics tracked during an rlm session.

    Only metrics with no native verifiers-v1 equivalent live here. Per-branch,
    terminal, and sub-agent token accounting is deliberately absent: verifiers
    reconstructs branches (compaction + subagents) from the message graph and
    reports their token counts natively, so tracking it here would double-count.
    What remains is rlm-internal behaviour verifiers can't see: compaction
    volume, ipython input size, and programmatic tool calls made from inside the
    REPL (which never hit the API proxy).
    """

    # Compaction metrics (auto-summarization at a token threshold)
    num_compactions: int = 0
    has_compacted: int = 0  # 1 if num_compactions > 0, else 0; aggregates as the per-batch fraction of rollouts that compacted
    turns_since_last_compaction: int = 0
    turns_between_compactions_mean: float = 0.0
    compaction_chars_dropped_mean: float = 0.0
    compaction_summary_chars_mean: float = 0.0

    # IPython input size metrics
    ipython_input_chars_mean: float = 0.0
    ipython_input_loc_mean: float = 0.0

    # Skill-CLI invocations from inside the ipython REPL (invisible to verifiers:
    # they run within a single tool call and never reach the API proxy).
    num_ptc_calls_python: int = 0
    num_ptc_calls_bash: int = 0
    sub_rlm_num_calls: int = 0
    sub_rlm_num_ptc_calls_python: int = 0
    sub_rlm_num_ptc_calls_bash: int = 0

    stop_reason: str = ""

    # Internal counters for derived metrics
    _ipython_call_count: int = field(default=0, repr=False)
    _ipython_input_chars_total: int = field(default=0, repr=False)
    _ipython_input_loc_total: int = field(default=0, repr=False)
    _turns_between_compactions_total: int = field(default=0, repr=False)
    _compaction_chars_dropped_total: int = field(default=0, repr=False)
    _compaction_summary_chars_total: int = field(default=0, repr=False)
    _sub_rlm_enabled: bool = field(default=False, repr=False)

    def apply_programmatic_tool_call_stats(
        self,
        direct: ProgrammaticToolCallStats,
        child: ProgrammaticToolCallStats,
        num_child_sessions: int = 0,
    ) -> None:
        self.num_ptc_calls_python = direct.python_total
        self.num_ptc_calls_bash = direct.bash_total
        self.sub_rlm_num_calls = num_child_sessions
        self.sub_rlm_num_ptc_calls_python = child.python_total
        self.sub_rlm_num_ptc_calls_bash = child.bash_total

    def record(self, event: BuiltinMetricEvent) -> None:
        if isinstance(event, IpythonExecuted):
            self._ipython_call_count += 1
            self._ipython_input_chars_total += event.input_chars
            self._ipython_input_loc_total += event.input_loc
        elif isinstance(event, CompactionApplied):
            self.num_compactions += 1
            self._turns_between_compactions_total += event.turns_since_last_compaction
            self._compaction_chars_dropped_total += event.dropped_chars
            self._compaction_summary_chars_total += event.summary_chars
        else:
            raise TypeError(f"Unsupported builtin metric event: {type(event)!r}")

        self._refresh_derived_metrics()

    def _refresh_derived_metrics(self) -> None:
        if self._ipython_call_count:
            self.ipython_input_chars_mean = (
                self._ipython_input_chars_total / self._ipython_call_count
            )
            self.ipython_input_loc_mean = (
                self._ipython_input_loc_total / self._ipython_call_count
            )
        self.has_compacted = 1 if self.num_compactions > 0 else 0
        if self.num_compactions:
            self.turns_between_compactions_mean = (
                self._turns_between_compactions_total / self.num_compactions
            )
            self.compaction_chars_dropped_mean = (
                self._compaction_chars_dropped_total / self.num_compactions
            )
            self.compaction_summary_chars_mean = (
                self._compaction_summary_chars_total / self.num_compactions
            )

    def to_dict(self) -> dict[str, Any]:
        self._refresh_derived_metrics()
        sub_rlm_enabled = self._sub_rlm_enabled
        return {
            key: value
            for key, value in asdict(self).items()
            if not key.startswith("_")
            and (sub_rlm_enabled or not key.startswith("sub_rlm_"))
        }


@dataclass
class RLMResult:
    answer: str
    session_dir: Path | None = None
    usage: TokenUsage = field(default_factory=TokenUsage)
    turns: int = 0
