"""Shared builtin tool types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from rlm.types import BuiltinMetricEvent, RLMMetrics, TokenUsage


@dataclass
class ToolOutcome:
    """Result of executing a builtin tool."""

    content: str
    metric_events: list[BuiltinMetricEvent] = field(default_factory=list)


@dataclass
class ToolContext:
    """Runtime context passed to builtin tool handlers."""

    messages: list[dict[str, Any]]
    metrics: RLMMetrics
    total_usage: TokenUsage
    last_prompt_tokens: int
    exec_timeout: int
    max_tool_output_chars: int | None = None
    allow_git: bool = False
    repl: Any | None = None
    state: dict[str, Any] = field(default_factory=dict)
    cwd: str = ""


class BuiltinTool(Protocol):
    """Interface for builtin OpenAI tool handlers."""

    name: str

    def schema(self) -> dict[str, Any]:
        """Return the OpenAI schema for this tool."""

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        """Execute the tool and return a string tool result plus side effects."""
