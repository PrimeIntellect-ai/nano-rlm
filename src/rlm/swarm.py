"""One population episode behind an ACP session.

The swarm admits the founders, waits without any model call of its own until no agent
can act again (or a budget or the wall clock runs out), stops every agent, and reports
the episode answer with population-wide accounting. All agents share one supervisor,
so their model calls share the session's provider endpoint and semantic edges.

An agent's engine only raises for infrastructure (a model call that failed after its
retries, a kernel that could not be released): tool errors stay in the conversation. So
one failed agent ends the episode as an error instead of a scored outcome.
"""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any

from rlm.client import ModelTransportError
from rlm.config import RuntimeConfig
from rlm.mcp import MCPServer
from rlm.session import Session
from rlm.supervisor import SessionTreeSupervisor
from rlm.types import ProgrammaticToolCallStats, RLMResult, TokenUsage

BOARD_SUMMARY_BYTES = 65_536


async def _settle(awaitable) -> None:
    """Finish cleanup even if the caller is cancelled meanwhile, then re-raise that."""
    task = asyncio.ensure_future(awaitable)
    cancelled = False
    while True:
        try:
            await asyncio.shield(task)
            break
        except asyncio.CancelledError:
            if task.done():
                raise
            cancelled = True
    if cancelled:
        raise asyncio.CancelledError


class Swarm:
    def __init__(
        self,
        *,
        cwd: str,
        session: Session,
        runtime_config: RuntimeConfig,
        mcp_servers: dict[str, MCPServer] | None = None,
        invocation_id: str | None = None,
    ) -> None:
        if runtime_config.policy.population is None:
            raise ValueError("a swarm needs a population policy")
        self.runtime_config = runtime_config
        self.policy = runtime_config.policy.population
        self.session = session
        self.supervisor = SessionTreeSupervisor(
            root_session=session,
            runtime_config=runtime_config,
            cwd=cwd,
            mcp_servers=mcp_servers,
            root_invocation_id=invocation_id,
        )
        self._stop_reason = ""
        self._summary: dict[str, Any] | None = None
        self._started = False

    @property
    def stop_reason(self) -> str:
        return self._stop_reason

    async def steer(self, text: str, message_id: str | None = None) -> dict:
        raise ValueError("a population episode does not accept steering")

    async def prompt(self, prompt: str) -> RLMResult:
        """Run the whole episode on this task and return its episode answer."""
        if self._started:
            raise RuntimeError("a population episode runs a single task")
        self._started = True
        started = time.monotonic()
        await self.supervisor.start()
        try:
            self.supervisor.admit_founders(prompt)
            self._stop_reason = await self._run(started)
        except asyncio.CancelledError:
            self._stop_reason = "cancelled"
            raise
        finally:
            self.supervisor.close_population()
            summary = self.supervisor.population_snapshot()
            summary["stop_reason"] = self._stop_reason
            summary["seconds"] = round(time.monotonic() - started, 3)
            summary["answer"] = self.supervisor.episode_answer
            board = self.supervisor.board_text.encode()
            summary["board"] = board[:BOARD_SUMMARY_BYTES].decode(errors="ignore")
            self._summary = summary
            await _settle(self.supervisor.stop_population())
            (self.session.dir / "population_summary.json").write_text(
                json.dumps(summary, indent=2)
            )
        if failures := self.supervisor.population_failures():
            transport = [e for e in failures if isinstance(e, ModelTransportError)]
            raise (transport or failures)[0]
        usage = self._usage()
        return RLMResult(
            answer=self.supervisor.episode_answer["text"],
            session_dir=self.session.dir,
            usage=usage,
            turns=self.supervisor.total_turns,
        )

    async def _run(self, started: float) -> str:
        deadline = (
            None
            if self.policy.max_seconds is None
            else started + self.policy.max_seconds
        )
        activity = self.supervisor.activity
        while True:
            activity.clear()
            if self.supervisor.population_failures():
                return "agent_failed"
            if self.supervisor.population_quiescent():
                return self._quiescent_reason()
            now = time.monotonic()
            if deadline is not None and now >= deadline:
                return "max_seconds"
            recheck = self.supervisor.population_recheck_at()
            wake = min((t for t in (deadline, recheck) if t is not None), default=None)
            try:
                await asyncio.wait_for(
                    activity.wait(), None if wake is None else wake - now
                )
            except asyncio.TimeoutError:
                pass

    def _quiescent_reason(self) -> str:
        policy = self.runtime_config.policy
        if (
            policy.max_total_tokens is not None
            and self.supervisor.total_tokens >= policy.max_total_tokens
        ):
            return "max_total_tokens"
        if (
            policy.max_total_turns is not None
            and self.supervisor.total_turns >= policy.max_total_turns
        ):
            return "max_total_turns"
        if self.supervisor.all_agents_turn_capped():
            return "max_agent_turns"
        counts = self.supervisor.population_counts()
        if counts["ended"] == counts["admitted"]:
            return "extinct"
        return "quiescent"

    def _usage(self) -> TokenUsage:
        agents = (self._summary or self.supervisor.population_snapshot())["agents"]
        return TokenUsage(
            prompt_tokens=sum(agent["prompt_tokens"] for agent in agents),
            completion_tokens=sum(agent["completion_tokens"] for agent in agents),
        )

    def execution_snapshot(self) -> dict:
        """Credential-free episode snapshot in the engine's schema, plus the population."""
        summary = self._summary or self.supervisor.population_snapshot()
        usage = self._usage()
        policy = self.runtime_config.policy
        metrics: dict[str, int | float] = {
            f"population_{key}": value
            for key, value in summary.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        metrics.update(
            population_founders=self.policy.founders,
            population_failed=sum(a["status"] == "failed" for a in summary["agents"]),
            population_answered=int(summary["answer_revision"] > 0),
            population_turns=self.supervisor.total_turns,
            population_new_tokens=self.supervisor.total_tokens,
            population_prompt_tokens=usage.prompt_tokens,
            population_completion_tokens=usage.completion_tokens,
        )
        stats = ProgrammaticToolCallStats()
        for agent in summary["agents"]:
            direct, _ = self.supervisor.programmatic_tool_call_stats(agent["id"])
            stats = stats.merge(direct)
        return {
            "model": self.runtime_config.model,
            "turns": self.supervisor.total_turns,
            "usage": {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total,
            },
            "metrics": metrics,
            "programmatic_tool_call_stats": stats.to_dict(),
            "supervisor": {
                "subagent_calls": self.supervisor.total_calls,
                "active_subagent_calls": self.supervisor.active_calls,
            },
            "limits": {
                "max_depth": policy.max_depth,
                "max_concurrent_subagents": policy.max_concurrent_subagents,
                "max_subagent_calls": policy.max_subagent_calls,
                "max_tokens": policy.max_tokens,
                "compaction": policy.compaction,
                "summarize_at_tokens": policy.summarize_at_tokens,
                "max_compactions": policy.max_compactions,
                "max_compaction_attempts": policy.max_compaction_attempts,
                "allow_git": policy.allow_git,
            },
            "semantic_edges": self.supervisor.semantic_edges.snapshot(),
            "population": summary,
        }

    async def aclose(self) -> None:
        try:
            await self.supervisor.aclose()
        finally:
            self.session.close()
