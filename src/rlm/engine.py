"""The agent loop."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
import uuid
from copy import deepcopy
from pathlib import Path
from typing import Any

from openai import APIStatusError, AsyncOpenAI

from rlm.client import (
    call_with_retries,
    extract_usage,
    make_client,
    model_call_headers,
)
from rlm.compaction import (
    CHECKPOINT_PROMPT,
    TOOL_OUTPUT_MAX_BYTES,
    CompactionFailed,
    REPL_NOTE,
    SUMMARY_FRAMING,
    compactable,
    discover_threshold,
    estimated_tokens,
    is_context_overflow,
    truncate_tool_output,
)
from rlm.config import RuntimeConfig
from rlm.semantic import SemanticEdgeTracker
from rlm.mcp import MCPServer, validate_mcp_servers
from rlm.prompt import build_system_prompt
from rlm.session import Session
from rlm.skills import enable_builtin_skills
from rlm.supervisor import SessionTreeSupervisor
from rlm.tools import (
    SKILLS_DIR,
    BuiltinTool,
    IPythonREPL,
    ToolContext,
    ToolOutcome,
    discover_skills,
    get_active_builtin_tools,
    get_builtin_tool,
    get_installed_skills,
)
from rlm.types import (
    CompactionApplied,
    ProgrammaticToolCallStats,
    RLMMetrics,
    RLMResult,
    TokenUsage,
)

logger = logging.getLogger(__name__)


def _parse_tool_call_args(raw: str) -> tuple[dict | None, dict | None]:
    """Parse a tool-call arguments blob. Returns (args, error_info).

    On success, args is the parsed dict and error_info is None.
    On failure (invalid JSON, wrong type, or non-object JSON like ``null`` /
    ``42`` / ``"foo"`` / ``[]``), args is None and error_info is a dict
    suitable for logging (with ``_parse_error`` and ``_raw`` keys). Callers
    that need a string error message should read ``error_info["_parse_error"]``.

    Tool schemas require objects, so anything that parses to a non-dict is
    treated as an error — otherwise ``args is None`` would be ambiguous
    (parse failure vs. JSON ``null``) and non-dict values would silently
    reach ``tool.execute`` and crash there with a less useful message.
    """
    try:
        args = json.loads(raw)
    except json.JSONDecodeError as exc:
        return None, {
            "_parse_error": f"{exc.msg} at line {exc.lineno} column {exc.colno}",
            "_raw": raw,
        }
    except TypeError as exc:
        return None, {"_parse_error": str(exc), "_raw": raw}
    if not isinstance(args, dict):
        return None, {
            "_parse_error": f"expected JSON object, got {type(args).__name__}",
            "_raw": raw,
        }
    return args, None


def _new_tokens(response, usage: TokenUsage) -> int:
    """One call's contribution to the tree budget: completion + uncached prompt tokens.
    The cached context prefix re-billed on every call is not new work; a provider that
    reports no cache detail counts the full prompt (conservative)."""
    details = getattr(getattr(response, "usage", None), "prompt_tokens_details", None)
    cached = getattr(details, "cached_tokens", 0) or 0
    return max(usage.prompt_tokens - cached, 0) + usage.completion_tokens


def _last_assistant_text(messages: list[dict]) -> str:
    """The most recent non-empty assistant content, for a graceful capped stop."""
    for msg in reversed(messages):
        if msg.get("role") == "assistant" and msg.get("content"):
            return msg["content"]
    return ""


class RLMEngine:
    def __init__(
        self,
        *,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
        mcp_servers: dict[str, MCPServer] | None = None,
        runtime_config: RuntimeConfig | None = None,
        supervisor: SessionTreeSupervisor | None = None,
        invocation_id: str | None = None,
        semantic_edges: SemanticEdgeTracker | None = None,
        parent_session_id: str | None = None,
        spawned_by_request_id: str | None = None,
    ):
        if runtime_config is None:
            raise ValueError(
                "RLMEngine requires an explicit runtime_config: standalone "
                "environment configuration was removed (rlm is consumed via "
                "the ACP runtime contract; children inherit in-memory)."
            )
        self.runtime_config = runtime_config
        config = self.runtime_config
        self.model = config.model
        self.cwd = cwd or os.getcwd()
        self.exec_timeout = config.policy.exec_timeout
        self.max_total_turns = config.policy.max_total_turns
        self.max_tool_output_bytes = config.policy.max_tool_output_bytes
        self.compaction = config.policy.compaction
        self.summarize_at_tokens = config.policy.summarize_at_tokens
        self.max_compactions = config.policy.max_compactions
        self.max_compaction_attempts = config.policy.max_compaction_attempts
        self.system_prompt_path = config.system_prompt_path
        self.append_to_system_prompt = config.resolved_append_to_system_prompt
        self.max_depth = config.policy.max_depth
        self.depth = config.invocation.depth
        self.allow_git = config.policy.allow_git

        # Task MCP tool servers to expose as IPython skills; kwarg wins, otherwise
        # parse RLM_MCP_CONFIG (a standard mcpServers config).
        self.mcp_servers = validate_mcp_servers(mcp_servers or {})

        # Built-in skills (rlm.skills) to enable for this run, from RLM_SKILLS (comma-separated).
        self.skills = list(config.skills)
        self.kernel_env = dict(config.kernel_env)
        self.max_tokens = config.policy.max_tokens

        self._owns_client = client is None
        self.client = client or make_client(config.provider)
        self.session = session
        self._supervisor = supervisor
        self._invocation_id = invocation_id or (
            supervisor.root_id if supervisor is not None else uuid.uuid4().hex
        )
        self._semantic_edges = semantic_edges or (
            supervisor.semantic_edges
            if supervisor is not None
            else SemanticEdgeTracker()
        )
        self._semantic_edges.register_session(
            self._invocation_id,
            parent_session_id=parent_session_id,
            spawned_by_request_id=spawned_by_request_id,
        )
        self._owns_supervisor = False
        self._total_usage = TokenUsage()
        # Engine-local tree-cap accounting. No supervisor exists when nothing needs
        # brokering (max_depth=0, no MCP, no search); that session is a one-engine
        # tree, so its own counters are the tree totals.
        self._own_turns = 0
        self._own_new_tokens = 0
        self._last_handoff_summary: str | None = None
        self._last_prompt_tokens = 0
        self._last_good = 0
        """Message count of the newest state that passed a threshold check - by
        definition a state with a full reserve of room, so a checkpoint over it fits."""
        self._compacted = False
        self._last_call_id: str | None = None

        # Metrics
        self._metrics = RLMMetrics()
        self._metrics._sub_rlm_enabled = self.max_depth > 0

        self._tool_state: dict[str, object] = {}

        # IPython REPL (started lazily in single-agent execution)
        self._repl: IPythonREPL | None = None

        # Turn index (0-based) at the start of the current branch. Used to
        # report "turns since last compaction" when a compaction fires.
        self._branch_start_turn: int = 0

        self._messages: list[dict] | None = None
        self._active_tools: list[BuiltinTool] = []
        self._active_tool_schemas: list[dict] = []
        self._turn = 0
        self._last_answer = ""
        self._has_result = False
        self._started = False
        self._closed = False
        self._close_task: asyncio.Task[None] | None = None

    def _ensure_session(self):
        """Create session if not set."""
        if self.session is not None:
            return
        session_dir = os.environ.get("RLM_SESSION_DIR")
        self.session = Session(session_dir)

    @property
    def stop_reason(self) -> str:
        """Reason the most recent prompt stopped."""
        return self._metrics.stop_reason

    async def run(self, prompt: str) -> RLMResult:
        """Run a single agent loop to completion."""
        try:
            return await self.prompt(prompt)
        finally:
            await self.aclose()

    async def prompt(self, prompt: str) -> RLMResult:
        """Run one user turn while preserving conversation and kernel state."""
        if self._closed or self._close_task is not None:
            raise RuntimeError("RLM engine is closed")

        if self.depth > self.max_depth:
            answer = f"[depth limit {self.max_depth} reached, cannot start]"
            self._metrics.stop_reason = "depth_limit"
            self._last_answer = answer
            self._has_result = True
            return RLMResult(
                answer=answer,
                turns=0,
                session_dir=self.session.dir if self.session is not None else None,
            )

        self._has_result = False

        if not self._started:
            try:
                await self._start(prompt)
            except BaseException as exc:
                self._metrics.stop_reason = (
                    "cancelled" if isinstance(exc, asyncio.CancelledError) else "error"
                )
                raise
            messages_before = self._messages[:1]
            last_good_before = len(messages_before)
        else:
            messages_before = list(self._messages)
            last_good_before = self._last_good
            self._messages.append({"role": "user", "content": prompt})
            # This turn's opening state is the floor for checkpoint fallbacks:
            # a fallback must never drop the newest user instruction.
            self._last_good = len(self._messages)
        branch_start_before = self._branch_start_turn
        compacted_before = self._compacted
        semantic_edges_before = self._semantic_edges.checkpoint(self._invocation_id)
        turn_before = self._turn
        usage_before = TokenUsage(
            prompt_tokens=self._total_usage.prompt_tokens,
            completion_tokens=self._total_usage.completion_tokens,
        )
        self._metrics.stop_reason = ""
        try:
            result = await self._run_loop()
        except BaseException as exc:
            attempted_turns = self._turn - turn_before
            try:
                self.session.log(
                    {
                        "type": "prompt_rollback",
                        "start_turn": turn_before,
                        "attempted_turns": attempted_turns,
                        "reason": (
                            "cancelled"
                            if isinstance(exc, asyncio.CancelledError)
                            else "error"
                        ),
                    }
                )
            except OSError:
                logger.warning("rlm: failed to log prompt rollback", exc_info=True)
            # Restore only the resumable conversation position. Usage, metrics,
            # kernel/tool side effects, and the append-only audit log describe work
            # that really ran and remain part of session accounting.
            self._messages[:] = messages_before
            self._last_good = last_good_before
            self._compacted = compacted_before
            self._branch_start_turn = branch_start_before
            self._semantic_edges.restore(self._invocation_id, semantic_edges_before)
            self._turn = turn_before
            if isinstance(exc, asyncio.CancelledError):
                self._metrics.stop_reason = "cancelled"
            else:
                self._metrics.stop_reason = "error"
            raise
        result.usage = TokenUsage(
            prompt_tokens=self._total_usage.prompt_tokens - usage_before.prompt_tokens,
            completion_tokens=(
                self._total_usage.completion_tokens - usage_before.completion_tokens
            ),
        )
        result.turns = self._turn - turn_before
        self._last_answer = result.answer
        self._has_result = True
        return result

    async def _start(self, prompt: str) -> None:
        """Initialize the session, tools, conversation, and persistent kernel."""

        if self.compaction and self.summarize_at_tokens is None:
            self.summarize_at_tokens = await discover_threshold(self.client, self.model)

        self._ensure_session()

        self.session.write_meta(
            session_id=self.session.dir.name,
            model=self.model,
            depth=self.depth,
            status="running",
            start_time=time.time(),
            prompt_preview=prompt[:200],
            cwd=self.cwd,
        )
        # Credential-free built-ins run locally; privileged skills are brokered.
        local_skills = [name for name in self.skills if name != "search"]
        enable_builtin_skills(local_skills, self.session.dir)
        broker_endpoint = None
        if self.depth < self.max_depth or self.mcp_servers or "search" in self.skills:
            if self._supervisor is None:
                self._supervisor = SessionTreeSupervisor(
                    root_session=self.session,
                    runtime_config=self.runtime_config,
                    cwd=self.cwd,
                    mcp_servers=self.mcp_servers,
                    root_invocation_id=self._invocation_id,
                    semantic_edges=self._semantic_edges,
                )
                self._owns_supervisor = True
            try:
                await self._supervisor.start()
                broker_endpoint = self._supervisor.endpoint_for(self._invocation_id)
                if self.mcp_servers or "search" in self.skills:
                    reserved_names = {"rlm", *local_skills, *discover_skills()}
                    brokered_skills = self._supervisor.write_brokered_skill_modules(
                        self.session.dir, reserved_names
                    )
                    logger.info(
                        "rlm: exposed %d supervisor-owned skill(s) - %s",
                        len(brokered_skills),
                        ", ".join(brokered_skills),
                    )
            except BaseException:
                if self._owns_supervisor:
                    await self._supervisor.aclose()
                    self._supervisor = None
                    self._owns_supervisor = False
                raise

        self._repl = IPythonREPL(
            cwd=self.cwd,
            session=self.session,
            kernel_env=self.kernel_env,
            depth=self.depth,
            max_depth=self.max_depth,
            broker_endpoint=broker_endpoint,
            exec_timeout=self.exec_timeout,
            allow_git=self.allow_git,
        )
        try:
            self._repl.start()

            self._active_tools = get_active_builtin_tools(self.exec_timeout)
            self._active_tool_schemas = [tool.schema() for tool in self._active_tools]
            system_prompt = self._load_system_prompt(self._active_tools)

            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            # The initial conversation is the floor for checkpoint fallbacks: a
            # first-turn checkpoint must never retry from an empty base.
            self._last_good = len(self._messages)
            self._started = True
        except BaseException:
            self._repl.shutdown()
            self._repl = None
            if self._owns_supervisor and self._supervisor is not None:
                await self._supervisor.aclose()
                self._supervisor = None
                self._owns_supervisor = False
            raise

    async def _run_loop(self) -> RLMResult:
        messages = self._messages
        if messages is None:
            raise RuntimeError("RLM engine is not started")

        final_text = ""
        # Cap-stop salvage looks only at messages produced by THIS prompt, so a later
        # prompt on an already-capped session can't replay a stale prior answer.
        salvage_from = len(messages)
        self._last_handoff_summary = None

        for turn in itertools.count(self._turn):
            # Cap checks run before the turn is counted, so a capped stop reports the
            # true number of model calls; the final answer falls back to this prompt's
            # last assistant text, then a compaction handoff summary, then a marker —
            # a capped sub-agent still hands its parent something.
            if capped := self._spent_tree_cap():
                self._metrics.stop_reason = capped
                final_text = (
                    _last_assistant_text(messages[salvage_from:])
                    or self._last_handoff_summary
                    or (
                        "[turn budget reached]"
                        if capped == "max_total_turns"
                        else "[token budget reached]"
                    )
                )
                break
            self._turn = turn + 1
            try:
                response, usage = await self._complete(messages, turn)
            except CompactionFailed:
                # The context is exhausted and could not be summarized: end the run
                # cleanly with what the conversation holds - still a trainable sample.
                self._metrics.stop_reason = "compaction_failed"
                final_text = "[context exhausted: compaction failed]"
                break
            call_id = self._last_call_id

            self._metrics.turns_since_last_compaction = (
                turn + 1 - self._branch_start_turn
            )

            msg = response.choices[0].message
            msg_dict = msg.model_dump(exclude_none=True)
            msg_dict.setdefault("content", "")
            messages.append(msg_dict)

            # Log assistant message; parse tool-call args once, reuse below.
            tool_calls_log: list[dict] | None = None
            parsed_args: list[dict | None] = []
            if msg.tool_calls:
                tool_calls_log = []
                for tc in msg.tool_calls:
                    args, err = _parse_tool_call_args(tc.function.arguments)
                    parsed_args.append(args)
                    tool_calls_log.append(
                        {
                            "name": tc.function.name,
                            "args": err if args is None else args,
                        }
                    )
            self.session.log_assistant(turn, tool_calls_log, msg.content)

            if msg.tool_calls and len(msg.tool_calls) > 1:
                feedback = "Error: only one tool call per turn allowed"
                for tc in msg.tool_calls:
                    self.session.log_tool_result(turn, tc.function.name, feedback, 0.0)
                    messages.append(
                        {"role": "tool", "tool_call_id": tc.id, "content": feedback}
                    )
                continue

            if msg.tool_calls and parsed_args[0] is None:
                tc = msg.tool_calls[0]
                tool_name = tc.function.name
                err_info = tool_calls_log[0]["args"]
                feedback = (
                    f"Error: invalid JSON arguments for tool '{tool_name}': "
                    f"{err_info['_parse_error']}"
                )
                self.session.log_tool_result(turn, tool_name, feedback, 0.0)
                messages.append(
                    {"role": "tool", "tool_call_id": tc.id, "content": feedback}
                )
                continue

            # Token budget check
            if (
                self.max_tokens
                and self._total_usage.completion_tokens >= self.max_tokens
            ):
                self._metrics.stop_reason = "token_budget"
                final_text = msg.content or "[token budget exhausted]"
                break

            # No tool calls → done
            if not msg.tool_calls:
                self._metrics.stop_reason = "done"
                final_text = msg.content or ""
                break

            tc = msg.tool_calls[0]
            tool_name = tc.function.name
            tool_args = parsed_args[0]
            t0 = time.time()
            tool = get_builtin_tool(tool_name)
            if tool is None:
                tool_result = ToolOutcome(content=f"Error: unknown tool '{tool_name}'")
            else:
                repl = self._repl
                scope_id = None
                if (
                    tool_name == "ipython"
                    and repl is not None
                    and self._supervisor is not None
                    and self._invocation_id is not None
                ):
                    scope_id = await self._supervisor.open_scope(
                        self._invocation_id, call_id
                    )
                try:
                    if scope_id is not None:
                        repl.set_broker_scope(
                            scope_id,
                            self._supervisor.broker_waits(scope_id),
                        )
                    tool_task = asyncio.create_task(
                        asyncio.to_thread(
                            tool.execute, tool_args, self._tool_context(messages)
                        )
                    )
                    try:
                        tool_result = await asyncio.shield(tool_task)
                    except asyncio.CancelledError:
                        settled_result = None
                        if repl is not None:
                            repl.interrupt()
                        try:
                            while True:
                                try:
                                    settled_result = await asyncio.shield(tool_task)
                                except asyncio.CancelledError:
                                    if repl is not None:
                                        repl.interrupt()
                                    continue
                                except Exception:
                                    logger.warning(
                                        "rlm: tool failed while cancellation was settling",
                                        exc_info=True,
                                    )
                                break
                        finally:
                            if repl is not None:
                                repl.finish_interrupt()
                        if settled_result is not None:
                            for event in settled_result.metric_events:
                                self._metrics.record(event)
                        raise
                finally:
                    if scope_id is not None:
                        try:
                            await self._supervisor.close_scope(scope_id)
                        finally:
                            if repl is not None:
                                try:
                                    repl.set_broker_scope(None)
                                except Exception:
                                    logger.warning(
                                        "rlm: failed to clear broker scope",
                                        exc_info=True,
                                    )
            duration = time.time() - t0
            for event in tool_result.metric_events:
                self._metrics.record(event)

            result = tool_result.content

            self.session.log_tool_result(turn, tool_name, result, duration)
            content = truncate_tool_output(
                result, self.max_tool_output_bytes or TOOL_OUTPUT_MAX_BYTES
            )
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": content,
                }
            )

            if self._should_compact(messages, usage, content):
                try:
                    await self._compact_branch(messages, turn)
                except CompactionFailed:
                    self._metrics.stop_reason = "compaction_failed"
                    final_text = "[context exhausted: compaction failed]"
                    break

        result = RLMResult(
            answer=final_text,
            session_dir=self.session.dir,
            usage=self._total_usage,
            turns=self._turn,
        )
        return result

    async def aclose(self) -> None:
        """Finalize artifacts and stop the complete recursive session tree."""
        if self._closed and self._close_task is None:
            return
        if self._close_task is None:
            self._close_task = asyncio.create_task(self._aclose_impl())
        cancelled = False
        while True:
            try:
                await asyncio.shield(self._close_task)
                break
            except asyncio.CancelledError:
                if self._close_task.done():
                    raise
                cancelled = True
        if cancelled:
            raise asyncio.CancelledError

    async def _aclose_impl(self) -> None:
        self._closed = True
        try:
            if self._owns_supervisor and self._supervisor is not None:
                await self._supervisor.aclose()
        finally:
            try:
                if self._owns_client:
                    await self.client.close()
            finally:
                self._close_local()

    def close(self) -> None:
        """Close synchronous resources, or run async cleanup outside an event loop."""
        if self._closed:
            return
        if self._owns_supervisor or self._owns_client:
            try:
                asyncio.get_running_loop()
            except RuntimeError:
                asyncio.run(self.aclose())
            else:
                raise RuntimeError("use 'await engine.aclose()' inside an event loop")
            return
        self._closed = True
        self._close_local()

    def _close_local(self) -> None:
        if self._repl is not None:
            try:
                self._repl.shutdown()
            except Exception:
                logger.warning("rlm: failed to stop IPython kernel", exc_info=True)
            self._repl = None
        if self.session is not None:
            if self._has_result:
                direct_tool_stats = None
                child_tool_stats = None
                if self._supervisor is not None and self._invocation_id is not None:
                    direct_tool_stats, child_tool_stats = (
                        self._supervisor.programmatic_tool_call_stats(
                            self._invocation_id
                        )
                    )
                self.session.finalize(
                    self._last_answer,
                    usage={
                        "prompt_tokens": self._total_usage.prompt_tokens,
                        "completion_tokens": self._total_usage.completion_tokens,
                    },
                    turns=self._turn,
                    metrics=self._metrics,
                    trusted_direct_tool_stats=direct_tool_stats,
                    trusted_child_tool_stats=child_tool_stats,
                )
            else:
                self.session.close()

    def _programmatic_tool_call_stats(
        self,
    ) -> tuple[ProgrammaticToolCallStats, ProgrammaticToolCallStats, int]:
        if self.session is None:
            return ProgrammaticToolCallStats(), ProgrammaticToolCallStats(), 0
        direct = ProgrammaticToolCallStats.from_log(
            self.session.dir / "programmatic_tool_calls.jsonl"
        )
        child_aggregate = self.session.aggregate_child_metrics(
            "local_programmatic_tool_call_stats"
        )
        child = child_aggregate.tool_call_stats
        if self._supervisor is not None:
            trusted_direct, trusted_child = (
                self._supervisor.programmatic_tool_call_stats(self._invocation_id)
            )
            direct = direct.merge(trusted_direct)
            child = child.merge(trusted_child)
        return direct, child, child_aggregate.num_sessions

    def _can_compact(self) -> bool:
        return self.compaction and (
            self.max_compactions is None
            or self._metrics.num_compactions < self.max_compactions
        )

    def _spent_tree_cap(self) -> str | None:
        """The stop_reason of a spent tree budget, or None. Live supervisor totals
        cover the whole tree; without a supervisor this engine is the whole tree."""
        if self._supervisor is not None:
            turns = self._supervisor.total_turns
            tokens = self._supervisor.total_tokens
        else:
            turns = self._own_turns
            tokens = self._own_new_tokens
        if self.max_total_turns is not None and turns >= self.max_total_turns:
            return "max_total_turns"
        budget = self.runtime_config.policy.max_total_tokens
        if budget is not None and tokens >= budget:
            return "max_total_tokens"
        return None

    def _should_compact(
        self, messages: list[dict], usage: TokenUsage, extra_text: str = ""
    ) -> bool:
        if self.summarize_at_tokens is None or not self._can_compact():
            return False
        # A spent tree cap stops the session next iteration: don't burn a model
        # call summarizing a conversation that is about to end. (The reactive
        # overflow path stays available - that call was already permitted.)
        if self._spent_tree_cap() is not None:
            return False
        if not compactable(messages):
            return False
        tokens = usage.total + estimated_tokens(extra_text)
        return tokens >= self.summarize_at_tokens

    async def _call_model(
        self,
        messages: list[dict],
        *,
        checkpoint: bool = False,
        compaction_id: str | None = None,
    ) -> tuple[Any, TokenUsage]:
        request_id = self._semantic_edges.start_request(
            self._invocation_id, compaction_id=compaction_id
        )
        request: dict = {
            "model": self.model,
            "messages": messages,
            "extra_headers": model_call_headers(request_id),
        }
        if self._active_tool_schemas:
            request["tools"] = self._active_tool_schemas
            if checkpoint:
                request["tool_choice"] = "none"
            else:
                request["parallel_tool_calls"] = False

        try:
            response = await call_with_retries(
                self.client.chat.completions.create, **request
            )
        except BaseException:
            self._semantic_edges.fail_request(request_id)
            raise
        self._semantic_edges.finish_request(request_id)
        usage = extract_usage(response)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens
        new_tokens = _new_tokens(response, usage)
        self._own_new_tokens += new_tokens
        if not checkpoint:
            self._own_turns += 1
        if self._supervisor is not None:
            if checkpoint:
                self._supervisor.record_usage(new_tokens)
            else:
                self._supervisor.record_call(new_tokens)
        if not checkpoint:
            self._last_prompt_tokens = usage.prompt_tokens
            self._last_call_id = request_id
        return response, usage

    async def _complete(
        self, messages: list[dict], turn: int
    ) -> tuple[Any, TokenUsage]:
        """Complete one turn, with at most one compact-and-retry cycle."""
        try:
            response, usage = await self._call_model(messages)
        except APIStatusError as error:
            # Reactive compaction needs no discovered threshold: the overflow
            # itself is the signal. The checkpoint fallback chain handles a
            # summary request that is itself too large.
            if not self._can_compact() or not is_context_overflow(error):
                raise
            if not compactable(messages):
                if self._compacted:
                    # The conversation is already a compaction floor and still
                    # overflows - out of moves, end cleanly.
                    raise CompactionFailed(
                        "the compacted conversation still overflows"
                    ) from error
                raise
        else:
            choice = response.choices[0]
            if (
                self.summarize_at_tokens is not None
                and usage.total < self.summarize_at_tokens
            ):
                # Usage-verified: this exact prompt was accepted with a full
                # reserve of room, so it is a safe checkpoint fallback.
                self._last_good = len(messages)
            if choice.finish_reason != "length" or not self._should_compact(
                messages, usage
            ):
                return response, usage

        await self._compact_branch(messages, turn)
        try:
            return await self._call_model(messages)
        except APIStatusError as error:
            # The rebuilt conversation is sized to fit, so this is out of moves.
            if is_context_overflow(error):
                raise CompactionFailed(
                    "the rebuilt conversation still overflows"
                ) from error
            raise

    async def _compact_branch(
        self,
        messages: list[dict],
        turn: int,
    ) -> None:
        """Ask the model for a handoff summary and rebuild ``messages``.

        Called in-place: mutates ``messages`` to ``[system, user(framing +
        summary)]`` while preserving the IPython kernel. A summary attempt is
        housekeeping, not a work turn: it does not count toward ``max_total_turns``.
        Its tokens still land in ``_total_usage`` for cost accounting and count
        toward token budgets. Every committed attempt remains represented in the
        semantic graph.

        Active tools are forwarded with ``tool_choice="none"`` so the system prompt matches
        regular turns (vLLM's chat-completions layer injects the tools
        block into the system message only when ``tools=`` is set). With
        a matching system prompt, prime-rl's RL trajectory walker keeps
        the extension property across the compaction boundary instead
        of opening an extra training-sample split. ``tool_choice="none"``
        keeps the original "text-only summary" behaviour by forbidding
        tool calls on this turn.
        """
        dropped_chars = _count_messages_chars(messages[1:])
        turns_since_last = turn + 1 - self._branch_start_turn

        checkpoint_prompt = CHECKPOINT_PROMPT
        if self._repl is not None:
            checkpoint_prompt += REPL_NOTE
        compaction = self._semantic_edges.begin_compaction(self._invocation_id)
        try:
            # A rejected checkpoint falls back to the last good snapshot (which has a
            # full reserve of room, so it fits); an empty or tool-calling reply is
            # resampled. Reasoning is never part of the summary.
            base = messages
            summary_text = ""
            for _ in range(self.max_compaction_attempts):
                checkpoint = [
                    *base,
                    {"role": "user", "content": checkpoint_prompt},
                ]
                try:
                    response, usage = await self._call_model(
                        checkpoint,
                        checkpoint=True,
                        compaction_id=compaction.compaction_id,
                    )
                except APIStatusError as e:
                    if not is_context_overflow(e):
                        raise
                    base = messages[: self._last_good]
                    continue
                message = response.choices[0].message
                # Reasoning never enters the summary: only the reply's final text
                # counts, so a reply that lives entirely in the reasoning channel
                # is resampled like an empty one.
                text = (message.content or "").strip()
                if not message.tool_calls and text:
                    summary_text = text
                    break
                self._semantic_edges.release_summary_request(compaction.compaction_id)
            if not summary_text:
                raise CompactionFailed(
                    f"no usable summary after {self.max_compaction_attempts} attempts"
                )
        except BaseException as exc:
            self._semantic_edges.finish_compaction(
                compaction.compaction_id,
                "cancelled" if isinstance(exc, asyncio.CancelledError) else "failed",
            )
            raise

        system_msg = messages[0]
        self._last_handoff_summary = summary_text
        compacted_user_content = SUMMARY_FRAMING + "\n\n" + summary_text
        messages[:] = [
            system_msg,
            {"role": "user", "content": compacted_user_content},
        ]
        self._last_good = len(messages)
        self._compacted = True
        self._semantic_edges.finish_compaction(compaction.compaction_id, "completed")

        # Log the compaction for traceability.
        self.session.log(
            {
                "type": "compaction",
                "turn": turn,
                "summary": summary_text,
                "summary_chars": len(summary_text),
                "dropped_chars": dropped_chars,
                "turns_since_last_compaction": turns_since_last,
                "usage": {
                    "prompt_tokens": usage.prompt_tokens,
                    "completion_tokens": usage.completion_tokens,
                },
            }
        )

        # Metrics: close the old branch.
        self._metrics.record(
            CompactionApplied(
                dropped_chars=dropped_chars,
                summary_chars=len(summary_text),
                turns_since_last_compaction=turns_since_last,
            )
        )
        self._branch_start_turn = turn + 1
        self._metrics.turns_since_last_compaction = 0

    def execution_snapshot(self) -> dict:
        """Return a credential-free snapshot of cumulative execution state."""
        if self.session is None:
            raise RuntimeError("RLM session is not initialized")
        direct_tool_stats, child_tool_stats, num_child_sessions = (
            self._programmatic_tool_call_stats()
        )
        metrics = deepcopy(self._metrics)
        metrics.apply_programmatic_tool_call_stats(
            direct_tool_stats, child_tool_stats, num_child_sessions
        )
        metric_values = {
            key: value
            for key, value in metrics.to_dict().items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        snapshot = {
            "model": self.model,
            "turns": self._turn,
            "usage": {
                "prompt_tokens": self._total_usage.prompt_tokens,
                "completion_tokens": self._total_usage.completion_tokens,
                "total_tokens": self._total_usage.total,
            },
            "metrics": metric_values,
            "programmatic_tool_call_stats": direct_tool_stats.merge(
                child_tool_stats
            ).to_dict(),
            "supervisor": {
                "subagent_calls": self._supervisor.total_calls
                if self._supervisor is not None
                else 0,
                "active_subagent_calls": self._supervisor.active_calls
                if self._supervisor is not None
                else 0,
            },
            "limits": {
                "max_depth": self.runtime_config.policy.max_depth,
                "max_concurrent_subagents": self.runtime_config.policy.max_concurrent_subagents,
                "max_subagent_calls": self.runtime_config.policy.max_subagent_calls,
                "max_tokens": self.runtime_config.policy.max_tokens,
                "compaction": self.compaction,
                "summarize_at_tokens": self.summarize_at_tokens,
                "max_compactions": self.runtime_config.policy.max_compactions,
                "max_compaction_attempts": self.max_compaction_attempts,
                "allow_git": self.runtime_config.policy.allow_git,
            },
            "semantic_edges": self._semantic_edges.snapshot(),
        }
        return snapshot

    def _load_system_prompt(self, active_tools: list[BuiltinTool]) -> str:
        if self.system_prompt_path:
            return Path(self.system_prompt_path).read_text()
        system_prompt = build_system_prompt(
            self.cwd,
            str(SKILLS_DIR) if SKILLS_DIR is not None else None,
            discover_skills(self.session.dir),
            depth=self.depth,
            session_dir=str(self.session.dir),
            allow_recursion=self.depth < self.max_depth,
            allow_git=self.allow_git,
            active_tools=active_tools,
            shell_skills=get_installed_skills(),
        )
        if self.append_to_system_prompt:
            system_prompt += "\n\n" + self.append_to_system_prompt
        return system_prompt

    def _tool_context(self, messages: list[dict]) -> ToolContext:
        return ToolContext(
            messages=messages,
            metrics=self._metrics,
            total_usage=self._total_usage,
            last_prompt_tokens=self._last_prompt_tokens,
            exec_timeout=self.exec_timeout,
            allow_git=self.allow_git,
            repl=self._repl,
            state=self._tool_state,
            cwd=self.cwd,
        )


def _count_messages_chars(messages: list[dict]) -> int:
    """Sum the content-char length across ``messages`` (text + tool-call args).

    Used as a rough "how much was dropped" metric on compaction. Tool-call
    argument strings are counted since they consume context just like
    message content does.
    """
    total = 0
    for message in messages:
        total += _content_chars(message.get("content"))
        tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for tc in tool_calls:
                total += _tool_call_chars(tc)
    return total


def _content_chars(content) -> int:
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(_content_chars(item) for item in content)
    if isinstance(content, dict):
        total = 0
        for field_name in ("text", "input_text", "output_text"):
            value = content.get(field_name)
            if isinstance(value, str):
                total += len(value)
        nested = content.get("content")
        if nested is not None:
            total += _content_chars(nested)
        return total
    return 0


def _tool_call_chars(tool_call) -> int:
    if isinstance(tool_call, dict):
        function = tool_call.get("function")
    else:
        function = getattr(tool_call, "function", None)
    if function is None:
        return 0
    if isinstance(function, dict):
        name = function.get("name")
        arguments = function.get("arguments")
    else:
        name = getattr(function, "name", None)
        arguments = getattr(function, "arguments", None)
    total = 0
    if isinstance(name, str):
        total += len(name)
    if isinstance(arguments, str):
        total += len(arguments)
    return total
