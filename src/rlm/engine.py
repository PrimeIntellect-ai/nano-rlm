"""The agent loop."""

from __future__ import annotations

import asyncio
import itertools
import json
import logging
import os
import time
from pathlib import Path

from openai import AsyncOpenAI, BadRequestError

from rlm.client import call_with_retries, extract_usage, make_client
from rlm.config import RuntimeConfig
from rlm.mcp import (
    MCP_CONFIG_ENV,
    MCPServer,
    dump_mcp_servers,
    generate_mcp_skills,
    load_mcp_servers,
)
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
from rlm.types import CompactionApplied, RLMMetrics, RLMResult, TokenUsage

logger = logging.getLogger(__name__)


# Injected as a user message when the branch's context size reaches the
# compaction threshold. The model's next reply is expected to be a
# plain-text handoff summary; any tool calls it emits are ignored and
# the message is compacted in place of them.
CHECKPOINT_COMPACTION_PROMPT = (
    "You are performing a CONTEXT CHECKPOINT COMPACTION. "
    "Create a handoff summary for another LLM that will resume the task.\n"
    "\n"
    "Include:\n"
    "- Current progress and key decisions made\n"
    "- Important context, constraints, or user preferences\n"
    "- What remains to be done (clear next steps)\n"
    "- Any critical data, examples, or references needed to continue\n"
    "\n"
    "Be concise, structured, and focused on helping the next LLM "
    "seamlessly continue the work."
)

# Appended to the checkpoint prompt when the IPython REPL is active.
REPL_RESTART_NOTE = (
    "\n\n"
    "Note: the IPython kernel stays running across this compaction. "
    "All variables, imports, and in-memory data are preserved. "
    "Mention important variable names and what they contain so the "
    "next LLM knows what's available."
)

# Wrapper text that frames the summary as the sole user-facing context
# for the post-compaction branch. The original task prompt is dropped;
# the summary is responsible for carrying the goal.
POST_COMPACTION_FRAMING = (
    "Another language model started to solve this problem and produced "
    "a summary of its thinking process. Use this to build on the work "
    "that has already been done and avoid duplicating work. Here is "
    "the summary produced by the other language model, use the "
    "information in this summary to assist with your own analysis:"
)


def _is_request_too_large(e: BadRequestError) -> bool:
    """True if a 400 matches the proxy's "Request Entity Too Large" body."""
    haystack = f"{e} {getattr(e, 'body', '') or ''}".lower()
    return "request entity too large" in haystack


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


class RLMEngine:
    def __init__(
        self,
        model: str | None = None,
        summarize_at_tokens: int | None = None,
        system_prompt_path: str | None = None,
        append_to_system_prompt: str | None = None,
        cwd: str | None = None,
        session: Session | None = None,
        client: AsyncOpenAI | None = None,
        mcp_servers: dict[str, MCPServer] | None = None,
        runtime_config: RuntimeConfig | None = None,
        supervisor: SessionTreeSupervisor | None = None,
        invocation_id: str | None = None,
    ):
        if runtime_config is not None and any(
            value is not None
            for value in (
                model,
                summarize_at_tokens,
                system_prompt_path,
                append_to_system_prompt,
            )
        ):
            raise ValueError(
                "runtime_config cannot be combined with runtime configuration kwargs"
            )
        self.runtime_config = runtime_config or RuntimeConfig.from_env(
            model=model,
            summarize_at_tokens=summarize_at_tokens,
            system_prompt_path=system_prompt_path,
            append_to_system_prompt=append_to_system_prompt,
        )
        config = self.runtime_config
        self.model = config.model
        self.cwd = cwd or os.getcwd()
        self.exec_timeout = config.policy.exec_timeout
        self.max_output = config.policy.max_output
        self.summarize_at_tokens = config.policy.summarize_at_tokens
        self.max_compactions = config.policy.max_compactions
        self.system_prompt_path = config.system_prompt_path
        self.append_to_system_prompt = config.append_to_system_prompt
        self.max_depth = config.policy.max_depth
        self.depth = config.invocation.depth

        # Task MCP tool servers to expose as IPython skills; kwarg wins, otherwise
        # parse RLM_MCP_CONFIG (a standard mcpServers config).
        self.mcp_servers = (
            mcp_servers if mcp_servers is not None else load_mcp_servers()
        )

        # Built-in skills (rlm.skills) to enable for this run, from RLM_SKILLS (comma-separated).
        self.skills = list(config.skills)
        self.max_tokens = config.policy.max_tokens

        self._owns_client = client is None
        self.client = client or make_client(config.provider, config.invocation)
        self.session = session
        self._supervisor = supervisor
        self._invocation_id = invocation_id
        self._owns_supervisor = False
        self._total_usage = TokenUsage()
        self._last_prompt_tokens = 0

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
            await self._start(prompt)
            messages_before = self._messages[:1]
        else:
            messages_before = list(self._messages)
            self._messages.append({"role": "user", "content": prompt})
        branch_start_before = self._branch_start_turn
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
            self._branch_start_turn = branch_start_before
            self._turn = turn_before
            if isinstance(exc, asyncio.CancelledError):
                self._metrics.stop_reason = "cancelled"
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

        # Skills the kernel pre-imports, all written into the session dir (the REPL and prompt
        # read them back from there): enabled built-in skills (rlm.skills) + any wired MCP tools
        # (PTC). ipython is the sole builtin tool, so the kernel always starts.
        enable_builtin_skills(self.skills, self.session.dir)
        if self.mcp_servers:
            mcp_skills = await generate_mcp_skills(
                self.mcp_servers, self.session.dir, self.cwd
            )
            logger.info(
                "rlm: exposed %d MCP tool(s) as skills - %s",
                len(mcp_skills),
                ", ".join(mcp_skills),
            )

        repl_env = (
            {MCP_CONFIG_ENV: dump_mcp_servers(self.mcp_servers)}
            if self.mcp_servers
            else None
        )
        broker_endpoint = None
        if self.depth < self.max_depth:
            if self._supervisor is None:
                self._supervisor = SessionTreeSupervisor(
                    root_session=self.session,
                    runtime_config=self.runtime_config,
                    cwd=self.cwd,
                    mcp_servers=self.mcp_servers,
                )
                self._invocation_id = self._supervisor.root_id
                self._owns_supervisor = True
            await self._supervisor.start()
            if self._invocation_id is None:
                raise RuntimeError("recursive engine has no supervisor invocation")
            broker_endpoint = self._supervisor.endpoint_for(self._invocation_id)

        self._repl = IPythonREPL(
            cwd=self.cwd,
            session=self.session,
            env=repl_env,
            depth=self.depth,
            max_depth=self.max_depth,
            broker_endpoint=broker_endpoint,
        )
        try:
            self._repl.start()

            self._active_tools = get_active_builtin_tools()
            self._active_tool_schemas = [tool.schema() for tool in self._active_tools]
            messages_path = str(self.session.dir / "messages.jsonl")
            system_prompt = self._load_system_prompt(messages_path, self._active_tools)

            self._messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ]
            self._started = True
        except BaseException:
            self._repl.shutdown()
            self._repl = None
            if self._owns_supervisor and self._supervisor is not None:
                await self._supervisor.aclose()
                self._supervisor = None
                self._invocation_id = None
                self._owns_supervisor = False
            raise

    async def _run_loop(self) -> RLMResult:
        messages = self._messages
        if messages is None:
            raise RuntimeError("RLM engine is not started")

        final_text = ""

        for turn in itertools.count(self._turn):
            self._turn = turn + 1
            # Call LLM
            request_kwargs = {
                "model": self.model,
                "messages": messages,
            }
            if self._active_tool_schemas:
                request_kwargs["tools"] = self._active_tool_schemas
                request_kwargs["parallel_tool_calls"] = False
            try:
                response = await call_with_retries(
                    self.client.chat.completions.create,
                    **request_kwargs,
                )
            except BadRequestError as e:
                if not _is_request_too_large(e):
                    raise
                self._metrics.stop_reason = "request_too_large"
                final_text = "[request body too large]"
                break

            usage = extract_usage(response)
            self._total_usage.prompt_tokens += usage.prompt_tokens
            self._total_usage.completion_tokens += usage.completion_tokens
            self._last_prompt_tokens = usage.prompt_tokens

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
                    scope_id = await self._supervisor.open_scope(self._invocation_id)
                try:
                    if scope_id is not None:
                        repl.set_broker_scope(scope_id)
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

            if self.max_output > 0 and len(result) > self.max_output:
                result = result[: self.max_output] + "\n... [output truncated]"

            self.session.log_tool_result(turn, tool_name, result, duration)
            messages.append(
                {
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                }
            )

            # Auto-compaction: if this turn's prompt_tokens reached the
            # configured threshold, ask the model for a handoff summary and
            # rebuild the branch around it. Fires at most once per loop
            # iteration; the compaction op takes its own LLM call. A
            # max_compactions cap, once hit, disables further compaction so
            # the context grows to the model's natural limit.
            if (
                self.summarize_at_tokens is not None
                and usage.prompt_tokens >= self.summarize_at_tokens
                and (
                    self.max_compactions is None
                    or self._metrics.num_compactions < self.max_compactions
                )
            ):
                try:
                    await self._compact_branch(
                        messages, turn, self._active_tool_schemas
                    )
                except BadRequestError as e:
                    if not _is_request_too_large(e):
                        raise
                    self._metrics.stop_reason = "request_too_large"
                    final_text = "[request body too large]"
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
                self.session.finalize(
                    self._last_answer,
                    usage={
                        "prompt_tokens": self._total_usage.prompt_tokens,
                        "completion_tokens": self._total_usage.completion_tokens,
                    },
                    turns=self._turn,
                    metrics=self._metrics,
                )
            else:
                self.session.close()

    async def _compact_branch(
        self, messages: list[dict], turn: int, active_tools: list[dict]
    ) -> None:
        """Ask the model for a handoff summary and rebuild ``messages``.

        Called in-place: mutates ``messages`` to ``[system, user(framing +
        summary)]`` and restarts the ipython kernel. The LLM call for the
        summary is housekeeping, not a work turn, but its tokens land in
        ``_total_usage`` for cost accounting.

        ``active_tools`` is forwarded as ``tools=`` with
        ``tool_choice="none"`` so the rendered system prompt matches
        regular turns (vLLM's chat-completions layer injects the tools
        block into the system message only when ``tools=`` is set). With
        a matching system prompt, prime-rl's RL trajectory walker keeps
        the extension property across the compaction boundary instead
        of opening an extra training-sample split. ``tool_choice="none"``
        keeps the original "text-only summary" behaviour by forbidding
        tool calls on this turn.
        """
        # Measure what's about to be dropped BEFORE appending the
        # checkpoint prompt — otherwise the prompt's own chars get
        # counted as "dropped conversation content", inflating the
        # metric and the session log's dropped_chars field.
        dropped_chars = _count_messages_chars(messages[1:])
        turns_since_last = turn + 1 - self._branch_start_turn

        # Append the checkpoint prompt and ask the model for a text-only
        # summary turn. Tools are advertised to the server (so the system
        # prompt renders identically to regular turns) but
        # ``tool_choice="none"`` forbids the model from calling any.
        # Warn about the REPL restart only when a kernel is actually running.
        checkpoint_prompt = CHECKPOINT_COMPACTION_PROMPT
        if self._repl is not None:
            checkpoint_prompt += REPL_RESTART_NOTE
        messages.append({"role": "user", "content": checkpoint_prompt})
        request_kwargs: dict = {"model": self.model, "messages": messages}
        if active_tools:
            request_kwargs["tools"] = active_tools
            request_kwargs["tool_choice"] = "none"
        response = await call_with_retries(
            self.client.chat.completions.create,
            **request_kwargs,
        )
        usage = extract_usage(response)
        self._total_usage.prompt_tokens += usage.prompt_tokens
        self._total_usage.completion_tokens += usage.completion_tokens

        summary_text = response.choices[0].message.content or ""

        system_msg = messages[0]
        compacted_user_content = POST_COMPACTION_FRAMING + "\n\n" + summary_text
        messages[:] = [
            system_msg,
            {"role": "user", "content": compacted_user_content},
        ]

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

    def _load_system_prompt(
        self, messages_path: str, active_tools: list[BuiltinTool]
    ) -> str:
        if self.system_prompt_path:
            return Path(self.system_prompt_path).read_text()
        system_prompt = build_system_prompt(
            self.cwd,
            str(SKILLS_DIR) if SKILLS_DIR is not None else None,
            discover_skills(self.session.dir),
            messages_path,
            allow_recursion=self.depth < self.max_depth,
            active_tools=active_tools,
            cli_skills=get_installed_skills(),
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
