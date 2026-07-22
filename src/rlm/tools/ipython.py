"""Builtin IPython tool and persistent REPL implementation."""

from __future__ import annotations

import copy
import os
from queue import Empty
import re
import sys
import threading
import time
from typing import TYPE_CHECKING, Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_in_ipython, refusal
from rlm.tools.skills import discover_skills
from rlm.types import IpythonExecuted

if TYPE_CHECKING:
    from rlm.session import Session


IPYTHON_SCHEMA = {
    "type": "function",
    "function": {
        "name": "ipython",
        "description": (
            "Execute code in a persistent IPython session. Variables, imports, "
            "and function definitions persist across calls. "
            "Use !command for shell commands (e.g. !ls -la, !cat file.py, !pip install foo). "
            "Use !python3 to run code with the project's own packages "
            "(e.g. !python3 -m pytest, !python3 -c 'import numpy'). "
            "Use %%bash for multi-line shell scripts."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "code": {
                    "type": "string",
                    "description": "Python or IPython code to execute.",
                },
                "timeout": {
                    "type": "integer",
                    "description": None,  # filled by schema()
                },
            },
            "required": ["code"],
        },
    },
}

_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
IPYTHON_TIMEOUT_MAX_SECONDS = 600


class IpythonTool:
    """Builtin tool handler for the persistent IPython session."""

    name = "ipython"

    def schema(self) -> dict[str, Any]:
        timeout = min(
            int(os.environ.get("RLM_EXEC_TIMEOUT", "300")),
            IPYTHON_TIMEOUT_MAX_SECONDS,
        )
        schema = copy.deepcopy(IPYTHON_SCHEMA)
        schema["function"]["parameters"]["properties"]["timeout"]["description"] = (
            "Optional timeout in seconds. "
            f"Default: {timeout}s. Maximum: {IPYTHON_TIMEOUT_MAX_SECONDS}s."
        )
        return schema

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        code = args.get("code", "")
        if not isinstance(code, str):
            code = str(code)
        input_chars = len(code)
        input_loc = self._count_nonempty_lines(code)
        metric_events = [IpythonExecuted(input_chars=input_chars, input_loc=input_loc)]

        timeout = args.get("timeout")
        if timeout is None:
            timeout = context.exec_timeout
        else:
            try:
                timeout = int(timeout)
            except (TypeError, ValueError):
                timeout = context.exec_timeout
        timeout = min(timeout, IPYTHON_TIMEOUT_MAX_SECONDS)

        if context.repl is None:
            return ToolOutcome(
                content="Error: IPython REPL is not available",
                metric_events=metric_events,
            )

        blocked = find_blocked_in_ipython(code)
        if blocked is not None:
            return ToolOutcome(
                content=refusal(blocked),
                metric_events=metric_events,
            )

        return ToolOutcome(
            content=self._maybe_truncate_output(
                context.repl.execute(code, timeout=timeout)
            ),
            metric_events=metric_events,
        )

    @staticmethod
    def _count_nonempty_lines(code: str) -> int:
        return sum(1 for line in code.splitlines() if line.strip())

    @staticmethod
    def _maybe_truncate_output(content: str) -> str:
        """Truncate ``content`` to ``RLM_MAX_TOOL_OUTPUT_CHARS`` if set (off by default)."""
        cap = int(os.environ.get("RLM_MAX_TOOL_OUTPUT_CHARS", "-1"))
        if cap <= 0 or len(content) <= cap:
            return content
        head, tail = cap // 2, cap - cap // 2
        return f"{content[:head]}\n...[{len(content) - cap} chars truncated]...\n{content[-tail:]}"


class IPythonREPL:
    """Persistent IPython kernel communicating via Jupyter protocol."""

    def __init__(
        self,
        cwd: str,
        session: "Session | None" = None,
        env: dict[str, str] | None = None,
    ):
        self.cwd = cwd
        self.session = session
        self.env = env or {}
        self._km = None
        self._kc = None
        self._lock = threading.Lock()

    def start(self):
        """Start the IPython kernel."""
        from jupyter_client import KernelManager

        self._km = KernelManager()
        self._km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        self._km.start_kernel(cwd=self.cwd, env={**os.environ, **self.env})
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _inject_startup(self):
        """Set up kernel: cwd, env vars, nest_asyncio, skill pre-imports."""
        session_dir = str(self.session.dir) if self.session else None
        depth = int(os.environ.get("RLM_DEPTH", "0"))
        max_depth = int(os.environ.get("RLM_MAX_DEPTH", "0"))
        allow_recursion = depth < max_depth
        # Pip-installed skills + the MCP-tool modules generated into the session dir (rlm.mcp);
        # the session dir goes on the kernel's sys.path so those import by name.
        skill_names = discover_skills(self.session.dir if self.session else None)

        setup_code = f"""\
import os, sys, types, json, time, functools, inspect
os.chdir({self.cwd!r})
if {bool(session_dir)!r}:
    sys.path.append({session_dir!r})
os.environ['RLM_SESSION_DIR'] = {session_dir!r} or ''
os.environ['RLM_DEPTH'] = str({depth!r} + 1)
os.environ['NO_COLOR'] = '1'

import nest_asyncio
nest_asyncio.apply()


def _log_programmatic_call(tool_name, source):
    # Matches the line format written by install.sh's bash wrapper so
    # ProgrammaticToolCallStats.from_log parses both sources identically.
    session_dir = os.environ.get('RLM_SESSION_DIR', '')
    if not session_dir:
        return
    try:
        with open(os.path.join(session_dir, 'programmatic_tool_calls.jsonl'), 'a') as f:
            f.write(json.dumps({{
                'tool': tool_name,
                'source': source,
                'timestamp': time.time(),
            }}) + '\\n')
    except OSError:
        pass


class _CallableModule(types.ModuleType):
    # Make `await <skill>(...)` shorthand for `await <skill>.run(...)`.
    # __call__ is looked up on the type, not the instance, so the
    # override has to live on the class.
    async def __call__(self, *args, **kwargs):
        return await self.run(*args, **kwargs)


def _wrap_callable(mod, log_source):
    # log_source: 'python' for skills (logged to programmatic_tool_calls.jsonl),
    # None for rlm (already aggregated via Session.aggregate_child_metrics).
    wrapped = _CallableModule(mod.__name__)
    wrapped.__dict__.update(mod.__dict__)
    if log_source is not None:
        _original_run = wrapped.run
        @functools.wraps(_original_run)
        async def _logged_run(*args, **kwargs):
            _log_programmatic_call(mod.__name__, log_source)
            return await _original_run(*args, **kwargs)
        wrapped.run = _logged_run
    # Mirror run's signature and docstring onto the module so
    # `inspect.signature(<skill>)` and `help(<skill>)` expose the real API
    # surface instead of `_CallableModule.__call__`'s `(*args, **kwargs)`
    # and the file-level module docstring.
    wrapped.__signature__ = inspect.signature(wrapped.run)
    wrapped.__doc__ = wrapped.run.__doc__
    sys.modules[mod.__name__] = wrapped
    return wrapped


for _name in {skill_names!r}:
    globals()[_name] = _wrap_callable(__import__(_name), 'python')

if {allow_recursion!r}:
    globals()['rlm'] = _wrap_callable(__import__('rlm'), None)
"""
        self._execute_silent(setup_code)

    def _execute_silent(self, code: str):
        """Execute code without capturing output (for setup)."""
        self._kc.execute(code, silent=True)
        self._kc.get_shell_msg(timeout=30)

    def _wait_for_idle(self, timeout: float) -> bool:
        """Wait briefly for the kernel to report an idle state."""
        deadline = time.monotonic() + timeout
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return False
            try:
                msg = self._kc.get_iopub_msg(timeout=remaining)
            except Empty:
                return False
            if (
                msg["msg_type"] == "status"
                and msg["content"].get("execution_state") == "idle"
            ):
                return True

    def restart_kernel(self):
        """Restart the kernel and restore the initial REPL state."""
        if self._kc:
            self._kc.stop_channels()
        self._km.restart_kernel(now=True)
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _interrupt_and_recover(self):
        """Interrupt the running cell and restart the kernel if needed."""
        self._km.interrupt_kernel()
        if not self._wait_for_idle(timeout=2):
            self.restart_kernel()

    def execute(self, code: str, timeout: int | None = None) -> str:
        """Execute code and return combined output."""
        with self._lock:
            return self._execute_locked(code, timeout)

    def _execute_locked(self, code: str, timeout: int | None) -> str:
        msg_id = self._kc.execute(code)
        deadline = None if timeout is None else time.monotonic() + timeout

        outputs: list[str] = []
        try:
            while True:
                if deadline is None:
                    wait_timeout = None
                else:
                    wait_timeout = deadline - time.monotonic()
                    if wait_timeout <= 0:
                        self._interrupt_and_recover()
                        outputs.append(
                            f"\n[execution timed out after {timeout}s and was interrupted]"
                        )
                        break
                try:
                    msg = self._kc.get_iopub_msg(timeout=wait_timeout)
                except Empty:
                    self._interrupt_and_recover()
                    outputs.append(
                        f"\n[execution timed out after {timeout}s and was interrupted]"
                    )
                    break

                if msg["parent_header"].get("msg_id") != msg_id:
                    continue

                msg_type = msg["msg_type"]
                content = msg["content"]

                if msg_type == "stream":
                    outputs.append(content["text"])
                elif msg_type == "execute_result":
                    text = content.get("data", {}).get("text/plain", "")
                    if text:
                        outputs.append(text + "\n")
                elif msg_type == "error":
                    tb = "\n".join(content.get("traceback", []))
                    tb = _ANSI_RE.sub("", tb)
                    outputs.append(tb)
                elif msg_type == "status" and content["execution_state"] == "idle":
                    break
        finally:
            try:
                self._kc.get_shell_msg(timeout=5)
            except Exception:
                pass

        return "".join(outputs)

    def shutdown(self):
        """Stop the kernel."""
        if self._kc:
            self._kc.stop_channels()
            self._kc = None
        if self._km:
            self._km.shutdown_kernel(now=True)
            self._km = None
