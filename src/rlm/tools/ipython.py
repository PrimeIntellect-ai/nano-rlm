"""Builtin IPython tool and persistent REPL implementation."""

from __future__ import annotations

import copy
import os
from queue import Empty
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections.abc import Mapping
from pathlib import Path
from typing import TYPE_CHECKING, Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_in_ipython, refusal
from rlm.tools.skills import discover_skills
from rlm.types import IpythonExecuted

if TYPE_CHECKING:
    from rlm.broker import BrokerEndpoint
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
_KERNEL_BASE_ENV_NAMES = {
    "CURL_CA_BUNDLE",
    "HOME",
    "LANG",
    "LOGNAME",
    "PATH",
    "REQUESTS_CA_BUNDLE",
    "SHELL",
    "SSL_CERT_FILE",
    "TERM",
    "TMPDIR",
    "TZ",
    "USER",
    "VIRTUAL_ENV",
}


def build_kernel_env(
    task_env: Mapping[str, str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    private_dir: str | None = None,
) -> dict[str, str]:
    """Build a minimal kernel environment plus explicitly supplied task variables."""
    source = os.environ if environ is None else environ
    explicit = dict(task_env or {})
    invalid_types = [
        key
        for key, value in explicit.items()
        if not isinstance(key, str) or not isinstance(value, str)
    ]
    if invalid_types:
        raise TypeError("kernel environment keys and values must be strings")
    kernel_env = {
        key: value
        for key, value in source.items()
        if key in _KERNEL_BASE_ENV_NAMES or key.startswith("LC_")
    }
    kernel_env.update(explicit)
    kernel_env["NO_COLOR"] = "1"
    if private_dir is not None:
        root = Path(private_dir)
        private_paths = {
            "IPYTHONDIR": root / "ipython",
            "JUPYTER_CONFIG_DIR": root / "jupyter-config",
            "JUPYTER_DATA_DIR": root / "jupyter-data",
            "JUPYTER_RUNTIME_DIR": root / "jupyter-runtime",
        }
        for path in private_paths.values():
            path.mkdir(mode=0o700, exist_ok=True)
        kernel_env.update({name: str(path) for name, path in private_paths.items()})
    return kernel_env


class IpythonTool:
    """Builtin tool handler for the persistent IPython session."""

    name = "ipython"

    def __init__(self, exec_timeout: int = 300) -> None:
        self.exec_timeout = exec_timeout

    def schema(self) -> dict[str, Any]:
        timeout = min(self.exec_timeout, IPYTHON_TIMEOUT_MAX_SECONDS)
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

        blocked = find_blocked_in_ipython(code, allow_git=context.allow_git)
        if blocked is not None:
            return ToolOutcome(
                content=refusal(blocked),
                metric_events=metric_events,
            )

        return ToolOutcome(
            content=context.repl.execute(code, timeout=timeout),
            metric_events=metric_events,
        )

    @staticmethod
    def _count_nonempty_lines(code: str) -> int:
        return sum(1 for line in code.splitlines() if line.strip())


class IPythonREPL:
    """Persistent IPython kernel communicating via Jupyter protocol."""

    def __init__(
        self,
        cwd: str,
        session: "Session | None" = None,
        kernel_env: Mapping[str, str] | None = None,
        depth: int | None = None,
        max_depth: int | None = None,
        broker_endpoint: BrokerEndpoint | None = None,
        exec_timeout: int | None = None,
        allow_git: bool | None = None,
    ):
        self.cwd = cwd
        self.session = session
        self.kernel_env = dict(kernel_env or {})
        self.depth = depth
        self.max_depth = max_depth
        self.broker_endpoint = broker_endpoint
        self.exec_timeout = exec_timeout
        self.allow_git = allow_git
        self._km = None
        self._kc = None
        self._ipc_dir = None
        self._lock = threading.Lock()
        self._interrupt_requested = threading.Event()

    def start(self):
        """Start the IPython kernel."""
        from jupyter_client import KernelManager

        # IPC instead of the default TCP (ipykernel >= 7.3 warns about
        # unencrypted TCP). The socket path must be absolute and short
        # (macOS caps Unix socket paths at 104 bytes), hence a temp dir.
        self._ipc_dir = tempfile.mkdtemp(prefix="rlm-ipc-")
        self._km = KernelManager(
            transport="ipc", ip=os.path.join(self._ipc_dir, "kernel")
        )
        self._km.kernel_spec.argv = [
            sys.executable,
            "-m",
            "ipykernel_launcher",
            "-f",
            "{connection_file}",
        ]
        self._km.kernel_spec.env = {}
        kernel_env = build_kernel_env(
            self.kernel_env,
            private_dir=self._ipc_dir,
        )
        launcher = shutil.which(sys.argv[0]) or os.path.abspath(sys.argv[0])
        launcher_dir = os.path.dirname(os.path.abspath(launcher))
        path_entries = kernel_env.get("PATH", "").split(os.pathsep)
        if launcher_dir not in path_entries:
            kernel_env["PATH"] = os.pathsep.join([launcher_dir, *path_entries])
        self._km.start_kernel(
            cwd=self.cwd,
            env=kernel_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        self._kc = self._km.client()
        self._kc.start_channels()
        self._kc.wait_for_ready(timeout=30)
        self._inject_startup()

    def _inject_startup(self):
        """Set up kernel: cwd, env vars, nest_asyncio, skill pre-imports."""
        session_dir = str(self.session.dir) if self.session else None
        depth = (
            int(os.environ.get("RLM_DEPTH", "0")) if self.depth is None else self.depth
        )
        max_depth = (
            int(os.environ.get("RLM_MAX_DEPTH", "0"))
            if self.max_depth is None
            else self.max_depth
        )
        allow_recursion = depth < max_depth and self.broker_endpoint is not None
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
if {self.exec_timeout!r} is not None:
    os.environ['RLM_EXEC_TIMEOUT'] = str({self.exec_timeout!r})
if {self.allow_git!r} is not None:
    os.environ['RLM_ALLOW_GIT'] = '1' if {self.allow_git!r} else '0'

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


def _wrap_callable(mod, log_source, register=True):
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
    if register:
        sys.modules[mod.__name__] = wrapped
    return wrapped


if {bool(self.broker_endpoint)!r}:
    import rlm.broker as _rlm_broker
    _rlm_broker.configure(_rlm_broker.BrokerEndpoint(
        {self.broker_endpoint.socket_path if self.broker_endpoint else None!r},
        {self.broker_endpoint.capability if self.broker_endpoint else None!r},
    ))

for _name in {skill_names!r}:
    _module = __import__(_name)
    _source = None if getattr(_module, '__rlm_brokered__', False) else 'python'
    globals()[_name] = _wrap_callable(_module, _source)

if {allow_recursion!r}:
    import rlm as _rlm_package
    globals()['rlm'] = _wrap_callable(_rlm_package, None)
"""
        self._execute_silent(setup_code)

    def set_broker_scope(self, scope_id: str | None) -> None:
        """Set the active recursive-call scope inside the kernel."""
        if self.broker_endpoint is not None:
            self._execute_silent(f"_rlm_broker.set_scope({scope_id!r})")

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

    def interrupt(self):
        """Request interruption and recovery of a running cell."""
        self._interrupt_requested.set()
        if self._km:
            self._km.interrupt_kernel()

    def finish_interrupt(self):
        """Clear an interrupt after the execution worker has settled."""
        self._interrupt_requested.clear()

    def _interrupt_and_recover(self):
        """Interrupt the running cell and restart the kernel if needed."""
        self._km.interrupt_kernel()
        if not self._wait_for_idle(timeout=2):
            self.restart_kernel()

    def execute(self, code: str, timeout: int | None = None) -> str:
        """Execute code and return combined output."""
        with self._lock:
            try:
                if self._interrupt_requested.is_set():
                    return ""
                return self._execute_locked(code, timeout)
            finally:
                self._interrupt_requested.clear()

    def _execute_locked(self, code: str, timeout: int | None) -> str:
        msg_id = self._kc.execute(code)
        deadline = None if timeout is None else time.monotonic() + timeout

        outputs: list[str] = []
        try:
            while True:
                if self._interrupt_requested.is_set():
                    self._interrupt_and_recover()
                    break
                if deadline is None:
                    wait_timeout = 0.1
                else:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        self._interrupt_and_recover()
                        outputs.append(
                            f"\n[execution timed out after {timeout}s and was interrupted]"
                        )
                        break
                    wait_timeout = min(remaining, 0.1)
                try:
                    msg = self._kc.get_iopub_msg(timeout=wait_timeout)
                except Empty:
                    continue

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
                timeout = 0.1 if self._interrupt_requested.is_set() else 5
                self._kc.get_shell_msg(timeout=timeout)
            except Exception:
                pass

        return "".join(outputs)

    def shutdown(self):
        """Stop the kernel."""
        if self._kc:
            self._kc.stop_channels()
            self._kc = None
        if self._km and self._km.has_kernel:
            self._km.shutdown_kernel(now=True)
        if self._km:
            self._km = None
        if self._ipc_dir:
            shutil.rmtree(self._ipc_dir, ignore_errors=True)
            self._ipc_dir = None
