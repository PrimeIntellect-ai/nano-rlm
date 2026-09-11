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
            "Execute Python in a persistent kernel, including top-level await. "
            "Use the pre-imported rlm API to manage agents, Bash jobs, inboxes, and subscriptions "
            "as described in the runtime guide. For quick blocking shell commands use !command; "
            "for multiline Bash use %%bash on the first line, with Bash through the end of the cell. "
            "Resume Python in a new cell. Use rlm.shell.run for background Bash jobs. "
            "The blocking magics belong to the kernel's lifecycle; supervisor jobs survive its restart. "
            "Variables persist across cells and compaction, but are lost on kernel restart. "
            "Read recovery notices and reconstruct state before retrying interrupted work."
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


class _KernelDied(RuntimeError):
    def __init__(self, output: str = ""):
        self.output = output
        super().__init__("IPython kernel exited")


MAX_RECOVERY_ATTEMPTS = 3


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
        self._scope_id: str | None = None
        self._recovery_attempts = 0
        self._recovery_failed = False
        self._recovery_notices: list[str] = []

    def start(self):
        """Start the IPython kernel."""
        from jupyter_client import KernelManager

        # IPC instead of the default TCP (ipykernel >= 7.3 warns about
        # unencrypted TCP). The socket path must be absolute and short
        # (macOS caps Unix socket paths at 104 bytes), hence a temp dir.
        self._ipc_dir = tempfile.mkdtemp(prefix="rlm-ipc-")
        self._km = KernelManager(
            transport="ipc", ip=os.path.join(self._ipc_dir, "kernel"), autorestart=False
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
        # Pip-installed skills + the MCP-tool modules generated into the session dir (rlm.mcp);
        # the session dir goes on the kernel's sys.path so those import by name.
        skill_names = discover_skills(self.session.dir if self.session else None)

        setup_code = f"""\
import os, sys, asyncio, types, json, time, functools, inspect
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
    # Brokered skills are counted by the supervisor.
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

import rlm
"""
        self._execute_silent(setup_code)

    def set_broker_scope(self, scope_id: str | None) -> None:
        """Set the scope to install when the next cell starts executing."""
        self._scope_id = scope_id

    def take_recovery_notices(self) -> list[str]:
        notices, self._recovery_notices = self._recovery_notices, []
        return notices

    def _execute_silent(self, code: str):
        """Execute setup and verify the matching reply succeeded."""
        msg_id = self._kc.execute(code, silent=True)
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            if not self._km.is_alive():
                raise _KernelDied()
            try:
                reply = self._kc.get_shell_msg(timeout=0.1)
            except Empty:
                continue
            if reply["parent_header"].get("msg_id") != msg_id:
                continue
            if reply["content"].get("status") != "ok":
                raise RuntimeError(
                    f"IPython setup failed: {reply['content'].get('ename', 'unknown error')}"
                )
            return
        raise TimeoutError("IPython setup did not respond within 30 seconds")

    def _recover(self, reason: str) -> bool:
        self._recovery_failed = True
        if self._recovery_attempts >= MAX_RECOVERY_ATTEMPTS:
            self._recovery_notices.append(
                "Supervisor: IPython is unavailable: the three-attempt recovery limit was reached. "
                "The cell was not replayed. Conversation and supervisor-owned resources remain available."
            )
            return False
        self._recovery_attempts += 1
        try:
            self.restart_kernel()
        except Exception as exc:
            self._recovery_notices.append(
                f"Supervisor: IPython restart failed (attempt {self._recovery_attempts}/3): {type(exc).__name__}: {exc}. "
                "The cell was not replayed. A later IPython call can retry within the recovery limit. "
                "Conversation and supervisor-owned resources remain available."
            )
            return False
        self._recovery_failed = False
        self._recovery_notices.append(
            f"Supervisor: Your IPython kernel {reason} and has been restarted. "
            "Python variables, imports, and in-kernel tasks were lost. Your conversation, inbox, "
            "agents, shell jobs, and subscriptions remain available. Recreate the variables you need and recover "
            "handles through rlm.agent.list/get, rlm.shell.list/get, and rlm.watch.list/get. Inbox read state is unchanged. "
            "The interrupted cell may have produced partial side effects; it was not replayed. "
            "Inspect existing resources before retrying a spawn, send, or shell command."
        )
        return True

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
        if self._km and self._km.is_alive():
            self._km.interrupt_kernel()

    def finish_interrupt(self):
        """Clear an interrupt after the execution worker has settled."""
        self._interrupt_requested.clear()

    def _interrupt_and_recover(self):
        """Interrupt the cell; report any restart that loses Python state."""
        if not self._km.is_alive():
            self._recover("crashed")
            return
        self._km.interrupt_kernel()
        if not self._wait_for_idle(timeout=2):
            self._recover("did not respond to interruption")

    def execute(self, code: str, timeout: int | None = None) -> str:
        """Execute once; a lost kernel is recovered without replaying the cell."""
        with self._lock:
            try:
                if self._interrupt_requested.is_set():
                    return ""
                if self._recovery_failed or not self._km.is_alive():
                    self._recover("was unavailable")
                    return "[cell not executed: IPython was unavailable; see recovery notice]"
                if self.broker_endpoint is not None:
                    self._execute_silent(f"_rlm_broker.set_scope({self._scope_id!r})")
                result = self._execute_locked(code, timeout)
                if not self._recovery_notices:
                    self._recovery_attempts = 0
                return result
            except _KernelDied as exc:
                self._recover("crashed")
                return exc.output + "\n[cell interrupted by kernel exit; not replayed]"
            finally:
                self._interrupt_requested.clear()

    def _execute_locked(self, code: str, timeout: int | None) -> str:
        client = self._kc
        msg_id = client.execute(code)
        deadline = None if timeout is None else time.monotonic() + timeout

        outputs: list[str] = []
        try:
            while True:
                if not self._km.is_alive():
                    raise _KernelDied("".join(outputs))
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
                if self._kc is client and self._km.is_alive():
                    client.get_shell_msg(timeout=timeout)
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
