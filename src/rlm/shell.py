"""Supervisor-owned Bash jobs, available from an active IPython cell."""

from __future__ import annotations

import shlex

import builtins
from dataclasses import dataclass
from typing import Literal

from rlm import broker


MAX_READ_BYTES = 65_536


@dataclass(frozen=True)
class ShellResult:
    text: str
    exit_code: int | None
    job_id: str
    truncated: bool
    error: str | None
    timed_out: bool = False

    @property
    def ok(self) -> bool:
        """True only when the command ran to completion and exited 0."""
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def __getattr__(self, name: str):
        # Frozen dataclass: only unknown attributes reach here. Point common
        # JobHandle/subprocess habits at the right place instead of a bare error.
        if name in ("read", "info", "cancel", "wait"):
            raise AttributeError(
                f"ShellResult has no {name}(); run() already finished. Use .text/"
                ".exit_code, or `await rlm.shell.get(result.job_id)` for a JobHandle, "
                "or rlm.shell.start() for background work."
            )
        if name in ("stdout", "stderr", "output"):
            raise AttributeError(
                f"ShellResult has no .{name}; stdout and stderr are combined in .text."
            )
        if name in ("returncode", "status"):
            raise AttributeError(f"ShellResult has no .{name}; use .exit_code.")
        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}"
        )


@dataclass(frozen=True)
class JobInfo:
    id: str
    owner_id: str
    command: str
    cwd: str
    status: Literal[
        "starting", "running", "completed", "failed", "cancelled", "timed_out"
    ]
    created_at: float
    elapsed_seconds: float
    exit_code: int | None
    output_path: str
    output_bytes: int
    output_complete: bool
    output_truncated: bool
    error: str | None
    timeout: float | None = None

    @property
    def timed_out(self) -> bool:
        """True when the job's timeout killed it (status == "timed_out")."""
        return self.status == "timed_out"

    def handle(self) -> "JobHandle":
        """The JobHandle for this job; a JobInfo is a snapshot, not a handle."""
        return JobHandle(self.id)


@dataclass(frozen=True)
class JobOutput:
    text: str
    next_cursor: int
    done: bool
    truncated: bool


@dataclass(frozen=True)
class JobHandle:
    id: str

    async def info(self) -> JobInfo:
        return JobInfo(**await broker.agent_request("shell.info", job_id=self.id))

    async def read(self, *, cursor: int = 0, max_bytes: int = 16_384) -> JobOutput:
        """Read combined stdout/stderr using a byte cursor; reads do not consume output.

        Chunks decode as UTF-8 with replacement. A cursor may split a multibyte
        character; the output file retains the exact captured bytes. max_bytes is
        clamped to MAX_READ_BYTES (65536); continue with next_cursor for more.
        """
        return JobOutput(
            **await broker.agent_request(
                "shell.read",
                job_id=self.id,
                cursor=cursor,
                max_bytes=max(1, min(int(max_bytes), MAX_READ_BYTES)),
            )
        )

    async def cancel(self) -> JobInfo:
        """Terminate the job's process group and collect remaining output."""
        return JobInfo(**await broker.agent_request("shell.cancel", job_id=self.id))


def _command(command: str | builtins.list[str]) -> str:
    """Accept a Bash string or a non-empty argv list (joined with shell quoting)."""
    if isinstance(command, str):
        return command
    if (
        isinstance(command, builtins.list)
        and command
        and all(isinstance(c, str) for c in command)
    ):
        return shlex.join(command)
    raise TypeError("command must be a Bash string or a non-empty list of argv strings")


def _env(env: dict[str, str] | None) -> dict[str, str] | None:
    if env is None:
        return None
    if not isinstance(env, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in env.items()
    ):
        raise TypeError("env must map str names to str values")
    return dict(env)


async def setenv(
    variables: dict[str, str] | None = None, /, **more: str
) -> dict[str, str]:
    """Set environment variables for every later run()/start() of this agent.

    Returns the full persistent overlay. Per-call env= wins over it; the image's own
    environment sits underneath. Survives kernel restarts (supervisor-owned).
    """
    merged = {**(variables or {}), **more}
    return await broker.agent_request("shell.setenv", variables=_env(merged))


async def getenv() -> dict[str, str]:
    """The persistent overlay set with setenv()."""
    return await broker.agent_request("shell.getenv", variables=None)


def _timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number of seconds or None")
    if timeout <= 0:
        raise ValueError("timeout must be positive seconds")
    return float(timeout)


async def run(
    command: str | list[str],
    *,
    cwd: str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> ShellResult:
    """Wait for Bash and return combined output (up to 16 KiB) and its exit code.

    Nonzero exit codes are returned; startup/capture failures populate error.
    truncated indicates omitted output; get(result.job_id) can read more.
    timeout (seconds) kills the whole process group when exceeded: the result then
    has timed_out=True, exit_code None, and the output captured so far.
    Cancelling the cell stops waiting, not the job; list() can recover its ID.
    Only start() publishes a completion event to the inbox; run() returns directly.
    """
    return ShellResult(
        **await broker.agent_request(
            "shell.run",
            command=_command(command),
            cwd=cwd,
            timeout=_timeout(timeout),
            env=_env(env),
        )
    )


async def start(
    command: str | list[str],
    *,
    cwd: str | None = None,
    timeout: float | None = None,
    env: dict[str, str] | None = None,
) -> JobHandle:
    """Register a Bash job and return immediately. Defaults to the agent's cwd.

    Jobs survive cell completion. Completion posts an inbox event. No interactive
    stdin is provided; stdout and stderr share one captured stream. timeout
    (seconds) kills the process group when exceeded; the job's status becomes
    timed_out.
    """
    info = await broker.agent_request(
        "shell.start",
        command=_command(command),
        cwd=cwd,
        timeout=_timeout(timeout),
        env=_env(env),
    )
    return JobHandle(info["id"])


async def get(job_id: str) -> JobHandle:
    """Recover a job owned by this agent, including completed jobs."""
    info = await broker.agent_request("shell.info", job_id=job_id)
    return JobHandle(info["id"])


async def list() -> builtins.list[JobInfo]:
    """List this agent's jobs, including completed jobs."""
    return [JobInfo(**item) for item in await broker.agent_request("shell.list")]
