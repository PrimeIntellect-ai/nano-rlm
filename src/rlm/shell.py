"""Supervisor-owned Bash jobs, available from an active IPython cell."""

from __future__ import annotations

import shlex

import builtins
from dataclasses import dataclass
from typing import Literal

from rlm import broker


MAX_READ_BYTES = 65_536


@dataclass(frozen=True)
class ShellJob:
    """A supervisor-owned Bash job: the snapshot run() or result() returned, plus the
    methods that wait for, read, or cancel the job. Fields describe the job at the moment
    the object was returned; `await job.result()` returns a fresh, finished snapshot."""

    id: str
    text: str = ""
    exit_code: int | None = None
    truncated: bool = False
    error: str | None = None
    timed_out: bool = False
    running: bool = True

    @property
    def ok(self) -> bool:
        """True only when the command ran to completion and exited 0."""
        return (
            self.exit_code == 0
            and not self.timed_out
            and not self.running
            and self.error is None
        )

    async def result(self, wait: float | None = None, **rejected) -> ShellJob:
        """Wait for the job to finish and return its finished snapshot.

        Blocks inside the cell for up to `wait` seconds (default and cap: 300). A job
        still running afterwards comes back with running=True and its output so far;
        call result() again later. Repeatable and non-consuming: on a finished job it
        returns the same result every time.
        """
        _reject_kwargs("result", rejected)
        return ShellJob(
            **await broker.agent_request(
                "shell.result", job_id=self.id, wait=_wait(wait)
            )
        )

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

    def __getattr__(self, name: str):
        # Frozen dataclass: only unknown attributes reach here. Point common
        # subprocess habits at the right place instead of a bare error.
        if name in ("stdout", "stderr", "output"):
            raise AttributeError(
                f"ShellJob has no .{name}; stdout and stderr are combined in .text."
            )
        if name in ("returncode", "status"):
            raise AttributeError(f"ShellJob has no .{name}; use .exit_code.")
        if name == "wait":
            raise AttributeError(
                "ShellJob has no wait(); `await job.result()` waits for the job."
            )
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


@dataclass(frozen=True)
class JobOutput:
    text: str
    next_cursor: int
    done: bool
    truncated: bool


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


def _wait(wait: float | None) -> float | None:
    if wait is None:
        return None
    if isinstance(wait, bool) or not isinstance(wait, (int, float)):
        raise TypeError("wait must be a number of seconds or None")
    if wait < 0:
        raise ValueError("wait must be a non-negative number of seconds")
    return float(wait)


def _timeout(timeout: float | None) -> float | None:
    if timeout is None:
        return None
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)):
        raise TypeError("timeout must be a number of seconds or None")
    if timeout <= 0:
        raise ValueError("timeout must be positive seconds")
    return float(timeout)


def _reject_kwargs(name: str, rejected: dict) -> None:
    """Name the replacement for an argument this API used to have: an unknown kwarg
    costs the agent a cell, a bare TypeError costs it another."""
    if not rejected:
        return
    hints = {"background": "use wait=0 to return at once"}
    key = next(iter(rejected))
    raise TypeError(
        f"{name}() got an unexpected keyword argument {key!r}: "
        + hints.get(key, "see help(rlm.shell.run)")
    )


async def run(
    command: str | list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
    wait: float | None = None,
    timeout: float | None = None,
    **rejected,
) -> ShellJob:
    """Run Bash under the supervisor and return a ShellJob.

    wait is how long to wait for the command, in seconds: default 10, 0 returns at once,
    values above 300 are capped at 300. A command that finished has running=False,
    exit_code, ok, text (combined output, up to 16 KiB: first and last 8 KiB around a
    marker naming the omitted bytes), truncated, error and timed_out. A command still
    going when the wait ends comes back with running=True, exit_code None and the output
    so far, and keeps running; `await job.result()` waits (up to 300 s per call) for the
    finished result. timeout, if given, kills the process group after that many seconds
    (timed_out=True, exit_code None); `await job.cancel()` stops a job at any time.
    Cancelling the cell stops waiting, not the job; list() can recover its ID. A job
    handed back running posts a quiet shell.completed inbox event when it ends (it wakes
    the native wait tool but is not counted in the unread notice); finished results post
    nothing.
    """
    _reject_kwargs("run", rejected)
    return ShellJob(
        **await broker.agent_request(
            "shell.run",
            command=_command(command),
            cwd=cwd,
            wait=_wait(wait),
            timeout=_timeout(timeout),
            env=_env(env),
        )
    )


async def get(job_id: str) -> ShellJob:
    """Recover a job owned by this agent as a ShellJob snapshot (finished jobs carry
    their result; running ones have running=True). Works after kernel restarts."""
    return ShellJob(
        **await broker.agent_request("shell.result", job_id=job_id, wait=0.0)
    )


async def list() -> builtins.list[JobInfo]:
    """List this agent's jobs, including completed jobs."""
    return [JobInfo(**item) for item in await broker.agent_request("shell.list")]
