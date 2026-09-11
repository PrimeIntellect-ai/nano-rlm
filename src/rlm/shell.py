"""Supervisor-owned Bash jobs, available from an active IPython cell."""

from __future__ import annotations

import builtins
from dataclasses import dataclass
from typing import Literal

from rlm import broker


@dataclass(frozen=True)
class ShellResult:
    text: str
    exit_code: int | None
    job_id: str
    truncated: bool
    error: str | None


@dataclass(frozen=True)
class JobInfo:
    id: str
    owner_id: str
    command: str
    cwd: str
    status: Literal["starting", "running", "completed", "failed", "cancelled"]
    created_at: float
    elapsed_seconds: float
    exit_code: int | None
    output_path: str
    output_bytes: int
    output_complete: bool
    output_truncated: bool
    error: str | None


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
        character; the output file retains the exact captured bytes.
        """
        return JobOutput(
            **await broker.agent_request(
                "shell.read", job_id=self.id, cursor=cursor, max_bytes=max_bytes
            )
        )

    async def cancel(self) -> JobInfo:
        """Terminate the job's process group and collect remaining output."""
        return JobInfo(**await broker.agent_request("shell.cancel", job_id=self.id))


async def run(command: str, *, cwd: str | None = None) -> ShellResult:
    """Wait for Bash and return combined output (up to 16 KiB) and its exit code.

    Nonzero exit codes are returned; startup/capture failures populate error.
    truncated indicates omitted output; get(result.job_id) can read more.
    Cancelling the cell stops waiting, not the job; list() can recover its ID.
    Both run() and start() publish completion events to the inbox.
    """
    return ShellResult(
        **await broker.agent_request("shell.run", command=command, cwd=cwd)
    )


async def start(command: str, *, cwd: str | None = None) -> JobHandle:
    """Register a Bash job and return immediately. Defaults to the agent's cwd.

    Jobs survive cell completion. Completion posts an inbox event. No interactive
    stdin is provided; stdout and stderr share one captured stream.
    """
    info = await broker.agent_request("shell.start", command=command, cwd=cwd)
    return JobHandle(info["id"])


async def get(job_id: str) -> JobHandle:
    """Recover a job owned by this agent, including completed jobs."""
    info = await broker.agent_request("shell.info", job_id=job_id)
    return JobHandle(info["id"])


async def list() -> builtins.list[JobInfo]:
    """List this agent's jobs, including completed jobs."""
    return [JobInfo(**item) for item in await broker.agent_request("shell.list")]
