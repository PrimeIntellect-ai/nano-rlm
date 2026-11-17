"""Long-lived services, available from an active IPython cell.

A Bash job (`rlm.shell.run`) belongs to your session: when the job ends, or when you finish,
its processes are stopped, including anything it started with `&` or `nohup`. A service is
for a process that must keep running after you finish, such as a server or daemon a task
asks you to leave up: it runs in its own session with its output appended to a log file, and
only `stop()` ends it.
"""

from __future__ import annotations

import builtins
from dataclasses import dataclass

from rlm import broker
from rlm.shell import _command, _env


@dataclass(frozen=True)
class Service:
    """A snapshot of a service when the object was returned; `get()` returns a fresh one."""

    name: str
    pid: int
    command: str
    cwd: str
    log: str
    running: bool
    exit_code: int | None
    started_at: float
    owner_id: str = ""

    async def tail(self, lines: int = 50) -> str:
        """Return the last `lines` lines of the service's log (stdout and stderr), at most 1000."""
        return await broker.agent_request(
            "services.tail", name=self.name, lines=int(lines)
        )

    async def stop(self, grace: float = 5.0) -> Service:
        """Send SIGTERM to the service's process group, then SIGKILL after `grace` seconds."""
        return Service(
            **await broker.agent_request(
                "services.stop", name=self.name, grace=float(grace)
            )
        )


async def start(
    name: str,
    command: str | builtins.list[str],
    *,
    cwd: str | None = None,
    env: dict[str, str] | None = None,
) -> Service:
    """Start `command` as a long-lived service called `name` and return a Service.

    The service runs in its own session, with stdin from /dev/null and stdout/stderr
    appended to its log file (`.log`), and it keeps running after your session ends. Use it
    for anything that must still be running after you finish; use rlm.shell.run for
    everything else. `name` is 1-64 letters, digits, `.`, `_` or `-`. Starting a name that is
    still running raises; stop it first. The environment and default cwd are those of
    rlm.shell.run. Starting returns at once: check the service came up with `tail()` or by
    connecting to it.
    """
    return Service(
        **await broker.agent_request(
            "services.start",
            name=name,
            command=_command(command),
            cwd=cwd,
            env=_env(env),
        )
    )


async def get(name: str) -> Service:
    """Return a fresh snapshot of the service called `name`. Works after kernel restarts."""
    return Service(**await broker.agent_request("services.get", name=name))


async def list() -> builtins.list[Service]:
    """List every service started in this sandbox, running or not."""
    return [Service(**item) for item in await broker.agent_request("services.list")]
