"""Per-agent process identities for population episodes (process_isolation="per_agent").

Each agent's kernel and shell commands run under the agent's own uid in a shared agents
group, so agents cannot signal each other or the supervisor (which keeps running as root),
and cannot read each other's session directories. The workspace stays group-writable for
collaboration. Granting these identities needs a root host.
"""

from __future__ import annotations

import os
import stat
from dataclasses import dataclass
from pathlib import Path

AGENT_ID_BASE = 61_000
"""Agent n runs as uid AGENT_ID_BASE + n; every agent shares gid AGENT_ID_BASE."""


@dataclass(frozen=True)
class ProcessIdentity:
    uid: int
    gid: int
    home: str

    def popen_kwargs(self) -> dict:
        return {"user": self.uid, "group": self.gid, "extra_groups": [], "umask": 0o002}

    def env(self) -> dict[str, str]:
        # Repositories in the shared workspace belong to other agents' uids.
        return {
            "HOME": self.home,
            "GIT_CONFIG_COUNT": "1",
            "GIT_CONFIG_KEY_0": "safe.directory",
            "GIT_CONFIG_VALUE_0": "*",
        }


def require_root() -> None:
    if os.geteuid() != 0:
        raise RuntimeError(
            "process_isolation='per_agent' needs the RLM host to run as root, to start "
            "each agent's processes under its own uid"
        )


def agent_identity(index: int, session_dir: Path) -> ProcessIdentity:
    """Own the agent's session directory (with a home inside) as the agent, privately."""
    identity = ProcessIdentity(
        uid=AGENT_ID_BASE + index,
        gid=AGENT_ID_BASE,
        home=str(session_dir / "home"),
    )
    Path(identity.home).mkdir(exist_ok=True)
    for path in (session_dir, *session_dir.iterdir()):
        if path.is_dir():
            os.chown(path, identity.uid, identity.gid)
    session_dir.chmod(0o700)
    return identity


def share_with_agents(root: Path) -> None:
    """Make a workspace group-writable for the agents group; new files keep the group."""
    for directory, _, files in os.walk(root):
        os.chown(directory, -1, AGENT_ID_BASE)
        os.chmod(directory, stat.S_IMODE(os.stat(directory).st_mode) | 0o2070)
        for name in files:
            path = os.path.join(directory, name)
            if not os.path.islink(path):
                os.chown(path, -1, AGENT_ID_BASE)
                os.chmod(path, stat.S_IMODE(os.stat(path).st_mode) | 0o060)


def own_kernel_dir(directory: str, identity: ProcessIdentity) -> None:
    """Give a kernel's private IPC directory (sockets, connection file) to its agent."""
    for path, dirs, files in os.walk(directory):
        os.chown(path, identity.uid, identity.gid)
        for name in [*dirs, *files]:
            os.chown(os.path.join(path, name), identity.uid, identity.gid)
