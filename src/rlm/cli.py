"""CLI entry point: an Agent Client Protocol agent over stdio."""

from __future__ import annotations

import asyncio
import resource

# Each agent's kernel holds several ZeroMQ sockets, so a population of a few dozen agents
# outgrows the usual 1,024-descriptor soft limit.
FD_LIMIT = 65536


def raise_fd_limit(target: int = FD_LIMIT) -> None:
    """Raise RLIMIT_NOFILE to `target` (lifting the hard limit when privileged) so the
    kernels this process starts inherit it."""
    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    if hard != resource.RLIM_INFINITY and hard < target:
        try:
            resource.setrlimit(resource.RLIMIT_NOFILE, (target, target))
            return
        except (ValueError, OSError):  # unprivileged: the hard limit is the ceiling
            target = hard
    if soft != resource.RLIM_INFINITY and soft < target:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target, hard))


def main():
    import argparse

    parser = argparse.ArgumentParser(
        prog="rlm",
        description="A minimalistic recursive agent, served over the Agent Client Protocol.",
    )
    parser.add_argument(
        "--acp",
        action="store_true",
        help="Serve as an Agent Client Protocol agent over stdio (the only mode)",
    )
    args = parser.parse_args()
    if not args.acp:
        parser.error("rlm runs only as an ACP agent: use `rlm --acp`")
    from rlm.acp import serve_acp

    raise_fd_limit()
    asyncio.run(serve_acp())


if __name__ == "__main__":
    main()
