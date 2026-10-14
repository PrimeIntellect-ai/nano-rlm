"""CLI entry point: an Agent Client Protocol agent over stdio."""

from __future__ import annotations

import asyncio


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
    parser.add_argument(
        "--skill",
        action="append",
        default=[],
        metavar="PATH",
        help=(
            "A SKILL.md skill to list in every session's prompt: a skill folder or a "
            "folder of skill folders (repeatable)"
        ),
    )
    args = parser.parse_args()
    if not args.acp:
        parser.error("rlm runs only as an ACP agent: use `rlm --acp`")
    from rlm.acp import serve_acp

    asyncio.run(serve_acp(tuple(args.skill)))


if __name__ == "__main__":
    main()
