"""Shared CLI entry for Python tool packages.

A tool package exposes an async ``run(...)``. Its bash CLI
comes from ``tyro.cli(run)`` via a single shared entry point —
``[project.scripts]`` in each package's ``pyproject.toml`` just points at
``rlm.skill:cli``.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import sys
from pathlib import Path
from typing import Any

import tyro


def run_cli(func: Any, prog: str | None = None) -> None:
    """Run ``func`` as a CLI with arguments parsed from the signature.

    Async returns are awaited; non-``None`` return values are printed.
    """
    tyro._experimental_options["utf8_boxes"] = False
    result = tyro.cli(func, prog=prog or Path(sys.argv[0]).stem)
    if inspect.isawaitable(result):
        result = asyncio.run(result)
    if result is not None:
        print(result)


def cli() -> None:
    """Entry point for Python tool console scripts.

    Dispatches based on ``sys.argv[0]`` basename — invoked as ``say`` it
    imports the ``say`` module and runs its ``run`` via ``run_cli``.
    """
    name = Path(sys.argv[0]).stem
    module = importlib.import_module(name)
    run_cli(module.run, prog=name)
