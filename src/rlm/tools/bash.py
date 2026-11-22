"""Native ``bash`` builtin tool.

Runs one shell command per call in a fresh subshell, like a plain bash agent.
Selected via the runtime contract's ``builtin_tools`` (the default tool set is
ipython alone).
"""

from __future__ import annotations

import codecs
import io
import os
import select
import selectors
import signal
import struct
import sys
import termios
import time
import contextlib
import fcntl
import subprocess
from threading import Event, Thread
from typing import Any

from rlm.tools.base import ToolContext, ToolOutcome
from rlm.tools.git_block import find_blocked_command, refusal
from rlm.tools.output import OUTPUT_KEEP_BYTES, OutputCapture, OutputSize

_OWN_GROUP = (
    {"process_group": 0} if sys.version_info >= (3, 11) else {"preexec_fn": os.setpgrp}
)
"""A command leads a process group in its caller's session: a timeout or a cancel stops
the command and what it started, a kernel restart or shutdown stops a command the kernel
ran, and a daemon, in a session of its own, outlives both."""

_BACKGROUND_NOTE = (
    "[a process started by this command is still running and holds its output; later"
    " output is not shown]"
)
"""Ends a result whose output a process the command left running still holds open."""

_EXIT_GRACE = 0.5
"""Seconds the output is still read after bash exits, for processes that finish writing
just after it: a process substitution, or a short job bash did not wait for."""

BASH_SCHEMA = {
    "type": "function",
    "function": {
        "name": "bash",
        "description": "Run a shell command and return its output (stdout and stderr).",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": "The shell command to run.",
                },
            },
            "required": ["command"],
        },
    },
}

EDIT_SCHEMA = {
    "type": "function",
    "function": {
        "name": "edit",
        "description": (
            "Replace a string in a file. old_str must occur exactly once in the file; "
            "the file is rewritten with old_str replaced by new_str."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Path of the file to edit."},
                "old_str": {
                    "type": "string",
                    "description": "Exact string to replace (must be unique).",
                },
                "new_str": {"type": "string", "description": "Replacement string."},
            },
            "required": ["path", "old_str", "new_str"],
        },
    },
}


def _pending(fd: int) -> int:
    """Bytes waiting in a pipe right now."""
    return struct.unpack("i", fcntl.ioctl(fd, termios.FIONREAD, b"\0" * 4))[0]


def _at_end(fd: int) -> bool:
    """Whether every process has closed a pipe whose waiting bytes were just read. Bytes
    that arrived since are a background job's, written after the grace, and dropped."""
    # poll, not select: select rejects descriptors at or above FD_SETSIZE (1024).
    poller = select.poll()
    poller.register(fd, select.POLLIN)
    if not poller.poll(0):
        return False
    return not os.read(fd, 65536)


def _discard_until_eof(fd: int) -> None:
    """Read a pipe a background job still holds to its end, so the job's writes neither
    block on a full pipe nor fail on a closed one once the call has returned."""
    with open(fd, "rb", buffering=0) as pipe:
        while pipe.read(65536):
            pass


def _drain_off_call(fd: int) -> None:
    """Start reading a pipe a background job still holds on a thread of its own. If that
    cannot start, the pipe is closed with the call instead, and the call still returns."""
    try:
        copy = os.dup(fd)
    except OSError:
        return
    try:
        Thread(target=_discard_until_eof, args=(copy,), daemon=True).start()
    except (OSError, RuntimeError):
        os.close(copy)


def run_bash(
    command: str,
    timeout: int,
    cwd: str | None = None,
    allow_git: bool | None = None,
    env: dict[str, str] | None = None,
    interrupt_event: Event | None = None,
) -> str:
    """Guarded ``bash -c`` execution shared by the bash tool and the bash skill."""
    return bash_outcome(command, timeout, cwd, allow_git, env, interrupt_event).content


def bash_outcome(
    command: str,
    timeout: int,
    cwd: str | None = None,
    allow_git: bool | None = None,
    env: dict[str, str] | None = None,
    interrupt_event: Event | None = None,
) -> ToolOutcome:
    """`run_bash`, with how much of a long output its content left out."""
    if not isinstance(command, str) or not command.strip():
        return ToolOutcome(content="Error: empty command")
    blocked = find_blocked_command(command, allow_git=allow_git)
    if blocked:
        return ToolOutcome(content=refusal(blocked))
    deadline = time.monotonic() + timeout
    streams = ("stdout", "stderr")
    # stdout and stderr each keep half of one result's bound.
    captures = {name: OutputCapture(OUTPUT_KEEP_BYTES // 2) for name in streams}
    decoders = {
        name: io.IncrementalNewlineDecoder(
            codecs.getincrementaldecoder("utf-8")(errors="replace"), translate=True
        )
        for name in streams
    }
    reason = ""
    held_open = False
    with (
        subprocess.Popen(
            ["bash", "-c", command],
            # The caller's stdin may be its terminal, where a background group stops on
            # a read, or its own protocol stream; a command reads end-of-file instead.
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd or None,
            env=env,
            **_OWN_GROUP,
        ) as proc,
        selectors.DefaultSelector() as selector,
    ):
        selector.register(proc.stdout, selectors.EVENT_READ, "stdout")
        selector.register(proc.stderr, selectors.EVENT_READ, "stderr")
        exited_at = None
        try:
            while selector.get_map():
                if interrupt_event is not None and interrupt_event.is_set():
                    reason = "[command interrupted]"
                    break
                now = time.monotonic()
                if exited_at is None and proc.poll() is not None:
                    exited_at = now
                if exited_at is not None:
                    # Bash has exited in time; its output is read until every process
                    # has closed it, or for a short grace after the exit.
                    wait = exited_at + _EXIT_GRACE - now
                    if wait <= 0:
                        # A pipe still open is held by a job left running. Read what
                        # is waiting in it now and return, as a terminal prompt does.
                        for key in list(selector.get_map().values()):
                            pending = _pending(key.fd)
                            while pending > 0:
                                data = os.read(key.fd, min(pending, 65536))
                                if not data:
                                    break
                                pending -= len(data)
                                captures[key.data].add(decoders[key.data].decode(data))
                            if _at_end(key.fd):
                                selector.unregister(key.fileobj)
                        held_open = bool(selector.get_map())
                        break
                else:
                    wait = deadline - now
                    if wait <= 0:
                        reason = f"Error: command timed out after {timeout}s"
                        break
                for key, _ in selector.select(min(wait, 0.1)):
                    data = os.read(key.fileobj.fileno(), 65536)
                    if not data:
                        selector.unregister(key.fileobj)
                        continue
                    captures[key.data].add(decoders[key.data].decode(data))
            if not reason:
                try:
                    proc.wait(timeout=max(0.01, deadline - time.monotonic()))
                except subprocess.TimeoutExpired:
                    reason = f"Error: command timed out after {timeout}s"
        finally:
            if proc.poll() is None:
                # A timeout or an interrupt: the command and what it started stop, and
                # their pipes close with the call.
                with contextlib.suppress(ProcessLookupError):
                    os.killpg(proc.pid, signal.SIGKILL)
                proc.wait()
            elif exited_at is not None:
                # Bash exited by itself, so what still holds its output is a job it put
                # in the background, and that job outlives the call. Its pipes are read
                # to their end off the call, never closed under it.
                for key in list(selector.get_map().values()):
                    _drain_off_call(key.fd)
    for name in streams:
        captures[name].add(decoders[name].decode(b"", final=True))
    (stdout, stdout_omitted), (stderr, stderr_omitted) = (
        captures[name].result() for name in streams
    )
    out = stdout + (("\n" + stderr) if stderr else "")
    if reason:
        out += "\n" + reason
    elif proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    if held_open and not reason:
        out += "\n" + _BACKGROUND_NOTE
    omitted = [size for size in (stdout_omitted, stderr_omitted) if size is not None]
    return ToolOutcome(
        content=out.strip() or "(no output)",
        omitted=sum(omitted, OutputSize()) if omitted else None,
    )


class BashTool:
    """One shell command per call, fresh subshell, cwd-anchored."""

    name = "bash"

    def schema(self) -> dict[str, Any]:
        return BASH_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        return bash_outcome(
            args.get("command", ""),
            context.exec_timeout,
            cwd=context.cwd,
            allow_git=context.allow_git,
            env=context.env,
            interrupt_event=context.interrupt_event,
        )


class EditTool:
    """Single-occurrence string replacement, mirroring the edit skill semantics."""

    name = "edit"

    def schema(self) -> dict[str, Any]:
        return EDIT_SCHEMA

    def execute(self, args: dict[str, Any], context: ToolContext) -> ToolOutcome:
        path, old, new = args.get("path"), args.get("old_str"), args.get("new_str")
        if not path or old is None or new is None:
            return ToolOutcome(content="Error: path, old_str and new_str are required")
        if context.cwd and not os.path.isabs(path):
            path = os.path.join(context.cwd, path)
        try:
            text = open(path, encoding="utf-8").read()
        except OSError as e:
            return ToolOutcome(content=f"Error: cannot read {path}: {e}")
        count = text.count(old)
        if count == 0:
            return ToolOutcome(content=f"Error: old_str not found in {path}")
        if count > 1:
            return ToolOutcome(
                content=f"Error: old_str occurs {count} times in {path}; must be unique"
            )
        open(path, "w", encoding="utf-8").write(text.replace(old, new, 1))
        return ToolOutcome(content=f"Edited {path} (1 replacement).")
