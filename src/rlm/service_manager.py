"""Supervisor side of `rlm.services`: processes meant to outlive the agent's session.

A Bash job belongs to the session that started it, and its whole process group is stopped
when it ends (see shell_jobs). A service is the opposite: it starts in its own session, so
no job's process-group kill reaches it; its output goes to a log file rather than a pipe,
so it keeps running after the supervisor exits; and nothing here stops it except stop().
Each service's record is a JSON file in the services directory, so services can be listed
and stopped after a kernel or supervisor restart.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import signal
import subprocess
import threading
import time
from pathlib import Path

NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}")
TAIL_MAX_LINES = 1000
TAIL_MAX_BYTES = 256 * 1024


def _check_name(name: str) -> None:
    if not NAME.fullmatch(name):
        raise ValueError(
            f"invalid service name {name!r}: use 1-64 letters, digits, '.', '_' or '-', "
            "starting with a letter or digit"
        )


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        state = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[0]
    except (OSError, IndexError):
        return True
    return state != "Z"


def _signal_group(pid: int, sig: int) -> None:
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


class ServiceManager:
    def __init__(self, directory: Path) -> None:
        self.directory = directory
        # Processes this supervisor started: their exit code is known once they exit.
        self._procs: dict[str, subprocess.Popen] = {}

    def _path(self, name: str) -> Path:
        return self.directory / f"{name}.json"

    def _load(self, name: str) -> dict:
        _check_name(name)
        path = self._path(name)
        if not path.exists():
            raise KeyError(
                f"no service named {name!r}; await rlm.services.list() shows the services that exist"
            )
        return json.loads(path.read_text())

    def _snapshot(self, record: dict) -> dict:
        proc = self._procs.get(record["name"])
        exit_code = (
            proc.poll() if proc is not None and proc.pid == record["pid"] else None
        )
        return {
            **record,
            "running": exit_code is None and _alive(record["pid"]),
            "exit_code": exit_code,
        }

    def start(
        self, *, name: str, command: str, cwd: str, env: dict[str, str], owner_id: str
    ) -> dict:
        _check_name(name)
        self.directory.mkdir(parents=True, exist_ok=True)
        path = self._path(name)
        if path.exists():
            current = self._snapshot(json.loads(path.read_text()))
            if current["running"]:
                raise ValueError(
                    f"service {name!r} is already running (pid {current['pid']}); "
                    "stop it first or choose another name"
                )
        log = self.directory / f"{name}.log"
        with open(os.devnull, "rb") as stdin, open(log, "ab") as output:
            proc = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-c", command],
                cwd=cwd,
                env=env,
                stdin=stdin,
                stdout=output,
                stderr=subprocess.STDOUT,
                start_new_session=True,
                close_fds=True,
            )
        self._procs[name] = proc
        # Reap it when it exits so a finished service never lingers as the supervisor's zombie.
        threading.Thread(
            target=proc.wait, name=f"rlm-service-{name}", daemon=True
        ).start()
        record = {
            "name": name,
            "pid": proc.pid,
            "command": command,
            "cwd": cwd,
            "log": str(log),
            "started_at": time.time(),
            "owner_id": owner_id,
        }
        path.write_text(json.dumps(record))
        return self._snapshot(record)

    def get(self, name: str) -> dict:
        return self._snapshot(self._load(name))

    def list(self) -> list[dict]:
        if not self.directory.exists():
            return []
        return [
            self._snapshot(json.loads(p.read_text()))
            for p in sorted(self.directory.glob("*.json"))
        ]

    def tail(self, name: str, lines: int) -> str:
        record = self._load(name)
        lines = max(1, min(int(lines), TAIL_MAX_LINES))
        try:
            with open(record["log"], "rb") as fh:
                fh.seek(max(0, os.fstat(fh.fileno()).st_size - TAIL_MAX_BYTES))
                data = fh.read()
        except FileNotFoundError:
            return ""
        return "\n".join(data.decode(errors="replace").splitlines()[-lines:])

    async def stop(self, name: str, grace: float) -> dict:
        record = self._load(name)
        if self._snapshot(record)["running"]:
            _signal_group(record["pid"], signal.SIGTERM)
            deadline = time.monotonic() + max(0.0, grace)
            while time.monotonic() < deadline and self._snapshot(record)["running"]:
                await asyncio.sleep(0.05)
            if self._snapshot(record)["running"]:
                _signal_group(record["pid"], signal.SIGKILL)
                for _ in range(40):
                    if not self._snapshot(record)["running"]:
                        break
                    await asyncio.sleep(0.05)
        return self._snapshot(record)
