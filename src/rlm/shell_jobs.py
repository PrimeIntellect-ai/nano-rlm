"""Bash process ownership and bounded output capture outside execution kernels."""

from __future__ import annotations

import asyncio
import json
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable

from rlm.shell import JobInfo

MAX_ACTIVE_JOBS = 32
MAX_JOBS = 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
DRAIN_SECONDS = 2.0


@dataclass
class ShellJob:
    info: JobInfo
    source_request_id: str | None
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    cancel_requested: bool = False
    task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            **asdict(self.info),
            "elapsed_seconds": (self.finished or time.monotonic()) - self.started,
        }

    def update(self, **values) -> None:
        self.info = JobInfo(**{**asdict(self.info), **values})


class ShellJobs:
    def __init__(self, publish: Callable[[ShellJob], None]):
        self.jobs: dict[str, ShellJob] = {}
        self.publish = publish

    def start(
        self,
        *,
        owner_id: str,
        command: str,
        cwd: str,
        directory: Path,
        env: dict[str, str],
        source_request_id: str | None,
    ) -> dict:
        if len(self.jobs) >= MAX_JOBS:
            raise RuntimeError("shell job limit reached")
        if sum(job.finished is None for job in self.jobs.values()) >= MAX_ACTIVE_JOBS:
            raise RuntimeError("active shell job limit reached")
        job_id = uuid.uuid4().hex
        directory = directory / "jobs" / job_id
        directory.mkdir(parents=True)
        output = directory / "output.bin"
        output.touch(exist_ok=False)
        job = ShellJob(
            JobInfo(
                id=job_id,
                owner_id=owner_id,
                command=command,
                cwd=cwd,
                status="starting",
                created_at=time.time(),
                elapsed_seconds=0,
                exit_code=None,
                output_path=str(output),
                output_bytes=0,
                output_complete=False,
                output_truncated=False,
                error=None,
            ),
            source_request_id,
        )
        self.jobs[job_id] = job
        job.task = asyncio.create_task(self._run(job, env))
        return job.snapshot()

    def get(self, owner_id: str, job_id: str) -> ShellJob:
        job = self.jobs.get(job_id)
        if job is None or job.info.owner_id != owner_id:
            raise PermissionError("unknown job or job is not owned by this agent")
        return job

    def read(self, job: ShellJob, cursor: int, max_bytes: int) -> dict:
        if cursor > job.info.output_bytes:
            raise ValueError("cursor is beyond captured output")
        with open(job.info.output_path, "rb") as stream:
            stream.seek(cursor)
            data = stream.read(max_bytes)
        return {
            "text": data.decode("utf-8", errors="replace"),
            "next_cursor": cursor + len(data),
            "done": job.finished is not None
            and cursor + len(data) == job.info.output_bytes,
            "truncated": job.info.output_truncated,
        }

    async def cancel(self, job: ShellJob) -> dict:
        job.cancel_requested = True
        await asyncio.shield(job.task)
        return job.snapshot()

    async def close(self, owner_id: str | None = None) -> None:
        jobs = [
            job
            for job in self.jobs.values()
            if owner_id is None or job.info.owner_id == owner_id
        ]
        for job in jobs:
            job.cancel_requested = True
        if jobs:
            await asyncio.gather(*(asyncio.shield(job.task) for job in jobs))

    @staticmethod
    def _signal(process: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(process.pid, sig)
        except ProcessLookupError:
            pass

    async def _run(self, job: ShellJob, env: dict[str, str]) -> None:
        process = None
        try:
            if job.cancel_requested:
                job.update(status="cancelled", output_complete=True)
                return
            process = subprocess.Popen(
                ["/bin/bash", "--noprofile", "--norc", "-c", job.info.command],
                cwd=job.info.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            os.set_blocking(process.stdout.fileno(), False)
            job.update(status="running")
            exit_at = None
            cancel_at = None
            eof = False
            with open(job.info.output_path, "ab", buffering=0) as output:
                while True:
                    now = time.monotonic()
                    if job.cancel_requested and cancel_at is None:
                        self._signal(process, signal.SIGTERM)
                        cancel_at = now
                    if cancel_at is not None and now - cancel_at >= 0.2:
                        self._signal(process, signal.SIGKILL)
                    if not eof:
                        try:
                            data = os.read(process.stdout.fileno(), 65_536)
                        except BlockingIOError:
                            data = None
                        if data == b"":
                            eof = True
                        elif data:
                            retained = data[
                                : max(0, MAX_OUTPUT_BYTES - job.info.output_bytes)
                            ]
                            output.write(retained)
                            job.update(
                                output_bytes=job.info.output_bytes + len(retained),
                                output_truncated=job.info.output_truncated
                                or len(retained) < len(data),
                            )
                    code = process.poll()
                    if code is not None:
                        if exit_at is None:
                            exit_at = now
                        if eof or now - exit_at >= DRAIN_SECONDS:
                            job.update(
                                exit_code=code,
                                output_complete=eof,
                                status="cancelled"
                                if job.cancel_requested
                                else "completed",
                                output_truncated=job.info.output_truncated or not eof,
                            )
                            break
                    await asyncio.sleep(0.01)
        except Exception as exc:
            job.update(status="failed", error=str(exc))
        finally:
            if process is not None:
                # Descendants must not outlive their job, even if they closed stdout.
                self._signal(process, signal.SIGKILL)
                if process.stdout is not None:
                    process.stdout.close()
                await asyncio.to_thread(process.wait)
            job.finished = time.monotonic()
            Path(job.info.output_path).with_name("meta.json").write_text(
                json.dumps(job.snapshot()), encoding="utf-8"
            )
            self.publish(job)
