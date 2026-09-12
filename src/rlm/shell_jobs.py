"""Bash process ownership and bounded output capture outside execution kernels."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import signal
import subprocess
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from collections.abc import Callable

from rlm.shell import JobInfo

logger = logging.getLogger(__name__)

MAX_ACTIVE_JOBS = 32
MAX_JOBS = 1024
MAX_OUTPUT_BYTES = 16 * 1024 * 1024
DRAIN_SECONDS = 2.0
RUN_TEXT_BYTES = 16 * 1024  # run() returns at most this much: head + tail of the output
# A blocking run() with no explicit timeout is killed after this many seconds: a command
# that never exits (a server started in the foreground) must come back as timed_out
# instead of hanging the cell past the kernel's 600 s limit and forcing a kernel restart.
DEFAULT_RUN_TIMEOUT = 540.0


@dataclass
class ShellJob:
    info: JobInfo
    source_request_id: str | None
    started: float = field(default_factory=time.monotonic)
    finished: float | None = None
    cancel_requested: bool = False
    timed_out: bool = False  # set by the loop when the deadline fires; JobInfo.timed_out derives from status
    notify: bool = (
        True  # publish shell.completed to the owner's inbox (start(); not run())
    )
    task: asyncio.Task | None = None

    def snapshot(self) -> dict:
        return {
            **asdict(self.info),
            "elapsed_seconds": (self.finished or time.monotonic()) - self.started,
        }

    def update(self, **values) -> None:
        self.info = JobInfo(**{**asdict(self.info), **values})


class ShellJobs:
    def __init__(
        self,
        publish: Callable[[ShellJob], None],
        output: Callable[[ShellJob, int], None] | None = None,
    ):
        self.jobs: dict[str, ShellJob] = {}
        self.publish = publish
        self.output = output

    def start(
        self,
        *,
        owner_id: str,
        command: str,
        cwd: str,
        directory: Path,
        env: dict[str, str],
        source_request_id: str | None,
        timeout: float | None = None,
        notify: bool = True,
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
                timeout=timeout,
            ),
            source_request_id,
            notify=notify,
        )
        self.jobs[job_id] = job
        job.task = asyncio.create_task(self._run(job, env))
        job.task.add_done_callback(self._observe_failure)
        return job.snapshot()

    @staticmethod
    def _observe_failure(task: asyncio.Task) -> None:
        if not task.cancelled() and (error := task.exception()) is not None:
            logger.error(
                "Bash job finalization failed",
                exc_info=(type(error), error, error.__traceback__),
            )

    def get(self, owner_id: str, job_id: str) -> ShellJob:
        job = self.jobs.get(job_id)
        if job is None or job.info.owner_id != owner_id:
            raise PermissionError("unknown job or job is not owned by this agent")
        return job

    def read(self, job: ShellJob, cursor: int, max_bytes: int) -> dict:
        if cursor < 0:
            raise ValueError("cursor must be >= 0")
        # A cursor past the retained output is not an error: it returns an empty
        # chunk positioned at the end, so "read until done" loops terminate.
        cursor = min(cursor, job.info.output_bytes)
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

    def run_text(self, job: ShellJob) -> dict:
        """Text for a finished run(): the whole output when it fits RUN_TEXT_BYTES,
        otherwise its first and last halves around an explicit gap marker (test
        runners put the failure at the top and the summary at the bottom)."""
        total = job.info.output_bytes
        if total <= RUN_TEXT_BYTES:
            chunk = self.read(job, 0, RUN_TEXT_BYTES)
            return {
                "text": chunk["text"],
                "truncated": chunk["truncated"] or not chunk["done"],
            }
        half = RUN_TEXT_BYTES // 2
        head = self.read(job, 0, half)["text"]
        tail = self.read(job, total - half, half)["text"]
        omitted = total - 2 * half
        marker = (
            f"\n[... {omitted} bytes omitted; output is retained up to 16 MiB: "
            f"job = await rlm.shell.get(result.job_id); await job.read(cursor={half}) ...]\n"
        )
        return {"text": head + marker + tail, "truncated": True}

    async def cancel(self, job: ShellJob) -> dict:
        job.cancel_requested = True
        try:
            await asyncio.shield(job.task)
        except Exception:
            if job.finished is None:
                raise
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
            results = await asyncio.gather(
                *(asyncio.shield(job.task) for job in jobs), return_exceptions=True
            )
            for job, result in zip(jobs, results):
                if isinstance(result, BaseException) and job.finished is None:
                    raise result

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
                    # Poll before judging the deadline: a process that already exited
                    # keeps its real status and exit code even if the deadline passed.
                    code = process.poll()
                    if (
                        code is None
                        and job.info.timeout is not None
                        and cancel_at is None
                        and not job.cancel_requested
                        and now - job.started >= job.info.timeout
                    ):
                        job.timed_out = True
                        job.update(
                            error=f"timed out after {job.info.timeout:g}s; process group killed"
                        )
                        job.cancel_requested = True
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
                            if retained and self.output is not None:
                                self.output(job, job.info.output_bytes - len(retained))
                    if code is None:
                        code = process.poll()
                    if code is not None:
                        if exit_at is None:
                            exit_at = now
                        if eof or now - exit_at >= DRAIN_SECONDS:
                            job.update(
                                exit_code=None if job.timed_out else code,
                                output_complete=eof,
                                status="timed_out"
                                if job.timed_out
                                else "cancelled"
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
            errors = []
            try:
                Path(job.info.output_path).with_name("meta.json").write_text(
                    json.dumps(job.snapshot()), encoding="utf-8"
                )
            except Exception as exc:
                errors.append(exc)
                job.update(error=f"Metadata persistence failed: {exc}")
            try:
                self.publish(job)
            except Exception as exc:
                errors.append(exc)
                job.update(
                    error=f"{job.info.error + '; ' if job.info.error else ''}Completion publication failed: {exc}"
                )
            if errors:
                raise errors[0]
