"""Shell jobs, services and path watches under split execution.

The supervisor stays with the agent loop, but every process the agent starts must run
in the task workspace, beside its kernel. The execution worker hosts them there with the
same job, service and watch code a local supervisor uses (`WorkspaceHost`); the
supervisor keeps its records and reaches the worker through `RemoteHost`.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import dataclasses
import os
import threading
import time
import uuid
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rlm.service_manager import ServiceManager
from rlm.shell import JobInfo
from rlm.shell_jobs import JobRecord, ShellJobs
from rlm.subscriptions import PathChanges, watch_path
from rlm.tools.ipython import build_kernel_env

EVENT_BYTES = 4 * 1024 * 1024
"""Job output one events call carries, so its reply stays well under a frame."""
EVENT_WAIT = 1.0
"""Seconds an events call waits for something to report."""
_JOB_FIELDS = [field.name for field in dataclasses.fields(JobInfo)]


def _env(layers: dict) -> dict[str, str]:
    """A command's environment, built where it runs from the supervisor's layers."""
    return {
        **build_kernel_env(layers["kernel_env"]),
        **layers["shell_env"],
        **layers["env"],
    }


class WorkspaceHost:
    """Runs the supervisor's shell jobs, services and path watches in this workspace,
    on an event loop of its own beside the worker's kernel."""

    def __init__(self, directory: Path) -> None:
        self.directory = directory
        self.loop = asyncio.new_event_loop()
        threading.Thread(
            target=self.loop.run_forever, name="rlm-host", daemon=True
        ).start()
        self.jobs = ShellJobs(self._finished, self._output)
        self.services = ServiceManager(
            Path(os.environ.get("RLM_SERVICES_DIR", "/tmp/rlm-services"))
        )
        self.watches: dict[str, asyncio.Task] = {}
        self._changed = threading.Condition()
        self._jobs_changed: dict[str, bool] = {}  # job id -> finished
        self._sent: dict[str, int] = {}  # job id -> output bytes reported
        self._paths: list[dict] = []

    def handle(self, method: str, params: dict) -> Any:
        if method == "host.events":
            return self._events(params["wait"])
        if method == "jobs.start":
            return self._run(self._start(params))
        if method == "jobs.cancel":
            return self._run(self.jobs.cancel(self.jobs.jobs[params["id"]]))
        if method == "jobs.close":
            return self._run(self.jobs.close(params["owner_id"]))
        if method == "services.start":
            return self.services.start(
                name=params["name"],
                command=params["command"],
                cwd=os.path.realpath(params["cwd"]),
                env=_env(params["env"]),
                owner_id=params["owner_id"],
            )
        if method == "services.list":
            return self.services.list()
        if method == "services.get":
            return self.services.get(params["name"])
        if method == "services.tail":
            return self.services.tail(params["name"], params["lines"])
        if method == "services.stop":
            return self._run(self.services.stop(params["name"], params["grace"]))
        if method == "watch.path":
            return self._run(self._watch(params))
        if method == "watch.cancel":
            if (task := self.watches.pop(params["id"], None)) is not None:
                self.loop.call_soon_threadsafe(task.cancel)
            return None
        raise ValueError(f"unknown execution method: {method}")

    def close(self) -> None:
        """End the session's jobs and watches; services outlive it, as they do a
        supervisor."""
        self._run(self.jobs.close(None))
        for task in self.watches.values():
            self.loop.call_soon_threadsafe(task.cancel)
        self.loop.call_soon_threadsafe(self.loop.stop)

    def _run(self, coroutine) -> Any:
        return asyncio.run_coroutine_threadsafe(coroutine, self.loop).result()

    async def _start(self, params: dict) -> dict:
        info = self.jobs.start(
            owner_id=params["owner_id"],
            command=params["command"],
            cwd=os.path.realpath(params["cwd"]),
            directory=self.directory,
            env=_env(params["env"]),
            source_request_id=None,
            timeout=params["timeout"],
        )
        job = self.jobs.jobs[info["id"]]
        while job.info.status == "starting" and job.finished is None:
            await asyncio.sleep(0.01)
        return job.snapshot()

    def _output(self, job: JobRecord, offset: int) -> None:
        with self._changed:
            self._jobs_changed.setdefault(job.info.id, False)
            self._changed.notify_all()

    def _finished(self, job: JobRecord) -> None:
        with self._changed:
            self._jobs_changed[job.info.id] = True
            self._changed.notify_all()

    def _events(self, wait: float) -> list[dict]:
        """What changed since the last call: each changed job's state and new output,
        in all at most EVENT_BYTES of it, and the path watches' changes."""
        with self._changed:
            self._changed.wait_for(
                lambda: self._jobs_changed or self._paths, timeout=wait
            )
            changed, self._jobs_changed = self._jobs_changed, {}
            paths, self._paths = self._paths, []
        events, budget = [], EVENT_BYTES
        for job_id, finished in changed.items():
            job = self.jobs.jobs[job_id]
            sent = self._sent.get(job_id, 0)
            with open(job.info.output_path, "rb") as output:
                output.seek(sent)
                data = output.read(budget)
            budget -= len(data)
            self._sent[job_id] = sent + len(data)
            more = self._sent[job_id] < job.info.output_bytes
            if more:  # the rest goes with the next call
                with self._changed:
                    self._jobs_changed[job_id] = (
                        self._jobs_changed.get(job_id, False) or finished
                    )
            events.append(
                {
                    "type": "job",
                    "job": job.snapshot(),
                    "data": base64.b64encode(data).decode(),
                    "finished": finished and not more,
                }
            )
        return events + paths

    async def _watch(self, params: dict) -> str:
        target, changes = await watch_path(Path(params["path"]), params["recursive"])
        self.watches[params["id"]] = asyncio.create_task(
            self._forward(params["id"], changes)
        )
        return str(target)

    async def _forward(self, watch_id: str, changes: PathChanges) -> None:
        first = True
        try:
            async for paths, exists in changes:
                # The first report says the watch is ready; later ones only changes.
                if first or paths or not exists:
                    self._push_path(
                        {
                            "type": "path",
                            "id": watch_id,
                            "paths": sorted(paths),
                            "exists": exists,
                        }
                    )
                first = False
                if not exists:
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._push_path({"type": "path", "id": watch_id, "error": str(exc)})
        finally:
            await changes.aclose()

    def _push_path(self, event: dict) -> None:
        with self._changed:
            self._paths.append(event)
            self._changed.notify_all()


class RemoteShellJobs(ShellJobs):
    """Shell jobs that run in the execution workspace. The worker owns each process and
    its output file; the supervisor keeps the job's record and a copy of its output."""

    def __init__(
        self,
        host: RemoteHost,
        publish: Callable[[JobRecord], None],
        output: Callable[[JobRecord, int], None] | None = None,
    ):
        super().__init__(publish, output)
        self.host = host
        self._done: dict[str, asyncio.Future] = {}
        self._copies: dict[str, Path] = {}

    async def start_remote(
        self,
        *,
        owner_id: str,
        command: str,
        cwd: str,
        directory: Path,
        env: dict,
        source_request_id: str | None,
        timeout: float | None = None,
        notify: bool = True,
    ) -> dict:
        snapshot = await self.host.call(
            "jobs.start",
            {
                "owner_id": owner_id,
                "command": command,
                "cwd": cwd,
                "env": env,
                "timeout": timeout,
            },
        )
        job = JobRecord(
            JobInfo(**{name: snapshot[name] for name in _JOB_FIELDS}),
            source_request_id,
            notify=notify,
        )
        job.update(output_bytes=0)  # counts the copy, which fills from events
        copy = directory / "jobs" / job.info.id / "output.bin"
        copy.parent.mkdir(parents=True)
        copy.touch()
        self._copies[job.info.id] = copy
        self._done[job.info.id] = asyncio.get_running_loop().create_future()
        self.jobs[job.info.id] = job
        job.task = asyncio.create_task(self._follow(job))
        job.task.add_done_callback(self._observe_failure)
        return job.snapshot()

    def output_file(self, job: JobRecord) -> str:
        return str(self._copies[job.info.id])

    async def cancel(self, job: JobRecord) -> dict:
        job.cancel_requested = True
        if job.finished is None:
            await self.host.call("jobs.cancel", {"id": job.info.id})
        await asyncio.shield(job.task)
        return job.snapshot()

    async def close(self, owner_id: str | None = None) -> None:
        jobs = [
            job
            for job in self.jobs.values()
            if job.finished is None
            and (owner_id is None or job.info.owner_id == owner_id)
        ]
        for job in jobs:
            job.cancel_requested = True
        if jobs:
            try:
                await self.host.call("jobs.close", {"owner_id": owner_id})
            except Exception as exc:
                self.disconnect(str(exc))
            await asyncio.gather(
                *(asyncio.shield(job.task) for job in jobs), return_exceptions=True
            )

    def apply(self, event: dict) -> None:
        job = self.jobs.get(event["job"]["id"])
        if job is None or job.finished is not None:
            return
        data = base64.b64decode(event["data"])
        start = job.info.output_bytes
        if data:
            with open(self._copies[job.info.id], "ab") as copy:
                copy.write(data)
        job.info = JobInfo(
            **{
                **{name: event["job"][name] for name in _JOB_FIELDS},
                "output_bytes": start + len(data),
            }
        )
        if data and self.output is not None:
            self.output(job, start)
        if event["finished"]:
            self._done[job.info.id].set_result(None)

    def disconnect(self, error: str) -> None:
        """Fail every running job: the worker that ran them is gone."""
        for job_id, done in self._done.items():
            if not done.done():
                self.jobs[job_id].update(
                    status="failed", error=f"execution worker disconnected: {error}"
                )
                done.set_result(None)

    async def _follow(self, job: JobRecord) -> None:
        try:
            await self._done[job.info.id]
        finally:
            job.finished = time.monotonic()
            self.publish(job)


class RemoteHost:
    """The supervisor's handle on the WorkspaceHost behind the root agent's worker."""

    def __init__(self, call: Callable[[str, dict], Any]) -> None:
        self._call = call
        self.jobs: RemoteShellJobs | None = None
        self._paths: dict[str, asyncio.Queue] = {}
        self._events: asyncio.Task | None = None

    async def call(self, method: str, params: dict) -> Any:
        return await asyncio.to_thread(self._call, method, params)

    def follow(self, jobs: RemoteShellJobs) -> None:
        self.jobs = jobs
        self._events = asyncio.create_task(self._pump())

    async def close(self) -> None:
        if self._events is not None:
            self._events.cancel()
            await asyncio.gather(self._events, return_exceptions=True)

    async def watch(self, target: Path, recursive: bool) -> tuple[Path, PathChanges]:
        watch_id = uuid.uuid4().hex
        queue: asyncio.Queue = asyncio.Queue()
        self._paths[watch_id] = queue
        try:
            resolved = await self.call(
                "watch.path",
                {"id": watch_id, "path": str(target), "recursive": recursive},
            )
        except BaseException:
            del self._paths[watch_id]
            raise
        return Path(resolved), self._changes(watch_id, queue)

    async def _changes(self, watch_id: str, queue: asyncio.Queue) -> PathChanges:
        try:
            while True:
                event = await queue.get()
                if "error" in event:
                    raise RuntimeError(event["error"])
                yield set(event["paths"]), event["exists"]
        finally:
            self._paths.pop(watch_id, None)
            with contextlib.suppress(Exception):
                await self.call("watch.cancel", {"id": watch_id})

    async def _pump(self) -> None:
        assert self.jobs is not None
        try:
            while True:
                for event in await self.call("host.events", {"wait": EVENT_WAIT}):
                    if event["type"] == "job":
                        self.jobs.apply(event)
                    elif (queue := self._paths.get(event["id"])) is not None:
                        queue.put_nowait(event)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # the worker is gone
            self.jobs.disconnect(str(exc))
            for queue in self._paths.values():
                queue.put_nowait({"error": f"execution worker disconnected: {exc}"})
