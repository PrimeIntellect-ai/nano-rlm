"""Subscription lifetimes and bounded batching, independent of IPython kernels."""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

from watchfiles import awatch

from rlm.watch import SubscriptionInfo

BATCH_SECONDS = 0.2
MAX_ACTIVE_SUBSCRIPTIONS = 64
MAX_SUBSCRIPTIONS = 1024
MAX_PATH_BYTES = 32_768


@dataclass
class Subscription:
    info: SubscriptionInfo
    cursor: int = 0
    pending: dict = field(default_factory=dict)
    timer: asyncio.TimerHandle | None = None
    task: asyncio.Task | None = None
    ready: asyncio.Event = field(default_factory=asyncio.Event)


class Subscriptions:
    def __init__(
        self,
        publish: Callable[[Subscription, str, dict], None],
        record: Callable[[Subscription], None],
    ):
        self.items: dict[str, Subscription] = {}
        self.publish = publish
        self.record = record

    def register(
        self,
        owner_id: str,
        kind: str,
        target: str,
        *,
        cursor: int = 0,
        recursive: bool = False,
        completed: bool = False,
    ) -> Subscription:
        if len(self.items) >= MAX_SUBSCRIPTIONS:
            raise RuntimeError("subscription limit reached")
        if not completed and (
            sum(s.info.status == "active" for s in self.items.values())
            >= MAX_ACTIVE_SUBSCRIPTIONS
        ):
            raise RuntimeError("active subscription limit reached")
        sub = Subscription(
            SubscriptionInfo(
                uuid.uuid4().hex,
                owner_id,
                kind,
                target,
                recursive,
                "completed" if completed else "active",
                time.time(),
                None,
            ),
            cursor,
        )
        self.record(sub)
        self.items[sub.info.id] = sub
        return sub

    async def path(self, owner_id: str, target: Path, recursive: bool) -> Subscription:
        if not target.exists():
            raise FileNotFoundError(target)
        sub = self.register(owner_id, "path", str(target), recursive=recursive)
        sub.task = asyncio.create_task(self._watch_path(sub, target))
        try:
            await sub.ready.wait()
        except BaseException:
            await self.cancel(sub)
            raise
        if sub.info.status == "failed":
            raise RuntimeError(sub.info.error)
        return sub

    def finish(self, kind: str, target: str) -> None:
        for sub in self.items.values():
            if sub.info.status == "active" and (sub.info.kind, sub.info.target) == (
                kind,
                target,
            ):
                try:
                    if sub.timer is not None:
                        sub.timer.cancel()
                    self._flush(sub)
                finally:
                    if sub.info.status == "active":
                        sub.info = replace(sub.info, status="completed")
                        self.record(sub)

    def get(self, owner_id: str, subscription_id: str) -> Subscription:
        sub = self.items.get(subscription_id)
        if sub is None or sub.info.owner_id != owner_id:
            raise PermissionError("unknown subscription or not owned by this agent")
        return sub

    def activity(self, kind: str, target: str, start: int, end: int) -> None:
        for sub in self.items.values():
            if sub.info.status != "active" or (sub.info.kind, sub.info.target) != (
                kind,
                target,
            ):
                continue
            start = max(start, sub.cursor)
            if end <= start:
                continue
            if not sub.pending:
                sub.pending = {"start": start, "end": end}
            else:
                sub.pending["end"] = end
            sub.cursor = end
            self._schedule(sub)

    def _schedule(self, sub: Subscription) -> None:
        if sub.timer is None:
            sub.timer = asyncio.get_running_loop().call_later(
                BATCH_SECONDS, self._flush, sub
            )

    def _flush(self, sub: Subscription) -> None:
        sub.timer = None
        if sub.info.status != "active" or not sub.pending:
            return
        payload, sub.pending = sub.pending, {}
        if sub.info.kind == "path":
            payload["paths"] = sorted(payload["paths"])
        try:
            self.publish(sub, "watch." + sub.info.kind, payload)
        except Exception as exc:
            self._fail(sub, str(exc))

    def _fail(self, sub: Subscription, error: str) -> None:
        sub.info = replace(sub.info, status="failed", error=error)
        if sub.timer is not None:
            sub.timer.cancel()
            sub.timer = None
        sub.pending.clear()
        self.record(sub)
        self.publish(sub, "watch.failed", {"error": error})

    async def _watch_path(self, sub: Subscription, target: Path) -> None:
        is_file = target.is_file()
        watched = target.parent if is_file else target
        try:
            async for changes in awatch(
                watched,
                recursive=sub.info.recursive if not is_file else False,
                watch_filter=None,
                debounce=1,
                step=1,
                rust_timeout=100,
                yield_on_timeout=True,
                ignore_permission_denied=False,
            ):
                sub.ready.set()
                if sub.info.status != "active":
                    return
                paths = {
                    name for _, name in changes if not is_file or name == str(target)
                }
                if paths:
                    pending = sub.pending or {"paths": set(), "truncated": False}
                    for name in sorted(paths):
                        if (
                            len(json.dumps(sorted(pending["paths"] | {name})).encode())
                            <= MAX_PATH_BYTES
                        ):
                            pending["paths"].add(name)
                        else:
                            pending["truncated"] = True
                    sub.pending = pending
                    self._schedule(sub)
                if not target.exists():
                    self._flush(sub)
                    if sub.info.status == "active":
                        self._fail(
                            sub,
                            "watched path was removed; register again after recreation",
                        )
                    return
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(sub, str(exc))
        finally:
            sub.ready.set()

    async def cancel(self, sub: Subscription) -> dict:
        if sub.info.status == "active":
            sub.info = replace(sub.info, status="cancelled")
            self.record(sub)
        if sub.timer is not None:
            sub.timer.cancel()
            sub.timer = None
        sub.pending.clear()
        if sub.task is not None:
            sub.task.cancel()
            await asyncio.gather(sub.task, return_exceptions=True)
        return asdict(sub.info)

    async def close(self, owner_id: str | None = None) -> None:
        for sub in list(self.items.values()):
            if owner_id is None or sub.info.owner_id == owner_id:
                await self.cancel(sub)
