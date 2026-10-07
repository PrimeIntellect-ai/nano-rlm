"""Persistent execution sessions over a trusted worker command's stdio.

The command is supplied by the orchestrator (for example docker exec -i). Only
kernel configuration and generated skills cross into the worker; model credentials,
conversation ownership, and recursive agent loops stay in the calling process.
"""

from __future__ import annotations

import concurrent.futures
import json
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import uuid
from pathlib import Path
from types import SimpleNamespace

from rlm.broker import BrokerEndpoint
from rlm.tools.ipython import IPythonREPL

MAX_FRAME = 16 * 1024 * 1024


def _read(stream, size):
    chunks = bytearray()
    while len(chunks) < size:
        chunk = stream.read(size - len(chunks))
        if not chunk:
            raise EOFError("execution worker disconnected")
        chunks.extend(chunk)
    return bytes(chunks)


def _receive(stream):
    size = struct.unpack("!I", _read(stream, 4))[0]
    if size > MAX_FRAME:
        raise ValueError("execution frame exceeds 16 MiB")
    value = json.loads(_read(stream, size))
    if not isinstance(value, dict):
        raise ValueError("execution frame must be an object")
    return value


def _send(stream, value):
    data = json.dumps(value).encode()
    if len(data) > MAX_FRAME:
        raise ValueError("execution frame exceeds 16 MiB")
    stream.write(struct.pack("!I", len(data)) + data)
    stream.flush()


class _Peer:
    """Bidirectional RPC: callbacks must keep flowing during a running cell."""

    def __init__(self, reader, writer, handler):
        self.reader, self.writer, self.handler = reader, writer, handler
        self.lock = threading.Lock()
        self.pending = {}
        self.closed = threading.Event()
        self.error = None
        self.slots = threading.BoundedSemaphore(32)
        self.thread = threading.Thread(target=self._listen, daemon=True)
        self.thread.start()

    def send(self, message):
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("execution connection closed") from self.error
            _send(self.writer, message)

    def call(self, method, params=None, timeout=None):
        key = uuid.uuid4().hex
        future = concurrent.futures.Future()
        with self.lock:
            if self.closed.is_set():
                raise RuntimeError("execution connection closed") from self.error
            self.pending[key] = future
        try:
            self.send({"id": key, "method": method, "params": params})
            return future.result(timeout=timeout)
        finally:
            with self.lock:
                self.pending.pop(key, None)

    def _answer(self, message):
        try:
            result = self.handler(message["method"], message.get("params"))
            reply = {"id": message["id"], "result": result}
        except Exception as exc:
            reply = {"id": message["id"], "error": str(exc)}
        try:
            try:
                self.send(reply)
            except ValueError as exc:
                self.send({"id": message["id"], "error": str(exc)})
        except (OSError, RuntimeError):
            pass  # the caller disconnected while its operation was settling
        finally:
            self.slots.release()

    def _listen(self):
        try:
            while True:
                message = _receive(self.reader)
                if "method" in message:
                    if not self.slots.acquire(blocking=False):
                        self.send(
                            {
                                "id": message["id"],
                                "error": "execution request limit reached",
                            }
                        )
                        continue
                    threading.Thread(
                        target=self._answer, args=(message,), daemon=True
                    ).start()
                else:
                    with self.lock:
                        future = self.pending.get(message["id"])
                    if future is not None:
                        if "error" in message:
                            future.set_exception(RuntimeError(message["error"]))
                        else:
                            future.set_result(message["result"])
        except Exception as exc:
            with self.lock:
                self.error = exc
                self.closed.set()
                for future in self.pending.values():
                    if not future.done():
                        future.set_exception(exc)


class RemoteREPL(IPythonREPL):
    """The IPython execution-session interface backed by a separate worker."""

    def __init__(self, *, command, **kwargs):
        super().__init__(**kwargs)
        self.command = command
        self._process = None
        self._peer = None
        self.session_dir = None

    def _broker(self, method, request):
        if method != "broker" or self.broker_endpoint is None:
            raise PermissionError("unknown execution callback")
        if request.get("capability") != self.broker_endpoint.capability:
            raise PermissionError("invalid execution capability")
        # These operations currently launch jobs/watchers in the supervisor process.
        # Refuse them until they have execution-session implementations.
        if (
            request.get("op", "").startswith("shell.")
            or request.get("op") == "watch.path"
        ):
            raise ValueError(
                "split execution does not support rlm.shell or watch.path; use bash/IPython in the task workspace"
            )
        with socket.socket(socket.AF_UNIX) as connection:
            connection.connect(self.broker_endpoint.socket_path)
            with connection.makefile("rwb", buffering=0) as stream:
                _send(stream, request)
                return _receive(stream)

    def start(self):
        self._process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            start_new_session=True,
        )
        self._peer = _Peer(self._process.stdout, self._process.stdin, self._broker)
        modules = {
            path.name: path.read_text() for path in self.session.dir.glob("*.py")
        }
        self.session_dir = self._peer.call(
            "start",
            {
                "cwd": self.cwd,
                "kernel_env": self.kernel_env,
                "depth": self.depth,
                "max_depth": self.max_depth,
                "exec_timeout": self.exec_timeout,
                "allow_git": self.allow_git,
                "capability": self.broker_endpoint.capability
                if self.broker_endpoint
                else None,
                "modules": modules,
            },
            timeout=120,
        )

    def execute(self, code, timeout=None):
        result = self._peer.call(
            "execute",
            {
                "code": code,
                "timeout": timeout,
                "scope_id": self._scope_id,
                "history": (self.session.dir / "messages.jsonl").read_text(),
            },
        )
        self._recovery_notices.extend(result["notices"])
        # Programmatic tool statistics are observational, just as with local kernels.
        (self.session.dir / "programmatic_tool_calls.jsonl").write_text(
            result["tool_log"]
        )
        return result["output"]

    def interrupt(self):
        if self._peer is not None and not self._peer.closed.is_set():
            self._peer.call("interrupt", timeout=5)

    def finish_interrupt(self):
        if self._peer is not None and not self._peer.closed.is_set():
            self._peer.call("finish_interrupt", timeout=5)

    def shutdown(self):
        process, self._process = self._process, None
        if process is not None:
            try:
                if self._peer is not None and not self._peer.closed.is_set():
                    self._peer.call("close", timeout=30)
            finally:
                # EOF is also a worker shutdown request if the explicit close failed.
                process.stdin.close()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                process.stdout.close()
        self._executor.shutdown(wait=False)


class _Worker:
    def __init__(self):
        self.repl = None
        self.peer = None
        self.directory = tempfile.TemporaryDirectory(prefix="rlm-execution-")
        self.session_dir = Path(self.directory.name)
        self.listener = None
        self.closed = threading.Event()

    def _accept(self):
        while not self.closed.is_set():
            try:
                connection, _ = self.listener.accept()
            except OSError:
                return
            threading.Thread(
                target=self._proxy, args=(connection,), daemon=True
            ).start()

    def _proxy(self, connection):
        with connection, connection.makefile("rwb", buffering=0) as stream:
            try:
                request = _receive(stream)
                result = self.peer.call("broker", request)
            except Exception as exc:
                result = {"error": str(exc)}
            try:
                _send(stream, result)
            except OSError:
                pass  # interrupted kernels close their pending callbacks

    def handle(self, method, params):
        if method == "start":
            if self.repl is not None:
                raise ValueError("execution session already started")
            params = dict(params)
            modules = params.pop("modules")
            for name, source in modules.items():
                if Path(name).name != name or not name.endswith(".py"):
                    raise ValueError("invalid generated skill name")
                (self.session_dir / name).write_text(source)
            capability = params.pop("capability")
            endpoint = None
            if capability:
                path = str(self.session_dir / "broker.sock")
                self.listener = socket.socket(socket.AF_UNIX)
                self.listener.bind(path)
                self.listener.listen(32)
                threading.Thread(target=self._accept, daemon=True).start()
                endpoint = BrokerEndpoint(path, capability)
            self.repl = IPythonREPL(
                session=SimpleNamespace(dir=self.session_dir),
                broker_endpoint=endpoint,
                **params,
            )
            # All Jupyter operations use the same dedicated thread.
            self.repl._executor.submit(self.repl.start).result()
            return str(self.session_dir)
        if self.repl is None:
            raise ValueError("execution session has not started")
        if method == "execute":

            def execute():
                (self.session_dir / "messages.jsonl").write_text(params["history"])
                self.repl.set_broker_scope(params["scope_id"])
                output = self.repl.execute(params["code"], params["timeout"])
                log = self.session_dir / "programmatic_tool_calls.jsonl"
                return {
                    "output": output,
                    "notices": self.repl.take_recovery_notices(),
                    "tool_log": log.read_text() if log.exists() else "",
                }

            return self.repl._executor.submit(execute).result()
        if method == "interrupt":
            self.repl.interrupt()
        elif method == "finish_interrupt":
            self.repl.finish_interrupt()
        elif method == "close":
            self.repl.shutdown()
            self.repl = None
        else:
            raise ValueError(f"unknown execution method: {method}")

    def close(self):
        self.closed.set()
        if self.listener is not None:
            self.listener.close()
        if self.repl is not None:
            self.repl.interrupt()
            self.repl.shutdown()
        self.directory.cleanup()


def main():
    worker = _Worker()
    output = sys.stdout.buffer
    sys.stdout = sys.stderr  # dependency diagnostics must not corrupt the wire
    worker.peer = _Peer(sys.stdin.buffer, output, worker.handle)
    try:
        worker.peer.closed.wait()
    finally:
        worker.close()


if __name__ == "__main__":
    main()
