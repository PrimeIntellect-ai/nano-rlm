"""Byte-stream transports for tool execution, independent of provisioning."""

from __future__ import annotations

import socket
import subprocess
from typing import BinaryIO, Protocol

from rlm.config import TcpExecutionConfig


class ExecutionTransport(Protocol):
    def connect(self) -> tuple[BinaryIO, BinaryIO]: ...

    def close(self) -> None: ...


class StdioTransport:
    def __init__(self, command):
        self.command = command
        self.process = None

    def connect(self):
        self.process = subprocess.Popen(
            self.command,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            start_new_session=True,
        )
        return self.process.stdout, self.process.stdin

    def close(self):
        process, self.process = self.process, None
        if process is not None:
            process.stdin.close()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            process.stdout.close()


class TcpTransport:
    """One execution session per connection on a trusted private network."""

    def __init__(self, endpoint: TcpExecutionConfig):
        self.endpoint = endpoint
        self.connection = None
        self.stream = None

    def connect(self):
        from rlm.execution import _receive, _send

        self.connection = socket.create_connection(
            (self.endpoint.host, self.endpoint.port), timeout=10
        )
        # Each frame is one write, and calls overlap (the event long-poll, kernel
        # callbacks): Nagle would hold a frame until the previous one is acknowledged.
        self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stream = self.connection.makefile("rwb", buffering=0)
        try:
            _send(self.stream, {"token": self.endpoint.token})
            if _receive(self.stream) != {"authenticated": True}:
                raise PermissionError("execution worker authentication failed")
            self.connection.settimeout(None)
        except BaseException:
            self.close()
            raise
        return self.stream, self.stream

    def close(self):
        if self.connection is not None:
            try:
                self.connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass  # the worker may already have disconnected
            self.connection.close()
            self.connection = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
