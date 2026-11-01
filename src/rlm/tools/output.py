"""Bounded capture of a command's output."""

from __future__ import annotations

from dataclasses import dataclass

OUTPUT_KEEP_BYTES = 1024 * 1024
"""The most of one tool result's output that is kept, half from its start and half from
its end. The conversation shows far less of it; the rest stays in the session log, and
a tool result always fits one execution frame."""


@dataclass(frozen=True)
class OutputSize:
    """Text measured as the conversation's truncation measures it."""

    nbytes: int = 0
    """UTF-8 bytes."""
    nchars: int = 0
    newlines: int = 0

    @classmethod
    def of(cls, text: str) -> OutputSize:
        return cls(
            len(text.encode("utf-8", "surrogatepass")), len(text), text.count("\n")
        )

    def __add__(self, other: OutputSize) -> OutputSize:
        return OutputSize(
            self.nbytes + other.nbytes,
            self.nchars + other.nchars,
            self.newlines + other.newlines,
        )

    def __sub__(self, other: OutputSize) -> OutputSize:
        return OutputSize(
            self.nbytes - other.nbytes,
            self.nchars - other.nchars,
            self.newlines - other.newlines,
        )


def _char_start(data: bytes | bytearray, at: int) -> int:
    """`at`, moved back to the start of the UTF-8 character it falls in."""
    while 0 < at < len(data) and data[at] & 0xC0 == 0x80:
        at -= 1
    return at


class OutputCapture:
    """Output as it is produced, kept to its first and last `keep // 2` bytes. The middle
    is dropped but measured, so a long command runs to its end in bounded memory and the
    whole output's size is still known."""

    def __init__(self, keep: int = OUTPUT_KEEP_BYTES):
        self._half = keep // 2
        self._head = bytearray()
        self._head_full = False
        self._tail = bytearray()
        self._dropped = False
        self.size = OutputSize()
        """The whole output's size."""

    def add(self, text: str) -> None:
        if not text:
            return
        self.size += OutputSize.of(text)
        data = text.encode("utf-8", "surrogatepass")
        if not self._head_full:
            room = self._half - len(self._head)
            if len(data) <= room:
                self._head += data
                return
            cut = _char_start(data, room)
            self._head += data[:cut]
            self._head_full = True
            data = data[cut:]
        self._tail += data
        if len(self._tail) > self._half:
            del self._tail[: _char_start(self._tail, len(self._tail) - self._half)]
            self._dropped = True

    def result(self) -> tuple[str, OutputSize]:
        """The kept text, and how much more the whole output measures than it: when
        its middle was dropped, the middle less the note that replaces it."""
        head = self._head.decode("utf-8", "surrogatepass")
        tail = self._tail.decode("utf-8", "surrogatepass")
        if not self._dropped:
            return head + tail, OutputSize()
        dropped = self.size.nbytes - len(self._head) - len(self._tail)
        text = f"{head}\n[... {dropped} bytes of output omitted ...]\n{tail}"
        return text, self.size - OutputSize.of(text)
