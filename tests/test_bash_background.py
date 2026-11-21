"""run_bash returns when bash exits, even if background jobs hold its pipes."""

from __future__ import annotations

import os
import signal
import socket
import subprocess
import threading
import time
import urllib.request
import uuid

import pytest

from rlm.tools.bash import _BACKGROUND_NOTE, bash_outcome, run_bash
from rlm.tools.output import OUTPUT_KEEP_BYTES

TAG_VAR = "RLM_BASH_TEST_TAG"

needs_proc = pytest.mark.skipif(
    not os.path.isdir("/proc/self"), reason="finds spawned processes through /proc"
)


def _tagged(tag: str) -> dict[int, list[str]]:
    """Live (non-zombie) processes, other than this one, carrying ``tag``."""
    needle = f"{TAG_VAR}={tag}".encode()
    found = {}
    for entry in os.listdir("/proc"):
        if not entry.isdigit() or int(entry) == os.getpid():
            continue
        try:
            with open(f"/proc/{entry}/environ", "rb") as f:
                env = f.read().split(b"\0")
            with open(f"/proc/{entry}/stat") as f:
                state = f.read().rsplit(")", 1)[1].split()[0]
            with open(f"/proc/{entry}/cmdline", "rb") as f:
                argv = f.read().decode(errors="replace").split("\0")
        except (OSError, IndexError):
            continue
        if needle in env and state != "Z":
            found[int(entry)] = argv
    return found


@pytest.fixture
def tag(monkeypatch):
    value = uuid.uuid4().hex
    monkeypatch.setenv(TAG_VAR, value)
    yield value
    for pid in _tagged(value):
        try:
            os.kill(pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _timed(command: str, timeout: int = 10, **kwargs) -> tuple[float, str]:
    start = time.monotonic()
    out = run_bash(command, timeout, **kwargs)
    return time.monotonic() - start, out


# (command, seconds the command itself sleeps before bash exits). Each leaves a job that
# holds the call's output, which kept the call open until its timeout before.
BACKGROUND_COMMANDS = [
    ("cd /tmp && nohup sleep 300 >/dev/null 2>&1 &", 0),
    ("cd /tmp && setsid sleep 300 </dev/null >/dev/null 2>&1 &", 0),
    ("sleep 300 &", 0),
    ("nohup sleep 300 &", 0),
    ("sleep 300 >f &", 0),
    ("(cd {cwd} && nohup sleep 300 >f 2>&1 &)", 0),
    ("bash -c 'sleep 300; :' & sleep 1; kill $!", 1),
    ("sleep 300 | cat & sleep 1; kill $!", 1),
]


@needs_proc
@pytest.mark.parametrize(("command", "own_sleep"), BACKGROUND_COMMANDS)
def test_background_job_does_not_hold_call_open(command, own_sleep, tag, tmp_path):
    elapsed, out = _timed(command.format(cwd=tmp_path), cwd=str(tmp_path))
    assert out == _BACKGROUND_NOTE
    assert elapsed < own_sleep + 1.5
    sleepers = [
        pid for pid, argv in _tagged(tag).items() if argv[:2] == ["sleep", "300"]
    ]
    assert len(sleepers) == 1, _tagged(tag)
    os.kill(sleepers[0], signal.SIGKILL)


@needs_proc
def test_unredirected_server_keeps_serving_after_return(tag):
    port = _free_port()
    elapsed, out = _timed(
        f"cd /tmp && python3 -I -m http.server {port} --bind 127.0.0.1 &"
    )
    assert elapsed < 1.5
    assert "timed out" not in out
    url = f"http://127.0.0.1:{port}/"
    give_up = time.monotonic() + 10
    while True:
        try:
            urllib.request.urlopen(url, timeout=2).read()
            break
        except OSError:
            if time.monotonic() > give_up:
                raise
            time.sleep(0.1)
    # The server logs every request to the stderr pipe it inherited from the call.
    for _ in range(3):
        with urllib.request.urlopen(url, timeout=5) as response:
            assert response.status == 200


@needs_proc
def test_output_before_exit_is_complete_while_a_job_holds_the_pipes(tag):
    """More than a pipe's capacity, all written before bash exits, comes back whole."""
    size = 300_000
    elapsed, out = _timed(
        f"head -c {size} /dev/zero | tr '\\0' a; echo; head -c {size} /dev/zero"
        " | tr '\\0' b >&2; sleep 300 &"
    )
    assert out == "a" * size + "\n\n" + "b" * size + "\n" + _BACKGROUND_NOTE
    assert elapsed < 5


@needs_proc
def test_output_written_after_bash_exits_is_not_shown(tag):
    elapsed, out = _timed("echo fg; (sleep 1; echo bg) &")
    assert out == "fg\n\n" + _BACKGROUND_NOTE
    assert elapsed < 1


@needs_proc
@pytest.mark.parametrize("command", ["sleep 30", "sleep 300 & sleep 30"])
def test_timeout_kills_process_group(command, tag):
    elapsed, out = _timed(command, timeout=3)
    assert out == "Error: command timed out after 3s"
    assert 2.9 < elapsed < 4.5
    assert _tagged(tag) == {}


@needs_proc
def test_interrupt_stops_the_command(tag):
    interrupt = threading.Event()
    threading.Timer(0.5, interrupt.set).start()
    elapsed, out = _timed("echo started; sleep 30", interrupt_event=interrupt)
    assert out == "started\n\n[command interrupted]"
    assert elapsed < 2
    assert _tagged(tag) == {}


def test_wait_still_waits_for_background_jobs():
    elapsed, out = _timed("(sleep 1; echo late) & wait; echo done")
    assert out == "late\ndone"
    assert elapsed >= 1


def _reference(command: str, cwd: str) -> str:
    proc = subprocess.run(
        ["bash", "-c", command],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        timeout=30,
        cwd=cwd,
        stdin=subprocess.DEVNULL,
    )
    out = proc.stdout + (("\n" + proc.stderr) if proc.stderr else "")
    if proc.returncode != 0:
        out += f"\n[exit code {proc.returncode}]"
    return out.strip() or "(no output)"


@pytest.mark.parametrize(
    "command",
    [
        "echo a; sleep 0.3; echo b >&2; echo c; exit 3",
        "echo only-stderr >&2",
        "true",
        "exit 7",
        "printf 'ok\\377\\376\\200\\n'; printf '\\377' >&2",
        "printf 'a\\r\\nb\\rc\\n'",
        "printf '  padded  \\n\\n'",
    ],
)
def test_output_matches_subprocess_run(command, tmp_path):
    assert run_bash(command, 30, cwd=str(tmp_path)) == _reference(
        command, str(tmp_path)
    )


def test_large_output_is_bounded():
    total = 5_000_000
    outcome = bash_outcome(f"head -c {total} /dev/zero | tr '\\0' a", 30)
    kept = OUTPUT_KEEP_BYTES // 4
    omitted = f"\n[... {total - 2 * kept} bytes of output omitted ...]\n"
    assert outcome.content == "a" * kept + omitted + "a" * kept
    assert outcome.omitted is not None
