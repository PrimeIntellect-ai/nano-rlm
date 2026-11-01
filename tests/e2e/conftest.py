"""End-to-end runs against a hosted model.

Each test starts `rlm --acp` the way an orchestrator does, gives it a task in a
throwaway workspace, and checks what the agent left there. The suite runs only with
`-m e2e`, needs `PRIME_API_KEY` (plus `PRIME_TEAM_ID` for a team's models), and skips
without it. `RLM_E2E_MODEL` and `RLM_E2E_BASE_URL` choose the model.

Every test runs in three placements: `local` (tools in the agent's process tree),
`split` (a worker over stdio) and `tcp` (a worker over TCP, started on its own
with no credentials in its environment).
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import secrets
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block

from rlm.acp import RUNTIME_METADATA_KEY, SESSION_METADATA_KEY

MODEL = os.environ.get("RLM_E2E_MODEL") or "internal/glm-5.3-fast"
BASE_URL = os.environ.get("RLM_E2E_BASE_URL") or "https://api.pinference.ai/api/v1"
MODES = ("local", "split", "tcp")
PROMPT_TIMEOUT = 600


def pytest_collection_modifyitems(items):
    for item in items:
        if "tests/e2e/" in item.nodeid.replace(os.sep, "/"):
            item.add_marker(pytest.mark.e2e)


@pytest.fixture(scope="session", autouse=True)
def install_fixture_skills():
    """The agent sees the skills a real install has, not the unit tests' fixtures."""
    yield


@pytest.fixture(params=MODES)
def mode(request) -> str:
    return request.param


@dataclass
class Outcome:
    workspace: Path
    answer: str
    stop_reason: str
    snapshot: dict[str, Any]
    session_dir: Path
    records: list[dict[str, Any]] = field(default_factory=list)

    def read(self, name: str) -> str:
        """A file the agent left in the workspace, stripped; empty when it left none."""
        path = self.workspace / name
        return path.read_text().strip() if path.exists() else ""

    def tool_calls(self) -> list[str]:
        """Names of the tools the root agent called, in order."""
        return [
            call["function"]["name"]
            for record in self.records
            if record.get("type") == "assistant"
            for call in record["message"].get("tool_calls") or ()
        ]

    def describe(self) -> str:
        return f"stop {self.stop_reason}; tools {self.tool_calls()}; answer {self.answer[-500:]!r}"


class _Client:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        content = getattr(update, "content", None)
        if getattr(update, "session_update", None) == "agent_message_chunk" and content:
            self.messages.append(getattr(content, "text", ""))


def _provider() -> dict[str, Any]:
    headers = {}
    if os.environ.get("PRIME_TEAM_ID"):
        headers["X-Prime-Team-ID"] = os.environ["PRIME_TEAM_ID"]
    return {
        "base_url": BASE_URL,
        "api_key": os.environ["PRIME_API_KEY"],
        "headers": headers,
        "max_retries": 5,
    }


async def _start_tcp_worker(
    workspace: Path, token: str
) -> tuple[subprocess.Popen, int]:
    """A worker the way a remote workspace runs one: its own process, its own
    environment, none of the agent's credentials."""
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("PRIME_", "RLM_E2E_"))
    }
    env["RLM_EXECUTION_TOKEN"] = token
    worker = subprocess.Popen(
        [sys.executable, "-m", "rlm.execution", "--listen", "127.0.0.1", "--port", "0"],
        cwd=workspace,
        env=env,
        stderr=subprocess.PIPE,
        text=True,
    )
    line = await asyncio.wait_for(asyncio.to_thread(worker.stderr.readline), 60)
    match = re.search(r"listening on \S+:(\d+)", line)
    if not match:
        worker.kill()
        raise RuntimeError(f"execution worker did not start: {line!r}")
    return worker, int(match.group(1))


@pytest.fixture
def run_agent(tmp_path, mode):
    """Run one prompt to its end in a fresh workspace, in this test's placement."""
    if not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    workers: list[subprocess.Popen] = []

    async def run(
        prompt: str,
        *,
        files: dict[str, str] | None = None,
        builtin_tools: list[str] | None = None,
        skills: tuple[str, ...] = ("edit",),
        policy: dict[str, Any] | None = None,
    ) -> Outcome:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for name, text in (files or {}).items():
            (workspace / name).write_text(text)
        home = tmp_path / "rlm"
        runtime: dict[str, Any] = {
            "session_id": "e2e",
            "model": MODEL,
            "provider": _provider(),
            "policy": {"max_depth": 1, "exec_timeout": 120, **(policy or {})},
            "system_prompt_path": None,
            "append_to_system_prompt": None,
            "skills": list(skills),
            "builtin_tools": builtin_tools,
            "kernel_env": {},
            "search_api_key": None,
        }
        if mode == "split":
            runtime["execution_command"] = [sys.executable, "-m", "rlm.execution"]
            runtime["execution_cwd"] = str(workspace)
        elif mode == "tcp":
            token = secrets.token_hex(32)
            worker, port = await _start_tcp_worker(workspace, token)
            workers.append(worker)
            runtime["execution_tcp"] = {
                "host": "127.0.0.1",
                "port": port,
                "token": token,
            }
            runtime["execution_cwd"] = str(workspace)
        client = _Client()
        executable = str(Path(sys.executable).parent / "rlm")
        env = {**os.environ, "RLM_HOME": str(home)}
        async with spawn_agent_process(client, executable, "--acp", env=env) as (
            connection,
            _process,
        ):
            await connection.initialize(PROTOCOL_VERSION)
            created = await connection.new_session(
                cwd=str(workspace), mcp_servers=[], **{RUNTIME_METADATA_KEY: runtime}
            )
            response = await asyncio.wait_for(
                connection.prompt(
                    session_id=created.session_id, prompt=[text_block(prompt)]
                ),
                PROMPT_TIMEOUT,
            )
            closed = await connection.close_session(created.session_id)
        (session_dir,) = (home / "sessions").iterdir()
        records = [
            json.loads(line)
            for line in (session_dir / "messages.jsonl").read_text().splitlines()
        ]
        return Outcome(
            workspace=workspace,
            answer="".join(client.messages),
            stop_reason=response.stop_reason,
            snapshot=closed.field_meta[SESSION_METADATA_KEY],
            session_dir=session_dir,
            records=records,
        )

    yield run
    for worker in workers:
        worker.kill()
        worker.wait()
