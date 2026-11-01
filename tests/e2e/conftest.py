"""End-to-end runs against a hosted model.

Each test starts `rlm --acp` the way an orchestrator does, gives it a task in a
throwaway workspace, and checks what the agent left there: files, its ledger and the
session snapshot. Where the model could get an answer without the feature under test,
the answer exists only behind that feature (a stamp only a tool call returns, a
codeword only a skill holds, a token only a sub-agent's history shows), so a pass proves
the feature ran. The suite runs only with `-m e2e`, needs `PRIME_API_KEY` (plus
`PRIME_TEAM_ID` for a team's models), and skips without it. `RLM_E2E_MODEL` and
`RLM_E2E_BASE_URL` choose the model.

Every test runs in three placements: `local` (tools in the agent's process tree),
`stdio` (a worker the agent starts, over its stdin and stdout) and `tcp` (a worker
over TCP, started on its own with no credentials in its environment). The last two
are split execution.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import re
import secrets
import subprocess
import sys
from collections.abc import AsyncIterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from acp import PROTOCOL_VERSION, spawn_agent_process, text_block

from rlm.acp import (
    ACP_SEMANTIC_EDGES_METADATA_KEY,
    RUNTIME_METADATA_KEY,
    SESSION_METADATA_KEY,
)

MODEL = os.environ.get("RLM_E2E_MODEL") or "internal/glm-5.3-fast"
BASE_URL = os.environ.get("RLM_E2E_BASE_URL") or "https://api.pinference.ai/api/v1"
MODES = ("local", "stdio", "tcp")
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
class Turn:
    stop_reason: str
    answer: str
    meta: dict[str, Any]


class _Client:
    def __init__(self) -> None:
        self.messages: list[str] = []

    async def session_update(self, session_id: str, update: Any, **kwargs: Any) -> None:
        content = getattr(update, "content", None)
        if getattr(update, "session_update", None) == "agent_message_chunk" and content:
            self.messages.append(getattr(content, "text", ""))


class AgentSession:
    """One ACP session: prompt, steer or cancel it while it lives, then read what it
    left behind."""

    def __init__(
        self,
        connection: Any,
        session_id: str,
        client: _Client,
        workspace: Path,
        home: Path,
    ) -> None:
        self._connection = connection
        self._client = client
        self.session_id = session_id
        self.workspace = workspace
        self.home = home
        self.turns: list[Turn] = []
        self.running: asyncio.Task[Turn] | None = None
        self._calls_before_running = 0
        self.closed_meta: dict[str, Any] | None = None

    async def prompt(self, text: str) -> Turn:
        """Run one prompt to its end."""
        self._client.messages.clear()
        response = await asyncio.wait_for(
            self._connection.prompt(
                session_id=self.session_id, prompt=[text_block(text)]
            ),
            PROMPT_TIMEOUT,
        )
        turn = Turn(
            response.stop_reason,
            "".join(self._client.messages),
            response.field_meta or {},
        )
        self.turns.append(turn)
        return turn

    def start(self, text: str) -> asyncio.Task[Turn]:
        """Start a prompt and return while it runs."""
        self._calls_before_running = len(self.tool_calls())
        self.running = asyncio.create_task(self.prompt(text))
        return self.running

    async def called(self, name: str, timeout: float = 300) -> None:
        """Wait until the running prompt's agent has called the tool `name`."""
        deadline = asyncio.get_running_loop().time() + timeout
        while name not in self.tool_calls()[self._calls_before_running :]:
            assert self.running is not None and not self.running.done(), (
                f"the prompt ended without calling {name}: {self.describe()}"
            )
            assert asyncio.get_running_loop().time() < deadline, (
                f"no {name} call within {timeout}s: {self.describe()}"
            )
            await asyncio.sleep(0.2)

    async def steer(self, text: str) -> dict[str, Any]:
        return await self._connection.ext_method(
            "session/steering",
            {"sessionId": self.session_id, "prompt": [{"type": "text", "text": text}]},
        )

    async def cancel(self) -> None:
        await self._connection.cancel(session_id=self.session_id)

    @property
    def session_dir(self) -> Path:
        (session_dir,) = (self.home / "sessions").iterdir()
        return session_dir

    @property
    def records(self) -> list[dict[str, Any]]:
        """The root agent's ledger as written so far."""
        return self.ledger(self.session_dir)

    def children(self) -> list[Path]:
        """Session directories of the root agent's sub-agents."""
        return sorted(
            path for path in self.session_dir.iterdir() if path.name.startswith("sub-")
        )

    @staticmethod
    def ledger(session_dir: Path) -> list[dict[str, Any]]:
        path = session_dir / "messages.jsonl"
        if not path.exists():
            return []
        records = []
        for line in path.read_text().splitlines():
            with contextlib.suppress(
                json.JSONDecodeError
            ):  # a line still being written
                records.append(json.loads(line))
        return records

    @property
    def snapshot(self) -> dict[str, Any]:
        """The authoritative snapshot `session/close` returned."""
        assert self.closed_meta is not None, "the session is still open"
        return self.closed_meta[SESSION_METADATA_KEY]

    @property
    def answer(self) -> str:
        return self.turns[-1].answer if self.turns else ""

    @property
    def stop_reason(self) -> str:
        return self.turns[-1].stop_reason if self.turns else ""

    def read(self, name: str) -> str:
        """A file the agent left in the workspace, stripped; empty when it left none."""
        path = self.workspace / name
        return path.read_text().strip() if path.exists() else ""

    def tool_calls(self) -> list[str]:
        """Names of the tools the root agent called, in order."""
        return [call["function"]["name"] for call in self.tool_call_records()]

    def tool_call_records(self) -> list[dict[str, Any]]:
        return [
            call
            for record in self.records
            if record.get("type") == "assistant"
            for call in record["message"].get("tool_calls") or ()
        ]

    def tool_results(self) -> list[str]:
        """What the root agent's tool calls returned, in order."""
        return [
            record["content"]
            for record in self.records
            if record.get("type") == "tool_result"
        ]

    def edges(self) -> set[str]:
        """Types of the semantic edges the session published."""
        metas = [turn.meta for turn in self.turns] + [self.closed_meta or {}]
        return {
            edge["type"]
            for meta in metas
            for edge in (meta.get(ACP_SEMANTIC_EDGES_METADATA_KEY) or {}).get(
                "edges", ()
            )
        }

    def describe(self) -> str:
        stops = [turn.stop_reason for turn in self.turns]
        return (
            f"stops {stops}; tools {self.tool_calls()}; answer {self.answer[-500:]!r}"
        )


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
def agent(tmp_path, mode):
    """Open a session on a fresh workspace, in this test's placement. Closing it
    records the final snapshot."""
    if not os.environ.get("PRIME_API_KEY"):
        pytest.skip("needs PRIME_API_KEY")
    workers: list[subprocess.Popen] = []

    @contextlib.asynccontextmanager
    async def open_session(
        *,
        files: dict[str, str] | None = None,
        builtin_tools: list[str] | None = None,
        skills: tuple[str, ...] = ("edit",),
        skill_docs: dict[str, str] | None = None,
        mcp_servers: Sequence[Any] = (),
        policy: dict[str, Any] | None = None,
    ) -> AsyncIterator[AgentSession]:
        workspace = tmp_path / "workspace"
        workspace.mkdir()
        for name, text in (files or {}).items():
            (workspace / name).write_text(text)
        home = tmp_path / "rlm"
        for name, text in (skill_docs or {}).items():
            (home / "skills" / name).mkdir(parents=True)
            (home / "skills" / name / "SKILL.md").write_text(text)
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
        if mode == "stdio":
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
                cwd=str(workspace),
                mcp_servers=list(mcp_servers),
                **{RUNTIME_METADATA_KEY: runtime},
            )
            session = AgentSession(
                connection, created.session_id, client, workspace, home
            )
            try:
                yield session
            finally:
                if session.running is not None and not session.running.done():
                    session.running.cancel()
                    await asyncio.gather(session.running, return_exceptions=True)
                closed = await connection.close_session(created.session_id)
                session.closed_meta = closed.field_meta

    yield open_session
    for worker in workers:
        worker.kill()
        worker.wait()


@pytest.fixture
def run_agent(agent):
    """Run one prompt to its end in a fresh session."""

    async def run(prompt: str, **options: Any) -> AgentSession:
        async with agent(**options) as session:
            await session.prompt(prompt)
        return session

    return run
