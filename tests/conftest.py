"""Shared fixtures and dummy-LLM scaffolding for the test suite."""

from __future__ import annotations

import json
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest
from fixtures.tools.add import AddTool
from fixtures.tools.boom import BoomTool

from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.session import Session
from rlm.tools import registry as tool_registry

SKILL_FIXTURES_DIR = Path(__file__).parent / "fixtures" / "skills"

# --- Dummy message types --------------------------------------
#
# Use these to build up the conversation a DummyClient will replay. The
# engine only reads ``response.choices[0].message`` + ``response.usage``,
# so mocking that surface is enough.


@dataclass
class DummyFunction:
    name: str
    arguments: str


@dataclass
class DummyToolCall:
    """A scripted tool call inside a DummyMessage.

    ``arguments`` may be a dict (auto-serialized to JSON) or a raw string
    (useful for scripting malformed-JSON cases).
    """

    name: str
    arguments: str | dict
    id: str = "call_0"
    type: str = "function"

    def __post_init__(self) -> None:
        if isinstance(self.arguments, dict):
            self.arguments = json.dumps(self.arguments)

    @property
    def function(self) -> DummyFunction:
        # __post_init__ normalizes arguments to str.
        return DummyFunction(name=self.name, arguments=cast(str, self.arguments))


@dataclass
class DummyMessage:
    """A scripted assistant turn."""

    content: str | None = None
    tool_calls: list[DummyToolCall] | None = None
    role: str = "assistant"

    def model_dump(self, exclude_none: bool = True) -> dict[str, Any]:
        out: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_calls is not None:
            out["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": tc.type,
                    "function": {"name": tc.name, "arguments": tc.arguments},
                }
                for tc in self.tool_calls
            ]
        if exclude_none:
            out = {k: v for k, v in out.items() if v is not None}
        return out


# --- Dummy client -------------------------------------------


@dataclass
class DummyChoice:
    message: DummyMessage
    finish_reason: str = "stop"


@dataclass
class DummyUsage:
    prompt_tokens: int = 1
    completion_tokens: int = 1


@dataclass
class DummyResponse:
    choices: list[DummyChoice]
    usage: DummyUsage = field(default_factory=DummyUsage)


class DummyClient:
    """Replays scripted DummyMessages, one per ``chat.completions.create`` call."""

    def __init__(self, messages: list[DummyMessage]):
        self.scripted = list(messages)
        self.calls: list[dict[str, Any]] = []
        self.chat = self
        self.completions = self

    async def create(self, **kwargs: Any) -> DummyResponse:
        self.calls.append(kwargs)
        if not self.scripted:
            raise AssertionError("DummyClient exhausted: no more scripted messages")
        return DummyResponse(choices=[DummyChoice(message=self.scripted.pop(0))])


def tool_result(client: DummyClient, turn: int = 0) -> str:
    """Return the content of the ``turn``-th ``tool`` message sent to the model.

    Tool results appear on the next LLM request after their tool call, so
    the default ``turn=0`` reads the result produced by the first tool
    call (visible in ``client.calls[1]["messages"]``).
    """
    request_messages = client.calls[turn + 1]["messages"]
    tool_messages = [m for m in request_messages if m.get("role") == "tool"]
    return tool_messages[turn]["content"]


def show_tool_result(output: str) -> None:
    """Pretty-print a tool result for ``pytest -s`` inspection."""
    print(f"\n── tool result ──\n{output.rstrip()}\n─────────────────")


def make_runtime_config(**overrides) -> RuntimeConfig:
    """A minimal explicit RuntimeConfig for engine tests (no environment resolution)."""
    defaults = dict(
        model="dummy-model",
        provider=ProviderConfig(base_url=None, api_key="EMPTY"),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(),
    )
    defaults.update(overrides)
    return RuntimeConfig(**defaults)


# --- Fixtures ------------------------------------------------------------


@pytest.fixture
def session(tmp_path):
    return Session(tmp_path / "session")


# --- Real-skill scaffolding ---------------------------------------------
#
# Each fixture skill in ``tests/fixtures/skills/`` is editable-installed
# once per test session (``install_fixture_skills``). That matches
# what the environment's install script does in production: the CLI
# lands in ``.venv/bin/<name>`` (so IPython's ``!<name>`` shell escape
# finds it) and the ``rlm-skill-<name>`` distribution shows up in
# ``importlib.metadata``, which is how ``get_installed_skills()``
# discovers it at kernel startup.


@pytest.fixture(scope="session", autouse=True)
def register_fixture_tools():
    """Session-wide: fixture tools (add/boom) join the builtin registry alongside ipython."""
    mp = pytest.MonkeyPatch()
    mp.setitem(tool_registry._TOOLS_BY_NAME, "add", AddTool())
    mp.setitem(tool_registry._TOOLS_BY_NAME, "boom", BoomTool())
    yield
    mp.undo()


_SKILL_WRAPPER_TEMPLATE = """\
#!/usr/bin/env bash
set -eo pipefail
REAL_TOOL={real_path!r}
TOOL_NAME={tool_name!r}
SOURCE="${{RLM_TOOL_CALL_SOURCE:-bash}}"
if [ -n "${{RLM_SESSION_DIR:-}}" ]; then
  printf '{{"tool":"%s","source":"%s","timestamp":%s}}\\n' \\
      "$TOOL_NAME" "$SOURCE" "$(date +%s.%N)" \\
      >> "${{RLM_SESSION_DIR}}/programmatic_tool_calls.jsonl" 2>/dev/null || true
fi
exec "$REAL_TOOL" "$@"
"""


@pytest.fixture(scope="session", autouse=True)
def install_fixture_skills():
    """Editable-install every skill under ``tests/fixtures/skills/`` once per session.

    Also installs the same bash wrapper ``install.sh`` creates around each
    skill CLI so that ``!<skill>`` invocations log a ``source="bash"`` entry
    to ``programmatic_tool_calls.jsonl``. Without this the fixture
    environment diverges from production and bash-path metrics tests fail.
    """
    installed: list[str] = []
    wrapped: list[tuple[Path, Path]] = []  # (wrapper_path, real_path)
    bin_dir = Path(sys.executable).parent
    real_tool_dir = bin_dir / ".rlm-real-tools"
    real_tool_dir.mkdir(exist_ok=True)

    for skill_dir in sorted(SKILL_FIXTURES_DIR.iterdir()):
        if not (skill_dir / "pyproject.toml").is_file():
            continue
        subprocess.run(
            [
                "uv",
                "pip",
                "install",
                "-e",
                str(skill_dir),
                "--python",
                sys.executable,
                "-q",
            ],
            check=True,
        )
        installed.append(f"rlm-skill-{skill_dir.name.replace('_', '-')}")

        tool_name = skill_dir.name
        wrapper_path = bin_dir / tool_name
        real_path = real_tool_dir / tool_name
        if wrapper_path.is_file() and not real_path.is_file():
            wrapper_path.rename(real_path)
            wrapper_path.write_text(
                _SKILL_WRAPPER_TEMPLATE.format(
                    real_path=str(real_path), tool_name=tool_name
                )
            )
            wrapper_path.chmod(0o755)
            wrapped.append((wrapper_path, real_path))
    yield
    for wrapper_path, real_path in wrapped:
        wrapper_path.unlink(missing_ok=True)
        if real_path.is_file():
            real_path.rename(wrapper_path)
    for dist in installed:
        subprocess.run(
            ["uv", "pip", "uninstall", dist, "--python", sys.executable],
            check=False,
            capture_output=True,
        )
