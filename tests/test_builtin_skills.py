"""Tests for built-in skills (``rlm.skills``): the ``edit``/``search`` skills + enable mechanism."""

from __future__ import annotations

import json

import pytest

from conftest import DummyClient, DummyMessage, DummyToolCall, tool_result

from rlm.config import (
    ExecutionPolicy,
    InvocationContext,
    ProviderConfig,
    RuntimeConfig,
)
from rlm.engine import RLMEngine
from rlm.skills import available_builtin_skills, enable_builtin_skills
from rlm.skills.edit import run as edit
from rlm.skills.search import format_results
from rlm.skills.search import run as run_search


async def test_edit_replaces_unique_string(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("hello world")
    result = await edit(path=str(f), old_str="world", new_str="there")
    assert result == f"Edited {f}"
    assert f.read_text() == "hello there"


async def test_edit_requires_exactly_one_occurrence(tmp_path):
    f = tmp_path / "a.txt"
    f.write_text("x x")
    with pytest.raises(ValueError, match="exactly once"):
        await edit(path=str(f), old_str="x", new_str="y")
    assert f.read_text() == "x x"


async def test_edit_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        await edit(path=str(tmp_path / "nope.txt"), old_str="a", new_str="b")


def test_enable_builtin_skills_writes_stub(tmp_path):
    assert "edit" in available_builtin_skills()
    assert enable_builtin_skills(["edit"], tmp_path) == ["edit"]
    assert (tmp_path / "edit.py").read_text() == "from rlm.skills.edit import run\n"


def test_enable_unknown_skill_raises(tmp_path):
    with pytest.raises(ValueError, match="unknown skill"):
        enable_builtin_skills(["nope"], tmp_path)


async def test_search_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    result = await run_search(query="anything")
    assert "SERPER_API_KEY" in result


def test_search_format_results():
    results = [
        {"title": "First", "link": "https://a", "snippet": "snippet one"},
        {"title": "", "link": "", "snippet": ""},
    ]
    out = format_results(results, "q")
    assert "Result 1: First" in out
    assert "URL: https://a" in out
    assert "- snippet one" in out
    assert "Result 2: Untitled" in out


def test_search_format_results_empty():
    assert format_results([], "q") == "No results returned for query: q"


async def test_real_kernel_search_is_brokered_without_key(monkeypatch, session):
    secret = "search-secret-do-not-leak"
    requests = []

    class Response:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "organic": [
                    {
                        "title": "Result",
                        "link": "https://example.com",
                        "snippet": "Found it",
                    }
                ]
            }

    class FakeClient:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return None

        async def post(self, url, **kwargs):
            requests.append((url, kwargs))
            return Response()

    monkeypatch.setattr("rlm.skills.search.httpx.AsyncClient", FakeClient)
    monkeypatch.setenv("SERPER_API_KEY", secret)
    config = RuntimeConfig(
        model="test-model",
        provider=ProviderConfig(
            base_url="http://interceptor", api_key="inference-secret"
        ),
        invocation=InvocationContext(),
        policy=ExecutionPolicy(),
        skills=("search",),
        search_api_key=secret,
    )
    client = DummyClient(
        [
            DummyMessage(
                tool_calls=[
                    DummyToolCall(
                        "ipython",
                        {
                            "code": """
import os, subprocess
print(await search(query='needle'))
child_env = subprocess.check_output(['env'], text=True)
print('SERPER_API_KEY' not in os.environ)
print('SERPER_API_KEY=' not in child_env)
"""
                        },
                    )
                ]
            ),
            DummyMessage(content="done"),
        ]
    )
    engine = RLMEngine(
        client=client,  # type: ignore[arg-type]
        session=session,
        runtime_config=config,
    )

    result = await engine.run("search")

    output = tool_result(client)
    assert result.answer == "done"
    assert "Result 1: Result" in output
    assert output.strip().splitlines()[-2:] == ["True", "True"]
    assert requests[0][1]["headers"]["X-API-KEY"] == secret
    source = (session.dir / "search.py").read_text()
    assert secret not in source
    meta = json.loads((session.dir / "meta.json").read_text())
    assert meta["programmatic_tool_call_stats"]["by_tool_python"] == {"search": 1}
