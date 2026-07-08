"""Tests for built-in skills (``rlm.skills``): the ``edit``/``search``/``open_webpage`` skills + enable mechanism."""

from __future__ import annotations

import pytest

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


def test_search_enable_writes_stub(tmp_path):
    assert "search" in available_builtin_skills()
    assert enable_builtin_skills(["search"], tmp_path) == ["search"]
    assert (tmp_path / "search.py").read_text() == "from rlm.skills.search import run\n"


def test_open_webpage_enable_writes_stub(tmp_path):
    assert "open_webpage" in available_builtin_skills()
    assert enable_builtin_skills(["open_webpage"], tmp_path) == ["open_webpage"]
    assert (
        tmp_path / "open_webpage.py"
    ).read_text() == "from rlm.skills.open_webpage import run\n"


async def test_search_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("SERPER_API_KEY", raising=False)
    result = await run_search(query="anything")
    assert "SERPER_API_KEY" in result


async def test_search_batches_queries_into_one_call(monkeypatch):
    import rlm.skills.search as search_skill

    captured = {}
    organic = [{"title": "t", "link": "https://x", "snippet": "s"}]

    class FakeResponse:
        def __init__(self, data):
            self._data = data

        def raise_for_status(self):
            pass

        def json(self):
            return self._data

    def fake_post(url, json=None, headers=None, timeout=None):
        captured["json"] = json
        if isinstance(json, list):
            return FakeResponse([{"organic": organic} for _ in json])
        return FakeResponse({"organic": organic})

    monkeypatch.setenv("SERPER_API_KEY", "test-key")
    monkeypatch.setattr(search_skill.httpx, "post", fake_post)

    single = await run_search(query="one", num_results=3)
    assert captured["json"] == {"q": "one", "num": 3}
    assert single.startswith("Result 1: t")

    batched = await run_search(query=["one", "two"])
    assert captured["json"] == [{"q": "one", "num": 10}, {"q": "two", "num": 10}]
    assert 'Results for query "one":' in batched
    assert 'Results for query "two":' in batched
    assert "\n\n==========\n\n" in batched


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
