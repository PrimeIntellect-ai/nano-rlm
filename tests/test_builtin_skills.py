"""Tests for built-in skills (``rlm.skills``): the ``edit``/``search`` skills + enable mechanism."""

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


from rlm.skills.search_mixedbread import format_results as format_mb_results
from rlm.skills.search_mixedbread import run as run_search_mixedbread


def test_search_mixedbread_enable_writes_stub(tmp_path):
    assert "search_mixedbread" in available_builtin_skills()
    assert enable_builtin_skills(["search_mixedbread"], tmp_path) == ["search_mixedbread"]
    assert (tmp_path / "search_mixedbread.py").read_text() == (
        "from rlm.skills.search_mixedbread import run\n"
    )


async def test_search_mixedbread_missing_api_key_returns_error(monkeypatch):
    monkeypatch.delenv("MIXEDBREAD_API_KEY", raising=False)
    result = await run_search_mixedbread(query="anything")
    assert "MIXEDBREAD_API_KEY" in result


def test_search_mixedbread_format_results():
    chunks = [
        {
            "text": "Page title\n\nRelevant excerpts from the page content...",
            "score": 0.95,
            "filename": "https://example.com/article",
            "metadata": {"title": "Page Title", "url": "https://example.com/article"},
        },
        {"text": "Another result body", "score": 0.7, "filename": "https://example.org/x", "metadata": {}},
    ]
    out = format_mb_results(chunks, "q")
    assert "Result 1: Page Title" in out
    assert "URL: https://example.com/article" in out
    assert "Score: 0.95" in out
    assert "Content: Page title\n\nRelevant excerpts from the page content..." in out
    assert "Result 2: Untitled" in out


def test_search_mixedbread_format_results_empty():
    assert format_mb_results([], "q") == "No results returned for query: q"
