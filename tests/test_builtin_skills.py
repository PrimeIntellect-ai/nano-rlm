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


def test_fetch_enable_writes_stub(tmp_path):
    from rlm.skills import available_builtin_skills, enable_builtin_skills

    assert "fetch" in available_builtin_skills()
    assert enable_builtin_skills(["fetch"], tmp_path) == ["fetch"]
    assert (tmp_path / "fetch.py").read_text() == "from rlm.skills.fetch import run\n"


def test_fetch_html_to_text_strips_markup():
    from rlm.skills.fetch import _html_to_text

    html = "<html><head><style>x{}</style></head><body><h1>T</h1><p>Hello &amp; world</p><script>bad()</script><p>Line2</p></body></html>"
    out = _html_to_text(html)
    assert "Hello & world" in out and "Line2" in out
    assert "bad()" not in out and "<" not in out


async def test_fetch_rejects_non_http():
    from rlm.skills.fetch import run as fetch

    out = await fetch(url="ftp://example.com/x")
    assert out.startswith("Error") and "http" in out
