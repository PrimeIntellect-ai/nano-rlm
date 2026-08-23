"""Tests for system prompt construction."""

from __future__ import annotations

from dataclasses import dataclass

from rlm.prompt import (
    EDIT_SKILL_PROMPT,
    GIT_HISTORY_GUARD_PROMPT,
    IPYTHON_CONTROL_PROMPT,
    SEARCH_SKILL_PROMPT,
    build_system_prompt,
)


@dataclass
class _Tool:
    name: str


def _prompt(
    active_tools: list[_Tool],
    *,
    installed_skills: list[str] | None = None,
    cli_skills: list[str] | None = None,
) -> str:
    return build_system_prompt(
        "/repo",
        None,
        installed_skills or [],
        allow_recursion=False,
        active_tools=active_tools,
        cli_skills=cli_skills,
    )


def test_git_history_guard_prompt_included_for_shell_tools(monkeypatch):
    monkeypatch.delenv("RLM_ALLOW_GIT", raising=False)
    prompt = _prompt([_Tool("ipython")])

    assert GIT_HISTORY_GUARD_PROMPT in prompt
    assert "Do not cheat" in prompt
    assert "online solutions or hints specific to this task" in prompt
    assert "other branches, tags, remotes" in prompt
    assert "`--all`" in prompt


def test_git_history_guard_prompt_omitted_when_unrestricted(monkeypatch):
    monkeypatch.setenv("RLM_ALLOW_GIT", "1")

    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("ipython")])


def test_git_history_guard_prompt_omitted_without_shell_tools(monkeypatch):
    monkeypatch.delenv("RLM_ALLOW_GIT", raising=False)

    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("summarize")])


def test_ipython_control_prompt_included_for_ipython_tool():
    prompt = _prompt([_Tool("ipython")])

    assert IPYTHON_CONTROL_PROMPT in prompt
    assert "`%%bash` as the very first line" in prompt
    assert "project's own environment" in prompt


def test_ipython_control_prompt_omitted_without_ipython_tool():
    assert IPYTHON_CONTROL_PROMPT not in _prompt([])


def test_edit_skill_prompt_included_only_when_edit_is_installed():
    prompt = _prompt([_Tool("ipython")], installed_skills=["edit"])

    assert EDIT_SKILL_PROMPT in prompt
    assert 'await edit(path="pkg/file.py", old_str=..., new_str=...)' in prompt
    assert EDIT_SKILL_PROMPT not in _prompt(
        [_Tool("ipython")], installed_skills=["search_docs"]
    )


def test_search_skill_prompt_included_only_when_search_is_installed():
    prompt = _prompt([_Tool("ipython")], installed_skills=["search"])

    assert SEARCH_SKILL_PROMPT in prompt
    assert "await search(query=" in prompt
    assert SEARCH_SKILL_PROMPT not in _prompt(
        [_Tool("ipython")], installed_skills=["search_docs"]
    )


def test_only_cli_skills_are_advertised_as_shell_commands():
    prompt = _prompt(
        [_Tool("ipython")],
        installed_skills=["installed", "generated"],
        cli_skills=["installed"],
    )

    assert "Skills with shell commands: `installed`" in prompt
    assert "`generated` --help" not in prompt
