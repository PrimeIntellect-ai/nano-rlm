"""Tests for system prompt construction."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

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
    allow_git: bool = False,
) -> str:
    return build_system_prompt(
        "/repo",
        None,
        installed_skills or [],
        allow_recursion=False,
        allow_git=allow_git,
        active_tools=active_tools,
    )


def test_git_history_guard_prompt_included_for_shell_tools():
    prompt = _prompt([_Tool("ipython")])

    assert GIT_HISTORY_GUARD_PROMPT in prompt
    assert "Do not cheat" in prompt
    assert "online solutions or hints specific to this task" in prompt
    assert "other branches, tags, remotes" in prompt
    assert "`--all`" in prompt


def test_git_history_guard_prompt_omitted_when_unrestricted():
    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("ipython")], allow_git=True)


def test_git_history_guard_prompt_omitted_without_shell_tools():
    assert GIT_HISTORY_GUARD_PROMPT not in _prompt([_Tool("summarize")])


def test_ipython_control_prompt_included_for_ipython_tool():
    prompt = _prompt([_Tool("ipython")])

    assert IPYTHON_CONTROL_PROMPT in prompt
    assert "await rlm.shell.run(command, cwd=...)" in prompt
    assert "project's interpreter" in prompt


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
    assert "one formatted text string" in prompt
    assert SEARCH_SKILL_PROMPT not in _prompt(
        [_Tool("ipython")], installed_skills=["search_docs"]
    )


def test_prompt_only_advertises_actual_shell_skills():
    prompt = build_system_prompt(
        "/repo",
        None,
        ["edit", "uploaded"],
        allow_recursion=False,
        allow_git=False,
        active_tools=[_Tool("ipython")],
        shell_skills=["uploaded"],
    )

    assert "Shell-enabled installed skills: `uploaded`" in prompt
    assert "Other listed skills are IPython-only" in prompt


def test_runtime_guidance_matches_agent_capabilities():
    leaf = build_system_prompt(
        "/repo",
        None,
        [],
        depth=1,
        allow_recursion=False,
        allow_git=False,
        active_tools=[_Tool("ipython")],
        agent_info={
            "id": "leaf",
            "parent_id": "root",
            "name": "researcher",
            "persistent": True,
        },
    )
    assert "rlm.agent.send_to_parent(message)" in leaf
    assert "rlm.agent.spawn(" not in leaf
    assert "rlm.watch.agent(" not in leaf
    assert "become idle" in leaf
    assert '"parent_id": "root"' in leaf
    assert "rlm.shell.run" in leaf

    native = build_system_prompt(
        "/repo",
        None,
        ["search"],
        allow_recursion=True,
        allow_git=False,
        active_tools=[_Tool("bash")],
    )
    assert "rlm.agent.spawn(" not in native
    assert "rlm.shell.run" not in native
    assert "await search" not in native
    assert "native bash tool" in native
    assert GIT_HISTORY_GUARD_PROMPT in native

    root = build_system_prompt(
        "/repo",
        None,
        [],
        allow_recursion=True,
        allow_git=False,
        active_tools=[_Tool("ipython")],
    )
    assert "rlm.agent.spawn(" in root
    assert "rlm.watch.agent(" in root
    assert "You have no parent" in root


@pytest.mark.parametrize("tool", ["add", "ipython"])
async def test_wait_registration_matches_runtime_capability(session, tool):
    from conftest import DummyClient, DummyMessage, make_runtime_config
    from rlm.engine import RLMEngine

    engine = RLMEngine(
        client=DummyClient([DummyMessage(content="done")]),
        session=session,
        runtime_config=make_runtime_config(builtin_tools=(tool,)),
    )
    try:
        await engine.prompt("task")
        registered = {
            schema["function"]["name"] for schema in engine._active_tool_schemas
        }
        assert ("wait" in registered) == (tool == "ipython")
    finally:
        await engine.aclose()
