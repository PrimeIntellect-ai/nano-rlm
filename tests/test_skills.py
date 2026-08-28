"""Tests for skills: skills invoked from inside the IPython tool.

Drives ``rlm.engine.RLMEngine`` with a scripted DummyClient and a real
skill fixture on PATH; the engine starts its own IPython kernel to
dispatch the ``ipython`` tool call, so the code inside the kernel hits
the real skill the model would see:

- python form → ``await <skill>(...)`` calls into the imported skill's ``run``
- bash form   → ``!<skill> ...`` runs the CLI directly from PATH

Skill fixtures live under ``tests/fixtures/skills/<name>/``.
Kernel startup is ~700ms per test; keep the set here small.
"""

from __future__ import annotations

import os
import sys

import pytest

from conftest import (
    DummyClient,
    DummyMessage,
    DummyToolCall,
    make_runtime_config,
    show_tool_result,
    tool_result,
)

from rlm.engine import RLMEngine
from rlm.tools.skills import discover_skills


async def test_python_skill_valid(session):
    """Python form: ``await say(...)`` in an ipython tool call hits the real skill."""
    prompt = "say hello"
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "await say(s='hello')"})]
        ),
        DummyMessage(content="ok"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "hello" in output
    assert result.answer == "ok"


async def test_bash_skill_valid(session, monkeypatch, tmp_path):
    """Bash form: ``!say ...`` in an ipython tool call runs the skill CLI via IPython's shell escape."""
    prompt = "say hello"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!say --s hello"})]),
        DummyMessage(content="ok"),
    ]

    client = DummyClient(messages)
    bin_dir = os.path.dirname(sys.executable)
    launcher_dir = tmp_path / "launcher-bin"
    launcher_dir.mkdir()
    (launcher_dir / "rlm").symlink_to(sys.executable)
    (launcher_dir / "say").symlink_to(os.path.join(bin_dir, "say"))
    monkeypatch.setenv(
        "PATH",
        os.pathsep.join(
            p
            for p in os.environ["PATH"].split(os.pathsep)
            if p not in {bin_dir, str(launcher_dir)}
        ),
    )
    monkeypatch.setattr(sys, "argv", [str(launcher_dir / "rlm")])
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "hello" in output
    assert result.answer == "ok"


def test_discover_skills_rejects_installed_generated_collision(monkeypatch, tmp_path):
    monkeypatch.setattr("rlm.tools.skills.get_installed_skills", lambda: ["search"])
    (tmp_path / "search.py").write_text("async def run(): pass\n")

    with pytest.raises(ValueError, match="skill name collision.*search"):
        discover_skills(tmp_path)


async def test_python_skill_invalid_args(session):
    """Python form: missing required arg → ``TypeError`` traceback is returned to the model."""
    prompt = "say something"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "await say()"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "TypeError" in output
    assert "missing 1 required positional argument: 's'" in output
    assert result.answer == "the call failed"


async def test_bash_skill_invalid_args(session):
    """Bash form: missing required arg → the CLI prints a usage error."""
    prompt = "say something"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!say"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "--s" in output
    assert "required" in output
    assert "╭" not in output
    assert "╰" not in output
    assert "│" not in output
    assert "─" not in output
    assert result.answer == "the call failed"


async def test_python_skill_raises(session):
    """Python form: ``await boom()`` raising inside the kernel surfaces the traceback."""
    prompt = "set off the boom skill"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "await boom()"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "boom" in output
    assert result.answer == "the call failed"


async def test_bash_skill_raises(session):
    """Bash form: ``!boom`` raising in the CLI surfaces the traceback on stderr."""
    prompt = "set off the boom skill"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!boom"})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    result = await engine.run(prompt)

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "boom" in output
    assert result.answer == "the call failed"


async def test_python_skill_halt_on_raise(session):
    """Python form: a raise halts the cell — ``say`` after ``await boom()`` is not executed."""
    # Source shows ``'R' + 'A' + 'N'``; say would print ``RAN`` only if it ran.
    code = "await boom(); await say(s='R' + 'A' + 'N')"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    await engine.run("try boom then say")

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "RAN" not in output


async def test_bash_skill_halt_on_raise(session):
    """Bash form: ``&&`` short-circuits — ``say`` is not executed after ``boom`` fails."""
    code = "!boom && say --s RAN"
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
        DummyMessage(content="the call failed"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    await engine.run("try boom then say")

    output = tool_result(client)
    show_tool_result(output)
    assert "RuntimeError" in output
    assert "RAN" not in output


async def test_valid_python_skill_metrics(session):
    """Python-form skill call increments num_ptc_calls_python."""
    messages = [
        DummyMessage(
            tool_calls=[DummyToolCall("ipython", {"code": "await say(s='hi')"})]
        ),
        DummyMessage(content="ok"),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    await engine.run("say hi")

    assert engine._metrics.num_ptc_calls_python == 1
    assert engine._metrics.num_ptc_calls_bash == 0


async def test_valid_bash_skill_metrics(session):
    """Bash-form skill call (``!say ...``) increments num_ptc_calls_bash."""
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": "!say --s hi"})]),
        DummyMessage(content="ok"),
    ]
    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    await engine.run("say hi")

    assert engine._metrics.num_ptc_calls_python == 0
    assert engine._metrics.num_ptc_calls_bash == 1


async def test_skill_introspection(session):
    """``inspect.signature`` and ``__doc__`` see through the logger wrapper.

    Without ``functools.wraps`` + a ``__signature__`` override on the
    callable module, ``inspect.signature(say)`` returns
    ``_CallableModule.__call__``'s ``(*args, **kwargs)`` and
    ``inspect.signature(say.run)`` returns the logger shim's
    ``(*args, **kwargs)`` — both useless for the model that prompt.py
    tells to introspect the skill API.
    """
    code = (
        "import inspect\n"
        "print('sig:', list(inspect.signature(say).parameters))\n"
        "print('sig.run:', list(inspect.signature(say.run).parameters))\n"
        "print('doc match:', say.__doc__ == say.run.__doc__)"
    )
    messages = [
        DummyMessage(tool_calls=[DummyToolCall("ipython", {"code": code})]),
        DummyMessage(content="ok"),
    ]

    client = DummyClient(messages)
    engine = RLMEngine(
        client=client, session=session, runtime_config=make_runtime_config()
    )  # type: ignore

    await engine.run("introspect say")

    output = tool_result(client)
    show_tool_result(output)
    assert "sig: ['s']" in output
    assert "sig.run: ['s']" in output
    assert "doc match: True" in output
